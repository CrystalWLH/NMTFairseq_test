'''
    CTC segmentation and NMT based on LSTM.
    Author: Lihui Wang && Shaojun Gao
    Create Date: 2020-01-02
    Update Date: 2020-01-09
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    FairseqEncoderDecoderDoubleModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import AdaptiveSoftmax


@register_model('seg_nmt_ctc_lstm')
class SegNmtCtcLSTMModel(FairseqEncoderDecoderDoubleModel):
    """
        seg_nmt model for training segmentation and NMT jointly.

        Args:
            shared_encoder(LSTMEncoder): seg and nmt shared encoder
            ctc_decoder (LSTMDecoder): the seg decoder for ctc
            nmt_encoder (LSTMEncoder): the nmt encoder
            nmt_decoder (LSTMDecoder): the nmt decoder
    """
    def __init__(self, shared_encoder, ctc_decoder, nmt_encoder, nmt_decoder):
        super().__init__(shared_encoder, ctc_decoder, nmt_encoder, nmt_decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--just-ctc', default=False, action='store_true',
                            help='ctc segmentation or ctc segmentation + nmt')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--shared-encoder-embed-dim', type=int, metavar='N',
                            help='shared encoder embedding dimension')
        parser.add_argument('--shared-encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained shared encoder embedding')
        parser.add_argument('--shared-encoder-freeze-embed', action='store_true',
                            help='freeze shared encoder embeddings')
        parser.add_argument('--shared-encoder-hidden-size', type=int, metavar='N',
                            help='shared encoder hidden size')
        parser.add_argument('--shared-encoder-layers', type=int, metavar='N',
                            help='number of shared encoder layers')
        parser.add_argument('--shared-encoder-bidirectional', action='store_true',
                            help='make all layers of shared encoder bidirectional')
        parser.add_argument('--nmt-encoder-embed-dim', type=int, metavar='N',
                            help='nmt encoder embedding dimension')
        parser.add_argument('--nmt-encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained nmt encoder embedding')
        parser.add_argument('--nmt-encoder-freeze-embed', action='store_true',
                            help='freeze nmt encoder embeddings')
        parser.add_argument('--nmt-encoder-hidden-size', type=int, metavar='N',
                            help='nmt encoder hidden size')
        parser.add_argument('--nmt-encoder-layers', type=int, metavar='N',
                            help='number of nmt encoder layers')
        parser.add_argument('--nmt-encoder-bidirectional', action='store_true',
                            help='make all layers of nmt encoder bidirectional')
        parser.add_argument('--ctc-decoder-embed-dim', type=int, metavar='N',
                            help='ctc decoder embedding dimension')
        parser.add_argument('--ctc-decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained ctc decoder embedding')
        parser.add_argument('--ctc-decoder-freeze-embed', action='store_true',
                            help='freeze ctc decoder embeddings')
        parser.add_argument('--ctc-decoder-hidden-size', type=int, metavar='N',
                            help='ctc decoder hidden size')
        parser.add_argument('--ctc-decoder-layers', type=int, metavar='N',
                            help='number of ctc decoder layers')
        parser.add_argument('--ctc-decoder-out-embed-dim', type=int, metavar='N',
                            help='ctc decoder output embedding dimension')
        parser.add_argument('--ctc-decoder-attention', type=str, metavar='BOOL',
                            help='ctc decoder attention')
        parser.add_argument('--nmt-decoder-embed-dim', type=int, metavar='N',
                            help='nmt decoder embedding dimension')
        parser.add_argument('--nmt-decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained nmt decoder embedding')
        parser.add_argument('--nmt-decoder-freeze-embed', action='store_true',
                            help='freeze nmt decoder embeddings')
        parser.add_argument('--nmt-decoder-hidden-size', type=int, metavar='N',
                            help='nmt decoder hidden size')
        parser.add_argument('--nmt-decoder-layers', type=int, metavar='N',
                            help='number of nmt decoder layers')
        parser.add_argument('--nmt-decoder-out-embed-dim', type=int, metavar='N',
                            help='nmt decoder output embedding dimension')
        parser.add_argument('--nmt-decoder-attention', type=str, metavar='BOOL',
                            help='nmt decoder attention')
        
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')
        parser.add_argument('--share-decoder-input-output-embed', default=False,
                            action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', default=False, action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')

        # Granular dropout settings (if not specified these default to --dropout)
        parser.add_argument('--shared-encoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for shared encoder input embedding')
        parser.add_argument('--shared-encoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for shared encoder output')
        parser.add_argument('--ctc-decoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for ctc decoder input embedding')
        parser.add_argument('--ctc-decoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for ctc decoder output')
        parser.add_argument('--nmt-encoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for nmt encoder input embedding')
        parser.add_argument('--nmt-encoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for nmt encoder output')
        parser.add_argument('--nmt-decoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for nmt decoder input embedding')
        parser.add_argument('--nmt-decoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for nmt decoder output embedding.')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        if args.nmt_encoder_layers != args.nmt_decoder_layers:
            raise ValueError('--nmt-encoder-layers must match --nmt-decoder-layers')

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        if args.shared_encoder_embed_path:
            pretrained_shared_encoder_embed = load_pretrained_embedding_from_file(
                args.shared_encoder_embed_path, task.source_dictionary, args.shared_encoder_embed_dim)
        else:
            num_embeddings = len(task.source_dictionary)
            pretrained_shared_encoder_embed = Embedding(
                num_embeddings, args.shared_encoder_embed_dim, task.source_dictionary.pad()
            )
        if args.nmt_encoder_embed_path:
            pretrained_shared_encoder_embed = load_pretrain_embedding_from_file(
                args.nmt_encoder_embed_path, task.seg_dictionary, args.nmt_encoder_embed_dim)
        else:
            num_embedding = len(task.seg_dictionary)
            pretrained_shared_encoder_embed = Embedding(
                num_embeddings, args.nmt_encoder_embed_dim, task.seg_dictionary.pad()
            )
        
        if args.share_all_embeddings:
            # double check all parameters combinations are valid
            if task.source_dictionary != task.target_dictionary:
                raise ValueError('--share-all-embeddings requires a joint dictionary')
            if args.nmt_decoder_embed_path and (
                    args.nmt_decoder_embed_path != args.shared_encoder_embed_path):
                raise ValueError(
                    '--share-all-embed not compatible with --shared-decoder-embed-path'
                )
            if args.shared_encoder_embed_dim != args.nmt_decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --shared-encoder-embed-dim to '
                    'match --nmt-decoder-embed-dim'
                )
            pretrained_nmt_decoder_embed = pretrained_shared_encoder_embed
            args.share_decoder_input_output_embed = True
        else:
            # separate decoder input embeddings
            pretrained_ctc_decoder_embed = None
            pretrained_nmt_decoder_embed = None
            if args.ctc_decoder_embed_path:
                pretrained_ctc_decoder_embed = load_pretrained_embedding_from_file(
                    args.ctc_decoder_embed_path,
                    task.seg_dictionary,
                    args.ctc_decoder_embed_dim
                )
            if args.nmt_decoder_embed_path:
                pretrained_nmt_decoder_embed = load_pretrained_embedding_from_file(
                    args.nmt_decoder_embed_path,
                    task.target_dictionary,
                    args.nmt_decoder_embed_dim
                )
        # one last double check of parameter combinations
        if args.share_decoder_input_output_embed and (
                args.nmt_decoder_embed_dim != args.nmt_decoder_out_embed_dim):
            raise ValueError(
                '--share-decoder-input-output-embeddings requires '
                '--nmt_decoder-embed-dim to match --nmt_decoder-out-embed-dim'
            )

        if args.shared_encoder_freeze_embed:
            pretrained_shared_encoder_embed.weight.requires_grad = False
        if args.ctc_decoder_freeze_embed:
            pretrained_ctc_decoder_embed.weight.requires_grad = False
        if args.nmt_encoder_freeze_embed:
            pretrained_nmt_encoder_embed.weight.requires_grad = False
        if args.nmt_decoder_freeze_embed:
            pretrained_nmt_decoder_embed.weight.requires_grad = False
        
        shared_encoder = LSTMEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args.shared_encoder_embed_dim,
            hidden_size=args.shared_encoder_hidden_size,
            num_layers=args.shared_encoder_layers,
            dropout_in=args.shared_encoder_dropout_in,
            dropout_out=args.shared_encoder_dropout_out,
            bidirectional=args.shared_encoder_bidirectional,
            pretrained_embed=pretrained_shared_encoder_embed,
        )
        nmt_encoder = LSTMEncoder(
            dictionary=task.target_dictionary,
            embed_dim=args.nmt_encoder_embed_dim,
            hidden_size=args.nmt_encoder_hidden_size,
            num_layers=args.nmt_encoder_layers,
            dropout_in=args.nmt_encoder_dropout_in,
            dropout_out=args.nmt_encoder_dropout_out,
            bidirectional=args.nmt_encoder_bidirectional,
            pretrained_embed=pretrained_nmt_encoder_embed,
        )
        ctc_decoder = CTCDecoder(
            encoder_output_units=args.shared_encoder_hidden_size,
            out_dim=len(task.seg_dictionary),
            hidden_size=args.ctc_decoder_hidden_size,
            num_layers=args.ctc_decoder_layers,
            )
        nmt_decoder = LSTMDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.nmt_decoder_embed_dim,
            hidden_size=args.nmt_decoder_hidden_size,
            out_embed_dim=args.nmt_decoder_out_embed_dim,
            num_layers=args.nmt_decoder_layers,
            dropout_in=args.nmt_decoder_dropout_in,
            dropout_out=args.nmt_decoder_dropout_out,
            attention=options.eval_bool(args.nmt_decoder_attention),
            encoder_output_units=nmt_encoder.output_units,
            pretrained_embed=pretrained_nmt_decoder_embed,
            share_input_output_embed=args.share_decoder_input_output_embed,
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == 'adaptive_loss' else None
            ),
        )
        return cls(shared_encoder, nmt_encoder, ctc_decoder, nmt_decoder)


class LSTMEncoder(FairseqEncoder):
    """LSTM encoder."""
    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, num_layers=1,
        dropout_in=0.1, dropout_out=0.1, bidirectional=False,
        left_pad=True, pretrained_embed=None, padding_value=0.,
    ):
        super().__init__(dictionary)
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.lstm = LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout_out if num_layers > 1 else 0.,
            bidirectional=bidirectional,
        )
        self.left_pad = left_pad
        self.padding_value = padding_value

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def forward(self, src_tokens, src_lengths):
        if self.left_pad:
            # nn.utils.rnn.pack_padded_sequence requires right-padding;
            # convert left-padding to right-padding
            src_tokens = utils.convert_padding_direction(
                src_tokens,
                self.padding_idx,
                left_to_right=True,
            )

        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.data.tolist())

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=self.padding_value)
        x = F.dropout(x, p=self.dropout_out, training=self.training)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.bidirectional:

            def combine_bidir(outs):
                out = outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous()
                return out.view(self.num_layers, bsz, -1)

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        return {
            'encoder_out': (x, final_hiddens, final_cells),
            'encoder_padding_mask': encoder_padding_mask if encoder_padding_mask.any() else None
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out['encoder_out'] = tuple(
            eo.index_select(1, new_order)
            for eo in encoder_out['encoder_out']
        )
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(1, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number


class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, bias=False):
        super().__init__()

        self.input_proj = Linear(input_embed_dim, source_embed_dim, bias=bias)
        self.output_proj = Linear(input_embed_dim + source_embed_dim, output_embed_dim, bias=bias)

    def forward(self, input, source_hids, encoder_padding_mask):
        # input: bsz x input_embed_dim
        # source_hids: srclen x bsz x source_embed_dim

        # x: bsz x source_embed_dim
        x = self.input_proj(input)

        # compute attention
        attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)

        # don't attend over padding
        if encoder_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                encoder_padding_mask,
                float('-inf')
            ).type_as(attn_scores)  # FP16 support: cast to float and back

        attn_scores = F.softmax(attn_scores, dim=0)  # srclen x bsz

        # sum weighted sources
        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)

        x = torch.tanh(self.output_proj(torch.cat((x, input), dim=1)))
        return x, attn_scores


class LSTMDecoder(FairseqIncrementalDecoder):
    """LSTM decoder."""
    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=True,
        encoder_output_units=512, pretrained_embed=None,
        share_input_output_embed=False, adaptive_softmax_cutoff=None,
    ):
        super().__init__(dictionary)
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed
        self.need_attn = True

        self.adaptive_softmax = None
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.encoder_output_units = encoder_output_units
        if encoder_output_units != hidden_size:
            self.encoder_hidden_proj = Linear(encoder_output_units, hidden_size)
            self.encoder_cell_proj = Linear(encoder_output_units, hidden_size)
        else:
            self.encoder_hidden_proj = self.encoder_cell_proj = None
        self.layers = nn.ModuleList([
            LSTMCell(
                input_size=hidden_size + embed_dim if layer == 0 else hidden_size,
                hidden_size=hidden_size,
            )
            for layer in range(num_layers)
        ])
        if attention:
            # TODO make bias configurable
            self.attention = AttentionLayer(hidden_size, encoder_output_units, hidden_size, bias=False)
        else:
            self.attention = None
        if hidden_size != out_embed_dim:
            self.additional_fc = Linear(hidden_size, out_embed_dim)
        if adaptive_softmax_cutoff is not None:
            # setting adaptive_softmax dropout to dropout_out for now but can be redefined
            self.adaptive_softmax = AdaptiveSoftmax(num_embeddings, hidden_size, adaptive_softmax_cutoff,
                                                    dropout=dropout_out)
        elif not self.share_input_output_embed:
            self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)

    def forward(self, prev_output_tokens, encoder_out, incremental_state=None):
        encoder_padding_mask = encoder_out['encoder_padding_mask']
        encoder_out = encoder_out['encoder_out']

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, seqlen = prev_output_tokens.size()

        # get outputs from encoder
        encoder_outs, encoder_hiddens, encoder_cells = encoder_out[:3]
        srclen = encoder_outs.size(0)

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed = cached_state
        else:
            num_layers = len(self.layers)
            prev_hiddens = [encoder_hiddens[i] for i in range(num_layers)]
            prev_cells = [encoder_cells[i] for i in range(num_layers)]
            if self.encoder_hidden_proj is not None:
                prev_hiddens = [self.encoder_hidden_proj(x) for x in prev_hiddens]
                prev_cells = [self.encoder_cell_proj(x) for x in prev_cells]
            input_feed = x.new_zeros(bsz, self.hidden_size)

        attn_scores = x.new_zeros(srclen, seqlen, bsz)
        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            input = torch.cat((x[j, :, :], input_feed), dim=1)

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout_out, training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            if self.attention is not None:
                out, attn_scores[:, j, :] = self.attention(hidden, encoder_outs, encoder_padding_mask)
            else:
                out = hidden
            out = F.dropout(out, p=self.dropout_out, training=self.training)

            # input feeding
            input_feed = out

            # save final output
            outs.append(out)

        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self, incremental_state, 'cached_state',
            (prev_hiddens, prev_cells, input_feed),
        )

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        if not self.training and self.need_attn:
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None

        # project back to size of vocabulary
        if self.adaptive_softmax is None:
            if hasattr(self, 'additional_fc'):
                x = self.additional_fc(x)
                x = F.dropout(x, p=self.dropout_out, training=self.training)
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = self.fc_out(x)
        return x, attn_scores

    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is None:
            return

        def reorder_state(state):
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            return state.index_select(0, new_order)

        new_state = tuple(map(reorder_state, cached_state))
        utils.set_incremental_state(self, incremental_state, 'cached_state', new_state)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e5)  # an arbitrary large number

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

class CTCDecoder(nn.Module):
    def __init__(self, encoder_output_units=512, out_dim=512, hidden_size=512, num_layers=1):
        super().__init__()
        self.encoder_output_units = encoder_output_units
        self.rnn = LSTM(input_size=encoder_output_units, hidden_size=hidden_size, num_layers=num_layers)
        self.fc_out = Linear(hidden_size, out_dim)

    def forward(self, encoder_out):
        encoder_out = encoder_out['encoder_out']
        rnn_out = self.rnn(encoder_out)
        fc_out = self.fc_out(rnn_out)
        out = fc_out.log_softmax(2).detach().requires_grad_()
        return out




def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def Linear(in_features, out_features, bias=True, dropout=0):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m


@register_model_architecture('seg_nmt_ctc_lstm', 'seg_nmt_ctc_lstm')
def base_architecture(args):
    args.just_ctc = getattr(args, 'just_ctc', args.just_ctc)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.shared_encoder_embed_dim = getattr(args, 'shared_encoder_embed_dim', 512)
    args.shared_encoder_embed_path = getattr(args, 'shared_encoder_embed_path', None)
    args.shared_encoder_freeze_embed = getattr(args, 'shared_encoder_freeze_embed', False)
    args.shared_encoder_hidden_size = getattr(args, 'shared_encoder_hidden_size', args.shared_encoder_embed_dim)
    args.shared_encoder_layers = getattr(args, 'shared_encoder_layers', 1)
    args.shared_encoder_bidirectional = getattr(args, 'shared_encoder_bidirectional', False)
    args.shared_encoder_dropout_in = getattr(args, 'shared_encoder_dropout_in', args.dropout)
    args.shared_encoder_dropout_out = getattr(args, 'shared_encoder_dropout_out', args.dropout)
    args.nmt_encoder_embed_dim = getattr(args, 'nmt_encoder_embed_dim', 512)
    args.nmt_encoder_embed_path = getattr(args, 'nmt_encoder_embed_path', None)
    args.nmt_encoder_freeze_embed = getattr(args, 'nmt_encoder_freeze_embed', False)
    args.nmt_encoder_hidden_size = getattr(args, 'nmt_encoder_hidden_size', args.nmt_encoder_embed_dim)
    args.nmt_encoder_layers = getattr(args, 'nmt_encoder_layers', 1)
    args.nmt_encoder_bidirectional = getattr(args, 'nmt_encoder_bidirectional', False)
    args.nmt_encoder_dropout_in = getattr(args, 'nmt_encoder_dropout_in', args.dropout)
    args.nmt_encoder_dropout_out = getattr(args, 'nmt_encoder_dropout_out', args.dropout)
    
    args.ctc_decoder_embed_dim = getattr(args, 'ctc_decoder_embed_dim', 512)
    args.ctc_decoder_embed_path = getattr(args, 'ctc_decoder_embed_path', None)
    args.ctc_decoder_freeze_embed = getattr(args, 'ctc_decoder_freeze_embed', False)
    args.ctc_decoder_hidden_size = getattr(args, 'ctc_decoder_hidden_size', args.ctc_decoder_embed_dim)
    args.ctc_decoder_layers = getattr(args, 'ctc_decoder_layers', 1)
    args.ctc_decoder_out_embed_dim = getattr(args, 'ctc_decoder_out_embed_dim', 512)
    args.ctc_decoder_attention = getattr(args, 'ctc_decoder_attention', '1')
    args.ctc_decoder_dropout_in = getattr(args, 'ctc_decoder_dropout_in', args.dropout)
    args.ctc_decoder_dropout_out = getattr(args, 'ctc_decoder_dropout_out', args.dropout)
    args.nmt_decoder_embed_dim = getattr(args, 'nmt_decoder_embed_dim', 512)
    args.nmt_decoder_embed_path = getattr(args, 'nmt_decoder_embed_path', None)
    args.nmt_decoder_freeze_embed = getattr(args, 'nmt_decoder_freeze_embed', False)
    args.nmt_decoder_hidden_size = getattr(args, 'nmt_decoder_hidden_size', args.nmt_decoder_embed_dim)
    args.nmt_decoder_layers = getattr(args, 'nmt_decoder_layers', 1)
    args.nmt_decoder_out_embed_dim = getattr(args, 'nmt_decoder_out_embed_dim', 512)
    args.nmt_decoder_attention = getattr(args, 'nmt_decoder_attention', '1')
    args.nmt_decoder_dropout_in = getattr(args, 'nmt_decoder_dropout_in', args.dropout)
    args.nmt_decoder_dropout_out = getattr(args, 'nmt_decoder_dropout_out', args.dropout)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', '10000,50000,200000')


@register_model_architecture('seg_nmt_ctc_lstm', 'my_ctc_nmt_lstm')
def my_ctc_nmt_lstm(args):
    args.just_ctc = getattr(args, 'just_ctc', args.just_ctc)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.shared_encoder_embed_dim = getattr(args, 'shared_encoder_embed_dim', args.shared_encoder_embed_dim)
    args.shared_encoder_hidden_size = getattr(args, 'shared_encoder_hidden_size', args.shared_encoder_hidden_size)
    args.shared_encoder_layers = getattr(args, 'shared_encoder_layers', args.shared_encoder_layers)
    args.shared_encoder_dropout_in = getattr(args, 'shared_encoder_dropout_in', 0)
    args.shared_encoder_dropout_out = getattr(args, 'shared_encoder_dropout_out', 0)

    args.nmt_encoder_embed_dim = getattr(args, 'nmt_encoder_embed_dim', args.nmt_encoder_embed_dim)
    args.nmt_encoder_hidden_size = getattr(args, 'nmt_encoder_hidden_size', args.nmt_encoder_hidden_size)
    args.nmt_encoder_layers = getattr(args, 'nmt_encoder_layers', args.nmt_encoder_layers)
    args.nmt_encoder_dropout_in = getattr(args, 'nmt_encoder_dropout_in', 0)
    args.nmt_encoder_dropout_out = getattr(args, 'nmt_encoder_dropout_out', 0)

    args.ctc_decoder_embed_dim = getattr(args, 'ctc_decoder_embed_dim', args.ctc_decoder_embed_dim)
    args.ctc_decoder_out_embed_dim = getattr(args, 'ctc_decoder_out_embed_dim', args.ctc_decoder_out_embed_dim)
    args.ctc_decoder_hidden_size = getattr(args, 'ctc_decoder_hidden_size', args.ctc_decoder_hidden_size)
    args.ctc_decoder_layers = getattr(args, 'ctc_decoder_layers', args.ctc_decoder_layers)
    args.ctc_decoder_dropout_in = getattr(args, 'ctc_decoder_dropout_in', 0)
    args.ctc_decoder_dropout_out = getattr(args, 'ctc_decoder_dropout_out', args.dropout)

    args.nmt_decoder_embed_dim = getattr(args, 'nmt_decoder_embed_dim', args.nmt_decoder_embed_dim)
    args.nmt_decoder_out_embed_dim = getattr(args, 'nmt_decoder_out_embed_dim', args.nmt_decoder_out_embed_dim)
    args.nmt_decoder_hidden_size = getattr(args, 'nmt_decoder_hidden_size', args.nmt_decoder_hidden_size)
    args.nmt_decoder_layers = getattr(args, 'nmt_decoder_layers', args.nmt_decoder_layers)
    args.nmt_decoder_dropout_in = getattr(args, 'nmt_decoder_dropout_in', 0)
    args.nmt_decoder_dropout_out = getattr(args, 'nmt_decoder_dropout_out', args.dropout)

    base_architecture(args)

