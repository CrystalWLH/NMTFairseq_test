'''
    CTC Segmentation and NMT Model.
    Author: Shaojun Gao & Lihui Wang
    Create Date: 2019-12-09
    Update Date: 2019-12-10
'''

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
import random

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_Seg_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model('seg_nmt_ctc')
class SegNmtCTCModel(FairseqEncoderDecoderDoubleModel):
    """
    seg_nmt model for training segmentation and NMT jointly.

    Args:
        shared_encoder(TransformerEncoder): seg and nmt shared encoder
        ctc_decoder (TransformerDecoder): the seg decoder for ctc
        nmt_encoder (TransformerEncoder): the nmt encoder
        nmt_decoder (TransformerDecoder): the nmt decoder

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """


    def __init__(self, shared_encoder, ctc_decoder, nmt_encoder, nmt_decoder):
        super().__init__()
        # self.shared_encoder = shared_encoder
        # self.ctc_decoder = ctc_decoder
        # self.nmt_encoder = nmt_encoder
        # self.nmt_decoder = nmt_decoder
        self.supports_align_args = True

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        #modified by Lihui Wang (2019-12-10)
        parser.add_argument('--shared-encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained shared encoder embedding')
        parser.add_argument('--shared-encoder-embed-dim', type=int, metavar='N',
                            help='shared encoder embedding dimension')
        parser.add_argument('--shared-encoder-ffn-embed-dim', type=int, metavar='N',
                            help='shared encoder embedding dimension for FFN')
        parser.add_argument('--shared encoder-layers', type=int, metavar='N',
                            help='num shared encoder layers')
        parser.add_argument('--shared-encoder-attention-heads', type=int, metavar='N',
                            help='num shared encoder attention heads')
        parser.add_argument('--shared encoder-normalize-before', action='store_true',
                            help='apply layernorm before each shared encoder block')
        parser.add_argument('--shared-encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the shared encoder')
        parser.add_argument('--ctc-decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained ctc decoder embedding')
        parser.add_argument('--ctc-decoder-embed-dim', type=int, metavar='N',
                            help='ctc decoder embedding dimension')
        parser.add_argument('--ctc-decoder-ffn-embed-dim', type=int, metavar='N',
                            help='ctc decoder embedding dimension for FFN')
        parser.add_argument('--ctc-decoder-layers', type=int, metavar='N',
                            help='num ctc decoder layers')
        parser.add_argument('--ctc-decoder-attention-heads', type=int, metavar='N',
                            help='num ctc decoder attention heads')
        parser.add_argument('--ctc-decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the ctc decoder')
        parser.add_argument('--ctc-decoder-normalize-before', action='store_true',
                            help='apply layernorm before each ctc decoder block')
        parser.add_argument('--nmt-encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained nmt encoder embedding')
        parser.add_argument('--nmt-encoder-embed-dim', type=int, metavar='N',
                            help='nmt encoder embedding dimension')
        parser.add_argument('--nmt-encoder-ffn-embed-dim', type=int, metavar='N',
                            help='nmt encoder embedding dimension for FFN')
        parser.add_argument('--nmt encoder-layers', type=int, metavar='N',
                            help='num nmt encoder layers')
        parser.add_argument('--nmt-encoder-attention-heads', type=int, metavar='N',
                            help='num nmt encoder attention heads')
        parser.add_argument('--nmt encoder-normalize-before', action='store_true',
                            help='apply layernorm before each nmt encoder block')
        parser.add_argument('--nmt-encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the nmt encoder')
        parser.add_argument('--nmt-decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained nmt decoder embedding')
        parser.add_argument('--nmt-decoder-embed-dim', type=int, metavar='N',
                            help='nmt decoder embedding dimension')
        parser.add_argument('--nmt-decoder-ffn-embed-dim', type=int, metavar='N',
                            help='nmt decoder embedding dimension for FFN')
        parser.add_argument('--nmt-decoder-layers', type=int, metavar='N',
                            help='num nmt decoder layers')
        parser.add_argument('--nmt-decoder-attention-heads', type=int, metavar='N',
                            help='num nmt decoder attention heads')
        parser.add_argument('--nmt-decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the nmt decoder')
        parser.add_argument('--nmt-decoder-normalize-before', action='store_true',
                            help='apply layernorm before each nmt decoder block')
        
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        parser.add_argument('--layer-wise-attention', default=False, action='store_true',
                            help='perform layer-wise attention (cross-attention or cross+self-attention)')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS
        #add by Lihui Wang (2019-12-10)
        if not hasattr(args, 'max_seg_positions'):
            args.max_seg_positions = DEFAULT_MAX_Seg_POSITIONS

        src_dict, ctc_dict, tgt_dict = task.source_dictionary, task.seg_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.shared_encoder_embed_dim != args.nmt_decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --shared-encoder-embed-dim to match --nmt-decoder-embed-dim')
            if args.nmt_decoder_embed_path and (
                    args.nmt_decoder_embed_path != args.shared_encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --nmt-decoder-embed-path')
            shared_encoder_embed_tokens = build_embedding(
                src_dict, args.shared_encoder_embed_dim, args.shared_encoder_embed_path
            )
            nmt_decoder_embed_tokens = shared_encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            shared_encoder_embed_tokens = build_embedding(
                src_dict, args.shared_encoder_embed_dim, args.shared_encoder_embed_path
            )
            ctc_decoer_embed_tokens = build_embedding(
                ctc_dict, args.ctc_decoder_embed_dim, args.ctc_embed_path
            )
            nmt_encoder_embed_tokens = build_embedding(
                ctc_dict, args.nmt_encoder_embed_dim, args.nmt_encoder_embed_path
            )
            nmt_decoder_embed_tokens = build_embedding(
                tgt_dict, args.nmt_decoder_embed_dim, args.nmt_decoder_embed_path
            )

        #questions!!!
        shared_encoder = cls.build_shared_encoder(args, src_dict, shared_encoder_embed_tokens)
        nmt_encoder = cls.build_nmt_encoder(args, src_dict, encoder_embed_tokens)
        nmt_decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        ctc_decoder = cls.build_ctc_decoder(args, input_dim, output_dim)
        return cls(shared_encoder, ctc_decoder, nmt_encoder, mmt_decoder)

    @classmethod
    def build_shared_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_nmt_encoder(cls, args, src_dict, embed_tokens):
        return NmtTransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, 'no_cross_attention', False),
        )

    @classmethod
    def build_ctc_decoder(args, input_dim, output_dim):
        return CtcDecoder(args, input_dim, output_dim)

    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample['ctc_target'], sample['nmt_target']

    def forward(args, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        shared_encoder_out = self.shared_encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        ctc_decoder_out = self.ctc_decoder(shared_encoder_out)
        nmt_encoder_out = self.nmt_encoder(shared_encoder_out, src_lengths)
        nmt_decoder_out = self.nmt_decoder(prev_output_tokens, encoder_out=nmt_encoder_out, **kwargs)
        return ctc_decoder_out, nmt_decoder_out

class CtcDecoder(nn.module):
    """
    input: (Tensor) seq_len, batch, input_dim
    output: (Tensor) seq_len, batch, words_num
    """
    def __init__(self, args, input_dim, output_dim):
        super().__init__()
        self.fc_layer = nn.Sequential(nn.Linear(input_dim, output_dim), nn.RELU(True))

    def forward(input):
        x = self.fc(input)
        return x

class NmtTransformerEncoder(nn.module):
    """
    Transformer NMT encoder, whose input is the output of shared encoder.
    """
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout
        self.encoder_layerdrop = args.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(self, input, src_lengths, cls_input=None, return_all_hiddens=False):
        """
        Args:
            x (Tensor): input (src_len, batch, input_dim)
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        if self.layer_wise_attention:
            return_all_hiddens = True

        #x, encoder_embedding = self.forward_embedding(src_tokens)

        # B x T x C -> T x B x C
        x = input.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if not self.training or (dropout_probability > self.encoder_layerdrop):
                x = layer(x, encoder_padding_mask)
                if return_all_hiddens:
                    encoder_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)
            if return_all_hiddens:
                encoder_states[-1] = x

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
            'encoder_states': encoder_states,  # List[T x B x C]
        }


class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout
        self.encoder_layerdrop = args.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None


    def forward_embedding(self, src_tokens):
        # embed tokens and positions
        embed = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, embed

    def forward(self, src_tokens, src_lengths, cls_input=None, return_all_hiddens=False):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        if self.layer_wise_attention:
            return_all_hiddens = True

        x, encoder_embedding = self.forward_embedding(src_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if not self.training or (dropout_probability > self.encoder_layerdrop):
                x = layer(x, encoder_padding_mask)
                if return_all_hiddens:
                    encoder_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)
            if return_all_hiddens:
                encoder_states[-1] = x

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
            'encoder_embedding': encoder_embedding,  # B x T x C
            'encoder_states': encoder_states,  # List[T x B x C]
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        if encoder_out.get('encoder_states', None) is not None:
            for idx, state in enumerate(encoder_out['encoder_states']):
                encoder_out['encoder_states'][idx] = state.index_select(1, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
            if self._future_mask.size(0) < dim:
                self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)
        for i in range(len(self.layers)):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(state_dict, "{}.layers.{}".format(name, i))

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None

        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, self.padding_idx,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.cross_self_attention = getattr(args, 'cross_self_attention', False)
        self.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(args, no_encoder_attn)
            for _ in range(args.decoder_layers)
        ])

        self.adaptive_softmax = None

        self.project_out_dim = Linear(embed_dim, self.output_embed_dim, bias=False) \
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights else None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), self.output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

        if args.decoder_normalize_before and not getattr(args, 'no_decoder_final_norm', False):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(
        self,
        prev_output_tokens,
        encoder_out=None,
        incremental_state=None,
        features_only=False,
        **extra_args,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state, **extra_args,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
        incremental_state=None,
        full_context_alignment=False,
        alignment_layer=None,
        alignment_heads=None,
        **unused,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = len(self.layers) - 1

        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)
        if not self_attn_padding_mask.any() and not self.cross_self_attention:
            self_attn_padding_mask = None

        # decoder layers
        attn = None
        inner_states = [x]
        for idx, layer in enumerate(self.layers):
            encoder_state = None
            if encoder_out is not None:
                if self.layer_wise_attention:
                    encoder_state = encoder_out['encoder_states'][idx]
                else:
                    encoder_state = encoder_out['encoder_out']

            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if not self.training or (dropout_probability > self.decoder_layerdrop):
                x, layer_attn = layer(
                    x,
                    encoder_state,
                    encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=(idx == alignment_layer),
                    need_head_weights=(idx == alignment_layer),
                )
                inner_states.append(x)
                if layer_attn is not None and idx == alignment_layer:
                    attn = layer_attn.float()

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {'attn': attn, 'inner_states': inner_states}

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                return F.linear(features, self.embed_tokens.weight)
            else:
                return F.linear(features, self.embed_out)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, '_future_mask')
            or self._future_mask is None
            or self._future_mask.device != tensor.device
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = '{}.layers.{}.layer_norms.{}.{}'.format(name, i, old, m)
                    if k in state_dict:
                        state_dict['{}.layers.{}.{}.{}'.format(name, i, new, m)] = state_dict[k]
                        del state_dict[k]

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


@register_model_architecture('transformer', 'transformer')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)
    args.no_cross_attention = getattr(args, 'no_cross_attention', False)
    args.cross_self_attention = getattr(args, 'cross_self_attention', False)
    args.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)


@register_model_architecture('transformer', 'transformer_iwslt_de_en')
def transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    base_architecture(args)


@register_model_architecture('transformer', 'transformer_wmt_en_de')
def transformer_wmt_en_de(args):
    base_architecture(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture('transformer', 'transformer_vaswani_wmt_en_de_big')
def transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.dropout = getattr(args, 'dropout', 0.3)
    base_architecture(args)


@register_model_architecture('transformer', 'transformer_vaswani_wmt_en_fr_big')
def transformer_vaswani_wmt_en_fr_big(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture('transformer', 'transformer_wmt_en_de_big')
def transformer_wmt_en_de_big(args):
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    transformer_vaswani_wmt_en_de_big(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture('transformer', 'transformer_wmt_en_de_big_t2t')
def transformer_wmt_en_de_big_t2t(args):
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture('transformer_align', 'transformer_align')
def transformer_align(args):
    args.alignment_heads = getattr(args, 'alignment_heads', 1)
    args.alignment_layer = getattr(args, 'alignment_layer', 4)
    args.full_context_alignment = getattr(args, 'full_context_alignment', False)
    base_architecture(args)


@register_model_architecture('transformer_align', 'transformer_wmt_en_de_big_align')
def transformer_wmt_en_de_big_align(args):
    args.alignment_heads = getattr(args, 'alignment_heads', 1)
    args.alignment_layer = getattr(args, 'alignment_layer', 4)
    transformer_wmt_en_de_big(args)
