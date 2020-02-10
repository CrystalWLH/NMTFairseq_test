'''
    Segmentation + NMT task
    Author: Lihui Wang
    Create Date: 2019-11-27
    Update Date: 2019-12-09
'''

import itertools
import os

from fairseq import options, utils
from fairseq.data import (
    ConcatDataset,
    data_utils,
    indexed_dataset,
    LanguagePairDataset,
    LanguageTraidDataset,
    PrependTokenDataset,
)
from . import FairseqTask, register_task


def load_langpair_dataset(
    data_path, split,
    src, src_dict,
    seg, seg_dict,
    tgt, tgt_dict,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_seg, left_pad_target, max_source_positions,
    max_seg_positions, max_target_positions, prepend_bos=False, load_alignments=False,
):
    def split_exists(split, src, seg, tgt, lang, data_path):
        filename = os.path.join(data_path, '{}.{}-{}-{}.{}'.format(split, src, seg, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)
    src_datasets = []
    seg_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, seg, tgt, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}-{}.'.format(split_k, src, seg, tgt))
        elif split_exists(split_k, tgt, seg, src, src, data_path):
            prefix = os.path.join(data_path, '{}.{}-{}-{}.'.format(split_k, tgt, seg, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        src_datasets.append(
            data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl)
        )
        seg_datasets.append(
            data_utils.load_indexed_dataset(prefix + seg, seg_dict, dataset_impl)        
        )
        tgt_datasets.append(
            data_utils.load_indexed_dataset(prefix + tgt, tgt_dict, dataset_impl)
        )

        print('| {} {} {}-{}-{} {} examples'.format(data_path, split_k, src, seg, tgt, len(src_datasets[-1])))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets), len(src_datasets)  == len(seg_datasets)

    if len(src_datasets) == 1:
        src_dataset, seg_dataset, tgt_dataset = src_datasets[0], seg_datasets[0], tgt_datasets[0]
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        seg_dataset = ConcatDataset(seg_datasets, sample_ratios)
        tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index") and hasattr(seg_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        seg_dataset = PrependTokenDataset(seg_dataset, seg_dict.bos())
        tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, '{}.align.{}-{}-{}'.format(split, src, seg, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(align_path, None, dataset_impl)

    return LanguageTraidDataset(
        src_dataset, src_dataset.sizes, src_dict,
        seg=seg_dataset, seg_sizes=seg_dataset.sizes, seg_dict=seg_dict,
        tgt=tgt_dataset, tgt_sizes=tgt_dataset.sizes, tgt_dict=tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
        align_dataset=align_dataset,
    )


@register_task('seg_translation')
class SegTranslationTask(FairseqTask):
    """
    Segement one (source) language to source word sequence, then translate from the source word sequences to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source char language
        seg_dict (~fairseq.data.Dictionary): dictionary for the source word language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The seg_translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The seg_translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source char language')
        parser.add_argument('-c', '--seg-lang', default=None, metavar='CTC',
                            help='source word language (CTC)')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--load-alignments', action='store_true',
                            help='load the binarized alignments')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-seg', default='True', type=str, metavar='BOOL',
                            help='pad the seg on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-seg-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the seg sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')

        # fmt: on

    def __init__(self, args, src_dict, seg_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.seg_dict = seg_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_seg = options.eval_bool(args.left_pad_seg)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        if getattr(args, 'raw_text', False):
            utils.deprecation_warning('--raw-text is deprecated, please use --dataset-impl=raw')
            args.dataset_impl = 'raw'
        elif getattr(args, 'lazy_load', False):
            utils.deprecation_warning('--lazy-load is deprecated, please use --dataset-impl=lazy')
            args.dataset_impl = 'lazy'

        paths = args.data.split(':')
        assert len(paths) > 0
        # find language pair automatically
        #if args.source_lang is None or args.seg_lang or args.target_lang is None:
        #    args.source_lang, args.seg_lang, args.target_lang = data_utils.infer_language_pair(paths[0])

        # Must give source_lang seg_lang target_lang mannuly
        if args.source_lang is None or args.seg_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        seg_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.seg_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad(), src_dict.pad() == seg_dict.pad()
        assert src_dict.eos() == tgt_dict.eos(), src_dict.eos() == seg_dict.pad()
        assert src_dict.unk() == tgt_dict.unk(), src_dict.unk() == seg_dict.pad()
        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.seg_lang, len(seg_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, seg_dict, tgt_dict)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]


        # infer langcode
        src, seg, tgt = self.args.source_lang, self.args.seg_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, seg, self.seg_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_seg=self.args.left_pad_seg,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_seg_positions=self.args.max_seg_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=self.args.load_alignments,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return LanguageTraidDataset(src_tokens, src_lengths, self.source_dictionary)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_seg_positions, self.args.max_target_positions)

    def build_generator(self, args):
        # if getattr(args, 'score_reference', False):
        #     from fairseq.sequence_scorer import SequenceScorer
        #     return SequenceScorer(self.target_dictionary)
        # else:
        #     from fairseq.sequence_generator import SequenceGenerator, SequenceGeneratorWithAlignment
        #     if getattr(args, 'print_alignment', False):
        #         seq_gen_cls = SequenceGeneratorWithAlignment
        #     else:
        #         seq_gen_cls = SequenceGenerator
        from fairseq.sequence_generator import SequenceGeneratorCTC
        seq_gen_cls = SequenceGeneratorCTC
        return seq_gen_cls(
            self.target_dictionary,
            beam_size=getattr(args, 'beam', 5),
            max_len_a=getattr(args, 'max_len_a', 0),
            max_len_b=getattr(args, 'max_len_b', 200),
            min_len=getattr(args, 'min_len', 1),
            normalize_scores=(not getattr(args, 'unnormalized', False)),
            len_penalty=getattr(args, 'lenpen', 1),
            unk_penalty=getattr(args, 'unkpen', 0),
            sampling=getattr(args, 'sampling', False),
            sampling_topk=getattr(args, 'sampling_topk', -1),
            sampling_topp=getattr(args, 'sampling_topp', -1.0),
            temperature=getattr(args, 'temperature', 1.),
            diverse_beam_groups=getattr(args, 'diverse_beam_groups', -1),
            diverse_beam_strength=getattr(args, 'diverse_beam_strength', 0.5),
            match_source_len=getattr(args, 'match_source_len', False),
            no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
        )
    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def seg_dictionary(self):
        """Return the source word:class:`~fairseq.data.Dictionary`."""
        return self.seg_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

