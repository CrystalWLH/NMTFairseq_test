'''
    CTC Segmentation and NMT loss.
    Author: Shaojun Gao
    Create Date: 2019-12-09
    Update Date: 2020-01-09
'''

import math
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('ctc_nmt_loss')
class CtcNmtCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self._ctc_weight = args.ctc_weight
        self._nmt_weight = args.nmt_weight


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # if self.args.just_ctc:
        #     ctc_out = model(**sample['net_input'], True)
        #     ctc_loss, _ = self.compute_ctc_loss(model, ctc_out, sample, reduce=reduce)
        #     sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        #     logging_output = {
        #         'ctc_loss': utils.item(ctc_loss.data) if reduce else ctc_loss.data,
        #         'nmt_loss': 0,
        #         'nll_loss': 0,
        #         'ntokens': sample['ntokens'],
        #         'nsentences': sample['target'].size(0),
        #         'sample_size': sample_size,
        #     }
        #     return ctc_loss, sample_size, logging_output
        #else:
        ctc_out, nmt_out = model(**sample['net_input'], False)
        ctc_loss, _ = self.compute_ctc_loss(model, ctc_out, sample, reduce=reduce)
        nmt_loss, _ = self.compute_nmt_loss(model, nmt_out, sample, reduct=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'ctc_loss': utils.item(ctc_loss.data) if reduce else ctc_loss.data,
            'nmt_loss': utils.item(nmt_loss.data) if reduce else nmt_loss.data,
            'nll_loss': utils.item(nmt_loss.data) if reduce else nmt_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        #TODO: all loss = ctcloss * alpla + nmtloss * beta
        loss = ctc_loss * self._ctc_weight + nmt_loss * self._nmt_weight
        return loss , sample_size, logging_output

    def compute_nmt_loss(self, model, nmt_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(nmt_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        _, nmt_target = model.get_targets(sample, nmt_output).view(-1)
        loss = F.nll_loss(
            lprobs,
            nmt_target,
            ignore_index=self.padding_idx,
            reduction='sum' if reduce else 'none',
        )
        return loss, loss

    def compute_ctc_loss(self, model, ctc_output, sample, reduct=True):
        ctc_loss = torch.nn.CTCLoss(blank=len(self.task.seg_dict), reduction='mean')
        seg_target, _ = model.get_targets(sample, num_output).view(-1)
        src_len = sample['net_input']['src_lengths']
        seg_len = sample['net_input']['segmentation_lengths']
        loss = ctc_loss(ctc_output, seg_target, src_len, seg_len)
        return loss, loss


    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ctc_loss_sum = sum(log.get('ctc_loss', 0) for log in logging_outputs)
        nmt_loss_sum = sum(log.get('nmt_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'ctc_loss': ctc_loss_sum / sample_size / math.log(2) if sample_size > 0 else 0.,
            'nmt_loss': nmt_loss_sum / sample_size / math.log(2) if sample_size > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = nmt_loss_sum / ntokens / math.log(2)
        return agg_output
