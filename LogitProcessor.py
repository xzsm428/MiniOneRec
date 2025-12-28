from transformers.generation import LogitsProcessor
from transformers import AutoTokenizer
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import math
import numpy as np
import torch
import warnings

from transformers.utils import add_start_docstrings

LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search

    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

"""

class ConstrainedLogitsProcessor(LogitsProcessor):

    def __init__(
        self,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        num_beams: int,
        base_model: str = None,
        eos_token_id: int = None
    ):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams
        self.count=0
        self.base_model = base_model
        self.eos_token_id = eos_token_id
        if self.base_model.lower().find("gpt2") > -1:
            self.prefix_index = 4
        else:
            self.prefix_index = 3

    
    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nn.functional.log_softmax(scores, dim=-1) # [batch_size*beam_size, vocab_size],这个是原始的softmax后的概率分布值
        mask = torch.full_like(scores, float('-inf')) # [batch_size*beam_size, vocab_size]，一开始把所有token的概率都设为无穷小
        
        # input_ids:[batch_size*beam_size, sequence_length] -> [batch_size, beam_size, sequence_length]
        # beam_size就是num_generations
        # TODO: sequence_length是什么？token的emb维度？
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent): # 遍历beam search的每一轮（即每组num_generations）
                if self.count == 0: # count初始为0，表示当前beam search刚开始（当前生成的是这组问题的第一个token），每一轮token生成后count+1
                    hash_key = sent[-self.prefix_index:] # 以'### Response:\n '为prefix
                else:
                    hash_key=sent[-self.count:] # 取已有前缀
                hash_key = hash_key.tolist()
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, hash_key) # 查哈希表得到可选token

                if len(prefix_allowed_tokens) == 0: # 无可选token，说明前缀完全不合法
                    warnings.warn(
                        f"No valid tokens found for hash_key {hash_key} at step {self.count}. "
                        f"This indicates the model generated an unexpected token. "
                    )
                    # Force EOS token to end invalid sequence
                    if self.eos_token_id is not None:
                        mask[batch_id * self._num_beams + beam_id, self.eos_token_id] = 0
                    continue 
                
                # 把mask中prefix_allowed_tokens对应的token的概率设为0，而不是原来的-1000000
                mask[batch_id * self._num_beams + beam_id, prefix_allowed_tokens] = 0

        self.count += 1

        scores = scores + mask # 把原始的概率分布值加上mask后，那些合法的token不变，不合法的加上-1000000后就变得非常小了
        return scores