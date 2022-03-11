# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""T5 Style dataset."""

import collections

import numpy as np
import torch

from nemo.collections.common.tokenizers import SentencePieceTokenizer, YouTokenToMeTokenizer
from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import (
    create_masked_lm_predictions,
    get_samples_mapping,
)
from nemo.collections.nlp.data.language_modeling.megatron.megatron_dataset import MegatronDataset


class BARTDataset(MegatronDataset):
    def __init__(
        self,
        cfg,
        trainer,
        tokenizer,
        name,
        indexed_dataset,
        data_prefix,
        num_epochs,
        max_num_samples,
        max_seq_length,
        seed,
        masked_lm_prob=0.15,
        short_seq_prob=0.1,
        max_ngram_size=10,
        mean_ngram_size=None,
        geometric_dist=True,
        permutation=False,
        whole_word_masking=True,
        favor_long_ngrams=False,
        delete_mask_prob=0,
    ):
        super().__init__(cfg, trainer=trainer)

        # Params to store.
        self.name = name
        self.seed = seed
        self.masked_lm_prob = masked_lm_prob
        self.max_seq_length = max_seq_length
        self.short_seq_prob = short_seq_prob
        self.max_ngram_size = max_ngram_size
        self.mean_ngram_size = mean_ngram_size
        self.geometric_dist = geometric_dist
        self.permutation = permutation
        self.whole_word_masking = whole_word_masking
        self.favor_long_ngrams = favor_long_ngrams
        self.delete_mask_prob = delete_mask_prob

        # Dataset.
        self.indexed_dataset = indexed_dataset

        # Build the samples mapping.
        self.samples_mapping = get_samples_mapping(
            indexed_dataset=self.indexed_dataset,
            data_prefix=data_prefix,
            num_epochs=num_epochs,
            max_num_samples=max_num_samples,
            max_seq_length=self.max_seq_length - 2,  # account for added tokens
            short_seq_prob=self.short_seq_prob,
            seed=self.seed,
            name=self.name,
            binary_head=False,
        )

        self.tokenizer = tokenizer
        self.tokenizer_type = 'wordpiece'  # TODO: better checks for tokenizer types. How do we do this for HF tokenizers that are not BERT?
        if isinstance(self.tokenizer, YouTokenToMeTokenizer):
            raise ValueError(f"YTTM does not support special tokens and cannot be used with T5 datasets.")

        if isinstance(self.tokenizer, SentencePieceTokenizer):
            if not self.tokenizer.legacy:
                raise ValueError("Sentencepiece Tokenizer must have legacy = False to add special tokens.")
            self.tokenizer_type = 'sentencepiece'

        self.cls_id = tokenizer.cls_id
        self.sep_id = tokenizer.sep_id
        self.mask_id = tokenizer.mask_id
        self.pad_id = tokenizer.pad_id
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id

        self.vocab_id_list = self.tokenizer.vocab
        self.vocab_id_to_token_dict = {idx: token for idx, token in enumerate(self.vocab_id_list)}

    def __len__(self):
        return self.samples_mapping.shape[0]

    def __getitem__(self, idx):

        start_index, end_index, seq_length = self.samples_mapping[idx]
        sample = []
        for index in range(start_index, end_index):
            sample.append(self.indexed_dataset[index])
        # Note that this rng state should be numpy and not python since
        # python randint is inclusive whereas the numpy one is exclusive.
        np_rng = np.random.RandomState(seed=(self.seed + idx))
        training_sample = build_training_sample(
            sample=sample,
            target_seq_length=seq_length,
            max_seq_length=self.max_seq_length,
            vocab_id_list=self.vocab_id_list,
            vocab_id_to_token_dict=self.vocab_id_to_token_dict,
            cls_id=self.cls_id,
            sep_id=self.sep_id,
            mask_id=self.mask_id,
            pad_id=self.pad_id,
            masked_lm_prob=self.masked_lm_prob,
            max_ngram_size=self.max_ngram_size,
            mean_ngram_size=self.mean_ngram_size,
            geometric_dist=self.geometric_dist,
            permutation=self.permutation,
            whole_word_masking=self.whole_word_masking,
            favor_long_ngrams=self.favor_long_ngrams,
            np_rng=np_rng,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            tokenizer_type=self.tokenizer_type,
            delete_mask_prob=self.delete_mask_prob,
        )
        return training_sample


def build_training_sample(
    sample,
    target_seq_length,
    max_seq_length,
    vocab_id_list,
    vocab_id_to_token_dict,
    cls_id,
    sep_id,
    mask_id,
    pad_id,
    masked_lm_prob,
    max_ngram_size,
    mean_ngram_size,
    geometric_dist,
    permutation,
    whole_word_masking,
    favor_long_ngrams,
    np_rng,
    bos_id=None,
    eos_id=None,
    tokenizer_type="wordpiece",
    delete_mask_prob=0,
):
    """Build BART training sample.
    Arguments:
        sample: A list of sentences in which each sentence is a list token ids.
        target_seq_length: Desired sequence length.
        max_seq_length: Maximum length of the sequence. All values are padded to
            this length.
        vocab_id_list: List of vocabulary ids. Used to pick a random id.
        vocab_id_to_token_dict: A dictionary from vocab ids to text tokens.
        cls_id: Start of example id.
        sep_id: Separator id.
        mask_id: Mask token id.
        pad_id: Padding token id.
        masked_lm_prob: Probability to mask tokens.
        max_ngram_size: maximum size of ngrams to be masked.
        mean_ngram_size: mean size of ngrams to be masked (only used if geometric_dist=True).
        geometric_dist: Uses a geometric distribution to sample ngram size.
        permutation: Permutes the ngrams.
        whole_word_masking: Always masks entire words instead of individual sub-word tokens.
        favor_long_ngrams: Favor longer ngrams over shorter ones.
        np_rng: Random number genenrator. Note that this rng state should be
              numpy and not python since python randint is inclusive for
              the opper bound whereas the numpy one is exclusive.
        bos_id: start of decoder example id
        eos_id: end of generation id
        tokenizer_type: wordpiece (BERT-style) or sentencepiece tokenizer. Used for whole word masking logic.
        delete_mask_prob: probability of deleting a mask
    """
    assert target_seq_length <= max_seq_length

    # flatten sentences into one list
    tokens = [token for sentence in sample for token in sentence]

    # Truncate to `target_sequence_length`.
    max_num_tokens = target_seq_length
    truncated = len(tokens) > max_num_tokens
    tokens = tokens[:max_num_tokens]

    # Masking.
    max_predictions_per_seq = max_num_tokens
    (output_tokens, masked_positions, masked_labels, _, masked_spans) = create_masked_lm_predictions(
        tokens=tokens,
        vocab_id_list=vocab_id_list,
        vocab_id_to_token_dict=vocab_id_to_token_dict,
        masked_lm_prob=masked_lm_prob,
        cls_id=cls_id,
        sep_id=sep_id,
        mask_id=mask_id,
        max_predictions_per_seq=max_predictions_per_seq,
        np_rng=np_rng,
        max_ngram_size=max_ngram_size,
        whole_word_masking=whole_word_masking,
        favor_long_ngrams=favor_long_ngrams,
        mean_ngram_size=mean_ngram_size,
        permutation=permutation,
        geometric_dist=geometric_dist,
        masking_style="bart",
        tokenizer_type=tokenizer_type,
    )

    # Padding.
    tokens_enc, tokens_dec_in, labels, enc_mask, dec_mask, loss_mask = pad_and_convert_to_numpy(
        tokens,
        output_tokens,
        masked_positions,
        masked_labels,
        pad_id,
        mask_id,
        max_seq_length,
        masked_spans,
        bos_id,
        eos_id,
        delete_mask_prob=delete_mask_prob,
        np_rng=np_rng,
    )

    train_sample = {
        'text_enc': tokens_enc,
        'text_dec': tokens_dec_in,
        'labels': labels,
        'loss_mask': loss_mask,
        'truncated': int(truncated),
        'enc_mask': enc_mask,
        'dec_mask': dec_mask,
    }
    return train_sample


def pad_and_convert_to_numpy(
    tokens,
    output_tokens,
    masked_positions,
    masked_labels,
    pad_id,
    mask_id,
    max_seq_length,
    masked_spans=None,
    bos_id=None,
    eos_id=None,
    delete_mask_prob=0,
    np_rng=None,
):
    """Pad sequences and convert them to numpy."""
    bart_decoder_in = [bos_id] + tokens[:-1]
    bart_decoder_out = tokens

    # construct bart input by collapsing multiple <mask> into one, and delete randomly
    bart_input = []
    (start_index, end_index) = (0, None)
    for span in masked_spans:
        end_index = span.index[0]
        bart_input.extend(output_tokens[start_index:end_index])
        # delete mask with probability delete_mask_prob
        if np_rng.rand() >= delete_mask_prob:
            bart_input.append(mask_id)

        # the next start index is the token after the last span token
        start_index = span.index[-1] + 1

    # Add the remaining tokens to the BART input
    bart_input.extend(output_tokens[start_index:])


    # Some checks.
    # Encoder-side padding mask.
    num_tokens = len(bart_input)
    padding_length = max_seq_length - num_tokens
    assert padding_length >= 0
    assert len(masked_positions) == len(masked_labels)

    # Tokens..
    filler = [pad_id] * padding_length
    tokens_enc = np.array(bart_input + filler, dtype=np.int64)

    # Decoder-side padding mask.
    num_tokens_dec = len(bart_decoder_in)
    padding_length_dec = max_seq_length - num_tokens_dec
    assert padding_length_dec >= 0
    filler_dec = [pad_id] * padding_length_dec
    tokens_dec_in = np.array(bart_decoder_in + filler_dec, dtype=np.int64)

    # Create attention masks
    enc_mask = (tokens_enc != pad_id).astype(np.int64)
    dec_mask = (tokens_dec_in != pad_id).astype(np.int64)

    # Labels mask.
    labels = bart_decoder_out + ([-1] * padding_length_dec)
    labels = np.array(labels, dtype=np.int64)

    # Loss mask
    loss_mask = ([1] * num_tokens_dec) + ([0] * padding_length_dec)
    loss_mask = np.array(loss_mask, dtype=np.int64)

    return tokens_enc, tokens_dec_in, labels, enc_mask, dec_mask, loss_mask
