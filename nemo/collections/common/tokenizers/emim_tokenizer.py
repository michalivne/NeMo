# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from pathlib import Path

import youtokentome as yttm

from nemo.collections.common.tokenizers import TokenizerSpec

__all__ = ['EmbeddingMIMTokenizer']


class EmbeddingMIMTokenizer(TokenizerSpec):
    """
    A wrapper around a character-level word SentenceMIM (i.e., EmbeddingMIM).
    """
    def __init__(self, smim):
        self.smim_voc = smim.voc
        self.vocab_size = self.smim_voc.size
        self.special_tokens = self.tokens_to_ids(["<PAD>", "<UNK>", "<BOS>", "<EOS>"])

    def text_to_tokens(self, text):
        raise NotImplementedError("Not supported in this class")

    def tokens_to_text(self, tokens):
        raise NotImplementedError("Not supported in this class")

    def text_to_ids(self, text):
        return self.smim.encode(text)

    def ids_to_text(self, ids):
        ids_ = [id_ for id_ in ids if id_ not in self.special_tokens]
        return self.smim.decode(ids_)

    def tokens_to_ids(self, tokens):
        raise NotImplementedError("Not supported in this class")

    def ids_to_tokens(self, ids):
        raise NotImplementedError("Not supported in this class")

    @property
    def pad_id(self):
        return self.smim_voc.pad_idx

    @property
    def bos_id(self):
        return self.smim_voc.bot_idx

    @property
    def eos_id(self):
        return self.smim_voc.eot_idx

    @property
    def unk_id(self):
        return self.smim_voc.unk_idx
