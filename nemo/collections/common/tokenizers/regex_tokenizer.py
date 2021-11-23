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

import os
import re
from typing import Optional

from nemo.collections.common.tokenizers.char_tokenizer import TokenizerSpec

__all__ = ['RegExTokenizer']

DEFAULT_REGEX = r"""\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]"""

class RegExTokenizer(TokenizerSpec):
    "Tokenizes at word boundary defined thru regular expression"

    def __init__(
        self,
        vocab_file: str,
        regex: Optional[str] = None,
        mask_token: Optional[str] = '<MASK>',
        bos_token: Optional[str] = '^',
        eos_token: Optional[str] = '&',
        pad_token: Optional[str] = '<PAD>',
        sep_token: Optional[str] = '<SEP>',
        unk_token: Optional[str] = '?',
    ):
        """
        Args:
            vocab_file: path to file with vocabulary which consists
                of characters separated by \n
            mask_token: mask token
            bos_token: the beginning of sequence token
            eos_token: the end of sequence token. Usually equal to sep_token
            pad_token: token to use for padding
            sep_token: token used for separating sequences
            cls_token: class token. Usually equal to bos_token
            unk_token: token to use for unknown tokens
        """
        self.mask_token = mask_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.sep_token = sep_token
        self.unk_token = unk_token
        self.regex = regex if regex else DEFAULT_REGEX

        if not vocab_file or not os.path.exists(vocab_file):
            raise ValueError(f"Vocab file: {vocab_file} is invalid")
        self.vocab_file = vocab_file
        self.load_vocab()

        # Computed attributes
        self._compiled_regex = None
        self._compile_regex()

        ## Cache data/attributes required for tokenization
        self._unk_id = self.vocab.get(unk_token, '?')
        self._decode_vocab = {i: t for t, i in self.vocab.items()}

    def _compile_regex(self):
        regex_string = r"("

        regex_string += self.regex + r"|"
        regex_string += r".)"
        self._compiled_regex = re.compile(regex_string)

    def text_to_tokens(self, text):
        # Begin token
        tokens = [self.bos_token]
        tokens.extend(self._compiled_regex.findall(text))
        # End token
        tokens.append(self.eos_token)

        return tokens

    def tokens_to_text(self, tokens):
        tokens_list = []
        for token in tokens:
            if token[0] == self.bos_token:
                token = token[1:]

            # Remove end token and the following values
            if self.eos_token in token:
                eos_idx = token.index(self.eos_token)
                token = token[:eos_idx]

            tokens_list.append(token)

        text = ["".join(tokens) for tokens in tokens_list]
        return text

    def token_to_ids(self, tokens):
        ids_list = []
        for token in tokens:
            ids_list.append(self.vocab.get(token, self._unk_id))
        return ids_list

    def tokens_to_ids(self, token_data):
        if isinstance(token_data, str):
            token_data = [token_data]

        ids_list = []
        for tokens in token_data:
            ids = self.token_to_ids(tokens)
            ids_list.append(ids)
        return ids_list

    def ids_to_tokens(self, ids):
        tokens_list = []
        for ids in ids:
            for token_id in ids:
                token = self._decode_vocab.get(token_id)
                if token is None:
                    raise ValueError(f"Token id {token_id} is not recognised")

            tokens = [self._decode_vocab.get(token_id) for token_id in ids]
            tokens_list.append(tokens)

        return tokens_list

    def text_to_ids(self, text):
        tokens = self.text_to_tokens(text)
        tokens = [tokens]
        return self.tokens_to_ids(tokens)[0]

    def ids_to_text(self, ids):
        tokens = self.ids_to_tokens(ids)
        return self.tokens_to_text(tokens)

    def load_vocab(self):
        vocab = {}
        with open(self.vocab_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    vocab[line] = len(vocab)
        self.vocab = vocab

    @staticmethod
    def create_vocab(data_csv_file, vocab_file, smiles_col="smiles"):
        import pandas as pd
        if not os.path.exists(data_csv_file):
            raise ValueError(f"Data file: {data_csv_file} is invalid")

        # Create empty vocab file
        if not os.path.exists(vocab_file):
            fp = open(vocab_file, 'w')
            fp.close()

        df = pd.read_csv(data_csv_file)
        tokenizer = RegExTokenizer(vocab_file=vocab_file)

        vocab = {
            '<PAD>' : 0, # pad_token
            '?'     : 1, # unk_token
            '^'     : 2, # begin_token
            '&'     : 3, # end_token
            '<MASK>': 4, # mask_token
            '<SEP>' : 5  # sep_token
        }
        for smiles in df[smiles_col]:
            tokens = tokenizer.text_to_tokens(smiles)
            print(smiles, tokens)
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)

        vocab = sorted(vocab.items(), key=lambda k_v: k_v[1])
        print(vocab)
        with open(vocab_file, 'w') as fp:
            for token in vocab:
                fp.write(f"{token[0]}\n")