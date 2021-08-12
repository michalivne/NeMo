# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.de.graph_utils import GraphFst, insert_space
from nemo_text_processing.inverse_text_normalization.de.utils import get_abs_path

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone numbers, e.g. 
        null vier eins eins eins zwei drei vier eins zwei drei vier -> { number_part: "(0411) 1234-1234" }
    """

    def __init__(self):
        super().__init__(name="telephone", kind="classify")
        # country code, number_part, extension
        separator = pynini.accep(" ")  # between components
        zero = pynini.invert(pynini.string_file(get_abs_path("data/numbers/zero.tsv")))
        digit = (pynini.invert(pynini.string_file(get_abs_path("data/numbers/digit.tsv"))) | zero).optimize()

        number_part = (
            pynutil.delete("(")
            + zero
            + insert_space
            + pynini.closure(digit + insert_space, 2, 2)
            + digit
            + pynutil.delete(")")
            + separator
            + pynini.closure(digit + insert_space, 3, 3)
            + digit
            + pynutil.delete("-")
            + insert_space
            + pynini.closure(digit + insert_space, 3, 3)
            + digit
        )
        number_part = pynutil.insert("number_part: \"") + pynini.invert(number_part) + pynutil.insert("\"")

        graph = number_part
        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
