import os
import string
from pathlib import Path
from typing import Dict

import pynini
from fun_text_processing.inverse_text_normalization.ko.utils import get_abs_path
from pynini import Far
from pynini.examples import plurals
from pynini.export import export
from pynini.lib import byte, pynutil, utf8

DAMO_CHAR = utf8.VALID_UTF8_CHAR

DAMO_DIGIT = byte.DIGIT
DAMO_LOWER = pynini.union(*string.ascii_lowercase).optimize()
DAMO_UPPER = pynini.union(*string.ascii_uppercase).optimize()
DAMO_ALPHA = pynini.union(DAMO_LOWER, DAMO_UPPER).optimize()
DAMO_ALNUM = pynini.union(DAMO_DIGIT, DAMO_ALPHA).optimize()
DAMO_HEX = pynini.union(*string.hexdigits).optimize()
DAMO_NON_BREAKING_SPACE = "\u00A0"
DAMO_SPACE = " "
DAMO_WHITE_SPACE = pynini.union(" ", "\t", "\n", "\r", "\u00A0").optimize()
DAMO_NOT_SPACE = pynini.difference(DAMO_CHAR, DAMO_WHITE_SPACE).optimize()
DAMO_NOT_QUOTE = pynini.difference(DAMO_CHAR, r'"').optimize()

DAMO_PUNCT = pynini.union(*map(pynini.escape, string.punctuation)).optimize()
DAMO_GRAPH = pynini.union(DAMO_ALNUM, DAMO_PUNCT).optimize()

DAMO_SIGMA = pynini.closure(DAMO_CHAR)

delete_space = pynutil.delete(pynini.closure(DAMO_WHITE_SPACE))
delete_zero_or_one_space = pynutil.delete(pynini.closure(DAMO_WHITE_SPACE, 0, 1))
insert_space = pynutil.insert(" ")
delete_extra_space = pynini.cross(pynini.closure(DAMO_WHITE_SPACE, 1), " ")
delete_preserve_order = pynini.closure(
    pynutil.delete(" preserve_order: true")
    | (pynutil.delete(' field_order: "') + DAMO_NOT_QUOTE + pynutil.delete('"'))
)

suppletive = pynini.string_file(get_abs_path("data/suppletive.tsv"))
# _v = pynini.union("a", "e", "i", "o", "u")
_c = pynini.union(
    "b",
    "c",
    "d",
    "f",
    "g",
    "h",
    "j",
    "k",
    "l",
    "m",
    "n",
    "p",
    "q",
    "r",
    "s",
    "t",
    "v",
    "w",
    "x",
    "y",
    "z",
)
_ies = DAMO_SIGMA + _c + pynini.cross("y", "ies")
_es = DAMO_SIGMA + pynini.union("s", "sh", "ch", "x", "z") + pynutil.insert("es")
_s = DAMO_SIGMA + pynutil.insert("s")

graph_plural = plurals._priority_union(
    suppletive,
    plurals._priority_union(_ies, plurals._priority_union(_es, _s, DAMO_SIGMA), DAMO_SIGMA),
    DAMO_SIGMA,
).optimize()

SINGULAR_TO_PLURAL = graph_plural
PLURAL_TO_SINGULAR = pynini.invert(graph_plural)
TO_LOWER = pynini.union(
    *[pynini.cross(x, y) for x, y in zip(string.ascii_uppercase, string.ascii_lowercase)]
)
TO_UPPER = pynini.invert(TO_LOWER)
MIN_NEG_WEIGHT = -0.0001
MIN_POS_WEIGHT = 0.0001


def generator_main(file_name: str, graphs: Dict[str, "pynini.FstLike"]):
    """
    Exports graph as OpenFst finite state archive (FAR) file with given file name and rule name.

    Args:
        file_name: exported file name
        graphs: Mapping of a rule name and Pynini WFST graph to be exported
    """
    exporter = export.Exporter(file_name)
    for rule, graph in graphs.items():
        exporter[rule] = graph.optimize()
    exporter.close()
    print(f"Created {file_name}")


def get_plurals(fst):
    """
    Given singular returns plurals

    Args:
        fst: Fst

    Returns plurals to given singular forms
    """
    return SINGULAR_TO_PLURAL @ fst


def get_singulars(fst):
    """
    Given plural returns singulars

    Args:
        fst: Fst

    Returns singulars to given plural forms
    """
    return PLURAL_TO_SINGULAR @ fst


def convert_space(fst) -> "pynini.FstLike":
    """
    Converts space to nonbreaking space.
    Used only in tagger grammars for transducing token values within quotes, e.g. name: "hello kitty"
    This is making transducer significantly slower, so only use when there could be potential spaces within quotes, otherwise leave it.

    Args:
        fst: feats fst

    Returns output fst where breaking spaces are converted to non breaking spaces
    """
    return fst @ pynini.cdrewrite(
        pynini.cross(DAMO_SPACE, DAMO_NON_BREAKING_SPACE), "", "", DAMO_SIGMA
    )


class GraphFst:
    """
    Base class for all grammar fsts.

    Args:
        name: name of grammar class
        kind: either 'classify' or 'verbalize'
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, name: str, kind: str, deterministic: bool = True):
        self.name = name
        self.kind = kind
        self._fst = None
        self.deterministic = deterministic

        self.far_path = Path(os.path.dirname(__file__) + "/grammars/" + kind + "/" + name + ".far")
        if self.far_exist():
            self._fst = Far(
                self.far_path, mode="r", arc_type="standard", far_type="default"
            ).get_fst()

    def far_exist(self) -> bool:
        """
        Returns true if FAR can be loaded
        """
        return self.far_path.exists()

    @property
    def fst(self) -> "pynini.FstLike":
        return self._fst

    @fst.setter
    def fst(self, fst):
        self._fst = fst

    def add_tokens(self, fst) -> "pynini.FstLike":
        """
        Wraps class name around to given fst

        Args:
            fst: feats fst

        Returns:
            Fst: fst
        """
        return pynutil.insert(f"{self.name} {{ ") + fst + pynutil.insert(" }")

    def delete_tokens(self, fst) -> "pynini.FstLike":
        """
        Deletes class name wrap around output of given fst

        Args:
            fst: feats fst

        Returns:
            Fst: fst
        """
        res = (
            pynutil.delete(f"{self.name}")
            + delete_space
            + pynutil.delete("{")
            + delete_space
            + fst
            + delete_space
            + pynutil.delete("}")
        )
        return res @ pynini.cdrewrite(pynini.cross("\u00A0", " "), "", "", DAMO_SIGMA)
