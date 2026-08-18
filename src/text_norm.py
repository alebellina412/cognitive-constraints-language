#!/usr/bin/env python3
"""Tokenisation shared by every corpus, matching the SPGC convention.

SPGC stores lowercase word tokens with punctuation removed and apostrophes
treated as separators — verified against the raw counts files: "don't" is stored
as `don` + `t`, "Seneca's" as `seneca` + `s`, French "d'eau" as `d` + `eau`.

Any other corpus we compare with SPGC (COREFL, for Figure S4) has to be tokenised
the same way, otherwise the vocabulary sizes are not comparable. Applying it also
removes the COREFL transcription markers for free: the pause marker "/" and
trailing punctuation such as "baby." disappear, while fillers ("uh") — which are
language, not annotation — are kept.
"""

from __future__ import annotations

import re

#: everything that is not a letter (accented included) or a digit separates tokens
SEPARATOR = re.compile(
    r"[^0-9a-zà-öø-ÿœæßñçğışâêîôûäëïöüáéíóúý]+", re.IGNORECASE
)


def tokens(text: str) -> list[str]:
    """SPGC-style token list of a raw text."""
    return [t for t in SEPARATOR.split(text.lower()) if t]


def phrase_tokens(surface: str) -> tuple[str, ...]:
    """Same, as a tuple — for looking a printed phrase up as an n-gram."""
    return tuple(tokens(surface))
