"""Term dataclasses for formula parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union


@dataclass
class SimpleTerm:
    """A simple variable term like 'x1'."""

    name: str


@dataclass
class InteractionTerm:
    """An interaction term like 'x1:x2' (product of variables)."""

    variables: List[str]


@dataclass
class PolyTerm:
    """A polynomial term like 'poly(x, 2)'."""

    variable: str
    degree: int
    raw: bool = False  # If True, use raw polynomials (x, x^2); if False, center them


@dataclass
class TransformTerm:
    """An explicit transform like 'I(x^2)'."""

    variable: str
    power: int


# Union of all term types
Term = Union[SimpleTerm, InteractionTerm, PolyTerm, TransformTerm]


@dataclass
class ParsedFormula:
    """Result of parsing a formula string."""

    response: str  # The response variable (left of ~)
    terms: List[Term]  # The predictor terms (right of ~)
