"""Formula parsing module for R-style regression formulas.

This module provides formula parsing for regression models, supporting:
- Main effects: 'y ~ x1 + x2'
- Interactions: 'y ~ x1:x2' (product only) or 'y ~ x1 * x2' (main + interaction)
- Polynomials: 'y ~ poly(x, 2)' (centered) or 'y ~ poly(x, 2, raw=True)'
- Transforms: 'y ~ I(x^2)'

Example
-------
>>> from polars_statistics.formula import FormulaParser, expand_terms_to_expressions
>>>
>>> parser = FormulaParser()
>>> parsed = parser.parse("y ~ x1 + x2 + x1:x2")
>>> print(parsed.response)  # 'y'
>>> print(len(parsed.terms))  # 3
>>>
>>> expressions, names = expand_terms_to_expressions(parsed.terms)
>>> print(names)  # ['x1', 'x2', 'x1:x2']
"""

from polars_statistics.formula.expander import expand_terms_to_expressions
from polars_statistics.formula.parser import FormulaParser
from polars_statistics.formula.terms import (
    InteractionTerm,
    ParsedFormula,
    PolyTerm,
    SimpleTerm,
    Term,
    TransformTerm,
)

__all__ = [
    "FormulaParser",
    "expand_terms_to_expressions",
    "ParsedFormula",
    "Term",
    "SimpleTerm",
    "InteractionTerm",
    "PolyTerm",
    "TransformTerm",
]
