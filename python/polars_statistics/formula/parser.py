"""Formula parser for R-style regression formulas."""

from __future__ import annotations

import re
from typing import List, Union

from polars_statistics.formula.terms import (
    InteractionTerm,
    ParsedFormula,
    PolyTerm,
    SimpleTerm,
    Term,
    TransformTerm,
)


class FormulaParser:
    """Parse R-style formula strings like 'y ~ x1 + x2 + x1:x2'.

    Supported syntax:
    - Main effects: 'y ~ x1 + x2'
    - Interaction only: 'y ~ x1:x2'
    - Full crossing (main + interaction): 'y ~ x1 * x2' -> x1 + x2 + x1:x2
    - Polynomial: 'y ~ poly(x, 2)' or 'y ~ poly(x, 2, raw=True)'
    - Explicit transform: 'y ~ I(x^2)'
    """

    # Pattern for poly(var, degree) or poly(var, degree, raw=True/False)
    POLY_PATTERN = re.compile(
        r"poly\(\s*(\w+)\s*,\s*(\d+)\s*(?:,\s*raw\s*=\s*(True|False))?\s*\)"
    )

    # Pattern for I(var^power)
    TRANSFORM_PATTERN = re.compile(r"I\(\s*(\w+)\s*\^\s*(\d+)\s*\)")

    def parse(self, formula: str) -> ParsedFormula:
        """Parse a formula string into a ParsedFormula.

        Parameters
        ----------
        formula : str
            Formula string like 'y ~ x1 + x2 + x1:x2'

        Returns
        -------
        ParsedFormula
            Parsed formula with response variable and list of terms

        Raises
        ------
        ValueError
            If formula syntax is invalid
        """
        if "~" not in formula:
            raise ValueError("Formula must contain '~' separator (e.g., 'y ~ x1 + x2')")

        lhs, rhs = formula.split("~", 1)
        response = lhs.strip()

        if not response:
            raise ValueError("Formula must have a response variable before '~'")

        rhs = rhs.strip()
        if not rhs:
            raise ValueError("Formula must have predictor terms after '~'")

        terms = self._parse_terms(rhs)
        return ParsedFormula(response=response, terms=terms)

    def _parse_terms(self, rhs: str) -> List[Term]:
        """Parse the right-hand side into a list of terms."""
        terms: List[Term] = []

        # Split by + while respecting parentheses
        term_strings = self._split_by_plus(rhs)

        for term_str in term_strings:
            term_str = term_str.strip()
            if not term_str:
                continue

            parsed = self._parse_single_term(term_str)
            if isinstance(parsed, list):
                terms.extend(parsed)
            else:
                terms.append(parsed)

        return terms

    def _split_by_plus(self, s: str) -> List[str]:
        """Split string by '+' while respecting parentheses."""
        parts = []
        current = []
        depth = 0

        for char in s:
            if char == "(":
                depth += 1
                current.append(char)
            elif char == ")":
                depth -= 1
                current.append(char)
            elif char == "+" and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(char)

        if current:
            parts.append("".join(current))

        return parts

    def _parse_single_term(self, term_str: str) -> Union[Term, List[Term]]:
        """Parse a single term string into a Term or list of Terms."""
        term_str = term_str.strip()

        # Check for poly()
        poly_match = self.POLY_PATTERN.match(term_str)
        if poly_match:
            var, degree, raw = poly_match.groups()
            return PolyTerm(variable=var, degree=int(degree), raw=(raw == "True"))

        # Check for I(x^n)
        transform_match = self.TRANSFORM_PATTERN.match(term_str)
        if transform_match:
            var, power = transform_match.groups()
            return TransformTerm(variable=var, power=int(power))

        # Check for * (full crossing) - must check before : since * has higher precedence
        if "*" in term_str:
            return self._expand_star(term_str)

        # Check for : (interaction only)
        if ":" in term_str:
            variables = [v.strip() for v in term_str.split(":")]
            return InteractionTerm(variables=variables)

        # Simple term
        return SimpleTerm(name=term_str)

    def _expand_star(self, term_str: str) -> List[Term]:
        """Expand 'x1 * x2' to [x1, x2, x1:x2].

        For 'x1 * x2 * x3', expands to:
        [x1, x2, x3, x1:x2, x1:x3, x2:x3, x1:x2:x3]
        """
        variables = [v.strip() for v in term_str.split("*")]
        terms: List[Term] = []

        # Add main effects
        for var in variables:
            terms.append(SimpleTerm(name=var))

        # Add all interaction combinations
        from itertools import combinations

        for r in range(2, len(variables) + 1):
            for combo in combinations(variables, r):
                terms.append(InteractionTerm(variables=list(combo)))

        return terms
