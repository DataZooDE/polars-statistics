"""Expand parsed formula terms to Polars expressions."""

from __future__ import annotations

import polars as pl

from polars_statistics.formula.terms import (
    InteractionTerm,
    PolyTerm,
    SimpleTerm,
    Term,
    TransformTerm,
)


def expand_terms_to_expressions(terms: list[Term]) -> tuple[list[pl.Expr], list[str]]:
    """Convert parsed terms to Polars expressions and term names.

    Parameters
    ----------
    terms : List[Term]
        List of parsed terms from FormulaParser

    Returns
    -------
    Tuple[List[pl.Expr], List[str]]
        (expressions, term_names) - parallel lists of Polars expressions
        and their corresponding string names for output
    """
    expressions: list[pl.Expr] = []
    names: list[str] = []

    for term in terms:
        term_exprs, term_names = _expand_single_term(term)
        expressions.extend(term_exprs)
        names.extend(term_names)

    return expressions, names


def _expand_single_term(term: Term) -> tuple[list[pl.Expr], list[str]]:
    """Expand a single term to expressions and names."""
    if isinstance(term, SimpleTerm):
        return _expand_simple(term)
    elif isinstance(term, InteractionTerm):
        return _expand_interaction(term)
    elif isinstance(term, PolyTerm):
        return _expand_poly(term)
    elif isinstance(term, TransformTerm):
        return _expand_transform(term)
    else:
        raise TypeError(f"Unknown term type: {type(term)}")


def _expand_simple(term: SimpleTerm) -> tuple[list[pl.Expr], list[str]]:
    """Expand a simple term like 'x1'."""
    expr = pl.col(term.name).cast(pl.Float64)
    return [expr], [term.name]


def _expand_interaction(term: InteractionTerm) -> tuple[list[pl.Expr], list[str]]:
    """Expand an interaction term like 'x1:x2'.

    Creates the product of all variables.
    """
    # Build product expression
    expr = pl.col(term.variables[0]).cast(pl.Float64)
    for var in term.variables[1:]:
        expr = expr * pl.col(var).cast(pl.Float64)

    name = ":".join(term.variables)
    return [expr], [name]


def _expand_poly(term: PolyTerm) -> tuple[list[pl.Expr], list[str]]:
    """Expand a polynomial term like 'poly(x, 2)'.

    For raw=False (default): Centers each polynomial term for partial orthogonalization.
    For raw=True: Uses raw polynomial terms (x, x^2, x^3, ...).

    Centering is computed using .mean() which works per-group in over()/group_by().
    """
    expressions: list[pl.Expr] = []
    names: list[str] = []

    base = pl.col(term.variable).cast(pl.Float64)

    for d in range(1, term.degree + 1):
        if d == 1:
            if term.raw:
                expr = base
            else:
                # Center the first term
                expr = base - base.mean()
            name = term.variable
        else:
            powered = base.pow(d)
            if term.raw:
                expr = powered
            else:
                # Center higher degree terms
                expr = powered - powered.mean()
            name = f"I({term.variable}^{d})"

        expressions.append(expr)
        names.append(name)

    return expressions, names


def _expand_transform(term: TransformTerm) -> tuple[list[pl.Expr], list[str]]:
    """Expand an explicit transform like 'I(x^2)'."""
    expr = pl.col(term.variable).cast(pl.Float64).pow(term.power)
    name = f"I({term.variable}^{term.power})"
    return [expr], [name]
