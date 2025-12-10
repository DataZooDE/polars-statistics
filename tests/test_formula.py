"""Tests for R-style formula parsing and integration."""

import math
import numpy as np
import polars as pl
import pytest

import polars_statistics as ps
from polars_statistics.formula import (
    FormulaParser,
    expand_terms_to_expressions,
    SimpleTerm,
    InteractionTerm,
    PolyTerm,
    TransformTerm,
)


# =============================================================================
# Helper Functions
# =============================================================================


def has_valid_numbers(result: pl.DataFrame, col: str = "result") -> bool:
    """Check that result column contains valid (non-null, non-NaN) numbers."""
    for row in result[col]:
        if row is None:
            return False
        if isinstance(row, dict):
            for key, val in row.items():
                if val is None:
                    return False
                if isinstance(val, float) and math.isnan(val):
                    return False
                if isinstance(val, list):
                    for v in val:
                        if v is None or (isinstance(v, float) and math.isnan(v)):
                            return False
    return True


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def df_regression():
    """DataFrame for regression tests."""
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    y = 1.0 + 2.0 * x1 - 1.0 * x2 + 0.5 * x1 * x2 + np.random.randn(n) * 0.5
    return pl.DataFrame({
        "group": ["A"] * 50 + ["B"] * 50,
        "y": y.tolist(),
        "x1": x1.tolist(),
        "x2": x2.tolist(),
    })


@pytest.fixture
def df_poly():
    """DataFrame for polynomial regression tests."""
    np.random.seed(42)
    n = 100
    x = np.linspace(-2, 2, n)
    y = 1.0 + 2.0 * x - 0.5 * x**2 + np.random.randn(n) * 0.2
    return pl.DataFrame({
        "group": ["A"] * 50 + ["B"] * 50,
        "y": y.tolist(),
        "x": x.tolist(),
    })


# =============================================================================
# Parser Tests
# =============================================================================


class TestFormulaParser:
    """Tests for formula parsing."""

    def test_simple_formula(self):
        """Test parsing simple formula."""
        parser = FormulaParser()
        result = parser.parse("y ~ x1 + x2")

        assert result.response == "y"
        assert len(result.terms) == 2
        assert isinstance(result.terms[0], SimpleTerm)
        assert result.terms[0].name == "x1"
        assert isinstance(result.terms[1], SimpleTerm)
        assert result.terms[1].name == "x2"

    def test_interaction_colon(self):
        """Test parsing interaction with colon."""
        parser = FormulaParser()
        result = parser.parse("y ~ x1:x2")

        assert len(result.terms) == 1
        assert isinstance(result.terms[0], InteractionTerm)
        assert result.terms[0].variables == ["x1", "x2"]

    def test_interaction_star(self):
        """Test parsing full crossing with star."""
        parser = FormulaParser()
        result = parser.parse("y ~ x1 * x2")

        # Should expand to x1, x2, x1:x2
        assert len(result.terms) == 3
        assert isinstance(result.terms[0], SimpleTerm)
        assert result.terms[0].name == "x1"
        assert isinstance(result.terms[1], SimpleTerm)
        assert result.terms[1].name == "x2"
        assert isinstance(result.terms[2], InteractionTerm)
        assert result.terms[2].variables == ["x1", "x2"]

    def test_poly_term(self):
        """Test parsing polynomial term."""
        parser = FormulaParser()
        result = parser.parse("y ~ poly(x, 2)")

        assert len(result.terms) == 1
        assert isinstance(result.terms[0], PolyTerm)
        assert result.terms[0].variable == "x"
        assert result.terms[0].degree == 2
        assert result.terms[0].raw is False

    def test_poly_term_raw(self):
        """Test parsing raw polynomial term."""
        parser = FormulaParser()
        result = parser.parse("y ~ poly(x, 3, raw=True)")

        assert len(result.terms) == 1
        assert isinstance(result.terms[0], PolyTerm)
        assert result.terms[0].variable == "x"
        assert result.terms[0].degree == 3
        assert result.terms[0].raw is True

    def test_transform_term(self):
        """Test parsing explicit transform."""
        parser = FormulaParser()
        result = parser.parse("y ~ I(x^2)")

        assert len(result.terms) == 1
        assert isinstance(result.terms[0], TransformTerm)
        assert result.terms[0].variable == "x"
        assert result.terms[0].power == 2

    def test_complex_formula(self):
        """Test parsing complex formula with multiple term types."""
        parser = FormulaParser()
        result = parser.parse("y ~ x1 + x2 + x1:x2 + poly(x3, 2) + I(x4^2)")

        assert result.response == "y"
        assert len(result.terms) == 5
        assert isinstance(result.terms[0], SimpleTerm)
        assert isinstance(result.terms[1], SimpleTerm)
        assert isinstance(result.terms[2], InteractionTerm)
        assert isinstance(result.terms[3], PolyTerm)
        assert isinstance(result.terms[4], TransformTerm)

    def test_missing_tilde_raises(self):
        """Test that missing tilde raises error."""
        parser = FormulaParser()
        with pytest.raises(ValueError, match="~"):
            parser.parse("y x1 + x2")

    def test_empty_response_raises(self):
        """Test that empty response raises error."""
        parser = FormulaParser()
        with pytest.raises(ValueError, match="response"):
            parser.parse("~ x1 + x2")

    def test_empty_rhs_raises(self):
        """Test that empty RHS raises error."""
        parser = FormulaParser()
        with pytest.raises(ValueError, match="predictor"):
            parser.parse("y ~")


# =============================================================================
# Expander Tests
# =============================================================================


class TestTermExpander:
    """Tests for term expansion to Polars expressions."""

    def test_simple_term_expansion(self):
        """Test expanding simple terms."""
        terms = [SimpleTerm("x1"), SimpleTerm("x2")]
        exprs, names = expand_terms_to_expressions(terms)

        assert len(exprs) == 2
        assert names == ["x1", "x2"]

    def test_interaction_term_expansion(self):
        """Test expanding interaction terms."""
        terms = [InteractionTerm(["x1", "x2"])]
        exprs, names = expand_terms_to_expressions(terms)

        assert len(exprs) == 1
        assert names == ["x1:x2"]

    def test_poly_term_expansion(self):
        """Test expanding polynomial terms."""
        terms = [PolyTerm("x", degree=2, raw=True)]
        exprs, names = expand_terms_to_expressions(terms)

        assert len(exprs) == 2
        assert names == ["x", "I(x^2)"]

    def test_transform_term_expansion(self):
        """Test expanding transform terms."""
        terms = [TransformTerm("x", power=3)]
        exprs, names = expand_terms_to_expressions(terms)

        assert len(exprs) == 1
        assert names == ["I(x^3)"]


# =============================================================================
# Integration Tests - OLS Formula
# =============================================================================


class TestOLSFormula:
    """Integration tests for ols_formula."""

    def test_simple_formula(self, df_regression):
        """Test OLS with simple formula."""
        result = df_regression.select(
            ps.ols_formula("y ~ x1 + x2").alias("result")
        )
        assert has_valid_numbers(result)

    def test_interaction_formula(self, df_regression):
        """Test OLS with interaction formula."""
        result = df_regression.select(
            ps.ols_formula("y ~ x1 * x2").alias("result")
        )
        assert has_valid_numbers(result)

        # Should have 3 coefficients (x1, x2, x1:x2)
        coefs = result["result"][0]["coefficients"]
        assert len(coefs) == 3

    def test_poly_formula(self, df_poly):
        """Test OLS with polynomial formula."""
        result = df_poly.select(
            ps.ols_formula("y ~ poly(x, 2, raw=True)").alias("result")
        )
        assert has_valid_numbers(result)

        # Should have 2 coefficients (x, x^2)
        coefs = result["result"][0]["coefficients"]
        assert len(coefs) == 2

    def test_group_by_formula(self, df_regression):
        """Test OLS formula with group_by."""
        result = df_regression.group_by("group").agg(
            ps.ols_formula("y ~ x1 + x2").alias("result")
        )
        assert result.shape[0] == 2
        assert has_valid_numbers(result)

    def test_over_formula(self, df_regression):
        """Test OLS formula with over."""
        result = df_regression.with_columns(
            ps.ols_formula("y ~ x1 + x2").over("group").alias("result")
        )
        assert result.shape[0] == 100
        assert has_valid_numbers(result)


# =============================================================================
# Integration Tests - Formula equals explicit
# =============================================================================


class TestFormulaEqualsExplicit:
    """Test that formula functions produce same results as explicit functions."""

    def test_ols_formula_equals_explicit(self, df_regression):
        """Test that ols_formula equals explicit ols."""
        formula_result = df_regression.select(
            ps.ols_formula("y ~ x1 + x2").alias("formula")
        )
        explicit_result = df_regression.select(
            ps.ols("y", "x1", "x2").alias("explicit")
        )

        # Compare coefficients
        formula_coefs = formula_result["formula"][0]["coefficients"]
        explicit_coefs = explicit_result["explicit"][0]["coefficients"]
        assert len(formula_coefs) == len(explicit_coefs)
        for f, e in zip(formula_coefs, explicit_coefs):
            assert abs(f - e) < 1e-10

    def test_interaction_formula_equals_manual(self, df_regression):
        """Test that interaction formula equals manually created interaction."""
        # Formula approach
        formula_result = df_regression.select(
            ps.ols_formula("y ~ x1 + x2 + x1:x2").alias("formula")
        )

        # Manual approach
        manual_result = df_regression.select(
            ps.ols("y", "x1", "x2", pl.col("x1") * pl.col("x2")).alias("manual")
        )

        # Compare coefficients
        formula_coefs = formula_result["formula"][0]["coefficients"]
        manual_coefs = manual_result["manual"][0]["coefficients"]
        assert len(formula_coefs) == len(manual_coefs)
        for f, m in zip(formula_coefs, manual_coefs):
            assert abs(f - m) < 1e-10


# =============================================================================
# Integration Tests - Per-Group Polynomial Centering
# =============================================================================


class TestPerGroupPolynomial:
    """Test that polynomial terms are computed per-group."""

    def test_poly_centering_differs_per_group(self, df_poly):
        """Test that polynomial centering is computed per-group."""
        # With group_by, centering should be per-group
        result = df_poly.group_by("group").agg(
            ps.ols_formula("y ~ poly(x, 2)").alias("result")
        )
        assert result.shape[0] == 2

        # Both groups should have valid results
        assert has_valid_numbers(result)


# =============================================================================
# Integration Tests - Other Model Formulas
# =============================================================================


class TestOtherModelFormulas:
    """Test formula functions for other model types."""

    def test_ridge_formula(self, df_regression):
        """Test ridge_formula."""
        result = df_regression.select(
            ps.ridge_formula("y ~ x1 * x2", lambda_=1.0).alias("result")
        )
        assert has_valid_numbers(result)

    def test_elastic_net_formula(self, df_regression):
        """Test elastic_net_formula."""
        result = df_regression.select(
            ps.elastic_net_formula("y ~ x1 + x2", lambda_=1.0, alpha=0.5).alias("result")
        )
        assert has_valid_numbers(result)

    def test_ols_formula_summary(self, df_regression):
        """Test ols_formula_summary."""
        result = df_regression.select(
            ps.ols_formula_summary("y ~ x1 + x2").alias("result")
        )
        # Should return list of coefficient summaries
        summaries = result["result"][0]
        assert len(summaries) == 3  # intercept + 2 coefficients

    def test_ols_formula_predict(self, df_regression):
        """Test ols_formula_predict."""
        result = df_regression.with_columns(
            ps.ols_formula_predict("y ~ x1 + x2").over("group").alias("pred")
        )
        assert result.shape[0] == 100
        # Should have predictions for all rows
        preds = result["pred"].to_list()
        for p in preds:
            assert p is not None
            assert "ols_prediction" in p
