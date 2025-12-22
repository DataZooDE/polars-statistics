"""Tests for categorical statistical test expressions."""

import polars as pl
import pytest

import polars_statistics as ps


class TestBinomTest:
    """Tests for exact binomial test."""

    def test_binom_test_basic(self):
        """Test basic binomial test."""
        result = pl.select(
            ps.binom_test(successes=7, n=10, p0=0.5).alias("binom")
        )

        assert result.shape == (1, 1)
        binom = result["binom"][0]
        assert "estimate" in binom
        assert "statistic" in binom
        assert "p_value" in binom
        assert "ci_lower" in binom
        assert "ci_upper" in binom

    def test_binom_test_fair_coin(self):
        """Test binomial test for fair coin."""
        # 50 heads out of 100 flips
        result = pl.select(
            ps.binom_test(successes=50, n=100, p0=0.5).alias("binom")
        )
        binom = result["binom"][0]

        # Should not reject null hypothesis
        assert binom["p_value"] > 0.05
        assert binom["estimate"] == pytest.approx(0.5, abs=1e-10)

    def test_binom_test_biased_coin(self):
        """Test binomial test for biased coin."""
        # 80 heads out of 100 flips
        result = pl.select(
            ps.binom_test(successes=80, n=100, p0=0.5).alias("binom")
        )
        binom = result["binom"][0]

        # Should reject null hypothesis
        assert binom["p_value"] < 0.05

    def test_binom_test_one_sided(self):
        """Test one-sided binomial test."""
        result = pl.select(
            ps.binom_test(successes=7, n=10, p0=0.5, alternative="greater").alias("binom")
        )
        binom = result["binom"][0]

        assert "p_value" in binom


class TestPropTestOne:
    """Tests for one-sample proportion test."""

    def test_prop_test_one_basic(self):
        """Test basic one-sample proportion test."""
        result = pl.select(
            ps.prop_test_one(successes=50, n=100, p0=0.5).alias("prop")
        )

        assert result.shape == (1, 1)
        prop = result["prop"][0]
        assert "estimate" in prop
        assert "statistic" in prop
        assert "p_value" in prop

    def test_prop_test_one_significant(self):
        """Test significant proportion difference."""
        result = pl.select(
            ps.prop_test_one(successes=80, n=100, p0=0.5).alias("prop")
        )
        prop = result["prop"][0]

        assert prop["p_value"] < 0.05


class TestPropTestTwo:
    """Tests for two-sample proportion test."""

    def test_prop_test_two_basic(self):
        """Test basic two-sample proportion test."""
        result = pl.select(
            ps.prop_test_two(
                successes1=50, n1=100,
                successes2=60, n2=100
            ).alias("prop")
        )

        assert result.shape == (1, 1)
        prop = result["prop"][0]
        assert "estimate" in prop
        assert "statistic" in prop
        assert "p_value" in prop

    def test_prop_test_two_equal(self):
        """Test proportion test with equal proportions."""
        result = pl.select(
            ps.prop_test_two(
                successes1=50, n1=100,
                successes2=50, n2=100
            ).alias("prop")
        )
        prop = result["prop"][0]

        # Should not reject null
        assert prop["p_value"] > 0.05

    def test_prop_test_two_different(self):
        """Test proportion test with different proportions."""
        result = pl.select(
            ps.prop_test_two(
                successes1=30, n1=100,
                successes2=70, n2=100
            ).alias("prop")
        )
        prop = result["prop"][0]

        # Should reject null
        assert prop["p_value"] < 0.05

    def test_prop_test_two_with_correction(self):
        """Test proportion test with Yates' correction."""
        result = pl.select(
            ps.prop_test_two(
                successes1=50, n1=100,
                successes2=60, n2=100,
                correction=True
            ).alias("prop")
        )
        prop = result["prop"][0]

        assert "p_value" in prop


class TestChiSquareTest:
    """Tests for chi-square test."""

    def test_chisq_test_basic(self):
        """Test basic chi-square test."""
        # 2x2 table: [[10, 20], [30, 40]]
        df = pl.DataFrame({"counts": [10, 20, 30, 40]})

        result = df.select(
            ps.chisq_test("counts", n_rows=2, n_cols=2).alias("chisq")
        )

        assert result.shape == (1, 1)
        chisq = result["chisq"][0]
        assert "statistic" in chisq
        assert "p_value" in chisq
        assert "df" in chisq

    def test_chisq_test_independent(self):
        """Test chi-square test with independent data."""
        # Proportional table - should be independent
        df = pl.DataFrame({"counts": [10, 20, 10, 20]})

        result = df.select(
            ps.chisq_test("counts", n_rows=2, n_cols=2).alias("chisq")
        )
        chisq = result["chisq"][0]

        # Should not reject independence
        assert chisq["p_value"] > 0.05

    def test_chisq_test_associated(self):
        """Test chi-square test with associated data."""
        # Highly non-proportional table
        df = pl.DataFrame({"counts": [50, 5, 5, 50]})

        result = df.select(
            ps.chisq_test("counts", n_rows=2, n_cols=2).alias("chisq")
        )
        chisq = result["chisq"][0]

        # Should reject independence
        assert chisq["p_value"] < 0.05

    def test_chisq_test_with_correction(self):
        """Test chi-square test with Yates' correction."""
        df = pl.DataFrame({"counts": [10, 20, 30, 40]})

        result = df.select(
            ps.chisq_test("counts", n_rows=2, n_cols=2, correction=True).alias("chisq")
        )
        chisq = result["chisq"][0]

        assert "statistic" in chisq

    def test_chisq_test_3x3(self):
        """Test chi-square test with 3x3 table."""
        df = pl.DataFrame({
            "counts": [10, 20, 30, 15, 25, 35, 20, 30, 40]
        })

        result = df.select(
            ps.chisq_test("counts", n_rows=3, n_cols=3).alias("chisq")
        )
        chisq = result["chisq"][0]

        assert chisq["df"] == 4  # (3-1) * (3-1) = 4


class TestChiSquareGoodnessOfFit:
    """Tests for chi-square goodness of fit test."""

    def test_chisq_gof_basic(self):
        """Test basic chi-square goodness of fit."""
        # Observed counts for die rolls
        df = pl.DataFrame({"observed": [18, 22, 16, 19, 24, 21]})

        result = df.select(
            ps.chisq_goodness_of_fit("observed").alias("chisq")
        )

        assert result.shape == (1, 1)
        chisq = result["chisq"][0]
        assert "statistic" in chisq
        assert "p_value" in chisq
        assert "df" in chisq

    def test_chisq_gof_fair_die(self):
        """Test goodness of fit for fair die."""
        # Equal counts - should be fair
        df = pl.DataFrame({"observed": [20, 20, 20, 20, 20, 20]})

        result = df.select(
            ps.chisq_goodness_of_fit("observed").alias("chisq")
        )
        chisq = result["chisq"][0]

        # Should not reject null (die is fair)
        assert chisq["statistic"] == pytest.approx(0.0, abs=1e-10)
        assert chisq["p_value"] == pytest.approx(1.0, abs=1e-10)

    def test_chisq_gof_biased_die(self):
        """Test goodness of fit for biased die."""
        # Very unequal counts
        df = pl.DataFrame({"observed": [50, 10, 10, 10, 10, 10]})

        result = df.select(
            ps.chisq_goodness_of_fit("observed").alias("chisq")
        )
        chisq = result["chisq"][0]

        # Should reject null (die is biased)
        assert chisq["p_value"] < 0.05

    def test_chisq_gof_with_expected(self):
        """Test goodness of fit with custom expected values."""
        df = pl.DataFrame({
            "observed": [100, 200, 300],
            "expected": [200.0, 200.0, 200.0]
        })

        result = df.select(
            ps.chisq_goodness_of_fit("observed", "expected").alias("chisq")
        )
        chisq = result["chisq"][0]

        assert "p_value" in chisq


class TestGTest:
    """Tests for G-test (likelihood ratio test)."""

    def test_g_test_basic(self):
        """Test basic G-test."""
        df = pl.DataFrame({"counts": [10, 20, 30, 40]})

        result = df.select(
            ps.g_test("counts", n_rows=2, n_cols=2).alias("g")
        )

        assert result.shape == (1, 1)
        g = result["g"][0]
        assert "statistic" in g
        assert "p_value" in g
        assert "df" in g

    def test_g_test_independent(self):
        """Test G-test with independent data."""
        df = pl.DataFrame({"counts": [10, 20, 10, 20]})

        result = df.select(
            ps.g_test("counts", n_rows=2, n_cols=2).alias("g")
        )
        g = result["g"][0]

        assert g["p_value"] > 0.05


class TestFisherExact:
    """Tests for Fisher's exact test."""

    def test_fisher_exact_basic(self):
        """Test basic Fisher's exact test."""
        result = pl.select(
            ps.fisher_exact(a=10, b=2, c=3, d=15).alias("fisher")
        )

        assert result.shape == (1, 1)
        fisher = result["fisher"][0]
        assert "statistic" in fisher  # odds ratio
        assert "p_value" in fisher

    def test_fisher_exact_no_association(self):
        """Test Fisher's exact with no association."""
        # Proportional table
        result = pl.select(
            ps.fisher_exact(a=10, b=10, c=10, d=10).alias("fisher")
        )
        fisher = result["fisher"][0]

        # Odds ratio should be close to 1
        assert fisher["statistic"] == pytest.approx(1.0, abs=0.1)
        assert fisher["p_value"] > 0.05

    def test_fisher_exact_strong_association(self):
        """Test Fisher's exact with strong association."""
        result = pl.select(
            ps.fisher_exact(a=20, b=1, c=1, d=20).alias("fisher")
        )
        fisher = result["fisher"][0]

        # Strong association
        assert fisher["p_value"] < 0.05

    def test_fisher_exact_one_sided(self):
        """Test one-sided Fisher's exact test."""
        result = pl.select(
            ps.fisher_exact(a=10, b=2, c=3, d=15, alternative="greater").alias("fisher")
        )
        fisher = result["fisher"][0]

        assert "p_value" in fisher


class TestMcNemar:
    """Tests for McNemar's test."""

    def test_mcnemar_test_basic(self):
        """Test basic McNemar's test."""
        # Before/after: 45 yes->yes, 15 yes->no, 5 no->yes, 35 no->no
        result = pl.select(
            ps.mcnemar_test(a=45, b=15, c=5, d=35).alias("mcnemar")
        )

        assert result.shape == (1, 1)
        mcnemar = result["mcnemar"][0]
        assert "statistic" in mcnemar
        assert "p_value" in mcnemar

    def test_mcnemar_test_no_change(self):
        """Test McNemar's test with symmetric changes."""
        # Equal number of b and c (discordant pairs)
        result = pl.select(
            ps.mcnemar_test(a=40, b=10, c=10, d=40).alias("mcnemar")
        )
        mcnemar = result["mcnemar"][0]

        # Should not reject null (no systematic change)
        assert mcnemar["p_value"] > 0.05

    def test_mcnemar_test_significant_change(self):
        """Test McNemar's test with significant change."""
        # Many more b->c changes than c->b
        result = pl.select(
            ps.mcnemar_test(a=40, b=30, c=2, d=40).alias("mcnemar")
        )
        mcnemar = result["mcnemar"][0]

        # Should reject null (systematic change)
        assert mcnemar["p_value"] < 0.05

    def test_mcnemar_test_with_correction(self):
        """Test McNemar's test with continuity correction."""
        result = pl.select(
            ps.mcnemar_test(a=45, b=15, c=5, d=35, correction=True).alias("mcnemar")
        )
        mcnemar = result["mcnemar"][0]

        assert "statistic" in mcnemar

    def test_mcnemar_exact_basic(self):
        """Test McNemar's exact test."""
        result = pl.select(
            ps.mcnemar_exact(a=10, b=5, c=1, d=10).alias("mcnemar")
        )

        assert result.shape == (1, 1)
        mcnemar = result["mcnemar"][0]
        assert "statistic" in mcnemar
        assert "p_value" in mcnemar


class TestCohenKappa:
    """Tests for Cohen's Kappa."""

    def test_cohen_kappa_basic(self):
        """Test basic Cohen's Kappa."""
        # Confusion matrix: [[20, 5], [3, 22]]
        df = pl.DataFrame({"counts": [20, 5, 3, 22]})

        result = df.select(
            ps.cohen_kappa("counts", n_categories=2).alias("kappa")
        )

        assert result.shape == (1, 1)
        kappa = result["kappa"][0]
        assert "estimate" in kappa
        assert "p_value" in kappa

    def test_cohen_kappa_perfect_agreement(self):
        """Test Cohen's Kappa with perfect agreement."""
        # Perfect agreement: only diagonal cells
        df = pl.DataFrame({"counts": [50, 0, 0, 50]})

        result = df.select(
            ps.cohen_kappa("counts", n_categories=2).alias("kappa")
        )
        kappa = result["kappa"][0]

        # Perfect agreement = 1
        assert kappa["estimate"] == pytest.approx(1.0, abs=1e-10)

    def test_cohen_kappa_no_agreement(self):
        """Test Cohen's Kappa with chance agreement."""
        # Proportional to marginals - chance agreement
        df = pl.DataFrame({"counts": [25, 25, 25, 25]})

        result = df.select(
            ps.cohen_kappa("counts", n_categories=2).alias("kappa")
        )
        kappa = result["kappa"][0]

        # No agreement beyond chance = 0
        assert kappa["estimate"] == pytest.approx(0.0, abs=0.1)

    def test_cohen_kappa_weighted(self):
        """Test weighted Cohen's Kappa."""
        df = pl.DataFrame({"counts": [20, 5, 3, 22]})

        result = df.select(
            ps.cohen_kappa("counts", n_categories=2, weighted=True).alias("kappa")
        )
        kappa = result["kappa"][0]

        assert "estimate" in kappa

    def test_cohen_kappa_3x3(self):
        """Test Cohen's Kappa for 3 categories."""
        df = pl.DataFrame({
            "counts": [30, 5, 2, 3, 25, 4, 1, 3, 27]
        })

        result = df.select(
            ps.cohen_kappa("counts", n_categories=3).alias("kappa")
        )
        kappa = result["kappa"][0]

        assert "estimate" in kappa
        assert kappa["estimate"] > 0.5  # Good agreement


class TestCramersV:
    """Tests for Cramer's V."""

    def test_cramers_v_basic(self):
        """Test basic Cramer's V."""
        df = pl.DataFrame({"counts": [10, 20, 30, 40]})

        result = df.select(
            ps.cramers_v("counts", n_rows=2, n_cols=2).alias("v")
        )

        assert result.shape == (1, 1)
        v = result["v"][0]
        assert "estimate" in v
        assert "p_value" in v

    def test_cramers_v_no_association(self):
        """Test Cramer's V with no association."""
        df = pl.DataFrame({"counts": [10, 20, 10, 20]})

        result = df.select(
            ps.cramers_v("counts", n_rows=2, n_cols=2).alias("v")
        )
        v = result["v"][0]

        # No association = 0
        assert v["estimate"] == pytest.approx(0.0, abs=0.1)

    def test_cramers_v_perfect_association(self):
        """Test Cramer's V with perfect association."""
        df = pl.DataFrame({"counts": [50, 0, 0, 50]})

        result = df.select(
            ps.cramers_v("counts", n_rows=2, n_cols=2).alias("v")
        )
        v = result["v"][0]

        # Perfect association = 1
        assert v["estimate"] == pytest.approx(1.0, abs=0.1)


class TestPhiCoefficient:
    """Tests for Phi coefficient."""

    def test_phi_coefficient_basic(self):
        """Test basic Phi coefficient."""
        result = pl.select(
            ps.phi_coefficient(a=10, b=20, c=30, d=40).alias("phi")
        )

        assert result.shape == (1, 1)
        phi = result["phi"][0]
        assert "estimate" in phi
        assert "p_value" in phi

    def test_phi_coefficient_no_correlation(self):
        """Test Phi coefficient with no correlation."""
        result = pl.select(
            ps.phi_coefficient(a=10, b=10, c=10, d=10).alias("phi")
        )
        phi = result["phi"][0]

        assert phi["estimate"] == pytest.approx(0.0, abs=0.1)

    def test_phi_coefficient_positive(self):
        """Test Phi coefficient with positive correlation."""
        result = pl.select(
            ps.phi_coefficient(a=40, b=5, c=5, d=40).alias("phi")
        )
        phi = result["phi"][0]

        assert phi["estimate"] > 0.5


class TestContingencyCoef:
    """Tests for contingency coefficient."""

    def test_contingency_coef_basic(self):
        """Test basic contingency coefficient."""
        df = pl.DataFrame({"counts": [10, 20, 30, 40]})

        result = df.select(
            ps.contingency_coef("counts", n_rows=2, n_cols=2).alias("c")
        )

        assert result.shape == (1, 1)
        c = result["c"][0]
        assert "estimate" in c
        assert "p_value" in c

    def test_contingency_coef_range(self):
        """Test contingency coefficient is in valid range."""
        df = pl.DataFrame({"counts": [50, 5, 5, 50]})

        result = df.select(
            ps.contingency_coef("counts", n_rows=2, n_cols=2).alias("c")
        )
        c = result["c"][0]

        # Contingency coefficient is between 0 and 1
        assert 0 <= c["estimate"] <= 1


class TestCategoricalGroupBy:
    """Tests for categorical tests with group_by operations."""

    def test_chisq_group_by(self):
        """Test chi-square test with group_by."""
        df = pl.DataFrame({
            "group": ["A"] * 4 + ["B"] * 4,
            "counts": [10, 20, 30, 40, 15, 25, 35, 45],
        })

        result = df.group_by("group").agg(
            ps.chisq_test("counts", n_rows=2, n_cols=2).alias("chisq")
        ).sort("group")

        assert result.shape == (2, 2)
        assert result["group"].to_list() == ["A", "B"]

        for chisq in result["chisq"]:
            assert "statistic" in chisq
            assert "p_value" in chisq

    def test_cramers_v_group_by(self):
        """Test Cramer's V with group_by."""
        df = pl.DataFrame({
            "group": ["A"] * 4 + ["B"] * 4,
            "counts": [10, 20, 30, 40, 50, 5, 5, 50],
        })

        result = df.group_by("group").agg(
            ps.cramers_v("counts", n_rows=2, n_cols=2).alias("v")
        ).sort("group")

        assert result.shape == (2, 2)

        for v in result["v"]:
            assert "estimate" in v
