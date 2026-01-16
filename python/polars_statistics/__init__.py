"""polars-statistics: Statistical testing and regression for Polars DataFrames."""

from pathlib import Path

# Import Rust bindings
from polars_statistics._polars_statistics import (
    # Linear models
    OLS,
    Ridge,
    ElasticNet,
    WLS,
    RLS,
    BLS,
    # Robust regression
    Quantile,
    Isotonic,
    # GLM models
    Logistic,
    Poisson,
    NegativeBinomial,
    Tweedie,
    Probit,
    Cloglog,
    # Augmented Linear Model
    ALM,
    # Dynamic Linear Model
    LmDynamic,
    # Demand Classification
    Aid,
    AidResult,
    # Bootstrap
    StationaryBootstrap,
    CircularBlockBootstrap,
    # Parametric Tests
    TTestInd,
    TTestPaired,
    BrownForsythe,
    YuenTest,
    # Non-Parametric Tests
    MannWhitneyU,
    WilcoxonSignedRank,
    KruskalWallis,
    BrunnerMunzel,
    # Distributional Tests
    ShapiroWilk,
    DAgostino,
)

# Import expression API
from polars_statistics import exprs
from polars_statistics.exprs import (
    # Parametric tests
    ttest_ind,
    ttest_paired,
    brown_forsythe,
    yuen_test,
    # Non-parametric tests
    mann_whitney_u,
    wilcoxon_signed_rank,
    kruskal_wallis,
    brunner_munzel,
    # Distributional tests
    shapiro_wilk,
    dagostino,
    # Forecast tests
    diebold_mariano,
    permutation_t_test,
    clark_west,
    spa_test,
    model_confidence_set,
    mspe_adjusted,
    # Modern tests
    energy_distance,
    mmd_test,
    # TOST equivalence tests
    tost_t_test_one_sample,
    tost_t_test_two_sample,
    tost_t_test_paired,
    tost_correlation,
    tost_prop_one,
    tost_prop_two,
    tost_wilcoxon_paired,
    tost_wilcoxon_two_sample,
    tost_bootstrap,
    tost_yuen,
    # Correlation tests
    pearson,
    spearman,
    kendall,
    distance_cor,
    partial_cor,
    semi_partial_cor,
    icc,
    # Categorical tests
    binom_test,
    prop_test_one,
    prop_test_two,
    chisq_test,
    chisq_goodness_of_fit,
    g_test,
    fisher_exact,
    mcnemar_test,
    mcnemar_exact,
    cohen_kappa,
    cramers_v,
    phi_coefficient,
    contingency_coef,
    # Regression expressions
    ols,
    ridge,
    elastic_net,
    wls,
    rls,
    bls,
    nnls,
    quantile,
    isotonic,
    # Diagnostics expressions
    condition_number,
    check_binary_separation,
    check_count_sparsity,
    # GLM expressions
    logistic,
    poisson,
    negative_binomial,
    tweedie,
    probit,
    cloglog,
    alm,
    # Summary expressions
    ols_summary,
    ridge_summary,
    elastic_net_summary,
    wls_summary,
    rls_summary,
    bls_summary,
    logistic_summary,
    poisson_summary,
    negative_binomial_summary,
    tweedie_summary,
    probit_summary,
    cloglog_summary,
    alm_summary,
    # Prediction expressions
    ols_predict,
    ridge_predict,
    elastic_net_predict,
    wls_predict,
    rls_predict,
    bls_predict,
    nnls_predict,
    logistic_predict,
    poisson_predict,
    negative_binomial_predict,
    tweedie_predict,
    probit_predict,
    cloglog_predict,
    alm_predict,
    # Formula-based regression expressions
    ols_formula,
    ridge_formula,
    elastic_net_formula,
    wls_formula,
    rls_formula,
    bls_formula,
    nnls_formula,
    logistic_formula,
    poisson_formula,
    negative_binomial_formula,
    tweedie_formula,
    probit_formula,
    cloglog_formula,
    alm_formula,
    # Formula-based summary expressions
    ols_formula_summary,
    ridge_formula_summary,
    elastic_net_formula_summary,
    logistic_formula_summary,
    poisson_formula_summary,
    alm_formula_summary,
    # Formula-based prediction expressions
    ols_formula_predict,
    ridge_formula_predict,
    elastic_net_formula_predict,
    logistic_formula_predict,
    poisson_formula_predict,
    alm_formula_predict,
    # AID and Dynamic Linear Model expressions
    aid,
    aid_anomalies,
    lm_dynamic,
)

__version__ = "0.3.0"

# Library path for plugin registration
LIB = Path(__file__).parent

__all__ = [
    # Linear Models
    "OLS",
    "Ridge",
    "ElasticNet",
    "WLS",
    "RLS",
    "BLS",
    # Robust Regression Models
    "Quantile",
    "Isotonic",
    # GLM Models
    "Logistic",
    "Poisson",
    "NegativeBinomial",
    "Tweedie",
    "Probit",
    "Cloglog",
    # Augmented Linear Model
    "ALM",
    # Dynamic Linear Model
    "LmDynamic",
    # Demand Classification
    "Aid",
    "AidResult",
    # Bootstrap
    "StationaryBootstrap",
    "CircularBlockBootstrap",
    # Parametric Test Models
    "TTestInd",
    "TTestPaired",
    "BrownForsythe",
    "YuenTest",
    # Non-Parametric Test Models
    "MannWhitneyU",
    "WilcoxonSignedRank",
    "KruskalWallis",
    "BrunnerMunzel",
    # Distributional Test Models
    "ShapiroWilk",
    "DAgostino",
    # Expression API module
    "exprs",
    # Parametric test expressions
    "ttest_ind",
    "ttest_paired",
    "brown_forsythe",
    "yuen_test",
    # Non-parametric test expressions
    "mann_whitney_u",
    "wilcoxon_signed_rank",
    "kruskal_wallis",
    "brunner_munzel",
    # Distributional test expressions
    "shapiro_wilk",
    "dagostino",
    # Forecast test expressions
    "diebold_mariano",
    "permutation_t_test",
    "clark_west",
    "spa_test",
    "model_confidence_set",
    "mspe_adjusted",
    # Modern test expressions
    "energy_distance",
    "mmd_test",
    # TOST equivalence test expressions
    "tost_t_test_one_sample",
    "tost_t_test_two_sample",
    "tost_t_test_paired",
    "tost_correlation",
    "tost_prop_one",
    "tost_prop_two",
    "tost_wilcoxon_paired",
    "tost_wilcoxon_two_sample",
    "tost_bootstrap",
    "tost_yuen",
    # Correlation test expressions
    "pearson",
    "spearman",
    "kendall",
    "distance_cor",
    "partial_cor",
    "semi_partial_cor",
    "icc",
    # Categorical test expressions
    "binom_test",
    "prop_test_one",
    "prop_test_two",
    "chisq_test",
    "chisq_goodness_of_fit",
    "g_test",
    "fisher_exact",
    "mcnemar_test",
    "mcnemar_exact",
    "cohen_kappa",
    "cramers_v",
    "phi_coefficient",
    "contingency_coef",
    # Regression expressions
    "ols",
    "ridge",
    "elastic_net",
    "wls",
    "rls",
    "bls",
    "nnls",
    "quantile",
    "isotonic",
    # Diagnostics expressions
    "condition_number",
    "check_binary_separation",
    "check_count_sparsity",
    # GLM expressions
    "logistic",
    "poisson",
    "negative_binomial",
    "tweedie",
    "probit",
    "cloglog",
    "alm",
    # Summary expressions
    "ols_summary",
    "ridge_summary",
    "elastic_net_summary",
    "wls_summary",
    "rls_summary",
    "bls_summary",
    "logistic_summary",
    "poisson_summary",
    "negative_binomial_summary",
    "tweedie_summary",
    "probit_summary",
    "cloglog_summary",
    "alm_summary",
    # Prediction expressions
    "ols_predict",
    "ridge_predict",
    "elastic_net_predict",
    "wls_predict",
    "rls_predict",
    "bls_predict",
    "nnls_predict",
    "logistic_predict",
    "poisson_predict",
    "negative_binomial_predict",
    "tweedie_predict",
    "probit_predict",
    "cloglog_predict",
    "alm_predict",
    # Formula-based regression expressions
    "ols_formula",
    "ridge_formula",
    "elastic_net_formula",
    "wls_formula",
    "rls_formula",
    "bls_formula",
    "nnls_formula",
    "logistic_formula",
    "poisson_formula",
    "negative_binomial_formula",
    "tweedie_formula",
    "probit_formula",
    "cloglog_formula",
    "alm_formula",
    # Formula-based summary expressions
    "ols_formula_summary",
    "ridge_formula_summary",
    "elastic_net_formula_summary",
    "logistic_formula_summary",
    "poisson_formula_summary",
    "alm_formula_summary",
    # Formula-based prediction expressions
    "ols_formula_predict",
    "ridge_formula_predict",
    "elastic_net_formula_predict",
    "logistic_formula_predict",
    "poisson_formula_predict",
    "alm_formula_predict",
    # AID and Dynamic Linear Model expressions
    "aid",
    "aid_anomalies",
    "lm_dynamic",
    # Library path
    "LIB",
]
