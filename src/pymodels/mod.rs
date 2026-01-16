//! Python-facing model wrappers using PyO3.

// Regression models
mod py_aid;
mod py_alm;
mod py_bls;
mod py_bootstrap;
mod py_cloglog;
mod py_elastic_net;
mod py_isotonic;
mod py_lm_dynamic;
mod py_logistic;
mod py_negative_binomial;
mod py_ols;
mod py_poisson;
mod py_probit;
mod py_quantile;
mod py_ridge;
mod py_rls;
mod py_tweedie;
mod py_wls;

// Statistical test models
mod py_brown_forsythe;
mod py_brunner_munzel;
mod py_dagostino;
mod py_kruskal_wallis;
mod py_mann_whitney;
mod py_shapiro_wilk;
mod py_ttest_ind;
mod py_ttest_paired;
mod py_wilcoxon;
mod py_yuen_test;

// Regression model exports
pub use py_aid::{PyAid, PyAidResult};
pub use py_alm::PyALM;
pub use py_bls::PyBLS;
pub use py_bootstrap::{PyCircularBlockBootstrap, PyStationaryBootstrap};
pub use py_cloglog::PyCloglog;
pub use py_elastic_net::PyElasticNet;
pub use py_isotonic::PyIsotonic;
pub use py_lm_dynamic::PyLmDynamic;
pub use py_logistic::PyLogistic;
pub use py_negative_binomial::PyNegativeBinomial;
pub use py_ols::PyOLS;
pub use py_poisson::PyPoisson;
pub use py_probit::PyProbit;
pub use py_quantile::PyQuantile;
pub use py_ridge::PyRidge;
pub use py_rls::PyRLS;
pub use py_tweedie::PyTweedie;
pub use py_wls::PyWLS;

// Statistical test model exports
pub use py_brown_forsythe::PyBrownForsythe;
pub use py_brunner_munzel::PyBrunnerMunzel;
pub use py_dagostino::PyDAgostino;
pub use py_kruskal_wallis::PyKruskalWallis;
pub use py_mann_whitney::PyMannWhitneyU;
pub use py_shapiro_wilk::PyShapiroWilk;
pub use py_ttest_ind::PyTTestInd;
pub use py_ttest_paired::PyTTestPaired;
pub use py_wilcoxon::PyWilcoxonSignedRank;
pub use py_yuen_test::PyYuenTest;
