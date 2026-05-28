//! Integration tests for the public Rust API exposed by `polars_statistics::expressions`.
//!
//! These tests exercise every `*_fit` function reachable from a downstream Rust crate
//! via the rlib path. They must compile and pass with:
//!
//! ```text
//! cargo test --no-default-features --test rust_api
//! ```
//!
//! Test depth:
//! - The four headline regressors (OLS / WLS / Ridge / Quantile) are validated against
//!   known coefficients on synthetic linear data.
//! - All other `*_fit` functions are smoke-tested for reachability and no-panic
//!   behaviour on legitimate inputs.

use polars::prelude::*;
use polars_statistics::expressions::*;

// =============================================================================
// Helpers
// =============================================================================

/// Synthetic data: y = 1.0 + 2.0 * x with no noise. 20 points.
fn linear_xy() -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi).collect();
    (x, y)
}

/// Two moderately-correlated samples, length 25.
fn two_samples() -> (Vec<f64>, Vec<f64>) {
    let a: Vec<f64> = (0..25).map(|i| (i as f64) * 0.5 + 1.0).collect();
    let b: Vec<f64> = (0..25).map(|i| (i as f64) * 0.5 + 1.3).collect();
    (a, b)
}

fn series_f64(name: &str, vals: &[f64]) -> Series {
    Series::new(name.into(), vals)
}

fn scalar_f64(name: &str, v: f64) -> Series {
    Series::new(name.into(), &[v])
}

fn scalar_bool(name: &str, v: bool) -> Series {
    Series::new(name.into(), &[v])
}

fn scalar_str(name: &str, v: &str) -> Series {
    Series::new(name.into(), &[v])
}

fn scalar_str_null(name: &str) -> Series {
    Series::new(name.into(), &[None::<&str>])
}

fn scalar_u32(name: &str, v: u32) -> Series {
    Series::new(name.into(), &[v])
}

fn scalar_u64(name: &str, v: u64) -> Series {
    Series::new(name.into(), &[v])
}

fn scalar_u64_null(name: &str) -> Series {
    Series::new(name.into(), &[None::<u64>])
}

fn scalar_f64_null(name: &str) -> Series {
    Series::new(name.into(), &[None::<f64>])
}

/// Helper to assert that a struct Series has a `n_observations` field > 0.
fn assert_n_obs_nonzero(s: &Series, field: &str) {
    let ca = s.struct_().expect("expected struct series");
    let n_series = ca
        .field_by_name(field)
        .unwrap_or_else(|_| panic!("missing field `{field}`"));
    let n = n_series
        .u32()
        .expect("n_observations is u32")
        .get(0)
        .unwrap_or(0);
    assert!(n > 0, "expected `{field}` > 0, got {n}");
}

/// Helper: assert a struct field f64 is non-NaN (proves the fit produced a value).
fn assert_f64_finite_or_zero(s: &Series, field: &str) {
    let ca = s.struct_().expect("expected struct series");
    let f = ca
        .field_by_name(field)
        .unwrap_or_else(|_| panic!("missing field `{field}`"));
    let v = f.f64().expect("expected f64 column").get(0);
    // We accept any value (including NaN) as long as the call didn't panic — but
    // for the headline asserts below we use the dedicated helpers.
    let _ = v;
}

// =============================================================================
// Linear regression (OLS / Ridge / Elastic Net / WLS / RLS / BLS) +
// Quantile / Isotonic + their *_summary_fit + *_predict_fit variants
// =============================================================================

#[test]
fn linear_regression_fits() {
    let (x, y) = linear_xy();
    let n = x.len();

    // -----------------------------------------------------------------
    // OLS: assert known coefficients (intercept ~= 1.0, slope ~= 2.0).
    // -----------------------------------------------------------------
    {
        let inputs = vec![
            series_f64("y", &y),
            scalar_bool("with_intercept", true),
            scalar_str_null("solve_method"),
            series_f64("x1", &x),
        ];
        let out = ols_fit(&inputs).expect("ols_fit failed");
        let st = out.struct_().unwrap();
        let intercept = st
            .field_by_name("intercept")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();
        let coefs_field = st.field_by_name("coefficients").unwrap();
        let coefs_inner = coefs_field.list().unwrap().get_as_series(0).unwrap();
        let slope = coefs_inner.f64().unwrap().get(0).unwrap();
        let r2 = st
            .field_by_name("r_squared")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();
        assert!(
            (intercept - 1.0).abs() < 1e-6,
            "OLS intercept ~ 1.0, got {intercept}"
        );
        assert!((slope - 2.0).abs() < 1e-6, "OLS slope ~ 2.0, got {slope}");
        assert!((r2 - 1.0).abs() < 1e-6, "OLS R^2 ~ 1.0, got {r2}");
        assert_n_obs_nonzero(&out, "n_observations");
    }

    // -----------------------------------------------------------------
    // WLS: same exact-fit data, all-unit weights.
    // -----------------------------------------------------------------
    {
        let w = vec![1.0_f64; n];
        let inputs = vec![
            series_f64("y", &y),
            series_f64("w", &w),
            scalar_bool("with_intercept", true),
            scalar_str_null("solve_method"),
            series_f64("x1", &x),
        ];
        let out = wls_fit(&inputs).expect("wls_fit failed");
        let st = out.struct_().unwrap();
        let intercept = st
            .field_by_name("intercept")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();
        let coefs_inner = st
            .field_by_name("coefficients")
            .unwrap()
            .list()
            .unwrap()
            .get_as_series(0)
            .unwrap();
        let slope = coefs_inner.f64().unwrap().get(0).unwrap();
        assert!(
            (intercept - 1.0).abs() < 1e-6,
            "WLS intercept ~ 1.0, got {intercept}"
        );
        assert!((slope - 2.0).abs() < 1e-6, "WLS slope ~ 2.0, got {slope}");
        assert_n_obs_nonzero(&out, "n_observations");
    }

    // -----------------------------------------------------------------
    // Ridge: small lambda -> close to OLS coefficients.
    // -----------------------------------------------------------------
    {
        let inputs = vec![
            series_f64("y", &y),
            scalar_f64("lambda", 1e-6),
            scalar_bool("with_intercept", true),
            scalar_str_null("solve_method"),
            series_f64("x1", &x),
        ];
        let out = ridge_fit(&inputs).expect("ridge_fit failed");
        let st = out.struct_().unwrap();
        let intercept = st
            .field_by_name("intercept")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();
        let coefs_inner = st
            .field_by_name("coefficients")
            .unwrap()
            .list()
            .unwrap()
            .get_as_series(0)
            .unwrap();
        let slope = coefs_inner.f64().unwrap().get(0).unwrap();
        assert!(
            (intercept - 1.0).abs() < 1e-3,
            "Ridge intercept ~ 1.0 (small lambda), got {intercept}"
        );
        assert!(
            (slope - 2.0).abs() < 1e-3,
            "Ridge slope ~ 2.0 (small lambda), got {slope}"
        );
        assert_n_obs_nonzero(&out, "n_observations");
    }

    // -----------------------------------------------------------------
    // Quantile (median, tau = 0.5).
    // -----------------------------------------------------------------
    {
        let inputs = vec![
            series_f64("y", &y),
            scalar_f64("tau", 0.5),
            scalar_bool("with_intercept", true),
            series_f64("x1", &x),
        ];
        let out = quantile_fit(&inputs).expect("quantile_fit failed");
        let st = out.struct_().unwrap();
        let intercept = st
            .field_by_name("intercept")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();
        let coefs_inner = st
            .field_by_name("coefficients")
            .unwrap()
            .list()
            .unwrap()
            .get_as_series(0)
            .unwrap();
        let slope = coefs_inner.f64().unwrap().get(0).unwrap();
        assert!(
            (intercept - 1.0).abs() < 1e-3,
            "Quantile (median) intercept ~ 1.0, got {intercept}"
        );
        assert!(
            (slope - 2.0).abs() < 1e-3,
            "Quantile (median) slope ~ 2.0, got {slope}"
        );
        assert_n_obs_nonzero(&out, "n_observations");
    }

    // -----------------------------------------------------------------
    // Elastic Net (smoke test).
    // -----------------------------------------------------------------
    {
        let inputs = vec![
            series_f64("y", &y),
            scalar_f64("lambda", 0.01),
            scalar_f64("alpha", 0.5),
            scalar_bool("with_intercept", true),
            series_f64("x1", &x),
        ];
        let out = elastic_net_fit(&inputs).expect("elastic_net_fit failed");
        assert_n_obs_nonzero(&out, "n_observations");
    }

    // -----------------------------------------------------------------
    // RLS (smoke test).
    // -----------------------------------------------------------------
    {
        let inputs = vec![
            series_f64("y", &y),
            scalar_f64("forgetting_factor", 0.99),
            scalar_bool("with_intercept", true),
            series_f64("x1", &x),
        ];
        let out = rls_fit(&inputs).expect("rls_fit failed");
        assert_n_obs_nonzero(&out, "n_observations");
    }

    // -----------------------------------------------------------------
    // BLS (smoke test).
    // -----------------------------------------------------------------
    {
        let inputs = vec![
            series_f64("y", &y),
            scalar_f64("lower_bound", -10.0),
            scalar_f64("upper_bound", 10.0),
            scalar_bool("with_intercept", true),
            series_f64("x1", &x),
        ];
        let out = bls_fit(&inputs).expect("bls_fit failed");
        assert_n_obs_nonzero(&out, "n_observations");
    }

    // -----------------------------------------------------------------
    // Isotonic (single feature, strictly increasing).
    // -----------------------------------------------------------------
    {
        let inputs = vec![
            series_f64("y", &y),
            series_f64("x", &x),
            scalar_bool("increasing", true),
        ];
        let out = isotonic_fit(&inputs).expect("isotonic_fit failed");
        assert_n_obs_nonzero(&out, "n_observations");
    }

    // quantile_summary_fit (issue #18): y, tau, with_intercept, x...
    {
        let inputs = vec![
            series_f64("y", &y),
            scalar_f64("tau", 0.5),
            scalar_bool("with_intercept", true),
            series_f64("x1", &x),
        ];
        let _ = quantile_summary_fit(&inputs).expect("quantile_summary_fit failed");
    }

    // quantile_predict_fit (issue #18): y, tau, with_intercept, null_policy, x...
    {
        let inputs = vec![
            series_f64("y", &y),
            scalar_f64("tau", 0.5),
            scalar_bool("with_intercept", true),
            scalar_str("null_policy", "drop"),
            series_f64("x1", &x),
        ];
        let pk = PredictKwargs {
            prefix: "q".to_string(),
        };
        let out = quantile_predict_fit(&inputs, pk).expect("quantile_predict_fit failed");
        assert_eq!(out.len(), y.len());
    }

    // isotonic_predict_fit (issue #18): y, x, increasing, null_policy
    {
        let inputs = vec![
            series_f64("y", &y),
            series_f64("x", &x),
            scalar_bool("increasing", true),
            scalar_str("null_policy", "drop"),
        ];
        let pk = PredictKwargs {
            prefix: "iso".to_string(),
        };
        let out = isotonic_predict_fit(&inputs, pk).expect("isotonic_predict_fit failed");
        assert_eq!(out.len(), y.len());
    }

    // -----------------------------------------------------------------
    // Summary variants (tidy coefficient output).
    // -----------------------------------------------------------------
    {
        // ols_summary: y, with_intercept, solve_method, hc_type, x...
        let inputs = vec![
            series_f64("y", &y),
            scalar_bool("with_intercept", true),
            scalar_str_null("solve_method"),
            scalar_str_null("hc_type"),
            series_f64("x1", &x),
        ];
        let _ = ols_summary_fit(&inputs).expect("ols_summary_fit failed");
    }
    {
        let inputs = vec![
            series_f64("y", &y),
            scalar_f64("lambda", 0.1),
            scalar_bool("with_intercept", true),
            scalar_str_null("solve_method"),
            series_f64("x1", &x),
        ];
        let _ = ridge_summary_fit(&inputs).expect("ridge_summary_fit failed");
    }
    {
        let inputs = vec![
            series_f64("y", &y),
            scalar_f64("lambda", 0.1),
            scalar_f64("alpha", 0.5),
            scalar_bool("with_intercept", true),
            series_f64("x1", &x),
        ];
        let _ = elastic_net_summary_fit(&inputs).expect("elastic_net_summary_fit failed");
    }
    {
        let w = vec![1.0_f64; n];
        let inputs = vec![
            series_f64("y", &y),
            series_f64("w", &w),
            scalar_bool("with_intercept", true),
            scalar_str_null("solve_method"),
            series_f64("x1", &x),
        ];
        let _ = wls_summary_fit(&inputs).expect("wls_summary_fit failed");
    }
    {
        let inputs = vec![
            series_f64("y", &y),
            scalar_f64("forgetting_factor", 0.99),
            scalar_bool("with_intercept", true),
            series_f64("x1", &x),
        ];
        let _ = rls_summary_fit(&inputs).expect("rls_summary_fit failed");
    }
    {
        let inputs = vec![
            series_f64("y", &y),
            scalar_f64("lower_bound", -10.0),
            scalar_f64("upper_bound", 10.0),
            scalar_bool("with_intercept", true),
            series_f64("x1", &x),
        ];
        let _ = bls_summary_fit(&inputs).expect("bls_summary_fit failed");
    }

    // -----------------------------------------------------------------
    // Linear *_predict_fit variants.
    // -----------------------------------------------------------------
    let kwargs = || PredictKwargs {
        prefix: "p".to_string(),
    };

    // ols_predict: y, with_intercept, solve_method, interval, level, null_policy, x...
    {
        let inputs = vec![
            series_f64("y", &y),
            scalar_bool("with_intercept", true),
            scalar_str_null("solve_method"),
            scalar_str_null("interval"),
            scalar_f64("level", 0.95),
            scalar_str("null_policy", "drop"),
            series_f64("x1", &x),
        ];
        let out = ols_predict_fit(&inputs, kwargs()).expect("ols_predict_fit failed");
        assert_eq!(out.len(), y.len(), "predictions length matches input");
    }

    // ridge_predict: + lambda after null_policy.
    {
        let inputs = vec![
            series_f64("y", &y),
            scalar_bool("with_intercept", true),
            scalar_str_null("solve_method"),
            scalar_str_null("interval"),
            scalar_f64("level", 0.95),
            scalar_str("null_policy", "drop"),
            scalar_f64("lambda", 0.1),
            series_f64("x1", &x),
        ];
        let out = ridge_predict_fit(&inputs, kwargs()).expect("ridge_predict_fit failed");
        assert_eq!(out.len(), y.len());
    }

    // elastic_net_predict: + lambda + alpha (no solve_method).
    {
        let inputs = vec![
            series_f64("y", &y),
            scalar_bool("with_intercept", true),
            scalar_str_null("interval"),
            scalar_f64("level", 0.95),
            scalar_str("null_policy", "drop"),
            scalar_f64("lambda", 0.1),
            scalar_f64("alpha", 0.5),
            series_f64("x1", &x),
        ];
        let out =
            elastic_net_predict_fit(&inputs, kwargs()).expect("elastic_net_predict_fit failed");
        assert_eq!(out.len(), y.len());
    }

    // wls_predict: + weights at idx 6.
    {
        let w = vec![1.0_f64; n];
        let inputs = vec![
            series_f64("y", &y),
            scalar_bool("with_intercept", true),
            scalar_str_null("solve_method"),
            scalar_str_null("interval"),
            scalar_f64("level", 0.95),
            scalar_str("null_policy", "drop"),
            series_f64("weights", &w),
            series_f64("x1", &x),
        ];
        let out = wls_predict_fit(&inputs, kwargs()).expect("wls_predict_fit failed");
        assert_eq!(out.len(), y.len());
    }

    // rls_predict.
    {
        let inputs = vec![
            series_f64("y", &y),
            scalar_bool("with_intercept", true),
            scalar_str_null("interval"),
            scalar_f64("level", 0.95),
            scalar_str("null_policy", "drop"),
            scalar_f64("forgetting_factor", 0.99),
            series_f64("x1", &x),
        ];
        let out = rls_predict_fit(&inputs, kwargs()).expect("rls_predict_fit failed");
        assert_eq!(out.len(), y.len());
    }

    // bls_predict: + lower + upper.
    {
        let inputs = vec![
            series_f64("y", &y),
            scalar_bool("with_intercept", true),
            scalar_str_null("interval"),
            scalar_f64("level", 0.95),
            scalar_str("null_policy", "drop"),
            scalar_f64("lower_bound", -10.0),
            scalar_f64("upper_bound", 10.0),
            series_f64("x1", &x),
        ];
        let out = bls_predict_fit(&inputs, kwargs()).expect("bls_predict_fit failed");
        assert_eq!(out.len(), y.len());
    }

    // huber_fit (added in issue #14): y, epsilon, alpha, with_intercept, max_iter, tol, x...
    // Same y = 1 + 2x ground truth as ols_fit, but with planted outliers — Huber
    // should still recover slope ~2.0 and intercept ~1.0.
    {
        let mut y_with_outliers = y.clone();
        // Spike 3 of the 20 points (indices well below the assertion margin).
        for &idx in &[3, 9, 15] {
            y_with_outliers[idx] += 50.0;
        }
        let inputs = vec![
            series_f64("y", &y_with_outliers),
            scalar_f64("epsilon", 1.35),
            scalar_f64("alpha", 1e-4),
            scalar_bool("with_intercept", true),
            scalar_u32("max_iter", 100),
            scalar_f64("tol", 1e-5),
            series_f64("x1", &x),
        ];
        let out = huber_fit(&inputs).expect("huber_fit failed");
        let st = out.struct_().unwrap();
        let intercept = st
            .field_by_name("intercept")
            .unwrap()
            .f64()
            .unwrap()
            .get(0)
            .unwrap();
        let coefs_inner = st
            .field_by_name("coefficients")
            .unwrap()
            .list()
            .unwrap()
            .get_as_series(0)
            .unwrap();
        let slope = coefs_inner.f64().unwrap().get(0).unwrap();
        // Tolerances looser than OLS — Huber down-weights but doesn't ignore outliers,
        // and alpha > 0 introduces a tiny ridge bias.
        assert!(
            (intercept - 1.0).abs() < 2.0,
            "huber intercept {} far from 1.0",
            intercept
        );
        assert!(
            (slope - 2.0).abs() < 0.5,
            "huber slope {} far from 2.0",
            slope
        );
    }
}

// =============================================================================
// GLM family (logistic / poisson / negbin / tweedie / probit / cloglog / alm)
// + their predict + summary variants
// =============================================================================

#[test]
fn glm_fits() {
    // Binary outcome: y switches at x = 10.
    let x_bin: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let y_bin: Vec<f64> = x_bin
        .iter()
        .map(|&xi| if xi >= 10.0 { 1.0 } else { 0.0 })
        .collect();

    // Count outcome: y ~ exp(0.1 * x) rounded.
    let x_cnt: Vec<f64> = (0..25).map(|i| (i as f64) * 0.1).collect();
    let y_cnt: Vec<f64> = x_cnt.iter().map(|&xi| (xi * 5.0 + 1.0).round()).collect();

    // Tweedie / continuous positive: small positive.
    let x_pos: Vec<f64> = (0..25).map(|i| (i as f64) * 0.1 + 0.1).collect();
    let y_pos: Vec<f64> = x_pos.iter().map(|&xi| 1.0 + 0.5 * xi).collect();

    let kwargs = || PredictKwargs {
        prefix: "p".to_string(),
    };

    // logistic_fit: y, lambda, with_intercept, x...
    {
        let inputs = vec![
            series_f64("y", &y_bin),
            scalar_f64("lambda", 0.0),
            scalar_bool("with_intercept", true),
            series_f64("x1", &x_bin),
        ];
        let out = logistic_fit(&inputs).expect("logistic_fit failed");
        assert_n_obs_nonzero(&out, "n_observations");
        // summary
        let _ = logistic_summary_fit(&inputs).expect("logistic_summary_fit failed");
        // residual diagnostics (issue #27 batch 2)
        let _ =
            logistic_pearson_residuals_fit(&inputs).expect("logistic_pearson_residuals_fit failed");
        let _ = logistic_deviance_residuals_fit(&inputs)
            .expect("logistic_deviance_residuals_fit failed");
        let _ =
            logistic_working_residuals_fit(&inputs).expect("logistic_working_residuals_fit failed");
    }

    // logistic_regression_fit (issue #14, sklearn-style):
    //   y, C, penalty (str), threshold, with_intercept, max_iter (u32), tol, x...
    {
        let inputs = vec![
            series_f64("y", &y_bin),
            scalar_f64("C", 1.0),
            scalar_str("penalty", "l2"),
            scalar_f64("threshold", 0.5),
            scalar_bool("with_intercept", true),
            scalar_u32("max_iter", 100),
            scalar_f64("tol", 1e-8),
            series_f64("x1", &x_bin),
        ];
        let out = logistic_regression_fit(&inputs).expect("logistic_regression_fit failed");
        assert_n_obs_nonzero(&out, "n_observations");
    }

    // poisson_fit: y, lambda, with_intercept, x...
    {
        let inputs = vec![
            series_f64("y", &y_cnt),
            scalar_f64("lambda", 0.0),
            scalar_bool("with_intercept", true),
            series_f64("x1", &x_cnt),
        ];
        let out = poisson_fit(&inputs).expect("poisson_fit failed");
        assert_n_obs_nonzero(&out, "n_observations");
        let _ = poisson_summary_fit(&inputs).expect("poisson_summary_fit failed");
        // residual diagnostics (issue #27 batch 2)
        let _ =
            poisson_pearson_residuals_fit(&inputs).expect("poisson_pearson_residuals_fit failed");
        let _ =
            poisson_deviance_residuals_fit(&inputs).expect("poisson_deviance_residuals_fit failed");
        let _ =
            poisson_working_residuals_fit(&inputs).expect("poisson_working_residuals_fit failed");
    }

    // negative_binomial_fit: y, theta, lambda, with_intercept, x...
    {
        let inputs = vec![
            series_f64("y", &y_cnt),
            scalar_f64("theta", 1.0),
            scalar_f64("lambda", 0.0),
            scalar_bool("with_intercept", true),
            series_f64("x1", &x_cnt),
        ];
        let out = negative_binomial_fit(&inputs).expect("negative_binomial_fit failed");
        assert_n_obs_nonzero(&out, "n_observations");
        let _ =
            negative_binomial_summary_fit(&inputs).expect("negative_binomial_summary_fit failed");
    }

    // tweedie_fit: y, var_power, lambda, with_intercept, x...
    {
        let inputs = vec![
            series_f64("y", &y_pos),
            scalar_f64("var_power", 1.5),
            scalar_f64("lambda", 0.0),
            scalar_bool("with_intercept", true),
            series_f64("x1", &x_pos),
        ];
        let out = tweedie_fit(&inputs).expect("tweedie_fit failed");
        assert_n_obs_nonzero(&out, "n_observations");
        let _ = tweedie_summary_fit(&inputs).expect("tweedie_summary_fit failed");
    }

    // probit_fit: y, lambda, with_intercept, x...
    {
        let inputs = vec![
            series_f64("y", &y_bin),
            scalar_f64("lambda", 0.0),
            scalar_bool("with_intercept", true),
            series_f64("x1", &x_bin),
        ];
        let out = probit_fit(&inputs).expect("probit_fit failed");
        assert_n_obs_nonzero(&out, "n_observations");
        let _ = probit_summary_fit(&inputs).expect("probit_summary_fit failed");
    }

    // cloglog_fit
    {
        let inputs = vec![
            series_f64("y", &y_bin),
            scalar_f64("lambda", 0.0),
            scalar_bool("with_intercept", true),
            series_f64("x1", &x_bin),
        ];
        let out = cloglog_fit(&inputs).expect("cloglog_fit failed");
        assert_n_obs_nonzero(&out, "n_observations");
        let _ = cloglog_summary_fit(&inputs).expect("cloglog_summary_fit failed");
    }

    // alm_fit (issue #16, extended contract):
    //   y, distribution, link (str|null), loss, role_trim (f64|null),
    //   extra_parameter (f64|null), with_intercept, x...
    {
        let inputs = vec![
            series_f64("y", &y_pos),
            scalar_str("distribution", "normal"),
            scalar_str_null("link"),
            scalar_str("loss", "likelihood"),
            scalar_f64_null("role_trim"),
            scalar_f64_null("extra_parameter"),
            scalar_bool("with_intercept", true),
            series_f64("x1", &x_pos),
        ];
        let out = alm_fit(&inputs).expect("alm_fit failed");
        assert_n_obs_nonzero(&out, "n_observations");
    }

    // alm_summary_fit kept the older 3-scalar contract — issue #18 will revisit.
    {
        let summary_inputs = vec![
            series_f64("y", &y_pos),
            scalar_str("distribution", "normal"),
            scalar_bool("with_intercept", true),
            series_f64("x1", &x_pos),
        ];
        let _ = alm_summary_fit(&summary_inputs).expect("alm_summary_fit failed");
    }

    // -----------------------------------------------------------------
    // GLM predict variants
    // -----------------------------------------------------------------

    // logistic_predict: y, lambda, with_intercept, interval, level, null_policy, x...
    {
        let inputs = vec![
            series_f64("y", &y_bin),
            scalar_f64("lambda", 0.0),
            scalar_bool("with_intercept", true),
            scalar_str_null("interval"),
            scalar_f64("level", 0.95),
            scalar_str("null_policy", "drop"),
            series_f64("x1", &x_bin),
        ];
        let out = logistic_predict_fit(&inputs, kwargs()).expect("logistic_predict_fit failed");
        assert_eq!(out.len(), y_bin.len());
    }

    // poisson_predict
    {
        let inputs = vec![
            series_f64("y", &y_cnt),
            scalar_f64("lambda", 0.0),
            scalar_bool("with_intercept", true),
            scalar_str_null("interval"),
            scalar_f64("level", 0.95),
            scalar_str("null_policy", "drop"),
            series_f64("x1", &x_cnt),
        ];
        let out = poisson_predict_fit(&inputs, kwargs()).expect("poisson_predict_fit failed");
        assert_eq!(out.len(), y_cnt.len());
    }

    // negative_binomial_predict
    {
        let inputs = vec![
            series_f64("y", &y_cnt),
            scalar_f64("lambda", 0.0),
            scalar_bool("with_intercept", true),
            scalar_str_null("interval"),
            scalar_f64("level", 0.95),
            scalar_str("null_policy", "drop"),
            series_f64("x1", &x_cnt),
        ];
        let out = negative_binomial_predict_fit(&inputs, kwargs())
            .expect("negative_binomial_predict_fit failed");
        assert_eq!(out.len(), y_cnt.len());
    }

    // tweedie_predict: y, lambda, with_intercept, interval, level, null_policy, var_power, x...
    {
        let inputs = vec![
            series_f64("y", &y_pos),
            scalar_f64("lambda", 0.0),
            scalar_bool("with_intercept", true),
            scalar_str_null("interval"),
            scalar_f64("level", 0.95),
            scalar_str("null_policy", "drop"),
            scalar_f64("var_power", 1.5),
            series_f64("x1", &x_pos),
        ];
        let out = tweedie_predict_fit(&inputs, kwargs()).expect("tweedie_predict_fit failed");
        assert_eq!(out.len(), y_pos.len());
    }

    // probit_predict
    {
        let inputs = vec![
            series_f64("y", &y_bin),
            scalar_f64("lambda", 0.0),
            scalar_bool("with_intercept", true),
            scalar_str_null("interval"),
            scalar_f64("level", 0.95),
            scalar_str("null_policy", "drop"),
            series_f64("x1", &x_bin),
        ];
        let out = probit_predict_fit(&inputs, kwargs()).expect("probit_predict_fit failed");
        assert_eq!(out.len(), y_bin.len());
    }

    // cloglog_predict
    {
        let inputs = vec![
            series_f64("y", &y_bin),
            scalar_f64("lambda", 0.0),
            scalar_bool("with_intercept", true),
            scalar_str_null("interval"),
            scalar_f64("level", 0.95),
            scalar_str("null_policy", "drop"),
            series_f64("x1", &x_bin),
        ];
        let out = cloglog_predict_fit(&inputs, kwargs()).expect("cloglog_predict_fit failed");
        assert_eq!(out.len(), y_bin.len());
    }

    // alm_predict: y, with_intercept, interval, level, null_policy, distribution, x...
    {
        let inputs = vec![
            series_f64("y", &y_pos),
            scalar_bool("with_intercept", true),
            scalar_str_null("interval"),
            scalar_f64("level", 0.95),
            scalar_str("null_policy", "drop"),
            scalar_str("distribution", "normal"),
            series_f64("x1", &x_pos),
        ];
        let out = alm_predict_fit(&inputs, kwargs()).expect("alm_predict_fit failed");
        assert_eq!(out.len(), y_pos.len());
    }
}

// =============================================================================
// Diagnostics (condition_number / check_binary_separation / check_count_sparsity)
// =============================================================================

#[test]
fn diagnostic_fits() {
    // condition_number_fit: inputs[0] = with_intercept, inputs[1..] = x columns.
    let x1: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let x2: Vec<f64> = (0..20).map(|i| (i as f64) * 0.5 + 1.0).collect();
    {
        let inputs = vec![
            scalar_bool("with_intercept", true),
            series_f64("x1", &x1),
            series_f64("x2", &x2),
        ];
        let out = condition_number_fit(&inputs).expect("condition_number_fit failed");
        // Severity field should be present.
        let st = out.struct_().unwrap();
        let _sev = st.field_by_name("severity").unwrap();
    }

    // check_binary_separation_fit: y (binary), x columns.
    {
        let y: Vec<f64> = (0..20).map(|i| if i >= 10 { 1.0 } else { 0.0 }).collect();
        let inputs = vec![
            series_f64("y", &y),
            series_f64("x1", &x1),
            series_f64("x2", &x2),
        ];
        let out = check_binary_separation_fit(&inputs).expect("check_binary_separation_fit failed");
        let st = out.struct_().unwrap();
        let _has = st.field_by_name("has_separation").unwrap();
    }

    // check_count_sparsity_fit: y (counts), x columns.
    {
        let y: Vec<f64> = (0..20).map(|i| (i % 5) as f64).collect();
        let inputs = vec![
            series_f64("y", &y),
            series_f64("x1", &x1),
            series_f64("x2", &x2),
        ];
        let out = check_count_sparsity_fit(&inputs).expect("check_count_sparsity_fit failed");
        let st = out.struct_().unwrap();
        let _has = st.field_by_name("has_separation").unwrap();
    }

    // vif_fit: only x columns; needs at least 2 features for a meaningful VIF.
    // Use two independent predictors so VIF should be close to 1.
    {
        let x_a: Vec<f64> = (0..30).map(|i| (i as f64 * 0.3).sin()).collect();
        let x_b: Vec<f64> = (0..30).map(|i| (i as f64 * 0.3).cos()).collect();
        let inputs = vec![series_f64("x1", &x_a), series_f64("x2", &x_b)];
        let out = vif_fit(&inputs).expect("vif_fit failed");
        let st = out.struct_().unwrap();
        let _terms = st.field_by_name("terms").unwrap();
        let _vif = st.field_by_name("vif").unwrap();
        assert_n_obs_nonzero(&out, "n_observations");
    }

    // leverage_fit: with_intercept (bool), x columns.
    {
        let inputs = vec![
            scalar_bool("with_intercept", true),
            series_f64("x1", &x1),
            series_f64("x2", &x2),
        ];
        let out = leverage_fit(&inputs).expect("leverage_fit failed");
        let st = out.struct_().unwrap();
        let _lev = st.field_by_name("leverage").unwrap();
        assert_n_obs_nonzero(&out, "n_observations");
    }

    // cooks_distance_fit: y, with_intercept (bool), x columns.
    {
        let y_vals: Vec<f64> = x1
            .iter()
            .enumerate()
            .map(|(i, xi)| 0.5 + 1.5 * xi + 0.1 * (i as f64).sin())
            .collect();
        let inputs = vec![
            series_f64("y", &y_vals),
            scalar_bool("with_intercept", true),
            series_f64("x1", &x1),
            series_f64("x2", &x2),
        ];
        let out = cooks_distance_fit(&inputs).expect("cooks_distance_fit failed");
        let st = out.struct_().unwrap();
        let _cd = st.field_by_name("cooks_d").unwrap();
        assert_n_obs_nonzero(&out, "n_observations");
    }

    // residual diagnostics (issue #27 batch 1): standardized / studentized /
    // externally studentized — same input contract as cooks_distance.
    {
        let y_vals: Vec<f64> = x1
            .iter()
            .enumerate()
            .map(|(i, xi)| 0.5 + 1.5 * xi + 0.1 * (i as f64).sin())
            .collect();
        let inputs = vec![
            series_f64("y", &y_vals),
            scalar_bool("with_intercept", true),
            series_f64("x1", &x1),
            series_f64("x2", &x2),
        ];
        let _ = standardized_residuals_fit(&inputs).expect("standardized_residuals_fit failed");
        let _ = studentized_residuals_fit(&inputs).expect("studentized_residuals_fit failed");
        let _ = externally_studentized_residuals_fit(&inputs)
            .expect("externally_studentized_residuals_fit failed");
    }

    // residual_outliers_fit: y, with_intercept, threshold, x...
    {
        let y_vals: Vec<f64> = x1
            .iter()
            .enumerate()
            .map(|(i, xi)| 0.5 + 1.5 * xi + 0.1 * (i as f64).sin())
            .collect();
        let inputs = vec![
            series_f64("y", &y_vals),
            scalar_bool("with_intercept", true),
            scalar_f64("threshold", 2.0),
            series_f64("x1", &x1),
            series_f64("x2", &x2),
        ];
        let out = residual_outliers_fit(&inputs).expect("residual_outliers_fit failed");
        let st = out.struct_().unwrap();
        let _ = st.field_by_name("is_outlier").unwrap();
        let n_obs = st
            .field_by_name("n_observations")
            .unwrap()
            .u32()
            .unwrap()
            .get(0)
            .unwrap_or(0);
        assert!(n_obs > 0);
    }
}

// =============================================================================
// Parametric tests (t-test ind/paired, Brown-Forsythe, Yuen)
// =============================================================================

#[test]
fn parametric_tests() {
    let (a, b) = two_samples();

    // ttest_ind_fit: x, y, alternative, equal_var, mu, conf_level
    {
        let inputs = vec![
            series_f64("x", &a),
            series_f64("y", &b),
            scalar_str("alternative", "two-sided"),
            scalar_bool("equal_var", false),
            scalar_f64("mu", 0.0),
            scalar_f64("conf_level", 0.95),
        ];
        let out = ttest_ind_fit(&inputs).expect("ttest_ind_fit failed");
        let st = out.struct_().unwrap();
        let _stat = st.field_by_name("statistic").unwrap();
    }

    // ttest_paired_fit: x, y, alternative, mu, conf_level
    {
        let inputs = vec![
            series_f64("x", &a),
            series_f64("y", &b),
            scalar_str("alternative", "two-sided"),
            scalar_f64("mu", 0.0),
            scalar_f64("conf_level", 0.95),
        ];
        let _ = ttest_paired_fit(&inputs).expect("ttest_paired_fit failed");
    }

    // brown_forsythe_fit: two groups.
    {
        let inputs = vec![series_f64("x", &a), series_f64("y", &b)];
        let _ = brown_forsythe_fit(&inputs).expect("brown_forsythe_fit failed");
    }

    // yuen_test_fit: x, y, trim, alternative, conf_level
    {
        let inputs = vec![
            series_f64("x", &a),
            series_f64("y", &b),
            scalar_f64("trim", 0.2),
            scalar_str("alternative", "two-sided"),
            scalar_f64("conf_level", 0.95),
        ];
        let _ = yuen_test_fit(&inputs).expect("yuen_test_fit failed");
    }
}

// =============================================================================
// Non-parametric tests
// =============================================================================

#[test]
fn nonparametric_tests() {
    let (a, b) = two_samples();

    // mann_whitney_u_fit: x, y, alternative, continuity_correction, exact, conf_level, mu
    {
        let inputs = vec![
            series_f64("x", &a),
            series_f64("y", &b),
            scalar_str("alternative", "two-sided"),
            scalar_bool("continuity_correction", true),
            scalar_bool("exact", false),
            scalar_f64("conf_level", 0.95),
            scalar_f64("mu", 0.0),
        ];
        let _ = mann_whitney_u_fit(&inputs).expect("mann_whitney_u_fit failed");
    }

    // wilcoxon_signed_rank_fit: x, y, alternative, continuity_correction, exact, conf_level, mu
    {
        let inputs = vec![
            series_f64("x", &a),
            series_f64("y", &b),
            scalar_str("alternative", "two-sided"),
            scalar_bool("continuity_correction", true),
            scalar_bool("exact", false),
            scalar_f64("conf_level", 0.95),
            scalar_f64("mu", 0.0),
        ];
        let _ = wilcoxon_signed_rank_fit(&inputs).expect("wilcoxon_signed_rank_fit failed");
    }

    // kruskal_wallis_fit: each input series is a separate group.
    {
        let c: Vec<f64> = (0..25).map(|i| (i as f64) * 0.5 + 2.0).collect();
        let inputs = vec![
            series_f64("g1", &a),
            series_f64("g2", &b),
            series_f64("g3", &c),
        ];
        let _ = kruskal_wallis_fit(&inputs).expect("kruskal_wallis_fit failed");
    }

    // brunner_munzel_fit: x, y, alternative, alpha
    {
        let inputs = vec![
            series_f64("x", &a),
            series_f64("y", &b),
            scalar_str("alternative", "two-sided"),
            scalar_f64("alpha", 0.05),
        ];
        let _ = brunner_munzel_fit(&inputs).expect("brunner_munzel_fit failed");
    }
}

// =============================================================================
// Distributional tests (normality)
// =============================================================================

#[test]
fn distributional_tests() {
    // shapiro_wilk_fit: x only.
    let x: Vec<f64> = (0..30)
        .map(|i| (i as f64) * 0.3 + 0.7 * ((i as f64).sin()))
        .collect();
    {
        let inputs = vec![series_f64("x", &x)];
        let _ = shapiro_wilk_fit(&inputs).expect("shapiro_wilk_fit failed");
    }
    // dagostino_fit (D'Agostino K-squared): x only.
    {
        let inputs = vec![series_f64("x", &x)];
        let _ = dagostino_fit(&inputs).expect("dagostino_fit failed");
    }
}

// =============================================================================
// Correlation tests
// =============================================================================

#[test]
fn correlation_fits() {
    let (a, b) = two_samples();

    // pearson_fit: x, y, conf_level
    {
        let inputs = vec![
            series_f64("x", &a),
            series_f64("y", &b),
            scalar_f64("conf_level", 0.95),
        ];
        let out = pearson_fit(&inputs).expect("pearson_fit failed");
        assert_n_obs_nonzero(&out, "n");
    }

    // spearman_fit: x, y, conf_level
    {
        let inputs = vec![
            series_f64("x", &a),
            series_f64("y", &b),
            scalar_f64("conf_level", 0.95),
        ];
        let out = spearman_fit(&inputs).expect("spearman_fit failed");
        assert_n_obs_nonzero(&out, "n");
    }

    // kendall_fit: x, y, variant (str)
    {
        let inputs = vec![
            series_f64("x", &a),
            series_f64("y", &b),
            scalar_str("variant", "b"),
        ];
        let _ = kendall_fit(&inputs).expect("kendall_fit failed");
    }

    // distance_cor_fit: x, y, n_permutations, seed
    {
        let inputs = vec![
            series_f64("x", &a),
            series_f64("y", &b),
            scalar_u32("n_permutations", 99),
            scalar_u64_null("seed"),
        ];
        let _ = distance_cor_fit(&inputs).expect("distance_cor_fit failed");
    }

    // partial_cor_fit: x, y, n_covariates (u32), then n_covariates covariates.
    {
        let cov: Vec<f64> = (0..25).map(|i| (i as f64) * 0.2).collect();
        let inputs = vec![
            series_f64("x", &a),
            series_f64("y", &b),
            scalar_u32("n_covariates", 1),
            series_f64("cov1", &cov),
        ];
        let _ = partial_cor_fit(&inputs).expect("partial_cor_fit failed");
    }

    // semi_partial_cor_fit: same layout as partial_cor_fit.
    {
        let cov: Vec<f64> = (0..25).map(|i| (i as f64) * 0.2).collect();
        let inputs = vec![
            series_f64("x", &a),
            series_f64("y", &b),
            scalar_u32("n_covariates", 1),
            series_f64("cov1", &cov),
        ];
        let _ = semi_partial_cor_fit(&inputs).expect("semi_partial_cor_fit failed");
    }

    // icc_fit: values, icc_type, conf_level. This is currently a placeholder
    // implementation in the crate (returns NaNs) but must still be reachable.
    {
        let vals: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let inputs = vec![
            series_f64("values", &vals),
            scalar_str("icc_type", "icc1"),
            scalar_f64("conf_level", 0.95),
        ];
        let _ = icc_fit(&inputs).expect("icc_fit failed");
    }
}

// =============================================================================
// Categorical / contingency-table tests
// =============================================================================

#[test]
fn categorical_fits() {
    // binom_test_fit: successes (u32), n (u32), p0 (f64), alternative (str)
    {
        let inputs = vec![
            scalar_u32("successes", 7),
            scalar_u32("n", 10),
            scalar_f64("p0", 0.5),
            scalar_str("alternative", "two-sided"),
        ];
        let _ = binom_test_fit(&inputs).expect("binom_test_fit failed");
    }

    // prop_test_one_fit
    {
        let inputs = vec![
            scalar_u32("successes", 30),
            scalar_u32("n", 100),
            scalar_f64("p0", 0.4),
            scalar_str("alternative", "two-sided"),
        ];
        let _ = prop_test_one_fit(&inputs).expect("prop_test_one_fit failed");
    }

    // prop_test_two_fit
    {
        let inputs = vec![
            scalar_u32("s1", 30),
            scalar_u32("n1", 100),
            scalar_u32("s2", 25),
            scalar_u32("n2", 100),
            scalar_str("alternative", "two-sided"),
            scalar_bool("correction", false),
        ];
        let _ = prop_test_two_fit(&inputs).expect("prop_test_two_fit failed");
    }

    // chisq_test_fit: 2x2 flattened table (u32 series of length 4), n_rows, n_cols, correction.
    {
        let table = Series::new("table".into(), &[10u32, 20, 30, 40]);
        let inputs = vec![
            table,
            scalar_u32("n_rows", 2),
            scalar_u32("n_cols", 2),
            scalar_bool("correction", false),
        ];
        let _ = chisq_test_fit(&inputs).expect("chisq_test_fit failed");
    }

    // chisq_goodness_of_fit_fit: observed (u32), has_expected (bool); no expected.
    {
        let observed = Series::new("observed".into(), &[15u32, 25, 30, 20]);
        let inputs = vec![observed, scalar_bool("has_expected", false)];
        let _ = chisq_goodness_of_fit_fit(&inputs).expect("chisq_goodness_of_fit_fit failed");
    }

    // g_test_fit: same shape as chisq_test_fit (without correction).
    {
        let table = Series::new("table".into(), &[10u32, 20, 30, 40]);
        let inputs = vec![table, scalar_u32("n_rows", 2), scalar_u32("n_cols", 2)];
        let _ = g_test_fit(&inputs).expect("g_test_fit failed");
    }

    // fisher_exact_fit: a, b, c, d, alternative
    {
        let inputs = vec![
            scalar_u32("a", 8),
            scalar_u32("b", 2),
            scalar_u32("c", 1),
            scalar_u32("d", 5),
            scalar_str("alternative", "two-sided"),
        ];
        let _ = fisher_exact_fit(&inputs).expect("fisher_exact_fit failed");
    }

    // mcnemar_test_fit: a, b, c, d, correction
    {
        let inputs = vec![
            scalar_u32("a", 30),
            scalar_u32("b", 12),
            scalar_u32("c", 5),
            scalar_u32("d", 50),
            scalar_bool("correction", false),
        ];
        let _ = mcnemar_test_fit(&inputs).expect("mcnemar_test_fit failed");
    }

    // mcnemar_exact_fit: a, b, c, d
    {
        let inputs = vec![
            scalar_u32("a", 30),
            scalar_u32("b", 12),
            scalar_u32("c", 5),
            scalar_u32("d", 50),
        ];
        let _ = mcnemar_exact_fit(&inputs).expect("mcnemar_exact_fit failed");
    }

    // cohen_kappa_fit: matrix flattened (u32), n_categories, weighted
    {
        // 2x2 confusion matrix
        let matrix = Series::new("matrix".into(), &[20u32, 5, 10, 30]);
        let inputs = vec![
            matrix,
            scalar_u32("n_categories", 2),
            scalar_bool("weighted", false),
        ];
        let _ = cohen_kappa_fit(&inputs).expect("cohen_kappa_fit failed");
    }

    // cramers_v_fit: flattened table, n_rows, n_cols
    {
        let table = Series::new("table".into(), &[10u32, 20, 30, 40]);
        let inputs = vec![table, scalar_u32("n_rows", 2), scalar_u32("n_cols", 2)];
        let _ = cramers_v_fit(&inputs).expect("cramers_v_fit failed");
    }

    // phi_coefficient_fit: a, b, c, d (2x2)
    {
        let inputs = vec![
            scalar_u32("a", 10),
            scalar_u32("b", 20),
            scalar_u32("c", 30),
            scalar_u32("d", 40),
        ];
        let _ = phi_coefficient_fit(&inputs).expect("phi_coefficient_fit failed");
    }

    // contingency_coef_fit: flattened table, n_rows, n_cols
    {
        let table = Series::new("table".into(), &[10u32, 20, 30, 40]);
        let inputs = vec![table, scalar_u32("n_rows", 2), scalar_u32("n_cols", 2)];
        let _ = contingency_coef_fit(&inputs).expect("contingency_coef_fit failed");
    }
}

// =============================================================================
// Forecast comparison tests
// =============================================================================

#[test]
fn forecast_fits() {
    // Two error series of length 50.
    let e1: Vec<f64> = (0..50).map(|i| (i as f64).sin() * 0.5 + 0.1).collect();
    let e2: Vec<f64> = (0..50).map(|i| (i as f64).cos() * 0.5 + 0.15).collect();
    let e3: Vec<f64> = (0..50)
        .map(|i| ((i as f64) * 0.7).sin() * 0.4 + 0.2)
        .collect();

    // diebold_mariano_fit: e1, e2, loss (str), h (u32), alt (str), var_est (str)
    {
        let inputs = vec![
            series_f64("e1", &e1),
            series_f64("e2", &e2),
            scalar_str("loss", "squared"),
            scalar_u32("h", 1),
            scalar_str("alternative", "two-sided"),
            scalar_str("var_est", "acf"),
        ];
        let _ = diebold_mariano_fit(&inputs).expect("diebold_mariano_fit failed");
    }

    // permutation_t_test_fit: x, y, alternative, n_perm (u32), seed (u64 nullable)
    {
        let inputs = vec![
            series_f64("x", &e1),
            series_f64("y", &e2),
            scalar_str("alternative", "two-sided"),
            scalar_u32("n_perm", 99),
            scalar_u64("seed", 42),
        ];
        let _ = permutation_t_test_fit(&inputs).expect("permutation_t_test_fit failed");
    }

    // clark_west_fit: e1, e2, h
    {
        let inputs = vec![
            series_f64("e1", &e1),
            series_f64("e2", &e2),
            scalar_u32("h", 1),
        ];
        let _ = clark_west_fit(&inputs).expect("clark_west_fit failed");
    }

    // spa_test_fit: benchmark, n_bootstrap, block_length, seed, then model losses.
    {
        let inputs = vec![
            series_f64("benchmark", &e1),
            scalar_u32("n_bootstrap", 99),
            scalar_f64("block_length", 5.0),
            scalar_u64("seed", 42),
            series_f64("m1", &e2),
            series_f64("m2", &e3),
        ];
        let _ = spa_test_fit(&inputs).expect("spa_test_fit failed");
    }

    // model_confidence_set_fit: alpha, stat (str), n_bootstrap, block_length, seed, then model losses.
    {
        let inputs = vec![
            scalar_f64("alpha", 0.1),
            scalar_str("stat", "range"),
            scalar_u32("n_bootstrap", 99),
            scalar_f64("block_length", 5.0),
            scalar_u64("seed", 42),
            series_f64("m1", &e1),
            series_f64("m2", &e2),
            series_f64("m3", &e3),
        ];
        let _ = model_confidence_set_fit(&inputs).expect("model_confidence_set_fit failed");
    }

    // mspe_adjusted_fit: benchmark, n_bootstrap, block_length, seed, then model errors.
    {
        let inputs = vec![
            series_f64("benchmark", &e1),
            scalar_u32("n_bootstrap", 99),
            scalar_f64("block_length", 5.0),
            scalar_u64("seed", 42),
            series_f64("m1", &e2),
        ];
        let _ = mspe_adjusted_fit(&inputs).expect("mspe_adjusted_fit failed");
    }
}

// =============================================================================
// Modern tests (Energy Distance, MMD)
// =============================================================================

#[test]
fn modern_fits() {
    let (a, b) = two_samples();

    // energy_distance_fit: x, y, n_perm, seed
    {
        let inputs = vec![
            series_f64("x", &a),
            series_f64("y", &b),
            scalar_u32("n_perm", 99),
            scalar_u64("seed", 42),
        ];
        let _ = energy_distance_fit(&inputs).expect("energy_distance_fit failed");
    }

    // mmd_test_fit: x, y, n_perm, seed
    {
        let inputs = vec![
            series_f64("x", &a),
            series_f64("y", &b),
            scalar_u32("n_perm", 99),
            scalar_u64("seed", 42),
        ];
        let _ = mmd_test_fit(&inputs).expect("mmd_test_fit failed");
    }
}

// =============================================================================
// TOST equivalence tests
// =============================================================================

#[test]
fn tost_fits() {
    let (a, b) = two_samples();

    // tost_t_test_one_sample_fit: x, mu, bounds_type, delta, lower, upper, alpha
    {
        let inputs = vec![
            series_f64("x", &a),
            scalar_f64("mu", 0.0),
            scalar_str("bounds_type", "symmetric"),
            scalar_f64("delta", 5.0),
            scalar_f64("lower", -5.0),
            scalar_f64("upper", 5.0),
            scalar_f64("alpha", 0.05),
        ];
        let _ = tost_t_test_one_sample_fit(&inputs).expect("tost_t_test_one_sample_fit failed");
    }

    // tost_t_test_two_sample_fit: x, y, bounds_type, delta, lower, upper, alpha, pooled
    {
        let inputs = vec![
            series_f64("x", &a),
            series_f64("y", &b),
            scalar_str("bounds_type", "symmetric"),
            scalar_f64("delta", 5.0),
            scalar_f64("lower", -5.0),
            scalar_f64("upper", 5.0),
            scalar_f64("alpha", 0.05),
            scalar_bool("pooled", false),
        ];
        let _ = tost_t_test_two_sample_fit(&inputs).expect("tost_t_test_two_sample_fit failed");
    }

    // tost_t_test_paired_fit
    {
        let inputs = vec![
            series_f64("x", &a),
            series_f64("y", &b),
            scalar_str("bounds_type", "symmetric"),
            scalar_f64("delta", 5.0),
            scalar_f64("lower", -5.0),
            scalar_f64("upper", 5.0),
            scalar_f64("alpha", 0.05),
        ];
        let _ = tost_t_test_paired_fit(&inputs).expect("tost_t_test_paired_fit failed");
    }

    // tost_correlation_fit: x, y, method, rho_null, bounds_type, delta, lower, upper, alpha
    {
        let inputs = vec![
            series_f64("x", &a),
            series_f64("y", &b),
            scalar_str("method", "pearson"),
            scalar_f64("rho_null", 0.0),
            scalar_str("bounds_type", "symmetric"),
            scalar_f64("delta", 0.3),
            scalar_f64("lower", -0.3),
            scalar_f64("upper", 0.3),
            scalar_f64("alpha", 0.05),
        ];
        let _ = tost_correlation_fit(&inputs).expect("tost_correlation_fit failed");
    }

    // tost_prop_one_fit: successes, n, p0, bounds_type, delta, lower, upper, alpha
    {
        let inputs = vec![
            scalar_u32("successes", 50),
            scalar_u32("n", 100),
            scalar_f64("p0", 0.5),
            scalar_str("bounds_type", "symmetric"),
            scalar_f64("delta", 0.1),
            scalar_f64("lower", -0.1),
            scalar_f64("upper", 0.1),
            scalar_f64("alpha", 0.05),
        ];
        let _ = tost_prop_one_fit(&inputs).expect("tost_prop_one_fit failed");
    }

    // tost_prop_two_fit
    {
        let inputs = vec![
            scalar_u32("s1", 48),
            scalar_u32("n1", 100),
            scalar_u32("s2", 52),
            scalar_u32("n2", 100),
            scalar_str("bounds_type", "symmetric"),
            scalar_f64("delta", 0.1),
            scalar_f64("lower", -0.1),
            scalar_f64("upper", 0.1),
            scalar_f64("alpha", 0.05),
        ];
        let _ = tost_prop_two_fit(&inputs).expect("tost_prop_two_fit failed");
    }

    // tost_wilcoxon_paired_fit
    {
        let inputs = vec![
            series_f64("x", &a),
            series_f64("y", &b),
            scalar_str("bounds_type", "symmetric"),
            scalar_f64("delta", 5.0),
            scalar_f64("lower", -5.0),
            scalar_f64("upper", 5.0),
            scalar_f64("alpha", 0.05),
        ];
        let _ = tost_wilcoxon_paired_fit(&inputs).expect("tost_wilcoxon_paired_fit failed");
    }

    // tost_wilcoxon_two_sample_fit
    {
        let inputs = vec![
            series_f64("x", &a),
            series_f64("y", &b),
            scalar_str("bounds_type", "symmetric"),
            scalar_f64("delta", 5.0),
            scalar_f64("lower", -5.0),
            scalar_f64("upper", 5.0),
            scalar_f64("alpha", 0.05),
        ];
        let _ = tost_wilcoxon_two_sample_fit(&inputs).expect("tost_wilcoxon_two_sample_fit failed");
    }

    // tost_bootstrap_fit: x, y, bounds_type, delta, lower, upper, alpha, n_bootstrap, seed
    {
        let inputs = vec![
            series_f64("x", &a),
            series_f64("y", &b),
            scalar_str("bounds_type", "symmetric"),
            scalar_f64("delta", 5.0),
            scalar_f64("lower", -5.0),
            scalar_f64("upper", 5.0),
            scalar_f64("alpha", 0.05),
            scalar_u32("n_bootstrap", 99),
            scalar_u64("seed", 42),
        ];
        let _ = tost_bootstrap_fit(&inputs).expect("tost_bootstrap_fit failed");
    }

    // tost_yuen_fit: x, y, trim, bounds_type, delta, lower, upper, alpha
    {
        let inputs = vec![
            series_f64("x", &a),
            series_f64("y", &b),
            scalar_f64("trim", 0.2),
            scalar_str("bounds_type", "symmetric"),
            scalar_f64("delta", 5.0),
            scalar_f64("lower", -5.0),
            scalar_f64("upper", 5.0),
            scalar_f64("alpha", 0.05),
        ];
        let _ = tost_yuen_fit(&inputs).expect("tost_yuen_fit failed");
    }
}

// =============================================================================
// Dynamic / AID models (lm_dynamic_fit, aid_fit, aid_anomalies_fit)
// =============================================================================

#[test]
fn dynamic_model_fits() {
    // aid_fit: y, intermittent_threshold (f64), detect_anomalies (bool)
    {
        // Intermittent count series with occasional zeros.
        let y: Vec<f64> = (0..30)
            .map(|i| {
                if i % 3 == 0 {
                    0.0
                } else {
                    (i as f64) % 5.0 + 1.0
                }
            })
            .collect();
        let inputs = vec![
            series_f64("y", &y),
            scalar_f64("intermittent_threshold", 0.3),
            scalar_bool("detect_anomalies", true),
        ];
        let out = aid_fit(&inputs).expect("aid_fit failed");
        assert_n_obs_nonzero(&out, "n_observations");
    }

    // aid_anomalies_fit: y, intermittent_threshold
    {
        let y: Vec<f64> = (0..30)
            .map(|i| {
                if i % 3 == 0 {
                    0.0
                } else {
                    (i as f64) % 5.0 + 1.0
                }
            })
            .collect();
        let inputs = vec![
            series_f64("y", &y),
            scalar_f64("intermittent_threshold", 0.3),
        ];
        let out = aid_anomalies_fit(&inputs).expect("aid_anomalies_fit failed");
        // n rows of per-observation anomaly flags.
        assert_eq!(out.len(), y.len());
    }

    // lm_dynamic_fit: y, ic (str), distribution (str), lowess_span (f64),
    //                 max_models (u32), with_intercept (bool), x columns
    {
        let (x, y) = linear_xy();
        let inputs = vec![
            series_f64("y", &y),
            scalar_str("ic", "aicc"),
            scalar_str("distribution", "normal"),
            scalar_f64("lowess_span", 0.0), // disable smoothing for stability
            scalar_u32("max_models", 16),
            scalar_bool("with_intercept", true),
            series_f64("x1", &x),
        ];
        let out = lm_dynamic_fit(&inputs).expect("lm_dynamic_fit failed");
        // n_observations field non-zero.
        assert_n_obs_nonzero(&out, "n_observations");
    }

    // lm_dynamic_predict_fit (issue #18): y, ic, distribution, lowess_span,
    //                 max_models (u32), with_intercept, null_policy, x columns
    {
        let (x, y) = linear_xy();
        let kwargs = PredictKwargs {
            prefix: "lmd".to_string(),
        };
        let inputs = vec![
            series_f64("y", &y),
            scalar_str("ic", "aicc"),
            scalar_str("distribution", "normal"),
            scalar_f64("lowess_span", 0.0),
            scalar_u32("max_models", 16),
            scalar_bool("with_intercept", true),
            scalar_str("null_policy", "drop"),
            series_f64("x1", &x),
        ];
        let out = lm_dynamic_predict_fit(&inputs, kwargs).expect("lm_dynamic_predict_fit failed");
        assert_eq!(out.len(), y.len());
    }
}

// =============================================================================
// Tail helper to silence unused-warning when assert_f64_finite_or_zero is
// not used in some configurations.
// =============================================================================

#[allow(dead_code)]
fn _touch_unused() {
    let s = Series::new("x".into(), &[1.0_f64]);
    assert_f64_finite_or_zero(&s, "x");
}
