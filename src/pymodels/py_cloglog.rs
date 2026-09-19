//! PyO3 wrapper for Complementary Log-Log regression.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use anofox_regression::solvers::{BinomialRegressor, FittedBinomial, FittedRegressor, Regressor};

use crate::utils::{IntoNumpy, ToFaer};

/// Complementary Log-Log regression model (Binomial GLM with cloglog link).
///
/// Uses the complementary log-log link function: g(μ) = log(-log(1-μ))
///
/// Useful when the probability of an event is asymmetric, particularly
/// for modeling time-to-event data or extreme value problems.
///
/// Parameters
/// ----------
/// with_intercept : bool, default True
///     Whether to include an intercept term.
/// max_iter : int, default 100
///     Maximum number of IRLS iterations.
/// tol : float, default 1e-6
///     Convergence tolerance.
#[pyclass(name = "Cloglog")]
pub struct PyCloglog {
    with_intercept: bool,
    max_iter: usize,
    tol: f64,
    lambda_: f64,
    fitted: Option<FittedBinomial>,
}

#[pymethods]
impl PyCloglog {
    #[new]
    #[pyo3(signature = (add_intercept=None, with_intercept=None, max_iter=100, tol=1e-6, lambda_=0.0))]
    fn new(
        py: Python<'_>,
        add_intercept: Option<bool>,
        with_intercept: Option<bool>,
        max_iter: usize,
        tol: f64,
        lambda_: f64,
    ) -> PyResult<Self> {
        let with_intercept =
            crate::pymodels::errors::resolve_intercept(py, add_intercept, with_intercept, true)?;
        Ok(Self {
            with_intercept,
            max_iter,
            tol,
            lambda_,
            fitted: None,
        })
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        crate::pymodels::errors::validate_xy("Cloglog", &x, &y)?;
        let x_mat = x.to_faer();
        let y_col = y.to_faer();

        let model = BinomialRegressor::cloglog()
            .with_intercept(slf.with_intercept)
            .max_iterations(slf.max_iter)
            .tolerance(slf.tol)
            .lambda(slf.lambda_)
            .build();

        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        slf.fitted = Some(fitted);
        Ok(slf)
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Cloglog"))?;

        let x_mat = x.to_faer();
        let predictions = fitted.predict(&x_mat);

        // Convert probabilities to binary predictions
        let binary: Vec<f64> = predictions
            .iter()
            .map(|&p| if p > 0.5 { 1.0 } else { 0.0 })
            .collect();
        Ok(PyArray1::from_vec(py, binary))
    }

    /// Mean classification accuracy on ``(X, y)`` using ``threshold``.
    ///
    /// Predicts class labels (``P(y=1|x) >= threshold``) and returns the
    /// fraction that match ``y``. Consistent with
    /// :meth:`sklearn.base.ClassifierMixin.score` (accuracy for classifiers).
    #[pyo3(signature = (x, y, threshold=0.5))]
    fn score<'py>(
        &self,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        threshold: f64,
    ) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Cloglog"))?;
        let x_mat = x.to_faer();
        let y_arr = y.as_array();
        let probabilities = fitted.predict_probability(&x_mat);
        let n = probabilities.nrows();
        if n != y_arr.len() {
            return Err(crate::pymodels::errors::x_y_row_mismatch_err(
                "Cloglog",
                n,
                y_arr.len(),
            ));
        }
        if n == 0 {
            return Err(crate::pymodels::errors::empty_input_err("Cloglog"));
        }
        let correct = (0..n)
            .filter(|&i| {
                let label = if probabilities[i] >= threshold {
                    1.0
                } else {
                    0.0
                };
                (label - y_arr[i]).abs() < 0.5
            })
            .count();
        Ok(correct as f64 / n as f64)
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Cloglog"))?;

        let x_mat = x.to_faer();
        let predictions = fitted.predict(&x_mat);

        Ok(predictions.into_numpy(py))
    }

    fn is_fitted(&self) -> bool {
        self.fitted.is_some()
    }

    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Cloglog"))?;

        Ok(fitted.coefficients().into_numpy(py))
    }

    #[getter]
    fn intercept(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Cloglog"))?;

        Ok(fitted.intercept())
    }

    #[getter]
    fn std_errors<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Cloglog"))?;

        Ok(fitted
            .result()
            .std_errors
            .as_ref()
            .map(|se| se.into_numpy(py)))
    }

    #[getter]
    fn p_values<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Cloglog"))?;

        Ok(fitted
            .result()
            .p_values
            .as_ref()
            .map(|pv| pv.into_numpy(py)))
    }

    #[getter]
    fn aic(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Cloglog"))?;

        Ok(Some(fitted.result().aic))
    }

    #[getter]
    fn bic(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| crate::pymodels::errors::not_fitted_err("Cloglog"))?;

        Ok(Some(fitted.result().bic))
    }
    /// Informative repr: class name plus key state; never panics if unfitted.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        crate::pymodels::ergonomics::repr(slf.as_any())
    }

    /// Return the results as a plain Python ``dict`` (``{"fitted": False}`` if unfitted).
    fn to_dict<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        crate::pymodels::ergonomics::to_dict(slf.as_any())
    }

    /// Readable multi-line summary of the results.
    fn summary(slf: &Bound<'_, Self>) -> PyResult<String> {
        crate::pymodels::ergonomics::generic_summary(slf.as_any())
    }
}
