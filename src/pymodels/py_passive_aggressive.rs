//! PyO3 wrapper for PassiveAggressive regression (online learner).

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use anofox_regression::solvers::{
    FittedPassiveAggressive, FittedRegressor, PaLoss, PaState, PassiveAggressiveRegressor,
    Regressor,
};

use crate::utils::{IntoNumpy, ToFaer};

/// Passive-Aggressive regression model (online learner).
///
/// Supports both batch fitting (``fit``) and incremental single-sample updates
/// (``partial_fit``). Particularly useful for streaming data where the full
/// design matrix cannot be materialised up front.
///
/// Parameters
/// ----------
/// c : float, default 1.0
///     Aggressiveness / regularisation parameter. Smaller values impose
///     stronger regularisation; must be > 0.
/// epsilon : float, default 0.1
///     Width of the insensitivity band. Residuals smaller than ``epsilon``
///     trigger no update.
/// with_intercept : bool, default True
///     Whether to fit an intercept term.
/// max_iter : int, default 1000
///     Maximum number of passes over the training data during ``fit``.
/// tol : float, default 1e-3
///     Convergence tolerance on weight change between passes.
/// shuffle : bool, default True
///     Whether to shuffle the training data on each pass during ``fit``.
/// loss : str, default "epsilon_insensitive"
///     Loss function: ``"epsilon_insensitive"`` or
///     ``"squared_epsilon_insensitive"``.
/// random_state : int, default 0
///     Seed for the random number generator (used when ``shuffle=True``).
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> from polars_statistics import PassiveAggressive
/// >>>
/// >>> X = np.random.randn(100, 3)
/// >>> y = X @ [1.0, -1.0, 0.5] + 0.1 * np.random.randn(100)
/// >>> model = PassiveAggressive().fit(X, y)
/// >>> model.predict(X).shape
/// (100,)
/// >>> # Online single-sample update:
/// >>> model.partial_fit(X[0], y[0])
#[pyclass(name = "PassiveAggressive")]
pub struct PyPassiveAggressive {
    c: f64,
    epsilon: f64,
    with_intercept: bool,
    max_iter: usize,
    tol: f64,
    shuffle: bool,
    loss: String,
    random_state: u64,
    fitted: Option<FittedPassiveAggressive>,
    state: Option<PaState>,
}

// Plain impl block — helper methods NOT exposed to Python.
impl PyPassiveAggressive {
    /// Build a PassiveAggressiveRegressor from the current field values.
    fn build_model(&self) -> PassiveAggressiveRegressor {
        let pa_loss = match self.loss.as_str() {
            "squared_epsilon_insensitive" => PaLoss::SquaredEpsilonInsensitive,
            _ => PaLoss::EpsilonInsensitive,
        };
        PassiveAggressiveRegressor::builder()
            .c(self.c)
            .epsilon(self.epsilon)
            .with_intercept(self.with_intercept)
            .max_iter(self.max_iter)
            .tolerance(self.tol)
            .shuffle(self.shuffle)
            .loss(pa_loss)
            .random_state(self.random_state)
            .build()
    }
}

#[pymethods]
impl PyPassiveAggressive {
    #[new]
    #[pyo3(signature = (c=1.0, epsilon=0.1, with_intercept=true, max_iter=1000, tol=1e-3, shuffle=true, loss="epsilon_insensitive", random_state=0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        c: f64,
        epsilon: f64,
        with_intercept: bool,
        max_iter: usize,
        tol: f64,
        shuffle: bool,
        loss: &str,
        random_state: u64,
    ) -> Self {
        Self {
            c,
            epsilon,
            with_intercept,
            max_iter,
            tol,
            shuffle,
            loss: loss.to_owned(),
            random_state,
            fitted: None,
            state: None,
        }
    }

    /// Fit the model on the full training set.
    ///
    /// Parameters
    /// ----------
    /// X : array-like of shape (n_samples, n_features)
    ///     Training data.
    /// y : array-like of shape (n_samples,)
    ///     Target values.
    ///
    /// Returns
    /// -------
    /// self
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let x_mat = x.to_faer();
        let y_col = y.to_faer();
        let model = slf.build_model();
        let fitted = model
            .fit(&x_mat, &y_col)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        slf.fitted = Some(fitted);
        Ok(slf)
    }

    /// Perform a single online update with one sample.
    ///
    /// Lazily initialises the streaming state on the first call. The state
    /// persists across successive ``partial_fit`` calls so the model learns
    /// incrementally.
    ///
    /// Parameters
    /// ----------
    /// x_row : 1-D array of shape (n_features,)
    ///     Feature values for a single sample.
    /// y_value : float
    ///     Target value for the sample.
    fn partial_fit(&mut self, x_row: PyReadonlyArray1<'_, f64>, y_value: f64) -> PyResult<()> {
        let slice = x_row.as_slice().unwrap();
        let n_features = slice.len();
        // Build model before taking mutable borrow of state to avoid borrow conflict.
        let model = self.build_model();
        let state = self.state.get_or_insert_with(|| PaState::new(n_features));
        model
            .partial_fit(state, slice, y_value)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// Predict response values.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        let x_mat = x.to_faer();
        Ok(fitted.predict(&x_mat).into_numpy(py))
    }

    fn is_fitted(&self) -> bool {
        self.fitted.is_some()
    }

    /// Number of training iterations performed (early-stop count).
    #[getter]
    fn n_iter(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        Ok(fitted.n_iter())
    }

    /// Final model coefficients (excluding intercept).
    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        Ok(fitted.coefficients().into_numpy(py))
    }

    /// Intercept term (None if with_intercept=False).
    #[getter]
    fn intercept(&self) -> PyResult<Option<f64>> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        Ok(fitted.intercept())
    }

    /// R-squared coefficient of determination.
    #[getter]
    fn r_squared(&self) -> PyResult<f64> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        Ok(fitted.result().r_squared)
    }

    /// Number of observations used in the batch fit.
    #[getter]
    fn n_observations(&self) -> PyResult<usize> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Model not fitted"))?;
        Ok(fitted.result().n_observations)
    }
}
