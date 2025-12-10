//! PyO3 wrapper for Automatic Identification of Demand (AID) classifier.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use anofox_regression::solvers::{
    AidClassifier, AnomalyType, DemandClassification, DemandDistribution, DemandType,
    InformationCriterion,
};

use crate::utils::ToFaer;

/// Convert string to InformationCriterion enum.
fn parse_ic(s: &str) -> PyResult<InformationCriterion> {
    match s.to_lowercase().as_str() {
        "aic" => Ok(InformationCriterion::AIC),
        "aicc" | "aic_c" => Ok(InformationCriterion::AICc),
        "bic" => Ok(InformationCriterion::BIC),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown information criterion: '{}'. Available: aic, aicc, bic",
            s
        ))),
    }
}

/// Convert DemandType to string.
fn demand_type_to_str(dt: DemandType) -> &'static str {
    match dt {
        DemandType::Regular => "regular",
        DemandType::Intermittent => "intermittent",
    }
}

/// Convert DemandDistribution to string.
fn distribution_to_str(dist: DemandDistribution) -> &'static str {
    match dist {
        DemandDistribution::Poisson => "poisson",
        DemandDistribution::NegativeBinomial => "negative_binomial",
        DemandDistribution::Geometric => "geometric",
        DemandDistribution::Normal => "normal",
        DemandDistribution::Gamma => "gamma",
        DemandDistribution::LogNormal => "lognormal",
        DemandDistribution::RectifiedNormal => "rectified_normal",
    }
}

/// Convert AnomalyType to string.
fn anomaly_to_str(anomaly: AnomalyType) -> &'static str {
    match anomaly {
        AnomalyType::None => "none",
        AnomalyType::Stockout => "stockout",
        AnomalyType::NewProduct => "new_product",
        AnomalyType::ObsoleteProduct => "obsolete_product",
        AnomalyType::HighOutlier => "high_outlier",
        AnomalyType::LowOutlier => "low_outlier",
    }
}

/// Result from AID classification.
#[pyclass(name = "AidResult")]
pub struct PyAidResult {
    result: DemandClassification,
}

#[pymethods]
impl PyAidResult {
    /// Get the primary demand type ("regular" or "intermittent").
    #[getter]
    fn demand_type(&self) -> &str {
        demand_type_to_str(self.result.demand_type)
    }

    /// Check if demand is regular.
    fn is_regular(&self) -> bool {
        self.result.demand_type == DemandType::Regular
    }

    /// Check if demand is intermittent.
    fn is_intermittent(&self) -> bool {
        self.result.demand_type == DemandType::Intermittent
    }

    /// Check if data contains fractional values.
    #[getter]
    fn is_fractional(&self) -> bool {
        self.result.is_fractional
    }

    /// Get the best-fitting distribution name.
    #[getter]
    fn distribution(&self) -> &str {
        distribution_to_str(self.result.distribution)
    }

    /// Get the estimated mean.
    #[getter]
    fn mean(&self) -> f64 {
        self.result.parameters.mean
    }

    /// Get the estimated variance.
    #[getter]
    fn variance(&self) -> f64 {
        self.result.parameters.variance
    }

    /// Get the shape parameter (for Gamma, NegBinom) if applicable.
    #[getter]
    fn shape(&self) -> Option<f64> {
        self.result.parameters.shape
    }

    /// Get the estimated probability of zero.
    #[getter]
    fn zero_prob(&self) -> Option<f64> {
        self.result.parameters.zero_prob
    }

    /// Get the scale parameter if applicable.
    #[getter]
    fn scale(&self) -> Option<f64> {
        self.result.parameters.scale
    }

    /// Get the zero proportion in the data.
    #[getter]
    fn zero_proportion(&self) -> f64 {
        self.result.zero_proportion
    }

    /// Get number of observations.
    #[getter]
    fn n_observations(&self) -> usize {
        self.result.n_observations
    }

    /// Get anomaly flags as a list of strings.
    #[getter]
    fn anomalies<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyList> {
        let anomaly_strs: Vec<&str> = self
            .result
            .anomalies
            .iter()
            .map(|a| anomaly_to_str(*a))
            .collect();
        pyo3::types::PyList::new(py, anomaly_strs).unwrap()
    }

    /// Get IC values for each candidate distribution as a dict.
    #[getter]
    fn ic_values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (dist, ic) in &self.result.ic_values {
            dict.set_item(distribution_to_str(*dist), *ic)?;
        }
        Ok(dict)
    }

    /// Check if any stockouts were detected.
    fn has_stockouts(&self) -> bool {
        self.result.has_stockouts()
    }

    /// Check if product appears to be new (leading zeros).
    fn is_new_product(&self) -> bool {
        self.result.is_new_product()
    }

    /// Check if product appears to be obsolete (trailing zeros).
    fn is_obsolete_product(&self) -> bool {
        self.result.is_obsolete_product()
    }

    /// Get anomaly counts as a dict.
    fn anomaly_counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let counts = self.result.anomaly_counts();
        let dict = PyDict::new(py);
        for (anomaly, count) in counts {
            dict.set_item(anomaly_to_str(anomaly), count)?;
        }
        Ok(dict)
    }
}

/// Automatic Identification of Demand (AID) classifier.
///
/// Analyzes time series data to identify whether demand is regular or intermittent,
/// and selects the best-fitting distribution. Based on the aid function from
/// the greybox R package.
///
/// Parameters
/// ----------
/// intermittent_threshold : float, default 0.3
///     Proportion of zeros above which demand is classified as intermittent.
/// detect_anomalies : bool, default True
///     Whether to detect anomalies (stockouts, lifecycle events, outliers).
/// ic : str, default "aicc"
///     Information criterion for distribution selection.
///     Options: "aic", "aicc", "bic"
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> from polars_statistics import Aid
/// >>>
/// >>> # Regular demand data
/// >>> demand = np.random.poisson(10, 100).astype(float)
/// >>> result = Aid().classify(demand)
/// >>> print(result.demand_type)  # "regular"
/// >>> print(result.distribution)  # e.g., "poisson"
/// >>>
/// >>> # Intermittent demand data
/// >>> demand = np.where(np.random.random(100) < 0.4, 0, np.random.poisson(5, 100))
/// >>> result = Aid(intermittent_threshold=0.3).classify(demand.astype(float))
/// >>> print(result.demand_type)  # "intermittent"
#[pyclass(name = "Aid")]
pub struct PyAid {
    intermittent_threshold: f64,
    detect_anomalies: bool,
    ic: String,
}

#[pymethods]
impl PyAid {
    #[new]
    #[pyo3(signature = (intermittent_threshold=0.3, detect_anomalies=true, ic="aicc"))]
    fn new(intermittent_threshold: f64, detect_anomalies: bool, ic: &str) -> Self {
        Self {
            intermittent_threshold,
            detect_anomalies,
            ic: ic.to_string(),
        }
    }

    /// Classify the demand pattern.
    ///
    /// Parameters
    /// ----------
    /// y : array-like of shape (n_samples,)
    ///     Demand time series data.
    ///
    /// Returns
    /// -------
    /// AidResult
    ///     Classification result containing demand type, best distribution, and anomalies.
    fn classify<'py>(&self, y: PyReadonlyArray1<'py, f64>) -> PyResult<PyAidResult> {
        let y_col = y.to_faer();
        let _ic_type = parse_ic(&self.ic)?;

        let classifier = AidClassifier::builder()
            .intermittent_threshold(self.intermittent_threshold)
            .detect_anomalies(self.detect_anomalies)
            .build();

        let result = classifier.classify(&y_col);

        Ok(PyAidResult { result })
    }
}
