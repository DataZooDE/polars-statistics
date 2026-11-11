//! Shared input-coercion helpers for every expression `*_fit` function.
//!
//! Every `*_fit` function in `src/expressions` used to read its inputs through Polars'
//! strict typed accessors (`Series::f64()`, `Series::u32()`), which error on any dtype
//! other than the exact one requested. That meant `ps.spearman("char_tokens", "char_types")`
//! on two `UInt32` columns raised `"invalid series dtype: expected `Float64`, got `u32`"`
//! (GH #52), even though every value trivially fits in a `f64`.
//!
//! `as_f64` and `as_u32` are the single coercion point for the fix: every input accessor
//! in `src/expressions` routes through one of them instead of calling the typed accessor
//! directly.

use polars::prelude::*;

/// Coerce a `Series` to an owned `Float64Chunked`, upcasting any numeric, boolean or
/// all-null dtype.
///
/// - `Float64` input is returned as-is (an `Arc` clone of the existing chunked array; no
///   data copy).
/// - Any dtype where [`DataType::is_primitive_numeric`] (`Int8`..`Int64`, `UInt8`..`UInt64`,
///   `Float32`), [`DataType::is_bool`] or [`DataType::is_null`] is true is cast to `Float64`.
///   Nulls are preserved through the cast. `Boolean` maps to `0.0` / `1.0`. Integers with a
///   magnitude above 2^53 round to the nearest representable `f64` — the same rounding the
///   Python wrappers already perform via `.cast(pl.Float64)`.
/// - Every other dtype (String, Date, Datetime, Duration, Decimal, Categorical, List,
///   Struct, ...) falls through to the plain `Float64` accessor, so the caller still gets
///   Polars' original `"expected `Float64`, got ..."` error.
pub(crate) fn as_f64(s: &Series) -> PolarsResult<Float64Chunked> {
    let dtype = s.dtype();
    if matches!(dtype, DataType::Float64) {
        return Ok(s.f64()?.clone());
    }
    if dtype.is_primitive_numeric() || dtype.is_bool() || dtype.is_null() {
        let casted = s.cast(&DataType::Float64)?;
        return Ok(casted.f64()?.clone());
    }
    Ok(s.f64()?.clone())
}

/// Coerce a `Series` to an owned `UInt32Chunked` via a *strict* cast, upcasting any
/// integer or all-null dtype.
///
/// - `UInt32` input is returned as-is (an `Arc` clone; no data copy).
/// - Any dtype where [`DataType::is_integer`] or [`DataType::is_null`] is true is
///   strict-cast to `UInt32`. The cast is strict on purpose: a negative value or a value
///   above `u32::MAX` must error, never be silently nulled (a non-strict cast's behaviour)
///   or wrapped/truncated.
/// - Float dtypes (and every other non-integer dtype) are rejected on purpose — `as_u32`
///   never silently truncates a fractional value — and fall through to the plain `UInt32`
///   accessor's original error.
pub(crate) fn as_u32(s: &Series) -> PolarsResult<UInt32Chunked> {
    let dtype = s.dtype();
    if matches!(dtype, DataType::UInt32) {
        return Ok(s.u32()?.clone());
    }
    if dtype.is_integer() || dtype.is_null() {
        let casted = s.strict_cast(&DataType::UInt32)?;
        return Ok(casted.u32()?.clone());
    }
    Ok(s.u32()?.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn as_f64_upcasts_every_integer_width_and_float32() {
        macro_rules! check {
            ($ctor:expr, $vals:expr) => {{
                let s = $ctor;
                let out = as_f64(&s).expect("as_f64 should upcast");
                let collected: Vec<Option<f64>> = out.into_iter().collect();
                assert_eq!(collected, $vals);
            }};
        }

        check!(
            Series::new("x".into(), &[1i8, 2, 3]),
            vec![Some(1.0), Some(2.0), Some(3.0)]
        );
        check!(
            Series::new("x".into(), &[1i16, 2, 3]),
            vec![Some(1.0), Some(2.0), Some(3.0)]
        );
        check!(
            Series::new("x".into(), &[1i32, 2, 3]),
            vec![Some(1.0), Some(2.0), Some(3.0)]
        );
        check!(
            Series::new("x".into(), &[1i64, 2, 3]),
            vec![Some(1.0), Some(2.0), Some(3.0)]
        );
        check!(
            Series::new("x".into(), &[1u8, 2, 3]),
            vec![Some(1.0), Some(2.0), Some(3.0)]
        );
        check!(
            Series::new("x".into(), &[1u16, 2, 3]),
            vec![Some(1.0), Some(2.0), Some(3.0)]
        );
        check!(
            Series::new("x".into(), &[1u32, 2, 3]),
            vec![Some(1.0), Some(2.0), Some(3.0)]
        );
        check!(
            Series::new("x".into(), &[1u64, 2, 3]),
            vec![Some(1.0), Some(2.0), Some(3.0)]
        );
        check!(
            Series::new("x".into(), &[1.0f32, 2.0, 3.0]),
            vec![Some(1.0), Some(2.0), Some(3.0)]
        );
        check!(
            Series::new("x".into(), &[true, false, true]),
            vec![Some(1.0), Some(0.0), Some(1.0)]
        );
    }

    #[test]
    fn as_f64_preserves_nulls() {
        let s: Series = Series::new("x".into(), &[Some(1i64), None, Some(3i64)]);
        let out = as_f64(&s).expect("as_f64 should upcast with a null");
        let collected: Vec<Option<f64>> = out.into_iter().collect();
        assert_eq!(collected, vec![Some(1.0), None, Some(3.0)]);
    }

    #[test]
    fn as_f64_on_float64_returns_equal_values() {
        let s = Series::new("x".into(), &[1.5f64, 2.5, 3.5]);
        let out = as_f64(&s).expect("as_f64 on Float64 must succeed");
        let collected: Vec<Option<f64>> = out.into_iter().collect();
        assert_eq!(collected, vec![Some(1.5), Some(2.5), Some(3.5)]);
    }

    #[test]
    fn as_f64_on_all_null_series_stays_all_null() {
        let s = Series::new_null("x".into(), 3);
        let out = as_f64(&s).expect("as_f64 on an all-null series must succeed");
        assert_eq!(out.len(), 3);
        let collected: Vec<Option<f64>> = out.into_iter().collect();
        assert_eq!(collected, vec![None, None, None]);
    }

    #[test]
    fn as_f64_on_string_errors_naming_float64() {
        let s = Series::new("x".into(), &["a", "b"]);
        let err = as_f64(&s).expect_err("as_f64 on a String series must error");
        assert!(
            err.to_string().contains("Float64"),
            "error should name Float64: {err}"
        );
    }

    #[test]
    fn as_u32_upcasts_integer_widths() {
        let s = Series::new("x".into(), &[0i64, 5, 7]);
        let out = as_u32(&s).expect("as_u32 on Int64 should succeed");
        let collected: Vec<Option<u32>> = out.into_iter().collect();
        assert_eq!(collected, vec![Some(0), Some(5), Some(7)]);

        let s = Series::new("x".into(), &[0u64, 5, 7]);
        let out = as_u32(&s);
        assert!(out.is_ok(), "as_u32 on UInt64 should succeed");
    }

    #[test]
    fn as_u32_rejects_negative_values() {
        let s = Series::new("x".into(), &[-1i64]);
        let out = as_u32(&s);
        assert!(out.is_err(), "as_u32 must error on a negative Int64 value");
    }

    #[test]
    fn as_u32_rejects_out_of_range_values() {
        let s = Series::new("x".into(), &[(u32::MAX as i64) + 1]);
        let out = as_u32(&s);
        assert!(
            out.is_err(),
            "as_u32 must error on an Int64 value above u32::MAX"
        );
    }

    #[test]
    fn as_u32_rejects_float_input() {
        let s = Series::new("x".into(), &[1.0f64]);
        let out = as_u32(&s);
        assert!(out.is_err(), "as_u32 must reject Float64 input");
    }

    #[test]
    fn as_u32_rejects_boolean_input() {
        let s = Series::new("x".into(), &[true]);
        let out = as_u32(&s);
        assert!(out.is_err(), "as_u32 must reject Boolean input");
    }
}
