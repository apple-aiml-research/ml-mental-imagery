//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
pub mod base;
pub mod games;
#[cfg(feature = "python")]
pub mod python;
pub mod solver;
pub mod stream;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(m)
}
