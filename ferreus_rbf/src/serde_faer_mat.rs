/////////////////////////////////////////////////////////////////////////////////////////////
//
// Provides methods for serializing/deserializing faer Mat, since faer dropped serde support.
//
// Created on: 5 August 2026     Author: Daniel Owen
//
// Copyright (c) 2026, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! # serde_faer_mat
//!
//! Methods for serializing/deserializing a faer Mat.

use faer::Mat;
use serde::{
    Deserialize, Deserializer, Serialize, Serializer, de,
    ser::{SerializeSeq, SerializeStruct},
};

pub fn serialize<S>(mat: &Mat<f64>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    struct MatrixData<'a>(&'a Mat<f64>);

    impl Serialize for MatrixData<'_> {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            let mat = self.0;
            let mut seq = serializer.serialize_seq(Some(mat.nrows() * mat.ncols()))?;

            for i in 0..mat.nrows() {
                for j in 0..mat.ncols() {
                    seq.serialize_element(&mat[(i, j)])?;
                }
            }

            seq.end()
        }
    }

    let mut st = serializer.serialize_struct("Mat", 3)?;
    st.serialize_field("nrows", &mat.nrows())?;
    st.serialize_field("ncols", &mat.ncols())?;
    st.serialize_field("data", &MatrixData(mat))?;
    st.end()
}

pub fn deserialize<'de, D>(deserializer: D) -> Result<Mat<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    let repr = MatRepr::deserialize(deserializer)?;
    repr.into_mat()
}

#[derive(Deserialize)]
struct MatRepr {
    nrows: usize,
    ncols: usize,
    data: Vec<f64>,
}

impl MatRepr {
    fn into_mat<E>(self) -> Result<Mat<f64>, E>
    where
        E: de::Error,
    {
        let expected = self
            .nrows
            .checked_mul(self.ncols)
            .ok_or_else(|| E::custom("matrix dimensions overflow"))?;

        if self.data.len() != expected {
            return Err(E::invalid_length(
                self.data.len(),
                &"nrows * ncols matrix elements",
            ));
        }

        Ok(Mat::from_fn(self.nrows, self.ncols, |i, j| {
            self.data[i * self.ncols + j]
        }))
    }
}

pub mod option {
    use super::*;

    pub fn serialize<S>(mat: &Option<Mat<f64>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match mat {
            Some(mat) => serializer.serialize_some(&SerializableMat(mat)),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<Mat<f64>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Option::<MatRepr>::deserialize(deserializer)?
            .map(MatRepr::into_mat)
            .transpose()
    }

    struct SerializableMat<'a>(&'a Mat<f64>);

    impl Serialize for SerializableMat<'_> {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            super::serialize(self.0, serializer)
        }
    }
}

pub mod arc {
    use super::*;
    use std::sync::Arc;

    pub fn serialize<S>(
        mat: &Arc<Mat<f64>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        super::serialize(mat.as_ref(), serializer)
    }

    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> Result<Arc<Mat<f64>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        super::deserialize(deserializer).map(Arc::new)
    }
}