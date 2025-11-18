/////////////////////////////////////////////////////////////////////////////////////////////
//
// Defines shared helpers for random point generation, extent padding, CSV I/O, and scaling utilities.
//
// Created on: 15 Nov 2025     Author: Daniel Owen 
//
// Copyright (c) 2025, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License. 
//
/////////////////////////////////////////////////////////////////////////////////////////////

use ferreus_rbf_utils;
use faer::{Mat, MatRef};
use csv::{ReaderBuilder, Writer};
use core::f64;
use std::fs::File;
use std::error::Error;
use std::fmt::Debug;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Round a value down to the nearest multiple of resolution
pub (crate) fn round_down(value: &f64, resolution: &f64) -> f64 {
    (value / resolution).floor() * resolution
}

/// Round a value up to the nearest multiple of resolution
pub (crate) fn round_up(value: &f64, resolution: &f64) -> f64 {
    (value / resolution).ceil() * resolution
}

/// Generate a matrix of random points in the unit hypercube.
///
/// # Parameters
/// - `n`: Number of points to generate (rows in the output matrix).
/// - `d`: Number of spatial dimensions per point (columns in the output matrix).
/// - `seed`: Optional random seed.  
///   - If `Some(seed)` is provided, the same sequence of points will be generated
///     deterministically across runs and platforms (useful for reproducible tests).
///   - If `None`, the generator is seeded from the operating system’s randomness source.
///
/// # Returns
/// A `Mat<f64>` of shape `(n, d)` where each element lies in `[0.0, 1.0)`.
///
/// # Example
/// ```
/// use ferreus_rbf::generate_random_points;
///
/// // Generate 100 reproducible 3D points
/// let pts = generate_random_points(100, 3, Some(42));
/// assert_eq!(pts.ncols(), 3);
/// ```
pub fn generate_random_points(n: usize, d: usize, seed: Option<u64>) -> Mat<f64> {
    let mut rng = match seed.is_some() {
        true => StdRng::seed_from_u64(seed.unwrap()),
        false => StdRng::from_os_rng(),
    };

    let source_points = Mat::from_fn(n, d, |_, _| rng.random_range(0.0..1.0));

    source_points
}

/// Pads and snaps the extents vector (2D or 3D) to the nearest multiple of resolution,
/// then expands the bounds by one resolution unit and the given buffer.
///
/// # Arguments
/// * `extents` - A `Vec<f64>` with either 4 elements (2D) or 6 elements (3D)
/// * `resolution` - Grid resolution
/// * `buffer` - Additional padding added to each side after snapping
///
/// # Returns
/// A new `Vec<f64>` with padded and snapped extents
pub fn pad_and_snap_extents(initial_extents: &Vec<f64>, resolution: &f64, buffer: &f64) -> Vec<f64> {
    let mut extents = initial_extents.clone();
    match extents.len() {
        4 => {
            // 2D: [xmin, ymin, xmax, ymax]
            extents[0] = round_down(&extents[0], &resolution) - resolution - buffer;
            extents[1] = round_down(&extents[1], &resolution) - resolution - buffer;
            extents[2] = round_up(&extents[2], &resolution) + resolution + buffer;
            extents[3] = round_up(&extents[3], &resolution) + resolution + buffer;
        }
        6 => {
            // 3D: [xmin, ymin, zmin, xmax, ymax, zmax]
            extents[0] = round_down(&extents[0], &resolution) - resolution - buffer;
            extents[1] = round_down(&extents[1], &resolution) - resolution - buffer;
            extents[2] = round_down(&extents[2], &resolution) - resolution - buffer;
            extents[3] = round_up(&extents[3], &resolution) + resolution + buffer;
            extents[4] = round_up(&extents[4], &resolution) + resolution + buffer;
            extents[5] = round_up(&extents[5], &resolution) + resolution + buffer;
        }
        _ => panic!("Expected extents of length 4 (2D) or 6 (3D), got {}", extents.len()),
    }

    extents
}

/// Create a regular evaluation grid from per-dimension ranges and sample counts.
///
/// # Arguments
/// * `ranges` - Inclusive `(min, max)` range for each dimension.
/// * `counts` - Number of grid samples per range; must match `ranges.len()`.
///
/// # Returns
/// A `Mat<f64>` with one row per grid point and one column per dimension.
pub fn create_evaluation_grid(
    ranges: &[(f64, f64)],
    counts: &[usize],
) -> Mat<f64> {
    assert_eq!(ranges.len(), counts.len());

    let dimensions = counts.to_vec();
    let total_points: usize = dimensions.iter().product();
    let num_dimensions = ranges.len();

    Mat::from_fn(total_points, num_dimensions, |row_idx, col_idx| {
        let dim_points = dimensions[col_idx];
        let (start, end) = ranges[col_idx];
        let step = (end - start) / (dim_points as f64 - 1.0);

        let stride = match col_idx == 0 {
            true => 1,
            false => dimensions[..col_idx].iter().product::<usize>(),
        };

        let index_in_dim = (row_idx / stride) % dim_points;
        start + step * index_in_dim as f64
    })
}

/// Load a CSV file into separate point and value matrices.
///
/// The last column is treated as the scalar value, and all preceding
/// columns form the point coordinates.
///
/// # Arguments
/// * `file_path` - Path to the CSV file.
/// * `has_headers` - Whether the file has a single header row to skip.
///
/// # Returns
/// On success, returns `(points, values)` where `points` has shape
/// `(n_rows, n_cols - 1)` and `values` has shape `(n_rows, 1)`.
pub fn csv_to_point_arrays(
    file_path: &str,
    has_headers: bool,
) -> Result<(Mat<f64>, Mat<f64>), Box<dyn Error>> {
    // Open the CSV file
    let file = File::open(file_path)?;
    let mut reader = ReaderBuilder::new()
        .has_headers(has_headers)
        .from_reader(file);

    // Initialize vectors for the points array and values
    let mut data = Vec::new();
    let mut last_column = Vec::new();
    let mut num_rows = 0;
    let mut num_cols = 0;

    // Read and parse CSV
    for result in reader.records() {
        let record = result?;
        if num_cols == 0 {
            num_cols = record.len();
        } else if record.len() != num_cols {
            return Err("Inconsistent number of columns in CSV".into());
        }

        // Extract values, separating the last column
        for (i, value) in record.iter().enumerate() {
            let parsed_value: f64 = value.parse()?;
            if i == num_cols - 1 {
                last_column.push(parsed_value);
            } else {
                data.push(parsed_value);
            }
        }

        num_rows += 1;
    }

    // Construct the point_arrays
    let points = MatRef::from_row_major_slice(data.as_slice(), num_rows, num_cols - 1).to_owned();
    let values = MatRef::from_row_major_slice(last_column.as_slice(), num_rows, 1).to_owned();

    
    Ok((points, values))
}

/// Write point coordinates and associated values to a CSV file.
///
/// Each row of `points` is written followed by the corresponding value
/// from `values`, with headers `X, Y, Z, InterpolatedValue`.
///
/// # Arguments
/// * `points` - Matrix of point coordinates (rows are points).
/// * `values` - Column matrix of scalar values; must match `points` rows.
/// * `filename` - Output CSV filename.
///
/// # Errors
/// Returns an error if writing to disk fails.
pub fn point_arrays_to_csv<T>(
    points: &Mat<T>,
    values: &Mat<T>,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>>
where
    T: std::fmt::Display + Debug + Clone + Send + Sync + PartialOrd + 'static,
{
    let num_points = points.shape().0;
    assert_eq!(num_points, values.shape().0,  "Points and values must have same length.");

    let mut wtr = Writer::from_path(filename)?;

    let headers = vec!["X", "Y", "Z", "InterpolatedValue"];
    wtr.write_record(&headers)?;

    for i in 0..num_points {
        let mut record: Vec<String> = points.row(i).iter().map(|c| c.to_string()).collect();
        record.push(values.get(i, 0).to_string());
        wtr.write_record(&record)?;
    }

    wtr.flush()?;
    Ok(())
}

/// Perform farthest point sampling on a point cloud.
///
/// Starting from the provided `seed_index`, iteratively selects points
/// that maximize the minimum distance to the already selected subset.
///
/// # Arguments
/// * `points` - Matrix of point coordinates (rows are points).
/// * `num_wanted_points` - Number of points to sample.
/// * `seed_index` - Index of the initial seed point.
///
/// # Returns
/// A vector of indices into `points` representing the sampled subset.
pub fn farthest_point_sampling(points: &Mat<f64>, num_wanted_points: &usize, seed_index: &usize) -> Vec<usize>
{
    let num_points = points.shape().0;
    let mut selected_points: Vec<usize> = Vec::with_capacity(*num_wanted_points);
    let mut is_selected = vec![false; num_points];
    let mut min_dists = vec![f64::INFINITY; num_points];

    selected_points.push(seed_index.clone());
    is_selected[*seed_index] = true;

    for _ in 1..*num_wanted_points {
        let last_selected = *selected_points.last().unwrap();

        for i in 0..num_points {
            if is_selected[i] {
                continue;
            }
            let dist = ferreus_rbf_utils::get_distance(points.row(last_selected), points.row(i));
            if dist < min_dists[i] {
                min_dists[i] = dist;
            }
        }

        // Select the farthest (max-min-distance) point
        let mut farthest_idx = 0;
        let mut max_dist = -1.0;
        for (i, &dist) in min_dists.iter().enumerate() {
            if !is_selected[i] && dist > max_dist {
                max_dist = dist;
                farthest_idx = i;
            }
        }

        selected_points.push(farthest_idx);
        is_selected[farthest_idx] = true;
    }

    selected_points
}

/// Compute translation and scale factors to map points into a Chebyshev cube.
///
/// The translation is the midpoint of each coordinate range and the scale
/// is half the range, with zeros replaced by `1.0` to avoid division by zero.
///
/// # Arguments
/// * `point_locations` - Matrix of point coordinates (rows are points).
///
/// # Returns
/// A tuple `(translation, scale)` where each is a per-dimension factor.
pub fn get_cheb_cube_scaling_factors(point_locations: &Mat<f64>) -> (Vec<f64>, Vec<f64>) 
{
    let dimensions = point_locations.shape().1;
    let extents = ferreus_rbf_utils::get_pointarray_extents(&point_locations);

    let mut translation_factor: Vec<f64> = Vec::with_capacity(dimensions);
    let mut scale_factor: Vec<f64> = Vec::with_capacity(dimensions);

    (0..dimensions)
        .into_iter()
        .for_each(|d| {
            let max_coord = extents[d + dimensions];
            let min_coord = extents[d];
            translation_factor.push((max_coord + min_coord) / 2.0);
            scale_factor.push((max_coord - min_coord) / 2.0);
        });

    scale_factor
        .iter_mut()
        .for_each(|element| {
            if *element == 0.0 {
                *element = 1.0;
            }
        });

    (translation_factor, scale_factor)
}

/// Apply translation and scaling to map points into a normalized cube.
///
/// For each coordinate `x`, applies `(x - translation_factor[d]) / scale_factor[d]`.
///
/// # Arguments
/// * `points` - Matrix of point coordinates to be transformed in-place.
/// * `translation_factor` - Per-dimension translation factors.
/// * `scale_factor` - Per-dimension scale factors.
pub fn scale_points(points: &mut Mat<f64>, translation_factor: &Vec<f64>, scale_factor: &Vec<f64>) 
{
    points
        .row_iter_mut()
        .for_each(|row| {
            row
                .iter_mut()
                .enumerate()
                .for_each(|(col_idx, element)| {
                    *element = (*element - translation_factor[col_idx]) / scale_factor[col_idx];
                });
        });
}
