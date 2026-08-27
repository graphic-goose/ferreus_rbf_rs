/////////////////////////////////////////////////////////////////////////////////////////////
//
// Defines point and triangle geometry primitives and relevant methods.
//
// Created on: 31 May 2026     Author: Daniel Owen
//
// Copyright (c) 2026, Maptek Pty Ltd. All rights reserved. Licensed under the MIT License.
//
/////////////////////////////////////////////////////////////////////////////////////////////

//! # geometry
//! This module defines geometry primitives and useful point operations.

mod point;
mod triangle;

pub(crate) use point::Point;
pub(crate) use triangle::{Triangle, Triangle3};
