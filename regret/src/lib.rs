//! Crate-level documentation for the regret library.
//!
//! This crate provides modules for regret minimization algorithms and related utilities.

#![warn(clippy::all)]
#![warn(clippy::style)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![warn(clippy::cargo)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::multiple_crate_versions)]
#![warn(
    missing_docs,
    clippy::missing_docs_in_private_items,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc
)]
pub mod percentage_tree;
pub mod provider;
pub mod regret_node;
pub mod roshambo;
mod tests;
pub mod user_data;
