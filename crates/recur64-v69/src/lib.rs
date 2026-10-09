//! Recur64 V69 fresh-data pipeline.
//!
//! Software only: no scientific asset from any earlier experiment is read here.
//! All inputs and outputs are confined to the V69 artifact namespace
//! (see [`custody`]) and, inside it, to role-restricted access ([`access`]).

pub mod access;
pub mod audit2;
pub mod canon;
pub mod custody;
pub mod d1;
pub mod d1_metrics;
pub mod d2;
pub mod dataset;
pub mod features;
pub mod g1;
pub mod generate;
pub mod metrics;
pub mod oracle;
pub mod provenance;
pub mod reference;
pub mod streams;
pub mod t1;
