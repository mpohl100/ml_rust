//! # Decision Module
//!
//! This module defines the trait for decision data used in regret nodes,
//! and provides a thread-safe wrapper for such data. Implement `DecisionTrait`
//! for your domain-specific data to use with the regret minimization framework.

use std::sync::{Arc, Mutex};
use utils::safer::safe_lock;

/// Trait for decision data in regret nodes, requiring probability management and string representation.
pub trait DecisionTrait: Default + Clone + std::fmt::Debug {
    /// Returns the probability associated with this decision data.
    fn get_probability(&self) -> f64;
    /// Sets the probability associated with this decision data.
    fn set_probability(
        &mut self,
        probability: f64,
    );
    /// Returns a string representation of the decision data.
    fn get_data_as_string(&self) -> String;
}

/// Thread-safe wrapper around decision data implementing `DecisionTrait`.
#[derive(Debug, Clone)]
pub struct WrappedDecision<Decision: DecisionTrait> {
    /// Thread-safe decision data implementing `DecisionTrait`.
    decision_data: Arc<Mutex<Decision>>,
}

impl<Decision: DecisionTrait> WrappedDecision<Decision> {
    /// Creates a new `WrappedDecision` containing the given decision data.
    #[must_use]
    pub fn new(decision_data: Decision) -> Self {
        Self { decision_data: Arc::new(Mutex::new(decision_data)) }
    }

    /// Returns a clone of the inner decision data.
    #[must_use]
    pub fn get_decision_data(&self) -> Decision {
        safe_lock(&self.decision_data).clone()
    }

    /// Sets the probability of the inner decision data.
    pub fn set_probability(
        &self,
        probability: f64,
    ) {
        let mut data = safe_lock(&self.decision_data);
        data.set_probability(probability);
    }

    /// Gets the probability of the inner decision data.
    #[must_use]
    pub fn get_probability(&self) -> f64 {
        safe_lock(&self.decision_data).get_probability()
    }

    /// Gets the string representation of the inner decision data.
    #[must_use]
    pub fn get_data_as_string(&self) -> String {
        safe_lock(&self.decision_data).get_data_as_string()
    }

    /// Returns a clone of the inner user data - backward compatibility method.
    #[must_use]
    pub fn get_user_data(&self) -> Decision {
        self.get_decision_data()
    }
}
