//! # Provider Module
//!
//! This module defines traits and wrappers for children and expected value providers
//! used in regret minimization trees. It also provides types for associating providers
//! with user data and for thread-safe access.

use crate::regret_node::WrappedRegret;
use crate::user_data::{DecisionTrait, WrappedDecision};
use std::sync::{Arc, Mutex};
use utils::safer::safe_lock;

/// Trait for types that can generate child nodes given parent data.
pub trait ChildrenProvider<Decision: DecisionTrait, ChildType = WrappedRegret<Decision>>:
    std::fmt::Debug
where
    ChildType: Clone + std::fmt::Debug,
{
    /// Returns the children nodes for the given parent data.
    fn get_children(
        &self,
        parents_data: Vec<WrappedDecision<Decision>>,
    ) -> Vec<ChildType>;
}

/// Thread-safe wrapper for a boxed `ChildrenProvider`.
#[derive(Debug, Clone)]
pub struct WrappedChildrenProvider<Decision: DecisionTrait, ChildType = WrappedRegret<Decision>>
where
    ChildType: Clone + std::fmt::Debug,
{
    /// The underlying provider, boxed and wrapped in Arc<Mutex>.
    provider: Arc<Mutex<Box<dyn ChildrenProvider<Decision, ChildType>>>>,
}

impl<Decision: DecisionTrait, ChildType> WrappedChildrenProvider<Decision, ChildType>
where
    ChildType: Clone + std::fmt::Debug,
{
    /// Creates a new wrapped children provider.
    #[must_use]
    pub fn new(provider: Box<dyn ChildrenProvider<Decision, ChildType>>) -> Self {
        Self { provider: Arc::new(Mutex::new(provider)) }
    }

    /// Gets the children nodes for the given parent data.
    #[must_use]
    pub fn get_children(
        &self,
        parents_data: Vec<WrappedDecision<Decision>>,
    ) -> Vec<ChildType> {
        safe_lock(&self.provider).get_children(parents_data)
    }
}

/// Trait for types that can compute expected values given parent data.
pub trait ExpectedValueProvider<Decision: DecisionTrait>: std::fmt::Debug {
    /// Returns the expected value for the given parent data.
    fn get_expected_value(
        &self,
        parents_data: Vec<WrappedDecision<Decision>>,
    ) -> f64;
}

/// Thread-safe wrapper for a boxed `ExpectedValueProvider`.
#[derive(Debug, Clone)]
pub struct WrappedExpectedValueProvider<Decision: DecisionTrait> {
    /// The underlying provider, boxed and wrapped in Arc<Mutex>.
    provider: Arc<Mutex<Box<dyn ExpectedValueProvider<Decision>>>>,
}

impl<Decision: DecisionTrait> WrappedExpectedValueProvider<Decision> {
    /// Creates a new wrapped expected value provider.
    #[must_use]
    pub fn new(provider: Box<dyn ExpectedValueProvider<Decision>>) -> Self {
        Self { provider: Arc::new(Mutex::new(provider)) }
    }

    /// Gets the expected value for the given parent data.
    #[must_use]
    pub fn get_expected_value(
        &self,
        parents_data: Vec<WrappedDecision<Decision>>,
    ) -> f64 {
        safe_lock(&self.provider).get_expected_value(parents_data)
    }
}

/// Enum representing either a children provider, expected value provider, or none (terminal node).
#[derive(Debug, Clone)]
pub enum ProviderType<Decision: DecisionTrait, ChildType = WrappedRegret<Decision>>
where
    ChildType: Clone + std::fmt::Debug,
{
    /// Children provider variant.
    Children(WrappedChildrenProvider<Decision, ChildType>),
    /// Expected value provider variant.
    ExpectedValue(WrappedExpectedValueProvider<Decision>),
    /// None variant for terminal nodes.
    None,
}

/// Associates a provider type with optional user data.
#[derive(Debug, Clone)]
pub struct Provider<Decision: DecisionTrait, ChildType = WrappedRegret<Decision>>
where
    ChildType: Clone + std::fmt::Debug,
{
    /// The provider type (children or expected value).
    pub provider_type: ProviderType<Decision, ChildType>,
    /// Optional user data associated with the provider.
    pub user_data: Option<WrappedDecision<Decision>>,
}

impl<Decision: DecisionTrait, ChildType> Provider<Decision, ChildType>
where
    ChildType: Clone + std::fmt::Debug,
{
    /// Creates a new provider with the given type and optional user data.
    #[must_use]
    pub const fn new(
        provider_type: ProviderType<Decision, ChildType>,
        user_data: Option<WrappedDecision<Decision>>,
    ) -> Self {
        Self { provider_type, user_data }
    }
}

/// Thread-safe wrapper for a `Provider`.
#[derive(Debug, Clone)]
pub struct WrappedProvider<Decision: DecisionTrait, ChildType = WrappedRegret<Decision>>
where
    ChildType: Clone + std::fmt::Debug,
{
    /// The underlying provider, wrapped in Arc<Mutex>.
    provider: Arc<Mutex<Provider<Decision, ChildType>>>,
}

impl<Decision: DecisionTrait, ChildType> WrappedProvider<Decision, ChildType>
where
    ChildType: Clone + std::fmt::Debug,
{
    /// Creates a new wrapped provider.
    #[must_use]
    pub fn new(provider: Provider<Decision, ChildType>) -> Self {
        Self { provider: Arc::new(Mutex::new(provider)) }
    }

    /// Gets the provider type.
    #[must_use]
    pub fn get_provider_type(&self) -> ProviderType<Decision, ChildType> {
        safe_lock(&self.provider).provider_type.clone()
    }

    /// Gets the optional user data.
    #[must_use]
    pub fn get_user_data(&self) -> Option<WrappedDecision<Decision>> {
        safe_lock(&self.provider).user_data.clone()
    }
}
