#![cfg(test)]
use crate::provider::ChildrenProvider;
use crate::provider::ExpectedValueProvider;
use crate::provider::{Provider, ProviderType, WrappedChildrenProvider, WrappedProvider};
use crate::regret_node::RegretNode;
use crate::roshambo::*;
use crate::user_data::WrappedDecision;

/// Test for children provider in Rock-Paper-Scissors.
#[test]
fn test_roshambo_children_provider() {
    let provider = RoshamboChildrenProvider::new();
    let parents_data = vec![];
    let children = provider.get_children(parents_data);
    assert_eq!(children.len(), 3); // Should have three children for Rock, Paper, Scissors
}

/// Test for expected value provider in Rock-Paper-Scissors.
#[test]
fn test_roshambo_expected_value_provider() {
    let provider = RoshamboExpectedValueProvider::new();
    let parents_data = vec![
        WrappedDecision::new(RoshamboData { choice: Choice::Rock, probability: 0.333 }),
        WrappedDecision::new(RoshamboData { choice: Choice::Paper, probability: 0.333 }),
    ];
    let expected_value = provider.get_expected_value(parents_data);
    assert!((expected_value + 1.0).abs() < f64::EPSILON, "Paper beats Rock");
}

/// Test for regret minimization in Rock-Paper-Scissors.
#[test]
fn test_roshambo_regret_minimization() {
    let mut node = RegretNode::new(
        1.0,
        0.01,
        vec![],
        WrappedProvider::new(Provider::new(
            ProviderType::Children(WrappedChildrenProvider::new(Box::new(
                RoshamboChildrenProvider {},
            ))),
            None,
        )),
        Some(1.0),
    );

    node.solve(1000);
    // print node as string
    println!("{}", node.get_data_as_string(0));
    let children = node.get_children();
    assert_eq!(children.len(), 3); // Should have three children for Rock, Paper, Scissors
    for child in children {
        let expected_value = child.get_average_expected_value();
        let probability = child.get_average_probability();
        // assert expected value is close to zero
        assert!((expected_value - 0.0).abs() < 0.01, "Expected value should be close to zero");
        // assert probability is close to 1/3
        assert!((probability - 1.0 / 3.0).abs() < 0.01, "Probability should be close to 1/3");
    }
}
