//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
/// Action template system for parametric action strings.
///
/// Templates describe the structure of action strings with fixed and variable
/// segments. `expand()` produces all concrete action strings via cartesian
/// product.

/// A single segment of an action template.
#[derive(Debug, Clone)]
pub enum ParamType {
    /// Fixed text: "place('"
    Literal(String),
    /// Finite set of options: piece IDs, directions
    OneOf(Vec<String>),
    /// Bounded integer range (inclusive): coordinates
    Int { min: i32, max: i32 },
}

/// A template describing a family of action strings.
///
/// Each segment is a `ParamType`; `expand()` computes the cartesian product.
#[derive(Debug, Clone)]
pub struct ActionTemplate {
    pub segments: Vec<ParamType>,
}

impl ActionTemplate {
    /// Expand template into all concrete action strings.
    /// Cartesian product of all segment values.
    pub fn expand(&self) -> Vec<String> {
        let mut results = vec![String::new()];
        for segment in &self.segments {
            let values: Vec<String> = match segment {
                ParamType::Literal(s) => vec![s.clone()],
                ParamType::OneOf(options) => options.clone(),
                ParamType::Int { min, max } => (*min..=*max).map(|i| i.to_string()).collect(),
            };
            results = results
                .into_iter()
                .flat_map(|prefix| values.iter().map(move |v| format!("{prefix}{v}")))
                .collect();
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sokoban_literals() {
        // Sokoban actions are simple literals: "up", "down", "left", "right"
        for dir in &["up", "down", "left", "right"] {
            let t = ActionTemplate {
                segments: vec![ParamType::Literal(dir.to_string())],
            };
            assert_eq!(t.expand(), vec![dir.to_string()]);
        }
    }

    #[test]
    fn tangram_compound_template() {
        // place('P0', 1, 0) style — compound template with OneOf + Int segments
        let t = ActionTemplate {
            segments: vec![
                ParamType::Literal("place('".into()),
                ParamType::OneOf(vec!["P0".into(), "P1".into()]),
                ParamType::Literal("', ".into()),
                ParamType::Int { min: 0, max: 1 },
                ParamType::Literal(", ".into()),
                ParamType::Int { min: 0, max: 1 },
                ParamType::Literal(")".into()),
            ],
        };
        let expanded = t.expand();
        // 1 * 2 * 1 * 2 * 1 * 2 * 1 = 8 actions
        assert_eq!(expanded.len(), 8);
        assert!(expanded.contains(&"place('P0', 0, 0)".to_string()));
        assert!(expanded.contains(&"place('P0', 0, 1)".to_string()));
        assert!(expanded.contains(&"place('P0', 1, 0)".to_string()));
        assert!(expanded.contains(&"place('P0', 1, 1)".to_string()));
        assert!(expanded.contains(&"place('P1', 0, 0)".to_string()));
        assert!(expanded.contains(&"place('P1', 0, 1)".to_string()));
        assert!(expanded.contains(&"place('P1', 1, 0)".to_string()));
        assert!(expanded.contains(&"place('P1', 1, 1)".to_string()));
    }

    #[test]
    fn oneof_only() {
        let t = ActionTemplate {
            segments: vec![ParamType::OneOf(vec![
                "alpha".into(),
                "beta".into(),
                "gamma".into(),
            ])],
        };
        assert_eq!(t.expand(), vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn single_literal() {
        let t = ActionTemplate {
            segments: vec![ParamType::Literal("done".into())],
        };
        assert_eq!(t.expand(), vec!["done"]);
    }

    #[test]
    fn empty_template() {
        let t = ActionTemplate { segments: vec![] };
        let expanded = t.expand();
        assert_eq!(expanded, vec![""]);
    }

    #[test]
    fn int_range_with_negatives() {
        let t = ActionTemplate {
            segments: vec![
                ParamType::Literal("slide('A', ".into()),
                ParamType::Int { min: -2, max: 2 },
                ParamType::Literal(")".into()),
            ],
        };
        let expanded = t.expand();
        assert_eq!(expanded.len(), 5);
        assert_eq!(expanded[0], "slide('A', -2)");
        assert_eq!(expanded[1], "slide('A', -1)");
        assert_eq!(expanded[2], "slide('A', 0)");
        assert_eq!(expanded[3], "slide('A', 1)");
        assert_eq!(expanded[4], "slide('A', 2)");
    }

    #[test]
    fn large_cartesian_product() {
        // 3 OneOf segments with 3, 4, 5 options = 60 combinations
        let t = ActionTemplate {
            segments: vec![
                ParamType::OneOf(vec!["a".into(), "b".into(), "c".into()]),
                ParamType::Literal("-".into()),
                ParamType::OneOf(vec!["1".into(), "2".into(), "3".into(), "4".into()]),
                ParamType::Literal("-".into()),
                ParamType::OneOf(vec![
                    "x".into(),
                    "y".into(),
                    "z".into(),
                    "w".into(),
                    "v".into(),
                ]),
            ],
        };
        let expanded = t.expand();
        assert_eq!(expanded.len(), 60);
        assert!(expanded.contains(&"a-1-x".to_string()));
        assert!(expanded.contains(&"c-4-v".to_string()));
        // All entries are unique
        let mut sorted = expanded.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 60);
    }

    #[test]
    fn combined_templates_with_meta_actions() {
        // Simulate enumerate_actions: expand templates + append meta-actions
        let place = ActionTemplate {
            segments: vec![
                ParamType::Literal("place('".into()),
                ParamType::OneOf(vec!["P0".into(), "P1".into()]),
                ParamType::Literal("', ".into()),
                ParamType::Int { min: 0, max: 2 },
                ParamType::Literal(", ".into()),
                ParamType::Int { min: 0, max: 1 },
                ParamType::Literal(")".into()),
            ],
        };
        let rotate = ActionTemplate {
            segments: vec![
                ParamType::Literal("rotatecw('".into()),
                ParamType::OneOf(vec!["P0".into(), "P1".into()]),
                ParamType::Literal("')".into()),
            ],
        };

        let templates = vec![place, rotate];
        let mut actions: Vec<String> = templates.iter().flat_map(|t| t.expand()).collect();
        actions.extend(["back".into(), "reset".into(), "done".into()]);

        // place: 2 pieces * 3 x-coords * 2 y-coords = 12
        // rotate: 2 pieces = 2
        // meta: 3
        assert_eq!(actions.len(), 17);
        assert!(actions.contains(&"place('P0', 0, 0)".to_string()));
        assert!(actions.contains(&"place('P1', 2, 1)".to_string()));
        assert!(actions.contains(&"rotatecw('P0')".to_string()));
        assert!(actions.contains(&"rotatecw('P1')".to_string()));
        assert!(actions.contains(&"back".to_string()));
        assert!(actions.contains(&"reset".to_string()));
        assert!(actions.contains(&"done".to_string()));
    }
}
