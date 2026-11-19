//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// Game auto-registration registry.
//
// Each game self-registers via `inventory::submit!(GameEntry { ... })` in its
// own module. The Python bindings discover games via `all_games()`.

use std::collections::HashMap;

use crate::stream::loader::GameFactory;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigParamType {
    Int,
    IntRange,
}

#[derive(Debug, Clone)]
pub enum ConfigValue {
    Int(i64),
    IntRange(i64, i64),
}

pub struct ConfigParam {
    pub key: &'static str,
    pub param_type: ConfigParamType,
    pub default: ConfigValue,
}

pub type BuildFactoryFn =
    fn(&str, &[ConfigParam], &HashMap<String, ConfigValue>) -> Result<Box<dyn GameFactory>, String>;

pub struct GameEntry {
    pub name: &'static str,
    pub params: &'static [ConfigParam],
    pub build: BuildFactoryFn,
}

inventory::collect!(GameEntry);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

pub fn all_games() -> Vec<&'static GameEntry> {
    let mut entries: Vec<&'static GameEntry> = inventory::iter::<GameEntry>.into_iter().collect();
    entries.sort_by_key(|e| e.name);
    entries
}

pub fn find_game(name: &str) -> Option<&'static GameEntry> {
    inventory::iter::<GameEntry>
        .into_iter()
        .find(|e| e.name == name)
}

fn resolve_raw<'a>(
    params: &'a [ConfigParam],
    overrides: &'a HashMap<String, ConfigValue>,
    key: &str,
) -> Result<&'a ConfigValue, String> {
    if let Some(val) = overrides.get(key) {
        return Ok(val);
    }
    for p in params {
        if p.key == key {
            return Ok(&p.default);
        }
    }
    Err(format!("unknown param '{key}'"))
}

pub fn resolve_int(
    params: &[ConfigParam],
    overrides: &HashMap<String, ConfigValue>,
    key: &str,
) -> Result<i64, String> {
    match resolve_raw(params, overrides, key)? {
        ConfigValue::Int(v) => Ok(*v),
        _ => Err(format!("expected int for '{key}'")),
    }
}

pub fn resolve_int_range(
    params: &[ConfigParam],
    overrides: &HashMap<String, ConfigValue>,
    key: &str,
) -> Result<(i64, i64), String> {
    match resolve_raw(params, overrides, key)? {
        ConfigValue::IntRange(lo, hi) => Ok((*lo, *hi)),
        _ => Err(format!("expected int_range for '{key}'")),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_game_nonexistent() {
        assert!(find_game("nonexistent").is_none());
    }

    #[test]
    fn registry_not_empty() {
        assert!(!all_games().is_empty());
    }

    #[test]
    fn all_registry_entries_buildable() {
        let overrides = HashMap::new();
        for entry in all_games() {
            let result = (entry.build)(entry.name, entry.params, &overrides);
            assert!(
                result.is_ok(),
                "failed to build game '{}': {:?}",
                entry.name,
                result.err()
            );
        }
    }

    #[test]
    fn resolve_int_override() {
        let params = &[ConfigParam {
            key: "size",
            param_type: ConfigParamType::Int,
            default: ConfigValue::Int(6),
        }];
        let mut overrides = HashMap::new();
        overrides.insert("size".to_string(), ConfigValue::Int(10));
        assert_eq!(resolve_int(params, &overrides, "size").unwrap(), 10);
    }

    #[test]
    fn resolve_int_default() {
        let params = &[ConfigParam {
            key: "size",
            param_type: ConfigParamType::Int,
            default: ConfigValue::Int(6),
        }];
        let overrides = HashMap::new();
        assert_eq!(resolve_int(params, &overrides, "size").unwrap(), 6);
    }

    #[test]
    fn resolve_range_override() {
        let params = &[ConfigParam {
            key: "range",
            param_type: ConfigParamType::IntRange,
            default: ConfigValue::IntRange(1, 2),
        }];
        let mut overrides = HashMap::new();
        overrides.insert("range".to_string(), ConfigValue::IntRange(3, 5));
        assert_eq!(
            resolve_int_range(params, &overrides, "range").unwrap(),
            (3, 5)
        );
    }
}
