//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
//! Compact JSON state descriptions for State CoT.
//!
//! Each game describes its state as an ordered list of `(key, value)` fields
//! whose values are already JSON-encoded (see `Game::state_fields`). The
//! episode appends a `moves` field and joins everything into one compact JSON
//! object without whitespace, e.g. `{"g":"#@.O#","moves":3}`.

/// One JSON field: key and JSON-encoded value.
pub type StateField = (&'static str, String);

/// Join fields and the move count into a compact JSON object.
pub fn format_state_json(fields: &[StateField], moves: usize) -> String {
    let mut parts: Vec<String> = fields
        .iter()
        .map(|(key, value)| format!("\"{key}\":{value}"))
        .collect();
    parts.push(format!("\"moves\":{moves}"));
    format!("{{{}}}", parts.join(","))
}

/// JSON string literal with the required escapes.
pub fn json_string(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len() + 2);
    escaped.push('"');
    for character in text.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            c if (c as u32) < 0x20 => escaped.push_str(&format!("\\u{:04x}", c as u32)),
            c => escaped.push(c),
        }
    }
    escaped.push('"');
    escaped
}

/// JSON boolean literal.
pub fn json_bool(value: bool) -> String {
    if value { "true" } else { "false" }.to_string()
}

/// JSON array of integers (or any `Display` numbers).
pub fn json_numbers<T: std::fmt::Display>(values: impl IntoIterator<Item = T>) -> String {
    let items: Vec<String> = values.into_iter().map(|value| value.to_string()).collect();
    format!("[{}]", items.join(","))
}

/// JSON array of optional integers, with `null` for `None`.
pub fn json_optional_numbers<T: std::fmt::Display>(
    values: impl IntoIterator<Item = Option<T>>,
) -> String {
    let items: Vec<String> = values
        .into_iter()
        .map(|value| value.map_or_else(|| "null".to_string(), |v| v.to_string()))
        .collect();
    format!("[{}]", items.join(","))
}

/// JSON array of strings.
pub fn json_strings<S: AsRef<str>>(values: impl IntoIterator<Item = S>) -> String {
    let items: Vec<String> = values
        .into_iter()
        .map(|value| json_string(value.as_ref()))
        .collect();
    format!("[{}]", items.join(","))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_compact_object_with_moves_last() {
        let fields = vec![("g", json_string("#@.")), ("n", "2".to_string())];
        assert_eq!(
            format_state_json(&fields, 3),
            r##"{"g":"#@.","n":2,"moves":3}"##
        );
    }

    #[test]
    fn escapes_strings() {
        assert_eq!(json_string("a\"b\\c\n"), r#""a\"b\\c\n""#);
    }

    #[test]
    fn encodes_arrays() {
        assert_eq!(json_numbers([0, 90]), "[0,90]");
        assert_eq!(json_optional_numbers([Some(1), None]), "[1,null]");
        assert_eq!(json_strings(["P2"]), r#"["P2"]"#);
        assert_eq!(json_bool(false), "false");
    }
}
