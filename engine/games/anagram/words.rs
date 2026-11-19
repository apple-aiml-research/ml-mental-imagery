//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
use std::collections::HashMap;
use std::sync::OnceLock;

/// The word list: 34,011 lowercase ASCII words of 6-10 letters, one per line,
/// from Webster's 1913 Unabridged Dictionary. Not distributed: scripts/download_words.py
/// recreates it from OPTED (see SOURCE.md).
const DICTIONARY_TEXT: &str = include_str!("dictionary.txt");

struct Dictionary {
    words: Vec<&'static str>,
    /// sorted-letters → list of words sharing that letter bag
    anagram_map: HashMap<String, Vec<&'static str>>,
}

fn dictionary() -> &'static Dictionary {
    static INSTANCE: OnceLock<Dictionary> = OnceLock::new();
    INSTANCE.get_or_init(|| {
        let words: Vec<&'static str> = DICTIONARY_TEXT.lines().filter(|l| !l.is_empty()).collect();
        let mut anagram_map: HashMap<String, Vec<&'static str>> = HashMap::new();
        for &w in &words {
            let key = sorted_letters(w);
            anagram_map.entry(key).or_default().push(w);
        }
        Dictionary { words, anagram_map }
    })
}

fn sorted_letters(word: &str) -> String {
    let mut chars: Vec<char> = word.chars().collect();
    chars.sort();
    chars.into_iter().collect()
}

#[cfg(test)]
pub fn all_words() -> &'static [&'static str] {
    &dictionary().words
}

pub fn words_in_length_range(min_len: usize, max_len: usize) -> Vec<&'static str> {
    dictionary()
        .words
        .iter()
        .filter(|w| w.len() >= min_len && w.len() <= max_len)
        .copied()
        .collect()
}

/// Return all dictionary words that are anagrams of the given word
/// (same letter bag), including the word itself.
pub fn anagram_group(word: &str) -> Vec<&'static str> {
    let key = sorted_letters(word);
    dictionary()
        .anagram_map
        .get(&key)
        .cloned()
        .unwrap_or_default()
}

#[cfg(test)]
/// Return all dictionary words matching a given sorted-letter key.
pub fn anagram_group_for_letters(letters: &[char]) -> Vec<&'static str> {
    let mut sorted = letters.to_vec();
    sorted.sort();
    let key: String = sorted.into_iter().collect();
    dictionary()
        .anagram_map
        .get(&key)
        .cloned()
        .unwrap_or_default()
}

/// Check if a word exists in the dictionary.
pub fn contains(word: &str) -> bool {
    dictionary().words.binary_search(&word).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dictionary_loads() {
        let words = all_words();
        assert!(
            words.len() > 30_000,
            "expected 30K+ words, got {}",
            words.len()
        );
    }

    #[test]
    fn all_words_in_length_range() {
        for w in all_words() {
            assert!(
                (6..=10).contains(&w.len()),
                "word '{}' has length {} (expected 6-10)",
                w,
                w.len()
            );
        }
    }

    #[test]
    fn all_words_lowercase_ascii() {
        for w in all_words() {
            assert!(
                w.chars().all(|c| c.is_ascii_lowercase()),
                "word '{}' contains non-lowercase-ascii",
                w
            );
        }
    }

    #[test]
    fn anagram_group_finds_partners() {
        let group = anagram_group("listen");
        assert!(
            group.len() >= 2,
            "expected 'listen' to have anagram partners, got {:?}",
            group
        );
    }

    #[test]
    fn anagram_group_for_single_word() {
        let letters: Vec<char> = "rhythm".chars().collect();
        let group = anagram_group_for_letters(&letters);
        assert!(
            group.contains(&"rhythm"),
            "expected group to contain 'rhythm', got {:?}",
            group
        );
    }

    #[test]
    fn contains_known_word() {
        assert!(contains("church"));
        assert!(!contains("zzzzzzz"));
    }

    #[test]
    fn words_in_length_range_filters() {
        let words = words_in_length_range(8, 10);
        for w in &words {
            assert!((8..=10).contains(&w.len()));
        }
        assert!(words.len() > 1000);
    }
}
