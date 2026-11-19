//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AnagramAction {
    Swap(usize, usize),
    Identify(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AnagramState {
    pub letters: Vec<char>,
    /// All dictionary words that can be formed from these letters.
    pub valid_words: Vec<String>,
    pub identified: bool,
}

impl AnagramState {
    pub fn word_len(&self) -> usize {
        self.letters.len()
    }

    pub fn current_word(&self) -> String {
        self.letters.iter().collect()
    }

    pub fn is_valid_word(&self, word: &str) -> bool {
        self.valid_words.iter().any(|w| w == word)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_clone_eq() {
        let s1 = AnagramState {
            letters: vec!['h', 't', 'y', 'm'],
            valid_words: vec!["myth".to_string()],
            identified: false,
        };
        let s2 = s1.clone();
        assert_eq!(s1, s2);
    }

    #[test]
    fn state_hash_deterministic() {
        let s1 = AnagramState {
            letters: vec!['a', 'b', 'c', 'd', 'e', 'f'],
            valid_words: vec!["abcdef".to_string()],
            identified: false,
        };
        let s2 = s1.clone();
        assert_eq!(
            crate::solver::bfs::state_hash(&s1),
            crate::solver::bfs::state_hash(&s2)
        );
    }

    #[test]
    fn word_len() {
        let state = AnagramState {
            letters: vec!['a', 'b', 'c', 'd', 'e', 'f'],
            valid_words: vec!["abcdef".to_string()],
            identified: false,
        };
        assert_eq!(state.word_len(), 6);
    }

    #[test]
    fn is_valid_word_checks_list() {
        let state = AnagramState {
            letters: vec!['l', 'i', 's', 't', 'e', 'n'],
            valid_words: vec![
                "listen".to_string(),
                "silent".to_string(),
                "tinsel".to_string(),
            ],
            identified: false,
        };
        assert!(state.is_valid_word("listen"));
        assert!(state.is_valid_word("silent"));
        assert!(!state.is_valid_word("enlist")); // not in this list
    }

    #[test]
    fn current_word_from_letters() {
        let state = AnagramState {
            letters: vec!['m', 'y', 't', 'h', 'i', 'c'],
            valid_words: vec!["mythic".to_string()],
            identified: false,
        };
        assert_eq!(state.current_word(), "mythic");
    }
}
