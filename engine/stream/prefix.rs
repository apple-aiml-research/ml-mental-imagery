//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// ActionTarget and prefix-matching primitives.
//
// Bridges action strings (from `enumerate_actions()`) to token IDs (for the model).
// EpisodeStream (episode_stream.rs) consumes these functions.

use tokenizers::Tokenizer;

pub type TokenId = u32;

/// A single valid action string, pre-tokenized for fast prefix matching.
#[derive(Debug, Clone)]
pub struct ActionTarget {
    pub action_str: String,
    pub tokens: Vec<TokenId>,
    /// Cumulative byte offset after each token.
    /// `offsets[i]` is the byte position in `action_str` right after token `i`.
    pub offsets: Vec<usize>,
}

impl ActionTarget {
    /// Create from an action string and a tokenizer.
    ///
    /// Encodes the string, extracts token IDs and cumulative byte offsets.
    /// Falls back to decode-and-accumulate if offsets don't match string length.
    pub fn new(action_str: &str, tokenizer: &Tokenizer) -> Self {
        let encoding = tokenizer
            .encode(action_str, false)
            .expect("tokenizer.encode failed");

        let tokens = encoding.get_ids().to_vec();

        // Compute cumulative byte offsets from encoding offsets (end of each token).
        let raw_offsets = encoding.get_offsets();
        let mut offsets: Vec<usize> = raw_offsets.iter().map(|&(_, end)| end).collect();

        // Validate: last offset should equal action_str byte length.
        if offsets.last().copied() != Some(action_str.len()) {
            // Fallback: decode each token individually and accumulate byte lengths.
            offsets.clear();
            let mut pos = 0usize;
            for &id in &tokens {
                let decoded = tokenizer
                    .decode(&[id], false)
                    .expect("tokenizer.decode failed");
                pos += decoded.len();
                offsets.push(pos);
            }
        }

        let result = ActionTarget {
            action_str: action_str.to_string(),
            tokens,
            offsets,
        };

        result
    }

    /// Return the canonical token at a given byte boundary.
    ///
    /// Returns `Some(token_id)` if `prefix_len` falls exactly on a token boundary
    /// (i.e., it equals the start byte of some token). Returns `None` if `prefix_len`
    /// falls mid-token.
    pub fn canonical_token_at(&self, prefix_len: usize) -> Option<TokenId> {
        if self.tokens.is_empty() {
            return None;
        }
        if prefix_len == 0 {
            return Some(self.tokens[0]);
        }
        // Token i (for i > 0) starts at offsets[i-1].
        for i in 1..self.tokens.len() {
            if self.offsets[i - 1] == prefix_len {
                return Some(self.tokens[i]);
            }
        }
        None
    }
}

/// Build sorted `ActionTarget` list from action strings.
pub fn build_valid_targets(action_strs: &[String], tokenizer: &Tokenizer) -> Vec<ActionTarget> {
    let mut targets: Vec<ActionTarget> = action_strs
        .iter()
        .map(|s| ActionTarget::new(s, tokenizer))
        .collect();
    targets.sort_by(|a, b| a.action_str.cmp(&b.action_str));

    targets
}

/// O(log N) exact-match check against sorted targets.
pub fn is_complete_action(prefix: &str, valid_targets: &[ActionTarget]) -> bool {
    valid_targets
        .binary_search_by(|t| t.action_str.as_str().cmp(prefix))
        .is_ok()
}

/// Return the contiguous slice of targets whose `action_str` starts with `prefix`.
///
/// Uses two `partition_point` calls on the sorted target list.
pub fn matching_targets<'a>(prefix: &str, valid_targets: &'a [ActionTarget]) -> &'a [ActionTarget] {
    let start = valid_targets.partition_point(|t| t.action_str.as_str() < prefix);
    let end = start + valid_targets[start..].partition_point(|t| t.action_str.starts_with(prefix));

    &valid_targets[start..end]
}

/// Return the next token to emit for `target` given that `prefix_len` bytes have
/// already been generated.
///
/// Uses the canonical tokenization when `prefix_len` is on a token boundary;
/// otherwise falls back to encoding the remainder.
pub fn next_token_of(target: &ActionTarget, prefix_len: usize, tokenizer: &Tokenizer) -> TokenId {
    if let Some(tok) = target.canonical_token_at(prefix_len) {
        return tok;
    }
    // Non-canonical: encode remainder and return its first token.

    let remainder = &target.action_str[prefix_len..];

    let encoding = tokenizer
        .encode(remainder, false)
        .expect("tokenizer.encode failed");

    let ids = encoding.get_ids();
    let result = if ids.is_empty() {
        // Remainder was unencodable (e.g. single char not in vocab).
        // Fall back to the canonical token whose byte range covers prefix_len.
        eprintln!(
            "next_token_of: empty encoding for remainder {:?} of {:?} at prefix_len={}",
            remainder, target.action_str, prefix_len
        );
        let mut fallback = *target.tokens.last().expect("target has no tokens");
        for i in 0..target.tokens.len() {
            if prefix_len < target.offsets[i] {
                fallback = target.tokens[i];
                break;
            }
        }
        fallback
    } else {
        ids[0]
    };

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::models::bpe::BPE;

    /// Build a deterministic BPE tokenizer with character-level vocab + known merges.
    ///
    /// Character vocab: a-z (0-25), 0-9 (26-35), punctuation (36-40), P (41)
    /// Merges: u+p→up(42), d+o→do(43), w+n→wn(44), n+e→ne(45),
    ///         do+wn→down(46), do+ne→done(47)
    fn test_tokenizer() -> Tokenizer {
        let mut vocab: ahash::AHashMap<String, u32> = ahash::AHashMap::new();
        let mut id = 0u32;

        for c in 'a'..='z' {
            vocab.insert(c.to_string(), id);
            id += 1;
        }
        // IDs 0-25: lowercase letters

        for c in '0'..='9' {
            vocab.insert(c.to_string(), id);
            id += 1;
        }
        // IDs 26-35: digits

        for c in ['(', ')', '\'', ',', ' ', '+', '-'] {
            vocab.insert(c.to_string(), id);
            id += 1;
        }
        // IDs 36-42: punctuation

        vocab.insert("P".to_string(), id);
        id += 1;
        // ID 43: uppercase P

        // Merged tokens
        vocab.insert("up".to_string(), id);
        id += 1; // 44
        vocab.insert("do".to_string(), id);
        id += 1; // 45
        vocab.insert("wn".to_string(), id);
        id += 1; // 46
        vocab.insert("ne".to_string(), id);
        id += 1; // 47
        vocab.insert("down".to_string(), id);
        id += 1; // 48
        vocab.insert("done".to_string(), id);
        id += 1; // 49
        vocab.insert("x+".to_string(), id);
        let _ = id + 1; // 50

        let merges = vec![
            ("u".into(), "p".into()),   // → up (44)
            ("d".into(), "o".into()),   // → do (45)
            ("w".into(), "n".into()),   // → wn (46)
            ("n".into(), "e".into()),   // → ne (47)
            ("do".into(), "wn".into()), // → down (48)
            ("do".into(), "ne".into()), // → done (49)
            ("x".into(), "+".into()),   // → x+ (50)
        ];

        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .build()
            .unwrap();

        Tokenizer::new(bpe)
    }

    // ---- Test 1: ActionTarget::new roundtrip ----
    #[test]
    fn new_roundtrip() {
        let tok = test_tokenizer();

        for action in &["up", "down", "left", "right", "done"] {
            let target = ActionTarget::new(action, &tok);
            // Last offset equals string length.
            assert_eq!(
                *target.offsets.last().unwrap(),
                action.len(),
                "last offset mismatch for {:?}",
                action
            );
            // Token strings concatenate back to the action string.
            let token_strs: Vec<String> = target
                .tokens
                .iter()
                .map(|&id| tok.id_to_token(id).unwrap())
                .collect();
            assert_eq!(
                token_strs.concat(),
                *action,
                "token concat mismatch for {:?}",
                action
            );
        }
    }

    // ---- Test 2: ActionTarget::new for compound tangram action ----
    #[test]
    fn new_compound_tangram() {
        let tok = test_tokenizer();
        let action = "place('P0', 0, 0)";
        let target = ActionTarget::new(action, &tok);

        // Should be character-level (no merges apply).
        assert_eq!(target.tokens.len(), action.len());
        assert_eq!(*target.offsets.last().unwrap(), action.len());

        // Each offset advances by 1.
        for (i, &off) in target.offsets.iter().enumerate() {
            assert_eq!(off, i + 1);
        }
    }

    // ---- Test 3: canonical_token_at ----
    #[test]
    fn canonical_token_at_boundaries() {
        let tok = test_tokenizer();

        // "left" → [l, e, f, t] — all character-level.
        let target = ActionTarget::new("left", &tok);
        assert_eq!(target.tokens.len(), 4);

        // Each byte boundary is a token boundary.
        assert_eq!(target.canonical_token_at(0), Some(target.tokens[0])); // l
        assert_eq!(target.canonical_token_at(1), Some(target.tokens[1])); // e
        assert_eq!(target.canonical_token_at(2), Some(target.tokens[2])); // f
        assert_eq!(target.canonical_token_at(3), Some(target.tokens[3])); // t

        // Past the end: no token.
        assert_eq!(target.canonical_token_at(4), None);

        // "down" → [down] — single merged token.
        let target = ActionTarget::new("down", &tok);
        assert_eq!(target.tokens.len(), 1);
        assert_eq!(target.canonical_token_at(0), Some(target.tokens[0])); // down
                                                                          // Mid-token positions return None.
        assert_eq!(target.canonical_token_at(1), None);
        assert_eq!(target.canonical_token_at(2), None);
        assert_eq!(target.canonical_token_at(3), None);
    }

    // ---- Test 4: build_valid_targets is sorted ----
    #[test]
    fn build_valid_targets_sorted() {
        let tok = test_tokenizer();
        let actions: Vec<String> = vec!["down", "up", "left", "right", "done"]
            .into_iter()
            .map(String::from)
            .collect();
        let targets = build_valid_targets(&actions, &tok);

        let strs: Vec<&str> = targets.iter().map(|t| t.action_str.as_str()).collect();
        let mut sorted = strs.clone();
        sorted.sort();
        assert_eq!(strs, sorted);
    }

    // ---- Test 5: is_complete_action ----
    #[test]
    fn is_complete_action_exact_only() {
        let tok = test_tokenizer();
        let actions: Vec<String> = vec!["done", "down", "left", "right", "up"]
            .into_iter()
            .map(String::from)
            .collect();
        let targets = build_valid_targets(&actions, &tok);

        // Exact matches return true.
        assert!(is_complete_action("up", &targets));
        assert!(is_complete_action("down", &targets));
        assert!(is_complete_action("left", &targets));
        assert!(is_complete_action("right", &targets));
        assert!(is_complete_action("done", &targets));

        // Partial prefixes and garbage return false.
        assert!(!is_complete_action("do", &targets));
        assert!(!is_complete_action("d", &targets));
        assert!(!is_complete_action("", &targets));
        assert!(!is_complete_action("elephant", &targets));
        assert!(!is_complete_action("downward", &targets));
    }

    // ---- Test 6: matching_targets ----
    #[test]
    fn matching_targets_prefix() {
        let tok = test_tokenizer();
        let actions: Vec<String> = vec![
            "done",
            "down",
            "left",
            "place('P0', 0, 0)",
            "place('P0', 0, 1)",
            "place('P1', 0, 0)",
            "right",
            "up",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        let targets = build_valid_targets(&actions, &tok);

        // "d" matches "done" and "down".
        let m = matching_targets("d", &targets);
        let strs: Vec<&str> = m.iter().map(|t| t.action_str.as_str()).collect();
        assert_eq!(strs, vec!["done", "down"]);

        // "place" matches all place actions.
        let m = matching_targets("place", &targets);
        assert_eq!(m.len(), 3);

        // "" matches all.
        let m = matching_targets("", &targets);
        assert_eq!(m.len(), actions.len());

        // "elephant" matches none.
        let m = matching_targets("elephant", &targets);
        assert!(m.is_empty());

        // "do" matches "done" and "down".
        let m = matching_targets("do", &targets);
        let strs: Vec<&str> = m.iter().map(|t| t.action_str.as_str()).collect();
        assert_eq!(strs, vec!["done", "down"]);
    }

    // ---- Test 7: next_token_of canonical path ----
    #[test]
    fn next_token_of_canonical() {
        let tok = test_tokenizer();
        let target = ActionTarget::new("left", &tok);

        // Every byte boundary is canonical for "left" (char-level tokens).
        for i in 0..target.tokens.len() {
            let prefix_len = if i == 0 { 0 } else { target.offsets[i - 1] };
            let got = next_token_of(&target, prefix_len, &tok);
            assert_eq!(got, target.tokens[i], "mismatch at token index {}", i);
        }
    }

    // ---- Test 8: next_token_of non-canonical fallback ----
    #[test]
    fn next_token_of_fallback() {
        let tok = test_tokenizer();

        // "down" is a single merged token. Positions 1, 2, 3 are mid-token.
        let target = ActionTarget::new("down", &tok);
        assert_eq!(target.tokens.len(), 1);

        // At prefix_len=1, canonical_token_at returns None → fallback.
        // Encodes "own" → [o, wn] → first token is 'o'.
        let t1 = next_token_of(&target, 1, &tok);
        let decoded = tok.decode(&[t1], false).unwrap();
        assert_eq!(decoded, "o", "fallback at prefix_len=1");

        // At prefix_len=2, encodes "wn" → [wn] → merged token.
        let t2 = next_token_of(&target, 2, &tok);
        let decoded = tok.decode(&[t2], false).unwrap();
        assert_eq!(decoded, "wn", "fallback at prefix_len=2");

        // At prefix_len=3, encodes "n" → [n].
        let t3 = next_token_of(&target, 3, &tok);
        let decoded = tok.decode(&[t3], false).unwrap();
        assert_eq!(decoded, "n", "fallback at prefix_len=3");
    }

    // ---- Test 9: Sokoban-style actions ----
    #[test]
    fn sokoban_actions() {
        let tok = test_tokenizer();
        let actions: Vec<String> = vec!["up", "down", "left", "right"]
            .into_iter()
            .map(String::from)
            .collect();
        let targets = build_valid_targets(&actions, &tok);

        // is_complete
        assert!(is_complete_action("up", &targets));
        assert!(is_complete_action("down", &targets));
        assert!(!is_complete_action("u", &targets));

        // matching_targets
        let m = matching_targets("l", &targets);
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].action_str, "left");

        let m = matching_targets("r", &targets);
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].action_str, "right");

        // next_token_of for "up" (single merged token)
        let up_target = targets.iter().find(|t| t.action_str == "up").unwrap();
        let t = next_token_of(up_target, 0, &tok);
        assert_eq!(t, up_target.tokens[0]);
    }

    // ---- Test 10: Tangram-style compound actions ----
    #[test]
    fn tangram_prefix_narrowing() {
        let tok = test_tokenizer();
        let actions: Vec<String> = vec![
            "place('P0', 0, 0)",
            "place('P0', 0, 1)",
            "place('P0', 1, 0)",
            "place('P0', 1, 1)",
            "place('P1', 0, 0)",
            "place('P1', 0, 1)",
            "place('P1', 1, 0)",
            "place('P1', 1, 1)",
            "done",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        let targets = build_valid_targets(&actions, &tok);

        // "place('P0" narrows to 4 matches.
        let m = matching_targets("place('P0", &targets);
        assert_eq!(m.len(), 4);
        for t in m {
            assert!(t.action_str.starts_with("place('P0"));
        }

        // "place('P1', 1" narrows to 2.
        let m = matching_targets("place('P1', 1", &targets);
        assert_eq!(m.len(), 2);

        // Full action is exact match.
        assert!(is_complete_action("place('P0', 0, 0)", &targets));
        assert!(!is_complete_action("place('P0', 0, ", &targets));
    }

    // ---- Test 11: Full action walkthrough ----
    #[test]
    fn full_walkthrough() {
        let tok = test_tokenizer();
        let target = ActionTarget::new("left", &tok);
        // "left" → [l, e, f, t], each 1 byte.

        let mut prefix = String::new();
        let mut prefix_len = 0usize;

        for i in 0..target.tokens.len() {
            // Get next token.
            let token_id = next_token_of(&target, prefix_len, &tok);
            assert_eq!(
                token_id, target.tokens[i],
                "walkthrough: wrong token at step {}",
                i
            );

            // Decode and append.
            let piece = tok.decode(&[token_id], false).unwrap();
            prefix.push_str(&piece);
            prefix_len += piece.len();
        }

        assert_eq!(prefix, "left");
        assert_eq!(prefix_len, target.action_str.len());

        // Also walk through "down" (single merged token).
        let target = ActionTarget::new("down", &tok);
        let token_id = next_token_of(&target, 0, &tok);
        let piece = tok.decode(&[token_id], false).unwrap();
        assert_eq!(piece, "down");
    }

    // ---- Test 12: next_token_of mid-token fallback when remainder is unencodable ----
    #[test]
    fn next_token_of_midtoken_fallback() {
        let tok = test_tokenizer();

        // "x+" merges into a single token. prefix_len=1 ("x" generated) means
        // remainder is "+", which re-encodes fine since '+' is in vocab.
        let target = ActionTarget::new("x+", &tok);
        assert_eq!(target.tokens.len(), 1, "x+ should be a single merged token");
        let t = next_token_of(&target, 1, &tok);
        let decoded = tok.decode(&[t], false).unwrap();
        assert_eq!(
            decoded, "+",
            "should return '+' for remainder at prefix_len=1"
        );

        // Simulate the scenario where remainder encoding returns empty by testing
        // with prefix_len == action_str.len() (empty remainder).
        // The fallback should return the last canonical token.
        let target = ActionTarget::new("down", &tok);
        let t = next_token_of(&target, 4, &tok);
        assert_eq!(
            t, target.tokens[0],
            "empty remainder should fall back to last canonical token"
        );
    }
}
