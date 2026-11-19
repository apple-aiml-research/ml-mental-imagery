//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
pub mod gen;
pub mod object_boxes;
pub mod render;
pub mod state;
pub mod words;

use rand_chacha::ChaChaRng;

use crate::base::game::{Game, RgbBuffer};
use crate::base::state_text::{json_bool, json_string, StateField};
use crate::base::template::{ActionTemplate, ParamType};

use gen::{build_permutation, closest_valid_word};
use state::{AnagramAction, AnagramState};

pub struct Anagram {
    pub word_len_range: (usize, usize),
    pub min_swaps: usize,
}

#[cfg(test)]
impl Default for Anagram {
    fn default() -> Self {
        Anagram {
            word_len_range: (6, 10),
            min_swaps: 2,
        }
    }
}

const SOLVE_BUDGET: usize = 50_000;

pub fn build_factory(
    name: &str,
    params: &[super::registry::ConfigParam],
    overrides: &std::collections::HashMap<String, super::registry::ConfigValue>,
) -> Result<Box<dyn crate::stream::loader::GameFactory>, String> {
    let (lo, hi) = super::registry::resolve_int_range(params, overrides, "word_len_range")?;
    let min_swaps = super::registry::resolve_int(params, overrides, "min_swaps")? as usize;
    let game = Anagram {
        word_len_range: (lo as usize, hi as usize),
        min_swaps,
    };
    Ok(crate::stream::loader::game_factory(
        game,
        name,
        SOLVE_BUDGET,
    ))
}

inventory::submit! {
    super::registry::GameEntry {
        name: "anagram",
        params: &[
            super::registry::ConfigParam {
                key: "word_len_range",
                param_type: super::registry::ConfigParamType::IntRange,
                default: super::registry::ConfigValue::IntRange(6, 10),
            },
            super::registry::ConfigParam {
                key: "min_swaps",
                param_type: super::registry::ConfigParamType::Int,
                default: super::registry::ConfigValue::Int(2),
            },
        ],
        build: build_factory,
    }
}

impl Game for Anagram {
    type State = AnagramState;
    type Action = AnagramAction;

    fn generate(&self, rng: &mut ChaChaRng) -> Option<AnagramState> {
        gen::generate(rng, self.word_len_range, self.min_swaps)
    }

    /// Raw, color-inclusive puzzle fingerprint.
    /// Fields hashed in order: word_len, raw ordered `letters` (the scramble
    /// order is the identity, so not sorted), sorted `valid_words`
    /// (length-prefixed), and the render tile-color palette index (derived from
    /// sorted letters). `identified` is false at generation and excluded.
    fn puzzle_fingerprint(&self, state: &AnagramState) -> u64 {
        use std::hash::Hasher;
        crate::base::fingerprint::fingerprint(|h| {
            // 1. word_len (letters length / rendered tile count).
            h.write_u64(state.letters.len() as u64);

            // 2. Raw ordered letters, each as its u32 scalar (lowercase ASCII).
            for &letter in &state.letters {
                h.write_u32(letter as u32);
            }

            // 3. valid_words, sorted so the fingerprint does not depend on the
            //    dictionary file order. The count and each word are
            //    length-prefixed to avoid boundary ambiguity.
            let mut sorted_words = state.valid_words.clone();
            sorted_words.sort();
            h.write_u64(sorted_words.len() as u64);
            for word in &sorted_words {
                h.write_u64(word.len() as u64);
                h.write(word.as_bytes());
            }

            // 4. tile_color_index: the palette index (0..8) that render.rs
            //    `tile_color` derives from the sorted letters.
            let mut sorted_letters = state.letters.clone();
            sorted_letters.sort();
            let color_hash: u32 = sorted_letters
                .iter()
                .fold(0u32, |acc, &c| acc.wrapping_mul(31).wrapping_add(c as u32));
            let tile_color_index = (color_hash as usize) % 8;
            h.write_u8(tile_color_index as u8);
        })
    }

    fn trivial(&self) -> AnagramState {
        let word = "church";
        let valid_words = words::anagram_group(word)
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let fallback = vec![word.to_string()];
        AnagramState {
            letters: word.chars().collect(),
            valid_words: if valid_words.is_empty() {
                fallback
            } else {
                valid_words
            },
            identified: false,
        }
    }

    fn actions(&self, state: &AnagramState) -> Vec<AnagramAction> {
        if state.identified {
            return vec![];
        }
        let n = state.word_len();
        let mut result = Vec::with_capacity(n * (n - 1) + 1);
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    result.push(AnagramAction::Swap(i, j));
                }
            }
        }
        let current_word = state.current_word();
        if state.is_valid_word(&current_word) {
            result.push(AnagramAction::Identify(current_word));
        }
        result
    }

    fn apply(&self, state: &AnagramState, action: &AnagramAction) -> Option<AnagramState> {
        if state.identified {
            return None;
        }
        match action {
            AnagramAction::Swap(i, j) => {
                if i == j || *i >= state.word_len() || *j >= state.word_len() {
                    return None;
                }
                let mut new_state = state.clone();
                new_state.letters.swap(*i, *j);
                Some(new_state)
            }
            // Any valid word is accepted, whatever the current arrangement.
            AnagramAction::Identify(word) => {
                if !state.is_valid_word(word) {
                    return None;
                }
                let mut new_state = state.clone();
                new_state.identified = true;
                Some(new_state)
            }
        }
    }

    fn is_solved(&self, state: &AnagramState) -> bool {
        state.identified
    }

    fn render(&self, state: &AnagramState, resolution: u32) -> RgbBuffer {
        render::render(state, resolution)
    }

    fn action_to_str(&self, action: &AnagramAction) -> String {
        match action {
            AnagramAction::Swap(i, j) => format!("swap({}, {})", i, j),
            AnagramAction::Identify(word) => format!("identify('{}')", word),
        }
    }

    fn parse_action(&self, state: &AnagramState, s: &str) -> Option<AnagramAction> {
        if let Some(word) = s
            .strip_prefix("identify('")
            .and_then(|s| s.strip_suffix("')"))
        {
            if word.is_empty() {
                return None;
            }
            return Some(AnagramAction::Identify(word.to_string()));
        }
        let s = s.strip_prefix("swap(")?;
        let s = s.strip_suffix(')')?;
        let mut parts = s.splitn(2, ", ");
        let i: usize = parts.next()?.parse().ok()?;
        let j: usize = parts.next()?.parse().ok()?;
        if i == j || i >= state.word_len() || j >= state.word_len() {
            return None;
        }
        Some(AnagramAction::Swap(i, j))
    }

    fn action_templates(&self, state: &AnagramState) -> Vec<ActionTemplate> {
        let n = state.word_len() as i32;
        vec![
            ActionTemplate {
                segments: vec![
                    ParamType::Literal("swap(".into()),
                    ParamType::Int { min: 0, max: n - 1 },
                    ParamType::Literal(", ".into()),
                    ParamType::Int { min: 0, max: n - 1 },
                    ParamType::Literal(")".into()),
                ],
            },
            ActionTemplate {
                segments: vec![
                    ParamType::Literal("identify('".into()),
                    ParamType::OneOf(state.valid_words.clone()),
                    ParamType::Literal("')".into()),
                ],
            },
        ]
    }

    fn prompt(&self, _state: &AnagramState) -> String {
        "The letters of a word have been scrambled. Swap letters to unscramble \
         the word, then submit with identify(word)."
            .to_string()
    }

    fn action_description(&self, state: &AnagramState) -> String {
        let n = state.word_len();
        format!(
            "swap(i, j) where i and j are slot indices 0 to {}, or identify(word) to submit the answer",
            n - 1
        )
    }

    fn object_boxes(&self, state: &AnagramState) -> Vec<crate::base::object_boxes::ObjectBox> {
        object_boxes::object_boxes(state)
    }

    /// `letters`: current arrangement; `sorted`: letters in order; `len`;
    /// `id`: whether an identification has been made; `n_valid`: number of
    /// valid words.
    fn state_fields(&self, state: &AnagramState) -> Vec<StateField> {
        let mut sorted = state.letters.clone();
        sorted.sort_unstable();
        vec![
            ("letters", json_string(&state.current_word())),
            ("sorted", json_string(&sorted.iter().collect::<String>())),
            ("len", state.word_len().to_string()),
            ("id", json_bool(state.identified)),
            ("n_valid", state.valid_words.len().to_string()),
        ]
    }

    /// Dynamically finds the closest valid word and returns the best next action.
    fn solve(&self, state: &AnagramState, _budget: usize) -> Option<(AnagramAction, usize)> {
        if state.identified {
            return None;
        }

        let current_word = state.current_word();
        if state.is_valid_word(&current_word) {
            return Some((AnagramAction::Identify(current_word), 1));
        }

        let (target_word, total_swaps) = closest_valid_word(&state.letters, &state.valid_words)?;
        let target: Vec<char> = target_word.chars().collect();
        let total_cost = total_swaps + 1; // swaps + identify

        let perm = build_permutation(&state.letters, &target);
        for i in 0..perm.len() {
            if perm[i] != i {
                return Some((AnagramAction::Swap(i, perm[i]), total_cost));
            }
        }

        unreachable!("letters don't form a valid word but no displaced position found")
    }

    fn solve_path(
        &self,
        state: &AnagramState,
        _budget: usize,
    ) -> Option<Vec<(AnagramAction, usize)>> {
        if state.identified {
            return None;
        }

        let mut path = Vec::new();
        let mut letters = state.letters.clone();

        loop {
            let current_word: String = letters.iter().collect();
            if state.is_valid_word(&current_word) {
                path.push((AnagramAction::Identify(current_word), 1));
                break;
            }

            let (target_word, remaining_swaps) = closest_valid_word(&letters, &state.valid_words)?;
            let target: Vec<char> = target_word.chars().collect();
            let cost = remaining_swaps + 1;

            let perm = build_permutation(&letters, &target);
            for i in 0..perm.len() {
                if perm[i] != i {
                    path.push((AnagramAction::Swap(i, perm[i]), cost));
                    letters.swap(i, perm[i]);
                    break;
                }
            }
        }

        Some(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::episode::{Episode, GameEpisode};
    use rand::SeedableRng;
    use std::sync::Arc;

    fn make_state(current: &str, valid: &[&str]) -> AnagramState {
        AnagramState {
            letters: current.chars().collect(),
            valid_words: valid.iter().map(|s| s.to_string()).collect(),
            identified: false,
        }
    }

    #[test]
    fn state_text_describes_arrangement_and_sorted_letters() {
        let fields = Anagram::default().state_fields(&make_state("tca", &["cat"]));
        assert_eq!(
            crate::base::state_text::format_state_json(&fields, 0),
            r#"{"letters":"tca","sorted":"act","len":3,"id":false,"n_valid":1,"moves":0}"#
        );
    }

    fn make_episode(current: &str, valid: &[&str]) -> GameEpisode<Anagram> {
        let state = make_state(current, valid);
        let game = Arc::new(Anagram::default());
        GameEpisode::new(game, state, 50_000)
    }

    // --- Registration ---

    #[test]
    fn registered_in_registry() {
        assert!(crate::games::registry::find_game("anagram").is_some());
    }

    // --- Trivial state ---

    #[test]
    fn trivial_is_solvable() {
        let game = Anagram::default();
        let state = game.trivial();
        let result = game.solve(&state, 50_000);
        assert!(result.is_some(), "trivial state should be solvable");
        let (action, cost) = result.unwrap();
        assert!(matches!(action, AnagramAction::Identify(_)));
        assert_eq!(cost, 1, "trivial state should be solvable in 1 action");
    }

    // --- Generation ---

    #[test]
    fn generate_produces_solvable() {
        let game = Anagram {
            word_len_range: (6, 8),
            min_swaps: 2,
        };
        for seed in 0..25u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            let state = game.generate(&mut rng).expect("seed should generate");
            let result = game.solve(&state, 50_000);
            assert!(result.is_some(), "seed {} produced unsolvable puzzle", seed);
        }
    }

    #[test]
    fn generate_not_already_solved() {
        let game = Anagram::default();
        for seed in 0..25u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            let state = game.generate(&mut rng).expect("seed should generate");
            assert!(!game.is_solved(&state), "seed {} was already solved", seed);
        }
    }

    // --- Game logic ---

    #[test]
    fn apply_swap_valid() {
        let game = Anagram::default();
        let state = make_state("hcurch", &["church"]);
        let next = game
            .apply(&state, &AnagramAction::Swap(0, 1))
            .expect("valid swap");
        assert_eq!(next.letters, vec!['c', 'h', 'u', 'r', 'c', 'h']);
    }

    #[test]
    fn apply_swap_same_index() {
        let game = Anagram::default();
        let state = make_state("hcurch", &["church"]);
        assert!(game.apply(&state, &AnagramAction::Swap(0, 0)).is_none());
    }

    #[test]
    fn apply_swap_out_of_bounds() {
        let game = Anagram::default();
        let state = make_state("hcurch", &["church"]);
        assert!(game.apply(&state, &AnagramAction::Swap(0, 10)).is_none());
    }

    #[test]
    fn apply_identify_correct() {
        let game = Anagram::default();
        let state = make_state("church", &["church"]);
        let next = game
            .apply(&state, &AnagramAction::Identify("church".to_string()))
            .expect("identify should succeed");
        assert!(next.identified);
        assert!(game.is_solved(&next));
    }

    #[test]
    fn apply_identify_any_valid_word() {
        let game = Anagram::default();
        let state = make_state("listen", &["listen", "silent", "tinsel"]);
        let next = game
            .apply(&state, &AnagramAction::Identify("listen".to_string()))
            .expect("identify should succeed for any valid word");
        assert!(game.is_solved(&next));
    }

    #[test]
    fn apply_identify_not_in_valid_words() {
        let game = Anagram::default();
        let state = make_state("church", &["church"]);
        assert!(game
            .apply(&state, &AnagramAction::Identify("zzzzz".to_string()))
            .is_none());
    }

    #[test]
    fn apply_after_identified() {
        let game = Anagram::default();
        let mut state = make_state("church", &["church"]);
        state.identified = true;
        assert!(game.apply(&state, &AnagramAction::Swap(0, 1)).is_none());
        assert!(game
            .apply(&state, &AnagramAction::Identify("church".to_string()))
            .is_none());
    }

    #[test]
    fn is_solved_only_after_identify() {
        let game = Anagram::default();
        let state = make_state("church", &["church"]);
        assert!(!game.is_solved(&state));
        let solved = game
            .apply(&state, &AnagramAction::Identify("church".to_string()))
            .unwrap();
        assert!(game.is_solved(&solved));
    }

    #[test]
    fn actions_includes_identify_when_correct() {
        let game = Anagram::default();
        let state = make_state("church", &["church"]);
        let actions = game.actions(&state);
        assert!(actions
            .iter()
            .any(|a| matches!(a, AnagramAction::Identify(_))));
    }

    #[test]
    fn actions_excludes_identify_when_wrong() {
        let game = Anagram::default();
        let state = make_state("hcurch", &["church"]);
        let actions = game.actions(&state);
        assert!(!actions
            .iter()
            .any(|a| matches!(a, AnagramAction::Identify(_))));
    }

    #[test]
    fn actions_empty_when_identified() {
        let game = Anagram::default();
        let mut state = make_state("church", &["church"]);
        state.identified = true;
        assert!(game.actions(&state).is_empty());
    }

    // --- Action serialization ---

    #[test]
    fn action_to_str_parse_roundtrip() {
        let game = Anagram::default();
        let state = make_state("hcurch", &["church"]);
        let actions = game.actions(&state);
        for action in &actions {
            let s = game.action_to_str(action);
            let parsed = game
                .parse_action(&state, &s)
                .unwrap_or_else(|| panic!("failed to parse '{}'", s));
            assert_eq!(
                game.action_to_str(&parsed),
                s,
                "roundtrip failed for '{}'",
                s
            );
        }
    }

    #[test]
    fn parse_action_rejects_same_index() {
        let game = Anagram::default();
        let state = make_state("hcurch", &["church"]);
        assert!(game.parse_action(&state, "swap(1, 1)").is_none());
    }

    #[test]
    fn parse_action_rejects_invalid() {
        let game = Anagram::default();
        let state = make_state("hcurch", &["church"]);
        assert!(game.parse_action(&state, "fly").is_none());
        assert!(game.parse_action(&state, "swap(0)").is_none());
        assert!(game.parse_action(&state, "swap(99, 0)").is_none());
    }

    #[test]
    fn action_templates_covers_all_valid() {
        let game = Anagram::default();
        let state = make_state("church", &["church"]);
        let templates = game.action_templates(&state);
        let expanded: Vec<String> = templates.iter().flat_map(|t| t.expand()).collect();
        let actions = game.actions(&state);
        for action in &actions {
            let s = game.action_to_str(action);
            assert!(
                expanded.contains(&s),
                "valid action '{}' not in template expansion",
                s
            );
        }
    }

    // --- Solver ---

    #[test]
    fn solve_one_swap_plus_identify() {
        let game = Anagram::default();
        let state = make_state("hcurch", &["church"]);
        let result = game.solve(&state, 50_000);
        assert!(result.is_some());
        let (action, cost) = result.unwrap();
        assert!(matches!(action, AnagramAction::Swap(_, _)));
        assert_eq!(cost, 2); // 1 swap + 1 identify
    }

    #[test]
    fn solve_already_correct_returns_identify() {
        let game = Anagram::default();
        let state = make_state("church", &["church"]);
        let result = game.solve(&state, 50_000);
        assert!(result.is_some());
        let (action, cost) = result.unwrap();
        assert!(matches!(action, AnagramAction::Identify(_)));
        assert_eq!(cost, 1);
    }

    #[test]
    fn solve_identified_returns_none() {
        let game = Anagram::default();
        let mut state = make_state("church", &["church"]);
        state.identified = true;
        assert!(game.solve(&state, 50_000).is_none());
    }

    #[test]
    fn solve_picks_closest_word() {
        let game = Anagram::default();
        // "listen" is 0 swaps from "listen", but some swaps from "silent"
        let state = make_state("listen", &["listen", "silent"]);
        let result = game.solve(&state, 50_000);
        assert!(result.is_some());
        let (action, cost) = result.unwrap();
        assert!(matches!(action, AnagramAction::Identify(ref w) if w == "listen"));
        assert_eq!(cost, 1);
    }

    #[test]
    fn solve_path_reaches_valid_word() {
        let game = Anagram::default();
        let state = make_state("hcurhc", &["church"]);
        let path = game.solve_path(&state, 50_000).unwrap();

        let mut letters = state.letters.clone();
        for (action, _cost) in &path {
            match action {
                AnagramAction::Swap(i, j) => letters.swap(*i, *j),
                AnagramAction::Identify(_) => {}
            }
        }
        let final_word: String = letters.iter().collect();
        assert!(state.is_valid_word(&final_word));
        assert!(matches!(path.last().unwrap().0, AnagramAction::Identify(_)));
    }

    #[test]
    fn solve_path_costs_decrease() {
        let game = Anagram::default();
        let state = make_state("hcurhc", &["church"]);
        let path = game.solve_path(&state, 50_000).unwrap();
        for i in 1..path.len() {
            assert!(
                path[i].1 <= path[i - 1].1,
                "costs should not increase along path"
            );
        }
    }

    // --- Episode integration tests ---

    #[test]
    fn episode_expert_action_returns_optimal() {
        let ep = make_episode("church", &["church"]);
        assert_eq!(ep.expert_action(), "identify('church')");
    }

    #[test]
    fn episode_expert_action_done_when_solved() {
        let mut state = make_state("church", &["church"]);
        state.identified = true;
        let game = Arc::new(Anagram::default());
        let ep = GameEpisode::new(game, state, 50_000);
        assert_eq!(ep.expert_action(), "done");
    }

    #[test]
    fn episode_expert_action_solves_puzzle() {
        let mut ep = make_episode("hcurch", &["church"]);
        let mut steps = 0;
        while !ep.is_solved() {
            let action = ep.expert_action();
            if action == "done" {
                break;
            }
            assert!(ep.apply(&action), "expert action '{}' failed", action);
            steps += 1;
            assert!(steps <= 20, "expert did not solve in 20 steps");
        }
        assert!(ep.is_solved());
    }

    #[test]
    fn episode_apply_undo_roundtrip() {
        let mut ep = make_episode("hcurch", &["church"]);
        let original = ep.state.clone();

        assert!(ep.apply("swap(0, 1)"));
        assert_ne!(ep.state, original);

        ep.undo();
        assert_eq!(ep.state, original);
    }

    #[test]
    fn episode_apply_invalid_returns_false() {
        let mut ep = make_episode("hcurch", &["church"]);
        let original = ep.state.clone();

        assert!(!ep.apply("swap(0, 0)"));
        assert_eq!(ep.state, original);

        assert!(!ep.apply("fly"));
        assert_eq!(ep.state, original);

        assert!(!ep.apply("identify()"));
        assert_eq!(ep.state, original);
    }

    #[test]
    fn episode_reset_restores_initial() {
        let mut ep = make_episode("hcurch", &["church"]);
        let initial = ep.state.clone();

        ep.apply("swap(0, 1)");
        ep.apply("swap(1, 2)");
        assert_ne!(ep.state, initial);

        ep.reset();
        assert_eq!(ep.state, initial);
        assert!(ep.history.is_empty());
    }

    #[test]
    fn episode_enumerate_includes_meta() {
        let ep = make_episode("hcurch", &["church"]);
        let actions = ep.enumerate_actions();
        assert!(actions.contains(&"swap(0, 1)".to_string()));
        assert!(actions.contains(&"back".to_string()));
        assert!(actions.contains(&"reset".to_string()));
        assert!(actions.contains(&"done".to_string()));
    }

    #[test]
    fn episode_cache_consistency() {
        let state = make_state("hcurch", &["church"]);
        let game = Arc::new(Anagram::default());
        let ep = GameEpisode::new(game.clone(), state.clone(), 50_000);

        let cached_cost = ep.cached_solve(&state);
        let fresh = game.solve(&state, 50_000).map(|(_, c)| c);
        assert_eq!(cached_cost, fresh);
    }

    // --- Word list tests ---

    #[test]
    fn dictionary_has_words() {
        let all = words::all_words();
        assert!(all.len() > 30_000, "expected 30K+ words, got {}", all.len());
    }

    #[test]
    fn dictionary_words_lowercase_ascii() {
        for word in words::all_words() {
            assert!(
                word.chars().all(|c| c.is_ascii_lowercase()),
                "word '{}' contains non-lowercase-ascii characters",
                word
            );
        }
    }

    #[test]
    fn dictionary_words_length_range() {
        for word in words::all_words() {
            let len = word.len();
            assert!(
                (6..=10).contains(&len),
                "word '{}' has length {} (expected 6-10)",
                word,
                len
            );
        }
    }

    // --- Identify accepts any valid word from any state ---

    #[test]
    fn identify_accepted_from_scrambled_state() {
        let game = Anagram::default();
        let state = make_state("hcurch", &["church"]);
        let next = game
            .apply(&state, &AnagramAction::Identify("church".to_string()))
            .expect("identify should be accepted from a scrambled state");
        assert!(next.identified);
        assert!(game.is_solved(&next));
    }

    #[test]
    fn identify_accepted_from_intermediate_state() {
        let game = Anagram::default();
        let state = make_state("hcurch", &["church"]);
        let intermediate = game
            .apply(&state, &AnagramAction::Swap(0, 5))
            .expect("swap should succeed");
        assert_ne!(intermediate.current_word(), "church");
        let next = game
            .apply(
                &intermediate,
                &AnagramAction::Identify("church".to_string()),
            )
            .expect("identify should be accepted from an intermediate state");
        assert!(game.is_solved(&next));
    }

    #[test]
    fn identify_rejects_invalid_word_from_any_state() {
        let game = Anagram::default();
        for letters in ["hcurch", "church", "hchurc"] {
            let state = make_state(letters, &["church"]);
            assert!(
                game.apply(&state, &AnagramAction::Identify("zzzzzz".to_string()))
                    .is_none(),
                "invalid word must be rejected from state '{letters}'"
            );
        }
    }
}
