//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
use rand::Rng;
use rand_chacha::ChaChaRng;

use super::state::AnagramState;
use super::words;

/// Generate a random solvable anagram puzzle.
///
/// Picks a word from the dictionary (filtered by length), finds all valid
/// anagram solutions, shuffles the letters, and verifies the scramble
/// requires at least `min_swaps` swaps to reach the *closest* valid word.
pub fn generate(
    rng: &mut ChaChaRng,
    word_len_range: (usize, usize),
    min_swaps: usize,
) -> Option<AnagramState> {
    let candidates = words::words_in_length_range(word_len_range.0, word_len_range.1);
    if candidates.is_empty() {
        return None;
    }

    for _ in 0..100 {
        let word = candidates[rng.gen_range(0..candidates.len())];
        let valid_words: Vec<String> = words::anagram_group(word)
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        let mut letters: Vec<char> = word.chars().collect();

        // Fisher-Yates shuffle
        for i in (1..letters.len()).rev() {
            let j = rng.gen_range(0..=i);
            letters.swap(i, j);
        }

        let closest_distance = closest_swap_distance(&letters, &valid_words);
        if closest_distance < min_swaps {
            continue;
        }

        return Some(AnagramState {
            letters,
            valid_words,
            identified: false,
        });
    }

    None
}

/// Minimum swap distance from `current` to the closest word in `valid_words`.
pub fn closest_swap_distance(current: &[char], valid_words: &[String]) -> usize {
    valid_words
        .iter()
        .map(|w| {
            let target: Vec<char> = w.chars().collect();
            minimum_swap_distance(current, &target)
        })
        .min()
        .unwrap_or(usize::MAX)
}

/// Find the closest valid word and return it along with its swap distance.
pub fn closest_valid_word(current: &[char], valid_words: &[String]) -> Option<(String, usize)> {
    valid_words
        .iter()
        .map(|w| {
            let target: Vec<char> = w.chars().collect();
            (w.clone(), minimum_swap_distance(current, &target))
        })
        .min_by_key(|(_, dist)| *dist)
}

/// Minimum number of swaps to transform `current` into `target`.
pub fn minimum_swap_distance(current: &[char], target: &[char]) -> usize {
    let perm = build_permutation(current, target);
    count_swaps_from_perm(&perm)
}

/// Build a permutation mapping: perm[i] = j means the letter that belongs
/// at position i is currently at position j.
///
/// Computes the permutation that minimises the swap distance from `current`
/// to `target` via a three-phase bipartite matching:
///
///   1. Lock in all fixed points (positions where current[i] == target[i]).
///   2. Maximum 2-cycle matching. For each unordered pair of pair-types
///      {(a, b), (b, a)} with a != b, positions of type (a, b) pair up with
///      positions of type (b, a); each pairing resolves as a single swap.
///      This is a bipartite matching in the swap-compatibility graph where
///      every edge is valid (complete bipartite structure), so greedy pairing
///      is optimal and attains min(|(a,b)|, |(b,a)|) matches per pair.
///   3. Remaining positions are forced into cycles of length >= 3. Assign
///      them greedily; Hall's condition guarantees a valid bijection.
///
/// Because every k-cycle costs (k-1) swaps and the per-position cost
/// (k-1)/k is strictly increasing in k, maximising 2-cycles minimises the
/// total swap count.
pub fn build_permutation(current: &[char], target: &[char]) -> Vec<usize> {
    let number_of_positions = current.len();
    let mut permutation = vec![usize::MAX; number_of_positions];
    let mut source_used = vec![false; number_of_positions];

    // Phase 1: fixed points.
    for position_index in 0..number_of_positions {
        if current[position_index] == target[position_index] {
            permutation[position_index] = position_index;
            source_used[position_index] = true;
        }
    }

    // Phase 2: maximum 2-cycle matching.
    // Group unfixed positions by their (current, target) letter-pair type.
    let mut positions_by_pair_type: std::collections::BTreeMap<(char, char), Vec<usize>> =
        std::collections::BTreeMap::new();
    for position_index in 0..number_of_positions {
        if permutation[position_index] == usize::MAX {
            positions_by_pair_type
                .entry((current[position_index], target[position_index]))
                .or_default()
                .push(position_index);
        }
    }
    let pair_types: Vec<(char, char)> = positions_by_pair_type.keys().copied().collect();
    for (letter_from, letter_to) in pair_types {
        // Process each unordered type-pair {(a, b), (b, a)} exactly once.
        if letter_from >= letter_to {
            continue;
        }
        let forward_positions: Vec<usize> = positions_by_pair_type
            .get(&(letter_from, letter_to))
            .cloned()
            .unwrap_or_default();
        let reverse_positions: Vec<usize> = positions_by_pair_type
            .get(&(letter_to, letter_from))
            .cloned()
            .unwrap_or_default();
        let matched_pair_count = forward_positions.len().min(reverse_positions.len());
        for match_index in 0..matched_pair_count {
            let forward_position = forward_positions[match_index];
            let reverse_position = reverse_positions[match_index];
            permutation[forward_position] = reverse_position;
            permutation[reverse_position] = forward_position;
            source_used[forward_position] = true;
            source_used[reverse_position] = true;
        }
    }

    // Phase 3: greedy assignment of the residual (cycles of length >= 3).
    for target_position in 0..number_of_positions {
        if permutation[target_position] != usize::MAX {
            continue;
        }
        for source_position in 0..number_of_positions {
            if !source_used[source_position] && current[source_position] == target[target_position]
            {
                permutation[target_position] = source_position;
                source_used[source_position] = true;
                break;
            }
        }
    }

    permutation
}

/// Count minimum swaps via cycle decomposition: n − number_of_cycles.
pub fn count_swaps_from_perm(perm: &[usize]) -> usize {
    let n = perm.len();
    let mut visited = vec![false; n];
    let mut num_cycles = 0;
    for i in 0..n {
        if visited[i] {
            continue;
        }
        let mut j = i;
        while !visited[j] {
            visited[j] = true;
            j = perm[j];
        }
        num_cycles += 1;
    }
    n - num_cycles
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn build_permutation_identity() {
        let word: Vec<char> = "church".chars().collect();
        let perm = build_permutation(&word, &word);
        assert_eq!(perm, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(count_swaps_from_perm(&perm), 0);
    }

    #[test]
    fn build_permutation_simple_swap() {
        let current: Vec<char> = "hcurch".chars().collect();
        let target: Vec<char> = "church".chars().collect();
        let perm = build_permutation(&current, &target);
        let swaps = count_swaps_from_perm(&perm);
        assert_eq!(swaps, 1);
    }

    #[test]
    fn build_permutation_repeated_letters() {
        let current: Vec<char> = "hcrcuh".chars().collect();
        let target: Vec<char> = "church".chars().collect();
        let perm = build_permutation(&current, &target);
        let swaps = count_swaps_from_perm(&perm);
        assert!(swaps <= 4);
    }

    #[test]
    fn build_permutation_maximises_fixed_points() {
        let current: Vec<char> = "chhucr".chars().collect();
        let target: Vec<char> = "church".chars().collect();
        let perm = build_permutation(&current, &target);
        // Count fixed points
        let fixed = perm.iter().enumerate().filter(|(i, &p)| p == *i).count();
        assert!(fixed >= 1);
    }

    #[test]
    fn generate_returns_some() {
        let mut rng = ChaChaRng::seed_from_u64(42);
        let state = generate(&mut rng, (6, 8), 2);
        assert!(state.is_some());
    }

    #[test]
    fn generate_not_already_solved() {
        for seed in 0..25u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            if let Some(state) = generate(&mut rng, (6, 8), 2) {
                let current_word: String = state.letters.iter().collect();
                assert!(
                    !state.is_valid_word(&current_word),
                    "seed {seed}: scrambled letters already form a valid word"
                );
            }
        }
    }

    #[test]
    fn generate_min_scramble_distance() {
        let min_swaps = 3;
        for seed in 0..25u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            if let Some(state) = generate(&mut rng, (6, 8), min_swaps) {
                let dist = closest_swap_distance(&state.letters, &state.valid_words);
                assert!(
                    dist >= min_swaps,
                    "seed {seed} produced distance {dist} (min {min_swaps})"
                );
            }
        }
    }

    #[test]
    fn generate_with_high_min_swaps() {
        let mut rng = ChaChaRng::seed_from_u64(42);
        let state = generate(&mut rng, (8, 10), 5);
        assert!(
            state.is_some(),
            "should generate with min_swaps=5 for 8-10 letter words"
        );
        let state = state.unwrap();
        let dist = closest_swap_distance(&state.letters, &state.valid_words);
        assert!(dist >= 5);
    }

    #[test]
    fn generate_empty_range_returns_none() {
        let mut rng = ChaChaRng::seed_from_u64(0);
        assert!(generate(&mut rng, (20, 30), 2).is_none());
    }

    #[test]
    fn generate_valid_words_are_anagrams() {
        let mut rng = ChaChaRng::seed_from_u64(42);
        let state = generate(&mut rng, (6, 8), 2).unwrap();
        let mut sorted_letters: Vec<char> = state.letters.clone();
        sorted_letters.sort();
        for word in &state.valid_words {
            let mut sorted_word: Vec<char> = word.chars().collect();
            sorted_word.sort();
            assert_eq!(
                sorted_letters, sorted_word,
                "word '{}' is not an anagram",
                word
            );
        }
    }

    #[test]
    fn closest_swap_distance_picks_nearest() {
        let current: Vec<char> = "listen".chars().collect();
        let valid = vec!["listen".to_string(), "silent".to_string()];
        let dist = closest_swap_distance(&current, &valid);
        assert_eq!(dist, 0); // "listen" is already a match
    }
}
