//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
//! Anagram object boxes (Boxes CoT): the tiles stay in place and only their
//! letters move, so each letter maps to the slot it is in, `{"c":[0],"a":[1]}`.
//! The slot index is the number drawn under each tile. A repeated letter gets a
//! `_k` suffix in slot order (`d_0`, `d_1`).

use std::collections::HashMap;

use crate::base::object_boxes::ObjectBox;

use super::state::AnagramState;

pub fn object_boxes(state: &AnagramState) -> Vec<ObjectBox> {
    let mut letter_counts: HashMap<char, usize> = HashMap::new();
    for &letter in &state.letters {
        *letter_counts.entry(letter).or_insert(0) += 1;
    }
    let mut letters_seen: HashMap<char, usize> = HashMap::new();
    let mut object_boxes = Vec::with_capacity(state.letters.len());
    for (slot, &letter) in state.letters.iter().enumerate() {
        let name = if letter_counts[&letter] > 1 {
            let occurrence = letters_seen.entry(letter).or_insert(0);
            let name = format!("{letter}_{occurrence}");
            *occurrence += 1;
            name
        } else {
            letter.to_string()
        };
        object_boxes.push((name, vec![slot as i32]));
    }
    object_boxes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(letters: &str) -> AnagramState {
        AnagramState {
            letters: letters.chars().collect(),
            valid_words: vec![letters.to_string()],
            identified: false,
        }
    }

    fn names(object_boxes: &[ObjectBox]) -> Vec<&str> {
        object_boxes.iter().map(|(name, _)| name.as_str()).collect()
    }

    #[test]
    fn maps_letters_to_slots_and_numbers_repeats() {
        assert_eq!(names(&object_boxes(&state("cat"))), vec!["c", "a", "t"]);
        assert_eq!(names(&object_boxes(&state("add"))), vec!["a", "d_0", "d_1"]);
        let slots: Vec<Vec<i32>> = object_boxes(&state("church"))
            .into_iter()
            .map(|(_, slot)| slot)
            .collect();
        assert_eq!(slots, (0..6).map(|slot| vec![slot]).collect::<Vec<_>>());
    }

    #[test]
    fn swapping_letters_moves_them_between_slots() {
        assert_eq!(object_boxes(&state("act"))[0], ("a".to_string(), vec![0]));
        assert_eq!(object_boxes(&state("cat"))[0], ("c".to_string(), vec![0]));
    }
}
