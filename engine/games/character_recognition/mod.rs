//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
pub mod gen;
pub mod object_boxes;
pub mod render;
pub mod state;

use std::sync::Arc;

use rand_chacha::ChaChaRng;

use crate::base::game::{Game, RgbBuffer};
use crate::base::state_text::{json_numbers, json_optional_numbers, StateField};
use crate::base::template::{ActionTemplate, ParamType};

use state::{CharacterRecognitionAction, CharacterRecognitionState};

// ---------------------------------------------------------------------------
// Game struct
// ---------------------------------------------------------------------------

pub struct CharacterRecognition {
    pub num_pieces_range: (usize, usize),
    pub grid_size_range: (usize, usize),
    pub slot_grid_dim: usize,
    pub solve_budget: usize,
}

// ---------------------------------------------------------------------------
// Game trait implementation
// ---------------------------------------------------------------------------

impl Game for CharacterRecognition {
    type State = CharacterRecognitionState;
    type Action = CharacterRecognitionAction;

    fn generate(&self, rng: &mut ChaChaRng) -> Option<CharacterRecognitionState> {
        let puzzle = gen::generate(
            rng,
            self.num_pieces_range,
            self.grid_size_range,
            self.slot_grid_dim,
        )?;
        let n = puzzle.num_pieces;
        let rotations = puzzle.initial_rotations.clone();
        let placements = vec![None; n];
        Some(CharacterRecognitionState {
            puzzle: Arc::new(puzzle),
            rotations,
            placements,
            answer: None,
        })
    }

    fn puzzle_fingerprint(&self, state: &CharacterRecognitionState) -> u64 {
        use std::hash::Hasher;
        let p = &*state.puzzle;
        crate::base::fingerprint::fingerprint(|h| {
            // 1-6: glyph + layout + color scalars.
            h.write_u32(p.character_char as u32);
            h.write_u64(p.font_idx as u64);
            h.write_u64(p.grid_size as u64);
            h.write_u64(p.num_pieces as u64);
            h.write_u64(p.slot_grid_dim as u64);
            h.write(&p.fill_color); // [u8; 3]

            // 7: fragment geometry, in piece-index order. Piece cells come from
            // HashSet iteration (nondeterministic order), so they are sorted
            // before hashing.
            for piece in &p.pieces {
                let mut cells = piece.cells.clone();
                cells.sort_unstable();
                h.write_u64(cells.len() as u64);
                for &(x, y) in &cells {
                    h.write_u32(x as u32);
                    h.write_u32(y as u32);
                }
            }

            // 8: the visible scramble orientation (Vec<u8>, piece-index order).
            h.write(&p.initial_rotations);

            // 9: slot answer key, in row-major generation order.
            for slot in &p.slots {
                h.write_u64(slot.id.0 as u64);
                h.write_u64(slot.id.1 as u64);
                match slot.matching_piece_idx {
                    Some(idx) => h.write_u64(idx as u64 + 1),
                    None => h.write_u64(0),
                }
            }
        })
    }

    fn trivial(&self) -> CharacterRecognitionState {
        let puzzle = gen::trivial(self.num_pieces_range.0, self.slot_grid_dim);
        let n = puzzle.num_pieces;
        let rotations = puzzle.initial_rotations.clone();
        let placements = vec![None; n];
        CharacterRecognitionState {
            puzzle: Arc::new(puzzle),
            rotations,
            placements,
            answer: None,
        }
    }

    fn actions(&self, state: &CharacterRecognitionState) -> Vec<CharacterRecognitionAction> {
        let mut acts = Vec::new();
        let n = state.puzzle.num_pieces;

        for pid in 0..n {
            if state.placements[pid].is_none() {
                acts.push(CharacterRecognitionAction::RotateCW {
                    piece_id: pid,
                    angle: 90,
                });
                acts.push(CharacterRecognitionAction::RotateCCW {
                    piece_id: pid,
                    angle: 90,
                });
                for (slot_idx, slot) in state.puzzle.slots.iter().enumerate() {
                    if !state.slot_occupied(slot_idx) {
                        acts.push(CharacterRecognitionAction::Place {
                            piece_id: pid,
                            row: slot.id.0,
                            col: slot.id.1,
                        });
                    }
                }
            } else {
                acts.push(CharacterRecognitionAction::Undo { piece_id: pid });
            }
        }

        if state.all_pieces_correctly_placed() {
            for name in &state.puzzle.allowed_chars {
                acts.push(CharacterRecognitionAction::Identify {
                    character: name.clone(),
                });
            }
        }

        acts
    }

    fn apply(
        &self,
        state: &CharacterRecognitionState,
        action: &CharacterRecognitionAction,
    ) -> Option<CharacterRecognitionState> {
        let n = state.puzzle.num_pieces;
        match action {
            CharacterRecognitionAction::RotateCW { piece_id, angle } => {
                if *piece_id >= n || *angle != 90 {
                    return None;
                }
                if state.placements[*piece_id].is_some() {
                    return None;
                }
                let mut new_state = state.clone();
                new_state.rotations[*piece_id] = (new_state.rotations[*piece_id] + 1) % 4;
                Some(new_state)
            }
            CharacterRecognitionAction::RotateCCW { piece_id, angle } => {
                if *piece_id >= n || *angle != 90 {
                    return None;
                }
                if state.placements[*piece_id].is_some() {
                    return None;
                }
                let mut new_state = state.clone();
                new_state.rotations[*piece_id] = (new_state.rotations[*piece_id] + 3) % 4;
                Some(new_state)
            }
            CharacterRecognitionAction::Place { piece_id, row, col } => {
                if *piece_id >= n {
                    return None;
                }
                if state.placements[*piece_id].is_some() {
                    return None;
                }
                let slot_idx = state
                    .puzzle
                    .slots
                    .iter()
                    .position(|s| s.id == (*row, *col))?;
                if state.slot_occupied(slot_idx) {
                    return None;
                }
                let mut new_state = state.clone();
                new_state.placements[*piece_id] = Some(slot_idx);
                Some(new_state)
            }
            CharacterRecognitionAction::Undo { piece_id } => {
                if *piece_id >= n {
                    return None;
                }
                if state.placements[*piece_id].is_none() {
                    return None;
                }
                let mut new_state = state.clone();
                new_state.placements[*piece_id] = None;
                Some(new_state)
            }
            // Any allowed character is accepted, whatever the assembly state.
            CharacterRecognitionAction::Identify { character } => {
                if !state.puzzle.allowed_chars.contains(character) {
                    return None;
                }
                let mut new_state = state.clone();
                new_state.answer = Some(character.clone());
                Some(new_state)
            }
        }
    }

    /// Solved once the character is identified correctly, from any assembly state.
    fn is_solved(&self, state: &CharacterRecognitionState) -> bool {
        state.answer.as_ref() == Some(&state.puzzle.character)
    }

    fn render(&self, state: &CharacterRecognitionState, resolution: u32) -> RgbBuffer {
        render::render(state, resolution)
    }

    fn action_to_str(&self, action: &CharacterRecognitionAction) -> String {
        action.to_str()
    }

    fn parse_action(
        &self,
        state: &CharacterRecognitionState,
        s: &str,
    ) -> Option<CharacterRecognitionAction> {
        CharacterRecognitionAction::parse(
            s,
            state.puzzle.num_pieces,
            state.puzzle.slot_grid_dim,
            &state.puzzle.allowed_chars,
        )
    }

    fn action_templates(&self, state: &CharacterRecognitionState) -> Vec<ActionTemplate> {
        let n = state.puzzle.num_pieces as i32;
        let grid_dim = state.puzzle.slot_grid_dim as i32;
        let char_strs: Vec<String> = state.puzzle.allowed_chars.clone();

        vec![
            ActionTemplate {
                segments: vec![
                    ParamType::Literal("rotatecw(".into()),
                    ParamType::Int { min: 0, max: n - 1 },
                    ParamType::Literal(", 90)".into()),
                ],
            },
            ActionTemplate {
                segments: vec![
                    ParamType::Literal("rotateccw(".into()),
                    ParamType::Int { min: 0, max: n - 1 },
                    ParamType::Literal(", 90)".into()),
                ],
            },
            ActionTemplate {
                segments: vec![
                    ParamType::Literal("place(".into()),
                    ParamType::Int { min: 0, max: n - 1 },
                    ParamType::Literal(", ".into()),
                    ParamType::Int {
                        min: 0,
                        max: grid_dim - 1,
                    },
                    ParamType::Literal(", ".into()),
                    ParamType::Int {
                        min: 0,
                        max: grid_dim - 1,
                    },
                    ParamType::Literal(")".into()),
                ],
            },
            ActionTemplate {
                segments: vec![
                    ParamType::Literal("undo(".into()),
                    ParamType::Int { min: 0, max: n - 1 },
                    ParamType::Literal(")".into()),
                ],
            },
            ActionTemplate {
                segments: vec![
                    ParamType::Literal("identify('".into()),
                    ParamType::OneOf(char_strs),
                    ParamType::Literal("')".into()),
                ],
            },
        ]
    }

    fn prompt(&self, _state: &CharacterRecognitionState) -> String {
        "A character has been broken into pieces that are scattered and rotated. \
         Rotate each numbered piece to the correct orientation and place it into \
         the matching lettered slot in the assembly area. Once all pieces are \
         correctly assembled, identify the character."
            .to_string()
    }

    fn action_description(&self, _state: &CharacterRecognitionState) -> String {
        "rotatecw(piece_id, 90), rotateccw(piece_id, 90), \
         place(piece_id, row, col), undo(piece_id), identify('char')"
            .to_string()
    }

    fn object_boxes(
        &self,
        state: &CharacterRecognitionState,
    ) -> Vec<crate::base::object_boxes::ObjectBox> {
        object_boxes::object_boxes(state)
    }

    /// `r`: rotation of each piece in quarter turns (0-3); `p`: slot index of
    /// each piece (`null` = unplaced); `n_placed`: placed pieces.
    fn state_fields(&self, state: &CharacterRecognitionState) -> Vec<StateField> {
        let num_placed = state
            .placements
            .iter()
            .filter(|slot| slot.is_some())
            .count();
        vec![
            ("r", json_numbers(state.rotations.iter())),
            ("p", json_optional_numbers(state.placements.iter().copied())),
            ("n_placed", num_placed.to_string()),
        ]
    }

    // --- Analytical solver ---

    fn solve(
        &self,
        state: &CharacterRecognitionState,
        _budget: usize,
    ) -> Option<(CharacterRecognitionAction, usize)> {
        // Phase 1: undo wrongly placed pieces
        for &i in &state.puzzle.piece_solve_order {
            if state.placements[i].is_some() && !state.piece_correct(i) {
                let cost = remaining_cost(state);
                return Some((CharacterRecognitionAction::Undo { piece_id: i }, cost));
            }
        }

        // Phase 2: rotate and place unplaced pieces
        for &i in &state.puzzle.piece_solve_order {
            if state.placements[i].is_some() {
                continue;
            }
            let slot_idx = state
                .puzzle
                .slots
                .iter()
                .position(|s| s.matching_piece_idx == Some(i))
                .unwrap();
            let correct_rot = state.puzzle.slots[slot_idx].correct_rotation;
            let current_rot = state.rotations[i];

            if current_rot != correct_rot {
                let action = best_rotation_90(i, current_rot, correct_rot);
                let cost = remaining_cost(state);
                return Some((action, cost));
            }

            let (row, col) = state.puzzle.slots[slot_idx].id;
            let cost = remaining_cost(state);
            return Some((
                CharacterRecognitionAction::Place {
                    piece_id: i,
                    row,
                    col,
                },
                cost,
            ));
        }

        // Phase 3: identify
        if state.all_pieces_correctly_placed()
            && state.answer.as_ref() != Some(&state.puzzle.character)
        {
            let cost = remaining_cost(state);
            return Some((
                CharacterRecognitionAction::Identify {
                    character: state.puzzle.character.clone(),
                },
                cost,
            ));
        }

        None
    }

    fn solve_path(
        &self,
        state: &CharacterRecognitionState,
        budget: usize,
    ) -> Option<Vec<(CharacterRecognitionAction, usize)>> {
        let mut path = Vec::new();
        let mut current = state.clone();
        for _ in 0..budget {
            if self.is_solved(&current) {
                break;
            }
            let (action, cost) = self.solve(&current, budget)?;
            path.push((action.clone(), cost));
            current = self.apply(&current, &action)?;
        }
        Some(path)
    }
}

// ---------------------------------------------------------------------------
// Solver helpers
// ---------------------------------------------------------------------------

fn remaining_cost(state: &CharacterRecognitionState) -> usize {
    let mut cost = 0;
    for i in 0..state.puzzle.num_pieces {
        if state.piece_correct(i) {
            continue;
        }
        let slot_idx = state
            .puzzle
            .slots
            .iter()
            .position(|s| s.matching_piece_idx == Some(i))
            .unwrap();
        let correct_rot = state.puzzle.slots[slot_idx].correct_rotation;
        let current_rot = state.rotations[i];
        let rot_cost = rotation_cost_90(current_rot, correct_rot);

        if state.placements[i].is_some() {
            cost += 1 + rot_cost + 1; // undo + rotate + place
        } else {
            cost += rot_cost + 1; // rotate + place
        }
    }
    if state.answer.as_ref() != Some(&state.puzzle.character) {
        cost += 1; // identify
    }
    cost
}

fn rotation_cost_90(current: u8, target: u8) -> usize {
    let diff = ((target as i8 - current as i8).rem_euclid(4)) as u8;
    // 0→0, 1→1, 2→2, 3→1 (CCW is shorter)
    diff.min(4 - diff) as usize
}

fn best_rotation_90(piece_id: usize, current: u8, target: u8) -> CharacterRecognitionAction {
    let diff_cw = ((target as i8 - current as i8).rem_euclid(4)) as u8;
    let diff_ccw = ((current as i8 - target as i8).rem_euclid(4)) as u8;

    if diff_cw <= diff_ccw {
        CharacterRecognitionAction::RotateCW {
            piece_id,
            angle: 90,
        }
    } else {
        CharacterRecognitionAction::RotateCCW {
            piece_id,
            angle: 90,
        }
    }
}

// ---------------------------------------------------------------------------
// Factory + registration
// ---------------------------------------------------------------------------

const SOLVE_BUDGET: usize = 50_000;

pub fn build_factory(
    name: &str,
    params: &[super::registry::ConfigParam],
    overrides: &std::collections::HashMap<String, super::registry::ConfigValue>,
) -> Result<Box<dyn crate::stream::loader::GameFactory>, String> {
    let (pieces_lo, pieces_hi) =
        super::registry::resolve_int_range(params, overrides, "num_pieces_range")?;
    let (grid_lo, grid_hi) =
        super::registry::resolve_int_range(params, overrides, "grid_size_range")?;
    let slot_grid_dim = super::registry::resolve_int(params, overrides, "slot_grid_dim")? as usize;

    let game = CharacterRecognition {
        num_pieces_range: (pieces_lo as usize, pieces_hi as usize),
        grid_size_range: (grid_lo as usize, grid_hi as usize),
        slot_grid_dim,
        solve_budget: SOLVE_BUDGET,
    };
    Ok(crate::stream::loader::game_factory(
        game,
        name,
        SOLVE_BUDGET,
    ))
}

inventory::submit! {
    super::registry::GameEntry {
        name: "character_recognition",
        params: &[
            super::registry::ConfigParam {
                key: "num_pieces_range",
                param_type: super::registry::ConfigParamType::IntRange,
                default: super::registry::ConfigValue::IntRange(3, 5),
            },
            super::registry::ConfigParam {
                key: "grid_size_range",
                param_type: super::registry::ConfigParamType::IntRange,
                default: super::registry::ConfigValue::IntRange(24, 30),
            },
            super::registry::ConfigParam {
                key: "slot_grid_dim",
                param_type: super::registry::ConfigParamType::Int,
                default: super::registry::ConfigValue::Int(4),
            },
        ],
        build: build_factory,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::episode::{Episode, GameEpisode};
    use rand::SeedableRng;

    fn make_game() -> CharacterRecognition {
        CharacterRecognition {
            num_pieces_range: (2, 2),
            grid_size_range: (24, 28),
            slot_grid_dim: 3,
            solve_budget: 50_000,
        }
    }

    fn make_state(game: &CharacterRecognition) -> CharacterRecognitionState {
        for seed in 0..200u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            if let Some(state) = game.generate(&mut rng) {
                return state;
            }
        }
        panic!("generation should succeed within 200 seeds")
    }

    fn make_episode(game: &CharacterRecognition) -> GameEpisode<CharacterRecognition> {
        let state = make_state(game);
        GameEpisode::new(Arc::new(make_game()), state, game.solve_budget)
    }

    #[test]
    fn state_text_reports_rotations_in_quarter_turns_and_slots() {
        let game = make_game();
        let mut state = make_state(&game);
        state.rotations = vec![1, 0];
        state.placements = vec![Some(3), None];
        let fields = game.state_fields(&state);
        assert_eq!(
            crate::base::state_text::format_state_json(&fields, 1),
            r#"{"r":[1,0],"p":[3,null],"n_placed":1,"moves":1}"#
        );
    }

    // --- Registry ---

    #[test]
    fn registered_in_registry() {
        assert!(crate::games::registry::find_game("character_recognition").is_some());
    }

    // --- Generation ---

    #[test]
    fn trivial_is_solvable() {
        let game = make_game();
        let state = game.trivial();
        let result = game.solve(&state, 50_000);
        assert!(result.is_some(), "trivial state should be solvable");
    }

    #[test]
    fn generate_produces_solvable() {
        let game = make_game();
        let mut generated = 0;
        for seed in 0..100 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            if let Some(state) = game.generate(&mut rng) {
                assert!(
                    !game.is_solved(&state),
                    "seed {seed}: generated state should not be already solved"
                );
                let result = game.solve(&state, 50_000);
                assert!(
                    result.is_some(),
                    "seed {seed}: generated state should be solvable"
                );
                generated += 1;
                if generated >= 20 {
                    break;
                }
            }
        }
        assert!(
            generated >= 20,
            "should generate at least 20 solvable puzzles, got {generated}"
        );
    }

    // --- Action parsing ---

    #[test]
    fn action_to_str_parse_roundtrip() {
        let game = make_game();
        let state = make_state(&game);
        let all_actions = game.actions(&state);
        for action in &all_actions {
            let s = game.action_to_str(action);
            let parsed = game
                .parse_action(&state, &s)
                .unwrap_or_else(|| panic!("failed to parse: {s}"));
            assert_eq!(*action, parsed, "roundtrip failed for: {s}");
        }
    }

    #[test]
    fn action_templates_expand_matches_actions() {
        let game = make_game();
        let state = make_state(&game);
        let templates = game.action_templates(&state);
        let expanded: Vec<String> = templates.iter().flat_map(|t| t.expand()).collect();
        for s in &expanded {
            assert!(
                game.parse_action(&state, s).is_some(),
                "template action should parse: {s}"
            );
        }
    }

    // --- Apply ---

    #[test]
    fn apply_rotate_unplaced() {
        let game = make_game();
        let state = make_state(&game);
        let rot_before = state.rotations[0];
        let new_state = game
            .apply(
                &state,
                &CharacterRecognitionAction::RotateCW {
                    piece_id: 0,
                    angle: 90,
                },
            )
            .expect("rotate should succeed");
        assert_eq!(new_state.rotations[0], (rot_before + 1) % 4);
    }

    #[test]
    fn apply_rotate_placed_fails() {
        let game = make_game();
        let mut state = make_state(&game);
        let (row, col) = state.puzzle.slots[0].id;
        state = game
            .apply(
                &state,
                &CharacterRecognitionAction::Place {
                    piece_id: 0,
                    row,
                    col,
                },
            )
            .unwrap();
        let result = game.apply(
            &state,
            &CharacterRecognitionAction::RotateCW {
                piece_id: 0,
                angle: 90,
            },
        );
        assert!(result.is_none(), "should not rotate placed piece");
    }

    #[test]
    fn apply_place_and_undo() {
        let game = make_game();
        let state = make_state(&game);
        let (row, col) = state.puzzle.slots[0].id;
        let placed = game
            .apply(
                &state,
                &CharacterRecognitionAction::Place {
                    piece_id: 0,
                    row,
                    col,
                },
            )
            .expect("place should succeed");
        assert!(placed.placements[0].is_some());

        let undone = game
            .apply(&placed, &CharacterRecognitionAction::Undo { piece_id: 0 })
            .expect("undo should succeed");
        assert!(undone.placements[0].is_none());
        assert_eq!(undone.rotations[0], placed.rotations[0]);
    }

    #[test]
    fn apply_place_occupied_slot_fails() {
        let game = make_game();
        let state = make_state(&game);
        let (row, col) = state.puzzle.slots[0].id;
        let placed = game
            .apply(
                &state,
                &CharacterRecognitionAction::Place {
                    piece_id: 0,
                    row,
                    col,
                },
            )
            .unwrap();
        let result = game.apply(
            &placed,
            &CharacterRecognitionAction::Place {
                piece_id: 1,
                row,
                col,
            },
        );
        assert!(result.is_none(), "should not place in occupied slot");
    }

    // --- Solved ---

    #[test]
    fn is_solved_all_correct_with_identify() {
        let game = make_game();
        let state = make_state(&game);
        let mut solved = state.clone();
        for i in 0..solved.puzzle.num_pieces {
            let slot_idx = solved
                .puzzle
                .slots
                .iter()
                .position(|s| s.matching_piece_idx == Some(i))
                .unwrap();
            solved.rotations[i] = solved.puzzle.slots[slot_idx].correct_rotation;
            solved.placements[i] = Some(slot_idx);
        }
        assert!(
            !game.is_solved(&solved),
            "should not be solved without identify"
        );
        solved.answer = Some(solved.puzzle.character.clone());
        assert!(game.is_solved(&solved));
    }

    #[test]
    fn is_solved_wrong_answer() {
        let game = make_game();
        let state = make_state(&game);
        let mut almost = state.clone();
        for i in 0..almost.puzzle.num_pieces {
            let slot_idx = almost
                .puzzle
                .slots
                .iter()
                .position(|s| s.matching_piece_idx == Some(i))
                .unwrap();
            almost.rotations[i] = almost.puzzle.slots[slot_idx].correct_rotation;
            almost.placements[i] = Some(slot_idx);
        }
        almost.answer = Some("?".to_string());
        assert!(!game.is_solved(&almost));
    }

    // --- Solver ---

    #[test]
    fn solver_produces_optimal_path() {
        let game = make_game();
        let state = make_state(&game);
        let path = game.solve_path(&state, 50_000).expect("should find path");
        let mut current = state.clone();
        for (action, _cost) in &path {
            current = game
                .apply(&current, action)
                .expect("solver action should be valid");
        }
        assert!(game.is_solved(&current), "should be solved after path");
    }

    #[test]
    fn solver_last_action_is_identify() {
        let game = make_game();
        let state = make_state(&game);
        let path = game.solve_path(&state, 50_000).expect("should find path");
        assert!(!path.is_empty());
        let (last_action, _) = path.last().unwrap();
        match last_action {
            CharacterRecognitionAction::Identify { character } => {
                assert_eq!(*character, state.puzzle.character);
            }
            _ => panic!("last action should be identify, got {:?}", last_action),
        }
    }

    // --- Episode integration ---

    #[test]
    fn episode_expert_action_done_when_solved() {
        let game = make_game();
        let mut state = make_state(&game);
        for i in 0..state.puzzle.num_pieces {
            let slot_idx = state
                .puzzle
                .slots
                .iter()
                .position(|s| s.matching_piece_idx == Some(i))
                .unwrap();
            state.rotations[i] = state.puzzle.slots[slot_idx].correct_rotation;
            state.placements[i] = Some(slot_idx);
        }
        state.answer = Some(state.puzzle.character.clone());
        let ep = GameEpisode::new(Arc::new(game), state, 50_000);
        assert_eq!(ep.expert_action(), "done");
    }

    #[test]
    fn episode_apply_undo_roundtrip() {
        let game = Arc::new(make_game());
        let state = make_state(&game);
        let mut ep = GameEpisode::new(game, state.clone(), 50_000);
        let (row, col) = state.puzzle.slots[0].id;
        let action_str = format!("place(0, {row}, {col})");
        assert!(ep.apply(&action_str));
        ep.undo();
        assert_eq!(ep.state, state);
    }

    #[test]
    fn episode_reset_restores_initial() {
        let game = Arc::new(make_game());
        let state = make_state(&game);
        let initial = state.clone();
        let mut ep = GameEpisode::new(game, state, 50_000);
        ep.apply("rotatecw(0, 90)");
        assert_ne!(ep.state, initial);
        ep.reset();
        assert_eq!(ep.state, initial);
    }

    #[test]
    fn episode_expert_action_returns_optimal() {
        let ep = make_episode(&make_game());
        let expert = ep.expert_action();
        assert_ne!(expert, "done", "should not be solved initially");
        let grid_dim = ep.state.puzzle.slot_grid_dim;
        let allowed = &ep.state.puzzle.allowed_chars;
        assert!(
            CharacterRecognitionAction::parse(
                &expert,
                ep.state.puzzle.num_pieces,
                grid_dim,
                allowed,
            )
            .is_some()
                || expert == "back"
                || expert == "reset",
            "expert action '{expert}' should be parseable"
        );
    }

    #[test]
    fn episode_expert_walk_to_solved() {
        let game = Arc::new(make_game());
        let state = make_state(&game);
        let mut ep = GameEpisode::new(game, state, 50_000);
        for _ in 0..100 {
            if ep.is_solved() {
                return;
            }
            let expert = ep.expert_action();
            if expert == "done" {
                break;
            }
            assert!(ep.apply(&expert), "expert action '{expert}' should succeed");
        }
        assert!(ep.is_solved(), "should be solved after expert walk");
    }

    #[test]
    fn episode_enumerate_includes_meta() {
        let ep = make_episode(&make_game());
        let actions = ep.enumerate_actions();
        assert!(actions.contains(&"back".to_string()));
        assert!(actions.contains(&"reset".to_string()));
        assert!(actions.contains(&"done".to_string()));
    }

    // --- Render ---

    #[test]
    fn render_dimensions() {
        let game = make_game();
        let state = make_state(&game);
        for res in [64, 128] {
            let buf = game.render(&state, res);
            assert_eq!(buf.width, res);
            assert_eq!(buf.height, res);
            assert_eq!(buf.data.len(), (res * res * 3) as usize);
        }
    }

    #[test]
    fn render_nonzero() {
        let game = make_game();
        let state = make_state(&game);
        let buf = game.render(&state, 128);
        let nonzero = buf.data.iter().filter(|&&b| b != 230).count();
        assert!(nonzero > 50, "render should have visual content");
    }

    // --- Rotation cost ---

    #[test]
    fn rotation_cost_90_examples() {
        assert_eq!(rotation_cost_90(0, 0), 0);
        assert_eq!(rotation_cost_90(0, 1), 1);
        assert_eq!(rotation_cost_90(0, 2), 2);
        assert_eq!(rotation_cost_90(0, 3), 1);
        assert_eq!(rotation_cost_90(3, 0), 1);
        assert_eq!(rotation_cost_90(1, 3), 2);
    }

    // --- Identify is accepted from any assembly state ---

    fn assemble_pieces(state: &mut CharacterRecognitionState) {
        for i in 0..state.puzzle.num_pieces {
            let slot_idx = state
                .puzzle
                .slots
                .iter()
                .position(|s| s.matching_piece_idx == Some(i))
                .unwrap();
            state.rotations[i] = state.puzzle.slots[slot_idx].correct_rotation;
            state.placements[i] = Some(slot_idx);
        }
    }

    fn wrong_char(state: &CharacterRecognitionState) -> String {
        state
            .puzzle
            .allowed_chars
            .iter()
            .find(|c| *c != &state.puzzle.character)
            .expect("allowed_chars must contain a distractor")
            .clone()
    }

    #[test]
    fn identify_accepted_with_zero_pieces_placed() {
        let game = make_game();
        let state = make_state(&game);
        assert!(!state.all_pieces_correctly_placed());
        let correct = state.puzzle.character.clone();
        let next = game
            .apply(
                &state,
                &CharacterRecognitionAction::Identify {
                    character: correct.clone(),
                },
            )
            .expect("identify should be accepted with no pieces placed");
        assert!(game.is_solved(&next));
    }

    #[test]
    fn identify_accepted_from_partial_assembly() {
        let game = make_game();
        let state = make_state(&game);
        // Place the first piece in any empty slot (does not assemble the
        // character), leaving the second piece unplaced.
        let mut partial = state.clone();
        partial.placements[0] = Some(0);
        assert!(!partial.all_pieces_correctly_placed());
        let correct = partial.puzzle.character.clone();
        let next = game
            .apply(
                &partial,
                &CharacterRecognitionAction::Identify { character: correct },
            )
            .expect("identify should be accepted from a partial assembly");
        assert!(game.is_solved(&next));
    }

    #[test]
    fn identify_accepted_from_assembled_state() {
        let game = make_game();
        let mut state = make_state(&game);
        assemble_pieces(&mut state);
        let correct = state.puzzle.character.clone();
        let next = game
            .apply(
                &state,
                &CharacterRecognitionAction::Identify { character: correct },
            )
            .expect("identify should succeed from assembled state");
        assert!(game.is_solved(&next));
    }

    #[test]
    fn wrong_identify_never_solves() {
        let game = make_game();
        let state = make_state(&game);
        let wrong = wrong_char(&state);

        // Unplaced state: identify accepted, but is_solved stays false (answer != character).
        let next_unplaced = game
            .apply(
                &state,
                &CharacterRecognitionAction::Identify {
                    character: wrong.clone(),
                },
            )
            .expect("wrong char from allowed_chars still applies (sets answer)");
        assert!(!game.is_solved(&next_unplaced));

        // Assembled state: same — wrong answer means not solved.
        let mut assembled = state.clone();
        assemble_pieces(&mut assembled);
        let next_assembled = game
            .apply(
                &assembled,
                &CharacterRecognitionAction::Identify { character: wrong },
            )
            .expect("apply should succeed but is_solved must be false");
        assert!(!game.is_solved(&next_assembled));
    }
}
