//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
pub mod gen;
pub mod object_boxes;
pub mod render;
pub mod solver;
pub mod state;

use rand_chacha::ChaChaRng;

use crate::base::game::{Game, RgbBuffer};
use crate::base::state_text::{json_string, json_strings, StateField};
use crate::base::template::{ActionTemplate, ParamType};

use state::{Orientation, RushHourState, Slide};

/// Rush Hour game configuration.
pub struct RushHour {
    pub grid_size_range: (u8, u8),
    pub num_cars_range: (u8, u8),
    pub solve_budget: usize,
}

#[cfg(test)]
impl Default for RushHour {
    fn default() -> Self {
        RushHour {
            grid_size_range: (6, 6),
            num_cars_range: (6, 12),
            solve_budget: 100_000,
        }
    }
}

const SOLVE_BUDGET: usize = 100_000;

pub fn build_factory(
    name: &str,
    params: &[super::registry::ConfigParam],
    overrides: &std::collections::HashMap<String, super::registry::ConfigValue>,
) -> Result<Box<dyn crate::stream::loader::GameFactory>, String> {
    let (gs_lo, gs_hi) = super::registry::resolve_int_range(params, overrides, "grid_size_range")?;
    let (lo, hi) = super::registry::resolve_int_range(params, overrides, "num_cars_range")?;
    let game = RushHour {
        grid_size_range: (gs_lo as u8, gs_hi as u8),
        num_cars_range: (lo as u8, hi as u8),
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
        name: "rush_hour",
        params: &[
            super::registry::ConfigParam {
                key: "grid_size_range",
                param_type: super::registry::ConfigParamType::IntRange,
                default: super::registry::ConfigValue::IntRange(6, 6),
            },
            super::registry::ConfigParam {
                key: "num_cars_range",
                param_type: super::registry::ConfigParamType::IntRange,
                default: super::registry::ConfigValue::IntRange(6, 12),
            },
        ],
        build: build_factory,
    }
}

impl Game for RushHour {
    type State = RushHourState;
    type Action = Slide;

    fn generate(&self, rng: &mut ChaChaRng) -> Option<RushHourState> {
        use rand::Rng;
        let grid_size = rng.gen_range(self.grid_size_range.0..=self.grid_size_range.1);
        gen::generate(rng, grid_size, self.num_cars_range, self.solve_budget)
    }

    /// Raw, color-inclusive puzzle fingerprint.
    ///
    /// Order: grid_size, exit_row, vehicle count, then per vehicle in stored
    /// (id-ascending) order `(id as u32, orientation as u8, length, fixed_axis,
    /// positions[i])`. Vehicle `id` is included because it determines the
    /// rendered color (render::vehicle_color). The stored order is canonical
    /// (generation assigns ids A, B, C… in placement order), so no re-sorting.
    fn puzzle_fingerprint(&self, state: &RushHourState) -> u64 {
        use std::hash::Hasher;
        let config = &*state.config;
        crate::base::fingerprint::fingerprint(|h| {
            h.write_u64(config.grid_size as u64);
            h.write_u64(config.exit_row as u64);
            h.write_u64(config.vehicles.len() as u64);
            for (idx, def) in config.vehicles.iter().enumerate() {
                let orientation_u8: u8 = match def.orientation {
                    Orientation::Horizontal => 0,
                    Orientation::Vertical => 1,
                };
                h.write_u64(def.id as u32 as u64);
                h.write_u64(orientation_u8 as u64);
                h.write_u64(def.length as u64);
                h.write_u64(def.fixed_axis as u64);
                h.write_u64(state.positions[idx] as u64);
            }
        })
    }

    fn trivial(&self) -> RushHourState {
        // Just car 'A' on exit_row at col 0. Slide right to exit. 1-slide solve.
        let gs = self.grid_size_range.0;
        let config = std::sync::Arc::new(state::BoardConfig {
            grid_size: gs,
            vehicles: vec![state::VehicleDef {
                id: 'A',
                orientation: Orientation::Horizontal,
                length: 2,
                fixed_axis: 2,
            }],
            exit_row: 2,
        });
        RushHourState {
            config,
            positions: vec![0],
        }
    }

    fn actions(&self, state: &RushHourState) -> Vec<Slide> {
        let gs = state.config.grid_size as usize;
        let occupancy = state.occupancy_grid();
        let mut result = Vec::new();

        for (idx, def) in state.config.vehicles.iter().enumerate() {
            let pos = state.positions[idx] as usize;
            let len = def.length as usize;

            match def.orientation {
                Orientation::Horizontal => {
                    let row = def.fixed_axis as usize;
                    // Scan left
                    let mut dist = 1i8;
                    let mut check = pos as i32 - 1;
                    while check >= 0 {
                        if occupancy[row * gs + check as usize].is_some() {
                            break;
                        }
                        result.push(Slide {
                            vehicle_id: def.id,
                            amount: -dist,
                        });
                        dist += 1;
                        check -= 1;
                    }
                    // Scan right
                    let mut dist = 1i8;
                    let mut check_col = pos + len;
                    while check_col < gs {
                        if occupancy[row * gs + check_col].is_some() {
                            break;
                        }
                        result.push(Slide {
                            vehicle_id: def.id,
                            amount: dist,
                        });
                        dist += 1;
                        check_col += 1;
                    }
                }
                Orientation::Vertical => {
                    let col = def.fixed_axis as usize;
                    // Scan up
                    let mut dist = 1i8;
                    let mut check = pos as i32 - 1;
                    while check >= 0 {
                        if occupancy[check as usize * gs + col].is_some() {
                            break;
                        }
                        result.push(Slide {
                            vehicle_id: def.id,
                            amount: -dist,
                        });
                        dist += 1;
                        check -= 1;
                    }
                    // Scan down
                    let mut dist = 1i8;
                    let mut check_row = pos + len;
                    while check_row < gs {
                        if occupancy[check_row * gs + col].is_some() {
                            break;
                        }
                        result.push(Slide {
                            vehicle_id: def.id,
                            amount: dist,
                        });
                        dist += 1;
                        check_row += 1;
                    }
                }
            }
        }

        result
    }

    fn apply(&self, state: &RushHourState, action: &Slide) -> Option<RushHourState> {
        if action.amount == 0 {
            return None;
        }

        let idx = state.config.vehicle_index(action.vehicle_id)?;
        let def = &state.config.vehicles[idx];
        let gs = state.config.grid_size as i32;
        let old_pos = state.positions[idx] as i32;
        let new_pos = old_pos + action.amount as i32;
        let length = def.length as i32;

        // Bounds check
        if new_pos < 0 || new_pos + length > gs {
            return None;
        }

        // Collision check on swept cells
        let occupancy = state.occupancy_grid();
        let gs_usize = state.config.grid_size as usize;

        let (sweep_start, sweep_end) = if action.amount > 0 {
            ((old_pos + length) as usize, (new_pos + length - 1) as usize)
        } else {
            (new_pos as usize, (old_pos - 1) as usize)
        };

        for cell in sweep_start..=sweep_end {
            let grid_idx = match def.orientation {
                Orientation::Horizontal => def.fixed_axis as usize * gs_usize + cell,
                Orientation::Vertical => cell * gs_usize + def.fixed_axis as usize,
            };
            if occupancy[grid_idx].is_some() {
                return None;
            }
        }

        let mut new_positions = state.positions.clone();
        new_positions[idx] = new_pos as u8;

        Some(RushHourState {
            config: state.config.clone(),
            positions: new_positions,
        })
    }

    fn is_solved(&self, state: &RushHourState) -> bool {
        state.positions[0] + state.config.vehicles[0].length == state.config.grid_size
    }

    fn render(&self, state: &RushHourState, resolution: u32) -> RgbBuffer {
        render::render(state, resolution)
    }

    fn action_to_str(&self, action: &Slide) -> String {
        format!("slide('{}', {})", action.vehicle_id, action.amount)
    }

    fn parse_action(&self, _state: &RushHourState, s: &str) -> Option<Slide> {
        let s = s.strip_prefix("slide('")?;
        let mut parts = s.splitn(2, "', ");
        let vehicle_id = parts.next()?.chars().next()?;
        let rest = parts.next()?;
        let rest = rest.strip_suffix(')')?;
        let amount: i8 = rest.parse().ok()?;
        if amount == 0 {
            return None;
        }
        Some(Slide { vehicle_id, amount })
    }

    fn action_templates(&self, state: &RushHourState) -> Vec<ActionTemplate> {
        let gs = state.config.grid_size as i32;
        let vehicle_ids: Vec<String> = state
            .config
            .vehicles
            .iter()
            .map(|v| v.id.to_string())
            .collect();
        vec![ActionTemplate {
            segments: vec![
                ParamType::Literal("slide('".into()),
                ParamType::OneOf(vehicle_ids),
                ParamType::Literal("', ".into()),
                ParamType::Int {
                    min: -(gs - 1),
                    max: gs - 1,
                },
                ParamType::Literal(")".into()),
            ],
        }]
    }

    fn prompt(&self, _state: &RushHourState) -> String {
        "Slide vehicles to free the red car (A) and move it to the right edge exit.".to_string()
    }

    fn action_description(&self, _state: &RushHourState) -> String {
        "slide('vehicle', distance) where distance is positive (right/down) or negative (left/up)"
            .to_string()
    }

    fn object_boxes(&self, state: &RushHourState) -> Vec<crate::base::object_boxes::ObjectBox> {
        object_boxes::object_boxes(state)
    }

    /// `g`: vehicle letters at their cells (`.` empty), row-major without row
    /// breaks; `exit_row`, `exit_col`; `n_vehicles`; `blocks_A`: vehicles
    /// between car A and the exit, nearest first.
    fn state_fields(&self, state: &RushHourState) -> Vec<StateField> {
        let config = &state.config;
        let grid_size = config.grid_size as usize;
        let occupancy = state.occupancy_grid();
        let grid: String = occupancy
            .iter()
            .map(|cell| cell.map_or('.', |index| config.vehicles[index].id))
            .collect();
        let exit_row = config.exit_row as usize;
        let target_front = state.positions[0] as usize + config.vehicles[0].length as usize;
        let mut blockers: Vec<String> = Vec::new();
        for col in target_front..grid_size {
            if let Some(index) = occupancy[exit_row * grid_size + col] {
                let id = config.vehicles[index].id.to_string();
                if !blockers.contains(&id) {
                    blockers.push(id);
                }
            }
        }
        vec![
            ("g", json_string(&grid)),
            ("exit_row", exit_row.to_string()),
            ("exit_col", (grid_size - 1).to_string()),
            ("n_vehicles", config.vehicles.len().to_string()),
            ("blocks_A", json_strings(&blockers)),
        ]
    }

    fn solve(&self, state: &RushHourState, budget: usize) -> Option<(Slide, usize)> {
        solver::iddfs_solve(&state.config, &state.positions, budget)
            .map(|r| (r.first_action().clone(), r.cost()))
    }

    fn solve_path(&self, state: &RushHourState, budget: usize) -> Option<Vec<(Slide, usize)>> {
        solver::iddfs_solve(&state.config, &state.positions, budget).map(|r| {
            let n = r.actions.len();
            r.actions
                .into_iter()
                .enumerate()
                .map(|(i, a)| (a, n - i))
                .collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::episode::{Episode, GameEpisode};
    use crate::games::rush_hour::state::from_ascii;
    use rand::SeedableRng;
    use std::sync::Arc;

    #[test]
    fn state_text_lists_blockers_of_the_target_car() {
        let state = from_ascii("......\n.....B\nAA.C.B\n...C..\n......\n......");
        let episode = GameEpisode::new(Arc::new(RushHour::default()), state, 100_000);
        assert_eq!(
            episode.state_text(1),
            concat!(
                r#"{"g":"...........BAA.C.B...C..............","exit_row":2,"exit_col":5,"#,
                r#""n_vehicles":3,"blocks_A":["C","B"],"moves":1}"#
            )
        );
    }

    // --- Game logic tests ---

    #[test]
    fn apply_simple_slide() {
        let state = from_ascii("......\n......\nAA....\n......\n......\n......");
        let game = RushHour::default();
        let new = game
            .apply(
                &state,
                &Slide {
                    vehicle_id: 'A',
                    amount: 1,
                },
            )
            .unwrap();
        assert_eq!(new.positions[0], 1);
    }

    #[test]
    fn apply_slide_multiple() {
        let state = from_ascii("......\n......\nAA....\n......\n......\n......");
        let game = RushHour::default();
        let new = game
            .apply(
                &state,
                &Slide {
                    vehicle_id: 'A',
                    amount: 3,
                },
            )
            .unwrap();
        assert_eq!(new.positions[0], 3);
    }

    #[test]
    fn apply_blocked_by_vehicle() {
        let state = from_ascii("......\n......\nAABB..\n......\n......\n......");
        let game = RushHour::default();
        assert!(game
            .apply(
                &state,
                &Slide {
                    vehicle_id: 'A',
                    amount: 1
                }
            )
            .is_none());
    }

    #[test]
    fn apply_blocked_by_wall() {
        let state = from_ascii("......\n......\n...AA.\n......\n......\n......");
        let game = RushHour::default();
        // Slide right by 2 would put A at pos 5, needing col 5 and 6 — col 6 is out of bounds
        assert!(game
            .apply(
                &state,
                &Slide {
                    vehicle_id: 'A',
                    amount: 2
                }
            )
            .is_none());
    }

    #[test]
    fn apply_invalid_amount_zero() {
        let state = from_ascii("......\n......\nAA....\n......\n......\n......");
        let game = RushHour::default();
        assert!(game
            .apply(
                &state,
                &Slide {
                    vehicle_id: 'A',
                    amount: 0
                }
            )
            .is_none());
    }

    #[test]
    fn apply_unknown_vehicle() {
        let state = from_ascii("......\n......\nAA....\n......\n......\n......");
        let game = RushHour::default();
        assert!(game
            .apply(
                &state,
                &Slide {
                    vehicle_id: 'Z',
                    amount: 1
                }
            )
            .is_none());
    }

    #[test]
    fn is_solved_positive() {
        let state = from_ascii("......\n......\n....AA\n......\n......\n......");
        let game = RushHour::default();
        assert!(game.is_solved(&state));
    }

    #[test]
    fn is_solved_negative() {
        let state = from_ascii("......\n......\nAA....\n......\n......\n......");
        let game = RushHour::default();
        assert!(!game.is_solved(&state));
    }

    #[test]
    fn action_to_str_parse_roundtrip() {
        let state = from_ascii("..BCCC\n..B...\nAAB..D\n.EE..D\n......\n......");
        let game = RushHour::default();
        let actions = game.actions(&state);
        assert!(!actions.is_empty());
        for action in &actions {
            let s = game.action_to_str(action);
            let parsed = game.parse_action(&state, &s).unwrap();
            assert_eq!(action.vehicle_id, parsed.vehicle_id);
            assert_eq!(action.amount, parsed.amount);
        }
    }

    #[test]
    fn action_templates_expansion() {
        let state = from_ascii("..BCCC\n..B...\nAAB..D\n.EE..D\n......\n......");
        let game = RushHour::default();
        let templates = game.action_templates(&state);
        let expanded: Vec<String> = templates.iter().flat_map(|t| t.expand()).collect();
        // All valid actions should appear in the expansion
        let actions = game.actions(&state);
        for action in &actions {
            let s = game.action_to_str(action);
            assert!(
                expanded.contains(&s),
                "Valid action '{}' not in template expansion",
                s
            );
        }
    }

    #[test]
    fn solve_trivial() {
        // A needs to slide right by 1 to reach the exit
        let state = from_ascii("......\n......\n...AA.\n......\n......\n......");
        let game = RushHour::default();
        let result = game.solve(&state, 10_000);
        assert!(result.is_some());
        let (first_action, cost) = result.unwrap();
        assert_eq!(first_action.vehicle_id, 'A');
        assert_eq!(first_action.amount, 1);
        assert_eq!(cost, 1);
    }

    #[test]
    fn registered_in_registry() {
        assert!(crate::games::registry::find_game("rush_hour").is_some());
    }

    #[test]
    fn test_aspect_ratio_default() {
        let game = RushHour::default();
        assert_eq!(game.aspect_ratio(), (1, 1));
    }

    #[test]
    fn generate_produces_solvable() {
        let game = RushHour {
            grid_size_range: (6, 6),
            num_cars_range: (6, 12),
            solve_budget: 100_000,
        };
        for seed in 0..10u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            let state = game.generate(&mut rng).expect("test seed should generate");
            let result = game.solve(&state, 100_000);
            assert!(result.is_some(), "Seed {} produced unsolvable puzzle", seed);
        }
    }

    #[test]
    fn trivial_is_solvable() {
        let game = RushHour::default();
        let state = game.trivial();
        let result = game.solve(&state, 100_000);
        assert!(result.is_some(), "trivial state should be solvable");
    }

    // --- GameEpisode integration tests ---

    fn make_episode(ascii: &str) -> GameEpisode<RushHour> {
        let state = from_ascii(ascii);
        let game = Arc::new(RushHour::default());
        GameEpisode::new(game, state, 100_000)
    }

    #[test]
    fn episode_apply_undo_roundtrip() {
        let mut ep = make_episode("......\n......\nAA....\n......\n......\n......");
        let original_pos = ep.state.positions[0];

        assert!(ep.apply("slide('A', 1)"));
        assert_ne!(ep.state.positions[0], original_pos);

        ep.undo();
        assert_eq!(ep.state.positions[0], original_pos);
    }

    #[test]
    fn episode_apply_invalid_returns_false() {
        let mut ep = make_episode("......\n......\nAABB..\n......\n......\n......");
        let original_pos = ep.state.positions[0];

        // Blocked by vehicle B
        assert!(!ep.apply("slide('A', 1)"));
        assert_eq!(ep.state.positions[0], original_pos);

        // Nonsense action
        assert!(!ep.apply("fly"));
    }

    #[test]
    fn episode_reset_restores_initial() {
        let mut ep = make_episode("......\n......\nAA....\n......\n......\n......");
        let initial_pos = ep.state.positions[0];

        ep.apply("slide('A', 1)");
        ep.apply("slide('A', 1)");
        assert_ne!(ep.state.positions[0], initial_pos);

        ep.reset();
        assert_eq!(ep.state.positions[0], initial_pos);
        assert!(ep.history.is_empty());
    }

    #[test]
    fn episode_expert_action_returns_optimal() {
        // A needs 1 move right to solve
        let ep = make_episode("......\n......\n...AA.\n......\n......\n......");
        let action = ep.expert_action();
        assert_eq!(action, "slide('A', 1)");
    }

    #[test]
    fn episode_expert_action_done_when_solved() {
        let ep = make_episode("......\n......\n....AA\n......\n......\n......");
        assert_eq!(ep.expert_action(), "done");
    }

    #[test]
    fn episode_enumerate_includes_meta() {
        let ep = make_episode("......\n......\n.AA...\n......\n......\n......");
        let actions = ep.enumerate_actions();
        assert!(actions.contains(&"slide('A', 1)".to_string()));
        assert!(actions.contains(&"slide('A', -1)".to_string()));
        assert!(actions.contains(&"back".to_string()));
        assert!(actions.contains(&"reset".to_string()));
        assert!(actions.contains(&"done".to_string()));
    }

    #[test]
    fn episode_cache_consistency() {
        let state = from_ascii("......\n......\n.AA...\n......\n......\n......");
        let game = Arc::new(RushHour::default());
        let ep = GameEpisode::new(game.clone(), state.clone(), 100_000);

        let cached_cost = ep.cached_solve(&state);
        let fresh = game.solve(&state, 100_000).map(|(_, c)| c);
        assert_eq!(cached_cost, fresh);
    }
}
