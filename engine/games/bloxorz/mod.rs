//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
pub mod gen;
pub mod object_boxes;
pub mod render;
pub mod state;

use rand_chacha::ChaChaRng;

use crate::base::game::{Game, RgbBuffer};
use crate::base::state_text::{json_numbers, json_string, StateField};
use crate::base::template::{ActionTemplate, ParamType};

use state::{BloxorzState, Direction, Orientation, ALL_DIRECTIONS};

/// Bloxorz game configuration.
pub struct Bloxorz {
    pub grid_size_range: (usize, usize),
    pub solve_budget: usize,
    pub fill_range: (usize, usize),
    pub num_fragile_tiles_range: (usize, usize),
}

impl Default for Bloxorz {
    fn default() -> Self {
        Bloxorz {
            grid_size_range: (6, 6),
            solve_budget: 100_000,
            fill_range: (50, 65),
            num_fragile_tiles_range: (0, 0),
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
    let (fill_lo, fill_hi) = super::registry::resolve_int_range(params, overrides, "fill_range")?;
    let (frag_lo, frag_hi) =
        super::registry::resolve_int_range(params, overrides, "num_fragile_tiles_range")?;
    let game = Bloxorz {
        grid_size_range: (gs_lo as usize, gs_hi as usize),
        solve_budget: SOLVE_BUDGET,
        fill_range: (fill_lo as usize, fill_hi as usize),
        num_fragile_tiles_range: (frag_lo as usize, frag_hi as usize),
    };
    Ok(crate::stream::loader::game_factory(
        game,
        name,
        SOLVE_BUDGET,
    ))
}

inventory::submit! {
    super::registry::GameEntry {
        name: "bloxorz",
        params: &[
            super::registry::ConfigParam {
                key: "grid_size_range",
                param_type: super::registry::ConfigParamType::IntRange,
                default: super::registry::ConfigValue::IntRange(6, 6),
            },
            super::registry::ConfigParam {
                key: "fill_range",
                param_type: super::registry::ConfigParamType::IntRange,
                default: super::registry::ConfigValue::IntRange(50, 65),
            },
            super::registry::ConfigParam {
                key: "num_fragile_tiles_range",
                param_type: super::registry::ConfigParamType::IntRange,
                default: super::registry::ConfigValue::IntRange(0, 0),
            },
        ],
        build: build_factory,
    }
}

impl Game for Bloxorz {
    type State = BloxorzState;
    type Action = Direction;

    fn generate(&self, rng: &mut ChaChaRng) -> Option<BloxorzState> {
        use rand::Rng;
        let grid_size = rng.gen_range(self.grid_size_range.0..=self.grid_size_range.1);
        let num_fragile =
            rng.gen_range(self.num_fragile_tiles_range.0..=self.num_fragile_tiles_range.1);
        gen::generate(
            rng,
            grid_size,
            self.solve_budget,
            self.fill_range,
            num_fragile,
        )
    }

    fn puzzle_fingerprint(&self, state: &BloxorzState) -> u64 {
        use std::hash::Hasher;
        // Identity = the immutable board (BloxorzGrid). The dynamic block
        // position (col/row/orientation/broken_mask) is excluded: it equals
        // `start` at generation and only changes mid-solve.
        let g = &*state.grid;
        crate::base::fingerprint::fingerprint(|h| {
            h.write_u64(g.width as u64);
            h.write_u64(g.height as u64);
            // Row-major tile map (walls/holes). Length-prefixed, index order.
            h.write_u64(g.cells.len() as u64);
            for &solid in &g.cells {
                h.write_u8(solid as u8);
            }
            // Goal.
            h.write_u64(g.goal.0 as u64);
            h.write_u64(g.goal.1 as u64);
            // Start (col, row, orientation).
            h.write_u64(g.start.0 as u64);
            h.write_u64(g.start.1 as u64);
            h.write_u8(match g.start.2 {
                Orientation::Standing => 0,
                Orientation::LyingX => 1,
                Orientation::LyingY => 2,
            });
            // Fragile tiles: order-independent set → sort before hashing.
            let mut fragile = g.fragile_tiles.clone();
            fragile.sort_unstable();
            h.write_u64(fragile.len() as u64);
            for (c, r) in fragile {
                h.write_u64(c as u64);
                h.write_u64(r as u64);
            }
        })
    }

    fn trivial(&self) -> BloxorzState {
        // Row of cells, goal at col 3, start standing at col 0. 2-move solve (right, right).
        let width = self.grid_size_range.0.max(6);
        let cells = vec![true; width];
        let grid = std::sync::Arc::new(state::BloxorzGrid {
            width,
            height: 1,
            cells,
            goal: (3, 0),
            start: (0, 0, Orientation::Standing),
            fragile_tiles: vec![],
        });
        BloxorzState {
            grid,
            col: 0,
            row: 0,
            orientation: Orientation::Standing,
            broken_mask: 0,
        }
    }

    fn actions(&self, _state: &BloxorzState) -> Vec<Direction> {
        ALL_DIRECTIONS.to_vec()
    }

    fn apply(&self, state: &BloxorzState, action: &Direction) -> Option<BloxorzState> {
        match state.apply_direction(*action) {
            Some(next) => Some(next),
            None => {
                // Fall off edge → reset to start
                let (col, row, orientation) = state.grid.start;
                Some(BloxorzState {
                    grid: state.grid.clone(),
                    col,
                    row,
                    orientation,
                    broken_mask: 0,
                })
            }
        }
    }

    fn is_solved(&self, state: &BloxorzState) -> bool {
        state.orientation == Orientation::Standing && (state.col, state.row) == state.grid.goal
    }

    fn render(&self, state: &BloxorzState, resolution: u32) -> RgbBuffer {
        render::render(state, resolution)
    }

    fn action_to_str(&self, action: &Direction) -> String {
        action.name().to_string()
    }

    fn parse_action(&self, _state: &BloxorzState, s: &str) -> Option<Direction> {
        match s {
            "up" => Some(Direction::Up),
            "down" => Some(Direction::Down),
            "left" => Some(Direction::Left),
            "right" => Some(Direction::Right),
            _ => None,
        }
    }

    fn action_templates(&self, _state: &BloxorzState) -> Vec<ActionTemplate> {
        vec![
            ActionTemplate {
                segments: vec![ParamType::Literal("up".into())],
            },
            ActionTemplate {
                segments: vec![ParamType::Literal("down".into())],
            },
            ActionTemplate {
                segments: vec![ParamType::Literal("left".into())],
            },
            ActionTemplate {
                segments: vec![ParamType::Literal("right".into())],
            },
        ]
    }

    fn prompt(&self, state: &BloxorzState) -> String {
        if state.grid.fragile_tiles.is_empty() {
            "Solve this Bloxorz puzzle. Roll the block so it stands on the red goal.".to_string()
        } else {
            "Solve this Bloxorz puzzle. Roll the block so it stands on the red goal. Striped tiles are fragile and will break after the block leaves them.".to_string()
        }
    }

    fn action_description(&self, _state: &BloxorzState) -> String {
        "up, down, left, right".to_string()
    }

    fn object_boxes(&self, state: &BloxorzState) -> Vec<crate::base::object_boxes::ObjectBox> {
        object_boxes::object_boxes(state)
    }

    /// `g`: the board without row breaks (`S` standing block, `=`/`-` block
    /// lying along X, `H`/`|` block lying along Y, `G` goal, `F` fragile tile,
    /// `.` tile, space for void); `b`: [col, row, "S"|"X"|"Y"]; `goal`:
    /// [col, row]; `broken`: broken fragile tiles; `distance`: Manhattan
    /// distance from the block to the goal.
    fn state_fields(&self, state: &BloxorzState) -> Vec<StateField> {
        let orientation = match state.orientation {
            Orientation::Standing => "S",
            Orientation::LyingX => "X",
            Orientation::LyingY => "Y",
        };
        let (goal_col, goal_row) = state.grid.goal;
        let distance = state.col.abs_diff(goal_col) + state.row.abs_diff(goal_row);
        vec![
            ("g", json_string(&state::to_ascii(state).replace('\n', ""))),
            (
                "b",
                format!("[{},{},{}]", state.col, state.row, json_string(orientation)),
            ),
            ("goal", json_numbers([goal_col, goal_row])),
            ("broken", state.broken_mask.count_ones().to_string()),
            ("distance", distance.to_string()),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::episode::{Episode, GameEpisode};
    use crate::games::bloxorz::state::from_ascii;
    use rand::SeedableRng;
    use std::sync::Arc;

    #[test]
    fn state_text_describes_block_goal_and_distance() {
        let state = from_ascii("....\n.=-.\n...G");
        let episode = GameEpisode::new(Arc::new(Bloxorz::default()), state, 100_000);
        assert_eq!(
            episode.state_text(2),
            r#"{"g":".....=-....G","b":[1,1,"X"],"goal":[3,2],"broken":0,"distance":3,"moves":2}"#
        );
    }

    #[test]
    fn apply_standing_roll() {
        let game = Bloxorz::default();

        // Standing at (2,3) on a 6x6 all-solid grid
        let state = from_ascii("......\n......\n......\n..S...\n......\n....G.");

        // Up: Standing(2,3) → LyingY(2,1)
        let up = game.apply(&state, &Direction::Up).unwrap();
        assert_eq!(
            (up.col, up.row, up.orientation),
            (2, 1, Orientation::LyingY)
        );

        // Down: Standing(2,3) → LyingY(2,4)
        let down = game.apply(&state, &Direction::Down).unwrap();
        assert_eq!(
            (down.col, down.row, down.orientation),
            (2, 4, Orientation::LyingY)
        );

        // Left: Standing(2,3) → LyingX(0,3)
        let left = game.apply(&state, &Direction::Left).unwrap();
        assert_eq!(
            (left.col, left.row, left.orientation),
            (0, 3, Orientation::LyingX)
        );

        // Right: Standing(2,3) → LyingX(3,3)
        let right = game.apply(&state, &Direction::Right).unwrap();
        assert_eq!(
            (right.col, right.row, right.orientation),
            (3, 3, Orientation::LyingX)
        );
    }

    #[test]
    fn apply_lying_x_roll() {
        let game = Bloxorz::default();

        // LyingX at (2,3) on a 6x6 all-solid grid
        let state = from_ascii("......\n......\n......\n..=-G.\n......\n......");

        // Up: LyingX(2,3) → LyingX(2,2)
        let up = game.apply(&state, &Direction::Up).unwrap();
        assert_eq!(
            (up.col, up.row, up.orientation),
            (2, 2, Orientation::LyingX)
        );

        // Down: LyingX(2,3) → LyingX(2,4)
        let down = game.apply(&state, &Direction::Down).unwrap();
        assert_eq!(
            (down.col, down.row, down.orientation),
            (2, 4, Orientation::LyingX)
        );

        // Left: LyingX(2,3) → Standing(1,3)
        let left = game.apply(&state, &Direction::Left).unwrap();
        assert_eq!(
            (left.col, left.row, left.orientation),
            (1, 3, Orientation::Standing)
        );

        // Right: LyingX(2,3) → Standing(4,3)
        let right = game.apply(&state, &Direction::Right).unwrap();
        assert_eq!(
            (right.col, right.row, right.orientation),
            (4, 3, Orientation::Standing)
        );
    }

    #[test]
    fn apply_lying_y_roll() {
        let game = Bloxorz::default();

        // LyingY at (2,2) on a 6x6 all-solid grid
        let state = from_ascii("......\n......\n..H...\n..|.G.\n......\n......");

        // Up: LyingY(2,2) → Standing(2,1)
        let up = game.apply(&state, &Direction::Up).unwrap();
        assert_eq!(
            (up.col, up.row, up.orientation),
            (2, 1, Orientation::Standing)
        );

        // Down: LyingY(2,2) → Standing(2,4)
        let down = game.apply(&state, &Direction::Down).unwrap();
        assert_eq!(
            (down.col, down.row, down.orientation),
            (2, 4, Orientation::Standing)
        );

        // Left: LyingY(2,2) → LyingY(1,2)
        let left = game.apply(&state, &Direction::Left).unwrap();
        assert_eq!(
            (left.col, left.row, left.orientation),
            (1, 2, Orientation::LyingY)
        );

        // Right: LyingY(2,2) → LyingY(3,2)
        let right = game.apply(&state, &Direction::Right).unwrap();
        assert_eq!(
            (right.col, right.row, right.orientation),
            (3, 2, Orientation::LyingY)
        );
    }

    #[test]
    fn apply_off_edge() {
        let game = Bloxorz::default();

        // Standing at (0,0) on a 3x3 grid
        let state = from_ascii("S..\n...\n..G");

        // Up and Left should reset to start (fall off edge)
        let up = game.apply(&state, &Direction::Up).unwrap();
        assert_eq!(
            (up.col, up.row, up.orientation),
            (state.col, state.row, state.orientation)
        );

        let left = game.apply(&state, &Direction::Left).unwrap();
        assert_eq!(
            (left.col, left.row, left.orientation),
            (state.col, state.row, state.orientation)
        );
    }

    #[test]
    fn apply_fall_resets_to_start() {
        let game = Bloxorz::default();

        // Standing at (0,0). Row 1 has void at col 0.
        // Roll down: Standing(0,0) → LyingY(0,1) occupies (0,1)[void] → fall → reset.
        let ascii = "S.G.\n ....\n....\n....";
        let state = from_ascii(ascii);
        assert_eq!(
            (state.col, state.row, state.orientation),
            (0, 0, Orientation::Standing)
        );

        let result = game.apply(&state, &Direction::Down).unwrap();
        assert_eq!(
            (result.col, result.row, result.orientation),
            (0, 0, Orientation::Standing)
        );
    }

    #[test]
    fn is_solved_positive() {
        let game = Bloxorz::default();
        // Standing on goal
        let state = BloxorzState {
            grid: Arc::new(crate::games::bloxorz::state::BloxorzGrid {
                width: 3,
                height: 3,
                cells: vec![true; 9],
                goal: (1, 1),
                start: (1, 1, Orientation::Standing),
                fragile_tiles: vec![],
            }),
            col: 1,
            row: 1,
            orientation: Orientation::Standing,
            broken_mask: 0,
        };
        assert!(game.is_solved(&state));
    }

    #[test]
    fn is_solved_negative_lying() {
        let game = Bloxorz::default();
        // LyingX on goal — not solved (must be standing)
        let state = BloxorzState {
            grid: Arc::new(crate::games::bloxorz::state::BloxorzGrid {
                width: 4,
                height: 3,
                cells: vec![true; 12],
                goal: (1, 1),
                start: (1, 1, Orientation::LyingX),
                fragile_tiles: vec![],
            }),
            col: 1,
            row: 1,
            orientation: Orientation::LyingX,
            broken_mask: 0,
        };
        assert!(!game.is_solved(&state));
    }

    #[test]
    fn is_solved_negative_wrong_cell() {
        let game = Bloxorz::default();
        // Standing but not on goal
        let state = BloxorzState {
            grid: Arc::new(crate::games::bloxorz::state::BloxorzGrid {
                width: 3,
                height: 3,
                cells: vec![true; 9],
                goal: (2, 2),
                start: (0, 0, Orientation::Standing),
                fragile_tiles: vec![],
            }),
            col: 0,
            row: 0,
            orientation: Orientation::Standing,
            broken_mask: 0,
        };
        assert!(!game.is_solved(&state));
    }

    #[test]
    fn action_to_str_parse_roundtrip() {
        let game = Bloxorz::default();
        let state = from_ascii("S..\n...\n..G");
        for dir in &ALL_DIRECTIONS {
            let s = game.action_to_str(dir);
            let parsed = game.parse_action(&state, &s).unwrap();
            assert_eq!(*dir, parsed);
        }
    }

    #[test]
    fn action_templates_no_meta() {
        let game = Bloxorz::default();
        let state = from_ascii("S..\n...\n..G");
        let templates = game.action_templates(&state);
        let mut actions: Vec<String> = templates.iter().flat_map(|t| t.expand()).collect();
        actions.sort();
        assert_eq!(actions, vec!["down", "left", "right", "up"]);
    }

    #[test]
    fn solve_trivial() {
        // Goal at (3,0), block standing at (0,0) on a 6x1 row:
        // Right: Standing(0,0) → LyingX(1,0)
        // Right: LyingX(1,0) → Standing(3,0), solved.
        let grid = Arc::new(crate::games::bloxorz::state::BloxorzGrid {
            width: 6,
            height: 1,
            cells: vec![true; 6],
            goal: (3, 0),
            start: (0, 0, Orientation::Standing),
            fragile_tiles: vec![],
        });
        let state = BloxorzState {
            grid,
            col: 0,
            row: 0,
            orientation: Orientation::Standing,
            broken_mask: 0,
        };
        let game = Bloxorz::default();
        let result = game.solve(&state, 10_000);
        assert!(result.is_some());
        let (first_action, cost) = result.unwrap();
        assert_eq!(first_action, Direction::Right);
        assert_eq!(cost, 2);
    }

    #[test]
    fn registered_in_registry() {
        assert!(crate::games::registry::find_game("bloxorz").is_some());
    }

    #[test]
    fn test_aspect_ratio_default() {
        let game = Bloxorz::default();
        assert_eq!(game.aspect_ratio(), (1, 1));
    }

    #[test]
    fn generate_produces_solvable() {
        let game = Bloxorz {
            grid_size_range: (6, 6),
            solve_budget: 100_000,
            fill_range: (50, 65),
            num_fragile_tiles_range: (0, 0),
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
        let game = Bloxorz::default();
        let state = game.trivial();
        let result = game.solve(&state, 100_000);
        assert!(result.is_some(), "trivial state should be solvable");
    }

    #[test]
    fn generate_large_succeeds() {
        let game = Bloxorz {
            grid_size_range: (20, 20),
            solve_budget: 100_000,
            fill_range: (50, 65),
            num_fragile_tiles_range: (0, 0),
        };
        for seed in 0..5u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            let result = game.generate(&mut rng);
            assert!(result.is_some(), "grid_size=20 seed={seed} should generate");
        }
    }

    // --- GameEpisode tests ---

    fn make_episode(state: BloxorzState) -> GameEpisode<Bloxorz> {
        let game = Arc::new(Bloxorz::default());
        GameEpisode::new(game, state, 100_000)
    }

    fn trivial_state() -> BloxorzState {
        // Standing at (0,0), goal at (3,0), on a 6x1 row
        BloxorzState {
            grid: Arc::new(crate::games::bloxorz::state::BloxorzGrid {
                width: 6,
                height: 1,
                cells: vec![true; 6],
                goal: (3, 0),
                start: (0, 0, Orientation::Standing),
                fragile_tiles: vec![],
            }),
            col: 0,
            row: 0,
            orientation: Orientation::Standing,
            broken_mask: 0,
        }
    }

    #[test]
    fn episode_apply_undo_roundtrip() {
        // A wide grid, so the block can move right
        let state = from_ascii("......\nS.....\n......\n....G.\n......\n......");
        let mut ep = make_episode(state);
        let original_col = ep.state.col;
        let original_row = ep.state.row;

        assert!(ep.apply("down"));
        assert_ne!(
            (ep.state.col, ep.state.row, ep.state.orientation),
            (original_col, original_row, Orientation::Standing)
        );

        ep.undo();
        assert_eq!(ep.state.col, original_col);
        assert_eq!(ep.state.row, original_row);
        assert_eq!(ep.state.orientation, Orientation::Standing);
    }

    #[test]
    fn episode_apply_invalid_returns_false() {
        // Standing at (0,0) in a 3x3 grid — can't go up or left (falls, resets to start)
        let state = from_ascii("S..\n...\n..G");
        let mut ep = make_episode(state);
        let orig_col = ep.state.col;
        let orig_row = ep.state.row;

        // "up" falls off edge → resets to start (same position), apply returns true
        assert!(ep.apply("up"));
        assert_eq!(ep.state.col, orig_col);
        assert_eq!(ep.state.row, orig_row);

        // Unknown action string → still false
        assert!(!ep.apply("fly"));
    }

    #[test]
    fn episode_reset_restores_initial() {
        let state = from_ascii("......\nS.....\n......\n....G.\n......\n......");
        let mut ep = make_episode(state);
        let initial_col = ep.state.col;
        let initial_row = ep.state.row;

        ep.apply("down");
        ep.apply("right");
        assert_ne!(ep.state.col, initial_col);

        ep.reset();
        assert_eq!(ep.state.col, initial_col);
        assert_eq!(ep.state.row, initial_row);
        assert!(ep.history.is_empty());
    }

    #[test]
    fn episode_expert_action_returns_optimal() {
        // Standing at (0,0), goal at (3,0) on 6x1 row → must go right
        let state = trivial_state();
        let ep = make_episode(state);
        let action = ep.expert_action();
        assert_eq!(action, "right");
    }

    #[test]
    fn episode_expert_action_done_when_solved() {
        // Standing on goal
        let state = BloxorzState {
            grid: Arc::new(crate::games::bloxorz::state::BloxorzGrid {
                width: 3,
                height: 3,
                cells: vec![true; 9],
                goal: (1, 1),
                start: (1, 1, Orientation::Standing),
                fragile_tiles: vec![],
            }),
            col: 1,
            row: 1,
            orientation: Orientation::Standing,
            broken_mask: 0,
        };
        let ep = make_episode(state);
        assert_eq!(ep.expert_action(), "done");
    }

    #[test]
    fn episode_cache_consistency() {
        let state = trivial_state();
        let game = Arc::new(Bloxorz::default());
        let ep = GameEpisode::new(game.clone(), state.clone(), 100_000);

        let cached_cost = ep.cached_solve(&state);
        let fresh = game.solve(&state, 100_000).map(|(_, c)| c);
        assert_eq!(cached_cost, fresh);
    }

    #[test]
    fn episode_enumerate_includes_meta() {
        let state = trivial_state();
        let ep = make_episode(state);
        let actions = ep.enumerate_actions();
        assert!(actions.contains(&"up".to_string()));
        assert!(actions.contains(&"down".to_string()));
        assert!(actions.contains(&"left".to_string()));
        assert!(actions.contains(&"right".to_string()));
        assert!(actions.contains(&"back".to_string()));
        assert!(actions.contains(&"reset".to_string()));
        assert!(actions.contains(&"done".to_string()));
        assert_eq!(actions.len(), 7); // 4 game + 3 meta
    }

    // --- Fragile tile tests ---

    #[test]
    fn apply_fall_resets_broken_mask() {
        let game = Bloxorz::default();
        // 4x1 row with fragile at (1,0), standing at (0,0)
        let grid = Arc::new(crate::games::bloxorz::state::BloxorzGrid {
            width: 4,
            height: 1,
            cells: vec![true; 4],
            goal: (3, 0),
            start: (0, 0, Orientation::Standing),
            fragile_tiles: vec![(1, 0)],
        });
        let state = BloxorzState {
            grid,
            col: 0,
            row: 0,
            orientation: Orientation::Standing,
            broken_mask: 1, // pre-broken
        };
        // Move up from row 0 → fall → reset
        let result = game.apply(&state, &Direction::Up).unwrap();
        assert_eq!(result.broken_mask, 0, "Reset should clear broken_mask");
        assert_eq!(result.col, 0);
    }

    #[test]
    fn generate_with_fragile_solvable() {
        let game = Bloxorz {
            grid_size_range: (6, 6),
            solve_budget: 100_000,
            fill_range: (50, 65),
            num_fragile_tiles_range: (2, 3),
        };
        let mut generated = 0;
        for seed in 0..20u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            if let Some(state) = game.generate(&mut rng) {
                let result = game.solve(&state, 100_000);
                assert!(
                    result.is_some(),
                    "Seed {} produced unsolvable fragile puzzle",
                    seed
                );
                generated += 1;
            }
        }
        assert!(
            generated >= 5,
            "Should generate at least some fragile puzzles"
        );
    }

    #[test]
    fn fragile_tiles_avoid_start_and_goal() {
        let game = Bloxorz {
            grid_size_range: (6, 6),
            solve_budget: 100_000,
            fill_range: (50, 65),
            num_fragile_tiles_range: (3, 3),
        };
        for seed in 0..10u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            if let Some(state) = game.generate(&mut rng) {
                let (sc, sr, so) = state.grid.start;
                let mut start_cells = vec![(sc, sr)];
                match so {
                    Orientation::LyingX => start_cells.push((sc + 1, sr)),
                    Orientation::LyingY => start_cells.push((sc, sr + 1)),
                    Orientation::Standing => {}
                }
                for &fragile in &state.grid.fragile_tiles {
                    assert_ne!(fragile, state.grid.goal, "Fragile should not be on goal");
                    assert!(
                        !start_cells.contains(&fragile),
                        "Fragile should not be on start cells"
                    );
                }
            }
        }
    }
}
