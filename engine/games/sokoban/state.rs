//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Cell types in the grid (static terrain).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Cell {
    Floor = 0,
    Wall = 1,
    Goal = 2,
}

/// Direction of movement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

impl Direction {
    pub fn dr(&self) -> i32 {
        match self {
            Direction::Up => -1,
            Direction::Down => 1,
            Direction::Left => 0,
            Direction::Right => 0,
        }
    }

    pub fn dc(&self) -> i32 {
        match self {
            Direction::Up => 0,
            Direction::Down => 0,
            Direction::Left => -1,
            Direction::Right => 1,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Direction::Up => "up",
            Direction::Down => "down",
            Direction::Left => "left",
            Direction::Right => "right",
        }
    }
}

pub const ALL_DIRECTIONS: [Direction; 4] = [
    Direction::Up,
    Direction::Down,
    Direction::Left,
    Direction::Right,
];

/// Board layout — the static part of the board (shared across states via Arc).
/// Uses a flat row-major Vec<Cell> for cache locality.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoardLayout {
    pub width: usize,
    pub height: usize,
    /// Flat row-major grid: grid[row * width + col]
    pub grid: Vec<Cell>,
    /// Goal positions as sorted (col, row) pairs.
    pub goal_positions: Vec<(usize, usize)>,
}

impl BoardLayout {
    pub fn in_bounds(&self, col: i32, row: i32) -> bool {
        col >= 0 && row >= 0 && (col as usize) < self.width && (row as usize) < self.height
    }

    pub fn cell_at(&self, col: usize, row: usize) -> Cell {
        self.grid[row * self.width + col]
    }

    pub fn is_goal(&self, col: usize, row: usize) -> bool {
        self.goal_positions.contains(&(col, row))
    }
}

/// Sokoban game state — player position + box positions.
///
/// `box_positions` is always kept sorted for deterministic Hash/Eq.
/// Layout is shared via Arc across states derived from the same puzzle.
#[derive(Debug, Clone)]
pub struct SokobanState {
    pub layout: Arc<BoardLayout>,
    pub player_col: usize,
    pub player_row: usize,
    /// Sorted (col, row) pairs — always sorted for deterministic Hash/Eq.
    pub box_positions: Vec<(usize, usize)>,
}

impl SokobanState {
    pub fn new(
        layout: Arc<BoardLayout>,
        player_col: usize,
        player_row: usize,
        mut box_positions: Vec<(usize, usize)>,
    ) -> Self {
        box_positions.sort();
        SokobanState {
            layout,
            player_col,
            player_row,
            box_positions,
        }
    }

    pub fn has_box(&self, col: usize, row: usize) -> bool {
        self.box_positions.contains(&(col, row))
    }
}

impl PartialEq for SokobanState {
    fn eq(&self, other: &Self) -> bool {
        self.player_col == other.player_col
            && self.player_row == other.player_row
            && self.box_positions == other.box_positions
    }
}

impl Eq for SokobanState {}

impl Hash for SokobanState {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.player_col.hash(hasher);
        self.player_row.hash(hasher);
        self.box_positions.hash(hasher);
    }
}

// --- ASCII helpers (for tests) ---

const CHAR_WALL: char = '#';
const CHAR_FLOOR: char = '.';
const CHAR_PLAYER: char = '@';
const CHAR_BOX: char = 'O';
const CHAR_GOAL: char = 'X';
const CHAR_BOX_ON_GOAL: char = '*';
const CHAR_PLAYER_ON_GOAL: char = '+';

/// Convert state to ASCII string.
pub fn to_ascii(state: &SokobanState) -> String {
    let mut lines = Vec::with_capacity(state.layout.height);
    for row in 0..state.layout.height {
        let mut chars = String::with_capacity(state.layout.width);
        for col in 0..state.layout.width {
            let is_player = col == state.player_col && row == state.player_row;
            let is_box = state.has_box(col, row);
            let is_goal = state.layout.is_goal(col, row);

            let ch = if is_player && is_goal {
                CHAR_PLAYER_ON_GOAL
            } else if is_player {
                CHAR_PLAYER
            } else if is_box && is_goal {
                CHAR_BOX_ON_GOAL
            } else if is_box {
                CHAR_BOX
            } else if is_goal {
                CHAR_GOAL
            } else if state.layout.cell_at(col, row) == Cell::Wall {
                CHAR_WALL
            } else {
                CHAR_FLOOR
            };
            chars.push(ch);
        }
        lines.push(chars);
    }
    lines.join("\n")
}

#[cfg(test)]
/// Parse an ASCII board into a SokobanState.
pub fn from_ascii(text: &str) -> SokobanState {
    let raw_lines: Vec<&str> = text.trim().split('\n').collect();
    let max_width = raw_lines.iter().map(|l| l.len()).max().unwrap_or(0);

    let mut grid = Vec::new();
    let mut player_pos: Option<(usize, usize)> = None;
    let mut box_positions = Vec::new();
    let mut goal_positions = Vec::new();

    for (row_idx, line) in raw_lines.iter().enumerate() {
        let chars: Vec<char> = {
            let mut c: Vec<char> = line.chars().collect();
            while c.len() < max_width {
                c.push('#');
            }
            c
        };

        for (col_idx, &ch) in chars.iter().enumerate() {
            let pos = (col_idx, row_idx);
            match ch {
                '#' => grid.push(Cell::Wall),
                '.' => grid.push(Cell::Floor),
                '@' => {
                    grid.push(Cell::Floor);
                    player_pos = Some(pos);
                }
                'O' => {
                    grid.push(Cell::Floor);
                    box_positions.push(pos);
                }
                'X' => {
                    grid.push(Cell::Goal);
                    goal_positions.push(pos);
                }
                '*' => {
                    grid.push(Cell::Goal);
                    box_positions.push(pos);
                    goal_positions.push(pos);
                }
                '+' => {
                    grid.push(Cell::Goal);
                    player_pos = Some(pos);
                    goal_positions.push(pos);
                }
                _ => grid.push(Cell::Wall),
            }
        }
    }

    let (player_col, player_row) = player_pos.expect("No player (@) found in ASCII board");
    let height = raw_lines.len();

    goal_positions.sort();
    let layout = Arc::new(BoardLayout {
        width: max_width,
        height,
        grid,
        goal_positions,
    });

    SokobanState::new(layout, player_col, player_row, box_positions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_roundtrip() {
        let ascii = "####\n#.@#\n#OX#\n####";
        let state = from_ascii(ascii);
        assert_eq!(state.layout.width, 4);
        assert_eq!(state.layout.height, 4);
        assert_eq!(state.player_col, 2);
        assert_eq!(state.player_row, 1);
        assert!(state.has_box(1, 2));
        assert!(state.layout.is_goal(2, 2));

        let back = to_ascii(&state);
        assert_eq!(back, ascii);
    }

    #[test]
    fn hash_eq_deterministic() {
        let layout = Arc::new(BoardLayout {
            width: 3,
            height: 3,
            grid: vec![Cell::Floor; 9],
            goal_positions: vec![],
        });

        let s1 = SokobanState::new(layout.clone(), 1, 1, vec![(0, 0), (2, 2)]);
        let s2 = SokobanState::new(layout.clone(), 1, 1, vec![(2, 2), (0, 0)]); // reversed
        let s3 = SokobanState::new(layout.clone(), 1, 1, vec![(0, 0), (2, 1)]); // different

        // Same player+boxes → same hash/eq
        assert_eq!(s1, s2);
        assert_eq!(
            crate::solver::bfs::state_hash(&s1),
            crate::solver::bfs::state_hash(&s2)
        );

        // Different → different
        assert_ne!(s1, s3);
        assert_ne!(
            crate::solver::bfs::state_hash(&s1),
            crate::solver::bfs::state_hash(&s3)
        );
    }

    #[test]
    fn sorted_box_invariant() {
        let layout = Arc::new(BoardLayout {
            width: 5,
            height: 5,
            grid: vec![Cell::Floor; 25],
            goal_positions: vec![],
        });

        let state = SokobanState::new(layout, 0, 0, vec![(3, 1), (1, 0), (2, 2)]);
        // Boxes should be sorted
        assert_eq!(state.box_positions, vec![(1, 0), (2, 2), (3, 1)]);
    }
}
