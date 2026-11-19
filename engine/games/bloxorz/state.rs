//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Block orientation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Orientation {
    /// Block stands upright on 1 cell at (col, row).
    Standing,
    /// Block lies flat along the X axis: occupies (col, row) and (col+1, row).
    LyingX,
    /// Block lies flat along the Y axis: occupies (col, row) and (col, row+1).
    LyingY,
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

/// Static grid layout shared across states via Arc.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BloxorzGrid {
    pub width: usize,
    pub height: usize,
    /// Flat row-major: cells[row * width + col]. true = solid tile.
    pub cells: Vec<bool>,
    /// Goal position (col, row).
    pub goal: (usize, usize),
    /// Start position (col, row, orientation).
    pub start: (usize, usize, Orientation),
    /// Positions of fragile tiles (max 32). Index = bit position in broken_mask.
    pub fragile_tiles: Vec<(usize, usize)>,
}

impl BloxorzGrid {
    pub fn is_solid(&self, col: usize, row: usize) -> bool {
        if col >= self.width || row >= self.height {
            return false;
        }
        self.cells[row * self.width + col]
    }

    /// Returns the fragile tile index for (col, row), or None if not fragile.
    pub fn fragile_index(&self, col: usize, row: usize) -> Option<usize> {
        self.fragile_tiles
            .iter()
            .position(|&(c, r)| c == col && r == row)
    }
}

/// Bloxorz game state: grid + block position and orientation.
#[derive(Debug, Clone)]
pub struct BloxorzState {
    pub grid: Arc<BloxorzGrid>,
    /// Anchor column (top-left of footprint).
    pub col: usize,
    /// Anchor row (top-left of footprint).
    pub row: usize,
    pub orientation: Orientation,
    /// Bitmask of broken fragile tiles. Bit i = 1 means fragile_tiles[i] is broken (void).
    pub broken_mask: u32,
}

impl PartialEq for BloxorzState {
    fn eq(&self, other: &Self) -> bool {
        self.col == other.col
            && self.row == other.row
            && self.orientation == other.orientation
            && self.broken_mask == other.broken_mask
    }
}

impl Eq for BloxorzState {}

impl Hash for BloxorzState {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.col.hash(hasher);
        self.row.hash(hasher);
        self.orientation.hash(hasher);
        self.broken_mask.hash(hasher);
    }
}

impl BloxorzState {
    /// Returns the cells occupied by the block.
    pub fn cells_occupied(&self) -> Vec<(usize, usize)> {
        match self.orientation {
            Orientation::Standing => vec![(self.col, self.row)],
            Orientation::LyingX => vec![(self.col, self.row), (self.col + 1, self.row)],
            Orientation::LyingY => vec![(self.col, self.row), (self.col, self.row + 1)],
        }
    }

    /// Check if the block position is valid: all occupied cells in bounds, solid, and not broken-fragile.
    pub fn is_valid(&self) -> bool {
        for (c, r) in self.cells_occupied() {
            if !self.grid.is_solid(c, r) {
                return false;
            }
            if let Some(idx) = self.grid.fragile_index(c, r) {
                if self.broken_mask & (1 << idx) != 0 {
                    return false;
                }
            }
        }
        true
    }

    /// Apply a direction to this state, returning a new state or None if invalid.
    pub fn apply_direction(&self, dir: Direction) -> Option<BloxorzState> {
        let (new_col, new_row, new_orient) = match (self.orientation, dir) {
            // Standing(c,r)
            (Orientation::Standing, Direction::Up) => {
                let r = self.row as i32 - 2;
                if r < 0 {
                    return None;
                }
                (self.col, r as usize, Orientation::LyingY)
            }
            (Orientation::Standing, Direction::Down) => {
                (self.col, self.row + 1, Orientation::LyingY)
            }
            (Orientation::Standing, Direction::Left) => {
                let c = self.col as i32 - 2;
                if c < 0 {
                    return None;
                }
                (c as usize, self.row, Orientation::LyingX)
            }
            (Orientation::Standing, Direction::Right) => {
                (self.col + 1, self.row, Orientation::LyingX)
            }

            // LyingX(c,r) — occupies (c,r) and (c+1,r)
            (Orientation::LyingX, Direction::Up) => {
                let r = self.row as i32 - 1;
                if r < 0 {
                    return None;
                }
                (self.col, r as usize, Orientation::LyingX)
            }
            (Orientation::LyingX, Direction::Down) => (self.col, self.row + 1, Orientation::LyingX),
            (Orientation::LyingX, Direction::Left) => {
                let c = self.col as i32 - 1;
                if c < 0 {
                    return None;
                }
                (c as usize, self.row, Orientation::Standing)
            }
            (Orientation::LyingX, Direction::Right) => {
                (self.col + 2, self.row, Orientation::Standing)
            }

            // LyingY(c,r) — occupies (c,r) and (c,r+1)
            (Orientation::LyingY, Direction::Up) => {
                let r = self.row as i32 - 1;
                if r < 0 {
                    return None;
                }
                (self.col, r as usize, Orientation::Standing)
            }
            (Orientation::LyingY, Direction::Down) => {
                (self.col, self.row + 2, Orientation::Standing)
            }
            (Orientation::LyingY, Direction::Left) => {
                let c = self.col as i32 - 1;
                if c < 0 {
                    return None;
                }
                (c as usize, self.row, Orientation::LyingY)
            }
            (Orientation::LyingY, Direction::Right) => {
                (self.col + 1, self.row, Orientation::LyingY)
            }
        };

        // Break any fragile tiles the block is leaving
        let mut new_mask = self.broken_mask;
        for (c, r) in self.cells_occupied() {
            if let Some(idx) = self.grid.fragile_index(c, r) {
                new_mask |= 1 << idx;
            }
        }

        let new_state = BloxorzState {
            grid: self.grid.clone(),
            col: new_col,
            row: new_row,
            orientation: new_orient,
            broken_mask: new_mask,
        };

        if new_state.is_valid() {
            Some(new_state)
        } else {
            None
        }
    }
}

// --- ASCII helpers (for tests) ---
//
// Legend:
//   S = standing block
//   = = lying-X left cell, - = lying-X right cell
//   H = lying-Y top cell, | = lying-Y bottom cell
//   . = solid tile
//   ' ' (space) = void
//   G = goal
//   F = fragile tile (unbroken)

const CHAR_STANDING: char = 'S';
const CHAR_LYINGX_LEFT: char = '=';
const CHAR_LYINGX_RIGHT: char = '-';
const CHAR_LYINGY_TOP: char = 'H';
const CHAR_LYINGY_BOTTOM: char = '|';
const CHAR_SOLID: char = '.';
const CHAR_VOID: char = ' ';
const CHAR_GOAL: char = 'G';
const CHAR_FRAGILE: char = 'F';

/// Convert state to ASCII string.
pub fn to_ascii(state: &BloxorzState) -> String {
    let occupied = state.cells_occupied();
    let mut lines = Vec::with_capacity(state.grid.height);
    for row in 0..state.grid.height {
        let mut chars = String::with_capacity(state.grid.width);
        for col in 0..state.grid.width {
            let is_goal = (col, row) == state.grid.goal;
            let ch = if occupied.contains(&(col, row)) {
                match state.orientation {
                    Orientation::Standing => CHAR_STANDING,
                    Orientation::LyingX => {
                        if col == state.col {
                            CHAR_LYINGX_LEFT
                        } else {
                            CHAR_LYINGX_RIGHT
                        }
                    }
                    Orientation::LyingY => {
                        if row == state.row {
                            CHAR_LYINGY_TOP
                        } else {
                            CHAR_LYINGY_BOTTOM
                        }
                    }
                }
            } else if is_goal {
                CHAR_GOAL
            } else if state.grid.fragile_index(col, row).is_some() {
                // Check if broken
                if let Some(idx) = state.grid.fragile_index(col, row) {
                    if state.broken_mask & (1 << idx) != 0 {
                        CHAR_VOID
                    } else {
                        CHAR_FRAGILE
                    }
                } else {
                    CHAR_FRAGILE
                }
            } else if state.grid.is_solid(col, row) {
                CHAR_SOLID
            } else {
                CHAR_VOID
            };
            chars.push(ch);
        }
        lines.push(chars);
    }
    lines.join("\n")
}

#[cfg(test)]
/// Parse an ASCII board into a BloxorzState.
pub fn from_ascii(text: &str) -> BloxorzState {
    let raw_lines: Vec<&str> = text.split('\n').collect();
    let max_width = raw_lines.iter().map(|l| l.len()).max().unwrap_or(0);
    let height = raw_lines.len();

    let mut cells = vec![false; max_width * height];
    let mut goal: Option<(usize, usize)> = None;
    let mut block_col: Option<usize> = None;
    let mut block_row: Option<usize> = None;
    let mut orientation: Option<Orientation> = None;
    let mut fragile_tiles: Vec<(usize, usize)> = Vec::new();

    for (row_idx, line) in raw_lines.iter().enumerate() {
        let chars: Vec<char> = {
            let mut c: Vec<char> = line.chars().collect();
            while c.len() < max_width {
                c.push(CHAR_VOID);
            }
            c
        };

        for (col_idx, &ch) in chars.iter().enumerate() {
            match ch {
                'S' => {
                    cells[row_idx * max_width + col_idx] = true;
                    block_col = Some(col_idx);
                    block_row = Some(row_idx);
                    orientation = Some(Orientation::Standing);
                }
                '=' => {
                    cells[row_idx * max_width + col_idx] = true;
                    // Left cell of LyingX — this is the anchor
                    block_col = Some(col_idx);
                    block_row = Some(row_idx);
                    orientation = Some(Orientation::LyingX);
                }
                '-' => {
                    cells[row_idx * max_width + col_idx] = true;
                    // Right cell of LyingX — only set anchor if not already set
                    if orientation != Some(Orientation::LyingX) {
                        block_col = Some(col_idx.saturating_sub(1));
                        block_row = Some(row_idx);
                        orientation = Some(Orientation::LyingX);
                    }
                }
                'H' => {
                    cells[row_idx * max_width + col_idx] = true;
                    // Top cell of LyingY — this is the anchor
                    block_col = Some(col_idx);
                    block_row = Some(row_idx);
                    orientation = Some(Orientation::LyingY);
                }
                '|' => {
                    cells[row_idx * max_width + col_idx] = true;
                    // Bottom cell of LyingY — only set anchor if not already set
                    if orientation != Some(Orientation::LyingY) {
                        block_col = Some(col_idx);
                        block_row = Some(row_idx.saturating_sub(1));
                        orientation = Some(Orientation::LyingY);
                    }
                }
                'G' => {
                    cells[row_idx * max_width + col_idx] = true;
                    goal = Some((col_idx, row_idx));
                }
                '.' => {
                    cells[row_idx * max_width + col_idx] = true;
                }
                'F' => {
                    cells[row_idx * max_width + col_idx] = true;
                    fragile_tiles.push((col_idx, row_idx));
                }
                _ => {
                    // space or anything else = void
                }
            }
        }
    }

    let goal = goal.expect("No goal (G) found in ASCII board");
    let block_col = block_col.expect("No block found in ASCII board");
    let block_row = block_row.expect("No block found in ASCII board");
    let orientation = orientation.expect("No block found in ASCII board");

    let grid = Arc::new(BloxorzGrid {
        width: max_width,
        height,
        cells,
        goal,
        start: (block_col, block_row, orientation),
        fragile_tiles,
    });

    BloxorzState {
        grid,
        col: block_col,
        row: block_row,
        orientation,
        broken_mask: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_roundtrip() {
        let ascii = "..G.\n....\nS...\n....";
        let state = from_ascii(ascii);
        assert_eq!(state.grid.width, 4);
        assert_eq!(state.grid.height, 4);
        assert_eq!(state.col, 0);
        assert_eq!(state.row, 2);
        assert_eq!(state.orientation, Orientation::Standing);
        assert_eq!(state.grid.goal, (2, 0));

        let back = to_ascii(&state);
        assert_eq!(back, ascii);
    }

    #[test]
    fn ascii_roundtrip_lying_x() {
        let ascii = ".G.\n=-.\n...";
        let state = from_ascii(ascii);
        assert_eq!(state.col, 0);
        assert_eq!(state.row, 1);
        assert_eq!(state.orientation, Orientation::LyingX);

        let back = to_ascii(&state);
        assert_eq!(back, ascii);
    }

    #[test]
    fn ascii_roundtrip_lying_y() {
        let ascii = ".G.\nH..\n|..\n...";
        let state = from_ascii(ascii);
        assert_eq!(state.col, 0);
        assert_eq!(state.row, 1);
        assert_eq!(state.orientation, Orientation::LyingY);

        let back = to_ascii(&state);
        assert_eq!(back, ascii);
    }

    #[test]
    fn hash_eq_deterministic() {
        let grid = Arc::new(BloxorzGrid {
            width: 4,
            height: 4,
            cells: vec![true; 16],
            goal: (3, 3),
            start: (1, 1, Orientation::Standing),
            fragile_tiles: vec![],
        });

        let s1 = BloxorzState {
            grid: grid.clone(),
            col: 1,
            row: 1,
            orientation: Orientation::Standing,
            broken_mask: 0,
        };
        let s2 = BloxorzState {
            grid: grid.clone(),
            col: 1,
            row: 1,
            orientation: Orientation::Standing,
            broken_mask: 0,
        };
        let s3 = BloxorzState {
            grid: grid.clone(),
            col: 1,
            row: 1,
            orientation: Orientation::LyingX,
            broken_mask: 0,
        };

        assert_eq!(s1, s2);
        assert_eq!(
            crate::solver::bfs::state_hash(&s1),
            crate::solver::bfs::state_hash(&s2)
        );

        assert_ne!(s1, s3);
        assert_ne!(
            crate::solver::bfs::state_hash(&s1),
            crate::solver::bfs::state_hash(&s3)
        );
    }

    #[test]
    fn cells_occupied_standing() {
        let grid = Arc::new(BloxorzGrid {
            width: 4,
            height: 4,
            cells: vec![true; 16],
            goal: (3, 3),
            start: (2, 1, Orientation::Standing),
            fragile_tiles: vec![],
        });
        let state = BloxorzState {
            grid,
            col: 2,
            row: 1,
            orientation: Orientation::Standing,
            broken_mask: 0,
        };
        assert_eq!(state.cells_occupied(), vec![(2, 1)]);
    }

    #[test]
    fn cells_occupied_lying_x() {
        let grid = Arc::new(BloxorzGrid {
            width: 4,
            height: 4,
            cells: vec![true; 16],
            goal: (3, 3),
            start: (1, 2, Orientation::LyingX),
            fragile_tiles: vec![],
        });
        let state = BloxorzState {
            grid,
            col: 1,
            row: 2,
            orientation: Orientation::LyingX,
            broken_mask: 0,
        };
        assert_eq!(state.cells_occupied(), vec![(1, 2), (2, 2)]);
    }

    #[test]
    fn cells_occupied_lying_y() {
        let grid = Arc::new(BloxorzGrid {
            width: 4,
            height: 4,
            cells: vec![true; 16],
            goal: (3, 3),
            start: (1, 0, Orientation::LyingY),
            fragile_tiles: vec![],
        });
        let state = BloxorzState {
            grid,
            col: 1,
            row: 0,
            orientation: Orientation::LyingY,
            broken_mask: 0,
        };
        assert_eq!(state.cells_occupied(), vec![(1, 0), (1, 1)]);
    }

    #[test]
    fn movement_standing_all_directions() {
        // 6x6 all-solid grid, standing at (2,3)
        let grid = Arc::new(BloxorzGrid {
            width: 6,
            height: 6,
            cells: vec![true; 36],
            goal: (0, 0),
            start: (2, 3, Orientation::Standing),
            fragile_tiles: vec![],
        });
        let state = BloxorzState {
            grid,
            col: 2,
            row: 3,
            orientation: Orientation::Standing,
            broken_mask: 0,
        };

        // Up: Standing(2,3) → LyingY(2, 1)
        let up = state.apply_direction(Direction::Up).unwrap();
        assert_eq!(
            (up.col, up.row, up.orientation),
            (2, 1, Orientation::LyingY)
        );

        // Down: Standing(2,3) → LyingY(2, 4)
        let down = state.apply_direction(Direction::Down).unwrap();
        assert_eq!(
            (down.col, down.row, down.orientation),
            (2, 4, Orientation::LyingY)
        );

        // Left: Standing(2,3) → LyingX(0, 3)
        let left = state.apply_direction(Direction::Left).unwrap();
        assert_eq!(
            (left.col, left.row, left.orientation),
            (0, 3, Orientation::LyingX)
        );

        // Right: Standing(2,3) → LyingX(3, 3)
        let right = state.apply_direction(Direction::Right).unwrap();
        assert_eq!(
            (right.col, right.row, right.orientation),
            (3, 3, Orientation::LyingX)
        );
    }

    #[test]
    fn movement_lying_x_all_directions() {
        let grid = Arc::new(BloxorzGrid {
            width: 6,
            height: 6,
            cells: vec![true; 36],
            goal: (0, 0),
            start: (2, 3, Orientation::LyingX),
            fragile_tiles: vec![],
        });
        let state = BloxorzState {
            grid,
            col: 2,
            row: 3,
            orientation: Orientation::LyingX,
            broken_mask: 0,
        };

        // Up: LyingX(2,3) → LyingX(2, 2)
        let up = state.apply_direction(Direction::Up).unwrap();
        assert_eq!(
            (up.col, up.row, up.orientation),
            (2, 2, Orientation::LyingX)
        );

        // Down: LyingX(2,3) → LyingX(2, 4)
        let down = state.apply_direction(Direction::Down).unwrap();
        assert_eq!(
            (down.col, down.row, down.orientation),
            (2, 4, Orientation::LyingX)
        );

        // Left: LyingX(2,3) → Standing(1, 3)
        let left = state.apply_direction(Direction::Left).unwrap();
        assert_eq!(
            (left.col, left.row, left.orientation),
            (1, 3, Orientation::Standing)
        );

        // Right: LyingX(2,3) → Standing(4, 3)
        let right = state.apply_direction(Direction::Right).unwrap();
        assert_eq!(
            (right.col, right.row, right.orientation),
            (4, 3, Orientation::Standing)
        );
    }

    #[test]
    fn movement_lying_y_all_directions() {
        let grid = Arc::new(BloxorzGrid {
            width: 6,
            height: 6,
            cells: vec![true; 36],
            goal: (0, 0),
            start: (2, 2, Orientation::LyingY),
            fragile_tiles: vec![],
        });
        let state = BloxorzState {
            grid,
            col: 2,
            row: 2,
            orientation: Orientation::LyingY,
            broken_mask: 0,
        };

        // Up: LyingY(2,2) → Standing(2, 1)
        let up = state.apply_direction(Direction::Up).unwrap();
        assert_eq!(
            (up.col, up.row, up.orientation),
            (2, 1, Orientation::Standing)
        );

        // Down: LyingY(2,2) → Standing(2, 4)
        let down = state.apply_direction(Direction::Down).unwrap();
        assert_eq!(
            (down.col, down.row, down.orientation),
            (2, 4, Orientation::Standing)
        );

        // Left: LyingY(2,2) → LyingY(1, 2)
        let left = state.apply_direction(Direction::Left).unwrap();
        assert_eq!(
            (left.col, left.row, left.orientation),
            (1, 2, Orientation::LyingY)
        );

        // Right: LyingY(2,2) → LyingY(3, 2)
        let right = state.apply_direction(Direction::Right).unwrap();
        assert_eq!(
            (right.col, right.row, right.orientation),
            (3, 2, Orientation::LyingY)
        );
    }

    #[test]
    fn movement_off_edge_returns_none() {
        // 3x3 grid, standing at (0,0)
        let grid = Arc::new(BloxorzGrid {
            width: 3,
            height: 3,
            cells: vec![true; 9],
            goal: (2, 2),
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

        // Up from row 0: needs row-2 = -2 → None
        assert!(state.apply_direction(Direction::Up).is_none());
        // Left from col 0: needs col-2 = -2 → None
        assert!(state.apply_direction(Direction::Left).is_none());
    }

    #[test]
    fn movement_onto_void_returns_none() {
        // 4x4 grid with a void cell at (3,1)
        let mut cells = vec![true; 16];
        cells[1 * 4 + 3] = false; // (3, 1) is void
        let grid = Arc::new(BloxorzGrid {
            width: 4,
            height: 4,
            cells,
            goal: (0, 0),
            start: (1, 1, Orientation::Standing),
            fragile_tiles: vec![],
        });

        // LyingX at (2,1) occupies (2,1) and (3,1) — (3,1) is void
        let state = BloxorzState {
            grid: grid.clone(),
            col: 1,
            row: 1,
            orientation: Orientation::Standing,
            broken_mask: 0,
        };
        // Roll right: Standing(1,1) → LyingX(2, 1), occupies (2,1) and (3,1)
        assert!(state.apply_direction(Direction::Right).is_none());
    }

    // --- Fragile tile tests ---

    #[test]
    fn fragile_tile_breaks_on_leave() {
        // 6x1 row: S . F . . G
        // Standing at (0,0), fragile at (2,0), goal at (5,0)
        let grid = Arc::new(BloxorzGrid {
            width: 6,
            height: 1,
            cells: vec![true; 6],
            goal: (5, 0),
            start: (0, 0, Orientation::Standing),
            fragile_tiles: vec![(2, 0)],
        });
        let state = BloxorzState {
            grid,
            col: 0,
            row: 0,
            orientation: Orientation::Standing,
            broken_mask: 0,
        };

        // Right: Standing(0,0) → LyingX(1,0), occupies (1,0) and (2,0)
        // Neither cell was previously occupied, so broken_mask should stay 0
        let step1 = state.apply_direction(Direction::Right).unwrap();
        assert_eq!(step1.broken_mask, 0);

        // Right: LyingX(1,0) → Standing(3,0), LEAVES cells (1,0) and (2,0)
        // Cell (2,0) is fragile → bit 0 should be set
        let step2 = step1.apply_direction(Direction::Right).unwrap();
        assert_eq!(step2.broken_mask, 1);
    }

    #[test]
    fn broken_fragile_blocks_movement() {
        // 4x1 row: all solid, fragile at (1,0)
        let grid = Arc::new(BloxorzGrid {
            width: 4,
            height: 1,
            cells: vec![true; 4],
            goal: (3, 0),
            start: (0, 0, Orientation::Standing),
            fragile_tiles: vec![(1, 0)],
        });
        // State with broken_mask=1 (fragile tile already broken)
        let state = BloxorzState {
            grid,
            col: 0,
            row: 0,
            orientation: Orientation::Standing,
            broken_mask: 1,
        };

        // Right: Standing(0,0) → LyingX(1,0), occupies (1,0) and (2,0)
        // (1,0) is broken fragile → should be invalid
        assert!(state.apply_direction(Direction::Right).is_none());
    }

    #[test]
    fn broken_mask_in_hash_eq() {
        let grid = Arc::new(BloxorzGrid {
            width: 4,
            height: 1,
            cells: vec![true; 4],
            goal: (3, 0),
            start: (0, 0, Orientation::Standing),
            fragile_tiles: vec![(1, 0)],
        });
        let s1 = BloxorzState {
            grid: grid.clone(),
            col: 0,
            row: 0,
            orientation: Orientation::Standing,
            broken_mask: 0,
        };
        let s2 = BloxorzState {
            grid: grid.clone(),
            col: 0,
            row: 0,
            orientation: Orientation::Standing,
            broken_mask: 1,
        };

        assert_ne!(s1, s2);
        assert_ne!(
            crate::solver::bfs::state_hash(&s1),
            crate::solver::bfs::state_hash(&s2)
        );
    }

    #[test]
    fn ascii_roundtrip_fragile() {
        let ascii = ".F.\n.S.\n..G";
        let state = from_ascii(ascii);
        assert_eq!(state.grid.fragile_tiles, vec![(1, 0)]);
        let back = to_ascii(&state);
        assert_eq!(back, ascii);
    }
}
