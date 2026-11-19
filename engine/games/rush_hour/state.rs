//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Orientation of a vehicle on the grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Orientation {
    Horizontal,
    Vertical,
}

/// Static definition of a vehicle (shared across states via Arc<BoardConfig>).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VehicleDef {
    pub id: char,
    pub orientation: Orientation,
    pub length: u8,
    /// Row for Horizontal vehicles, column for Vertical vehicles.
    pub fixed_axis: u8,
}

/// Board configuration — the static part of the puzzle (shared via Arc).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoardConfig {
    pub grid_size: u8,
    /// Vehicles sorted by id. Index 0 is always the target car ('A').
    pub vehicles: Vec<VehicleDef>,
    /// Row of the exit (same row as target car 'A').
    pub exit_row: u8,
}

impl BoardConfig {
    pub fn vehicle_index(&self, id: char) -> Option<usize> {
        self.vehicles.iter().position(|v| v.id == id)
    }
}

/// Rush Hour game state — position of each vehicle along its movable axis.
///
/// `positions[i]` is the column (horizontal) or row (vertical) of vehicle i's
/// leading edge. Parallel to `config.vehicles`.
#[derive(Debug, Clone)]
pub struct RushHourState {
    pub config: Arc<BoardConfig>,
    pub positions: Vec<u8>,
}

impl PartialEq for RushHourState {
    fn eq(&self, other: &Self) -> bool {
        self.positions == other.positions
    }
}

impl Eq for RushHourState {}

impl Hash for RushHourState {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.positions.hash(hasher);
    }
}

impl RushHourState {
    /// Build a flat occupancy grid: cell (col, row) → Some(vehicle_idx) or None.
    /// Size is grid_size × grid_size.
    pub fn occupancy_grid(&self) -> Vec<Option<usize>> {
        let gs = self.config.grid_size as usize;
        let mut grid = vec![None; gs * gs];
        for (idx, def) in self.config.vehicles.iter().enumerate() {
            let pos = self.positions[idx] as usize;
            for offset in 0..def.length as usize {
                let (col, row) = match def.orientation {
                    Orientation::Horizontal => (pos + offset, def.fixed_axis as usize),
                    Orientation::Vertical => (def.fixed_axis as usize, pos + offset),
                };
                grid[row * gs + col] = Some(idx);
            }
        }
        grid
    }
}

/// Slide action: move a vehicle by some amount along its axis.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Slide {
    pub vehicle_id: char,
    /// Positive = right/down, negative = left/up. Zero is invalid.
    pub amount: i8,
}

// --- ASCII helpers (for tests) ---

#[cfg(test)]
/// Parse an ASCII board into a RushHourState.
///
/// Each letter represents a vehicle's cells. '.' is empty.
/// Example:
/// ```text
/// ..BCCC
/// ..B...
/// AAB..D
/// .EE..D
/// ......
/// ......
/// ```
pub fn from_ascii(text: &str) -> RushHourState {
    let lines: Vec<&str> = text.trim().split('\n').collect();
    let height = lines.len();
    let width = lines.iter().map(|l| l.len()).max().unwrap_or(0);

    // Collect cells per vehicle letter
    let mut vehicle_cells: std::collections::BTreeMap<char, Vec<(usize, usize)>> =
        std::collections::BTreeMap::new();

    for (row, line) in lines.iter().enumerate() {
        for (col, ch) in line.chars().enumerate() {
            if ch != '.' {
                vehicle_cells.entry(ch).or_default().push((col, row));
            }
        }
    }

    let mut vehicles = Vec::new();
    let mut positions = Vec::new();
    let mut target_row = 2u8; // default exit row

    // BTreeMap iterates in sorted order by key (char), so 'A' comes first
    for (&id, cells) in &vehicle_cells {
        let length = cells.len() as u8;
        assert!(length >= 2, "Vehicle '{id}' has only {length} cell(s)");

        // Infer orientation: if all cells share the same row → horizontal
        let all_same_row = cells.iter().all(|c| c.1 == cells[0].1);
        let all_same_col = cells.iter().all(|c| c.0 == cells[0].0);
        assert!(
            all_same_row || all_same_col,
            "Vehicle '{id}' cells are not in a line"
        );

        let orientation = if all_same_row {
            Orientation::Horizontal
        } else {
            Orientation::Vertical
        };

        let (fixed_axis, pos) = match orientation {
            Orientation::Horizontal => {
                let row = cells[0].1 as u8;
                let min_col = cells.iter().map(|c| c.0).min().unwrap() as u8;
                (row, min_col)
            }
            Orientation::Vertical => {
                let col = cells[0].0 as u8;
                let min_row = cells.iter().map(|c| c.1).min().unwrap() as u8;
                (col, min_row)
            }
        };

        if id == 'A' {
            target_row = match orientation {
                Orientation::Horizontal => fixed_axis,
                Orientation::Vertical => {
                    panic!("Target car 'A' must be horizontal")
                }
            };
        }

        vehicles.push(VehicleDef {
            id,
            orientation,
            length,
            fixed_axis,
        });
        positions.push(pos);
    }

    assert!(
        vehicles.iter().any(|v| v.id == 'A'),
        "No target car 'A' found"
    );
    // 'A' is already first due to BTreeMap ordering

    let config = Arc::new(BoardConfig {
        grid_size: width.max(height) as u8,
        vehicles,
        exit_row: target_row,
    });

    RushHourState { config, positions }
}

#[cfg(test)]
/// Render a RushHourState as ASCII. '.' for empty, vehicle id char for occupied.
pub fn to_ascii(state: &RushHourState) -> String {
    let gs = state.config.grid_size as usize;
    let mut grid = vec!['.'; gs * gs];

    for (idx, def) in state.config.vehicles.iter().enumerate() {
        let pos = state.positions[idx] as usize;
        for offset in 0..def.length as usize {
            let (col, row) = match def.orientation {
                Orientation::Horizontal => (pos + offset, def.fixed_axis as usize),
                Orientation::Vertical => (def.fixed_axis as usize, pos + offset),
            };
            grid[row * gs + col] = def.id;
        }
    }

    let mut lines = Vec::with_capacity(gs);
    for row in 0..gs {
        let line: String = grid[row * gs..(row + 1) * gs].iter().collect();
        lines.push(line);
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_roundtrip() {
        let ascii = "..BCCC\n..B...\nAAB..D\n.EE..D\n......\n......";
        let state = from_ascii(ascii);
        assert_eq!(state.config.grid_size, 6);
        assert_eq!(state.config.vehicles.len(), 5);
        assert_eq!(state.config.vehicles[0].id, 'A');
        assert_eq!(state.config.exit_row, 2);

        let back = to_ascii(&state);
        assert_eq!(back, ascii);
    }

    #[test]
    fn hash_eq_deterministic() {
        let s1 = from_ascii("......\n......\nAA....\n......\n......\n......");
        let s2 = from_ascii("......\n......\nAA....\n......\n......\n......");
        let s3 = from_ascii("......\n......\n.AA...\n......\n......\n......");

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
    fn occupancy_grid_correct() {
        let state = from_ascii("......\n......\nAAB...\n..B...\n......\n......");
        let grid = state.occupancy_grid();
        let gs = 6;

        // A occupies (0,2) and (1,2)
        assert_eq!(grid[2 * gs + 0], Some(0));
        assert_eq!(grid[2 * gs + 1], Some(0));

        // B occupies (2,2) and (2,3)
        let b_idx = state.config.vehicle_index('B').unwrap();
        assert_eq!(grid[2 * gs + 2], Some(b_idx));
        assert_eq!(grid[3 * gs + 2], Some(b_idx));

        // Empty cell
        assert_eq!(grid[0 * gs + 0], None);
    }

    #[test]
    fn vehicle_index_lookup() {
        let state = from_ascii("..BCCC\n..B...\nAAB..D\n.EE..D\n......\n......");
        assert_eq!(state.config.vehicle_index('A'), Some(0));
        assert_eq!(state.config.vehicle_index('B'), Some(1));
        assert_eq!(state.config.vehicle_index('Z'), None);
    }
}
