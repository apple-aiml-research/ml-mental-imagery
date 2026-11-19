//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};

// ---------------------------------------------------------------------------
// Edge types for jigsaw-shaped pieces
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeType {
    Flat, // border edge — straight
    Tab,  // protrudes outward
    Slot, // indents inward
}

// ---------------------------------------------------------------------------
// Procedural-image generating parameters (for puzzle fingerprinting)
// ---------------------------------------------------------------------------

/// A filled rectangle primitive drawn during image generation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GenRect {
    /// Index into the fixed `PALETTE` (see gen.rs).
    pub color_idx: u8,
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// A filled circle primitive drawn during image generation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GenCircle {
    /// Index into the fixed `PALETTE` (see gen.rs).
    pub color_idx: u8,
    pub cx: i32,
    pub cy: i32,
    pub radius: i32,
}

/// The exact RNG-drawn primitives that generate the source image, so the
/// puzzle can be fingerprinted by these compact parameters instead of the full
/// pixel buffer.
/// `piece_images` is a deterministic function of `(grid_size, GenParams)`.
/// A `trivial()` fallback puzzle has empty `rects`/`circles` (a sentinel).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct JigsawGenParams {
    /// Gradient background endpoints (top and bottom colors).
    pub bg_a: [u8; 3],
    pub bg_b: [u8; 3],
    /// Rectangles then circles, in draw (paint) order.
    pub rects: Vec<GenRect>,
    pub circles: Vec<GenCircle>,
}

// ---------------------------------------------------------------------------
// Shared puzzle definition (immutable, shared via Arc)
// ---------------------------------------------------------------------------

/// The immutable puzzle configuration shared across all states in an episode.
#[derive(Clone)]
pub struct JigsawPuzzle {
    pub grid_size: usize,
    /// Width of each piece in pixels (internal resolution).
    pub piece_width: u32,
    /// Height of each piece in pixels (internal resolution).
    pub piece_height: u32,
    /// RGB data for each piece, indexed by piece_id.
    /// piece_id = row * grid_size + col (the correct position).
    pub piece_images: Vec<Vec<u8>>,
    /// Shuffled piece IDs for tray slot assignment.
    /// tray_order[slot] = piece_id displayed in that slot.
    pub tray_order: Vec<usize>,
    /// Horizontal edges between row r and r+1. Size: (N-1)*N.
    /// `h_tabs[r * N + c]` = true means the top cell has a tab on its bottom edge
    /// (and the bottom cell has a slot on its top edge).
    pub h_tabs: Vec<bool>,
    /// Vertical edges between col c and c+1. Size: N*(N-1).
    /// `v_tabs[r * (N-1) + c]` = true means the left cell has a tab on its right edge
    /// (and the right cell has a slot on its left edge).
    pub v_tabs: Vec<bool>,
    /// Tray layout `(resolution, cell_size, positions)`, computed at the first
    /// render's resolution and reused by later renders. The packer is not
    /// scale-invariant, so `object_boxes` needs the resolution it was packed for.
    pub tray_layout: OnceLock<(u32, u32, Vec<(u32, u32)>)>,
    /// The exact procedural-image primitives drawn during generation;
    /// reproduces `piece_images` via `gen::paint_image`. Used for puzzle
    /// fingerprinting. Empty for `trivial()`.
    pub gen_params: JigsawGenParams,
}

impl JigsawPuzzle {
    /// Get the edge types [Top, Right, Bottom, Left] for a given piece_id.
    pub fn edge_types(&self, piece_id: usize) -> [EdgeType; 4] {
        let n = self.grid_size;
        let row = piece_id / n;
        let col = piece_id % n;

        // Top edge
        let top = if row == 0 {
            EdgeType::Flat
        } else {
            // h_tabs[(row-1)*N+col]: true = cell above has tab → this cell has slot
            if self.h_tabs[(row - 1) * n + col] {
                EdgeType::Slot
            } else {
                EdgeType::Tab
            }
        };

        // Bottom edge
        let bottom = if row == n - 1 {
            EdgeType::Flat
        } else {
            // h_tabs[row*N+col]: true = this cell has tab
            if self.h_tabs[row * n + col] {
                EdgeType::Tab
            } else {
                EdgeType::Slot
            }
        };

        // Left edge
        let left = if col == 0 {
            EdgeType::Flat
        } else {
            // v_tabs[row*(N-1)+(col-1)]: true = cell to left has tab → this cell has slot
            if self.v_tabs[row * (n - 1) + (col - 1)] {
                EdgeType::Slot
            } else {
                EdgeType::Tab
            }
        };

        // Right edge
        let right = if col == n - 1 {
            EdgeType::Flat
        } else {
            // v_tabs[row*(N-1)+col]: true = this cell has tab
            if self.v_tabs[row * (n - 1) + col] {
                EdgeType::Tab
            } else {
                EdgeType::Slot
            }
        };

        [top, right, bottom, left]
    }

    /// Get edge types for a piece rotated by the given angle (0, 90, 180, 270).
    /// Rotation cyclically shifts the [Top, Right, Bottom, Left] array.
    pub fn rotated_edge_types(&self, piece_id: usize, rotation: i32) -> [EdgeType; 4] {
        let base = self.edge_types(piece_id);
        match ((rotation % 360) + 360) % 360 {
            0 => base,
            90 => [base[3], base[0], base[1], base[2]],
            180 => [base[2], base[3], base[0], base[1]],
            270 => [base[1], base[2], base[3], base[0]],
            _ => base,
        }
    }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// The mutable state of a jigsaw puzzle.
///
/// `grid[row * grid_size + col]` is `Some(piece_id)` when a piece is placed,
/// `None` when empty. A piece is solved when it is at its correct target cell
/// for some global orientation and has the corresponding target rotation.
#[derive(Clone)]
pub struct JigsawState {
    pub puzzle: Arc<JigsawPuzzle>,
    /// Flat grid: grid_size * grid_size cells.
    pub grid: Vec<Option<usize>>,
    /// Sorted list of unplaced piece IDs.
    pub unplaced: Vec<usize>,
    /// Current rotation of each piece in degrees (0, 90, 180, 270).
    /// Indexed by piece_id.
    pub rotations: Vec<i32>,
}

// Hash/Eq use grid layout + rotations — the puzzle is shared and immutable.
impl Hash for JigsawState {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.grid.hash(hasher);
        self.rotations.hash(hasher);
    }
}

impl PartialEq for JigsawState {
    fn eq(&self, other: &Self) -> bool {
        self.grid == other.grid && self.rotations == other.rotations
    }
}

impl Eq for JigsawState {}

// ---------------------------------------------------------------------------
// Action
// ---------------------------------------------------------------------------

#[derive(Clone, Hash, Eq, PartialEq, Debug)]
pub enum JigsawAction {
    /// Place an unplaced piece at an empty grid cell.
    Place { piece_id: usize, cell: usize },
    /// Rotate a piece 90° clockwise (works on placed or unplaced pieces).
    RotateCW { piece_id: usize },
    /// Rotate a piece 90° counter-clockwise (works on placed or unplaced pieces).
    RotateCCW { piece_id: usize },
    /// Remove a placed piece from the grid back to the tray, keeping its rotation.
    Remove { piece_id: usize },
}

// ---------------------------------------------------------------------------
// Cell label helpers (A, B, C, ... Z, AA, AB, ...)
// ---------------------------------------------------------------------------

/// Convert a cell index to a label string (A=0, B=1, ..., Z=25, AA=26, ...).
pub fn cell_label(idx: usize) -> String {
    let mut result = String::new();
    let mut n = idx;
    loop {
        result.insert(0, (b'A' + (n % 26) as u8) as char);
        if n < 26 {
            break;
        }
        n = n / 26 - 1;
    }
    result
}

/// Parse a cell label back to an index. Returns None if invalid.
pub fn parse_cell_label(s: &str) -> Option<usize> {
    if s.is_empty() || !s.chars().all(|c| c.is_ascii_uppercase()) {
        return None;
    }
    let mut idx: usize = 0;
    for ch in s.chars() {
        idx = idx * 26 + (ch as usize - 'A' as usize) + 1;
    }
    Some(idx - 1)
}

// ---------------------------------------------------------------------------
// Rotation helpers
// ---------------------------------------------------------------------------

/// Minimum number of 90° rotations to go from `current` to `target`.
/// Considers both CW and CCW directions, returns the shorter path (0, 1, or 2).
pub fn min_rotations(current: i32, target: i32) -> usize {
    let diff = ((target - current) % 360 + 360) % 360;
    let cw = (diff / 90) as usize;
    let ccw = ((360 - diff) / 90) as usize;
    cw.min(ccw)
}

/// Target cell for a piece under a global puzzle rotation theta.
///
/// theta=0: piece (r,c) → cell (r,c)
/// theta=90: piece (r,c) → cell (c, n-1-r)
/// theta=180: piece (r,c) → cell (n-1-r, n-1-c)
/// theta=270: piece (r,c) → cell (n-1-c, r)
pub fn target_cell(piece_id: usize, n: usize, theta: i32) -> usize {
    let row = piece_id / n;
    let col = piece_id % n;
    match ((theta % 360) + 360) % 360 {
        0 => row * n + col,
        90 => col * n + (n - 1 - row),
        180 => (n - 1 - row) * n + (n - 1 - col),
        270 => (n - 1 - col) * n + row,
        _ => piece_id,
    }
}
