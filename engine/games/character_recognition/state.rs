//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
use std::hash::{Hash, Hasher};
use std::sync::Arc;

pub const UPPERCASE_CHARS: &[char] = &[
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'J', 'K', 'L', 'P', 'Q', 'R', 'T', 'U', 'V', 'Y',
];
pub const LOWERCASE_CHARS: &[char] = &[
    'a', 'c', 'e', 'f', 'g', 'h', 'j', 'k', 'm', 'r', 't', 'v', 'w', 'y',
];
pub const DIGIT_CHARS: &[char] = &['2', '3', '4', '5', '7'];
pub const GREEK_UPPER_CHARS: &[char] = &['Γ', 'Δ', 'Λ', 'Π', 'Σ', 'Φ', 'Ψ', 'Ω'];
pub const GREEK_LOWER_CHARS: &[char] = &[
    'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'λ', 'μ', 'ξ', 'π', 'ρ', 'σ', 'φ', 'ψ',
];

pub fn char_to_name(c: char) -> String {
    match c {
        'Γ' | 'γ' => "gamma".into(),
        'Δ' | 'δ' => "delta".into(),
        'Λ' | 'λ' => "lambda".into(),
        'Π' | 'π' => "pi".into(),
        'Σ' | 'σ' => "sigma".into(),
        'Φ' | 'φ' => "phi".into(),
        'Ψ' | 'ψ' => "psi".into(),
        'Ω' => "omega".into(),
        'α' => "alpha".into(),
        'β' => "beta".into(),
        'ε' => "epsilon".into(),
        'ζ' => "zeta".into(),
        'η' => "eta".into(),
        'μ' => "mu".into(),
        'ξ' => "xi".into(),
        'ρ' => "rho".into(),
        other => other.to_string(),
    }
}

pub fn all_allowed_chars() -> Vec<char> {
    let mut chars = Vec::new();
    chars.extend_from_slice(UPPERCASE_CHARS);
    chars.extend_from_slice(LOWERCASE_CHARS);
    chars.extend_from_slice(DIGIT_CHARS);
    chars.extend_from_slice(GREEK_UPPER_CHARS);
    chars.extend_from_slice(GREEK_LOWER_CHARS);
    chars
}

// ---------------------------------------------------------------------------
// Puzzle configuration (immutable, shared via Arc)
// ---------------------------------------------------------------------------

/// Half-plane cut: a*x + b*y < c.
#[derive(Debug, Clone)]
pub struct CutLine {
    pub a: f64,
    pub b: f64,
    pub c: f64,
}

impl CutLine {
    pub fn is_above(&self, x: f64, y: f64) -> bool {
        self.a * x + self.b * y < self.c
    }
}

/// Tree of cuts that maps any point to a piece index.
#[derive(Debug, Clone)]
pub enum PieceCutTree {
    Leaf(usize),
    Split {
        cut: CutLine,
        above: Box<PieceCutTree>,
        below: Box<PieceCutTree>,
    },
}

impl PieceCutTree {
    pub fn piece_at(&self, x: f64, y: f64) -> usize {
        match self {
            PieceCutTree::Leaf(idx) => *idx,
            PieceCutTree::Split { cut, above, below } => {
                if cut.is_above(x, y) {
                    above.piece_at(x, y)
                } else {
                    below.piece_at(x, y)
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct PieceDef {
    /// Absolute cells in the character grid at rotation 0.
    pub cells: Vec<(i32, i32)>,
    /// Center of mass in absolute grid coordinates.
    pub centroid: (f64, f64),
}

impl PieceDef {
    /// Return cells rendered at the given rotation around the piece's centroid,
    /// translated so the rotated centroid aligns with `target_centroid`.
    pub fn rendered_cells(&self, rotation: u8, target_centroid: (f64, f64)) -> Vec<(i32, i32)> {
        self.cells
            .iter()
            .map(|&(x, y)| {
                let dx = x as f64 - self.centroid.0;
                let dy = y as f64 - self.centroid.1;
                let (rdx, rdy) = match rotation % 4 {
                    0 => (dx, dy),
                    1 => (-dy, dx),
                    2 => (-dx, -dy),
                    3 => (dy, -dx),
                    _ => unreachable!(),
                };
                (
                    (rdx + target_centroid.0).round() as i32,
                    (rdy + target_centroid.1).round() as i32,
                )
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct SlotDef {
    /// Grid intersection coordinates (row, col).
    pub id: (usize, usize),
    /// Position in absolute grid coordinates (intersection point).
    pub centroid: (f64, f64),
    /// Which piece matches this slot, or None for empty intersections.
    pub matching_piece_idx: Option<usize>,
    /// Rotation at which the piece correctly fills this slot (always 0).
    pub correct_rotation: u8,
}

#[derive(Debug, Clone)]
pub struct PieceLookup {
    pub points: Vec<(f64, f64)>,
    pub edges: Vec<(usize, usize)>,
    pub vertex_piece: Vec<usize>,
}

impl PieceLookup {
    pub fn piece_at(&self, x: f64, y: f64) -> usize {
        let mut best_dist = f64::MAX;
        let mut best_piece = 0;
        for &(u, v) in &self.edges {
            let (x1, y1) = self.points[u];
            let (x2, y2) = self.points[v];
            let d = point_to_segment_dist_sq(x, y, x1, y1, x2, y2);
            if d < best_dist {
                best_dist = d;
                let pu = self.vertex_piece[u];
                let pv = self.vertex_piece[v];
                if pu == pv {
                    best_piece = pu;
                } else {
                    let du = (x - x1).powi(2) + (y - y1).powi(2);
                    let dv = (x - x2).powi(2) + (y - y2).powi(2);
                    best_piece = if du <= dv { pu } else { pv };
                }
            }
        }
        best_piece
    }
}

fn point_to_segment_dist_sq(px: f64, py: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-12 {
        return (px - x1).powi(2) + (py - y1).powi(2);
    }
    let t = ((px - x1) * dx + (py - y1) * dy) / len_sq;
    let t = t.clamp(0.0, 1.0);
    let proj_x = x1 + t * dx;
    let proj_y = y1 + t * dy;
    (px - proj_x).powi(2) + (py - proj_y).powi(2)
}

#[derive(Debug, Clone)]
pub struct CharacterRecognitionPuzzle {
    /// English name of the character (e.g. "psi", "A", "7").
    pub character: String,
    /// The actual Unicode char, needed for rendering.
    pub character_char: char,
    pub font_idx: usize,
    pub grid_size: usize,
    pub num_pieces: usize,
    pub pieces: Vec<PieceDef>,
    pub slots: Vec<SlotDef>,
    pub fill_color: [u8; 3],
    pub piece_solve_order: Vec<usize>,
    /// Allowed answer names (English strings, e.g. "psi", "A", "7").
    pub allowed_chars: Vec<String>,
    pub initial_rotations: Vec<u8>,
    /// Sub-pixel piece boundary lookup via graph Voronoi.
    pub piece_lookup: Option<PieceLookup>,
    /// NxN grid dimension for slot placement overlay.
    pub slot_grid_dim: usize,
    /// Cached character rendering parameters (computed on first render).
    pub char_render_cache: super::render::CharRenderCache,
}

// ---------------------------------------------------------------------------
// Game state (mutable per step)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CharacterRecognitionState {
    pub puzzle: Arc<CharacterRecognitionPuzzle>,
    /// Current rotation per piece (0-3, x 90 degrees).
    pub rotations: Vec<u8>,
    /// Placement: None = unplaced, Some(slot_idx) = placed in that slot.
    pub placements: Vec<Option<usize>>,
    /// Character identification answer (English name string).
    pub answer: Option<String>,
}

impl Hash for CharacterRecognitionState {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.rotations.hash(hasher);
        self.placements.hash(hasher);
        self.answer.hash(hasher);
    }
}

impl PartialEq for CharacterRecognitionState {
    fn eq(&self, other: &Self) -> bool {
        self.rotations == other.rotations
            && self.placements == other.placements
            && self.answer == other.answer
    }
}

impl Eq for CharacterRecognitionState {}

unsafe impl Send for CharacterRecognitionState {}

impl CharacterRecognitionState {
    pub fn slot_occupied(&self, slot_idx: usize) -> bool {
        self.placements.iter().any(|p| *p == Some(slot_idx))
    }

    pub fn piece_correct(&self, piece_idx: usize) -> bool {
        if let Some(slot_idx) = self.placements[piece_idx] {
            let slot = &self.puzzle.slots[slot_idx];
            slot.matching_piece_idx == Some(piece_idx)
                && self.rotations[piece_idx] == slot.correct_rotation
        } else {
            false
        }
    }

    pub fn all_pieces_correctly_placed(&self) -> bool {
        (0..self.puzzle.num_pieces).all(|i| self.piece_correct(i))
    }
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CharacterRecognitionAction {
    RotateCW {
        piece_id: usize,
        angle: u16,
    },
    RotateCCW {
        piece_id: usize,
        angle: u16,
    },
    Place {
        piece_id: usize,
        row: usize,
        col: usize,
    },
    Undo {
        piece_id: usize,
    },
    Identify {
        character: String,
    },
}

impl CharacterRecognitionAction {
    pub fn to_str(&self) -> String {
        match self {
            Self::RotateCW { piece_id, angle } => format!("rotatecw({piece_id}, {angle})"),
            Self::RotateCCW { piece_id, angle } => format!("rotateccw({piece_id}, {angle})"),
            Self::Place { piece_id, row, col } => format!("place({piece_id}, {row}, {col})"),
            Self::Undo { piece_id } => format!("undo({piece_id})"),
            Self::Identify { character } => format!("identify('{character}')"),
        }
    }

    pub fn parse(
        s: &str,
        num_pieces: usize,
        grid_dim: usize,
        allowed_names: &[String],
    ) -> Option<Self> {
        let s = s.trim();

        if let Some(inner) = s
            .strip_prefix("rotatecw(")
            .and_then(|r| r.strip_suffix(')'))
        {
            let (pid, angle_str) = parse_two_args(inner)?;
            let piece_id: usize = pid.parse().ok()?;
            let angle: u16 = angle_str.parse().ok()?;
            if piece_id >= num_pieces || angle != 90 {
                return None;
            }
            return Some(Self::RotateCW { piece_id, angle });
        }

        if let Some(inner) = s
            .strip_prefix("rotateccw(")
            .and_then(|r| r.strip_suffix(')'))
        {
            let (pid, angle_str) = parse_two_args(inner)?;
            let piece_id: usize = pid.parse().ok()?;
            let angle: u16 = angle_str.parse().ok()?;
            if piece_id >= num_pieces || angle != 90 {
                return None;
            }
            return Some(Self::RotateCCW { piece_id, angle });
        }

        if let Some(inner) = s.strip_prefix("place(").and_then(|r| r.strip_suffix(')')) {
            let (pid, rest) = parse_two_args(inner)?;
            let piece_id: usize = pid.parse().ok()?;
            let (row_str, col_str) = parse_two_args(rest)?;
            let row: usize = row_str.parse().ok()?;
            let col: usize = col_str.parse().ok()?;
            if piece_id >= num_pieces || row >= grid_dim || col >= grid_dim {
                return None;
            }
            return Some(Self::Place { piece_id, row, col });
        }

        if let Some(inner) = s.strip_prefix("undo(").and_then(|r| r.strip_suffix(')')) {
            let piece_id: usize = inner.trim().parse().ok()?;
            if piece_id >= num_pieces {
                return None;
            }
            return Some(Self::Undo { piece_id });
        }

        if let Some(inner) = s
            .strip_prefix("identify(")
            .and_then(|r| r.strip_suffix(')'))
        {
            let inner = inner.trim();
            let name = inner
                .strip_prefix('\'')
                .and_then(|r| r.strip_suffix('\''))?;
            let name_string = name.to_string();
            if !allowed_names.contains(&name_string) {
                return None;
            }
            return Some(Self::Identify {
                character: name_string,
            });
        }

        None
    }
}

fn parse_two_args(s: &str) -> Option<(&str, &str)> {
    let comma = s.find(',')?;
    Some((s[..comma].trim(), s[comma + 1..].trim()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_roundtrip() {
        let grid_dim = 4;
        let allowed: Vec<String> = all_allowed_chars()
            .iter()
            .map(|c| char_to_name(*c).to_string())
            .collect();
        let actions = vec![
            CharacterRecognitionAction::RotateCW {
                piece_id: 0,
                angle: 90,
            },
            CharacterRecognitionAction::RotateCCW {
                piece_id: 1,
                angle: 90,
            },
            CharacterRecognitionAction::Place {
                piece_id: 0,
                row: 0,
                col: 1,
            },
            CharacterRecognitionAction::Place {
                piece_id: 1,
                row: 2,
                col: 3,
            },
            CharacterRecognitionAction::Undo { piece_id: 0 },
            CharacterRecognitionAction::Identify {
                character: "A".to_string(),
            },
            CharacterRecognitionAction::Identify {
                character: "a".to_string(),
            },
            CharacterRecognitionAction::Identify {
                character: "7".to_string(),
            },
            CharacterRecognitionAction::Identify {
                character: "psi".to_string(),
            },
            CharacterRecognitionAction::Identify {
                character: "alpha".to_string(),
            },
        ];
        for action in &actions {
            let s = action.to_str();
            let parsed = CharacterRecognitionAction::parse(&s, 2, grid_dim, &allowed)
                .unwrap_or_else(|| panic!("failed to parse: {s}"));
            assert_eq!(*action, parsed, "roundtrip failed for {s}");
        }
    }

    #[test]
    fn parse_invalid() {
        let grid_dim = 4;
        let allowed: Vec<String> = all_allowed_chars()
            .iter()
            .map(|c| char_to_name(*c).to_string())
            .collect();
        // Invalid angle
        assert!(
            CharacterRecognitionAction::parse("rotatecw(0, 45)", 2, grid_dim, &allowed).is_none()
        );
        // Out of range row/col
        assert!(
            CharacterRecognitionAction::parse("place(0, 5, 0)", 2, grid_dim, &allowed).is_none()
        );
        // Out of range piece
        assert!(CharacterRecognitionAction::parse("undo(5)", 2, grid_dim, &allowed).is_none());
        // Invalid character for identify (not in allowed names)
        assert!(
            CharacterRecognitionAction::parse("identify('O')", 2, grid_dim, &allowed).is_none()
        );
        // Garbage
        assert!(CharacterRecognitionAction::parse("foo()", 2, grid_dim, &allowed).is_none());
    }

    #[test]
    fn allowed_chars_count() {
        let chars = all_allowed_chars();
        assert_eq!(chars.len(), 59);
    }

    #[test]
    fn state_hash_eq() {
        use std::collections::hash_map::DefaultHasher;

        let allowed_names: Vec<String> = all_allowed_chars()
            .iter()
            .map(|c| char_to_name(*c).to_string())
            .collect();
        let puzzle = Arc::new(CharacterRecognitionPuzzle {
            character: "A".to_string(),
            character_char: 'A',
            font_idx: 0,
            grid_size: 16,
            num_pieces: 2,
            pieces: vec![],
            slots: vec![],
            fill_color: [255, 0, 0],
            piece_solve_order: vec![0, 1],
            allowed_chars: allowed_names,
            initial_rotations: vec![1, 0],
            piece_lookup: None,
            slot_grid_dim: 2,
            char_render_cache: std::sync::OnceLock::new(),
        });

        let s1 = CharacterRecognitionState {
            puzzle: puzzle.clone(),
            rotations: vec![1, 0],
            placements: vec![None, Some(0)],
            answer: None,
        };
        let s2 = CharacterRecognitionState {
            puzzle: puzzle.clone(),
            rotations: vec![1, 0],
            placements: vec![None, Some(0)],
            answer: None,
        };
        let s3 = CharacterRecognitionState {
            puzzle,
            rotations: vec![0, 0],
            placements: vec![None, Some(0)],
            answer: Some("A".to_string()),
        };

        assert_eq!(s1, s2);
        assert_ne!(s1, s3);

        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        s1.hash(&mut h1);
        s2.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }
}
