//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
use std::collections::{BTreeSet, VecDeque};
use std::sync::OnceLock;

use ab_glyph::{FontRef, PxScale};
use image::{Rgb, RgbImage};
use imageproc::drawing::draw_text_mut;
use rand::Rng;
use rand_chacha::ChaChaRng;

use super::state::*;

// ---------------------------------------------------------------------------
// Fonts
// ---------------------------------------------------------------------------

static DEJAVUSANS_BYTES: &[u8] = include_bytes!("../../../assets/fonts/DejaVuSans.ttf");
static INTER_BOLD_BYTES: &[u8] = include_bytes!("../../../assets/fonts/Inter-Bold.ttf");
static INTER_LIGHT_BYTES: &[u8] = include_bytes!("../../../assets/fonts/Inter-Light.ttf");
static INTER_REGULAR_BYTES: &[u8] = include_bytes!("../../../assets/fonts/Inter-Regular.ttf");
static NOTOSANS_BLACK_BYTES: &[u8] = include_bytes!("../../../assets/fonts/NotoSans-Black.ttf");
static NOTOSANS_BOLD_BYTES: &[u8] = include_bytes!("../../../assets/fonts/NotoSans-Bold.ttf");
static NOTOSANS_BOLDITALIC_BYTES: &[u8] =
    include_bytes!("../../../assets/fonts/NotoSans-BoldItalic.ttf");
static NOTOSANS_ITALIC_BYTES: &[u8] = include_bytes!("../../../assets/fonts/NotoSans-Italic.ttf");
static NOTOSANS_LIGHT_BYTES: &[u8] = include_bytes!("../../../assets/fonts/NotoSans-Light.ttf");
static NOTOSANS_REGULAR_BYTES: &[u8] = include_bytes!("../../../assets/fonts/NotoSans-Regular.ttf");
static NOTOSERIF_BOLD_BYTES: &[u8] = include_bytes!("../../../assets/fonts/NotoSerif-Bold.ttf");
static NOTOSERIF_BOLDITALIC_BYTES: &[u8] =
    include_bytes!("../../../assets/fonts/NotoSerif-BoldItalic.ttf");
static NOTOSERIF_ITALIC_BYTES: &[u8] = include_bytes!("../../../assets/fonts/NotoSerif-Italic.ttf");
static NOTOSERIF_REGULAR_BYTES: &[u8] =
    include_bytes!("../../../assets/fonts/NotoSerif-Regular.ttf");
static OSWALD_BOLD_BYTES: &[u8] = include_bytes!("../../../assets/fonts/Oswald-Bold.ttf");
static OSWALD_LIGHT_BYTES: &[u8] = include_bytes!("../../../assets/fonts/Oswald-Light.ttf");
static OSWALD_REGULAR_BYTES: &[u8] = include_bytes!("../../../assets/fonts/Oswald-Regular.ttf");
static PLAYFAIRDISPLAY_BLACK_BYTES: &[u8] =
    include_bytes!("../../../assets/fonts/PlayfairDisplay-Black.ttf");
static PLAYFAIRDISPLAY_BOLD_BYTES: &[u8] =
    include_bytes!("../../../assets/fonts/PlayfairDisplay-Bold.ttf");
static PLAYFAIRDISPLAY_BOLDITALIC_BYTES: &[u8] =
    include_bytes!("../../../assets/fonts/PlayfairDisplay-BoldItalic.ttf");
static PLAYFAIRDISPLAY_ITALIC_BYTES: &[u8] =
    include_bytes!("../../../assets/fonts/PlayfairDisplay-Italic.ttf");
static PLAYFAIRDISPLAY_REGULAR_BYTES: &[u8] =
    include_bytes!("../../../assets/fonts/PlayfairDisplay-Regular.ttf");
static ROBOTOBLACK_BYTES: &[u8] = include_bytes!("../../../assets/fonts/RobotoBlack.ttf");
static ROBOTOMONO_BOLD_BYTES: &[u8] = include_bytes!("../../../assets/fonts/RobotoMono-Bold.ttf");
static ROBOTOMONO_REGULAR_BYTES: &[u8] =
    include_bytes!("../../../assets/fonts/RobotoMono-Regular.ttf");
static ROBOTOSLAB_BOLD_BYTES: &[u8] = include_bytes!("../../../assets/fonts/RobotoSlab-Bold.ttf");
static ROBOTOSLAB_LIGHT_BYTES: &[u8] = include_bytes!("../../../assets/fonts/RobotoSlab-Light.ttf");
static ROBOTOSLAB_REGULAR_BYTES: &[u8] =
    include_bytes!("../../../assets/fonts/RobotoSlab-Regular.ttf");
static ZILLASLAB_BOLD_BYTES: &[u8] = include_bytes!("../../../assets/fonts/ZillaSlab-Bold.ttf");
static ZILLASLAB_BOLDITALIC_BYTES: &[u8] =
    include_bytes!("../../../assets/fonts/ZillaSlab-BoldItalic.ttf");
static ZILLASLAB_LIGHT_BYTES: &[u8] = include_bytes!("../../../assets/fonts/ZillaSlab-Light.ttf");

static FONTS: OnceLock<Vec<FontRef<'static>>> = OnceLock::new();

pub fn fonts() -> &'static [FontRef<'static>] {
    FONTS.get_or_init(|| {
        vec![
            FontRef::try_from_slice(DEJAVUSANS_BYTES).expect("invalid DejaVuSans font"),
            FontRef::try_from_slice(INTER_BOLD_BYTES).expect("invalid Inter-Bold font"),
            FontRef::try_from_slice(INTER_LIGHT_BYTES).expect("invalid Inter-Light font"),
            FontRef::try_from_slice(INTER_REGULAR_BYTES).expect("invalid Inter-Regular font"),
            FontRef::try_from_slice(NOTOSANS_BLACK_BYTES).expect("invalid NotoSans-Black font"),
            FontRef::try_from_slice(NOTOSANS_BOLD_BYTES).expect("invalid NotoSans-Bold font"),
            FontRef::try_from_slice(NOTOSANS_BOLDITALIC_BYTES)
                .expect("invalid NotoSans-BoldItalic font"),
            FontRef::try_from_slice(NOTOSANS_ITALIC_BYTES).expect("invalid NotoSans-Italic font"),
            FontRef::try_from_slice(NOTOSANS_LIGHT_BYTES).expect("invalid NotoSans-Light font"),
            FontRef::try_from_slice(NOTOSANS_REGULAR_BYTES).expect("invalid NotoSans-Regular font"),
            FontRef::try_from_slice(NOTOSERIF_BOLD_BYTES).expect("invalid NotoSerif-Bold font"),
            FontRef::try_from_slice(NOTOSERIF_BOLDITALIC_BYTES)
                .expect("invalid NotoSerif-BoldItalic font"),
            FontRef::try_from_slice(NOTOSERIF_ITALIC_BYTES).expect("invalid NotoSerif-Italic font"),
            FontRef::try_from_slice(NOTOSERIF_REGULAR_BYTES)
                .expect("invalid NotoSerif-Regular font"),
            FontRef::try_from_slice(OSWALD_BOLD_BYTES).expect("invalid Oswald-Bold font"),
            FontRef::try_from_slice(OSWALD_LIGHT_BYTES).expect("invalid Oswald-Light font"),
            FontRef::try_from_slice(OSWALD_REGULAR_BYTES).expect("invalid Oswald-Regular font"),
            FontRef::try_from_slice(PLAYFAIRDISPLAY_BLACK_BYTES)
                .expect("invalid PlayfairDisplay-Black font"),
            FontRef::try_from_slice(PLAYFAIRDISPLAY_BOLD_BYTES)
                .expect("invalid PlayfairDisplay-Bold font"),
            FontRef::try_from_slice(PLAYFAIRDISPLAY_BOLDITALIC_BYTES)
                .expect("invalid PlayfairDisplay-BoldItalic font"),
            FontRef::try_from_slice(PLAYFAIRDISPLAY_ITALIC_BYTES)
                .expect("invalid PlayfairDisplay-Italic font"),
            FontRef::try_from_slice(PLAYFAIRDISPLAY_REGULAR_BYTES)
                .expect("invalid PlayfairDisplay-Regular font"),
            FontRef::try_from_slice(ROBOTOBLACK_BYTES).expect("invalid RobotoBlack font"),
            FontRef::try_from_slice(ROBOTOMONO_BOLD_BYTES).expect("invalid RobotoMono-Bold font"),
            FontRef::try_from_slice(ROBOTOMONO_REGULAR_BYTES)
                .expect("invalid RobotoMono-Regular font"),
            FontRef::try_from_slice(ROBOTOSLAB_BOLD_BYTES).expect("invalid RobotoSlab-Bold font"),
            FontRef::try_from_slice(ROBOTOSLAB_LIGHT_BYTES).expect("invalid RobotoSlab-Light font"),
            FontRef::try_from_slice(ROBOTOSLAB_REGULAR_BYTES)
                .expect("invalid RobotoSlab-Regular font"),
            FontRef::try_from_slice(ZILLASLAB_BOLD_BYTES).expect("invalid ZillaSlab-Bold font"),
            FontRef::try_from_slice(ZILLASLAB_BOLDITALIC_BYTES)
                .expect("invalid ZillaSlab-BoldItalic font"),
            FontRef::try_from_slice(ZILLASLAB_LIGHT_BYTES).expect("invalid ZillaSlab-Light font"),
        ]
    })
}

fn is_greek(ch: char) -> bool {
    ('\u{0370}'..='\u{03FF}').contains(&ch) || ('\u{1F00}'..='\u{1FFF}').contains(&ch)
}

const GREEK_EXCLUDED_FONTS: &[&str] = &["Oswald", "Playfair", "ZillaSlab"];

fn font_indices_for_char(ch: char) -> Vec<usize> {
    let font_names: &[&str] = &[
        "DejaVuSans",
        "Inter-Bold",
        "Inter-Light",
        "Inter-Regular",
        "NotoSans-Black",
        "NotoSans-Bold",
        "NotoSans-BoldItalic",
        "NotoSans-Italic",
        "NotoSans-Light",
        "NotoSans-Regular",
        "NotoSerif-Bold",
        "NotoSerif-BoldItalic",
        "NotoSerif-Italic",
        "NotoSerif-Regular",
        "Oswald-Bold",
        "Oswald-Light",
        "Oswald-Regular",
        "PlayfairDisplay-Black",
        "PlayfairDisplay-Bold",
        "PlayfairDisplay-BoldItalic",
        "PlayfairDisplay-Italic",
        "PlayfairDisplay-Regular",
        "RobotoBlack",
        "RobotoMono-Bold",
        "RobotoMono-Regular",
        "RobotoSlab-Bold",
        "RobotoSlab-Light",
        "RobotoSlab-Regular",
        "ZillaSlab-Bold",
        "ZillaSlab-BoldItalic",
        "ZillaSlab-Light",
    ];
    if is_greek(ch) {
        (0..font_names.len())
            .filter(|&i| {
                !GREEK_EXCLUDED_FONTS
                    .iter()
                    .any(|exc| font_names[i].starts_with(exc))
            })
            .collect()
    } else {
        (0..font_names.len()).collect()
    }
}

// ---------------------------------------------------------------------------
// Color palette
// ---------------------------------------------------------------------------

const COLOR_PALETTE: &[[u8; 3]] = &[
    [0xCC, 0x33, 0x33],
    [0x33, 0x99, 0x33],
    [0x33, 0x55, 0xCC],
    [0xCC, 0x99, 0x33],
    [0x88, 0x33, 0xBB],
    [0x33, 0x99, 0x99],
    [0xCC, 0x55, 0x88],
    [0x66, 0x88, 0x33],
];

// ---------------------------------------------------------------------------
// Main generation
// ---------------------------------------------------------------------------

pub fn generate(
    rng: &mut ChaChaRng,
    num_pieces_range: (usize, usize),
    grid_size_range: (usize, usize),
    slot_grid_dim: usize,
) -> Option<CharacterRecognitionPuzzle> {
    let allowed = all_allowed_chars();
    let character = allowed[rng.gen_range(0..allowed.len())];
    let character_name = char_to_name(character).to_string();
    let allowed_names: Vec<String> = allowed
        .iter()
        .map(|c| char_to_name(*c).to_string())
        .collect();
    let valid_fonts = font_indices_for_char(character);
    let font_idx = valid_fonts[rng.gen_range(0..valid_fonts.len())];
    let grid_size = rng.gen_range(grid_size_range.0..=grid_size_range.1);
    let num_pieces = rng.gen_range(num_pieces_range.0..=num_pieces_range.1);
    let fill_color = COLOR_PALETTE[rng.gen_range(0..COLOR_PALETTE.len())];

    let font = &fonts()[font_idx];
    let cells = rasterize_character(font, character, grid_size);
    if cells.is_empty() {
        return None;
    }

    let (piece_cell_lists, piece_lookup) = decompose_via_graph(character, &cells, num_pieces, rng);

    let mut pieces = Vec::new();
    for pcells in &piece_cell_lists {
        let centroid = compute_centroid(pcells);
        pieces.push(PieceDef {
            cells: pcells.clone(),
            centroid,
        });
    }

    let slots = assign_grid_slots(&pieces, grid_size, slot_grid_dim);

    let initial_rotations: Vec<u8> = loop {
        let r: Vec<u8> = (0..num_pieces).map(|_| rng.gen_range(0..4u8)).collect();
        if r.iter().any(|&x| x != 0) {
            break r;
        }
    };

    let mut piece_solve_order: Vec<usize> = (0..num_pieces).collect();
    for i in (1..num_pieces).rev() {
        let j = rng.gen_range(0..=i);
        piece_solve_order.swap(i, j);
    }

    Some(CharacterRecognitionPuzzle {
        character: character_name,
        character_char: character,
        font_idx,
        grid_size,
        num_pieces,
        pieces,
        slots,
        fill_color,
        piece_solve_order,
        allowed_chars: allowed_names,
        initial_rotations,
        piece_lookup: Some(piece_lookup),
        slot_grid_dim,
        char_render_cache: std::sync::OnceLock::new(),
    })
}

pub fn trivial(num_pieces: usize, slot_grid_dim: usize) -> CharacterRecognitionPuzzle {
    let grid_size = 16;
    let num_pieces = num_pieces.clamp(2, 3);

    let mut all_cells = Vec::new();
    for y in 2..12 {
        for x in 4..7 {
            all_cells.push((x, y));
        }
    }
    for y in 9..12 {
        for x in 7..11 {
            all_cells.push((x, y));
        }
    }

    let piece_cells: Vec<Vec<(i32, i32)>> = if num_pieces == 3 {
        vec![
            all_cells.iter().filter(|&&(_, y)| y < 6).copied().collect(),
            all_cells
                .iter()
                .filter(|&&(x, y)| y >= 6 && x < 7)
                .copied()
                .collect(),
            all_cells
                .iter()
                .filter(|&&(x, y)| y >= 6 && x >= 7)
                .copied()
                .collect(),
        ]
    } else {
        vec![
            all_cells.iter().filter(|&&(_, y)| y < 9).copied().collect(),
            all_cells
                .iter()
                .filter(|&&(_, y)| y >= 9)
                .copied()
                .collect(),
        ]
    };

    let pieces: Vec<PieceDef> = piece_cells
        .iter()
        .map(|cells| PieceDef {
            cells: cells.clone(),
            centroid: compute_centroid(cells),
        })
        .collect();
    let slots = assign_grid_slots(&pieces, grid_size, slot_grid_dim);
    let allowed_names: Vec<String> = all_allowed_chars()
        .iter()
        .map(|c| char_to_name(*c).to_string())
        .collect();

    CharacterRecognitionPuzzle {
        character: "L".to_string(),
        character_char: 'L',
        font_idx: 0,
        grid_size,
        num_pieces,
        pieces,
        slots,
        fill_color: COLOR_PALETTE[0],
        piece_solve_order: (0..num_pieces).collect(),
        allowed_chars: allowed_names,
        initial_rotations: vec![1; num_pieces],
        piece_lookup: None,
        slot_grid_dim,
        char_render_cache: std::sync::OnceLock::new(),
    }
}

// ---------------------------------------------------------------------------
// Character rasterization
// ---------------------------------------------------------------------------

fn rasterize_character(font: &FontRef, ch: char, grid_size: usize) -> BTreeSet<(i32, i32)> {
    let render_size = 400u32;
    let mut img = RgbImage::from_pixel(render_size, render_size, Rgb([0, 0, 0]));
    let font_height = render_size as f32 * 0.70;
    let scale = PxScale::from(font_height);

    let x = (render_size as i32) / 6;
    let y = (render_size as i32) / 6;

    draw_text_mut(
        &mut img,
        Rgb([255, 255, 255]),
        x,
        y,
        scale,
        font,
        &ch.to_string(),
    );

    let mut min_px = render_size;
    let mut min_py = render_size;
    let mut max_px = 0u32;
    let mut max_py = 0u32;
    for py in 0..render_size {
        for px in 0..render_size {
            if img.get_pixel(px, py)[0] > 128 {
                min_px = min_px.min(px);
                min_py = min_py.min(py);
                max_px = max_px.max(px);
                max_py = max_py.max(py);
            }
        }
    }

    if max_px < min_px {
        return BTreeSet::new();
    }

    let src_w = (max_px - min_px + 1) as f64;
    let src_h = (max_py - min_py + 1) as f64;

    let margin = 1;
    let available = (grid_size - 2 * margin) as f64;
    if available <= 0.0 {
        return BTreeSet::new();
    }
    let sx = available / src_w;
    let sy = available / src_h;
    let scale_factor = sx.min(sy);

    let dest_w = (src_w * scale_factor).round() as usize;
    let dest_h = (src_h * scale_factor).round() as usize;
    let off_x = (grid_size - dest_w) / 2;
    let off_y = (grid_size - dest_h) / 2;

    let mut cells = BTreeSet::new();
    for gy in 0..grid_size {
        for gx in 0..grid_size {
            if gx < off_x || gy < off_y {
                continue;
            }
            let rel_x = gx - off_x;
            let rel_y = gy - off_y;
            if rel_x >= dest_w || rel_y >= dest_h {
                continue;
            }
            let src_fx = rel_x as f64 / scale_factor + min_px as f64;
            let src_fy = rel_y as f64 / scale_factor + min_py as f64;
            let src_px = src_fx.round() as u32;
            let src_py = src_fy.round() as u32;

            if src_px < render_size && src_py < render_size {
                if img.get_pixel(src_px, src_py)[0] > 128 {
                    cells.insert((gx as i32, gy as i32));
                }
            }
        }
    }
    cells
}

// ---------------------------------------------------------------------------
// Decomposition
// ---------------------------------------------------------------------------

pub fn compute_centroid(cells: &[(i32, i32)]) -> (f64, f64) {
    if cells.is_empty() {
        return (0.0, 0.0);
    }
    let sum_x: f64 = cells.iter().map(|&(x, _)| x as f64).sum();
    let sum_y: f64 = cells.iter().map(|&(_, y)| y as f64).sum();
    let n = cells.len() as f64;
    (sum_x / n, sum_y / n)
}

fn assign_grid_slots(pieces: &[PieceDef], grid_size: usize, slot_grid_dim: usize) -> Vec<SlotDef> {
    let n = slot_grid_dim;
    let mut intersections: Vec<((usize, usize), (f64, f64))> = Vec::with_capacity(n * n);
    for row in 0..n {
        for col in 0..n {
            let cx = (col + 1) as f64 * grid_size as f64 / (n + 1) as f64;
            let cy = (row + 1) as f64 * grid_size as f64 / (n + 1) as f64;
            intersections.push(((row, col), (cx, cy)));
        }
    }

    let mut pairs: Vec<(usize, usize, f64)> = Vec::new();
    for (pi, piece) in pieces.iter().enumerate() {
        for (ii, &(_, (cx, cy))) in intersections.iter().enumerate() {
            let dx = piece.centroid.0 - cx;
            let dy = piece.centroid.1 - cy;
            pairs.push((pi, ii, dx * dx + dy * dy));
        }
    }
    pairs.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    let mut piece_assigned = vec![false; pieces.len()];
    let mut cell_assigned = vec![false; intersections.len()];
    let mut cell_to_piece: Vec<Option<usize>> = vec![None; intersections.len()];

    for &(pi, ii, _) in &pairs {
        if !piece_assigned[pi] && !cell_assigned[ii] {
            piece_assigned[pi] = true;
            cell_assigned[ii] = true;
            cell_to_piece[ii] = Some(pi);
        }
    }

    intersections
        .iter()
        .enumerate()
        .map(|(ii, &(id, centroid))| SlotDef {
            id,
            centroid,
            matching_piece_idx: cell_to_piece[ii],
            correct_rotation: 0,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Abstract character graphs
// ---------------------------------------------------------------------------

struct CharGraph {
    points: Vec<(f64, f64)>,
    edges: Vec<(usize, usize)>,
}

fn char_graph(ch: char) -> CharGraph {
    let (points, edges): (Vec<(f64, f64)>, Vec<(usize, usize)>) = match ch {
        'A' => (
            vec![
                (0.10, 0.90),
                (0.17, 0.87),
                (0.20, 0.77),
                (0.23, 0.70),
                (0.27, 0.63),
                (0.30, 0.53),
                (0.30, 0.47),
                (0.33, 0.37),
                (0.37, 0.30),
                (0.40, 0.20),
                (0.47, 0.13),
                (0.33, 0.67),
                (0.43, 0.67),
                (0.53, 0.67),
                (0.63, 0.67),
                (0.70, 0.63),
                (0.47, 0.07),
                (0.47, 0.10),
                (0.50, 0.13),
                (0.57, 0.20),
                (0.60, 0.30),
                (0.63, 0.37),
                (0.67, 0.47),
                (0.67, 0.53),
                (0.77, 0.70),
                (0.77, 0.80),
                (0.83, 0.90),
                (0.90, 0.93),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (4, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (16, 17),
                (10, 17),
                (17, 18),
                (10, 18),
                (18, 19),
                (19, 20),
                (20, 21),
                (21, 22),
                (22, 23),
                (15, 23),
                (15, 24),
                (24, 25),
                (25, 26),
                (26, 27),
            ],
        ),
        'B' => (
            vec![
                (0.20, 0.43),
                (0.20, 0.33),
                (0.20, 0.23),
                (0.20, 0.13),
                (0.23, 0.07),
                (0.33, 0.07),
                (0.43, 0.07),
                (0.53, 0.10),
                (0.63, 0.10),
                (0.70, 0.17),
                (0.70, 0.27),
                (0.70, 0.37),
                (0.63, 0.43),
                (0.20, 0.47),
                (0.23, 0.47),
                (0.20, 0.50),
                (0.20, 0.60),
                (0.20, 0.70),
                (0.20, 0.80),
                (0.23, 0.90),
                (0.33, 0.90),
                (0.43, 0.90),
                (0.53, 0.90),
                (0.60, 0.90),
                (0.70, 0.83),
                (0.73, 0.77),
                (0.73, 0.67),
                (0.70, 0.57),
                (0.63, 0.50),
                (0.33, 0.47),
                (0.40, 0.47),
                (0.50, 0.47),
                (0.60, 0.47),
                (0.63, 0.47),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (0, 13),
                (0, 14),
                (13, 15),
                (13, 14),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),
                (19, 20),
                (20, 21),
                (21, 22),
                (22, 23),
                (23, 24),
                (24, 25),
                (25, 26),
                (26, 27),
                (27, 28),
                (14, 15),
                (14, 29),
                (29, 30),
                (30, 31),
                (31, 32),
                (12, 32),
                (32, 33),
                (28, 32),
                (12, 33),
                (28, 33),
            ],
        ),
        'C' => (
            vec![
                (0.77, 0.13),
                (0.67, 0.10),
                (0.60, 0.07),
                (0.50, 0.07),
                (0.40, 0.10),
                (0.33, 0.13),
                (0.27, 0.23),
                (0.23, 0.30),
                (0.20, 0.40),
                (0.17, 0.50),
                (0.20, 0.57),
                (0.20, 0.67),
                (0.27, 0.77),
                (0.33, 0.83),
                (0.40, 0.87),
                (0.50, 0.90),
                (0.60, 0.90),
                (0.70, 0.90),
                (0.80, 0.83),
                (0.83, 0.77),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),
            ],
        ),
        'D' => (
            vec![
                (0.13, 0.10),
                (0.13, 0.20),
                (0.13, 0.30),
                (0.13, 0.40),
                (0.13, 0.50),
                (0.13, 0.60),
                (0.13, 0.70),
                (0.13, 0.80),
                (0.17, 0.90),
                (0.27, 0.90),
                (0.37, 0.90),
                (0.47, 0.90),
                (0.57, 0.87),
                (0.63, 0.87),
                (0.73, 0.77),
                (0.77, 0.70),
                (0.80, 0.60),
                (0.80, 0.50),
                (0.80, 0.40),
                (0.80, 0.33),
                (0.73, 0.23),
                (0.70, 0.17),
                (0.60, 0.10),
                (0.50, 0.10),
                (0.40, 0.10),
                (0.33, 0.07),
                (0.23, 0.07),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),
                (19, 20),
                (20, 21),
                (21, 22),
                (22, 23),
                (23, 24),
                (24, 25),
                (25, 26),
                (0, 26),
            ],
        ),
        'E' => (
            vec![
                (0.23, 0.43),
                (0.23, 0.33),
                (0.23, 0.23),
                (0.23, 0.13),
                (0.30, 0.07),
                (0.40, 0.07),
                (0.50, 0.07),
                (0.60, 0.07),
                (0.70, 0.07),
                (0.23, 0.47),
                (0.27, 0.47),
                (0.23, 0.50),
                (0.23, 0.60),
                (0.23, 0.70),
                (0.23, 0.80),
                (0.27, 0.90),
                (0.33, 0.90),
                (0.43, 0.90),
                (0.53, 0.90),
                (0.63, 0.90),
                (0.73, 0.87),
                (0.37, 0.47),
                (0.50, 0.47),
                (0.60, 0.47),
                (0.70, 0.43),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (0, 9),
                (0, 10),
                (9, 11),
                (9, 10),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),
                (19, 20),
                (10, 11),
                (10, 21),
                (21, 22),
                (22, 23),
                (23, 24),
            ],
        ),
        'F' => (
            vec![
                (0.27, 0.43),
                (0.27, 0.33),
                (0.27, 0.23),
                (0.27, 0.10),
                (0.33, 0.07),
                (0.47, 0.07),
                (0.57, 0.07),
                (0.67, 0.07),
                (0.27, 0.47),
                (0.30, 0.47),
                (0.27, 0.50),
                (0.27, 0.60),
                (0.27, 0.70),
                (0.27, 0.80),
                (0.27, 0.90),
                (0.40, 0.47),
                (0.50, 0.47),
                (0.60, 0.47),
                (0.67, 0.43),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (0, 8),
                (0, 9),
                (8, 10),
                (8, 9),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (9, 10),
                (9, 15),
                (15, 16),
                (16, 17),
                (17, 18),
            ],
        ),
        'G' => (
            vec![
                (0.57, 0.53),
                (0.67, 0.53),
                (0.77, 0.53),
                (0.80, 0.63),
                (0.80, 0.73),
                (0.80, 0.83),
                (0.73, 0.87),
                (0.63, 0.90),
                (0.53, 0.90),
                (0.43, 0.90),
                (0.33, 0.87),
                (0.27, 0.80),
                (0.20, 0.73),
                (0.17, 0.63),
                (0.17, 0.53),
                (0.13, 0.47),
                (0.17, 0.37),
                (0.20, 0.27),
                (0.23, 0.20),
                (0.30, 0.13),
                (0.40, 0.10),
                (0.50, 0.07),
                (0.60, 0.07),
                (0.67, 0.10),
                (0.77, 0.13),
                (0.83, 0.20),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),
                (19, 20),
                (20, 21),
                (21, 22),
                (22, 23),
                (23, 24),
                (24, 25),
            ],
        ),
        'J' => (
            vec![
                (0.37, 0.90),
                (0.47, 0.90),
                (0.53, 0.87),
                (0.53, 0.77),
                (0.53, 0.67),
                (0.53, 0.57),
                (0.53, 0.47),
                (0.53, 0.37),
                (0.53, 0.27),
                (0.53, 0.17),
                (0.53, 0.07),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
            ],
        ),
        'K' => (
            vec![
                (0.17, 0.43),
                (0.17, 0.33),
                (0.17, 0.23),
                (0.17, 0.13),
                (0.20, 0.03),
                (0.17, 0.47),
                (0.20, 0.47),
                (0.17, 0.50),
                (0.17, 0.60),
                (0.17, 0.70),
                (0.17, 0.80),
                (0.17, 0.90),
                (0.33, 0.47),
                (0.37, 0.40),
                (0.43, 0.33),
                (0.50, 0.27),
                (0.57, 0.20),
                (0.63, 0.13),
                (0.73, 0.07),
                (0.80, 0.03),
                (0.37, 0.53),
                (0.47, 0.63),
                (0.53, 0.70),
                (0.60, 0.77),
                (0.67, 0.83),
                (0.73, 0.90),
                (0.83, 0.93),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (0, 5),
                (0, 6),
                (5, 7),
                (5, 6),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (6, 7),
                (6, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),
                (12, 20),
                (20, 21),
                (21, 22),
                (22, 23),
                (23, 24),
                (24, 25),
                (25, 26),
            ],
        ),
        'L' => (
            vec![
                (0.27, 0.03),
                (0.23, 0.13),
                (0.23, 0.23),
                (0.23, 0.33),
                (0.23, 0.43),
                (0.23, 0.53),
                (0.23, 0.63),
                (0.23, 0.73),
                (0.23, 0.83),
                (0.30, 0.90),
                (0.40, 0.90),
                (0.50, 0.90),
                (0.60, 0.90),
                (0.70, 0.87),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13),
            ],
        ),
        'P' => (
            vec![
                (0.23, 0.50),
                (0.23, 0.40),
                (0.23, 0.30),
                (0.23, 0.20),
                (0.23, 0.10),
                (0.30, 0.07),
                (0.40, 0.07),
                (0.50, 0.10),
                (0.60, 0.10),
                (0.67, 0.13),
                (0.70, 0.23),
                (0.70, 0.33),
                (0.70, 0.43),
                (0.63, 0.50),
                (0.57, 0.53),
                (0.47, 0.53),
                (0.37, 0.53),
                (0.27, 0.53),
                (0.23, 0.53),
                (0.23, 0.57),
                (0.23, 0.67),
                (0.23, 0.80),
                (0.23, 0.90),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (0, 18),
                (0, 17),
                (18, 19),
                (17, 18),
                (19, 20),
                (20, 21),
                (21, 22),
                (17, 19),
            ],
        ),
        'Q' => (
            vec![
                (0.57, 0.80),
                (0.47, 0.77),
                (0.37, 0.77),
                (0.30, 0.73),
                (0.23, 0.63),
                (0.20, 0.57),
                (0.20, 0.47),
                (0.20, 0.40),
                (0.20, 0.30),
                (0.23, 0.20),
                (0.30, 0.13),
                (0.40, 0.07),
                (0.50, 0.07),
                (0.57, 0.07),
                (0.67, 0.13),
                (0.73, 0.20),
                (0.77, 0.30),
                (0.80, 0.37),
                (0.80, 0.47),
                (0.77, 0.57),
                (0.73, 0.67),
                (0.67, 0.73),
                (0.60, 0.80),
                (0.60, 0.83),
                (0.67, 0.90),
                (0.73, 0.93),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),
                (19, 20),
                (20, 21),
                (21, 22),
                (0, 22),
                (0, 23),
                (22, 23),
                (23, 24),
                (24, 25),
            ],
        ),
        'R' => (
            vec![
                (0.17, 0.47),
                (0.17, 0.37),
                (0.17, 0.27),
                (0.17, 0.17),
                (0.20, 0.07),
                (0.30, 0.07),
                (0.40, 0.07),
                (0.50, 0.10),
                (0.60, 0.13),
                (0.63, 0.20),
                (0.63, 0.33),
                (0.63, 0.43),
                (0.57, 0.50),
                (0.17, 0.50),
                (0.20, 0.50),
                (0.17, 0.53),
                (0.17, 0.63),
                (0.17, 0.70),
                (0.17, 0.80),
                (0.17, 0.90),
                (0.30, 0.50),
                (0.43, 0.50),
                (0.53, 0.50),
                (0.57, 0.53),
                (0.60, 0.63),
                (0.67, 0.70),
                (0.70, 0.80),
                (0.73, 0.87),
                (0.83, 0.93),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (0, 13),
                (0, 14),
                (13, 15),
                (13, 14),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),
                (14, 15),
                (14, 20),
                (20, 21),
                (21, 22),
                (12, 22),
                (22, 23),
                (12, 23),
                (23, 24),
                (24, 25),
                (25, 26),
                (26, 27),
                (27, 28),
            ],
        ),
        'T' => (
            vec![
                (0.13, 0.10),
                (0.23, 0.07),
                (0.33, 0.07),
                (0.43, 0.07),
                (0.47, 0.07),
                (0.47, 0.10),
                (0.50, 0.07),
                (0.47, 0.20),
                (0.47, 0.30),
                (0.47, 0.40),
                (0.47, 0.50),
                (0.47, 0.60),
                (0.47, 0.70),
                (0.47, 0.80),
                (0.47, 0.90),
                (0.60, 0.07),
                (0.67, 0.07),
                (0.77, 0.07),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (3, 5),
                (4, 5),
                (4, 6),
                (5, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (5, 6),
                (6, 15),
                (15, 16),
                (16, 17),
            ],
        ),
        'U' => (
            vec![
                (0.20, 0.03),
                (0.17, 0.13),
                (0.17, 0.23),
                (0.17, 0.30),
                (0.17, 0.40),
                (0.17, 0.50),
                (0.17, 0.60),
                (0.20, 0.70),
                (0.23, 0.77),
                (0.30, 0.87),
                (0.37, 0.90),
                (0.47, 0.90),
                (0.57, 0.90),
                (0.67, 0.87),
                (0.73, 0.80),
                (0.73, 0.73),
                (0.77, 0.63),
                (0.77, 0.53),
                (0.77, 0.43),
                (0.77, 0.33),
                (0.77, 0.23),
                (0.77, 0.13),
                (0.77, 0.03),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),
                (19, 20),
                (20, 21),
                (21, 22),
            ],
        ),
        'V' => (
            vec![
                (0.10, 0.03),
                (0.13, 0.10),
                (0.20, 0.20),
                (0.20, 0.27),
                (0.23, 0.37),
                (0.27, 0.43),
                (0.30, 0.53),
                (0.37, 0.63),
                (0.37, 0.70),
                (0.43, 0.80),
                (0.50, 0.83),
                (0.57, 0.77),
                (0.60, 0.67),
                (0.63, 0.60),
                (0.67, 0.50),
                (0.70, 0.43),
                (0.73, 0.33),
                (0.77, 0.27),
                (0.80, 0.17),
                (0.83, 0.10),
                (0.90, 0.03),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),
                (19, 20),
            ],
        ),
        'Y' => (
            vec![
                (0.10, 0.03),
                (0.17, 0.07),
                (0.23, 0.13),
                (0.30, 0.23),
                (0.33, 0.30),
                (0.40, 0.37),
                (0.43, 0.43),
                (0.47, 0.43),
                (0.47, 0.47),
                (0.50, 0.43),
                (0.47, 0.57),
                (0.47, 0.67),
                (0.47, 0.80),
                (0.47, 0.90),
                (0.57, 0.37),
                (0.63, 0.30),
                (0.67, 0.23),
                (0.73, 0.13),
                (0.80, 0.07),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (6, 8),
                (7, 8),
                (7, 9),
                (8, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (8, 9),
                (9, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
            ],
        ),
        'a' => (
            vec![
                (0.17, 0.17),
                (0.23, 0.13),
                (0.33, 0.10),
                (0.43, 0.10),
                (0.50, 0.10),
                (0.60, 0.10),
                (0.67, 0.17),
                (0.73, 0.23),
                (0.77, 0.30),
                (0.77, 0.40),
                (0.73, 0.43),
                (0.63, 0.43),
                (0.53, 0.43),
                (0.43, 0.43),
                (0.33, 0.47),
                (0.27, 0.47),
                (0.20, 0.57),
                (0.17, 0.63),
                (0.17, 0.73),
                (0.23, 0.83),
                (0.30, 0.87),
                (0.40, 0.90),
                (0.50, 0.90),
                (0.57, 0.87),
                (0.63, 0.80),
                (0.73, 0.77),
                (0.77, 0.43),
                (0.77, 0.47),
                (0.77, 0.73),
                (0.77, 0.77),
                (0.77, 0.80),
                (0.77, 0.57),
                (0.77, 0.63),
                (0.77, 0.90),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),
                (19, 20),
                (20, 21),
                (21, 22),
                (22, 23),
                (23, 24),
                (24, 25),
                (9, 10),
                (10, 26),
                (10, 27),
                (25, 28),
                (25, 29),
                (25, 30),
                (9, 26),
                (26, 27),
                (27, 31),
                (31, 32),
                (28, 32),
                (28, 29),
                (29, 30),
                (30, 33),
            ],
        ),
        'c' => (
            vec![
                (0.80, 0.17),
                (0.73, 0.10),
                (0.63, 0.10),
                (0.53, 0.10),
                (0.43, 0.10),
                (0.33, 0.17),
                (0.27, 0.23),
                (0.23, 0.30),
                (0.20, 0.40),
                (0.20, 0.50),
                (0.20, 0.60),
                (0.23, 0.70),
                (0.30, 0.77),
                (0.37, 0.83),
                (0.47, 0.87),
                (0.53, 0.90),
                (0.63, 0.90),
                (0.73, 0.87),
                (0.80, 0.80),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
            ],
        ),
        'e' => (
            vec![
                (0.13, 0.43),
                (0.17, 0.33),
                (0.20, 0.27),
                (0.23, 0.20),
                (0.33, 0.13),
                (0.40, 0.10),
                (0.50, 0.10),
                (0.60, 0.10),
                (0.67, 0.13),
                (0.77, 0.23),
                (0.80, 0.30),
                (0.80, 0.40),
                (0.77, 0.47),
                (0.67, 0.47),
                (0.57, 0.47),
                (0.47, 0.47),
                (0.37, 0.47),
                (0.27, 0.47),
                (0.17, 0.47),
                (0.13, 0.47),
                (0.13, 0.50),
                (0.13, 0.60),
                (0.17, 0.70),
                (0.23, 0.77),
                (0.30, 0.83),
                (0.40, 0.87),
                (0.47, 0.90),
                (0.57, 0.90),
                (0.67, 0.87),
                (0.77, 0.87),
                (0.83, 0.80),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (0, 19),
                (0, 18),
                (19, 20),
                (18, 19),
                (20, 21),
                (21, 22),
                (22, 23),
                (23, 24),
                (24, 25),
                (25, 26),
                (26, 27),
                (27, 28),
                (28, 29),
                (29, 30),
                (18, 20),
            ],
        ),
        'f' => (
            vec![
                (0.27, 0.33),
                (0.40, 0.33),
                (0.43, 0.30),
                (0.43, 0.33),
                (0.43, 0.37),
                (0.43, 0.20),
                (0.47, 0.13),
                (0.53, 0.07),
                (0.63, 0.03),
                (0.47, 0.33),
                (0.43, 0.47),
                (0.43, 0.57),
                (0.43, 0.67),
                (0.43, 0.77),
                (0.43, 0.87),
                (0.60, 0.30),
            ],
            vec![
                (0, 1),
                (1, 2),
                (1, 3),
                (1, 4),
                (2, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (2, 3),
                (2, 9),
                (3, 4),
                (3, 9),
                (4, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (4, 9),
                (9, 15),
            ],
        ),
        'g' => (
            vec![
                (0.30, 0.83),
                (0.37, 0.90),
                (0.47, 0.90),
                (0.57, 0.90),
                (0.63, 0.87),
                (0.70, 0.80),
                (0.73, 0.70),
                (0.70, 0.60),
                (0.67, 0.13),
                (0.57, 0.10),
                (0.50, 0.07),
                (0.40, 0.10),
                (0.33, 0.13),
                (0.27, 0.23),
                (0.27, 0.33),
                (0.27, 0.40),
                (0.27, 0.50),
                (0.33, 0.60),
                (0.40, 0.63),
                (0.50, 0.67),
                (0.57, 0.63),
                (0.67, 0.60),
                (0.70, 0.13),
                (0.70, 0.17),
                (0.70, 0.57),
                (0.73, 0.07),
                (0.73, 0.27),
                (0.73, 0.37),
                (0.73, 0.47),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),
                (19, 20),
                (20, 21),
                (8, 22),
                (8, 23),
                (21, 24),
                (7, 21),
                (22, 23),
                (22, 25),
                (23, 26),
                (26, 27),
                (27, 28),
                (24, 28),
                (7, 24),
            ],
        ),
        'h' => (
            vec![
                (0.23, 0.37),
                (0.23, 0.27),
                (0.23, 0.13),
                (0.27, 0.03),
                (0.23, 0.40),
                (0.27, 0.40),
                (0.23, 0.43),
                (0.23, 0.53),
                (0.23, 0.63),
                (0.23, 0.70),
                (0.23, 0.80),
                (0.23, 0.90),
                (0.37, 0.37),
                (0.43, 0.33),
                (0.57, 0.33),
                (0.63, 0.37),
                (0.70, 0.47),
                (0.70, 0.57),
                (0.70, 0.67),
                (0.70, 0.77),
                (0.70, 0.87),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (0, 4),
                (0, 5),
                (4, 6),
                (4, 5),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (5, 6),
                (5, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),
                (19, 20),
            ],
        ),
        'j' => (
            vec![
                (0.40, 0.90),
                (0.50, 0.90),
                (0.53, 0.80),
                (0.53, 0.70),
                (0.53, 0.60),
                (0.53, 0.50),
                (0.53, 0.40),
                (0.53, 0.30),
                (0.53, 0.07),
                (0.53, 0.10),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (8, 9),
            ],
        ),
        'k' => (
            vec![
                (0.23, 0.57),
                (0.23, 0.47),
                (0.23, 0.33),
                (0.23, 0.23),
                (0.23, 0.13),
                (0.27, 0.03),
                (0.23, 0.60),
                (0.27, 0.60),
                (0.23, 0.63),
                (0.23, 0.73),
                (0.23, 0.80),
                (0.23, 0.90),
                (0.37, 0.60),
                (0.40, 0.57),
                (0.40, 0.60),
                (0.40, 0.63),
                (0.47, 0.50),
                (0.53, 0.47),
                (0.60, 0.40),
                (0.67, 0.33),
                (0.77, 0.30),
                (0.47, 0.70),
                (0.53, 0.77),
                (0.60, 0.83),
                (0.67, 0.90),
                (0.73, 0.93),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (0, 6),
                (0, 7),
                (6, 8),
                (6, 7),
                (8, 9),
                (9, 10),
                (10, 11),
                (7, 8),
                (7, 12),
                (12, 13),
                (12, 14),
                (12, 15),
                (13, 14),
                (13, 16),
                (16, 17),
                (17, 18),
                (18, 19),
                (19, 20),
                (14, 15),
                (15, 21),
                (21, 22),
                (22, 23),
                (23, 24),
                (24, 25),
            ],
        ),
        'm' => (
            vec![
                (0.07, 0.77),
                (0.07, 0.67),
                (0.07, 0.57),
                (0.07, 0.47),
                (0.07, 0.37),
                (0.10, 0.27),
                (0.10, 0.20),
                (0.10, 0.23),
                (0.13, 0.27),
                (0.23, 0.23),
                (0.33, 0.20),
                (0.43, 0.27),
                (0.50, 0.33),
                (0.50, 0.37),
                (0.53, 0.33),
                (0.50, 0.47),
                (0.50, 0.53),
                (0.50, 0.63),
                (0.50, 0.73),
                (0.60, 0.27),
                (0.67, 0.20),
                (0.77, 0.20),
                (0.87, 0.27),
                (0.90, 0.33),
                (0.90, 0.43),
                (0.90, 0.53),
                (0.90, 0.63),
                (0.90, 0.73),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (6, 7),
                (5, 7),
                (7, 8),
                (5, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (12, 14),
                (13, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (13, 14),
                (14, 19),
                (19, 20),
                (20, 21),
                (21, 22),
                (22, 23),
                (23, 24),
                (24, 25),
                (25, 26),
                (26, 27),
            ],
        ),
        'r' => (
            vec![
                (0.27, 0.90),
                (0.30, 0.80),
                (0.30, 0.70),
                (0.30, 0.60),
                (0.30, 0.47),
                (0.30, 0.37),
                (0.30, 0.27),
                (0.30, 0.20),
                (0.33, 0.10),
                (0.30, 0.23),
                (0.33, 0.23),
                (0.40, 0.20),
                (0.50, 0.13),
                (0.57, 0.10),
                (0.67, 0.10),
                (0.70, 0.03),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (7, 8),
                (7, 9),
                (7, 10),
                (6, 9),
                (9, 10),
                (6, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
            ],
        ),
        't' => (
            vec![
                (0.27, 0.27),
                (0.37, 0.27),
                (0.40, 0.23),
                (0.40, 0.27),
                (0.40, 0.30),
                (0.40, 0.03),
                (0.40, 0.13),
                (0.43, 0.27),
                (0.40, 0.40),
                (0.40, 0.50),
                (0.40, 0.60),
                (0.40, 0.70),
                (0.40, 0.80),
                (0.47, 0.87),
                (0.57, 0.90),
                (0.67, 0.87),
                (0.57, 0.27),
                (0.67, 0.23),
            ],
            vec![
                (0, 1),
                (1, 2),
                (1, 3),
                (1, 4),
                (5, 6),
                (2, 6),
                (2, 3),
                (2, 7),
                (3, 4),
                (3, 7),
                (4, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (4, 7),
                (7, 16),
                (16, 17),
            ],
        ),
        'v' => (
            vec![
                (0.07, 0.03),
                (0.13, 0.10),
                (0.17, 0.20),
                (0.20, 0.27),
                (0.23, 0.37),
                (0.27, 0.47),
                (0.30, 0.53),
                (0.33, 0.63),
                (0.37, 0.70),
                (0.40, 0.80),
                (0.47, 0.87),
                (0.57, 0.80),
                (0.60, 0.73),
                (0.63, 0.63),
                (0.67, 0.57),
                (0.67, 0.47),
                (0.73, 0.37),
                (0.77, 0.30),
                (0.77, 0.20),
                (0.83, 0.10),
                (0.87, 0.03),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),
                (19, 20),
            ],
        ),
        'w' => (
            vec![
                (0.07, 0.13),
                (0.10, 0.23),
                (0.13, 0.30),
                (0.17, 0.40),
                (0.17, 0.50),
                (0.20, 0.60),
                (0.23, 0.67),
                (0.30, 0.67),
                (0.37, 0.57),
                (0.37, 0.47),
                (0.40, 0.40),
                (0.43, 0.30),
                (0.47, 0.20),
                (0.53, 0.23),
                (0.57, 0.33),
                (0.60, 0.43),
                (0.60, 0.50),
                (0.63, 0.60),
                (0.67, 0.70),
                (0.77, 0.67),
                (0.77, 0.60),
                (0.80, 0.50),
                (0.83, 0.40),
                (0.87, 0.33),
                (0.87, 0.23),
                (0.90, 0.13),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),
                (19, 20),
                (20, 21),
                (21, 22),
                (22, 23),
                (23, 24),
                (24, 25),
            ],
        ),
        'y' => (
            vec![
                (0.20, 0.03),
                (0.23, 0.10),
                (0.30, 0.20),
                (0.30, 0.27),
                (0.37, 0.37),
                (0.37, 0.43),
                (0.43, 0.53),
                (0.47, 0.60),
                (0.23, 0.90),
                (0.33, 0.90),
                (0.40, 0.87),
                (0.43, 0.80),
                (0.47, 0.73),
                (0.50, 0.63),
                (0.50, 0.60),
                (0.57, 0.53),
                (0.60, 0.43),
                (0.63, 0.33),
                (0.67, 0.23),
                (0.70, 0.17),
                (0.77, 0.07),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (7, 14),
                (7, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),
                (19, 20),
            ],
        ),
        '2' => (
            vec![
                (0.20, 0.90),
                (0.30, 0.87),
                (0.23, 0.13),
                (0.33, 0.10),
                (0.43, 0.10),
                (0.53, 0.10),
                (0.60, 0.10),
                (0.70, 0.20),
                (0.70, 0.27),
                (0.70, 0.37),
                (0.63, 0.47),
                (0.60, 0.53),
                (0.53, 0.60),
                (0.47, 0.67),
                (0.37, 0.77),
                (0.33, 0.83),
                (0.33, 0.87),
                (0.43, 0.90),
                (0.53, 0.90),
                (0.63, 0.90),
                (0.73, 0.87),
            ],
            vec![
                (0, 1),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (1, 15),
                (1, 16),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),
                (19, 20),
            ],
        ),
        '3' => (
            vec![
                (0.23, 0.87),
                (0.33, 0.90),
                (0.43, 0.90),
                (0.53, 0.90),
                (0.60, 0.87),
                (0.70, 0.80),
                (0.73, 0.73),
                (0.73, 0.63),
                (0.67, 0.53),
                (0.60, 0.47),
                (0.27, 0.10),
                (0.37, 0.10),
                (0.43, 0.07),
                (0.53, 0.07),
                (0.63, 0.13),
                (0.70, 0.20),
                (0.70, 0.27),
                (0.67, 0.37),
                (0.60, 0.43),
                (0.37, 0.47),
                (0.47, 0.47),
                (0.57, 0.47),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (19, 20),
                (20, 21),
                (18, 21),
                (9, 21),
                (9, 18),
            ],
        ),
        '4' => (
            vec![
                (0.57, 0.13),
                (0.50, 0.17),
                (0.43, 0.27),
                (0.37, 0.33),
                (0.33, 0.40),
                (0.27, 0.47),
                (0.23, 0.57),
                (0.20, 0.67),
                (0.27, 0.70),
                (0.37, 0.70),
                (0.47, 0.70),
                (0.57, 0.70),
                (0.60, 0.10),
                (0.60, 0.13),
                (0.60, 0.17),
                (0.60, 0.67),
                (0.60, 0.70),
                (0.60, 0.73),
                (0.63, 0.07),
                (0.60, 0.27),
                (0.60, 0.37),
                (0.60, 0.47),
                (0.60, 0.57),
                (0.63, 0.70),
                (0.60, 0.80),
                (0.60, 0.90),
                (0.77, 0.67),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (0, 12),
                (0, 13),
                (0, 14),
                (11, 15),
                (11, 16),
                (11, 17),
                (12, 13),
                (12, 18),
                (13, 14),
                (14, 19),
                (19, 20),
                (20, 21),
                (21, 22),
                (15, 22),
                (15, 16),
                (15, 23),
                (16, 17),
                (16, 23),
                (17, 24),
                (24, 25),
                (17, 23),
                (23, 26),
            ],
        ),
        '5' => (
            vec![
                (0.23, 0.87),
                (0.33, 0.90),
                (0.43, 0.90),
                (0.53, 0.90),
                (0.60, 0.87),
                (0.67, 0.83),
                (0.70, 0.73),
                (0.70, 0.63),
                (0.70, 0.53),
                (0.63, 0.43),
                (0.57, 0.40),
                (0.47, 0.40),
                (0.37, 0.40),
                (0.30, 0.33),
                (0.30, 0.27),
                (0.30, 0.17),
                (0.33, 0.07),
                (0.43, 0.07),
                (0.53, 0.07),
                (0.63, 0.07),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),
            ],
        ),
        '7' => (
            vec![
                (0.23, 0.10),
                (0.33, 0.07),
                (0.43, 0.07),
                (0.53, 0.07),
                (0.63, 0.07),
                (0.40, 0.90),
                (0.43, 0.80),
                (0.47, 0.73),
                (0.53, 0.63),
                (0.53, 0.57),
                (0.57, 0.47),
                (0.60, 0.37),
                (0.63, 0.30),
                (0.67, 0.20),
                (0.67, 0.10),
                (0.67, 0.07),
                (0.70, 0.07),
                (0.73, 0.07),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (4, 15),
                (4, 14),
                (14, 15),
                (15, 16),
                (14, 16),
                (16, 17),
            ],
        ),
        'Γ' => (
            vec![
                (0.23, 0.90),
                (0.23, 0.80),
                (0.23, 0.70),
                (0.23, 0.60),
                (0.23, 0.50),
                (0.23, 0.37),
                (0.23, 0.27),
                (0.23, 0.17),
                (0.27, 0.07),
                (0.37, 0.07),
                (0.47, 0.07),
                (0.57, 0.07),
                (0.67, 0.07),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
            ],
        ),
        'Δ' => (
            vec![
                (0.10, 0.90),
                (0.13, 0.90),
                (0.17, 0.87),
                (0.17, 0.90),
                (0.20, 0.80),
                (0.23, 0.70),
                (0.27, 0.60),
                (0.30, 0.53),
                (0.30, 0.47),
                (0.33, 0.37),
                (0.37, 0.27),
                (0.40, 0.20),
                (0.47, 0.13),
                (0.20, 0.90),
                (0.30, 0.90),
                (0.40, 0.90),
                (0.50, 0.90),
                (0.60, 0.90),
                (0.70, 0.90),
                (0.77, 0.87),
                (0.47, 0.07),
                (0.47, 0.10),
                (0.50, 0.13),
                (0.57, 0.20),
                (0.60, 0.30),
                (0.63, 0.40),
                (0.67, 0.50),
                (0.70, 0.57),
                (0.73, 0.67),
                (0.77, 0.77),
                (0.83, 0.90),
                (0.90, 0.93),
            ],
            vec![
                (0, 1),
                (1, 2),
                (1, 3),
                (2, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (2, 3),
                (2, 13),
                (3, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),
                (20, 21),
                (12, 21),
                (21, 22),
                (12, 22),
                (22, 23),
                (23, 24),
                (24, 25),
                (25, 26),
                (26, 27),
                (27, 28),
                (28, 29),
                (19, 29),
                (19, 30),
                (30, 31),
            ],
        ),
        'Λ' => (
            vec![
                (0.10, 0.90),
                (0.17, 0.83),
                (0.20, 0.77),
                (0.23, 0.67),
                (0.27, 0.57),
                (0.30, 0.47),
                (0.33, 0.40),
                (0.37, 0.30),
                (0.40, 0.20),
                (0.47, 0.13),
                (0.47, 0.07),
                (0.47, 0.10),
                (0.50, 0.13),
                (0.57, 0.20),
                (0.60, 0.30),
                (0.63, 0.37),
                (0.67, 0.47),
                (0.70, 0.57),
                (0.73, 0.63),
                (0.77, 0.73),
                (0.77, 0.80),
                (0.83, 0.90),
                (0.90, 0.93),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (10, 11),
                (9, 11),
                (11, 12),
                (9, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),
                (19, 20),
                (20, 21),
                (21, 22),
            ],
        ),
        'Π' => (
            vec![
                (0.17, 0.90),
                (0.17, 0.80),
                (0.17, 0.70),
                (0.17, 0.60),
                (0.17, 0.50),
                (0.17, 0.40),
                (0.17, 0.30),
                (0.17, 0.20),
                (0.17, 0.10),
                (0.27, 0.07),
                (0.37, 0.07),
                (0.43, 0.07),
                (0.53, 0.07),
                (0.63, 0.07),
                (0.73, 0.07),
                (0.77, 0.17),
                (0.77, 0.27),
                (0.77, 0.37),
                (0.77, 0.47),
                (0.77, 0.57),
                (0.77, 0.67),
                (0.77, 0.77),
                (0.77, 0.87),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),
                (19, 20),
                (20, 21),
                (21, 22),
            ],
        ),
        'Σ' => (
            vec![
                (0.20, 0.90),
                (0.30, 0.87),
                (0.27, 0.07),
                (0.30, 0.10),
                (0.33, 0.17),
                (0.37, 0.27),
                (0.43, 0.33),
                (0.50, 0.40),
                (0.53, 0.50),
                (0.50, 0.57),
                (0.43, 0.63),
                (0.37, 0.70),
                (0.33, 0.80),
                (0.40, 0.07),
                (0.50, 0.07),
                (0.60, 0.07),
                (0.70, 0.07),
                (0.37, 0.90),
                (0.47, 0.90),
                (0.57, 0.90),
                (0.67, 0.90),
                (0.73, 0.87),
            ],
            vec![
                (0, 1),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (1, 12),
                (3, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (1, 17),
                (17, 18),
                (18, 19),
                (19, 20),
                (20, 21),
            ],
        ),
        'Φ' => (
            vec![
                (0.43, 0.20),
                (0.33, 0.20),
                (0.23, 0.27),
                (0.17, 0.33),
                (0.13, 0.40),
                (0.10, 0.50),
                (0.13, 0.60),
                (0.17, 0.67),
                (0.23, 0.73),
                (0.33, 0.80),
                (0.43, 0.80),
                (0.47, 0.17),
                (0.47, 0.20),
                (0.47, 0.23),
                (0.47, 0.77),
                (0.47, 0.80),
                (0.47, 0.83),
                (0.50, 0.03),
                (0.50, 0.20),
                (0.47, 0.33),
                (0.47, 0.43),
                (0.47, 0.57),
                (0.47, 0.67),
                (0.50, 0.80),
                (0.47, 0.90),
                (0.60, 0.20),
                (0.70, 0.23),
                (0.77, 0.27),
                (0.83, 0.37),
                (0.83, 0.43),
                (0.83, 0.57),
                (0.83, 0.63),
                (0.77, 0.73),
                (0.70, 0.77),
                (0.60, 0.80),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (0, 11),
                (0, 12),
                (0, 13),
                (10, 14),
                (10, 15),
                (10, 16),
                (11, 17),
                (11, 12),
                (11, 18),
                (12, 13),
                (12, 18),
                (13, 19),
                (19, 20),
                (20, 21),
                (21, 22),
                (14, 22),
                (13, 18),
                (14, 15),
                (14, 23),
                (15, 16),
                (15, 23),
                (16, 24),
                (16, 23),
                (18, 25),
                (25, 26),
                (26, 27),
                (27, 28),
                (28, 29),
                (29, 30),
                (30, 31),
                (31, 32),
                (32, 33),
                (33, 34),
                (23, 34),
            ],
        ),
        'Ψ' => (
            vec![
                (0.13, 0.03),
                (0.10, 0.13),
                (0.10, 0.20),
                (0.10, 0.30),
                (0.13, 0.40),
                (0.17, 0.47),
                (0.20, 0.57),
                (0.27, 0.63),
                (0.33, 0.70),
                (0.43, 0.70),
                (0.47, 0.67),
                (0.47, 0.70),
                (0.47, 0.73),
                (0.47, 0.57),
                (0.47, 0.47),
                (0.47, 0.33),
                (0.47, 0.23),
                (0.47, 0.13),
                (0.50, 0.03),
                (0.50, 0.70),
                (0.47, 0.80),
                (0.47, 0.90),
                (0.60, 0.70),
                (0.70, 0.67),
                (0.77, 0.60),
                (0.80, 0.50),
                (0.83, 0.43),
                (0.83, 0.33),
                (0.83, 0.23),
                (0.83, 0.13),
                (0.87, 0.03),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (9, 11),
                (9, 12),
                (10, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (10, 11),
                (10, 19),
                (11, 12),
                (11, 19),
                (12, 20),
                (20, 21),
                (12, 19),
                (19, 22),
                (22, 23),
                (23, 24),
                (24, 25),
                (25, 26),
                (26, 27),
                (27, 28),
                (28, 29),
                (29, 30),
            ],
        ),
        'Ω' => (
            vec![
                (0.07, 0.90),
                (0.20, 0.90),
                (0.30, 0.87),
                (0.30, 0.83),
                (0.27, 0.77),
                (0.17, 0.67),
                (0.17, 0.60),
                (0.13, 0.50),
                (0.13, 0.40),
                (0.17, 0.30),
                (0.20, 0.23),
                (0.27, 0.17),
                (0.37, 0.10),
                (0.47, 0.10),
                (0.57, 0.10),
                (0.67, 0.10),
                (0.73, 0.17),
                (0.80, 0.23),
                (0.83, 0.33),
                (0.83, 0.43),
                (0.83, 0.53),
                (0.83, 0.63),
                (0.80, 0.70),
                (0.73, 0.77),
                (0.70, 0.87),
                (0.33, 0.87),
                (0.40, 0.90),
                (0.63, 0.90),
                (0.77, 0.90),
                (0.87, 0.87),
            ],
            vec![
                (0, 1),
                (1, 2),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),
                (19, 20),
                (20, 21),
                (21, 22),
                (22, 23),
                (23, 24),
                (2, 3),
                (3, 25),
                (2, 25),
                (25, 26),
                (24, 27),
                (24, 28),
                (28, 29),
            ],
        ),
        'α' => (
            vec![
                (0.63, 0.73),
                (0.57, 0.80),
                (0.50, 0.87),
                (0.40, 0.90),
                (0.30, 0.87),
                (0.23, 0.83),
                (0.17, 0.77),
                (0.13, 0.67),
                (0.10, 0.57),
                (0.10, 0.47),
                (0.10, 0.37),
                (0.13, 0.30),
                (0.20, 0.20),
                (0.23, 0.13),
                (0.33, 0.10),
                (0.43, 0.10),
                (0.53, 0.13),
                (0.60, 0.20),
                (0.63, 0.27),
                (0.67, 0.37),
                (0.67, 0.70),
                (0.67, 0.73),
                (0.70, 0.33),
                (0.70, 0.37),
                (0.70, 0.40),
                (0.67, 0.60),
                (0.67, 0.50),
                (0.70, 0.73),
                (0.77, 0.27),
                (0.80, 0.20),
                (0.83, 0.10),
                (0.90, 0.07),
                (0.77, 0.80),
                (0.83, 0.87),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),
                (0, 20),
                (0, 21),
                (19, 22),
                (19, 23),
                (19, 24),
                (20, 25),
                (25, 26),
                (24, 26),
                (20, 21),
                (20, 27),
                (21, 27),
                (22, 23),
                (22, 28),
                (28, 29),
                (29, 30),
                (30, 31),
                (23, 24),
                (27, 32),
                (32, 33),
            ],
        ),
        'β' => (
            vec![
                (0.30, 0.70),
                (0.30, 0.60),
                (0.30, 0.50),
                (0.30, 0.40),
                (0.30, 0.27),
                (0.30, 0.17),
                (0.37, 0.10),
                (0.47, 0.07),
                (0.57, 0.10),
                (0.60, 0.17),
                (0.60, 0.27),
                (0.57, 0.37),
                (0.30, 0.73),
                (0.33, 0.73),
                (0.30, 0.77),
                (0.30, 0.87),
                (0.43, 0.73),
                (0.53, 0.73),
                (0.63, 0.67),
                (0.63, 0.57),
                (0.63, 0.47),
                (0.57, 0.40),
                (0.43, 0.37),
                (0.53, 0.37),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (0, 12),
                (0, 13),
                (12, 14),
                (12, 13),
                (14, 15),
                (13, 14),
                (13, 16),
                (16, 17),
                (17, 18),
                (18, 19),
                (19, 20),
                (20, 21),
                (22, 23),
                (11, 23),
                (21, 23),
                (11, 21),
            ],
        ),
        'γ' => (
            vec![
                (0.17, 0.07),
                (0.27, 0.10),
                (0.30, 0.17),
                (0.33, 0.27),
                (0.37, 0.33),
                (0.40, 0.43),
                (0.43, 0.50),
                (0.47, 0.60),
                (0.50, 0.60),
                (0.50, 0.63),
                (0.57, 0.53),
                (0.60, 0.43),
                (0.63, 0.37),
                (0.67, 0.27),
                (0.70, 0.20),
                (0.73, 0.10),
                (0.77, 0.03),
                (0.50, 0.73),
                (0.50, 0.87),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (7, 9),
                (8, 9),
                (8, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (9, 17),
                (17, 18),
            ],
        ),
        'δ' => (
            vec![
                (0.37, 0.30),
                (0.30, 0.23),
                (0.30, 0.13),
                (0.37, 0.10),
                (0.43, 0.07),
                (0.53, 0.07),
                (0.63, 0.07),
                (0.33, 0.37),
                (0.27, 0.47),
                (0.23, 0.53),
                (0.23, 0.63),
                (0.23, 0.73),
                (0.30, 0.83),
                (0.37, 0.90),
                (0.47, 0.90),
                (0.57, 0.90),
                (0.63, 0.87),
                (0.70, 0.77),
                (0.70, 0.67),
                (0.70, 0.57),
                (0.70, 0.47),
                (0.63, 0.40),
                (0.57, 0.33),
                (0.47, 0.30),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (0, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),
                (19, 20),
                (20, 21),
                (21, 22),
                (22, 23),
                (0, 23),
            ],
        ),
        'ε' => (
            vec![
                (0.40, 0.43),
                (0.33, 0.37),
                (0.27, 0.30),
                (0.27, 0.20),
                (0.33, 0.13),
                (0.43, 0.10),
                (0.53, 0.07),
                (0.60, 0.10),
                (0.70, 0.10),
                (0.40, 0.47),
                (0.43, 0.47),
                (0.33, 0.53),
                (0.27, 0.60),
                (0.23, 0.70),
                (0.27, 0.77),
                (0.33, 0.83),
                (0.43, 0.87),
                (0.53, 0.90),
                (0.63, 0.90),
                (0.70, 0.87),
                (0.80, 0.83),
                (0.53, 0.47),
                (0.60, 0.43),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (0, 9),
                (0, 10),
                (9, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),
                (19, 20),
                (9, 10),
                (10, 21),
                (21, 22),
            ],
        ),
        'ζ' => (
            vec![
                (0.30, 0.07),
                (0.40, 0.07),
                (0.53, 0.07),
                (0.57, 0.07),
                (0.57, 0.10),
                (0.60, 0.07),
                (0.53, 0.17),
                (0.43, 0.27),
                (0.37, 0.33),
                (0.30, 0.40),
                (0.30, 0.50),
                (0.30, 0.60),
                (0.37, 0.67),
                (0.47, 0.73),
                (0.57, 0.73),
                (0.63, 0.80),
                (0.60, 0.90),
                (0.63, 0.03),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (2, 4),
                (3, 4),
                (3, 5),
                (4, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (4, 5),
                (5, 17),
            ],
        ),
        'η' => (
            vec![
                (0.23, 0.13),
                (0.23, 0.17),
                (0.27, 0.07),
                (0.27, 0.17),
                (0.23, 0.20),
                (0.23, 0.30),
                (0.23, 0.40),
                (0.23, 0.47),
                (0.23, 0.57),
                (0.23, 0.67),
                (0.37, 0.13),
                (0.43, 0.10),
                (0.53, 0.07),
                (0.60, 0.10),
                (0.67, 0.17),
                (0.70, 0.27),
                (0.70, 0.37),
                (0.70, 0.47),
                (0.70, 0.57),
                (0.70, 0.67),
                (0.70, 0.77),
                (0.70, 0.87),
            ],
            vec![
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 4),
                (1, 3),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (3, 4),
                (3, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),
                (19, 20),
                (20, 21),
            ],
        ),
        'λ' => (
            vec![
                (0.27, 0.07),
                (0.37, 0.07),
                (0.43, 0.13),
                (0.47, 0.23),
                (0.50, 0.30),
                (0.50, 0.40),
                (0.27, 0.87),
                (0.30, 0.77),
                (0.33, 0.70),
                (0.37, 0.60),
                (0.43, 0.50),
                (0.50, 0.43),
                (0.53, 0.43),
                (0.60, 0.50),
                (0.63, 0.60),
                (0.67, 0.70),
                (0.70, 0.80),
                (0.73, 0.87),
                (0.80, 0.93),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (5, 11),
                (5, 12),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
            ],
        ),
        'μ' => (
            vec![
                (0.20, 0.60),
                (0.20, 0.50),
                (0.20, 0.40),
                (0.20, 0.30),
                (0.20, 0.20),
                (0.20, 0.10),
                (0.23, 0.03),
                (0.20, 0.63),
                (0.23, 0.63),
                (0.20, 0.67),
                (0.20, 0.77),
                (0.20, 0.90),
                (0.33, 0.63),
                (0.43, 0.67),
                (0.53, 0.63),
                (0.63, 0.60),
                (0.67, 0.57),
                (0.67, 0.60),
                (0.67, 0.03),
                (0.67, 0.13),
                (0.67, 0.23),
                (0.67, 0.37),
                (0.67, 0.47),
                (0.77, 0.63),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (0, 7),
                (0, 8),
                (7, 9),
                (7, 8),
                (9, 10),
                (10, 11),
                (8, 9),
                (8, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (15, 17),
                (18, 19),
                (19, 20),
                (20, 21),
                (21, 22),
                (16, 22),
                (16, 17),
                (17, 23),
            ],
        ),
        'ξ' => (
            vec![
                (0.30, 0.07),
                (0.37, 0.07),
                (0.40, 0.07),
                (0.40, 0.10),
                (0.43, 0.07),
                (0.37, 0.17),
                (0.33, 0.27),
                (0.37, 0.33),
                (0.43, 0.37),
                (0.40, 0.40),
                (0.40, 0.43),
                (0.43, 0.40),
                (0.33, 0.50),
                (0.30, 0.60),
                (0.33, 0.67),
                (0.43, 0.70),
                (0.53, 0.73),
                (0.63, 0.77),
                (0.63, 0.83),
                (0.57, 0.90),
                (0.53, 0.07),
                (0.63, 0.03),
                (0.47, 0.40),
                (0.60, 0.37),
            ],
            vec![
                (0, 1),
                (1, 2),
                (1, 3),
                (2, 3),
                (2, 4),
                (3, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (3, 4),
                (9, 10),
                (8, 9),
                (9, 11),
                (10, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),
                (10, 11),
                (4, 20),
                (20, 21),
                (8, 11),
                (8, 22),
                (11, 22),
                (22, 23),
            ],
        ),
        'π' => (
            vec![
                (0.10, 0.10),
                (0.20, 0.10),
                (0.23, 0.10),
                (0.23, 0.13),
                (0.20, 0.90),
                (0.23, 0.80),
                (0.23, 0.73),
                (0.23, 0.63),
                (0.23, 0.53),
                (0.23, 0.43),
                (0.23, 0.33),
                (0.23, 0.23),
                (0.27, 0.10),
                (0.37, 0.10),
                (0.47, 0.10),
                (0.57, 0.10),
                (0.67, 0.10),
                (0.70, 0.10),
                (0.70, 0.13),
                (0.73, 0.10),
                (0.70, 0.23),
                (0.70, 0.33),
                (0.70, 0.43),
                (0.70, 0.53),
                (0.70, 0.63),
                (0.70, 0.73),
                (0.73, 0.83),
                (0.80, 0.90),
                (0.87, 0.87),
                (0.87, 0.07),
            ],
            vec![
                (0, 1),
                (1, 2),
                (1, 3),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (3, 11),
                (2, 3),
                (2, 12),
                (3, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (16, 18),
                (17, 18),
                (17, 19),
                (18, 20),
                (20, 21),
                (21, 22),
                (22, 23),
                (23, 24),
                (24, 25),
                (25, 26),
                (26, 27),
                (27, 28),
                (18, 19),
                (19, 29),
            ],
        ),
        'ρ' => (
            vec![
                (0.23, 0.63),
                (0.23, 0.67),
                (0.27, 0.60),
                (0.27, 0.63),
                (0.23, 0.77),
                (0.23, 0.90),
                (0.23, 0.50),
                (0.27, 0.43),
                (0.27, 0.33),
                (0.27, 0.23),
                (0.33, 0.17),
                (0.40, 0.10),
                (0.50, 0.07),
                (0.60, 0.10),
                (0.67, 0.17),
                (0.70, 0.23),
                (0.73, 0.33),
                (0.73, 0.43),
                (0.70, 0.53),
                (0.67, 0.60),
                (0.60, 0.67),
                (0.50, 0.67),
                (0.40, 0.63),
                (0.30, 0.63),
            ],
            vec![
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 4),
                (4, 5),
                (1, 3),
                (2, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),
                (19, 20),
                (20, 21),
                (21, 22),
                (22, 23),
                (2, 3),
                (2, 23),
                (3, 23),
            ],
        ),
        'σ' => (
            vec![
                (0.67, 0.10),
                (0.57, 0.07),
                (0.47, 0.07),
                (0.40, 0.10),
                (0.30, 0.10),
                (0.23, 0.17),
                (0.17, 0.23),
                (0.13, 0.33),
                (0.10, 0.43),
                (0.10, 0.53),
                (0.10, 0.60),
                (0.13, 0.70),
                (0.20, 0.77),
                (0.27, 0.83),
                (0.37, 0.90),
                (0.47, 0.90),
                (0.53, 0.90),
                (0.63, 0.83),
                (0.70, 0.77),
                (0.73, 0.70),
                (0.77, 0.60),
                (0.77, 0.50),
                (0.77, 0.40),
                (0.77, 0.30),
                (0.73, 0.23),
                (0.70, 0.13),
                (0.70, 0.10),
                (0.73, 0.10),
                (0.83, 0.10),
                (0.90, 0.07),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),
                (19, 20),
                (20, 21),
                (21, 22),
                (22, 23),
                (23, 24),
                (24, 25),
                (0, 26),
                (0, 25),
                (25, 26),
                (26, 27),
                (25, 27),
                (27, 28),
                (28, 29),
            ],
        ),
        'φ' => (
            vec![
                (0.40, 0.07),
                (0.30, 0.13),
                (0.27, 0.20),
                (0.23, 0.30),
                (0.23, 0.40),
                (0.23, 0.50),
                (0.30, 0.60),
                (0.37, 0.63),
                (0.47, 0.67),
                (0.50, 0.63),
                (0.50, 0.67),
                (0.50, 0.70),
                (0.50, 0.53),
                (0.50, 0.43),
                (0.50, 0.33),
                (0.50, 0.23),
                (0.50, 0.13),
                (0.57, 0.07),
                (0.67, 0.10),
                (0.73, 0.17),
                (0.77, 0.23),
                (0.77, 0.33),
                (0.77, 0.43),
                (0.73, 0.53),
                (0.70, 0.60),
                (0.63, 0.63),
                (0.53, 0.67),
                (0.50, 0.77),
                (0.50, 0.87),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (8, 10),
                (8, 11),
                (9, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),
                (19, 20),
                (20, 21),
                (21, 22),
                (22, 23),
                (23, 24),
                (24, 25),
                (25, 26),
                (9, 10),
                (9, 26),
                (10, 11),
                (10, 26),
                (11, 27),
                (27, 28),
                (11, 26),
            ],
        ),
        'ψ' => (
            vec![
                (0.23, 0.03),
                (0.20, 0.13),
                (0.20, 0.23),
                (0.20, 0.33),
                (0.23, 0.40),
                (0.23, 0.50),
                (0.30, 0.60),
                (0.37, 0.63),
                (0.47, 0.67),
                (0.50, 0.63),
                (0.50, 0.67),
                (0.50, 0.70),
                (0.50, 0.03),
                (0.50, 0.13),
                (0.50, 0.23),
                (0.50, 0.33),
                (0.50, 0.43),
                (0.50, 0.53),
                (0.53, 0.67),
                (0.50, 0.77),
                (0.50, 0.87),
                (0.63, 0.63),
                (0.70, 0.60),
                (0.77, 0.53),
                (0.77, 0.43),
                (0.77, 0.33),
                (0.77, 0.23),
                (0.77, 0.13),
                (0.77, 0.03),
            ],
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (8, 10),
                (8, 11),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (9, 17),
                (9, 10),
                (9, 18),
                (10, 11),
                (10, 18),
                (11, 19),
                (19, 20),
                (11, 18),
                (18, 21),
                (21, 22),
                (22, 23),
                (23, 24),
                (24, 25),
                (25, 26),
                (26, 27),
                (27, 28),
            ],
        ),
        _ => (
            vec![(0.2, 0.0), (0.2, 0.5), (0.2, 1.0), (0.8, 0.0), (0.8, 1.0)],
            vec![(0, 1), (1, 2), (1, 3), (1, 4)],
        ),
    };
    CharGraph { points, edges }
}

// ---------------------------------------------------------------------------
// Graph decomposition
// ---------------------------------------------------------------------------

const CORNER_WEIGHT: f64 = 5.0;
const CORNER_ANGLE_THRESHOLD: f64 = 2.094; // 120 degrees (2π/3 radians)

fn identify_corners(graph: &CharGraph) -> Vec<bool> {
    let n = graph.points.len();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(u, v) in &graph.edges {
        adj[u].push(v);
        adj[v].push(u);
    }
    let mut is_corner = vec![false; n];
    for i in 0..n {
        let deg = adj[i].len();
        if deg >= 3 {
            is_corner[i] = true;
        } else if deg == 2 {
            let (vx, vy) = graph.points[i];
            let (ax, ay) = graph.points[adj[i][0]];
            let (bx, by) = graph.points[adj[i][1]];
            let v1x = ax - vx;
            let v1y = ay - vy;
            let v2x = bx - vx;
            let v2y = by - vy;
            let dot = v1x * v2x + v1y * v2y;
            let len1 = (v1x * v1x + v1y * v1y).sqrt();
            let len2 = (v2x * v2x + v2y * v2y).sqrt();
            if len1 > 1e-9 && len2 > 1e-9 {
                let cos_angle = (dot / (len1 * len2)).clamp(-1.0, 1.0);
                let angle = cos_angle.acos();
                if angle < CORNER_ANGLE_THRESHOLD {
                    is_corner[i] = true;
                }
            }
        }
    }
    is_corner
}

fn decompose_graph(graph: &CharGraph, num_pieces: usize, rng: &mut ChaChaRng) -> Vec<usize> {
    let n = graph.points.len();
    let mut vertex_piece: Vec<usize> = vec![0; n];

    if num_pieces <= 1 {
        return vertex_piece;
    }

    // Build adjacency from edges.
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(u, v) in &graph.edges {
        adj[u].push(v);
        adj[v].push(u);
    }

    // BFS spanning tree from random root.
    let root = rng.gen_range(0..n);
    let mut parent: Vec<Option<usize>> = vec![None; n];
    let mut bfs_order: Vec<usize> = Vec::with_capacity(n);
    let mut visited = vec![false; n];
    let mut queue = VecDeque::new();

    visited[root] = true;
    queue.push_back(root);
    while let Some(node) = queue.pop_front() {
        bfs_order.push(node);
        let mut neighbors: Vec<usize> = adj[node]
            .iter()
            .copied()
            .filter(|&nb| !visited[nb])
            .collect();
        for i in (1..neighbors.len()).rev() {
            let j = rng.gen_range(0..=i);
            neighbors.swap(i, j);
        }
        for nb in neighbors {
            visited[nb] = true;
            parent[nb] = Some(node);
            queue.push_back(nb);
        }
    }

    // Compute subtree sizes.
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &node in &bfs_order {
        if let Some(p) = parent[node] {
            children[p].push(node);
        }
    }
    let mut subtree_size: Vec<usize> = vec![1; n];
    for &node in bfs_order.iter().rev() {
        for &ch in &children[node] {
            subtree_size[node] += subtree_size[ch];
        }
    }

    let is_corner = identify_corners(graph);
    let mut subtree_corners: Vec<usize> =
        is_corner.iter().map(|&c| if c { 1 } else { 0 }).collect();
    for &node in bfs_order.iter().rev() {
        for &ch in &children[node] {
            subtree_corners[node] += subtree_corners[ch];
        }
    }

    // Greedily cut N-1 edges: always cut the largest subtree.
    let mut cuts_remaining = num_pieces - 1;
    let mut piece_id = 0;

    // Track which subtree roots define pieces.
    let mut subtree_roots: Vec<usize> = vec![root];

    while cuts_remaining > 0 {
        // Among current subtree roots, find the one containing the best cut edge.
        // Score combines balance with corner distribution to avoid revealing pieces.
        let mut best_node: Option<usize> = None;
        let mut best_score = 0.0f64;
        for &sr in &subtree_roots {
            fn find_best_cut(
                node: usize,
                children: &[Vec<usize>],
                subtree_size: &[usize],
                subtree_corners: &[usize],
                total: usize,
                total_corners: usize,
            ) -> Option<(usize, f64)> {
                let mut best: Option<(usize, f64)> = None;
                for &ch in &children[node] {
                    let sz = subtree_size[ch];
                    let balance = sz.min(total - sz) as f64;
                    let c_sub = subtree_corners[ch];
                    let c_rest = total_corners.saturating_sub(c_sub);
                    let corner_ratio = if c_sub + c_rest == 0 {
                        1.0
                    } else {
                        c_sub.min(c_rest) as f64 / c_sub.max(c_rest) as f64
                    };
                    let score = balance * (1.0 + CORNER_WEIGHT * corner_ratio);
                    if best.map_or(true, |(_, bs)| score > bs) {
                        best = Some((ch, score));
                    }
                    if let Some(deeper) = find_best_cut(
                        ch,
                        children,
                        subtree_size,
                        subtree_corners,
                        total,
                        total_corners,
                    ) {
                        if best.map_or(true, |(_, bs)| deeper.1 > bs) {
                            best = Some(deeper);
                        }
                    }
                }
                best
            }

            let total = subtree_size[sr];
            let total_corners = subtree_corners[sr];
            if let Some((node, score)) = find_best_cut(
                sr,
                &children,
                &subtree_size,
                &subtree_corners,
                total,
                total_corners,
            ) {
                if score > best_score {
                    best_score = score;
                    best_node = Some(node);
                }
            }
        }

        if let Some(cut_node) = best_node {
            piece_id += 1;
            // Assign all nodes in cut_node's subtree to the new piece.
            fn assign_piece(node: usize, children: &[Vec<usize>], piece: &mut [usize], pid: usize) {
                piece[node] = pid;
                for &ch in &children[node] {
                    assign_piece(ch, children, piece, pid);
                }
            }
            assign_piece(cut_node, &children, &mut vertex_piece, piece_id);

            // Update: the cut_node's parent's subtree_size and corner count decrease.
            let cut_sz = subtree_size[cut_node];
            let cut_corners = subtree_corners[cut_node];
            let mut cur = parent[cut_node];
            while let Some(p) = cur {
                subtree_size[p] = subtree_size[p].saturating_sub(cut_sz);
                subtree_corners[p] = subtree_corners[p].saturating_sub(cut_corners);
                if subtree_roots.contains(&p) {
                    break;
                }
                cur = parent[p];
            }
            // Remove cut_node from parent's children.
            if let Some(p) = parent[cut_node] {
                children[p].retain(|&c| c != cut_node);
            }
            subtree_roots.push(cut_node);
            cuts_remaining -= 1;
        } else {
            break;
        }
    }

    vertex_piece
}

// ---------------------------------------------------------------------------
// Pixel-to-edge assignment
// ---------------------------------------------------------------------------

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

fn assign_pixels_to_pieces(
    cells: &BTreeSet<(i32, i32)>,
    scaled_points: &[(f64, f64)],
    edges: &[(usize, usize)],
    vertex_piece: &[usize],
    num_pieces: usize,
) -> Vec<Vec<(i32, i32)>> {
    let mut pieces: Vec<Vec<(i32, i32)>> = vec![Vec::new(); num_pieces];

    for &(cx, cy) in cells {
        let px = cx as f64 + 0.5;
        let py = cy as f64 + 0.5;

        let mut best_dist = f64::MAX;
        let mut best_piece = 0;

        for &(u, v) in edges {
            let (x1, y1) = scaled_points[u];
            let (x2, y2) = scaled_points[v];
            let d = point_to_segment_dist_sq(px, py, x1, y1, x2, y2);

            if d < best_dist {
                best_dist = d;
                let pu = vertex_piece[u];
                let pv = vertex_piece[v];
                if pu == pv {
                    best_piece = pu;
                } else {
                    let du = (px - x1).powi(2) + (py - y1).powi(2);
                    let dv = (px - x2).powi(2) + (py - y2).powi(2);
                    best_piece = if du <= dv { pu } else { pv };
                }
            }
        }

        pieces[best_piece].push((cx, cy));
    }

    pieces
}

fn snap_to_nearest_cell(x: f64, y: f64, cells: &BTreeSet<(i32, i32)>) -> (f64, f64) {
    let mut best_dist = f64::MAX;
    let mut best = (x, y);
    for &(cx, cy) in cells {
        let fx = cx as f64 + 0.5;
        let fy = cy as f64 + 0.5;
        let d = (x - fx).powi(2) + (y - fy).powi(2);
        if d < best_dist {
            best_dist = d;
            best = (fx, fy);
        }
    }
    best
}

fn decompose_via_graph(
    ch: char,
    cells: &BTreeSet<(i32, i32)>,
    num_pieces: usize,
    rng: &mut ChaChaRng,
) -> (Vec<Vec<(i32, i32)>>, PieceLookup) {
    let graph = char_graph(ch);

    let vertex_piece = decompose_graph(&graph, num_pieces, rng);

    let min_x = cells.iter().map(|&(x, _)| x).min().unwrap() as f64;
    let max_x = cells.iter().map(|&(x, _)| x).max().unwrap() as f64 + 1.0;
    let min_y = cells.iter().map(|&(_, y)| y).min().unwrap() as f64;
    let max_y = cells.iter().map(|&(_, y)| y).max().unwrap() as f64 + 1.0;
    let scaled: Vec<(f64, f64)> = graph
        .points
        .iter()
        .map(|&(x, y)| {
            let sx = min_x + x * (max_x - min_x);
            let sy = min_y + y * (max_y - min_y);
            snap_to_nearest_cell(sx, sy, cells)
        })
        .collect();

    let lookup = PieceLookup {
        points: scaled.clone(),
        edges: graph.edges.clone(),
        vertex_piece: vertex_piece.clone(),
    };

    let pieces = assign_pixels_to_pieces(cells, &scaled, &graph.edges, &vertex_piece, num_pieces);
    (pieces, lookup)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn font_name_table_matches_loaded_fonts() {
        // font_indices_for_char indexes fonts() through its name table; the two
        // lists must stay aligned when fonts are added or removed.
        assert_eq!(font_indices_for_char('A').len(), fonts().len());
    }

    #[test]
    fn rasterize_produces_cells() {
        let font = &fonts()[0];
        for &ch in &['A', 'L', 'T', '7'] {
            let cells = rasterize_character(font, ch, 20);
            assert!(
                cells.len() > 5,
                "character '{ch}' should produce pixels, got {}",
                cells.len()
            );
        }
    }

    #[test]
    fn rasterize_both_fonts() {
        for (fi, font) in fonts().iter().enumerate() {
            let cells = rasterize_character(font, 'R', 20);
            assert!(cells.len() > 5, "font {fi}: 'R' should produce pixels");
        }
    }

    #[test]
    fn char_graph_defined_for_all() {
        for &ch in &all_allowed_chars() {
            let g = char_graph(ch);
            assert!(!g.points.is_empty(), "char '{ch}' should have points");
            assert!(!g.edges.is_empty(), "char '{ch}' should have edges");
            for &(u, v) in &g.edges {
                assert!(
                    u < g.points.len(),
                    "char '{ch}' edge ({u},{v}) out of range"
                );
                assert!(
                    v < g.points.len(),
                    "char '{ch}' edge ({u},{v}) out of range"
                );
            }
        }
    }

    #[test]
    fn decompose_via_graph_all_chars() {
        let font = &fonts()[0];
        for &ch in &all_allowed_chars() {
            let cells = rasterize_character(font, ch, 24);
            if cells.len() < 10 {
                continue;
            }
            let mut rng = ChaChaRng::seed_from_u64(42);
            let (pieces, _lookup) = decompose_via_graph(ch, &cells, 3, &mut rng);
            assert_eq!(pieces.len(), 3, "char '{ch}' should have 3 pieces");
            let total: usize = pieces.iter().map(|p| p.len()).sum();
            assert_eq!(
                total,
                cells.len(),
                "char '{ch}' pieces should cover all cells"
            );
        }
    }

    #[test]
    fn generate_succeeds() {
        for seed in 0..30u64 {
            let mut rng = ChaChaRng::seed_from_u64(seed);
            if generate(&mut rng, (3, 3), (24, 28), 4).is_some() {
                return;
            }
        }
        panic!("generation should succeed within 30 seeds");
    }

    #[test]
    fn trivial_puzzle_valid() {
        let puzzle = trivial(3, 4);
        assert_eq!(puzzle.character, "L");
        assert_eq!(puzzle.character_char, 'L');
        assert_eq!(puzzle.num_pieces, 3);
        assert_eq!(puzzle.pieces.len(), 3);
        assert_eq!(puzzle.slots.len(), 16); // 4x4 grid
        let assigned: Vec<_> = puzzle
            .slots
            .iter()
            .filter(|s| s.matching_piece_idx.is_some())
            .collect();
        assert_eq!(assigned.len(), 3);
        for piece in &puzzle.pieces {
            assert!(!piece.cells.is_empty());
        }
    }
}
