//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
use image::{Rgb, RgbImage};
use imageproc::drawing::draw_filled_rect_mut;

use crate::base::game::RgbBuffer;
use crate::base::render::{draw_text_centered, rect};

use super::state::AnagramState;

const BG_COLOR: Rgb<u8> = Rgb([0xE8, 0xE8, 0xE8]);
const TEXT_COLOR: Rgb<u8> = Rgb([0xFF, 0xFF, 0xFF]);
const INDEX_COLOR: Rgb<u8> = Rgb([0x60, 0x60, 0x60]);

const TILE_COLORS: [Rgb<u8>; 8] = [
    Rgb([0x4A, 0x90, 0xD9]), // Blue
    Rgb([0xE8, 0x6B, 0x6B]), // Red
    Rgb([0x6B, 0xB8, 0x6B]), // Green
    Rgb([0xE8, 0xA8, 0x4A]), // Orange
    Rgb([0x9B, 0x6B, 0xD9]), // Purple
    Rgb([0xD9, 0x6B, 0xA8]), // Pink
    Rgb([0x4A, 0xB8, 0xB8]), // Teal
    Rgb([0xB8, 0x8B, 0x4A]), // Brown
];

fn tile_color(target: &[char]) -> Rgb<u8> {
    let hash: u32 = target
        .iter()
        .fold(0u32, |acc, &c| acc.wrapping_mul(31).wrapping_add(c as u32));
    TILE_COLORS[(hash as usize) % TILE_COLORS.len()]
}

pub fn render(state: &AnagramState, resolution: u32) -> RgbBuffer {
    let mut img = RgbImage::from_pixel(resolution, resolution, BG_COLOR);
    let n = state.word_len() as u32;
    if n == 0 {
        return RgbBuffer::from_image(img);
    }

    let mut sorted_letters = state.letters.clone();
    sorted_letters.sort();
    let color = tile_color(&sorted_letters);

    let (row1_count, row2_count) = if n <= 5 {
        (n, 0u32)
    } else {
        let top = (n + 1) / 2;
        (top, n - top)
    };
    let num_rows = if row2_count > 0 { 2u32 } else { 1 };

    let margin = resolution / 10;
    let gap = (resolution / 40).max(2);
    let usable_width = resolution.saturating_sub(2 * margin);
    let ref_per_row = 5u32;
    let tile_size = ((usable_width.saturating_sub((ref_per_row - 1) * gap)) / ref_per_row).max(4);
    let letter_font_h = (tile_size as f32 * 0.6).max(8.0);
    let index_font_h = (tile_size as f32 * 0.3).max(6.0);
    let row_height = tile_size + gap * 2 + index_font_h as u32;
    let row_gap = gap * 2;
    let total_height = num_rows * row_height + num_rows.saturating_sub(1) * row_gap;
    let start_y = (resolution.saturating_sub(total_height)) / 2;

    let rows: Vec<(u32, u32)> = if row2_count > 0 {
        vec![(0, row1_count), (row1_count, row1_count + row2_count)]
    } else {
        vec![(0, n)]
    };

    for (row_idx, &(from, to)) in rows.iter().enumerate() {
        let count = to - from;
        let total_width = count * tile_size + count.saturating_sub(1) * gap;
        let start_x = (resolution - total_width) / 2;
        let tile_y = start_y + row_idx as u32 * (row_height + row_gap);

        for i in 0..count {
            let global_idx = from + i;
            let x = start_x + i * (tile_size + gap);

            draw_filled_rect_mut(&mut img, rect(x, tile_y, tile_size, tile_size), color);

            let letter = state.letters[global_idx as usize].to_ascii_uppercase();
            let cx = (x + tile_size / 2) as i32;
            let cy = (tile_y + tile_size / 2) as i32;
            draw_text_centered(&mut img, cx, cy, letter, letter_font_h, TEXT_COLOR);

            let idx_y = (tile_y + tile_size + gap * 2) as i32 + (index_font_h / 2.0) as i32;
            draw_index(&mut img, cx, idx_y, global_idx, index_font_h, INDEX_COLOR);
        }
    }

    RgbBuffer::from_image(img)
}

fn draw_index(img: &mut RgbImage, cx: i32, cy: i32, index: u32, font_height: f32, color: Rgb<u8>) {
    if index < 10 {
        let ch = char::from_digit(index, 10).unwrap_or('?');
        draw_text_centered(img, cx, cy, ch, font_height, color);
    } else {
        use crate::base::render::draw_str_centered;
        let label = index.to_string();
        draw_str_centered(img, cx, cy, &label, font_height, color);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_state() -> AnagramState {
        AnagramState {
            letters: vec!['h', 'c', 'u', 'r', 'c', 'h'],
            valid_words: vec!["church".to_string()],
            identified: false,
        }
    }

    #[test]
    fn render_dimensions() {
        let buf = render(&test_state(), 64);
        assert_eq!(buf.width, 64);
        assert_eq!(buf.height, 64);
        assert_eq!(buf.data.len(), 64 * 64 * 3);
    }

    #[test]
    fn render_nonzero() {
        let buf = render(&test_state(), 64);
        assert!(buf.data.iter().any(|&b| b != 0));
    }

    #[test]
    fn render_multiple_resolutions() {
        let state = test_state();
        for res in [64, 128, 256] {
            let buf = render(&state, res);
            assert_eq!(buf.width, res);
            assert_eq!(buf.height, res);
            assert_eq!(buf.data.len(), (res * res * 3) as usize);
        }
    }

    #[test]
    fn render_eight_letter_word() {
        let state = AnagramState {
            letters: "tgsrhnet".chars().collect(),
            valid_words: vec!["strength".to_string()],
            identified: false,
        };
        let buf = render(&state, 256);
        assert_eq!(buf.width, 256);
    }

    #[test]
    fn render_ten_letter_word_two_rows() {
        let state = AnagramState {
            letters: "lkbcismath".chars().collect(),
            valid_words: vec!["blacksmith".to_string()],
            identified: false,
        };
        let buf = render(&state, 256);
        assert_eq!(buf.width, 256);
        assert_eq!(buf.height, 256);
    }

    #[test]
    fn tile_color_deterministic() {
        let letters: Vec<char> = "church".chars().collect();
        assert_eq!(tile_color(&letters), tile_color(&letters));
    }

    #[test]
    fn tile_color_varies_by_word() {
        let w1: Vec<char> = "church".chars().collect();
        let w2: Vec<char> = "growth".chars().collect();
        let _ = tile_color(&w1);
        let _ = tile_color(&w2);
    }
}
