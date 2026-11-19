//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
use std::sync::OnceLock;

use ab_glyph::{FontRef, PxScale};
use image::{Rgb, RgbImage};
use imageproc::drawing::draw_text_mut;
use imageproc::rect::Rect;

use super::game::RgbBuffer;

static FONT: OnceLock<FontRef<'static>> = OnceLock::new();
// Board labels, coordinates, and letters (Roboto Mono Bold, SIL Open Font License 1.1).
static FONT_BYTES: &[u8] = include_bytes!("../../assets/fonts/RobotoMono-Bold.ttf");

pub fn font() -> &'static FontRef<'static> {
    FONT.get_or_init(|| FontRef::try_from_slice(FONT_BYTES).expect("invalid embedded font"))
}

pub fn fit_to_resolution(board_w: u32, board_h: u32, resolution: u32) -> (u32, u32, u32, u32) {
    if board_w == 0 || board_h == 0 {
        return (0, 0, resolution, resolution);
    }
    let scale_w = resolution / board_w;
    let scale_h = resolution / board_h;
    let scale = scale_w.min(scale_h).max(1);
    let img_w = board_w * scale;
    let img_h = board_h * scale;
    let offset_x = (resolution - img_w) / 2;
    let offset_y = (resolution - img_h) / 2;
    (offset_x, offset_y, img_w, scale)
}

pub fn draw_text_centered(
    img: &mut RgbImage,
    cx: i32,
    cy: i32,
    ch: char,
    font_height: f32,
    color: Rgb<u8>,
) {
    let scale = PxScale::from(font_height);
    let char_w = (font_height * 0.6) as i32;
    let char_h = font_height as i32;
    let x = cx - char_w / 2;
    let y = cy - char_h / 2;
    let s = String::from(ch);
    draw_text_mut(img, color, x, y, scale, font(), &s);
}

pub fn draw_str_centered(
    img: &mut RgbImage,
    cx: i32,
    cy: i32,
    text: &str,
    font_height: f32,
    color: Rgb<u8>,
) {
    let scale = PxScale::from(font_height);
    let char_w = (font_height * 0.6) as i32;
    let char_h = font_height as i32;
    let total_w = char_w * text.len() as i32;
    let x = cx - total_w / 2;
    let y = cy - char_h / 2;
    draw_text_mut(img, color, x, y, scale, font(), text);
}

pub fn fill_diamond(img: &mut RgbImage, cx: u32, cy: u32, half: u32, color: Rgb<u8>) {
    for dy in 0..=half * 2 {
        for dx in 0..=half * 2 {
            let px = (dx as i64 - half as i64).unsigned_abs() as u32;
            let py = (dy as i64 - half as i64).unsigned_abs() as u32;
            if px + py <= half {
                let fx = cx as i64 + dx as i64 - half as i64;
                let fy = cy as i64 + dy as i64 - half as i64;
                if fx >= 0 && fy >= 0 && (fx as u32) < img.width() && (fy as u32) < img.height() {
                    img.put_pixel(fx as u32, fy as u32, color);
                }
            }
        }
    }
}

pub fn luminance(c: &Rgb<u8>) -> f32 {
    0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32
}

pub fn rect(x: u32, y: u32, w: u32, h: u32) -> Rect {
    Rect::at(x as i32, y as i32).of_size(w.max(1), h.max(1))
}

/// Box-filter downsample: average each `factor x factor` block into one output pixel.
/// Uses integer accumulation (u32) with final division.
/// Truncates remainder pixels if dimensions are not evenly divisible.
pub fn downsample(buf: &RgbBuffer, factor: u32) -> RgbBuffer {
    if factor <= 1 {
        return RgbBuffer {
            width: buf.width,
            height: buf.height,
            data: buf.data.clone(),
        };
    }
    let out_w = buf.width / factor;
    let out_h = buf.height / factor;
    let area = factor * factor;
    let mut data = Vec::with_capacity((out_w * out_h * 3) as usize);
    for oy in 0..out_h {
        for ox in 0..out_w {
            let mut r: u32 = 0;
            let mut g: u32 = 0;
            let mut b: u32 = 0;
            for dy in 0..factor {
                for dx in 0..factor {
                    let sx = ox * factor + dx;
                    let sy = oy * factor + dy;
                    let idx = ((sy * buf.width + sx) * 3) as usize;
                    r += buf.data[idx] as u32;
                    g += buf.data[idx + 1] as u32;
                    b += buf.data[idx + 2] as u32;
                }
            }
            data.push((r / area) as u8);
            data.push((g / area) as u8);
            data.push((b / area) as u8);
        }
    }
    RgbBuffer {
        width: out_w,
        height: out_h,
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_square_board() {
        let (ox, oy, _img_w, cell) = fit_to_resolution(4, 4, 64);
        assert_eq!(cell, 16);
        assert_eq!(ox, 0);
        assert_eq!(oy, 0);
    }

    #[test]
    fn fit_nonsquare_board() {
        let (ox, oy, _img_w, cell) = fit_to_resolution(4, 2, 64);
        assert_eq!(cell, 16); // limited by width
        assert!(oy > 0); // vertical centering offset
        assert_eq!(ox, 0);
    }

    #[test]
    fn fit_zero_dimensions() {
        let (ox, oy, img_w, cell) = fit_to_resolution(0, 0, 64);
        assert_eq!(ox, 0);
        assert_eq!(oy, 0);
        assert_eq!(img_w, 64);
        assert_eq!(cell, 64);
    }

    #[test]
    fn luminance_known_values() {
        let white = Rgb([255, 255, 255]);
        assert!((luminance(&white) - 255.0).abs() < 0.01);

        let black = Rgb([0, 0, 0]);
        assert!((luminance(&black) - 0.0).abs() < 0.01);

        // Pure red: 0.299 * 255 ≈ 76.245
        let red = Rgb([255, 0, 0]);
        assert!((luminance(&red) - 76.245).abs() < 0.01);
    }

    #[test]
    fn fill_diamond_within_bounds() {
        let mut img = RgbImage::from_pixel(16, 16, Rgb([0, 0, 0]));
        let color = Rgb([255, 0, 0]);
        fill_diamond(&mut img, 8, 8, 3, color);

        // Center pixel should be colored
        assert_eq!(*img.get_pixel(8, 8), color);
        // Corner of bounding box should NOT be colored (diamond, not square)
        assert_ne!(*img.get_pixel(5, 5), color);
    }

    #[test]
    fn fill_diamond_symmetric() {
        let mut img = RgbImage::from_pixel(32, 32, Rgb([0, 0, 0]));
        let color = Rgb([255, 0, 0]);
        fill_diamond(&mut img, 16, 16, 4, color);

        // Check symmetry around center
        for dy in 0..=4u32 {
            for dx in 0..=4u32 {
                let top_left = *img.get_pixel(16 - dx, 16 - dy);
                let top_right = *img.get_pixel(16 + dx, 16 - dy);
                let bot_left = *img.get_pixel(16 - dx, 16 + dy);
                let bot_right = *img.get_pixel(16 + dx, 16 + dy);
                assert_eq!(top_left, top_right);
                assert_eq!(top_left, bot_left);
                assert_eq!(top_left, bot_right);
            }
        }
    }

    #[test]
    fn downsample_factor_1_identity() {
        let buf = RgbBuffer {
            width: 4,
            height: 4,
            data: vec![128; 4 * 4 * 3],
        };
        let out = downsample(&buf, 1);
        assert_eq!(out.width, 4);
        assert_eq!(out.height, 4);
        assert_eq!(out.data, buf.data);
    }

    #[test]
    fn downsample_2x_known_values() {
        // 2x2 image downsampled to 1x1
        // pixels: (0,0,0), (100,200,50), (200,100,150), (100,100,50)
        // avg: r=(0+100+200+100)/4=100, g=(0+200+100+100)/4=100, b=(0+50+150+50)/4=62
        let data = vec![0, 0, 0, 100, 200, 50, 200, 100, 150, 100, 100, 50];
        let buf = RgbBuffer {
            width: 2,
            height: 2,
            data,
        };
        let out = downsample(&buf, 2);
        assert_eq!(out.width, 1);
        assert_eq!(out.height, 1);
        assert_eq!(out.data, vec![100, 100, 62]);
    }

    #[test]
    fn downsample_uniform_color_preserved() {
        let buf = RgbBuffer {
            width: 4,
            height: 4,
            data: vec![42; 4 * 4 * 3],
        };
        let out = downsample(&buf, 2);
        assert_eq!(out.width, 2);
        assert_eq!(out.height, 2);
        assert!(out.data.iter().all(|&v| v == 42));
    }

    #[test]
    fn downsample_dimension_correctness() {
        // 6x4 image at factor 2 -> 3x2
        let buf = RgbBuffer {
            width: 6,
            height: 4,
            data: vec![0; 6 * 4 * 3],
        };
        let out = downsample(&buf, 2);
        assert_eq!(out.width, 3);
        assert_eq!(out.height, 2);
        assert_eq!(out.data.len(), 3 * 2 * 3);
    }
}
