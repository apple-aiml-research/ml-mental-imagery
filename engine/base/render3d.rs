//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
use glam::{Mat4, Vec3, Vec4};

use super::game::RgbBuffer;

#[derive(Clone)]
pub struct Vertex {
    pub pos: Vec3,
    pub color: [u8; 3],
}

pub struct Triangle {
    pub verts: [Vertex; 3],
}

pub struct Mesh {
    pub tris: Vec<Triangle>,
}

impl Mesh {
    /// Transform all vertex positions in-place by the given matrix.
    pub fn transform(&mut self, mat: Mat4) {
        for tri in &mut self.tris {
            for vert in &mut tri.verts {
                vert.pos = (mat * vert.pos.extend(1.0)).truncate();
            }
        }
    }
}

#[derive(Clone)]
pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub ortho_scale: f32, // world-units visible vertically
}

#[derive(Clone)]
pub struct Light {
    pub dir: Vec3,    // unit vector toward light
    pub ambient: f32, // 0.0–1.0
    pub diffuse: f32, // 0.0–1.0
}

pub struct Scene {
    pub meshes: Vec<Mesh>,
    pub camera: Camera,
    pub light: Light,
    pub bg: [u8; 3],
}

/// Build a look-at view matrix (world → view).
fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
    let f = (target - eye).normalize();
    let r = f.cross(up).normalize();
    let u = r.cross(f);
    // Column-major: glam Mat4 is column-major
    Mat4::from_cols(
        Vec4::new(r.x, u.x, -f.x, 0.0),
        Vec4::new(r.y, u.y, -f.y, 0.0),
        Vec4::new(r.z, u.z, -f.z, 0.0),
        Vec4::new(-r.dot(eye), -u.dot(eye), f.dot(eye), 1.0),
    )
}

/// Aspect-correct orthographic projection.
/// `scale` = vertical extent in world units. Horizontal scales by `aspect` = width/height.
fn ortho_rect(scale: f32, aspect: f32) -> Mat4 {
    let half = scale / 2.0;
    Mat4::from_cols(
        Vec4::new(1.0 / (half * aspect), 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0 / half, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0 / half, 0.0),
        Vec4::ZERO.truncate().extend(1.0),
    )
}

/// Edge function for triangle rasterization.
#[inline]
fn edge(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

/// Render a scene to a square RGB buffer.
pub fn render_scene(scene: &Scene, resolution: u32) -> RgbBuffer {
    render_scene_rect(scene, resolution, resolution)
}

/// Render a scene to a (possibly non-square) RGB buffer (opaque triangles, z-buffered).
pub fn render_scene_rect(scene: &Scene, width: u32, height: u32) -> RgbBuffer {
    let w = width as usize;
    let h = height as usize;
    let mut color_buf = vec![scene.bg; w * h];
    let mut z_buf = vec![f32::NEG_INFINITY; w * h];

    let view = look_at(scene.camera.eye, scene.camera.target, scene.camera.up);
    let aspect = width as f32 / height as f32;
    let proj = ortho_rect(scene.camera.ortho_scale, aspect);
    let vp = proj * view;

    let light_dir = scene.light.dir.normalize();

    // --- Collect triangles ---
    struct PreparedTri {
        screen: [[f32; 3]; 3], // [x_screen, y_screen, z_clip] per vertex
        shaded: [u8; 3],
    }

    let mut prepared_tris: Vec<PreparedTri> = Vec::new();

    for mesh in &scene.meshes {
        for tri in &mesh.tris {
            // Transform vertices to clip space
            let mut screen = [[0.0f32; 3]; 3];
            for i in 0..3 {
                let p = vp * tri.verts[i].pos.extend(1.0);
                screen[i][0] = (p.x + 1.0) * 0.5 * w as f32;
                screen[i][1] = (1.0 - p.y) * 0.5 * h as f32;
                screen[i][2] = p.z;
            }

            // Back-face cull
            let dx1 = screen[1][0] - screen[0][0];
            let dy1 = screen[1][1] - screen[0][1];
            let dx2 = screen[2][0] - screen[0][0];
            let dy2 = screen[2][1] - screen[0][1];
            let cross = dx1 * dy2 - dy1 * dx2;
            if cross >= 0.0 {
                continue;
            }

            // Lighting
            let e1 = tri.verts[1].pos - tri.verts[0].pos;
            let e2 = tri.verts[2].pos - tri.verts[0].pos;
            let normal = e1.cross(e2).normalize();
            let ndotl = normal.dot(light_dir).max(0.0);
            let shade = (scene.light.ambient + scene.light.diffuse * ndotl).min(1.0);

            let base = tri.verts[0].color;
            let shaded = [
                (base[0] as f32 * shade).min(255.0) as u8,
                (base[1] as f32 * shade).min(255.0) as u8,
                (base[2] as f32 * shade).min(255.0) as u8,
            ];

            prepared_tris.push(PreparedTri { screen, shaded });
        }
    }

    // Helper closure to rasterize a triangle
    let rasterize = |tri: &PreparedTri, color_buf: &mut Vec<[u8; 3]>, z_buf: &mut Vec<f32>| {
        let s = &tri.screen;
        let min_x = s[0][0].min(s[1][0]).min(s[2][0]).max(0.0) as i32;
        let max_x = s[0][0].max(s[1][0]).max(s[2][0]).min(w as f32 - 1.0) as i32;
        let min_y = s[0][1].min(s[1][1]).min(s[2][1]).max(0.0) as i32;
        let max_y = s[0][1].max(s[1][1]).max(s[2][1]).min(h as f32 - 1.0) as i32;

        let a = [s[0][0], s[0][1]];
        let b = [s[1][0], s[1][1]];
        let c = [s[2][0], s[2][1]];

        let area = edge(a, b, c);
        if area.abs() < 1e-6 {
            return;
        }
        let inv_area = 1.0 / area;

        for py in min_y..=max_y {
            for px in min_x..=max_x {
                let p = [px as f32 + 0.5, py as f32 + 0.5];

                let w0 = edge(b, c, p);
                let w1 = edge(c, a, p);
                let w2 = edge(a, b, p);

                if w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0 {
                    let bary0 = w0 * inv_area;
                    let bary1 = w1 * inv_area;
                    let bary2 = w2 * inv_area;
                    let z = bary0 * s[0][2] + bary1 * s[1][2] + bary2 * s[2][2];

                    let idx = py as usize * w + px as usize;
                    if z > z_buf[idx] {
                        z_buf[idx] = z;
                        color_buf[idx] = tri.shaded;
                    }
                }
            }
        }
    };

    for tri in &prepared_tris {
        rasterize(tri, &mut color_buf, &mut z_buf);
    }

    // Flatten to RGB bytes
    let mut data = Vec::with_capacity(w * h * 3);
    for pixel in &color_buf {
        data.push(pixel[0]);
        data.push(pixel[1]);
        data.push(pixel[2]);
    }

    RgbBuffer {
        width,
        height,
        data,
    }
}

/// Create a box mesh with 12 triangles (2 per face).
/// `face_colors` order: [+x, -x, +y, -y, +z, -z].
pub fn box_mesh(center: Vec3, half: Vec3, face_colors: [[u8; 3]; 6]) -> Mesh {
    let c = center;
    let h = half;

    // 8 corners
    let corners = [
        Vec3::new(c.x - h.x, c.y - h.y, c.z - h.z), // 0: ---
        Vec3::new(c.x + h.x, c.y - h.y, c.z - h.z), // 1: +--
        Vec3::new(c.x + h.x, c.y + h.y, c.z - h.z), // 2: ++-
        Vec3::new(c.x - h.x, c.y + h.y, c.z - h.z), // 3: -+-
        Vec3::new(c.x - h.x, c.y - h.y, c.z + h.z), // 4: --+
        Vec3::new(c.x + h.x, c.y - h.y, c.z + h.z), // 5: +-+
        Vec3::new(c.x + h.x, c.y + h.y, c.z + h.z), // 6: +++
        Vec3::new(c.x - h.x, c.y + h.y, c.z + h.z), // 7: -++
    ];

    // Each face: 2 triangles, CCW winding when viewed from outside
    let faces: [([usize; 4], usize); 6] = [
        ([1, 2, 6, 5], 0), // +x face
        ([0, 4, 7, 3], 1), // -x face
        ([3, 7, 6, 2], 2), // +y face (top)
        ([0, 1, 5, 4], 3), // -y face (bottom)
        ([4, 5, 6, 7], 4), // +z face
        ([0, 3, 2, 1], 5), // -z face
    ];

    let mut tris = Vec::with_capacity(12);
    for (indices, color_idx) in &faces {
        let col = face_colors[*color_idx];
        let v = |i: usize| Vertex {
            pos: corners[indices[i]],
            color: col,
        };
        tris.push(Triangle {
            verts: [v(0), v(1), v(2)],
        });
        tris.push(Triangle {
            verts: [v(0), v(2), v(3)],
        });
    }

    Mesh { tris }
}

/// Project a world-space point to pixel coordinates for a given camera and resolution.
/// Returns `(px, py)` in pixel space where (0,0) is top-left.
/// Points outside the viewport may return coordinates outside `[0, resolution)`.
pub fn project_to_screen(camera: &Camera, point: Vec3, width: u32, height: u32) -> (f32, f32) {
    let view = look_at(camera.eye, camera.target, camera.up);
    let aspect = width as f32 / height as f32;
    let proj = ortho_rect(camera.ortho_scale, aspect);
    let vp = proj * view;
    let p = vp * point.extend(1.0);
    let px = (p.x + 1.0) * 0.5 * width as f32;
    let py = (1.0 - p.y) * 0.5 * height as f32;
    (px, py)
}

/// Standard isometric camera: 45° azimuth, arctan(1/√2) ≈ 35.264° elevation.
pub fn isometric_camera(target: Vec3, distance: f32, ortho_scale: f32) -> Camera {
    // Classic isometric direction: rotate 45° around Y, then tilt up ~35.264°
    let angle_h = std::f32::consts::FRAC_PI_4; // 45°
    let angle_v = (1.0_f32 / 2.0_f32.sqrt()).atan(); // arctan(1/√2)

    let dir = Vec3::new(
        angle_h.cos() * angle_v.cos(),
        angle_v.sin(),
        angle_h.sin() * angle_v.cos(),
    );

    Camera {
        eye: target + dir * distance,
        target,
        up: Vec3::Y,
        ortho_scale,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_colors() -> [[u8; 3]; 6] {
        [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
        ]
    }

    fn simple_scene(meshes: Vec<Mesh>) -> Scene {
        Scene {
            meshes,
            camera: isometric_camera(Vec3::ZERO, 10.0, 4.0),
            light: Light {
                dir: Vec3::new(0.3, 1.0, 0.2).normalize(),
                ambient: 0.4,
                diffuse: 0.6,
            },
            bg: [0, 0, 0],
        }
    }

    #[test]
    fn render_dimensions() {
        let scene = simple_scene(vec![]);
        let buf = render_scene(&scene, 64);
        assert_eq!(buf.width, 64);
        assert_eq!(buf.height, 64);
        assert_eq!(buf.data.len(), 64 * 64 * 3);
    }

    #[test]
    fn render_single_cube() {
        let mesh = box_mesh(Vec3::ZERO, Vec3::splat(0.5), test_colors());
        let scene = simple_scene(vec![mesh]);
        let buf = render_scene(&scene, 64);
        // Should have non-background pixels
        assert!(buf.data.chunks(3).any(|p| p != [0, 0, 0]));
    }

    #[test]
    fn render_zbuffer() {
        // Front cube (closer to camera) should occlude back cube
        let colors_front: [[u8; 3]; 6] = [[255; 3]; 6];
        let colors_back: [[u8; 3]; 6] = [[100; 3]; 6];

        let front = box_mesh(Vec3::new(0.5, 0.5, 0.5), Vec3::splat(0.5), colors_front);
        let back = box_mesh(Vec3::new(-0.5, -0.5, -0.5), Vec3::splat(0.5), colors_back);

        let scene = simple_scene(vec![front, back]);
        let buf = render_scene(&scene, 64);

        // The front cube's pixels should dominate the center region: at least
        // some bright (255-range) pixels are visible
        let bright_count = buf
            .data
            .chunks(3)
            .filter(|p| p[0] > 200 && p[1] > 200 && p[2] > 200)
            .count();
        assert!(bright_count > 0, "Front cube should produce bright pixels");
    }

    #[test]
    fn render_backface_cull() {
        // Create a single triangle facing away from the camera
        // Camera looks in -Z direction (from positive Z), so a triangle with
        // normal pointing in +Z (away from viewer) should be culled
        let camera = Camera {
            eye: Vec3::new(0.0, 0.0, 5.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            ortho_scale: 4.0,
        };
        // This triangle's world-space normal points -Z (away from camera at +Z).
        // Reversed winding so the screen-space cross product marks it as back-facing.
        let tri = Triangle {
            verts: [
                Vertex {
                    pos: Vec3::new(-1.0, -1.0, 0.0),
                    color: [255, 0, 0],
                },
                Vertex {
                    pos: Vec3::new(0.0, 1.0, 0.0),
                    color: [255, 0, 0],
                },
                Vertex {
                    pos: Vec3::new(1.0, -1.0, 0.0),
                    color: [255, 0, 0],
                },
            ],
        };
        let scene = Scene {
            meshes: vec![Mesh { tris: vec![tri] }],
            camera,
            light: Light {
                dir: Vec3::Y,
                ambient: 1.0,
                diffuse: 0.0,
            },
            bg: [0, 0, 0],
        };
        let buf = render_scene(&scene, 32);
        // All pixels should be background (backface culled)
        assert!(
            buf.data.iter().all(|&b| b == 0),
            "Backface should be culled"
        );
    }

    #[test]
    fn render_empty_scene() {
        let scene = Scene {
            meshes: vec![],
            camera: isometric_camera(Vec3::ZERO, 10.0, 4.0),
            light: Light {
                dir: Vec3::Y,
                ambient: 0.5,
                diffuse: 0.5,
            },
            bg: [42, 42, 42],
        };
        let buf = render_scene(&scene, 16);
        // All pixels should be background
        assert!(buf.data.chunks(3).all(|p| p == [42, 42, 42]));
    }

    #[test]
    fn box_mesh_triangle_count() {
        let mesh = box_mesh(Vec3::ZERO, Vec3::ONE, test_colors());
        assert_eq!(mesh.tris.len(), 12);
    }

    #[test]
    fn mesh_transform_translates() {
        let mut mesh = box_mesh(Vec3::ZERO, Vec3::splat(0.5), test_colors());
        let translate = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        mesh.transform(translate);
        // All vertices should now be offset by (1,2,3)
        for tri in &mesh.tris {
            for vert in &tri.verts {
                assert!(vert.pos.x >= 0.5 && vert.pos.x <= 1.5);
                assert!(vert.pos.y >= 1.5 && vert.pos.y <= 2.5);
                assert!(vert.pos.z >= 2.5 && vert.pos.z <= 3.5);
            }
        }
    }

    #[test]
    fn render_scene_rect_nonsquare() {
        let mesh = box_mesh(Vec3::ZERO, Vec3::splat(0.5), test_colors());
        let scene = simple_scene(vec![mesh]);
        let buf = render_scene_rect(&scene, 128, 64);
        assert_eq!(buf.width, 128);
        assert_eq!(buf.height, 64);
        assert_eq!(buf.data.len(), 128 * 64 * 3);
        assert!(buf.data.chunks(3).any(|p| p != [0, 0, 0]));
    }
}
