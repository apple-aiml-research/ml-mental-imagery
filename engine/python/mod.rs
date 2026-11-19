//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// PyO3 bindings for the puzzle engine (`mental_imagery.engine`).
//
//   GameConfig         one puzzle family plus its difficulty parameters
//   VisualTokenConfig  visual-token layout for the Visual CoT method
//   PuzzleEngine       engine: tokenizer, special tokens, episode loader factory
//                      (board_text="state" writes boards as JSON state descriptions
//                      for State CoT, board_text="boxes" as object boxes for Boxes CoT)
//   Loader, Rollout    batches of episodes with expert labels and rendered frames
//   available_games    names of the registered games

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use tokenizers::Tokenizer;

use crate::base::game::RgbBuffer;
use crate::games::registry::{self, ConfigParamType, ConfigValue};
use crate::stream::episode_stream::{Phase, SpecialTokenIds, TextBoards, VisualTokenConfig};
use crate::stream::loader::{GameFactory, Loader, PuzzleEngine};

/// Label value for positions that receive no supervision. Python maps it to -100.
const IGNORE_TOKEN_ID: u32 = u32::MAX;
/// Placeholder written at visual-token positions during teacher forcing. Python
/// replaces it with real codebook tokens after encoding the rendered frames.
const VISUAL_TOKEN_PLACEHOLDER_ID: u32 = u32::MAX - 1;
/// Stand-ins for <board>/</board> when boards are disabled (Action Supervision, Visual
/// Supervision). They are never emitted or matched in that mode.
const UNUSED_BOARD_OPEN_ID: u32 = u32::MAX - 2;
const UNUSED_BOARD_CLOSE_ID: u32 = u32::MAX - 3;
/// Token budget used when generating a full (untruncated) episode before
/// random truncation. Every episode terminates well before this.
const FULL_EPISODE_TOKEN_BUDGET: usize = 1_000_000;

// ---------------------------------------------------------------------------
// VisualTokenConfig
// ---------------------------------------------------------------------------

#[pyclass(name = "VisualTokenConfig", from_py_object)]
#[derive(Clone)]
struct PyVisualTokenConfig {
    codebook_size: usize,
    short_edge_res: u32,
    token_short_edge: Option<u32>,
}

#[pymethods]
impl PyVisualTokenConfig {
    #[new]
    #[pyo3(signature = (*, codebook_size, short_edge_res, token_short_edge=None))]
    fn new(codebook_size: usize, short_edge_res: u32, token_short_edge: Option<u32>) -> Self {
        PyVisualTokenConfig {
            codebook_size,
            short_edge_res,
            token_short_edge,
        }
    }
}

// ---------------------------------------------------------------------------
// GameConfig
// ---------------------------------------------------------------------------

#[pyclass(name = "GameConfig", from_py_object)]
#[derive(Clone)]
struct PyGameConfig {
    game_name: String,
    overrides: HashMap<String, ConfigValue>,
}

#[pymethods]
impl PyGameConfig {
    #[new]
    #[pyo3(signature = (name, **kwargs))]
    fn new(name: &str, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let entry = registry::find_game(name).ok_or_else(|| {
            let names: Vec<&str> = registry::all_games().iter().map(|e| e.name).collect();
            pyo3::exceptions::PyValueError::new_err(format!(
                "unknown game '{name}'; available: {names:?}"
            ))
        })?;

        let mut overrides = HashMap::new();
        if let Some(kw) = kwargs {
            for (key, value) in kw.iter() {
                let key_str: String = key.extract()?;
                let param = entry
                    .params
                    .iter()
                    .find(|p| p.key == key_str)
                    .ok_or_else(|| {
                        let keys: Vec<&str> = entry.params.iter().map(|p| p.key).collect();
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "unknown param '{key_str}' for game '{name}'; available: {keys:?}"
                        ))
                    })?;
                let cv = match param.param_type {
                    ConfigParamType::Int => {
                        let v: i64 = value.extract()?;
                        ConfigValue::Int(v)
                    }
                    ConfigParamType::IntRange => {
                        let seq: Vec<i64> = value.extract()?;
                        if seq.len() != 2 {
                            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                "param '{key_str}' expects [lo, hi], got {}-element sequence",
                                seq.len()
                            )));
                        }
                        if seq[0] > seq[1] {
                            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                "param '{key_str}' expects lo <= hi, got [{}, {}]",
                                seq[0], seq[1]
                            )));
                        }
                        ConfigValue::IntRange(seq[0], seq[1])
                    }
                };
                overrides.insert(key_str, cv);
            }
        }

        Ok(PyGameConfig {
            game_name: name.to_string(),
            overrides,
        })
    }
}

impl PyGameConfig {
    fn into_factory(&self) -> PyResult<Box<dyn GameFactory>> {
        let entry = registry::find_game(&self.game_name).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("unknown game '{}'", self.game_name))
        })?;
        (entry.build)(entry.name, entry.params, &self.overrides).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "failed to build game '{}': {e}",
                self.game_name
            ))
        })
    }
}

// ---------------------------------------------------------------------------
// available_games()
// ---------------------------------------------------------------------------

/// Names of the registered games, sorted.
#[pyfunction]
fn available_games() -> Vec<&'static str> {
    registry::all_games()
        .iter()
        .map(|entry| entry.name)
        .collect()
}

// ---------------------------------------------------------------------------
// PuzzleEngine
// ---------------------------------------------------------------------------

#[pyclass(name = "PuzzleEngine")]
struct PyPuzzleEngine {
    inner: Arc<PuzzleEngine>,
}

fn require_token(vocab: &HashMap<String, u32>, token: &str, role: &str) -> PyResult<u32> {
    vocab.get(token).copied().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("tokenizer is missing {token:?} ({role})"))
    })
}

#[pymethods]
impl PyPuzzleEngine {
    #[new]
    #[pyo3(signature = (
        tokenizer_path,
        *,
        render_resolution,
        eos_token_id,
        pad_token_id,
        think_close_token,
        im_end_token,
        rank=0,
        seed=42,
        visual_tokens=None,
        board_text=None,
        action_limit_multiplier=1.5,
        parallel=true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        tokenizer_path: &str,
        render_resolution: u32,
        eos_token_id: u32,
        pad_token_id: u32,
        think_close_token: &str,
        im_end_token: &str,
        rank: usize,
        seed: u64,
        visual_tokens: Option<PyVisualTokenConfig>,
        board_text: Option<&str>,
        action_limit_multiplier: f64,
        parallel: bool,
    ) -> PyResult<Self> {
        let text_boards = match board_text {
            None => None,
            Some("state") => Some(TextBoards::StateDescriptions),
            Some("boxes") => Some(TextBoards::ObjectBoxes),
            Some(other) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "board_text must be \"state\" (State CoT), \"boxes\" (Boxes CoT) or None; got {other:?}"
                )))
            }
        };
        if text_boards.is_some() && visual_tokens.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "board_text (State CoT, Boxes CoT) and visual_tokens (Visual CoT) are mutually exclusive",
            ));
        }
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "failed to load tokenizer from {tokenizer_path}: {e}"
            ))
        })?;
        let vocab = tokenizer.get_vocab(true);

        let action_open = require_token(&vocab, "<action>", "action delimiter")?;
        let action_close = require_token(&vocab, "</action>", "action delimiter")?;
        let think_close = require_token(&vocab, think_close_token, "closes the reasoning span")?;
        let im_end = require_token(&vocab, im_end_token, "ends the assistant turn")?;

        let mut visual_token_config: Option<VisualTokenConfig> = None;
        let (board_open, board_close) = if let Some(ref visual_tokens_config) = visual_tokens {
            for code in 0..visual_tokens_config.codebook_size {
                require_token(&vocab, &format!("<v{code}>"), "visual token")?;
            }
            visual_token_config = Some(VisualTokenConfig {
                short_edge_res: visual_tokens_config.short_edge_res,
                token_short_edge: visual_tokens_config.token_short_edge,
            });
            (
                require_token(&vocab, "<board>", "board delimiter")?,
                require_token(&vocab, "</board>", "board delimiter")?,
            )
        } else if text_boards.is_some() {
            (
                require_token(&vocab, "<board>", "board delimiter")?,
                require_token(&vocab, "</board>", "board delimiter")?,
            )
        } else {
            (UNUSED_BOARD_OPEN_ID, UNUSED_BOARD_CLOSE_ID)
        };

        let special = SpecialTokenIds {
            action_open,
            action_close,
            eos: eos_token_id,
            ignore: IGNORE_TOKEN_ID,
            pad: pad_token_id,
            board_open,
            board_close,
            visual_token_placeholder: VISUAL_TOKEN_PLACEHOLDER_ID,
            think_close,
            im_end,
        };

        let mut puzzle_engine = PuzzleEngine::new(
            tokenizer,
            seed,
            rank,
            special,
            visual_token_config,
            render_resolution,
            action_limit_multiplier,
            parallel,
        );
        if let Some(text_boards) = text_boards {
            puzzle_engine = puzzle_engine.with_text_boards(text_boards);
        }
        Ok(PyPuzzleEngine {
            inner: Arc::new(puzzle_engine),
        })
    }

    /// Token ids of the structural tokens, keyed by role.
    #[getter]
    fn special_tokens(&self) -> HashMap<String, u32> {
        let s = &self.inner.special;
        let mut m = HashMap::new();
        m.insert("action_open".into(), s.action_open);
        m.insert("action_close".into(), s.action_close);
        m.insert("eos".into(), s.eos);
        m.insert("board_open".into(), s.board_open);
        m.insert("board_close".into(), s.board_close);
        m.insert(
            "visual_token_placeholder".into(),
            s.visual_token_placeholder,
        );
        m.insert("im_end".into(), s.im_end);
        m
    }

    /// Create an episode loader.
    ///
    /// Every game gets an equal share of each batch; `prefetch` batches are
    /// generated ahead of time on background threads (0 = synchronous).
    ///
    /// `excluded_seeds[i]`, if given, lists seeds whose puzzles game `i` never
    /// generates (training loaders exclude evaluation puzzles this way).
    #[pyo3(signature = (games, n, prefetch=2, excluded_seeds=None))]
    fn loader(
        &self,
        py: Python<'_>,
        games: Vec<PyGameConfig>,
        n: usize,
        prefetch: usize,
        excluded_seeds: Option<Vec<Vec<u64>>>,
    ) -> PyResult<PyLoader> {
        let mut factories: Vec<Box<dyn GameFactory>> = games
            .iter()
            .map(|g| g.into_factory())
            .collect::<PyResult<_>>()?;
        if let Some(excluded_seeds) = excluded_seeds {
            if excluded_seeds.len() != factories.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "excluded_seeds has {} entries for {} games",
                    excluded_seeds.len(),
                    factories.len()
                )));
            }
            py.detach(|| {
                for (factory, seeds) in factories.iter_mut().zip(&excluded_seeds) {
                    if !seeds.is_empty() {
                        factory.exclude_puzzles_of_seeds(seeds);
                    }
                }
            });
        }
        let loader = self.inner.loader(factories, n, prefetch);
        Ok(PyLoader {
            inner: Mutex::new(loader),
        })
    }
}

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

#[pyclass(name = "Loader")]
struct PyLoader {
    inner: Mutex<Loader>,
}

#[pymethods]
impl PyLoader {
    /// Next batch of freshly generated episodes (training).
    fn next(&self, py: Python<'_>) -> PyRollout {
        let prepared = py.detach(|| self.inner.lock().unwrap().next());
        PyRollout::from_prepared(prepared)
    }

    /// Batch of episodes with explicit per-episode puzzle seeds (evaluation).
    /// Every episode uses the first game of the loader.
    #[pyo3(signature = (seeds))]
    fn next_with_seeds(&self, py: Python<'_>, seeds: Vec<u64>) -> PyRollout {
        let prepared = py.detach(|| self.inner.lock().unwrap().next_with_seeds(&seeds));
        PyRollout::from_prepared(prepared)
    }

    /// `Game::puzzle_fingerprint` (u64) of the puzzle each seed generates for game
    /// `game_index`, without solving or rendering it.
    #[pyo3(signature = (game_index, seeds))]
    fn puzzle_fingerprints(
        &self,
        py: Python<'_>,
        game_index: usize,
        seeds: Vec<u64>,
    ) -> PyResult<Vec<u64>> {
        let loader = self.inner.lock().unwrap();
        if game_index >= loader.num_games() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "game_index {game_index} out of range"
            )));
        }
        Ok(py.detach(|| loader.puzzle_fingerprints(game_index, &seeds)))
    }

    /// Training episodes of batches `start .. start + count`, exactly as `next()` generates
    /// them, whose puzzle fingerprint is in `targets[game]` for their assigned game.
    /// Returns `(game_index, batch_index, seed)` for each match.
    #[pyo3(signature = (targets, start, count))]
    fn find_training_puzzles(
        &self,
        py: Python<'_>,
        targets: Vec<HashSet<u64>>,
        start: u64,
        count: u64,
    ) -> PyResult<Vec<(usize, u64, u64)>> {
        let loader = self.inner.lock().unwrap();
        if targets.len() != loader.num_games() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected {} target sets (one per game), got {}",
                loader.num_games(),
                targets.len()
            )));
        }
        Ok(py.detach(|| loader.find_training_puzzles(&targets, start, count)))
    }
}

// ---------------------------------------------------------------------------
// Rollout
// ---------------------------------------------------------------------------

#[pyclass(name = "Rollout")]
struct PyRollout {
    rollout: Mutex<crate::stream::batch::Rollout>,
    raw_images: Vec<RgbBuffer>,
    prompt_strings: Vec<String>,
    action_strings: Vec<String>,
    /// Result of `teacher_force_labels_randomly_truncated`, consumed by the next
    /// `teacher_force_labels` call.
    cached_tf_result: Option<crate::stream::batch::TeacherForceResult>,
    /// PIL images built from `raw_images`; invalidated when they are swapped.
    pil_cache: Option<Vec<Py<PyAny>>>,
}

impl PyRollout {
    fn from_prepared(prepared: crate::stream::loader::PreparedRollout) -> Self {
        PyRollout {
            rollout: Mutex::new(prepared.rollout),
            raw_images: prepared.initial_images,
            prompt_strings: prepared.prompts,
            action_strings: prepared.actions,
            cached_tf_result: None,
            pil_cache: None,
        }
    }
}

#[pymethods]
impl PyRollout {
    /// Execute one model token per episode (evaluation replay).
    ///
    /// `model_tokens` holds one token id per episode. Returns a numpy bool
    /// array: whether each episode has ended.
    fn step<'py>(
        &self,
        py: Python<'py>,
        model_tokens: PyReadonlyArray1<'py, u32>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let tokens = model_tokens.as_slice()?;
        let done = py.detach(|| self.rollout.lock().unwrap().step(tokens));
        Ok(done.into_pyarray(py))
    }

    /// Expert (teacher-forced) token sequences for the whole batch.
    ///
    /// Returns (inputs [B, max_tokens] uint32, labels [B, max_tokens] uint32,
    /// lengths [B] int64, num_truncated). Sequences are right-padded with PAD.
    /// Labels equal inputs (the expert trajectory). Visual-token positions hold
    /// the deferred sentinel in both arrays.
    ///
    /// If `teacher_force_labels_randomly_truncated` ran first, its cached result
    /// is returned.
    fn teacher_force_labels<'py>(
        &mut self,
        py: Python<'py>,
        max_tokens: usize,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>, Bound<'py, PyArray1<i64>>, usize)> {
        let result = if let Some(cached) = self.cached_tf_result.take() {
            cached
        } else {
            py.detach(|| {
                self.rollout
                    .lock()
                    .unwrap()
                    .teacher_force_labels(max_tokens)
            })
        };

        let batch_size = result.sequences.len();
        let pad = result.pad_token;
        let num_truncated = result.num_truncated;

        let mut flat_inputs = vec![pad; batch_size * max_tokens];
        for (i, sequence) in result.sequences.iter().enumerate() {
            let len = sequence.len().min(max_tokens);
            flat_inputs[i * max_tokens..i * max_tokens + len].copy_from_slice(&sequence[..len]);
        }
        let flat_labels = flat_inputs.clone();
        let lengths: Vec<i64> = result
            .lengths
            .iter()
            .map(|&l| l.min(max_tokens) as i64)
            .collect();

        let inputs_2d = flat_inputs
            .into_pyarray(py)
            .call_method1("reshape", ((batch_size, max_tokens),))?
            .unbind();
        let labels_2d = flat_labels
            .into_pyarray(py)
            .call_method1("reshape", ((batch_size, max_tokens),))?
            .unbind();

        Ok((
            inputs_2d,
            labels_2d,
            lengths.into_pyarray(py),
            num_truncated,
        ))
    }

    /// Teacher forcing with random truncation of long episodes.
    ///
    /// Generates each full episode, and for episodes longer than `max_tokens`
    /// picks a random step boundary as the new start. The rollout's initial
    /// images and accumulated frames are rewritten to match, and the result is
    /// cached for the next `teacher_force_labels` call.
    ///
    /// Returns per-episode stats: [{"original_length"}], the token count
    /// before truncation; `original_length == 0` means not truncated.
    fn teacher_force_labels_randomly_truncated(
        &mut self,
        py: Python<'_>,
        max_tokens: usize,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let (result, initial_frames, original_lengths) = py.detach(|| {
            let mut rollout = self.rollout.lock().unwrap();
            let mut result = rollout.teacher_force_labels(FULL_EPISODE_TOKEN_BUDGET);
            let (initial_frames, original_lengths) =
                rollout.apply_random_truncation(&mut result, max_tokens);
            (result, initial_frames, original_lengths)
        });

        // Swap the prompt images for the frames at each truncation start.
        if !initial_frames.is_empty() {
            self.raw_images = initial_frames;
            self.pil_cache = None;
        }
        self.cached_tf_result = Some(result);

        original_lengths
            .iter()
            .map(|&original_length| {
                let dict = PyDict::new(py);
                dict.set_item("original_length", original_length)?;
                Ok(dict.into_any().unbind())
            })
            .collect()
    }

    /// Drain the frames rendered at each action boundary.
    ///
    /// Returns (images, frame_counts): a flat list of PIL images and the number
    /// of frames per episode (the initial board followed by one frame per action).
    fn take_frames(&self, py: Python<'_>) -> PyResult<(Vec<Py<PyAny>>, Vec<usize>)> {
        let mut rollout = self.rollout.lock().unwrap();
        let mut all_images = Vec::new();
        let mut frame_counts = Vec::new();
        for stream in rollout.streams.iter_mut() {
            let frames = stream.take_accumulated_frames();
            frame_counts.push(frames.len());
            for frame in frames {
                all_images.push(rgb_to_pil(py, &frame)?);
            }
        }
        Ok((all_images, frame_counts))
    }

    /// Drain the frames rendered at each action boundary as one array.
    ///
    /// Returns (frames, frame_counts): a uint8 array (num_frames, height, width, 3) of all
    /// episodes' frames, episode after episode, and the number of frames per episode (the initial
    /// board followed by one frame per action). The frames must share one size.
    fn take_frames_array<'py>(&self, py: Python<'py>) -> PyResult<(Py<PyAny>, Vec<usize>)> {
        let (pixels, frame_counts, size) = py
            .detach(|| {
                let mut rollout = self.rollout.lock().unwrap();
                let mut pixels = Vec::new();
                let mut frame_counts = Vec::new();
                let mut size = None;
                for stream in rollout.streams.iter_mut() {
                    let frames = stream.take_accumulated_frames();
                    frame_counts.push(frames.len());
                    for frame in frames {
                        if *size.get_or_insert((frame.height, frame.width))
                            != (frame.height, frame.width)
                        {
                            return Err(format!(
                                "frames of different sizes: {:?} and {}x{}",
                                size.unwrap(),
                                frame.height,
                                frame.width
                            ));
                        }
                        pixels.extend_from_slice(&frame.data);
                    }
                }
                Ok((pixels, frame_counts, size.unwrap_or((0, 0))))
            })
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let num_frames: usize = frame_counts.iter().sum();
        let frames = pixels
            .into_pyarray(py)
            .call_method1(
                "reshape",
                ((num_frames, size.0 as usize, size.1 as usize, 3usize),),
            )?
            .unbind();
        Ok((frames, frame_counts))
    }

    /// Board image shown in each episode's prompt.
    #[getter]
    fn initial_images(&mut self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        if let Some(ref cached) = self.pil_cache {
            return Ok(cached.iter().map(|o| o.clone_ref(py)).collect());
        }
        let images: Vec<Py<PyAny>> = self
            .raw_images
            .iter()
            .map(|img| rgb_to_pil(py, img))
            .collect::<PyResult<_>>()?;
        let ret = images.iter().map(|o| o.clone_ref(py)).collect();
        self.pil_cache = Some(images);
        Ok(ret)
    }

    /// Task description of each episode.
    #[getter]
    fn prompts(&self) -> Vec<String> {
        self.prompt_strings.clone()
    }

    /// Action-space description of each episode.
    #[getter]
    fn actions(&self) -> Vec<String> {
        self.action_strings.clone()
    }

    /// Every action string the replay accepts, per episode (base-model evaluation
    /// maps free-text answers onto these).
    #[getter]
    fn valid_actions(&self) -> Vec<Vec<String>> {
        let rollout = self.rollout.lock().unwrap();
        rollout
            .streams
            .iter()
            .map(|s| {
                s.valid_targets
                    .iter()
                    .map(|t| t.action_str.clone())
                    .collect()
            })
            .collect()
    }

    /// Position of each episode within a board block of visual tokens: -2 at
    /// `<board>`, i at the i-th visual token, -3 at `</board>`, -1 outside a board
    /// (token-replacement test).
    #[getter]
    fn board_slots(&self) -> Vec<i64> {
        let rollout = self.rollout.lock().unwrap();
        rollout
            .streams
            .iter()
            .map(|s| match s.phase {
                Phase::BoardOpen => -2,
                Phase::BoardVisual(i) => i as i64,
                Phase::BoardClose => -3,
                _ => -1,
            })
            .collect()
    }

    /// Visual tokens per board of each episode.
    #[getter]
    fn board_num_tokens(&self) -> Vec<usize> {
        let rollout = self.rollout.lock().unwrap();
        rollout.streams.iter().map(|s| s.num_tokens).collect()
    }

    /// Render the current board of every episode.
    fn render(&self, py: Python<'_>, resolution: u32) -> PyResult<Vec<Py<PyAny>>> {
        let rollout = self.rollout.lock().unwrap();
        rollout
            .streams
            .iter()
            .map(|s| rgb_to_pil(py, &s.episode.render(resolution)))
            .collect()
    }

    /// Render every episode to raw RGB bytes with the GIL released.
    ///
    /// Returns (data, width, height): B concatenated H*W*3 images. All episodes
    /// must render at the same size.
    fn render_bytes(&self, py: Python<'_>, resolution: u32) -> PyResult<(Py<PyAny>, u32, u32)> {
        let rendered = py.detach(|| {
            let rollout = self.rollout.lock().unwrap();
            let first = rollout.streams[0].episode.render(resolution);
            let (w, h) = (first.width, first.height);
            let mut flat = Vec::with_capacity(rollout.streams.len() * (w * h * 3) as usize);
            flat.extend_from_slice(&first.data);
            for s in rollout.streams.iter().skip(1) {
                let img = s.episode.render(resolution);
                if img.width != w || img.height != h {
                    return Err(format!(
                        "render_bytes needs equal frame sizes, got {}x{} and {}x{}",
                        w, h, img.width, img.height
                    ));
                }
                flat.extend_from_slice(&img.data);
            }
            Ok((flat, w, h))
        });
        let (flat, w, h) = rendered.map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok((PyBytes::new(py, &flat).into_any().unbind(), w, h))
    }

    #[getter]
    fn is_solved(&self) -> Vec<bool> {
        let rollout = self.rollout.lock().unwrap();
        rollout
            .streams
            .iter()
            .map(|s| s.episode.is_solved())
            .collect()
    }

    /// Whether each episode hit its action limit (ceil(multiplier * optimal)).
    #[getter]
    fn timed_out(&self) -> Vec<bool> {
        let rollout = self.rollout.lock().unwrap();
        rollout.streams.iter().map(|s| s.timed_out).collect()
    }

    /// Number of non-terminal actions executed in each episode.
    #[getter]
    fn action_counts(&self) -> Vec<usize> {
        let rollout = self.rollout.lock().unwrap();
        rollout.streams.iter().map(|s| s.action_count).collect()
    }

    /// Expert actions needed to solve each episode from its current state.
    fn solution_length(&self, py: Python<'_>) -> Vec<usize> {
        py.detach(|| {
            let rollout = self.rollout.lock().unwrap();
            rollout
                .streams
                .iter()
                .map(|s| s.episode.solution_length().unwrap_or(0))
                .collect()
        })
    }

    /// Apply up to `steps[i]` expert actions to episode i (stops early when
    /// the expert's next action is terminal). Does not render.
    fn advance_expert(&self, py: Python<'_>, steps: Vec<usize>) -> PyResult<()> {
        py.detach(|| {
            let mut rollout = self.rollout.lock().unwrap();
            let batch_size = rollout.streams.len();
            if steps.len() != batch_size {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "steps length {} != batch size {}",
                    steps.len(),
                    batch_size,
                )));
            }
            for (stream, &n) in rollout.streams.iter_mut().zip(steps.iter()) {
                for _ in 0..n {
                    if stream.episode.is_solved() {
                        break;
                    }
                    let action = stream.episode.expert_action();
                    if action == "done" || action == "yes" || action == "no" {
                        break;
                    }
                    stream.episode.apply(&action);
                }
            }
            Ok(())
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn rgb_to_pil(py: Python<'_>, img: &RgbBuffer) -> PyResult<Py<PyAny>> {
    let pil = py.import("PIL.Image")?;
    let bytes = PyBytes::new(py, &img.data);
    pil.call_method1("frombytes", ("RGB", (img.width, img.height), bytes))
        .map(|obj| obj.unbind())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVisualTokenConfig>()?;
    m.add_class::<PyGameConfig>()?;
    m.add_class::<PyPuzzleEngine>()?;
    m.add_class::<PyLoader>()?;
    m.add_class::<PyRollout>()?;
    m.add_function(wrap_pyfunction!(available_games, m)?)?;
    Ok(())
}
