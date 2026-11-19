//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// Batched rollouts: teacher forcing, random truncation, and evaluation replay.

use rand::Rng;

use crate::base::game::RgbBuffer;
use crate::stream::episode_stream::{EpisodeStream, Phase};
use crate::stream::prefix::TokenId;

/// Expert token sequences from teacher forcing. Each sequence is both the
/// model input and its label.
pub struct TeacherForceResult {
    pub sequences: Vec<Vec<TokenId>>, // [batch_size][variable len]
    pub lengths: Vec<usize>,          // actual token count per episode
    pub pad_token: TokenId,           // for padding in Python binding
    pub num_truncated: usize,         // episodes that hit max_tokens without finishing
}

/// Batch of episodes.
pub struct Rollout {
    pub streams: Vec<EpisodeStream>,
}

impl Rollout {
    pub fn new(streams: Vec<EpisodeStream>) -> Self {
        Rollout { streams }
    }

    /// Execute one model token per episode (evaluation replay).
    ///
    /// Returns, per episode, whether it has ended.
    pub fn step(&mut self, model_tokens: &[TokenId]) -> Vec<bool> {
        assert_eq!(model_tokens.len(), self.streams.len());
        self.streams
            .iter_mut()
            .zip(model_tokens.iter())
            .map(|(stream, &token)| {
                if stream.phase != Phase::Done {
                    stream.advance(token);
                }
                stream.phase == Phase::Done
            })
            .collect()
    }

    /// Generate the expert token sequences for teacher forcing.
    ///
    /// Each position executes `expert_label()`. Frames rendered after each
    /// action are accumulated for deferred visual-token encoding or visual head
    /// supervision. visual-token positions hold a sentinel token (`visual_token_placeholder`) that
    /// Python replaces with actual visual tokens after batch encoding the frames.
    pub fn teacher_force_labels(&mut self, max_tokens: usize) -> TeacherForceResult {
        let batch_size = self.streams.len();
        let pad = self.streams[0].special.pad;
        let mut sequences: Vec<Vec<TokenId>> = vec![Vec::with_capacity(max_tokens); batch_size];
        let mut finished: Vec<bool> = vec![false; batch_size];

        for _pos in 0..max_tokens {
            // Check if all done before generating more tokens
            if finished.iter().all(|&f| f) {
                break;
            }

            for (i, stream) in self.streams.iter_mut().enumerate() {
                if finished[i] {
                    continue;
                }
                let label = stream.expert_label();
                stream.advance(label);
                sequences[i].push(label);
                if stream.phase == Phase::Done {
                    finished[i] = true;
                }
            }

            // Accumulate rendered frames for deferred visual-token encoding in Python
            for stream in self.streams.iter_mut() {
                if let Some(rgb) = stream.take_pending_render() {
                    stream.accumulated_frames.push(rgb);
                }
            }
        }

        let lengths: Vec<usize> = sequences.iter().map(|s| s.len()).collect();
        let num_truncated = finished.iter().filter(|&&f| !f).count();

        TeacherForceResult {
            sequences,
            lengths,
            pad_token: pad,
            num_truncated,
        }
    }

    /// Apply random truncation in-place: for episodes whose token sequence
    /// exceeds `max_tokens`, pick a random step boundary as the start,
    /// rewrite the sequence to contain only the truncated tokens, and adjust
    /// accumulated_frames to only include the truncated region's frames.
    ///
    /// With boards, uses `<board>` positions as step boundaries. In
    /// render-only mode (visual head), uses the position after each `</action>`.
    ///
    /// Returns (initial_frames, original_lengths): the frame at each episode's
    /// start, and its token count before truncation (0 if not truncated).
    pub fn apply_random_truncation(
        &mut self,
        result: &mut TeacherForceResult,
        max_tokens: usize,
    ) -> (Vec<RgbBuffer>, Vec<usize>) {
        let mut initial_frames = Vec::with_capacity(self.streams.len());
        let mut original_lengths = Vec::with_capacity(self.streams.len());

        for (i, stream) in self.streams.iter_mut().enumerate() {
            let sequence = &mut result.sequences[i];
            let frames = &mut stream.accumulated_frames;
            let has_frames = !frames.is_empty();
            let original_length = sequence.len();

            if sequence.len() <= max_tokens {
                // Episode fits — no truncation needed
                if has_frames {
                    initial_frames.push(frames[0].clone());
                }
                original_lengths.push(0);
                continue;
            }

            // Compute step boundaries: <board> positions with boards,
            // positions after </action> in render-only mode.
            let step_starts: Vec<usize> = if stream.has_boards {
                let board_open = stream.special.board_open;
                sequence
                    .iter()
                    .enumerate()
                    .filter(|(_, &t)| t == board_open)
                    .map(|(pos, _)| pos)
                    .collect()
            } else {
                let action_close = stream.special.action_close;
                let action_close_positions: Vec<usize> = sequence
                    .iter()
                    .enumerate()
                    .filter(|(_, &t)| t == action_close)
                    .map(|(pos, _)| pos)
                    .collect();
                std::iter::once(0)
                    .chain(action_close_positions.iter().map(|&pos| pos + 1))
                    .collect()
            };

            if step_starts.is_empty() || (step_starts.len() == 1 && step_starts[0] == 0) {
                // No meaningful boundaries — hard-truncate from position 0
                sequence.truncate(max_tokens);
                result.lengths[i] = sequence.len();
                if has_frames {
                    initial_frames.push(frames[0].clone());
                }
                original_lengths.push(original_length);
                continue;
            }

            // Find max_start_step: latest step where remaining > max_tokens.
            let mut max_start_step: usize = 0;
            for (step_idx, &pos) in step_starts.iter().enumerate() {
                if sequence.len() - pos > max_tokens {
                    max_start_step = step_idx;
                }
            }

            // Sample from [0, max_start_step + 1]
            let upper = (max_start_step + 1).min(step_starts.len() - 1);
            let start_step = stream.rng.gen_range(0..=upper);

            let token_start = step_starts[start_step];
            let hard_end = (token_start + max_tokens).min(sequence.len());

            // Snap token_end to the last step boundary that fits within
            // the budget, so an observation is never cut.
            let token_end = step_starts
                .iter()
                .rev()
                .find(|&&pos| pos <= hard_end && pos > token_start)
                .copied()
                .unwrap_or(hard_end);

            // Count boundary tokens in the truncated region to determine frame count.
            // With boards: count <board> tokens. Render-only: count </action> tokens + 1.
            let region_frame_count = if stream.has_boards {
                let board_open = stream.special.board_open;
                sequence[token_start..token_end]
                    .iter()
                    .filter(|&&t| t == board_open)
                    .count()
            } else {
                let action_close = stream.special.action_close;
                let action_close_count = sequence[token_start..token_end]
                    .iter()
                    .filter(|&&t| t == action_close)
                    .count();
                // Frames = initial + one per completed </action>
                action_close_count + 1
            };

            // Rewrite the sequence: shift the truncated region to start at 0.
            *sequence = sequence[token_start..token_end].to_vec();
            result.lengths[i] = sequence.len();

            // Adjust frames for the truncated region.
            // Frame k = obs_k; the truncated region uses frames
            // [start_step .. start_step + region_frame_count].
            if has_frames {
                initial_frames.push(frames[start_step].clone());

                let frame_end = (start_step + region_frame_count).min(frames.len());
                let truncated_frames: Vec<RgbBuffer> = frames[start_step..frame_end].to_vec();
                *frames = truncated_frames;
            }

            original_lengths.push(original_length);
        }

        // Update num_truncated to reflect episodes that were randomly truncated
        result.num_truncated = original_lengths
            .iter()
            .filter(|&&length| length > 0)
            .count();

        (initial_frames, original_lengths)
    }
}

#[cfg(test)]
impl Rollout {
    pub fn is_all_done(&self) -> bool {
        self.streams.iter().all(|s| s.phase == Phase::Done)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use rand::SeedableRng;
    use rand_chacha::ChaChaRng;
    use tokenizers::models::bpe::BPE;
    use tokenizers::Tokenizer;

    use crate::base::episode::GameEpisode;
    use crate::games::sokoban::state::from_ascii;
    use crate::games::sokoban::Sokoban;
    use crate::stream::episode_stream::SpecialTokenIds;

    const ACTION_OPEN: TokenId = 100;
    const ACTION_CLOSE: TokenId = 101;
    const EOS: TokenId = 102;
    const IGNORE: TokenId = 103;
    const PAD: TokenId = 104;
    const THINK_CLOSE: TokenId = 105;
    const IM_END: TokenId = 106;
    const BOARD_OPEN: TokenId = 200;
    const BOARD_CLOSE: TokenId = 201;

    fn test_special() -> SpecialTokenIds {
        SpecialTokenIds {
            action_open: ACTION_OPEN,
            action_close: ACTION_CLOSE,
            eos: EOS,
            ignore: IGNORE,
            pad: PAD,
            board_open: BOARD_OPEN,
            board_close: BOARD_CLOSE,
            visual_token_placeholder: 999,
            think_close: THINK_CLOSE,
            im_end: IM_END,
        }
    }

    fn test_tokenizer() -> Tokenizer {
        let mut vocab: ahash::AHashMap<String, u32> = ahash::AHashMap::new();
        let mut id = 0u32;

        for c in 'a'..='z' {
            vocab.insert(c.to_string(), id);
            id += 1;
        }
        for c in '0'..='9' {
            vocab.insert(c.to_string(), id);
            id += 1;
        }
        for c in ['(', ')', '\'', ',', ' '] {
            vocab.insert(c.to_string(), id);
            id += 1;
        }
        vocab.insert("P".to_string(), id);
        id += 1;
        vocab.insert("\n".to_string(), id);
        id += 1;
        vocab.insert("-".to_string(), id);
        id += 1;
        for c in 'A'..='Z' {
            if c != 'P' {
                vocab.insert(c.to_string(), id);
                id += 1;
            }
        }

        vocab.insert("up".to_string(), id);
        id += 1;
        vocab.insert("do".to_string(), id);
        id += 1;
        vocab.insert("wn".to_string(), id);
        id += 1;
        vocab.insert("ne".to_string(), id);
        id += 1;
        vocab.insert("down".to_string(), id);
        id += 1;
        vocab.insert("done".to_string(), id);
        let _ = id + 1;

        let merges = vec![
            ("u".into(), "p".into()),
            ("d".into(), "o".into()),
            ("w".into(), "n".into()),
            ("n".into(), "e".into()),
            ("do".into(), "wn".into()),
            ("do".into(), "ne".into()),
        ];

        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .build()
            .unwrap();

        Tokenizer::new(bpe)
    }

    fn make_stream_with_seed(ascii: &str, seed: u64) -> EpisodeStream {
        let state = from_ascii(ascii);
        let game = Arc::new(Sokoban::default());
        let episode = GameEpisode::new(game, state, 100_000);
        let tokenizer = test_tokenizer();
        let rng = ChaChaRng::seed_from_u64(seed);
        EpisodeStream::new(
            Box::new(episode),
            Arc::new(tokenizer),
            test_special(),
            rng,
            None,
            0,
            1.5,
        )
    }

    /// 1-step puzzle: push right to solve, then "done"
    const PUZZLE_1STEP: &str = "#####\n#@OX#\n#####";
    /// 2-step puzzle: move right, push right to solve, then "done"
    const PUZZLE_2STEP: &str = "######\n#@.OX#\n######";

    // =========================================================================
    // visual-token integration tests
    // =========================================================================

    use crate::stream::episode_stream::VisualTokenConfig;

    fn make_stream_visual_with_seed(ascii: &str, seed: u64) -> EpisodeStream {
        let state = from_ascii(ascii);
        let game = Arc::new(Sokoban::default());
        let episode = GameEpisode::new(game, state, 100_000);
        let tokenizer = test_tokenizer();
        let rng = ChaChaRng::seed_from_u64(seed);
        let visual_token_config = Some(VisualTokenConfig {
            short_edge_res: 64,
            token_short_edge: None,
        });
        EpisodeStream::new(
            Box::new(episode),
            Arc::new(tokenizer),
            test_special(),
            rng,
            visual_token_config,
            0,
            1.5,
        )
    }

    use crate::base::game::Game;

    fn make_generated_stream(seed: u64) -> EpisodeStream {
        let game = Arc::new(Sokoban {
            grid_size_range: (6, 6),
            num_boxes_range: (1, 1),
            solve_budget: 100_000,
            wall_density: 20,
        });
        let mut rng = ChaChaRng::seed_from_u64(seed);
        let state = game.generate(&mut rng).expect("test seed should generate");
        let episode = GameEpisode::new(game, state, 100_000);
        let tokenizer = test_tokenizer();
        let rng2 = ChaChaRng::seed_from_u64(seed.wrapping_add(1));
        EpisodeStream::new(
            Box::new(episode),
            Arc::new(tokenizer),
            test_special(),
            rng2,
            None,
            0,
            1.5,
        )
    }

    // =========================================================================
    // teacher_force_labels tests
    // =========================================================================

    #[test]
    fn teacher_force_staggered_done() {
        let seed = 42u64;
        let max_tokens = 200;

        let mut rollout = Rollout::new(vec![
            make_stream_with_seed(PUZZLE_1STEP, seed),
            make_stream_with_seed(PUZZLE_2STEP, seed),
        ]);

        let result = rollout.teacher_force_labels(max_tokens);

        // 1-step puzzle should be shorter than 2-step puzzle
        assert!(
            result.lengths[0] < result.lengths[1],
            "1-step puzzle should finish before 2-step: {} vs {}",
            result.lengths[0],
            result.lengths[1]
        );

        // Both should have real tokens (not all PAD)
        assert!(result.lengths[0] > 0);
        assert!(result.lengths[1] > 0);

        // Sequences should only contain non-PAD tokens up to their length
        assert_eq!(result.sequences[0].len(), result.lengths[0]);
        assert_eq!(result.sequences[1].len(), result.lengths[1]);
    }

    #[test]
    fn teacher_force_determinism() {
        let seed = 42u64;
        let max_tokens = 200;

        let mut rollout1 = Rollout::new(vec![
            make_stream_with_seed(PUZZLE_1STEP, seed),
            make_stream_with_seed(PUZZLE_2STEP, seed),
        ]);
        let result1 = rollout1.teacher_force_labels(max_tokens);

        let mut rollout2 = Rollout::new(vec![
            make_stream_with_seed(PUZZLE_1STEP, seed),
            make_stream_with_seed(PUZZLE_2STEP, seed),
        ]);
        let result2 = rollout2.teacher_force_labels(max_tokens);

        assert_eq!(result1.sequences, result2.sequences);
        assert_eq!(result1.lengths, result2.lengths);
    }

    #[test]
    fn teacher_force_visual_tokens_uses_placeholder() {
        const VISUAL_TOKEN_PLACEHOLDER: TokenId = 999;

        let mut rollout_tf = Rollout::new(vec![
            make_stream_visual_with_seed(PUZZLE_1STEP, 42),
            make_stream_visual_with_seed(PUZZLE_1STEP, 43),
        ]);
        // Push initial images
        for stream in rollout_tf.streams.iter_mut() {
            let img = stream.episode.render(128);
            stream.accumulated_frames.push(img);
        }
        let tf_result = rollout_tf.teacher_force_labels(200);
        let batch_size = tf_result.sequences.len();

        // Verify sentinel tokens appear (visual_token_placeholder = 999)
        for i in 0..batch_size {
            let has_sentinel = tf_result.sequences[i]
                .iter()
                .any(|&t| t == VISUAL_TOKEN_PLACEHOLDER);
            assert!(
                has_sentinel,
                "episode {i}: expected sentinel visual tokens in teacher_force output"
            );
        }

        // Verify frames were accumulated
        for (i, stream) in rollout_tf.streams.iter().enumerate() {
            assert!(
                !stream.accumulated_frames.is_empty(),
                "stream {i}: expected accumulated frames after teacher forcing"
            );
        }
    }

    #[test]
    fn teacher_force_all_done() {
        let mut rollout = Rollout::new(vec![
            make_stream_with_seed(PUZZLE_1STEP, 42),
            make_stream_with_seed(PUZZLE_2STEP, 43),
        ]);

        assert!(!rollout.is_all_done());
        rollout.teacher_force_labels(200);
        assert!(rollout.is_all_done());
    }

    /// Symmetric board where both "left" and "right" are equally optimal.
    const PUZZLE_SYMMETRIC: &str = "#######\n#XO@OX#\n#######";

    #[test]
    fn teacher_force_symmetric_board() {
        // Both "left" and "right" are equally optimal on this board.
        // teacher_force_labels should produce a valid optimal sequence
        // (following whichever solve_optimal picks).
        let mut rollout = Rollout::new(vec![
            make_stream_with_seed(PUZZLE_SYMMETRIC, 42),
            make_stream_with_seed(PUZZLE_SYMMETRIC, 43),
        ]);
        let result = rollout.teacher_force_labels(200);
        assert!(rollout.is_all_done());
        for sequence in &result.sequences {
            assert!(sequence.contains(&ACTION_OPEN));
            assert!(sequence.contains(&ACTION_CLOSE));
            assert!(sequence.contains(&IM_END));
        }
    }

    #[test]
    fn teacher_force_truncation() {
        let mut rollout = Rollout::new(vec![
            make_stream_with_seed(PUZZLE_2STEP, 42),
            make_stream_with_seed(PUZZLE_2STEP, 43),
        ]);

        let max_tokens = 5;
        let result = rollout.teacher_force_labels(max_tokens);

        // Should NOT be done — 5 tokens is not enough for a 2-step puzzle
        assert!(!rollout.is_all_done());

        for i in 0..2 {
            assert!(result.lengths[i] <= max_tokens);
            assert_eq!(result.sequences[i].len(), result.lengths[i]);
            assert!(result.lengths[i] > 0, "should produce at least some tokens");
        }
    }

    fn make_tangram_stream(seed: u64) -> EpisodeStream {
        use crate::games::tangram::Tangram;

        let game = Arc::new(Tangram {
            num_pieces_range: (2, 2),
            canvas_size_range: (4, 4),
            piece_size_range: (2, 4),
            solve_budget: 1_000_000,
            certificate_game: false,
        });
        let mut rng = ChaChaRng::seed_from_u64(seed);
        let state = game.generate(&mut rng).expect("test seed should generate");
        let episode = GameEpisode::new(game, state, 100_000);
        let tokenizer = test_tokenizer();
        let rng2 = ChaChaRng::seed_from_u64(seed.wrapping_add(1));
        EpisodeStream::new(
            Box::new(episode),
            Arc::new(tokenizer),
            test_special(),
            rng2,
            None,
            0,
            1.5,
        )
    }

    fn make_bloxorz_stream(seed: u64) -> EpisodeStream {
        use crate::games::bloxorz::Bloxorz;

        let game = Arc::new(Bloxorz {
            grid_size_range: (6, 6),
            solve_budget: 100_000,
            fill_range: (50, 65),
            num_fragile_tiles_range: (0, 0),
        });
        let mut rng = ChaChaRng::seed_from_u64(seed);
        let state = game.generate(&mut rng).expect("test seed should generate");
        let episode = GameEpisode::new(game, state, 100_000);
        let tokenizer = test_tokenizer();
        let rng2 = ChaChaRng::seed_from_u64(seed.wrapping_add(1));
        EpisodeStream::new(
            Box::new(episode),
            Arc::new(tokenizer),
            test_special(),
            rng2,
            None,
            0,
            1.5,
        )
    }

    fn make_rush_hour_stream(seed: u64) -> EpisodeStream {
        use crate::games::rush_hour::RushHour;

        let game = Arc::new(RushHour {
            grid_size_range: (6, 6),
            num_cars_range: (6, 12),
            solve_budget: 100_000,
        });
        let mut rng = ChaChaRng::seed_from_u64(seed);
        let state = game.generate(&mut rng).expect("test seed should generate");
        let episode = GameEpisode::new(game, state, 100_000);
        let tokenizer = test_tokenizer();
        let rng2 = ChaChaRng::seed_from_u64(seed.wrapping_add(1));
        EpisodeStream::new(
            Box::new(episode),
            Arc::new(tokenizer),
            test_special(),
            rng2,
            None,
            0,
            1.5,
        )
    }

    fn make_mental_rotation_stream(seed: u64) -> EpisodeStream {
        use crate::games::mental_rotation::MentalRotation;

        let game = Arc::new(MentalRotation {
            num_cubes_range: (9, 9),
            increment: 15,
            max_step: 30,
            max_delta: 180,
            solve_budget: 100_000,
            zoom_pct: 100,
            certificate_game: false,
            min_solve_steps: 5,
        });
        let mut rng = ChaChaRng::seed_from_u64(seed);
        let state = game.generate(&mut rng).expect("test seed should generate");
        let episode = GameEpisode::new(game, state, 100_000);
        let tokenizer = test_tokenizer();
        let rng2 = ChaChaRng::seed_from_u64(seed.wrapping_add(1));
        EpisodeStream::new(
            Box::new(episode),
            Arc::new(tokenizer),
            test_special(),
            rng2,
            None,
            0,
            1.5,
        )
    }

    #[test]
    fn teacher_force_tangram() {
        let max_tokens = 500;
        let mut rollout = Rollout::new(vec![make_tangram_stream(100), make_tangram_stream(200)]);

        let result = rollout.teacher_force_labels(max_tokens);

        assert!(rollout.is_all_done());

        for i in 0..2 {
            assert!(result.lengths[i] > 0);
            assert!(result.sequences[i].contains(&ACTION_OPEN));
            assert!(result.sequences[i].contains(&ACTION_CLOSE));
            assert!(result.sequences[i].contains(&IM_END));
        }
    }

    /// Replay the teacher-forced tokens through `step()`, as evaluation replays a
    /// model's tokens: every episode must end exactly at its last token, solved
    /// and within the action limit.
    fn verify_teacher_force_replay<F: Fn(u64) -> EpisodeStream>(
        seeds: &[u64],
        max_tokens: usize,
        make_stream: F,
    ) {
        let mut rollout_tf = Rollout::new(seeds.iter().map(|&s| make_stream(s)).collect());
        let tf_result = rollout_tf.teacher_force_labels(max_tokens);
        assert!(
            rollout_tf.is_all_done(),
            "teacher_force_labels did not complete all episodes within {max_tokens} tokens"
        );

        let mut rollout_replay = Rollout::new(seeds.iter().map(|&s| make_stream(s)).collect());
        let longest = tf_result.lengths.iter().copied().max().unwrap();
        for position in 0..longest {
            let model_tokens: Vec<TokenId> = tf_result
                .sequences
                .iter()
                .map(|sequence| sequence.get(position).copied().unwrap_or(PAD))
                .collect();
            let done = rollout_replay.step(&model_tokens);
            for (i, &episode_done) in done.iter().enumerate() {
                assert_eq!(
                    episode_done,
                    position + 1 >= tf_result.lengths[i],
                    "episode {i} (seed {}): done at position {position}",
                    seeds[i]
                );
            }
        }
        for (i, stream) in rollout_replay.streams.iter().enumerate() {
            assert!(
                stream.episode.is_solved(),
                "episode {i} (seed {}): not solved",
                seeds[i]
            );
            assert!(
                !stream.timed_out,
                "episode {i} (seed {}): timed out",
                seeds[i]
            );
        }
    }

    #[test]
    fn teacher_force_replay_sokoban() {
        verify_teacher_force_replay(&[100, 200, 300], 500, make_generated_stream);
    }

    #[test]
    fn teacher_force_replay_tangram() {
        verify_teacher_force_replay(&[100, 200], 500, make_tangram_stream);
    }

    #[test]
    fn teacher_force_replay_bloxorz() {
        verify_teacher_force_replay(&[100, 200, 300], 500, make_bloxorz_stream);
    }

    #[test]
    fn teacher_force_replay_rush_hour() {
        verify_teacher_force_replay(&[100, 200, 300], 500, make_rush_hour_stream);
    }

    #[test]
    fn teacher_force_replay_mental_rotation() {
        verify_teacher_force_replay(&[100, 200, 300], 500, make_mental_rotation_stream);
    }

    // =========================================================================
    // apply_random_truncation tests
    // =========================================================================

    #[test]
    fn random_truncation_short_episode_unchanged() {
        // Episode that fits within max_tokens should not be modified
        let mut rollout = Rollout::new(vec![
            make_stream_visual_with_seed(PUZZLE_1STEP, 42),
            make_stream_visual_with_seed(PUZZLE_1STEP, 43),
        ]);
        // Push initial images
        for stream in rollout.streams.iter_mut() {
            let img = stream.episode.render(128);
            stream.accumulated_frames.push(img);
        }

        let mut result = rollout.teacher_force_labels(1000);
        let original_lengths: Vec<usize> = result.lengths.clone();
        let original_input_seqs: Vec<Vec<TokenId>> = result.sequences.clone();
        let original_label_seqs: Vec<Vec<TokenId>> = result.sequences.clone();

        let (_initial_frames, _stats) = rollout.apply_random_truncation(&mut result, 1000);

        // Sequences should be unchanged (they fit)
        assert_eq!(result.lengths, original_lengths);
        assert_eq!(result.sequences, original_input_seqs);
        assert_eq!(result.sequences, original_label_seqs);
    }

    #[test]
    fn random_truncation_long_episode_truncated() {
        // Use a 2-step puzzle with visual tokens -- each step has board+action tokens
        // Set max_tokens very small to force truncation
        let mut rollout = Rollout::new(vec![make_stream_visual_with_seed(PUZZLE_2STEP, 42)]);
        for stream in rollout.streams.iter_mut() {
            let img = stream.episode.render(128);
            stream.accumulated_frames.push(img);
        }

        let mut result = rollout.teacher_force_labels(1000);
        let full_length = result.lengths[0];

        // Truncate to a small max -- must be shorter
        let max_tokens = 10;
        if full_length > max_tokens {
            let (_initial_frames, _stats) =
                rollout.apply_random_truncation(&mut result, max_tokens);
            assert!(
                result.lengths[0] <= max_tokens,
                "truncated length {} exceeds max_tokens {}",
                result.lengths[0],
                max_tokens
            );
            assert_eq!(result.sequences[0].len(), result.lengths[0]);
            assert_eq!(result.sequences[0].len(), result.lengths[0]);
        }
    }

    #[test]
    fn random_truncation_starts_at_board_token() {
        // The truncated sequence should start with a <board> token
        let mut rollout = Rollout::new(vec![make_stream_visual_with_seed(PUZZLE_2STEP, 42)]);
        for stream in rollout.streams.iter_mut() {
            let img = stream.episode.render(128);
            stream.accumulated_frames.push(img);
        }

        let mut result = rollout.teacher_force_labels(1000);
        let full_length = result.lengths[0];
        let max_tokens = 10;

        if full_length > max_tokens {
            let (_initial_frames, _stats) =
                rollout.apply_random_truncation(&mut result, max_tokens);
            assert_eq!(
                result.sequences[0][0], BOARD_OPEN,
                "truncated label sequence should start with <board>, got {}",
                result.sequences[0][0]
            );
        }
    }

    #[test]
    fn random_truncation_frames_match_board_tokens() {
        // Number of accumulated frames should match board tokens in truncated sequence
        let mut rollout = Rollout::new(vec![make_stream_visual_with_seed(PUZZLE_2STEP, 42)]);
        for stream in rollout.streams.iter_mut() {
            let img = stream.episode.render(128);
            stream.accumulated_frames.push(img);
        }

        let mut result = rollout.teacher_force_labels(1000);
        let full_length = result.lengths[0];
        let max_tokens = 10;

        if full_length > max_tokens {
            let (_initial_frames, _stats) =
                rollout.apply_random_truncation(&mut result, max_tokens);

            let board_count = result.sequences[0]
                .iter()
                .filter(|&&t| t == BOARD_OPEN)
                .count();
            let frame_count = rollout.streams[0].accumulated_frames.len();
            assert_eq!(
                frame_count, board_count,
                "frame count ({}) != board token count ({})",
                frame_count, board_count
            );
        }
    }

    #[test]
    fn random_truncation_returns_initial_frame() {
        // apply_random_truncation should return one initial frame per episode
        let mut rollout = Rollout::new(vec![
            make_stream_visual_with_seed(PUZZLE_2STEP, 42),
            make_stream_visual_with_seed(PUZZLE_2STEP, 43),
        ]);
        for stream in rollout.streams.iter_mut() {
            let img = stream.episode.render(128);
            stream.accumulated_frames.push(img);
        }

        let mut result = rollout.teacher_force_labels(1000);
        let initial_frames = rollout.apply_random_truncation(&mut result, 10);

        assert_eq!(
            initial_frames.0.len(),
            2,
            "should return one initial frame per episode"
        );
        // Each frame should be a valid RGB buffer (non-empty)
        for (i, frame) in initial_frames.0.iter().enumerate() {
            assert!(
                !frame.data.is_empty(),
                "initial frame {i} should not be empty"
            );
        }
    }

    #[test]
    fn random_truncation_deterministic_with_same_seed() {
        // Same seed should produce the same truncation point
        let max_tokens = 10;

        let run = |seed: u64| -> (Vec<TokenId>, Vec<TokenId>) {
            let mut rollout = Rollout::new(vec![make_stream_visual_with_seed(PUZZLE_2STEP, seed)]);
            for stream in rollout.streams.iter_mut() {
                let img = stream.episode.render(128);
                stream.accumulated_frames.push(img);
            }
            let mut result = rollout.teacher_force_labels(1000);
            let _ = rollout.apply_random_truncation(&mut result, max_tokens);
            (result.sequences[0].clone(), result.sequences[0].clone())
        };

        let (inp1, lbl1) = run(42);
        let (inp2, lbl2) = run(42);
        assert_eq!(
            inp1, inp2,
            "same seed should produce identical truncation (inputs)"
        );
        assert_eq!(
            lbl1, lbl2,
            "same seed should produce identical truncation (labels)"
        );
    }
}
