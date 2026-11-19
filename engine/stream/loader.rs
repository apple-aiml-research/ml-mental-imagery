//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// Episode loader: game assignment, seeding, and background prefetch threads.

use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use rayon::prelude::*;
use tokenizers::Tokenizer;

use crate::base::episode::{Episode, GameEpisode};
use crate::base::game::{Game, RgbBuffer};
use crate::stream::batch::Rollout;
use crate::stream::episode_stream::{
    EpisodeStream, SpecialTokenIds, TextBoards, VisualTokenConfig,
};

// ---------------------------------------------------------------------------
// GameFactory trait + GameConfig wrapper
// ---------------------------------------------------------------------------

/// Type-erased factory for creating episodes from games.
///
/// Bridges the gap between the generic `Game` trait (with associated types)
/// and the type-erased `Episode` trait. Enables mixed-game batches.
pub trait GameFactory: Send + Sync {
    fn create_episode(&self, rng: &mut ChaChaRng) -> Box<dyn Episode>;

    /// `Game::puzzle_fingerprint` of the puzzle `create_episode` would build from `rng`,
    /// without solving or rendering it.
    fn puzzle_fingerprint(&self, rng: &mut ChaChaRng) -> u64;

    /// Never generate the puzzles of these seeds (e.g. the evaluation puzzles, for a
    /// training loader): a generated puzzle with the same fingerprint as one of them is
    /// replaced by the next puzzle drawn from the same RNG.
    fn exclude_puzzles_of_seeds(&mut self, seeds: &[u64]);
}

/// Consecutive excluded puzzles after which generation fails loudly.
const MAX_EXCLUDED_DRAWS: usize = 1000;

struct GameConfig<G: Game> {
    game: Arc<G>,
    name: String,
    solve_budget: usize,
    /// Fingerprints of puzzles never to generate (see `exclude_puzzles_of_seeds`).
    excluded: HashSet<u64>,
}

impl<G: Game> GameConfig<G> {
    fn generate_or_trivial(&self, rng: &mut ChaChaRng) -> G::State {
        self.game.generate(rng).unwrap_or_else(|| {
            eprintln!("\x1b[31m[engine] WARNING: generation exhausted for '{}', using trivial fallback\x1b[0m", self.name);
            self.game.trivial()
        })
    }

    fn initial_state(&self, rng: &mut ChaChaRng) -> G::State {
        for _ in 0..MAX_EXCLUDED_DRAWS {
            let state = self.generate_or_trivial(rng);
            if self.excluded.is_empty()
                || !self
                    .excluded
                    .contains(&self.game.puzzle_fingerprint(&state))
            {
                return state;
            }
        }
        panic!(
            "'{}': {MAX_EXCLUDED_DRAWS} consecutive generated puzzles were excluded",
            self.name
        );
    }
}

impl<G: Game> GameFactory for GameConfig<G> {
    fn create_episode(&self, rng: &mut ChaChaRng) -> Box<dyn Episode> {
        let state = self.initial_state(rng);
        Box::new(GameEpisode::new(
            self.game.clone(),
            state,
            self.solve_budget,
        ))
    }

    fn puzzle_fingerprint(&self, rng: &mut ChaChaRng) -> u64 {
        let state = self.initial_state(rng);
        self.game.puzzle_fingerprint(&state)
    }

    fn exclude_puzzles_of_seeds(&mut self, seeds: &[u64]) {
        let fingerprints: Vec<u64> = seeds
            .par_iter()
            .map(|&seed| {
                self.game.puzzle_fingerprint(
                    &self.generate_or_trivial(&mut ChaChaRng::seed_from_u64(seed)),
                )
            })
            .collect();
        self.excluded.extend(fingerprints);
    }
}

/// Wrap any `Game` into a boxed `GameFactory`.
pub fn game_factory<G: Game>(game: G, name: &str, solve_budget: usize) -> Box<dyn GameFactory> {
    Box::new(GameConfig {
        game: Arc::new(game),
        name: name.to_string(),
        solve_budget,
        excluded: HashSet::new(),
    })
}

// ---------------------------------------------------------------------------
// PuzzleEngine — central config
// ---------------------------------------------------------------------------

/// Central configuration of the puzzle engine: tokenizer, special tokens, seeding
/// and episode layout. Shared across threads via `Arc`.
pub struct PuzzleEngine {
    pub tokenizer: Arc<Tokenizer>,
    pub seed: u64,
    pub rank: usize,
    pub special: SpecialTokenIds,
    pub visual_token_config: Option<VisualTokenConfig>,
    /// Text boards (State CoT or Boxes CoT; see `EpisodeStream::with_text_boards`).
    pub text_boards: Option<TextBoards>,
    pub render_resolution: u32,
    pub action_limit_multiplier: f64,
    pub parallel: bool,
}

impl PuzzleEngine {
    pub fn new(
        tokenizer: Tokenizer,
        seed: u64,
        rank: usize,
        special: SpecialTokenIds,
        visual_token_config: Option<VisualTokenConfig>,
        render_resolution: u32,
        action_limit_multiplier: f64,
        parallel: bool,
    ) -> Self {
        PuzzleEngine {
            tokenizer: Arc::new(tokenizer),
            seed,
            rank,
            special,
            visual_token_config,
            text_boards: None,
            render_resolution,
            action_limit_multiplier,
            parallel,
        }
    }

    /// Write every board as text: JSON state descriptions (State CoT) or object boxes (Boxes CoT).
    pub fn with_text_boards(mut self, text_boards: TextBoards) -> Self {
        assert!(
            self.visual_token_config.is_none(),
            "text boards cannot be combined with visual-token boards"
        );
        self.text_boards = Some(text_boards);
        self
    }

    /// Deterministic seed for a specific episode within a rollout.
    ///
    /// Seeds are computed as: seed + (rank << 40) + (rollout_index << 16) + episode_index
    ///
    /// This ensures:
    /// - Different ranks get non-overlapping seed ranges (separated by 2^40)
    /// - Sequential rollouts get sequential seeds
    /// - Eval (seed=0) uses seeds [0, N), training (seed=10000+) uses [10000+, ∞)
    pub fn episode_seed(&self, rollout_index: u64, episode_index: u64) -> u64 {
        self.seed
            .wrapping_add((self.rank as u64) << 40)
            .wrapping_add(rollout_index << 16)
            .wrapping_add(episode_index)
    }

    /// Convenience factory for creating a `Loader`.
    pub fn loader(
        self: &Arc<Self>,
        games: Vec<Box<dyn GameFactory>>,
        batch_size: usize,
        prefetch: usize,
    ) -> Loader {
        Loader::new(self.clone(), games, batch_size, prefetch)
    }
}

// ---------------------------------------------------------------------------
// PreparedRollout — rollout + prompt data
// ---------------------------------------------------------------------------

/// A rollout with the prompt data for the Python layer.
pub struct PreparedRollout {
    pub rollout: Rollout,
    pub initial_images: Vec<RgbBuffer>,
    pub prompts: Vec<String>,
    pub actions: Vec<String>,
}

// ---------------------------------------------------------------------------
// Loader — background prefetch queue
// ---------------------------------------------------------------------------

/// Produces rollouts on demand, with optional background prefetch.
///
/// Every game gets an equal share of each batch. Uses `VecDeque<JoinHandle>`
/// for FIFO ordering: each slot is a thread building one rollout. `next()`
/// pops the front handle and `join()`s it, then spawns a replacement at the back.
pub struct Loader {
    engine: Arc<PuzzleEngine>,
    games: Arc<Vec<Box<dyn GameFactory>>>,
    batch_size: usize,
    prefetch: usize,
    rollout_counter: u64,
    queue: VecDeque<JoinHandle<PreparedRollout>>,
}

impl Loader {
    pub fn new(
        engine: Arc<PuzzleEngine>,
        games: Vec<Box<dyn GameFactory>>,
        batch_size: usize,
        prefetch: usize,
    ) -> Self {
        let games = Arc::new(games);
        let mut queue = VecDeque::with_capacity(prefetch);

        for i in 0..prefetch {
            let handle = Self::spawn_build(engine.clone(), games.clone(), batch_size, i as u64);
            queue.push_back(handle);
        }

        Loader {
            engine,
            games,
            batch_size,
            prefetch,
            rollout_counter: prefetch as u64,
            queue,
        }
    }

    /// Get the next prepared rollout.
    ///
    /// If prefetch > 0: pops the front handle, joins it, spawns a replacement.
    /// If prefetch == 0: builds synchronously.
    pub fn next(&mut self) -> PreparedRollout {
        let mut prepared = if self.prefetch == 0 {
            let idx = self.rollout_counter;
            self.rollout_counter += 1;
            Self::build_rollout(&self.engine, &self.games, self.batch_size, idx)
        } else {
            let handle = self.queue.pop_front().expect("prefetch queue empty");
            let result = match handle.join() {
                Ok(r) => r,
                Err(_) => {
                    eprintln!("\x1b[31m[engine] WARNING: build_rollout thread panicked, retrying synchronously\x1b[0m");
                    Self::build_rollout(
                        &self.engine,
                        &self.games,
                        self.batch_size,
                        self.rollout_counter - 1,
                    )
                }
            };

            // Spawn replacement
            let replacement = Self::spawn_build(
                self.engine.clone(),
                self.games.clone(),
                self.batch_size,
                self.rollout_counter,
            );
            self.queue.push_back(replacement);
            self.rollout_counter += 1;

            result
        };

        // Push initial images into accumulated_frames for deferred visual-token encoding
        // or visual head supervision (any mode with rendering).
        if prepared.rollout.streams[0].short_edge_res > 0 {
            for (stream, image) in prepared
                .rollout
                .streams
                .iter_mut()
                .zip(prepared.initial_images.iter())
            {
                stream.accumulated_frames.push(image.clone());
            }
        }

        prepared
    }

    /// Number of games the loader mixes.
    pub fn num_games(&self) -> usize {
        self.games.len()
    }

    /// Get a rollout with explicit seeds for each episode.
    ///
    /// Used for evaluation, so that puzzles do not depend on the batch size.
    /// All episodes use game index 0 (assumes single-game loader).
    pub fn next_with_seeds(&self, seeds: &[u64]) -> PreparedRollout {
        let mut prepared = Self::build_rollout_with_seeds(&self.engine, &self.games, seeds);

        // Push initial images into accumulated_frames for deferred visual-token encoding
        if !prepared.rollout.streams.is_empty() && prepared.rollout.streams[0].has_boards {
            for (stream, image) in prepared
                .rollout
                .streams
                .iter_mut()
                .zip(prepared.initial_images.iter())
            {
                stream.accumulated_frames.push(image.clone());
            }
        }

        prepared
    }

    fn spawn_build(
        engine: Arc<PuzzleEngine>,
        games: Arc<Vec<Box<dyn GameFactory>>>,
        batch_size: usize,
        rollout_index: u64,
    ) -> JoinHandle<PreparedRollout> {
        thread::spawn(move || Self::build_rollout(&engine, &games, batch_size, rollout_index))
    }

    fn build_rollout(
        engine: &PuzzleEngine,
        games: &[Box<dyn GameFactory>],
        batch_size: usize,
        rollout_index: u64,
    ) -> PreparedRollout {
        let game_assignments =
            Self::game_assignments(engine, games.len(), batch_size, rollout_index);
        let render_res = engine.render_resolution;

        // Build episodes in parallel: generate puzzle, BFS solve, render, tokenize
        let episodes: Vec<_> = if engine.parallel {
            game_assignments
                .par_iter()
                .enumerate()
                .map(|(ep_idx, &game_idx)| {
                    Self::build_one_episode(
                        engine,
                        games,
                        game_idx,
                        ep_idx,
                        rollout_index,
                        render_res,
                    )
                })
                .collect()
        } else {
            game_assignments
                .iter()
                .enumerate()
                .map(|(ep_idx, &game_idx)| {
                    Self::build_one_episode(
                        engine,
                        games,
                        game_idx,
                        ep_idx,
                        rollout_index,
                        render_res,
                    )
                })
                .collect()
        };

        Self::collect_rollout(episodes)
    }

    /// The game of each episode of batch `rollout_index`: an equal share per game, in shuffled positions.
    fn game_assignments(
        engine: &PuzzleEngine,
        num_games: usize,
        batch_size: usize,
        rollout_index: u64,
    ) -> Vec<usize> {
        // Rollout-level RNG for game sampling
        let mut sampling_rng = ChaChaRng::seed_from_u64(engine.episode_seed(rollout_index, 0));

        let base = batch_size / num_games;
        let remainder = batch_size % num_games;

        // Shuffle game indices to randomize which games get the
        // `remainder` extra slots. Without this, the last games in
        // the list are systematically starved when batch_size < num_games.
        let mut game_order: Vec<usize> = (0..num_games).collect();
        for i in (1..num_games).rev() {
            let j = sampling_rng.gen_range(0..=i);
            game_order.swap(i, j);
        }

        // First `remainder` games in shuffled order get base+1, rest get base
        let mut game_assignments: Vec<usize> = Vec::with_capacity(batch_size);
        for (pos, &g) in game_order.iter().enumerate() {
            let count = base + if pos < remainder { 1 } else { 0 };
            for _ in 0..count {
                game_assignments.push(g);
            }
        }

        // Shuffle positions within the batch
        for i in (1..game_assignments.len()).rev() {
            let j = sampling_rng.gen_range(0..=i);
            game_assignments.swap(i, j);
        }
        game_assignments
    }

    /// Puzzle fingerprints of game `game_idx` at explicit seeds.
    pub fn puzzle_fingerprints(&self, game_idx: usize, seeds: &[u64]) -> Vec<u64> {
        seeds
            .par_iter()
            .map(|&seed| {
                self.games[game_idx].puzzle_fingerprint(&mut ChaChaRng::seed_from_u64(seed))
            })
            .collect()
    }

    /// Training episodes of batches `start..start + count` (as `next()` generates them)
    /// whose puzzle fingerprint is in `targets[game]` for the game the episode is assigned to.
    /// Returns `(game index, batch index, episode seed)` of each match.
    pub fn find_training_puzzles(
        &self,
        targets: &[HashSet<u64>],
        start: u64,
        count: u64,
    ) -> Vec<(usize, u64, u64)> {
        assert_eq!(targets.len(), self.games.len(), "one target set per game");
        let (engine, games, batch_size) = (&*self.engine, &self.games[..], self.batch_size);
        let mut matches: Vec<(usize, u64, u64)> = (start..start + count)
            .into_par_iter()
            .flat_map_iter(|rollout_index| {
                let assignments =
                    Self::game_assignments(engine, games.len(), batch_size, rollout_index);
                assignments
                    .into_iter()
                    .enumerate()
                    .filter_map(move |(ep_idx, game_idx)| {
                        if targets[game_idx].is_empty() {
                            return None;
                        }
                        let seed = engine.episode_seed(rollout_index, ep_idx as u64 + 1);
                        let fingerprint =
                            games[game_idx].puzzle_fingerprint(&mut ChaChaRng::seed_from_u64(seed));
                        targets[game_idx].contains(&fingerprint).then_some((
                            game_idx,
                            rollout_index,
                            seed,
                        ))
                    })
            })
            .collect();
        matches.sort_unstable();
        matches
    }

    /// Build a rollout with explicit seeds (for eval).
    ///
    /// All episodes use game index 0. Seeds are provided explicitly
    /// to ensure reproducibility regardless of batch size or GPU count.
    fn build_rollout_with_seeds(
        engine: &PuzzleEngine,
        games: &[Box<dyn GameFactory>],
        seeds: &[u64],
    ) -> PreparedRollout {
        let render_res = engine.render_resolution;

        // Build episodes in parallel with explicit seeds
        let episodes: Vec<_> = if engine.parallel {
            seeds
                .par_iter()
                .map(|&seed| Self::build_one_episode_with_seed(engine, games, 0, seed, render_res))
                .collect()
        } else {
            seeds
                .iter()
                .map(|&seed| Self::build_one_episode_with_seed(engine, games, 0, seed, render_res))
                .collect()
        };

        Self::collect_rollout(episodes)
    }

    fn collect_rollout(
        episodes: Vec<(EpisodeStream, RgbBuffer, String, String)>,
    ) -> PreparedRollout {
        let batch_size = episodes.len();
        let mut streams = Vec::with_capacity(batch_size);
        let mut initial_images = Vec::with_capacity(batch_size);
        let mut prompts = Vec::with_capacity(batch_size);
        let mut actions = Vec::with_capacity(batch_size);

        for (stream, image, prompt, action_desc) in episodes {
            streams.push(stream);
            initial_images.push(image);
            prompts.push(prompt);
            actions.push(action_desc);
        }

        PreparedRollout {
            rollout: Rollout::new(streams),
            initial_images,
            prompts,
            actions,
        }
    }

    fn build_one_episode(
        engine: &PuzzleEngine,
        games: &[Box<dyn GameFactory>],
        game_idx: usize,
        ep_idx: usize,
        rollout_index: u64,
        render_res: u32,
    ) -> (EpisodeStream, RgbBuffer, String, String) {
        let seed = engine.episode_seed(rollout_index, ep_idx as u64 + 1);
        Self::build_one_episode_with_seed(engine, games, game_idx, seed, render_res)
    }

    fn build_one_episode_with_seed(
        engine: &PuzzleEngine,
        games: &[Box<dyn GameFactory>],
        game_idx: usize,
        seed: u64,
        render_res: u32,
    ) -> (EpisodeStream, RgbBuffer, String, String) {
        let mut ep_rng = ChaChaRng::seed_from_u64(seed);
        let episode = games[game_idx].create_episode(&mut ep_rng);

        let image = episode.render(render_res);
        let prompt = episode.prompt();
        let action_desc = episode.action_description();

        let tokenizer = Arc::clone(&engine.tokenizer);

        let mut stream = EpisodeStream::new(
            episode,
            tokenizer,
            engine.special.clone(),
            ep_rng,
            engine.visual_token_config.clone(),
            engine.render_resolution,
            engine.action_limit_multiplier,
        );
        if let Some(text_boards) = engine.text_boards {
            stream = stream.with_text_boards(text_boards);
        }

        (stream, image, prompt, action_desc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::sokoban::Sokoban;
    use crate::games::tangram::Tangram;
    use crate::stream::episode_stream::Phase;
    use crate::stream::prefix::TokenId;
    use tokenizers::models::bpe::BPE;

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

    fn test_engine(seed: u64, rank: usize) -> Arc<PuzzleEngine> {
        Arc::new(PuzzleEngine::new(
            test_tokenizer(),
            seed,
            rank,
            test_special(),
            None,
            128,
            1.5,
            true,
        ))
    }

    fn sokoban_factory() -> Box<dyn GameFactory> {
        game_factory(
            Sokoban {
                grid_size_range: (6, 6),
                num_boxes_range: (1, 1),
                solve_budget: 100_000,
                wall_density: 20,
            },
            "sokoban",
            100_000,
        )
    }

    fn tangram_factory() -> Box<dyn GameFactory> {
        game_factory(
            Tangram {
                num_pieces_range: (2, 2),
                piece_size_range: (2, 12),
                canvas_size_range: (4, 4),
                solve_budget: 100_000,
                certificate_game: false,
            },
            "tangram",
            100_000,
        )
    }

    // =========================================================================
    // Tests
    // =========================================================================

    #[test]
    fn game_factory_creates_episodes() {
        let factory = sokoban_factory();
        let mut rng = ChaChaRng::seed_from_u64(42);
        let episode = factory.create_episode(&mut rng);

        assert!(!episode.expert_action().is_empty());
        assert!(!episode.enumerate_actions().is_empty());
    }

    #[test]
    fn excluded_puzzles_are_never_generated() {
        let mut factory = tangram_factory();
        let seeds: Vec<u64> = (0..20).collect();
        let fingerprints: HashSet<u64> = seeds
            .iter()
            .map(|&seed| factory.puzzle_fingerprint(&mut ChaChaRng::seed_from_u64(seed)))
            .collect();
        factory.exclude_puzzles_of_seeds(&seeds);
        for &seed in &seeds {
            let fingerprint = factory.puzzle_fingerprint(&mut ChaChaRng::seed_from_u64(seed));
            assert!(
                !fingerprints.contains(&fingerprint),
                "seed {seed} still generates an excluded puzzle"
            );
        }
        // Other seeds are unaffected unless they happen to generate an excluded puzzle.
        let plain = tangram_factory();
        for seed in 1000..1050u64 {
            let original = plain.puzzle_fingerprint(&mut ChaChaRng::seed_from_u64(seed));
            if !fingerprints.contains(&original) {
                assert_eq!(
                    factory.puzzle_fingerprint(&mut ChaChaRng::seed_from_u64(seed)),
                    original
                );
            }
        }
    }

    #[test]
    fn find_training_puzzles_reproduces_next() {
        // Every episode next() generates is found at its batch, game and seed.
        let engine = test_engine(10_000, 3);
        let games = || vec![sokoban_factory(), tangram_factory()];
        let mut loader = Loader::new(engine.clone(), games(), 5, 0);
        let finder = Loader::new(engine.clone(), games(), 5, 0);
        let mut targets = vec![HashSet::new(), HashSet::new()];
        let mut expected = HashSet::new();
        for batch in 0..3u64 {
            let prepared = loader.next();
            for ep_idx in 0..prepared.prompts.len() {
                let game_idx = usize::from(!prepared.prompts[ep_idx].contains("Sokoban"));
                let seed = engine.episode_seed(batch, ep_idx as u64 + 1);
                targets[game_idx].insert(finder.puzzle_fingerprints(game_idx, &[seed])[0]);
                expected.insert((game_idx, batch, seed));
            }
        }
        let found: HashSet<_> = finder
            .find_training_puzzles(&targets, 0, 3)
            .into_iter()
            .collect();
        assert!(
            expected.is_subset(&found),
            "missing {:?}",
            expected.difference(&found).collect::<Vec<_>>()
        );
    }

    #[test]
    fn loader_correct_batch_size() {
        let engine = test_engine(42, 0);
        for batch_size in [1, 4, 8] {
            let mut loader = Loader::new(engine.clone(), vec![sokoban_factory()], batch_size, 0);
            let prepared = loader.next();
            assert_eq!(
                prepared.rollout.streams.len(),
                batch_size,
                "batch_size={batch_size}"
            );
        }
    }

    #[test]
    fn balanced_sampling_distributes_evenly() {
        let engine = test_engine(42, 0);

        // 2 games, batch_size=8 → 4 each
        let mut loader = Loader::new(
            engine.clone(),
            vec![sokoban_factory(), tangram_factory()],
            8,
            0,
        );
        let prepared = loader.next();
        let sokoban_count = prepared
            .prompts
            .iter()
            .filter(|p| p.contains("Sokoban"))
            .count();
        let tangram_count = prepared
            .prompts
            .iter()
            .filter(|p| p.contains("tangram"))
            .count();
        assert_eq!(sokoban_count, 4, "expected 4 sokoban, got {sokoban_count}");
        assert_eq!(tangram_count, 4, "expected 4 tangram, got {tangram_count}");

        // 2 games, batch_size=7 → 3 each + 1 random extra
        let mut loader = Loader::new(
            engine.clone(),
            vec![sokoban_factory(), tangram_factory()],
            7,
            0,
        );
        let prepared = loader.next();
        let sokoban_count = prepared
            .prompts
            .iter()
            .filter(|p| p.contains("Sokoban"))
            .count();
        let tangram_count = prepared
            .prompts
            .iter()
            .filter(|p| p.contains("tangram"))
            .count();
        assert_eq!(sokoban_count + tangram_count, 7, "total should be 7");
        assert!(
            sokoban_count >= 3,
            "sokoban should get at least base=3, got {sokoban_count}"
        );
        assert!(
            tangram_count >= 3,
            "tangram should get at least base=3, got {tangram_count}"
        );
    }

    #[test]
    fn balanced_sampling_no_game_starved() {
        // With batch_size < num_games, every game should still appear
        // approximately uniformly over many rollouts.
        let engine = test_engine(42, 0);
        let num_rollouts = 200;

        // 2 games, batch_size=1: each rollout picks exactly 1 game.
        // Over 200 rollouts, each game should appear ~100 times.
        let mut loader = Loader::new(
            engine.clone(),
            vec![sokoban_factory(), tangram_factory()],
            1,
            0,
        );

        let mut sokoban_total = 0usize;
        let mut tangram_total = 0usize;
        for _ in 0..num_rollouts {
            let prepared = loader.next();
            if prepared.prompts[0].contains("Sokoban") {
                sokoban_total += 1;
            } else {
                tangram_total += 1;
            }
        }

        // Each game should appear at least 30% of the time (expect ~50%)
        let min_expected = num_rollouts * 30 / 100;
        assert!(
            sokoban_total >= min_expected,
            "sokoban appeared {sokoban_total}/{num_rollouts} times, expected >= {min_expected}"
        );
        assert!(
            tangram_total >= min_expected,
            "tangram appeared {tangram_total}/{num_rollouts} times, expected >= {min_expected}"
        );
    }

    #[test]
    fn loader_rollout_matches_manual() {
        let engine = test_engine(42, 0);
        let game = Arc::new(Sokoban {
            grid_size_range: (6, 6),
            num_boxes_range: (1, 1),
            solve_budget: 100_000,
            wall_density: 20,
        });

        // Loader path: prefetch=0, batch_size=1, single game
        let mut loader = Loader::new(engine.clone(), vec![sokoban_factory()], 1, 0);
        let mut prepared = loader.next();

        // Manual path: replicate the same seed derivation
        // rollout_index=0, episode_index=1 (since 0 is used for sampling)
        let ep_seed = engine.episode_seed(0, 1);
        let mut ep_rng = ChaChaRng::seed_from_u64(ep_seed);
        let state = game
            .generate(&mut ep_rng)
            .expect("test seed should generate");
        let episode = GameEpisode::new(game.clone(), state, 100_000);
        let manual_stream = EpisodeStream::new(
            Box::new(episode),
            Arc::new(test_tokenizer()),
            test_special(),
            ep_rng,
            None,
            0,
            1.5,
        );
        let mut manual_rollout = Rollout::new(vec![manual_stream]);

        let loader_result = prepared.rollout.teacher_force_labels(500);
        let manual_result = manual_rollout.teacher_force_labels(500);
        assert_eq!(loader_result.sequences, manual_result.sequences);
    }

    #[test]
    fn deterministic_seeding() {
        let engine = test_engine(42, 0);

        let mut loader1 = Loader::new(engine.clone(), vec![sokoban_factory()], 4, 0);
        let mut loader2 = Loader::new(engine.clone(), vec![sokoban_factory()], 4, 0);

        let mut p1 = loader1.next();
        let mut p2 = loader2.next();

        let r1 = p1.rollout.teacher_force_labels(500);
        let r2 = p2.rollout.teacher_force_labels(500);
        assert_eq!(r1.sequences, r2.sequences);
    }

    #[test]
    fn different_ranks_diverge() {
        let engine0 = test_engine(42, 0);
        let engine1 = test_engine(42, 1);

        let mut loader0 = Loader::new(engine0, vec![sokoban_factory()], 2, 0);
        let mut loader1 = Loader::new(engine1, vec![sokoban_factory()], 2, 0);

        let mut p0 = loader0.next();
        let mut p1 = loader1.next();

        let r0 = p0.rollout.teacher_force_labels(500);
        let r1 = p1.rollout.teacher_force_labels(500);
        assert_ne!(
            r0.sequences, r1.sequences,
            "different ranks should produce different puzzles"
        );
    }

    #[test]
    fn metadata_populated() {
        let engine = test_engine(42, 0);
        let mut loader = Loader::new(engine, vec![sokoban_factory()], 4, 0);
        let prepared = loader.next();

        assert_eq!(prepared.initial_images.len(), 4);
        assert_eq!(prepared.prompts.len(), 4);
        assert_eq!(prepared.actions.len(), 4);

        for i in 0..4 {
            assert!(
                !prepared.prompts[i].is_empty(),
                "prompt {i} should not be empty"
            );
            assert!(
                !prepared.actions[i].is_empty(),
                "action {i} should not be empty"
            );
            assert!(
                !prepared.initial_images[i].data.is_empty(),
                "image {i} should not be empty"
            );
        }
    }

    #[test]
    fn prefetch_produces_valid_rollouts() {
        let engine = test_engine(42, 0);
        let mut loader = Loader::new(engine, vec![sokoban_factory()], 4, 3);

        for i in 0..6 {
            let prepared = loader.next();
            assert_eq!(
                prepared.rollout.streams.len(),
                4,
                "rollout {i} should have batch_size=4"
            );
        }
    }

    #[test]
    fn prefetch_zero_works() {
        let engine = test_engine(42, 0);
        let mut loader = Loader::new(engine, vec![sokoban_factory()], 4, 0);

        for i in 0..3 {
            let prepared = loader.next();
            assert_eq!(
                prepared.rollout.streams.len(),
                4,
                "rollout {i} should have batch_size=4"
            );
        }
    }

    // =========================================================================
    // Integration and performance
    // =========================================================================

    use rand::Rng;

    #[test]
    fn training_loop_smoke_test() {
        let engine = test_engine(42, 0);
        let mut loader = Loader::new(engine, vec![sokoban_factory(), tangram_factory()], 8, 2);

        for rollout_idx in 0..5u32 {
            let mut prepared = loader.next();

            // Verify metadata
            assert_eq!(prepared.prompts.len(), 8, "rollout {rollout_idx}: prompts");
            assert_eq!(prepared.actions.len(), 8, "rollout {rollout_idx}: actions");
            assert_eq!(
                prepared.initial_images.len(),
                8,
                "rollout {rollout_idx}: images"
            );
            for i in 0..8 {
                assert!(
                    !prepared.prompts[i].is_empty(),
                    "rollout {rollout_idx} prompt {i} empty"
                );
                assert!(
                    !prepared.actions[i].is_empty(),
                    "rollout {rollout_idx} action {i} empty"
                );
            }

            // Teacher forcing: all streams (sokoban and tangram) should complete.
            prepared.rollout.teacher_force_labels(2000);
            for (i, stream) in prepared.rollout.streams.iter().enumerate() {
                assert_eq!(
                    stream.phase,
                    Phase::Done,
                    "rollout {rollout_idx} stream {i} should be done within 2000 tokens"
                );
            }
        }
    }

    #[test]
    fn random_tokens_replay_tangram() {
        let engine = test_engine(99, 0);
        let mut loader = Loader::new(engine, vec![tangram_factory()], 4, 0);

        let mut tok_rng = ChaChaRng::seed_from_u64(12345);

        for rollout_idx in 0..3u32 {
            let mut prepared = loader.next();

            // Replay random model tokens
            let mut steps = 0u32;
            while !prepared.rollout.is_all_done() {
                let model_toks: Vec<TokenId> =
                    (0..4).map(|_| tok_rng.gen_range(0u32..50)).collect();
                prepared.rollout.step(&model_toks);
                steps += 1;
                if steps >= 4096 {
                    break;
                }
            }
            // Just verify no panic — the rollout may or may not complete with random tokens
            assert!(steps <= 4096, "rollout {rollout_idx} exceeded 4096 steps");
        }
    }
}
