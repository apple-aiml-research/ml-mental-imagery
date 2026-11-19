//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//
// EpisodeStream: per-episode token state machine.
//
// Consumes prefix-matching primitives (ActionTarget, matching_targets, next_token_of)
// to emit the expert's token sequence (teacher forcing) and to replay a model's
// tokens (evaluation). Supports optional board phases, where each board is
// written either as visual tokens (Visual CoT) or as a JSON state description
// (State CoT), and render-only mode (visual head) where frames are captured
// without board tokens.

use std::sync::Arc;

use rand_chacha::ChaChaRng;
use tokenizers::Tokenizer;

use crate::base::episode::Episode;
use crate::base::game::{compute_render_size, RgbBuffer};
use crate::stream::prefix::{
    build_valid_targets, is_complete_action, matching_targets, next_token_of, ActionTarget, TokenId,
};

/// Special token IDs used by the episode stream.
/// Keeps EpisodeStream decoupled from the PuzzleEngine struct.
#[derive(Debug, Clone)]
pub struct SpecialTokenIds {
    pub action_open: TokenId,
    pub action_close: TokenId,
    pub eos: TokenId,
    pub ignore: TokenId,
    pub pad: TokenId,
    pub board_open: TokenId,
    pub board_close: TokenId,
    /// Sentinel token for deferred visual-token positions. Python scans for this
    /// and asserts all instances are replaced with real visual tokens.
    pub visual_token_placeholder: TokenId,
    pub think_close: TokenId, // </think>
    pub im_end: TokenId,      // <|im_end|>
}

/// Configuration for board visual tokens in the episode stream.
#[derive(Debug, Clone)]
pub struct VisualTokenConfig {
    pub short_edge_res: u32,
    pub token_short_edge: Option<u32>,
}

/// What text boards contain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextBoards {
    /// The game's JSON state description (State CoT, `Episode::state_text`).
    StateDescriptions,
    /// The positions of the task-relevant objects (Boxes CoT, `Episode::object_boxes_text`).
    ObjectBoxes,
}

/// Phase of the episode stream state machine.
///
/// Board variants are active when boards are enabled (`has_boards`): visual-token boards
/// (`BoardVisual`) or text boards (`BoardText`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Phase {
    BoardOpen,          // Before visual-token encoding: labels board_open
    BoardVisual(usize), // Board token emission: labels the visual_token_placeholder sentinel
    BoardClose,         // After visual-token encoding: labels board_close
    /// Text board: emits the i-th token of the state description, then
    /// board_close. Ends when board_close is executed.
    BoardText(usize),
    ActionOpen,
    Action,
    ActionClose,
    ThinkClose,      // emits </think>
    Response(usize), // emits \n\n{terminal_action} token by token
    ImEnd,           // emits <|im_end|>
    Done,
}

/// Per-episode token state machine.
pub struct EpisodeStream {
    pub episode: Box<dyn Episode>,
    pub tokenizer: Arc<Tokenizer>,
    pub valid_targets: Vec<ActionTarget>,
    pub special: SpecialTokenIds,
    /// Whether a board block follows the initial prompt and every non-terminal action.
    pub has_boards: bool,
    /// Boards are text (State CoT or Boxes CoT) instead of visual tokens.
    pub board_text: Option<TextBoards>,
    /// Tokens of the current text board's state description.
    pub board_text_tokens: Vec<TokenId>,
    pub phase: Phase,
    pub prefix: String,
    /// Expert action being emitted, set at each ActionOpen. Cleared when the
    /// executed tokens leave it (evaluation replay of a model's tokens).
    pub target: Option<ActionTarget>,
    pub render_pending: bool,
    pub max_action_str_len: usize,
    /// Draws the random truncation start (`Rollout::apply_random_truncation`).
    pub rng: ChaChaRng,
    pub num_tokens: usize,
    pub short_edge_res: u32,
    /// RGB frames accumulated during teacher forcing for deferred visual-token encoding
    /// or visual head supervision.
    pub accumulated_frames: Vec<RgbBuffer>,
    /// Cached \n\n{action} tokens for the response after terminal actions.
    pub response_tokens: Vec<TokenId>,
    /// Number of non-terminal actions completed (game moves, back, reset, invalid).
    pub action_count: usize,
    /// Maximum actions before the episode silently ends. ceil(1.5 * optimal_len).
    /// None means no limit (unknown optimal solution length).
    pub action_limit: Option<usize>,
    /// True if the episode ended because action_count reached action_limit.
    pub timed_out: bool,
    /// Consecutive model tokens skipped while waiting for a `<board>` or `<action>` tag
    /// (evaluation replay); MAX_STRAY_TOKENS of them end the episode.
    pub stray_tokens: usize,
}

impl EpisodeStream {
    pub fn new(
        episode: Box<dyn Episode>,
        tokenizer: Arc<Tokenizer>,
        special: SpecialTokenIds,
        rng: ChaChaRng,
        visual_token_config: Option<VisualTokenConfig>,
        render_resolution: u32,
        action_limit_multiplier: f64,
    ) -> Self {
        let action_strs = episode.enumerate_actions();
        let valid_targets = build_valid_targets(&action_strs, &tokenizer);
        let max_action_str_len = valid_targets
            .iter()
            .map(|t| t.action_str.len())
            .max()
            .unwrap_or(0);

        let (has_boards, phase, num_tokens, short_edge_res) = match visual_token_config {
            Some(cfg) => {
                let (token_w, token_h) = match cfg.token_short_edge {
                    Some(tse) => compute_render_size(episode.aspect_ratio(), tse),
                    None => {
                        let (w, h) =
                            compute_render_size(episode.aspect_ratio(), cfg.short_edge_res);
                        (w / 16, h / 16)
                    }
                };
                let num_tokens = (token_w * token_h) as usize;
                (true, Phase::BoardOpen, num_tokens, cfg.short_edge_res)
            }
            None => (false, Phase::ActionOpen, 0, render_resolution),
        };

        let action_limit = if action_limit_multiplier > 0.0 {
            episode
                .optimal_len()
                .map(|optimal_len| (action_limit_multiplier * optimal_len as f64).ceil() as usize)
        } else {
            None
        };

        EpisodeStream {
            episode,
            tokenizer,
            valid_targets,
            special,
            has_boards,
            board_text: None,
            board_text_tokens: Vec::new(),
            phase,
            prefix: String::new(),
            target: None,
            render_pending: false,
            max_action_str_len,
            rng,
            num_tokens,
            short_edge_res,
            accumulated_frames: Vec::new(),
            response_tokens: Vec::new(),
            action_count: 0,
            action_limit,
            timed_out: false,
            stray_tokens: 0,
        }
    }

    /// Skip a model token that is not the tag the stream waits for (evaluation replay).
    fn skip_stray_token(&mut self) {
        self.stray_tokens += 1;
        if self.stray_tokens >= MAX_STRAY_TOKENS {
            self.phase = Phase::Done;
        }
    }

    /// Write every board as text between `<board>` and `</board>`: the game's
    /// JSON state description (State CoT) or its object boxes (Boxes CoT).
    ///
    /// Frames are still rendered at `render_resolution` after each action, so
    /// random truncation can swap the prompt image to the board it starts at.
    pub fn with_text_boards(mut self, text_boards: TextBoards) -> Self {
        assert!(
            !self.has_boards,
            "text boards cannot be combined with visual-token boards"
        );
        self.has_boards = true;
        self.board_text = Some(text_boards);
        self.phase = Phase::BoardOpen;
        self
    }

    /// Label of a text-board position: the description token, then `</board>`.
    fn board_text_label(&self, index: usize) -> TokenId {
        self.board_text_tokens
            .get(index)
            .copied()
            .unwrap_or(self.special.board_close)
    }

    /// Look up the expert's optimal action among valid_targets.
    /// Uses binary search on precomputed tokenizations; falls back to ActionTarget::new().
    pub fn solve_optimal(&self) -> ActionTarget {
        let expert = self.episode.expert_action();
        // Binary search for exact match in sorted valid_targets
        if let Ok(idx) = self
            .valid_targets
            .binary_search_by(|t| t.action_str.as_str().cmp(expert.as_str()))
        {
            return self.valid_targets[idx].clone();
        }
        // Fallback: build fresh (shouldn't happen for well-formed episodes)
        ActionTarget::new(&expert, &self.tokenizer)
    }

    /// Check if a render is pending, and if so, render the current board
    /// at short_edge_res and clear the flag.
    pub fn take_pending_render(&mut self) -> Option<RgbBuffer> {
        if self.render_pending {
            self.render_pending = false;
            let img = self.episode.render(self.short_edge_res);
            Some(img)
        } else {
            None
        }
    }

    /// Drain accumulated RGB frames from this stream.
    ///
    /// Called after teacher forcing so Python can batch-encode them.
    pub fn take_accumulated_frames(&mut self) -> Vec<RgbBuffer> {
        std::mem::take(&mut self.accumulated_frames)
    }

    /// The expert's next token (teacher forcing).
    ///
    /// The executed token always equals this label, so `self.target` always
    /// matches `self.prefix`: just emit the next token of the cached optimal action.
    pub fn expert_label(&self) -> TokenId {
        match &self.phase {
            Phase::ActionOpen => self.special.action_open,
            Phase::ActionClose => self.special.action_close,
            Phase::ThinkClose => self.special.think_close,
            Phase::Response(i) => self.response_tokens[*i],
            Phase::ImEnd => self.special.im_end,
            Phase::Done => self.special.ignore,
            Phase::BoardOpen => self.special.board_open,
            Phase::BoardVisual(_) => self.special.visual_token_placeholder,
            Phase::BoardClose => self.special.board_close,
            Phase::BoardText(i) => self.board_text_label(*i),
            Phase::Action => {
                debug_assert!(
                    self.target.is_some(),
                    "expert_label: Action phase without target"
                );
                let tgt = self.target.as_ref().unwrap();
                next_token_of(tgt, self.prefix.len(), &self.tokenizer)
            }
        }
    }

    /// Advance the state machine after a token is executed.
    pub fn advance(&mut self, executed_token: TokenId) {
        // Global EOS override
        if executed_token == self.special.eos || executed_token == self.special.im_end {
            self.phase = Phase::Done;
            return;
        }

        match self.phase {
            Phase::ActionOpen => {
                // An action starts at <action>; anything written before it is skipped.
                if executed_token != self.special.action_open {
                    self.skip_stray_token();
                    return;
                }
                self.stray_tokens = 0;
                self.prefix.clear();
                self.target = Some(self.solve_optimal());
                self.phase = Phase::Action;
            }
            Phase::Action => {
                // Special tokens have structural meaning — don't decode as text.
                // EOS is already handled by the global override above.
                if executed_token == self.special.action_close {
                    self.phase = Phase::ActionClose;
                    return;
                }
                if executed_token == self.special.action_open
                    || executed_token == self.special.pad
                    || executed_token == self.special.ignore
                {
                    return;
                }

                let decoded = self
                    .tokenizer
                    .decode(&[executed_token], false)
                    .unwrap_or_default();

                self.prefix.push_str(&decoded);

                // While the prefix follows the expert target (always, in teacher
                // forcing), only completion is checked. Once it diverges, the
                // action ends as soon as no valid action matches the prefix.
                let target_matches = self
                    .target
                    .as_ref()
                    .map_or(false, |t| t.action_str.starts_with(&self.prefix));
                if !target_matches {
                    self.target = None;
                    if matching_targets(&self.prefix, &self.valid_targets).is_empty() {
                        self.phase = Phase::ActionClose;
                        return;
                    }
                }
                if is_complete_action(&self.prefix, &self.valid_targets) {
                    self.phase = Phase::ActionClose;
                    return;
                }

                // Safety net: force ActionClose if prefix exceeds max length
                if self.prefix.len() >= self.max_action_str_len {
                    self.phase = Phase::ActionClose;
                }
            }
            Phase::ActionClose => {
                // short_edge_res > 0 means frames are captured (visual tokens or visual
                // head); has_boards decides whether a board phase follows the action.
                let should_render = self.short_edge_res > 0;
                if self.prefix == "done" || self.prefix == "yes" || self.prefix == "no" {
                    // Apply terminal action so certificate_affirmed is set
                    // in certificate-mode games.
                    self.episode.apply(&self.prefix);
                    // Visual head needs the final-state render for </action>
                    // alignment. Visual-token mode doesn't render here (the board was already
                    // shown before this action).
                    self.render_pending = !self.has_boards && should_render;
                    let response = format!("\n\n{}", self.prefix);
                    let enc = self.tokenizer.encode(response, false).unwrap();
                    self.response_tokens = enc.get_ids().to_vec();
                    self.phase = Phase::ThinkClose;
                } else {
                    // Non-terminal action: back, reset, valid game action, or invalid.
                    if self.prefix == "back" {
                        self.episode.undo();
                    } else if self.prefix == "reset" {
                        self.episode.reset();
                    } else {
                        // Valid or invalid action (apply returns false for garbage)
                        self.episode.apply(&self.prefix);
                    }
                    self.render_pending = should_render;
                    self.phase = if self.has_boards {
                        Phase::BoardOpen
                    } else {
                        Phase::ActionOpen
                    };

                    // Count all non-terminal actions and enforce action limit
                    self.action_count += 1;
                    if let Some(limit) = self.action_limit {
                        if self.action_count >= limit {
                            self.phase = Phase::Done; // Silent termination, no EOS supervised
                            self.render_pending = false; // No board tokens will follow
                            self.timed_out = true;
                        }
                    }
                }
                self.prefix.clear();
                self.target = None;
            }
            Phase::ThinkClose => {
                self.phase = Phase::Response(0);
            }
            Phase::Response(i) => {
                if i + 1 < self.response_tokens.len() {
                    self.phase = Phase::Response(i + 1);
                } else {
                    self.phase = Phase::ImEnd;
                }
            }
            Phase::ImEnd => {
                self.phase = Phase::Done;
            }
            Phase::Done => {
                // No-op
            }
            // Board phases
            Phase::BoardOpen => {
                if executed_token == self.special.action_open {
                    // No board this step: read the action that follows.
                    self.phase = Phase::ActionOpen;
                    self.advance(executed_token);
                    return;
                }
                if executed_token != self.special.board_open {
                    self.skip_stray_token();
                    return;
                }
                self.stray_tokens = 0;
                if let Some(text_boards) = self.board_text {
                    let text = match text_boards {
                        TextBoards::StateDescriptions => self.episode.state_text(self.action_count),
                        TextBoards::ObjectBoxes => self.episode.object_boxes_text(),
                    };
                    self.board_text_tokens = self
                        .tokenizer
                        .encode(text, false)
                        .unwrap()
                        .get_ids()
                        .to_vec();
                    assert!(
                        self.board_text_tokens.len() < MAX_BOARD_TEXT_TOKENS,
                        "text board of {} tokens exceeds MAX_BOARD_TEXT_TOKENS",
                        self.board_text_tokens.len()
                    );
                    self.phase = Phase::BoardText(0);
                } else {
                    self.phase = Phase::BoardVisual(0);
                }
            }
            Phase::BoardVisual(i) => {
                if executed_token == self.special.board_close {
                    // The model closed its board early.
                    self.phase = Phase::ActionOpen;
                } else if executed_token == self.special.action_open {
                    self.phase = Phase::ActionOpen;
                    self.advance(executed_token);
                } else {
                    self.phase = if i + 1 == self.num_tokens {
                        Phase::BoardClose
                    } else {
                        Phase::BoardVisual(i + 1)
                    };
                }
            }
            Phase::BoardClose => {
                // The board ends at </board> (or at an <action> without it); extra
                // tokens before it are skipped.
                if executed_token == self.special.board_close {
                    self.phase = Phase::ActionOpen;
                } else if executed_token == self.special.action_open {
                    self.phase = Phase::ActionOpen;
                    self.advance(executed_token);
                } else {
                    self.skip_stray_token();
                }
            }
            Phase::BoardText(i) => {
                // The model writes free text; the board ends at </board>. During
                // evaluation an <action> without </board> also ends it, and a
                // runaway description is cut off after MAX_BOARD_TEXT_TOKENS tokens.
                if executed_token == self.special.board_close {
                    self.phase = Phase::ActionOpen;
                } else if executed_token == self.special.action_open {
                    self.phase = Phase::ActionOpen;
                    self.advance(executed_token);
                } else if i + 1 >= MAX_BOARD_TEXT_TOKENS {
                    self.phase = Phase::ActionOpen;
                } else {
                    self.phase = Phase::BoardText(i + 1);
                }
            }
        }
    }
}

/// Longest text board a model may write before it is force-closed during evaluation
/// (the longest state description of the benchmark is about 220 tokens).
const MAX_BOARD_TEXT_TOKENS: usize = 1024;

/// Consecutive stray tokens (not the expected `<board>`/`<action>` tag) after which
/// an evaluation episode ends as a failure.
const MAX_STRAY_TOKENS: usize = 256;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::episode::GameEpisode;
    use crate::base::game::RgbBuffer;
    use crate::games::sokoban::state::from_ascii;
    use crate::games::sokoban::Sokoban;
    use crate::stream::prefix::next_token_of;
    use rand::SeedableRng;
    use std::sync::Arc;
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
    const PLACEHOLDER: TokenId = 999;

    fn test_special() -> SpecialTokenIds {
        SpecialTokenIds {
            action_open: ACTION_OPEN,
            action_close: ACTION_CLOSE,
            eos: EOS,
            ignore: IGNORE,
            pad: PAD,
            board_open: BOARD_OPEN,
            board_close: BOARD_CLOSE,
            visual_token_placeholder: PLACEHOLDER,
            think_close: THINK_CLOSE,
            im_end: IM_END,
        }
    }

    /// Replicate the test tokenizer from prefix.rs
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
        id += 1; // 42
        vocab.insert("do".to_string(), id);
        id += 1; // 43
        vocab.insert("wn".to_string(), id);
        id += 1; // 44
        vocab.insert("ne".to_string(), id);
        id += 1; // 45
        vocab.insert("down".to_string(), id);
        id += 1; // 46
        vocab.insert("done".to_string(), id);
        id += 1; // 47
                 // Characters of JSON state descriptions (text-board tests): ids 48+.
        for c in ['{', '}', '"', ':', '#', '@', 'O', 'X', '.', '*', '+'] {
            vocab.insert(c.to_string(), id);
            id += 1;
        }

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

    fn make_stream(ascii: &str) -> EpisodeStream {
        let state = from_ascii(ascii);
        let game = Arc::new(Sokoban::default());
        let episode = GameEpisode::new(game, state, 100_000);
        let tokenizer = test_tokenizer();
        let rng = ChaChaRng::seed_from_u64(0);
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

    // Token ID helpers
    fn tok_id(tokenizer: &Tokenizer, s: &str) -> TokenId {
        let enc = tokenizer.encode(s, false).unwrap();
        enc.get_ids()[0]
    }

    /// Encode a full string and return all token IDs.
    fn encode_all(tokenizer: &Tokenizer, s: &str) -> Vec<TokenId> {
        let enc = tokenizer.encode(s, false).unwrap();
        enc.get_ids().to_vec()
    }

    // =========================================================================
    // Skeleton and phase transitions
    // =========================================================================

    #[test]
    fn phase_transitions_basic() {
        // #####
        // #@OX#  — player at (1,1), box at (2,1), goal at (3,1)
        // #####
        // Optimal: "right" (push box onto goal)
        let mut stream = make_stream("#####\n#@OX#\n#####");
        let tok = test_tokenizer();

        assert_eq!(stream.phase, Phase::ActionOpen);

        // Feed action_open token → ActionOpen advances to Action
        stream.advance(ACTION_OPEN);
        assert_eq!(stream.phase, Phase::Action);

        // Feed tokens for "right": r, i, g, h, t
        let right_tokens = encode_all(&tok, "right");
        assert_eq!(right_tokens.len(), 5); // all char-level

        for &t in &right_tokens[..right_tokens.len() - 1] {
            stream.advance(t);
            assert_eq!(stream.phase, Phase::Action);
        }
        // Last token completes the action
        stream.advance(right_tokens[right_tokens.len() - 1]);
        assert_eq!(stream.phase, Phase::ActionClose);

        // Feed action_close → applies "right", back to ActionOpen
        stream.advance(ACTION_CLOSE);
        assert_eq!(stream.phase, Phase::ActionOpen);
    }

    #[test]
    fn done_action_transitions_to_think_close() {
        // #####
        // #@OX#  — push right to solve, then "done"
        // #####
        let mut stream = make_stream("#####\n#@OX#\n#####");
        let tok = test_tokenizer();

        // First action: "right" to solve
        stream.advance(ACTION_OPEN); // → Action
        for &t in &encode_all(&tok, "right") {
            stream.advance(t);
        }
        assert_eq!(stream.phase, Phase::ActionClose);
        stream.advance(ACTION_CLOSE); // applies "right" → solved → ActionOpen

        // Now expert should say "done"
        assert_eq!(stream.phase, Phase::ActionOpen);

        // Second action: "done"
        stream.advance(ACTION_OPEN); // → Action
        let done_tokens = encode_all(&tok, "done");
        for &t in &done_tokens {
            stream.advance(t);
        }
        assert_eq!(stream.phase, Phase::ActionClose);
        stream.advance(ACTION_CLOSE); // "done" → ThinkClose
        assert_eq!(stream.phase, Phase::ThinkClose);

        // ThinkClose → Response(0)
        stream.advance(THINK_CLOSE);
        assert_eq!(stream.phase, Phase::Response(0));

        // Walk through response tokens (\n\ndone)
        let response_tokens = stream.response_tokens.clone();
        assert!(
            !response_tokens.is_empty(),
            "response_tokens should be populated"
        );
        for i in 0..response_tokens.len() {
            assert_eq!(stream.phase, Phase::Response(i));
            stream.advance(response_tokens[i]);
        }

        // After all response tokens → ImEnd
        assert_eq!(stream.phase, Phase::ImEnd);
        stream.advance(IM_END);
        assert_eq!(stream.phase, Phase::Done);
    }

    #[test]
    fn back_action_calls_undo() {
        // optimal_len=2 (right, right), limit=3 — room for "right" + "back"
        let mut stream = make_stream("######\n#@.OX#\n######");
        let tok = test_tokenizer();

        // Do a "right" move first
        stream.advance(ACTION_OPEN);
        for &t in &encode_all(&tok, "right") {
            stream.advance(t);
        }
        stream.advance(ACTION_CLOSE);

        // Now "back"
        stream.advance(ACTION_OPEN);
        let back_tokens = encode_all(&tok, "back");
        for &t in &back_tokens {
            stream.advance(t);
        }
        assert_eq!(stream.phase, Phase::ActionClose);
        stream.advance(ACTION_CLOSE);
        // "back" calls undo(), phase → ActionOpen
        assert_eq!(stream.phase, Phase::ActionOpen);
    }

    #[test]
    fn reset_action_calls_reset() {
        // optimal_len=2 (right, right), limit=3 — room for "right" + "reset"
        let mut stream = make_stream("######\n#@.OX#\n######");
        let tok = test_tokenizer();

        // Do a "right" move
        stream.advance(ACTION_OPEN);
        for &t in &encode_all(&tok, "right") {
            stream.advance(t);
        }
        stream.advance(ACTION_CLOSE);

        // Now "reset"
        stream.advance(ACTION_OPEN);
        let reset_tokens = encode_all(&tok, "reset");
        for &t in &reset_tokens {
            stream.advance(t);
        }
        assert_eq!(stream.phase, Phase::ActionClose);
        stream.advance(ACTION_CLOSE);
        assert_eq!(stream.phase, Phase::ActionOpen);
    }

    #[test]
    fn garbage_in_action_close_is_noop() {
        let mut stream = make_stream("#####\n#@OX#\n#...#\n#####");
        let tok = test_tokenizer();

        // Feed garbage tokens that don't match any action
        stream.advance(ACTION_OPEN);

        // "z" — matches no action, immediately transitions to ActionClose
        let z_id = tok_id(&tok, "z");
        stream.advance(z_id);
        assert_eq!(stream.phase, Phase::ActionClose);

        // Advance past ActionClose — garbage prefix "z" means no-op
        stream.advance(ACTION_CLOSE);
        assert_eq!(stream.phase, Phase::ActionOpen);
    }

    #[test]
    fn done_phase_emits_ignore() {
        let mut stream = make_stream("#####\n#@OX#\n#####");
        stream.phase = Phase::Done;

        let label = stream.expert_label();
        assert_eq!(label, IGNORE);

        // Advance is no-op
        stream.advance(42);
        assert_eq!(stream.phase, Phase::Done);
    }

    #[test]
    fn global_eos_override() {
        let mut stream = make_stream("#####\n#@OX#\n#####");
        assert_eq!(stream.phase, Phase::ActionOpen);

        // EOS in any phase → Done
        stream.advance(EOS);
        assert_eq!(stream.phase, Phase::Done);
    }

    #[test]
    fn expert_label_format_phases() {
        let stream = make_stream("#####\n#@OX#\n#####");

        // ActionOpen → action_open
        assert_eq!(stream.expert_label(), ACTION_OPEN);
    }

    // =========================================================================
    // On-path action labels
    // =========================================================================

    #[test]
    fn label_replay_full_episode() {
        // #####
        // #@OX#  — optimal: "right", then "done"
        // #####
        let mut stream = make_stream("#####\n#@OX#\n#####");
        let tok = test_tokenizer();

        // --- Action 1: "right" ---
        // ActionOpen label
        let label = stream.expert_label();
        assert_eq!(label, ACTION_OPEN);
        stream.advance(ACTION_OPEN);

        // Action phase: feed expert tokens for "right"
        let right_target = ActionTarget::new("right", &tok);
        let mut prefix_len = 0usize;
        for i in 0..right_target.tokens.len() {
            let expert_tok = next_token_of(&right_target, prefix_len, &tok);
            let label = stream.expert_label();
            assert_eq!(label, expert_tok, "label mismatch at right token {}", i);

            stream.advance(expert_tok);
            let decoded = tok.decode(&[expert_tok], false).unwrap();
            prefix_len += decoded.len();
        }

        // ActionClose label
        assert_eq!(stream.phase, Phase::ActionClose);
        let label = stream.expert_label();
        assert_eq!(label, ACTION_CLOSE);
        stream.advance(ACTION_CLOSE);

        // --- Action 2: "done" ---
        assert_eq!(stream.phase, Phase::ActionOpen);
        let label = stream.expert_label();
        assert_eq!(label, ACTION_OPEN);
        stream.advance(ACTION_OPEN);

        let done_target = ActionTarget::new("done", &tok);
        prefix_len = 0;
        for i in 0..done_target.tokens.len() {
            let expert_tok = next_token_of(&done_target, prefix_len, &tok);
            let label = stream.expert_label();
            assert_eq!(label, expert_tok, "label mismatch at done token {}", i);

            stream.advance(expert_tok);
            let decoded = tok.decode(&[expert_tok], false).unwrap();
            prefix_len += decoded.len();
        }

        assert_eq!(stream.phase, Phase::ActionClose);
        let label = stream.expert_label();
        assert_eq!(label, ACTION_CLOSE);
        stream.advance(ACTION_CLOSE);

        // "done" → ThinkClose → Response(0..n) → ImEnd → Done
        assert_eq!(stream.phase, Phase::ThinkClose);
        let label = stream.expert_label();
        assert_eq!(label, THINK_CLOSE);
        stream.advance(THINK_CLOSE);

        // Walk through response tokens
        let response_tokens = stream.response_tokens.clone();
        for (i, &resp_tok) in response_tokens.iter().enumerate() {
            assert_eq!(stream.phase, Phase::Response(i));
            let label = stream.expert_label();
            assert_eq!(label, resp_tok, "label mismatch at response token {}", i);
            stream.advance(resp_tok);
        }

        assert_eq!(stream.phase, Phase::ImEnd);
        let label = stream.expert_label();
        assert_eq!(label, IM_END);
        stream.advance(IM_END);
        assert_eq!(stream.phase, Phase::Done);
    }

    #[test]
    fn non_canonical_split_completes_action() {
        // "down" canonically merges to a single token; the expert emits it, while a
        // model may write it character by character. Both complete the action.
        let mut stream = make_stream("#####\n#...#\n#@..#\n#.OX#\n#####");
        let tok = test_tokenizer();

        stream.advance(ACTION_OPEN);
        assert_eq!(stream.target.as_ref().unwrap().action_str, "down");
        assert_eq!(stream.expert_label(), tok_id(&tok, "down"));

        stream.advance(tok_id(&tok, "d"));
        assert_eq!(stream.prefix, "d");
        assert_eq!(stream.phase, Phase::Action);
        stream.advance(tok_id(&tok, "o"));
        assert_eq!(stream.prefix, "do");
        stream.advance(tok_id(&tok, "wn"));
        assert_eq!(stream.phase, Phase::ActionClose);
    }

    // =========================================================================
    // Divergence from the expert
    // =========================================================================

    #[test]
    fn diverging_from_target_completes_other_action() {
        // #####
        // #@OX#
        // #...#
        // #####
        let mut stream = make_stream("#####\n#@OX#\n#...#\n#####");
        let tok = test_tokenizer();

        stream.advance(ACTION_OPEN);
        let original = stream.target.as_ref().unwrap().action_str.clone();
        assert_eq!(original, "right");

        // Feed "d" then "o" then "w" then "n" for "down"
        for ch in ["d", "o", "w", "n"] {
            let t = tok_id(&tok, ch);
            stream.advance(t);
        }

        // "down" is complete although the expert target was "right"
        assert_eq!(stream.phase, Phase::ActionClose);
        assert_eq!(stream.prefix, "down");
        assert!(stream.target.is_none());
    }

    // =========================================================================
    // Garbage prefix recovery
    // =========================================================================

    #[test]
    fn garbage_token_labels_action_close() {
        let mut stream = make_stream("#####\n#@OX#\n#####");
        let tok = test_tokenizer();

        stream.advance(ACTION_OPEN);

        // Feed "z" — matches no action (no sokoban action starts with 'z')
        let z_tok = tok_id(&tok, "z");
        stream.advance(z_tok);

        // Now prefix is "z", matching_targets("z") is empty → ActionClose
        assert_eq!(stream.phase, Phase::ActionClose);

        assert_eq!(stream.expert_label(), ACTION_CLOSE);

        // After ActionClose, state should be unchanged (no valid action applied)
        stream.advance(ACTION_CLOSE);
        assert_eq!(stream.phase, Phase::ActionOpen);
    }

    #[test]
    fn garbage_state_unchanged_after_close() {
        let mut stream = make_stream("#####\n#@OX#\n#####");
        let tok = test_tokenizer();

        // Get initial expert action
        let expert_before = stream.solve_optimal().action_str;

        // Feed garbage action
        stream.advance(ACTION_OPEN);
        let z_tok = tok_id(&tok, "z");
        stream.advance(z_tok); // garbage → ActionClose
        stream.advance(ACTION_CLOSE); // no-op → ActionOpen

        // Expert should be the same
        let expert_after = stream.solve_optimal().action_str;
        assert_eq!(expert_before, expert_after);
    }

    #[test]
    fn max_action_str_len_safety_net() {
        // Create a stream and manually set a very small max_action_str_len
        let mut stream = make_stream("#####\n#@OX#\n#...#\n#####");
        stream.max_action_str_len = 2;

        let tok = test_tokenizer();

        stream.advance(ACTION_OPEN);

        // Feed "ri" — 2 chars, hits the safety net
        let r_tok = tok_id(&tok, "r");
        stream.advance(r_tok);
        assert_eq!(stream.phase, Phase::Action); // "r" matches "right", len=1 < 2

        let i_tok = tok_id(&tok, "i");
        stream.advance(i_tok);
        // prefix "ri" has len 2 >= max_action_str_len → forced ActionClose
        assert_eq!(stream.phase, Phase::ActionClose);
    }

    #[test]
    fn consecutive_garbage_keeps_labeling_action_close() {
        let mut stream = make_stream("#####\n#@OX#\n#####");
        let tok = test_tokenizer();

        // Get to Action phase
        stream.advance(ACTION_OPEN);

        // Garbage "z" instead of the expert token
        let z_tok = tok_id(&tok, "z");
        stream.advance(z_tok);
        // → ActionClose (empty matching)
        assert_eq!(stream.phase, Phase::ActionClose);

        assert_eq!(stream.expert_label(), ACTION_CLOSE);
    }

    // =========================================================================
    // Model emits structural tokens during the Action phase (evaluation)
    // =========================================================================

    #[test]
    fn action_close_executed_mid_action_transitions_to_action_close() {
        // The model closes the action early. advance(ACTION_CLOSE) should
        // transition to ActionClose, not decode it as text.
        let mut stream = make_stream("#####\n#@OX#\n#####");

        stream.advance(ACTION_OPEN);
        assert_eq!(stream.phase, Phase::Action);

        // The model writes </action> mid-action
        stream.advance(ACTION_CLOSE);

        // Must transition to ActionClose — not stay in Action with garbage prefix
        assert_eq!(stream.phase, Phase::ActionClose);
        // Prefix should be empty (no text decoded from the special token)
        assert_eq!(stream.prefix, "");
    }

    #[test]
    fn action_close_mid_action_then_recovers() {
        // Full round-trip: enter Action, the model writes </action>, ActionClose
        // processes empty prefix as garbage no-op, back to ActionOpen, game state unchanged.
        let mut stream = make_stream("#####\n#@OX#\n#####");
        let expert_before = stream.solve_optimal().action_str;

        // ActionOpen → Action
        stream.advance(ACTION_OPEN);
        assert_eq!(stream.phase, Phase::Action);

        // Feed one content token, then </action>
        let tok = test_tokenizer();
        let r_tok = tok_id(&tok, "r");
        stream.advance(r_tok);
        assert_eq!(stream.phase, Phase::Action);
        assert_eq!(stream.prefix, "r");

        // Early close
        stream.advance(ACTION_CLOSE);
        assert_eq!(stream.phase, Phase::ActionClose);

        // ActionClose processes "r" — not a valid action → no-op → ActionOpen
        stream.advance(ACTION_CLOSE);
        assert_eq!(stream.phase, Phase::ActionOpen);

        // Game state unchanged — same expert action
        let expert_after = stream.solve_optimal().action_str;
        assert_eq!(expert_before, expert_after);
    }

    #[test]
    fn special_tokens_not_decoded_as_text_in_action() {
        // PAD/IGNORE/ACTION_OPEN should not be decoded and appended to prefix
        let mut stream = make_stream("#####\n#@OX#\n#####");

        stream.advance(ACTION_OPEN);
        assert_eq!(stream.phase, Phase::Action);
        assert_eq!(stream.prefix, "");

        // Execute PAD during Action — should be no-op, not append text
        stream.advance(PAD);
        assert_eq!(stream.phase, Phase::Action);
        assert_eq!(stream.prefix, "", "PAD should not be decoded into prefix");

        // Execute ACTION_OPEN during Action — should be no-op
        stream.advance(ACTION_OPEN);
        assert_eq!(stream.phase, Phase::Action);
        assert_eq!(
            stream.prefix, "",
            "ACTION_OPEN should not be decoded into prefix"
        );
    }

    // =========================================================================
    // Board text phase tests (State CoT)
    // =========================================================================

    /// Expert token sequence of a whole episode (teacher forcing).
    fn expert_sequence(stream: &mut EpisodeStream) -> Vec<TokenId> {
        let mut sequence = Vec::new();
        while stream.phase != Phase::Done {
            let label = stream.expert_label();
            sequence.push(label);
            stream.advance(label);
        }
        sequence
    }

    #[test]
    fn text_boards_describe_every_state_before_each_action() {
        let tok = test_tokenizer();
        let mut stream =
            make_stream("#####\n#@OX#\n#####").with_text_boards(TextBoards::StateDescriptions);
        assert_eq!(stream.phase, Phase::BoardOpen);
        let sequence = expert_sequence(&mut stream);

        let mut expected = vec![BOARD_OPEN];
        expected.extend(encode_all(&tok, "{\"g\":\"######@OX######\",\"moves\":0}"));
        expected.extend([BOARD_CLOSE, ACTION_OPEN]);
        expected.extend(encode_all(&tok, "right"));
        expected.extend([ACTION_CLOSE, BOARD_OPEN]);
        expected.extend(encode_all(&tok, "{\"g\":\"######.@*######\",\"moves\":1}"));
        expected.extend([BOARD_CLOSE, ACTION_OPEN]);
        expected.extend(encode_all(&tok, "done"));
        expected.extend([ACTION_CLOSE, THINK_CLOSE]);
        assert_eq!(&sequence[..expected.len()], &expected[..]);
        assert_eq!(*sequence.last().unwrap(), IM_END);
        assert!(stream.episode.is_solved());
    }

    #[test]
    fn box_boards_list_object_positions_before_each_action() {
        let tok = test_tokenizer();
        let mut stream =
            make_stream("#####\n#@OX#\n#####").with_text_boards(TextBoards::ObjectBoxes);
        let sequence = expert_sequence(&mut stream);

        let mut expected = vec![BOARD_OPEN];
        expected.extend(encode_all(
            &tok,
            "{\"player\":[1,1],\"box_0\":[2,1],\"goal_0\":[3,1]}",
        ));
        expected.extend([BOARD_CLOSE, ACTION_OPEN]);
        expected.extend(encode_all(&tok, "right"));
        expected.extend([ACTION_CLOSE, BOARD_OPEN]);
        expected.extend(encode_all(
            &tok,
            "{\"player\":[2,1],\"box_0\":[3,1],\"goal_0\":[3,1]}",
        ));
        expected.extend([BOARD_CLOSE, ACTION_OPEN]);
        assert_eq!(&sequence[..expected.len()], &expected[..]);
        assert!(stream.episode.is_solved());
    }

    #[test]
    fn text_board_ends_at_board_close_whatever_its_length() {
        // Evaluation replay: the model's description may differ from the expert's.
        let tok = test_tokenizer();
        let mut stream =
            make_stream("#####\n#@OX#\n#####").with_text_boards(TextBoards::StateDescriptions);
        stream.advance(BOARD_OPEN);
        stream.advance(tok_id(&tok, "x"));
        assert_eq!(stream.phase, Phase::BoardText(1));
        stream.advance(BOARD_CLOSE);
        assert_eq!(stream.phase, Phase::ActionOpen);
    }

    #[test]
    fn action_open_inside_text_board_starts_the_action() {
        let tok = test_tokenizer();
        let mut stream =
            make_stream("#####\n#@OX#\n#####").with_text_boards(TextBoards::StateDescriptions);
        stream.advance(BOARD_OPEN);
        stream.advance(tok_id(&tok, "{"));
        stream.advance(ACTION_OPEN);
        assert_eq!(stream.phase, Phase::Action);
        for &t in &encode_all(&tok, "right") {
            stream.advance(t);
        }
        stream.advance(ACTION_CLOSE);
        assert_eq!(stream.phase, Phase::BoardOpen);
    }

    #[test]
    fn runaway_text_board_is_closed() {
        let tok = test_tokenizer();
        let mut stream =
            make_stream("#####\n#@OX#\n#####").with_text_boards(TextBoards::StateDescriptions);
        stream.advance(BOARD_OPEN);
        for _ in 0..MAX_BOARD_TEXT_TOKENS {
            assert!(matches!(stream.phase, Phase::BoardText(_)));
            stream.advance(tok_id(&tok, "x"));
        }
        assert_eq!(stream.phase, Phase::ActionOpen);
    }

    #[test]
    fn missing_board_reads_the_action() {
        let tok = test_tokenizer();
        let mut stream =
            make_stream("#####\n#@OX#\n#####").with_text_boards(TextBoards::StateDescriptions);
        stream.advance(ACTION_OPEN);
        assert_eq!(stream.phase, Phase::Action);
        for &t in &encode_all(&tok, "right") {
            stream.advance(t);
        }
        stream.advance(ACTION_CLOSE);
        assert!(stream.episode.is_solved());
    }

    #[test]
    fn stray_tokens_before_a_tag_are_skipped() {
        let tok = test_tokenizer();
        let mut stream = make_stream("#####\n#@OX#\n#####");
        stream.advance(tok_id(&tok, "x"));
        assert_eq!(stream.phase, Phase::ActionOpen);
        stream.advance(ACTION_OPEN);
        assert_eq!(stream.phase, Phase::Action);
    }

    #[test]
    fn endless_stray_tokens_end_the_episode() {
        let tok = test_tokenizer();
        let mut stream = make_stream("#####\n#@OX#\n#####");
        for _ in 0..MAX_STRAY_TOKENS {
            stream.advance(tok_id(&tok, "x"));
        }
        assert_eq!(stream.phase, Phase::Done);
        assert!(!stream.episode.is_solved());
    }

    #[test]
    fn visual_board_ends_at_an_early_board_close() {
        let mut stream = make_stream_visual("#####\n#@OX#\n#####");
        stream.advance(BOARD_OPEN);
        stream.advance(PLACEHOLDER);
        stream.advance(BOARD_CLOSE);
        assert_eq!(stream.phase, Phase::ActionOpen);
    }

    #[test]
    fn extra_visual_tokens_are_skipped_until_board_close() {
        let mut stream = make_stream_visual("#####\n#@OX#\n#####");
        stream.advance(BOARD_OPEN);
        for _ in 0..stream.num_tokens + 3 {
            stream.advance(PLACEHOLDER);
        }
        assert_eq!(stream.phase, Phase::BoardClose);
        stream.advance(BOARD_CLOSE);
        assert_eq!(stream.phase, Phase::ActionOpen);
    }

    // =========================================================================
    // Board visual-token phase tests
    // =========================================================================

    fn make_stream_visual(ascii: &str) -> EpisodeStream {
        let state = from_ascii(ascii);
        let game = Arc::new(Sokoban::default());
        let episode = GameEpisode::new(game, state, 100_000);
        let tokenizer = test_tokenizer();
        let rng = ChaChaRng::seed_from_u64(0);
        let visual_token_config = Some(VisualTokenConfig {
            short_edge_res: 32,
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

    #[test]
    fn test_initial_phase_with_visual_tokens() {
        let stream = make_stream_visual("#####\n#@OX#\n#####");
        assert_eq!(stream.phase, Phase::BoardOpen);
        assert!(stream.has_boards);
        assert_eq!(stream.num_tokens, 4);
    }

    #[test]
    fn test_board_visual_sequence() {
        let mut stream = make_stream_visual("#####\n#@OX#\n#####");

        // BoardOpen → BoardVisual(0)
        assert_eq!(stream.phase, Phase::BoardOpen);
        stream.advance(BOARD_OPEN);
        assert_eq!(stream.phase, Phase::BoardVisual(0));

        // BoardVisual(0) → BoardVisual(1) → BoardVisual(2) → BoardVisual(3) → BoardClose
        // Any token advances a visual-token board position (visual-token ids 310..340).
        stream.advance(310);
        assert_eq!(stream.phase, Phase::BoardVisual(1));

        stream.advance(320);
        assert_eq!(stream.phase, Phase::BoardVisual(2));

        stream.advance(330);
        assert_eq!(stream.phase, Phase::BoardVisual(3));

        stream.advance(340); // last token
        assert_eq!(stream.phase, Phase::BoardClose);

        // BoardClose → ActionOpen
        stream.advance(BOARD_CLOSE);
        assert_eq!(stream.phase, Phase::ActionOpen);
    }

    #[test]
    fn test_action_close_to_board_open_with_visual_tokens() {
        // After applying a valid action, ActionClose → BoardOpen (not ActionOpen) when has_boards
        let mut stream = make_stream_visual("#####\n#@OX#\n#...#\n#####");
        let tok = test_tokenizer();

        // Skip initial board phase
        stream.advance(BOARD_OPEN);
        for _ in 0..4 {
            stream.advance(0);
        }
        stream.advance(BOARD_CLOSE);
        assert_eq!(stream.phase, Phase::ActionOpen);

        // Do a "right" action
        stream.advance(ACTION_OPEN);
        for &t in &encode_all(&tok, "right") {
            stream.advance(t);
        }
        assert_eq!(stream.phase, Phase::ActionClose);
        stream.advance(ACTION_CLOSE);

        // Should go to BoardOpen, not ActionOpen
        assert_eq!(stream.phase, Phase::BoardOpen);
        assert!(stream.render_pending);
    }

    #[test]
    fn test_garbage_recovery_with_visual_tokens() {
        let mut stream = make_stream_visual("#####\n#@OX#\n#...#\n#####");
        let tok = test_tokenizer();

        // Skip initial board phase
        stream.advance(BOARD_OPEN);
        for _ in 0..4 {
            stream.advance(0);
        }
        stream.advance(BOARD_CLOSE);

        // Enter action, feed garbage
        stream.advance(ACTION_OPEN);
        let z_tok = tok_id(&tok, "z");
        stream.advance(z_tok); // → ActionClose (empty matching)
        assert_eq!(stream.phase, Phase::ActionClose);

        stream.advance(ACTION_CLOSE);
        // Garbage no-op → BoardOpen (not ActionOpen) when has_boards
        assert_eq!(stream.phase, Phase::BoardOpen);
        assert!(stream.render_pending);
    }

    #[test]
    fn test_back_reset_with_visual_tokens() {
        // optimal_len=3 (right x3), limit=5 — room for right + back + reset
        let mut stream = make_stream_visual("#######\n#@..OX#\n#######");
        let tok = test_tokenizer();

        // Skip initial board
        stream.advance(BOARD_OPEN);
        for _ in 0..4 {
            stream.advance(0);
        }
        stream.advance(BOARD_CLOSE);

        // Do "right" action first
        stream.advance(ACTION_OPEN);
        for &t in &encode_all(&tok, "right") {
            stream.advance(t);
        }
        stream.advance(ACTION_CLOSE);
        assert_eq!(stream.phase, Phase::BoardOpen);
        stream.render_pending = false; // clear for next check

        // Skip board phase
        stream.advance(BOARD_OPEN);
        for _ in 0..4 {
            stream.advance(0);
        }
        stream.advance(BOARD_CLOSE);

        // Now "back"
        stream.advance(ACTION_OPEN);
        for &t in &encode_all(&tok, "back") {
            stream.advance(t);
        }
        stream.advance(ACTION_CLOSE);
        assert_eq!(stream.phase, Phase::BoardOpen);
        assert!(stream.render_pending);
        stream.render_pending = false;

        // Skip board
        stream.advance(BOARD_OPEN);
        for _ in 0..4 {
            stream.advance(0);
        }
        stream.advance(BOARD_CLOSE);

        // Now "reset"
        stream.advance(ACTION_OPEN);
        for &t in &encode_all(&tok, "reset") {
            stream.advance(t);
        }
        stream.advance(ACTION_CLOSE);
        assert_eq!(stream.phase, Phase::BoardOpen);
        assert!(stream.render_pending);
    }

    #[test]
    fn test_label_replay_full_episode_with_visual_tokens() {
        // #####
        // #@OX#  — optimal: "right", then "done"
        // #####
        // With visual tokens: <board> vq[0..3] </board> <action> right </action>
        //          <board> vq[0..3] </board> <action> done </action> EOS
        let mut stream = make_stream_visual("#####\n#@OX#\n#####");
        let tok = test_tokenizer();
        let special = test_special();

        // Collect token stream for verification
        let mut token_stream: Vec<TokenId> = Vec::new();

        // --- Initial Board Phase ---
        assert_eq!(stream.phase, Phase::BoardOpen);
        let label = stream.expert_label();
        assert_eq!(label, BOARD_OPEN);
        token_stream.push(label);
        stream.advance(label);

        // visual tokens — the deferred sentinel
        for i in 0..4 {
            assert_eq!(stream.phase, Phase::BoardVisual(i));
            let label = stream.expert_label();
            assert_eq!(label, special.visual_token_placeholder);
            token_stream.push(label);
            stream.advance(label);
        }

        // BoardClose
        assert_eq!(stream.phase, Phase::BoardClose);
        let label = stream.expert_label();
        assert_eq!(label, BOARD_CLOSE);
        token_stream.push(label);
        stream.advance(label);

        // --- Action 1: "right" ---
        assert_eq!(stream.phase, Phase::ActionOpen);
        let label = stream.expert_label();
        assert_eq!(label, ACTION_OPEN);
        token_stream.push(label);
        stream.advance(label);

        let right_target = ActionTarget::new("right", &tok);
        let mut prefix_len = 0usize;
        for _ in 0..right_target.tokens.len() {
            let expert_tok = next_token_of(&right_target, prefix_len, &tok);
            let label = stream.expert_label();
            assert_eq!(label, expert_tok);
            token_stream.push(label);
            stream.advance(label);
            let decoded = tok.decode(&[expert_tok], false).unwrap();
            prefix_len += decoded.len();
        }

        assert_eq!(stream.phase, Phase::ActionClose);
        let label = stream.expert_label();
        assert_eq!(label, ACTION_CLOSE);
        token_stream.push(label);
        stream.advance(label);

        // --- Second Board Phase (after "right" applied) ---
        assert_eq!(stream.phase, Phase::BoardOpen);
        assert!(stream.render_pending);
        stream.render_pending = false;

        let label = stream.expert_label();
        assert_eq!(label, BOARD_OPEN);
        token_stream.push(label);
        stream.advance(label);

        for i in 0..4 {
            assert_eq!(stream.phase, Phase::BoardVisual(i));
            let label = stream.expert_label();
            assert_eq!(label, special.visual_token_placeholder);
            token_stream.push(label);
            stream.advance(label);
        }

        assert_eq!(stream.phase, Phase::BoardClose);
        let label = stream.expert_label();
        assert_eq!(label, BOARD_CLOSE);
        token_stream.push(label);
        stream.advance(label);

        // --- Action 2: "done" ---
        assert_eq!(stream.phase, Phase::ActionOpen);
        let label = stream.expert_label();
        assert_eq!(label, ACTION_OPEN);
        token_stream.push(label);
        stream.advance(label);

        let done_target = ActionTarget::new("done", &tok);
        prefix_len = 0;
        for _ in 0..done_target.tokens.len() {
            let expert_tok = next_token_of(&done_target, prefix_len, &tok);
            let label = stream.expert_label();
            assert_eq!(label, expert_tok);
            token_stream.push(label);
            stream.advance(label);
            let decoded = tok.decode(&[expert_tok], false).unwrap();
            prefix_len += decoded.len();
        }

        assert_eq!(stream.phase, Phase::ActionClose);
        let label = stream.expert_label();
        assert_eq!(label, ACTION_CLOSE);
        token_stream.push(label);
        stream.advance(label);

        // "done" → ThinkClose → Response → ImEnd → Done
        assert_eq!(stream.phase, Phase::ThinkClose);
        let label = stream.expert_label();
        assert_eq!(label, THINK_CLOSE);
        token_stream.push(label);
        stream.advance(label);

        let response_tokens = stream.response_tokens.clone();
        for (i, &resp_tok) in response_tokens.iter().enumerate() {
            assert_eq!(stream.phase, Phase::Response(i));
            let label = stream.expert_label();
            assert_eq!(label, resp_tok);
            token_stream.push(label);
            stream.advance(label);
        }

        assert_eq!(stream.phase, Phase::ImEnd);
        let label = stream.expert_label();
        assert_eq!(label, IM_END);
        token_stream.push(label);
        stream.advance(label);
        assert_eq!(stream.phase, Phase::Done);

        // Verify token stream structure:
        // [BOARD_OPEN, vq*4, BOARD_CLOSE, ACTION_OPEN, right_tokens..., ACTION_CLOSE,
        //  BOARD_OPEN, vq*4, BOARD_CLOSE, ACTION_OPEN, done_tokens..., ACTION_CLOSE,
        //  THINK_CLOSE, response_tokens..., IM_END]
        assert_eq!(token_stream[0], BOARD_OPEN);
        assert_eq!(token_stream[5], BOARD_CLOSE);
        assert_eq!(token_stream[6], ACTION_OPEN);
        // right tokens are in between
        let right_end = 6 + 1 + right_target.tokens.len();
        assert_eq!(token_stream[right_end], ACTION_CLOSE);
        assert_eq!(token_stream[right_end + 1], BOARD_OPEN);
        assert_eq!(token_stream[right_end + 6], BOARD_CLOSE);
        assert_eq!(*token_stream.last().unwrap(), IM_END);
    }

    // =========================================================================
    // Variable token count visual-token tests
    // =========================================================================

    fn make_stream_visual_n(ascii: &str, short_edge_res: u32) -> EpisodeStream {
        let state = from_ascii(ascii);
        let game = Arc::new(Sokoban::default());
        let episode = GameEpisode::new(game, state, 100_000);
        let tokenizer = test_tokenizer();
        let rng = ChaChaRng::seed_from_u64(0);
        let visual_token_config = Some(VisualTokenConfig {
            short_edge_res,
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

    /// Count how many BoardVisual phases occur in one board block.
    fn count_board_visual_phases(stream: &mut EpisodeStream, _num_tokens: usize) -> usize {
        assert_eq!(stream.phase, Phase::BoardOpen);
        stream.advance(BOARD_OPEN);
        let mut count = 0;
        while matches!(stream.phase, Phase::BoardVisual(_)) {
            stream.advance(0); // model_token doesn't matter for phase transitions
            count += 1;
        }
        assert_eq!(stream.phase, Phase::BoardClose);
        stream.advance(BOARD_CLOSE);
        count
    }

    #[test]
    fn test_visual_phase_transitions_16_tokens() {
        // Sokoban 1:1, short_edge_res=64 → (64/16)^2 = 16 tokens
        let mut stream = make_stream_visual_n("#####\n#@OX#\n#####", 64);
        assert_eq!(count_board_visual_phases(&mut stream, 16), 16);
    }

    #[test]
    fn test_visual_phase_transitions_64_tokens() {
        // Sokoban 1:1, short_edge_res=128 → (128/16)^2 = 64 tokens
        let mut stream = make_stream_visual_n("#####\n#@OX#\n#####", 128);
        assert_eq!(count_board_visual_phases(&mut stream, 64), 64);
    }

    #[test]
    fn test_visual_phase_transitions_256_tokens() {
        // Sokoban 1:1, short_edge_res=256 → (256/16)^2 = 256 tokens
        let mut stream = make_stream_visual_n("#####\n#@OX#\n#####", 256);
        assert_eq!(count_board_visual_phases(&mut stream, 256), 256);
    }

    #[test]
    fn test_placeholder_count_varies_with_short_edge_res() {
        // Run teacher forcing for two different resolutions, count sentinels.
        let special = test_special();

        // Helper: count sentinel tokens in one full board block via expert_label
        fn count_sentinels_in_board(
            stream: &mut EpisodeStream,
            special: &SpecialTokenIds,
        ) -> usize {
            assert_eq!(stream.phase, Phase::BoardOpen);
            let label = stream.expert_label();
            assert_eq!(label, special.board_open);
            stream.advance(label);

            let mut sentinel_count = 0;
            while matches!(stream.phase, Phase::BoardVisual(_)) {
                let label = stream.expert_label();
                if label == special.visual_token_placeholder {
                    sentinel_count += 1;
                }
                stream.advance(label);
            }
            assert_eq!(stream.phase, Phase::BoardClose);
            let label = stream.expert_label();
            assert_eq!(label, special.board_close);
            stream.advance(label);
            sentinel_count
        }

        let mut stream_16 = make_stream_visual_n("#####\n#@OX#\n#####", 64);
        let sentinels_16 = count_sentinels_in_board(&mut stream_16, &special);
        assert_eq!(sentinels_16, 16);

        let mut stream_64 = make_stream_visual_n("#####\n#@OX#\n#####", 128);
        let sentinels_64 = count_sentinels_in_board(&mut stream_64, &special);
        assert_eq!(sentinels_64, 64);

        // 128 short_edge_res produces 4x the sentinels of 64
        assert_eq!(sentinels_64, 4 * sentinels_16);
    }

    // =========================================================================
    // Action limit tests
    // =========================================================================

    #[test]
    fn action_limit_computed_from_optimal_len() {
        // 1-step puzzle: optimal_len=1, limit=ceil(1.5*1)=2
        let stream = make_stream("#####\n#@OX#\n#####");
        assert_eq!(stream.episode.optimal_len(), Some(1));
        assert_eq!(stream.action_limit, Some(2));
        assert_eq!(stream.action_count, 0);
        assert!(!stream.timed_out);
    }

    #[test]
    fn action_limit_terminates_episode() {
        // 1-step puzzle: optimal_len=1, limit=2
        // Execute 2 non-optimal actions to hit the limit
        let mut stream = make_stream("#####\n#@OX#\n#####");
        let tok = test_tokenizer();
        assert_eq!(stream.action_limit, Some(2));

        // Action 1: "left" (invalid in this position, but counts as action)
        stream.advance(ACTION_OPEN);
        for &t in &encode_all(&tok, "left") {
            stream.advance(t);
        }
        assert_eq!(stream.phase, Phase::ActionClose);
        stream.advance(ACTION_CLOSE);
        assert_eq!(stream.action_count, 1);
        assert!(!stream.timed_out);
        // Episode should continue
        assert_ne!(stream.phase, Phase::Done);

        // Action 2: "left" again — should hit limit
        stream.advance(ACTION_OPEN);
        for &t in &encode_all(&tok, "left") {
            stream.advance(t);
        }
        assert_eq!(stream.phase, Phase::ActionClose);
        stream.advance(ACTION_CLOSE);
        assert_eq!(stream.action_count, 2);
        assert!(stream.timed_out);
        assert_eq!(stream.phase, Phase::Done);
    }

    #[test]
    fn done_does_not_increment_action_count() {
        // 1-step puzzle: solve with "right", then "done"
        let mut stream = make_stream("#####\n#@OX#\n#####");
        let tok = test_tokenizer();

        // Action 1: "right" — solves the puzzle
        stream.advance(ACTION_OPEN);
        for &t in &encode_all(&tok, "right") {
            stream.advance(t);
        }
        stream.advance(ACTION_CLOSE);
        assert_eq!(stream.action_count, 1);
        assert!(!stream.timed_out);

        // Action 2: "done" — terminal, should NOT increment action_count
        stream.advance(ACTION_OPEN);
        for &t in &encode_all(&tok, "done") {
            stream.advance(t);
        }
        assert_eq!(stream.phase, Phase::ActionClose);
        stream.advance(ACTION_CLOSE);
        // "done" goes to ThinkClose, not counted
        assert_eq!(stream.action_count, 1);
        assert_eq!(stream.phase, Phase::ThinkClose);
        assert!(!stream.timed_out);
    }

    #[test]
    fn action_limit_skips_eos_supervision() {
        // When the limit fires, phase goes to Done (not ThinkClose).
        // So the label is ignore, not think_close.
        let mut stream = make_stream("#####\n#@OX#\n#####");
        let tok = test_tokenizer();
        assert_eq!(stream.action_limit, Some(2));

        // Burn through 2 actions
        for _ in 0..2 {
            stream.advance(ACTION_OPEN);
            for &t in &encode_all(&tok, "left") {
                stream.advance(t);
            }
            stream.advance(ACTION_CLOSE);
        }
        assert_eq!(stream.phase, Phase::Done);
        assert!(stream.timed_out);

        // In Done phase, label is ignore (not eos)
        let label = stream.expert_label();
        assert_eq!(label, IGNORE);
    }

    #[test]
    fn solved_before_limit_not_timed_out() {
        // Solve optimally in 1 action, well under the limit of 2
        let mut stream = make_stream("#####\n#@OX#\n#####");
        let tok = test_tokenizer();

        // Action 1: "right" — solves
        stream.advance(ACTION_OPEN);
        for &t in &encode_all(&tok, "right") {
            stream.advance(t);
        }
        stream.advance(ACTION_CLOSE);

        // "done"
        stream.advance(ACTION_OPEN);
        for &t in &encode_all(&tok, "done") {
            stream.advance(t);
        }
        stream.advance(ACTION_CLOSE);
        assert_eq!(stream.phase, Phase::ThinkClose);

        // Walk through think termination sequence
        stream.advance(THINK_CLOSE);
        let response_tokens = stream.response_tokens.clone();
        for &resp_tok in &response_tokens {
            stream.advance(resp_tok);
        }
        stream.advance(IM_END);
        assert_eq!(stream.phase, Phase::Done);

        assert!(!stream.timed_out);
        assert_eq!(stream.action_count, 1);
        assert!(stream.episode.is_solved());
    }

    #[test]
    fn back_and_reset_count_toward_limit() {
        // 1-step puzzle: limit=2. Use "back" and "reset" to exhaust limit.
        let mut stream = make_stream("#####\n#@OX#\n#####");
        let tok = test_tokenizer();
        assert_eq!(stream.action_limit, Some(2));

        // Action 1: "back" (no history, invalid but still counts)
        stream.advance(ACTION_OPEN);
        for &t in &encode_all(&tok, "back") {
            stream.advance(t);
        }
        stream.advance(ACTION_CLOSE);
        assert_eq!(stream.action_count, 1);
        assert!(!stream.timed_out);

        // Action 2: "reset" — should hit limit
        stream.advance(ACTION_OPEN);
        for &t in &encode_all(&tok, "reset") {
            stream.advance(t);
        }
        stream.advance(ACTION_CLOSE);
        assert_eq!(stream.action_count, 2);
        assert!(stream.timed_out);
        assert_eq!(stream.phase, Phase::Done);
    }

    // =========================================================================
    // Visual head mode: render-only (no board tokens)
    // =========================================================================

    /// Create a stream with render_resolution but no visual-token config (visual head mode).
    fn make_stream_render(ascii: &str, render_resolution: u32) -> EpisodeStream {
        let state = from_ascii(ascii);
        let game = Arc::new(Sokoban::default());
        let episode = GameEpisode::new(game, state, 100_000);
        let tokenizer = test_tokenizer();
        let rng = ChaChaRng::seed_from_u64(0);
        EpisodeStream::new(
            Box::new(episode),
            Arc::new(tokenizer),
            test_special(),
            rng,
            None,
            render_resolution,
            1.5,
        )
    }

    #[test]
    fn render_mode_starts_at_action_open() {
        let stream = make_stream_render("#####\n#@OX#\n#####", 128);
        assert_eq!(stream.phase, Phase::ActionOpen);
        assert!(!stream.has_boards);
        assert_eq!(stream.short_edge_res, 128);
    }

    #[test]
    fn render_mode_renders_after_action() {
        let mut stream = make_stream_render("#####\n#@OX#\n#...#\n#####", 128);
        let tok = test_tokenizer();

        assert!(!stream.render_pending);

        // Do "right" action
        stream.advance(ACTION_OPEN);
        for &t in &encode_all(&tok, "right") {
            stream.advance(t);
        }
        assert_eq!(stream.phase, Phase::ActionClose);
        stream.advance(ACTION_CLOSE);

        // Should render but go straight to ActionOpen (no BoardOpen)
        assert!(
            stream.render_pending,
            "should render after action in render mode"
        );
        assert_eq!(stream.phase, Phase::ActionOpen, "should skip board phases");
    }

    #[test]
    fn render_mode_renders_on_done() {
        let mut stream = make_stream_render("#####\n#@OX#\n#####", 128);
        let tok = test_tokenizer();

        // Solve: "right" then "done"
        stream.advance(ACTION_OPEN);
        for &t in &encode_all(&tok, "right") {
            stream.advance(t);
        }
        stream.advance(ACTION_CLOSE);
        stream.render_pending = false; // consume the render

        stream.advance(ACTION_OPEN);
        for &t in &encode_all(&tok, "done") {
            stream.advance(t);
        }
        stream.advance(ACTION_CLOSE);

        assert!(
            stream.render_pending,
            "should render on done for visual head alignment"
        );
        assert_eq!(stream.phase, Phase::ThinkClose);
    }

    #[test]
    fn render_mode_renders_on_back_and_reset() {
        // Use a 3-step puzzle (optimal_len=3, limit=5) so back/reset don't hit the limit
        let mut stream = make_stream_render("#######\n#@..OX#\n#######", 128);
        let tok = test_tokenizer();

        // Do "right"
        stream.advance(ACTION_OPEN);
        for &t in &encode_all(&tok, "right") {
            stream.advance(t);
        }
        stream.advance(ACTION_CLOSE);
        stream.render_pending = false;

        // "back"
        stream.advance(ACTION_OPEN);
        for &t in &encode_all(&tok, "back") {
            stream.advance(t);
        }
        stream.advance(ACTION_CLOSE);
        assert!(stream.render_pending, "should render on back");
        assert_eq!(stream.phase, Phase::ActionOpen);
        stream.render_pending = false;

        // "reset"
        stream.advance(ACTION_OPEN);
        for &t in &encode_all(&tok, "reset") {
            stream.advance(t);
        }
        stream.advance(ACTION_CLOSE);
        assert!(stream.render_pending, "should render on reset");
        assert_eq!(stream.phase, Phase::ActionOpen);
    }

    #[test]
    fn render_mode_no_render_when_resolution_zero() {
        // render_resolution=0, no visual tokens: no rendering at all
        let mut stream = make_stream("#####\n#@OX#\n#...#\n#####");
        let tok = test_tokenizer();

        stream.advance(ACTION_OPEN);
        for &t in &encode_all(&tok, "right") {
            stream.advance(t);
        }
        stream.advance(ACTION_CLOSE);

        assert!(
            !stream.render_pending,
            "should NOT render when resolution is 0"
        );
        assert_eq!(stream.phase, Phase::ActionOpen);
    }

    #[test]
    fn visual_tokens_mode_does_not_render_on_done() {
        // visual-token mode: "done" should NOT set render_pending (board was already rendered)
        let mut stream = make_stream_visual("#####\n#@OX#\n#####");
        let tok = test_tokenizer();

        // Skip initial board phase
        stream.advance(BOARD_OPEN);
        for _ in 0..4 {
            stream.advance(0);
        }
        stream.advance(BOARD_CLOSE);

        // "right" action + board phase
        stream.advance(ACTION_OPEN);
        for &t in &encode_all(&tok, "right") {
            stream.advance(t);
        }
        stream.advance(ACTION_CLOSE);
        assert!(stream.render_pending);
        stream.render_pending = false;

        // Board phase after "right"
        stream.advance(BOARD_OPEN);
        for _ in 0..4 {
            stream.advance(0);
        }
        stream.advance(BOARD_CLOSE);

        // "done" action
        stream.advance(ACTION_OPEN);
        for &t in &encode_all(&tok, "done") {
            stream.advance(t);
        }
        stream.advance(ACTION_CLOSE);

        assert!(
            !stream.render_pending,
            "visual-token mode should NOT render on done"
        );
        assert_eq!(stream.phase, Phase::ThinkClose);
    }

    /// Mock episode that mimics certificate-mode behavior:
    /// `is_solved()` returns `certificate_affirmed`, which is only set
    /// when `apply("yes")` is called.
    struct CertificateMockEpisode {
        certificate_affirmed: bool,
    }

    impl Episode for CertificateMockEpisode {
        fn is_solved(&self) -> bool {
            self.certificate_affirmed
        }
        fn apply(&mut self, action_str: &str) -> bool {
            if action_str == "yes" {
                self.certificate_affirmed = true;
                return true;
            }
            false
        }
        fn undo(&mut self) {}
        fn reset(&mut self) {}
        fn expert_action(&self) -> String {
            "yes".to_string()
        }
        fn enumerate_actions(&self) -> Vec<String> {
            vec!["yes".to_string(), "no".to_string()]
        }
        fn render(&self, _resolution: u32) -> RgbBuffer {
            RgbBuffer {
                width: 1,
                height: 1,
                data: vec![0, 0, 0],
            }
        }
        fn prompt(&self) -> String {
            String::new()
        }
        fn action_description(&self) -> String {
            String::new()
        }
        fn state_text(&self, moves: usize) -> String {
            format!("{{\"moves\":{moves}}}")
        }
        fn object_boxes_text(&self) -> String {
            "{}".to_string()
        }
    }

    #[test]
    fn certificate_yes_via_stream_sets_is_solved() {
        // Terminal actions ("yes", "no") must reach episode.apply() so that
        // certificate_affirmed is set.
        let episode = CertificateMockEpisode {
            certificate_affirmed: false,
        };
        let tokenizer = Arc::new(test_tokenizer());
        let rng = ChaChaRng::seed_from_u64(0);
        let mut stream = EpisodeStream::new(
            Box::new(episode),
            tokenizer.clone(),
            test_special(),
            rng,
            None,
            0,
            1.5,
        );

        assert!(!stream.episode.is_solved(), "precondition: not solved yet");

        // Feed: <action> y e s </action>
        stream.advance(ACTION_OPEN);
        assert_eq!(stream.phase, Phase::Action);

        for &tok in &encode_all(&tokenizer, "yes") {
            stream.advance(tok);
        }

        assert_eq!(stream.phase, Phase::ActionClose);
        stream.advance(ACTION_CLOSE);

        // "yes" is a terminal action → stream transitions to ThinkClose.
        assert_eq!(stream.phase, Phase::ThinkClose);
        assert!(
            stream.episode.is_solved(),
            "is_solved() should be true after 'yes' goes through the stream"
        );
    }
}
