/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2026 Apple Inc. All Rights Reserved.
 */
// Shared data and helpers for the project page.
(function () {
  "use strict";

  // Paper names and task prompts (appendix, "Puzzle task prompts"), in the paper's table order.
  const GAMES = {
    tangram: { name: "Tangram", prompt: "Complete the tangram. Fill the grey silhouette using the colored pieces." },
    tangram_certificate: { name: "Tangram (yes/no)", prompt: "Determine if the pieces can fill the silhouette. Say ‘yes’ if they can, ‘no’ if they can’t." },
    mental_rotation: { name: "3D Shape Matching", prompt: "Rotate the left shape to match the right shape." },
    mental_rotation_certificate: { name: "3D Mental Rotation (yes/no)", prompt: "Determine if the left and right shapes are the same. Rotate to check, then say ‘yes’ if they match or ‘no’ if they don’t." },
    mental_rotation_2d: { name: "2D Shape Matching", prompt: "Rotate the left 2D shape to match the right shape." },
    mental_rotation_2d_certificate: { name: "2D Mental Rotation (yes/no)", prompt: "Determine if the left and right 2D shapes are the same (possibly rotated). Rotate to check, then say ‘yes’ if they match or ‘no’ if they don’t." },
    jigsaw: { name: "Jigsaw", prompt: "Reconstruct the original image by rotating each piece to the correct orientation and placing it at its correct grid position." },
    anagram: { name: "Anagram", prompt: "The letters of a word have been scrambled. Swap letters to unscramble the word, then submit with identify(word)." },
    character_recognition: { name: "Character Recognition", prompt: "A character has been broken into pieces that are scattered and rotated. Rotate each piece into its slot, then identify the character." },
    sokoban: { name: "Sokoban", prompt: "Solve this Sokoban puzzle. Push the brown boxes onto the red diamonds." },
    bloxorz: { name: "Bloxorz", prompt: "Solve this Bloxorz puzzle. Roll the block so it stands on the red goal." },
    rush_hour: { name: "Rush Hour", prompt: "Slide vehicles to free the red car (A) and move it to the right edge exit." },
  };

  function el(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  }

  // An action as the model writes it: one segment per Qwen3.5 token (data/action_tokens.js).
  function actionChip(action) {
    const chip = el("span", "token-action");
    const tokens = window.ACTION_TOKENS[action];
    if (!tokens) console.warn(`No tokens for action ${action}`);
    (tokens || [action]).forEach((token) => chip.append(el("span", "token-piece", token)));
    return chip;
  }

  // Moves a tape's track (place), then slides the tokens that were already there from where they were to
  // where they are now, so only they move: a new token fades in at its final place.
  function shiftTokens({ track, place }) {
    const tokens = [...track.children];
    const before = tokens.map((token) => token.getBoundingClientRect().left);
    place();
    tokens.forEach((token, index) => {
      const shift = before[index] - token.getBoundingClientRect().left;
      if (Math.abs(shift) > 0.5) {
        token.animate([{ transform: `translateX(${shift}px)` }, { transform: "none" }], { duration: 300, easing: "ease" });
      }
    });
  }

  // A row of tokens that scrolls left once it outgrows its box, keeping the newest token in view.
  function tokenTape() {
    const tape = el("div", "token-tape");
    const track = el("div", "tape-track");
    tape.append(track);
    function fit() {
      const overflow = Math.max(0, track.scrollWidth - tape.clientWidth);
      tape.classList.toggle("is-overflowing", overflow > 0);
      track.style.transform = `translateX(${-overflow}px)`;
    }
    return {
      element: tape,
      push(token) {
        shiftTokens({
          track,
          place: () => {
            track.appendChild(token);
            fit();
          },
        });
        track.querySelectorAll(".is-current").forEach((current) => current.classList.remove("is-current"));
        token.classList.add("is-current");
      },
      settle() {
        track.querySelectorAll(".is-current").forEach((current) => current.classList.remove("is-current"));
        // On the next frame, after any layout change (such as the outcome appearing beside the tape).
        requestAnimationFrame(() => shiftTokens({ track, place: fit }));
      },
      clear() {
        track.replaceChildren();
        tape.classList.remove("is-overflowing");
        track.style.transform = "";
      },
    };
  }

  window.Sheep = { GAMES, el, actionChip, shiftTokens, tokenTape };
})();
