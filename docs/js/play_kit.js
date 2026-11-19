/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2026 Apple Inc. All Rights Reserved.
 */
// Helpers shared by the game ports in js/games/ (see js/play.js for the game module contract). No DOM work
// at load time, so the ports can be loaded in Node too.
(function () {
  "use strict";

  // Games draw in the engine's render space, 512 px.
  const RENDER_SIZE = 512;

  function button({ label, title, className, onClick }) {
    const element = document.createElement("button");
    element.type = "button";
    element.className = `play-button${className ? ` ${className}` : ""}`;
    element.textContent = label;
    if (title) element.title = title;
    element.addEventListener("click", onClick);
    return element;
  }

  const KEY_DIRECTIONS = {
    ArrowUp: "up", ArrowDown: "down", ArrowLeft: "left", ArrowRight: "right",
    w: "up", s: "down", a: "left", d: "right", W: "up", S: "down", A: "left", D: "right",
  };

  // "up" | "down" | "left" | "right" for arrow keys and WASD, else null.
  function keyDirection(event) {
    if (event.metaKey || event.ctrlKey || event.altKey) return null;
    return KEY_DIRECTIONS[event.key] || null;
  }

  // An on-screen arrow pad, for touch screens.
  function directionPad({ onDirection }) {
    const pad = document.createElement("div");
    pad.className = "play-dpad";
    [["up", "↑"], ["left", "←"], ["down", "↓"], ["right", "→"]].forEach(([direction, arrow]) => {
      pad.append(button({ label: arrow, title: direction, className: `dpad-${direction}`, onClick: () => onDirection(direction) }));
    });
    return pad;
  }

  // The engine's fit_to_resolution (engine/base/render.rs): integer cells, centered.
  function fitToResolution({ boardWidth, boardHeight, resolution = RENDER_SIZE }) {
    const cellSize = Math.max(1, Math.min(Math.floor(resolution / boardWidth), Math.floor(resolution / boardHeight)));
    return {
      offsetX: Math.floor((resolution - boardWidth * cellSize) / 2),
      offsetY: Math.floor((resolution - boardHeight * cellSize) / 2),
      cellSize,
    };
  }

  function rgb([red, green, blue]) {
    return `rgb(${red}, ${green}, ${blue})`;
  }

  // The engine's text is Roboto Mono Bold (assets/fonts/RobotoMono-Bold.ttf), sized by ab_glyph's PxScale: the
  // ascent-to-descent height, 2701 font units, of which the em is 2048 and the ascent 2146.
  const EM_PER_ENGINE_HEIGHT = 2048 / 2701;
  const ASCENT_PER_ENGINE_HEIGHT = 2146 / 2701;

  function engineFont(emPx) {
    return `bold ${emPx}px "Roboto Mono", ui-monospace, monospace`;
  }

  // The engine's draw_str_centered (engine/base/render.rs; draw_text_centered is its one-character case): a text
  // box of 0.6 x fontHeight per character (computed in f32, as the engine does), centered on the point.
  // `color` is [r, g, b] or a CSS color.
  function drawEngineText({ context, text, centerX, centerY, fontHeight, color }) {
    const characterWidth = Math.trunc(Math.fround(Math.fround(fontHeight) * Math.fround(0.6)));
    const left = centerX - Math.trunc((characterWidth * text.length) / 2);
    const top = centerY - Math.trunc(Math.trunc(fontHeight) / 2);
    context.font = engineFont(fontHeight * EM_PER_ENGINE_HEIGHT);
    context.textAlign = "left";
    context.textBaseline = "alphabetic";
    context.fillStyle = Array.isArray(color) ? rgb(color) : color;
    context.fillText(text, left, top + fontHeight * ASCENT_PER_ENGINE_HEIGHT);
  }

  // A fixed yes/no variant of an existing renderer. Certificate answers are accepted from any board state,
  // exactly as in GameEpisode; only the correct answer changes the state and completes the task.
  function certificateGame({ base, goal, help }) {
    return {
      goal,
      help,
      initialState: base.initialState,
      apply({ puzzle, state, action }) {
        if (action === "yes" || action === "no") {
          return action === puzzle.answer ? { ...state, answer: action } : null;
        }
        return base.apply({ puzzle, state, action });
      },
      isSolved: ({ puzzle, state }) => state.answer === puzzle.answer,
      render: base.render,
      mount(args) {
        const mounted = base.mount?.(args) || {};
        const group = document.createElement("div");
        group.className = "play-group";
        const label = document.createElement("span");
        label.className = "play-group-label";
        label.textContent = "Answer";
        group.append(label);
        for (const answer of ["yes", "no"]) {
          const text = answer[0].toUpperCase() + answer.slice(1);
          group.append(button({ label: text, title: answer, onClick: () => args.act(answer) }));
        }
        args.controls.append(group);
        return mounted;
      },
    };
  }

  window.PuzzleKit = { RENDER_SIZE, button, keyDirection, directionPad, fitToResolution, rgb, engineFont, drawEngineText, certificateGame };
  window.PuzzleGames = window.PuzzleGames || {};
})();
