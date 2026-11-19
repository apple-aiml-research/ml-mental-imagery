/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2026 Apple Inc. All Rights Reserved.
 */
// Playable suite: clicking a wall tile swaps the wall for a player of that game. Each game's rules and
// renderer are ported to js/games/<game>.js; its puzzles (data/play/<game>.js) are exported from the Rust
// engine with their expert solutions. Both load on first play; the helpers the ports share are in
// js/play_kit.js.
//
// A game module registers itself as window.PuzzleGames[<game>] = {
//   goal:          one plain sentence on what solves the puzzle, shown above the controls.
//   help:          one short line on how to play, shown under the controls.
//   initialState({ puzzle }) -> state            puzzle = { seed, board, solution }; must not mutate puzzle.
//   apply({ puzzle, state, action }) -> state | null
//                  one action in the model's action language (the engine's action string), or null if the
//                  engine would reject it. Never mutates state.
//   isSolved({ puzzle, state }) -> boolean
//   render({ context, puzzle, state, ui })
//                  draws in the engine's 512 x 512 render space (the context is scaled to the canvas).
//   mount({ puzzle, canvas, controls, act, getState, ui, redraw, toLogical }) -> { onKey?, onStateChange?, destroy? }
//                  optional; builds the game's controls into `controls` and handles input. act(action)
//                  returns whether the move was legal; `ui` is a scratch object (selection, drag, ...)
//                  that render also receives; toLogical(pointerEvent) gives { x, y } in render space.
//                  onKey(event) returns true if it handled the key; onStateChange({ state }) runs after
//                  every move, undo, reset, and solution step.
//   comparableState?({ puzzle, state }) -> the state in the exporter's encode_state shape, if it differs
//                  (to check the port against engine trajectories).
// }
(function () {
  "use strict";

  const { GAMES, el } = window.Sheep;
  const { RENDER_SIZE, button } = window.PuzzleKit;
  const SOLUTION_STEP_MS = 650;
  const CONFETTI_COLORS = ["#4a3aa7", "#2a78d6", "#e08a1e", "#1f9d55", "#e8486b", "#f2c230"];
  const CONFETTI_PER_CANNON = 70;
  const CONFETTI_MS = 2800;
  const CONFETTI_GRAVITY = 0.3; // px per frame², at 60 frames per second
  const CONFETTI_FALL_SPEED = 2.6; // px per frame: the pieces flutter down slowly once they turn over

  // Confetti from the board's bottom corners, over the whole player: the reader solved the puzzle.
  function celebrate({ root, board }) {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const canvas = el("canvas", "player-confetti");
    root.append(canvas);
    const rootBounds = root.getBoundingClientRect();
    const boardBounds = board.getBoundingClientRect();
    const pixelRatio = window.devicePixelRatio || 1;
    canvas.width = Math.round(rootBounds.width * pixelRatio);
    canvas.height = Math.round(rootBounds.height * pixelRatio);
    const context = canvas.getContext("2d");
    const cannons = [
      { x: boardBounds.left - rootBounds.left, y: boardBounds.bottom - rootBounds.top, direction: 1 },
      { x: boardBounds.right - rootBounds.left, y: boardBounds.bottom - rootBounds.top, direction: -1 },
    ];
    const pieces = cannons.flatMap(({ x, y, direction }) => Array.from({ length: CONFETTI_PER_CANNON }, () => {
      const angle = (Math.PI / 180) * (50 + Math.random() * 30); // up and in, 50-80 degrees above the horizontal
      const speed = 17 + Math.random() * 9;
      return {
        x,
        y,
        velocityX: Math.cos(angle) * speed * direction,
        velocityY: -Math.sin(angle) * speed,
        rotation: Math.random() * Math.PI * 2,
        spin: (Math.random() - 0.5) * 0.35,
        flutter: Math.random() * Math.PI * 2,
        width: 7 + Math.random() * 5,
        height: 4 + Math.random() * 3,
        color: CONFETTI_COLORS[Math.floor(Math.random() * CONFETTI_COLORS.length)],
      };
    }));
    const startTime = performance.now();
    let previousTime = startTime;
    function frame(now) {
      const frames = Math.min(3, (now - previousTime) / (1000 / 60));
      previousTime = now;
      const elapsed = now - startTime;
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
      context.clearRect(0, 0, rootBounds.width, rootBounds.height);
      context.globalAlpha = Math.min(1, Math.max(0, (CONFETTI_MS - elapsed) / (CONFETTI_MS * 0.35)));
      for (const piece of pieces) {
        piece.velocityY = Math.min(CONFETTI_FALL_SPEED, piece.velocityY + CONFETTI_GRAVITY * frames);
        piece.velocityX *= Math.pow(0.97, frames);
        piece.x += piece.velocityX * frames;
        piece.y += piece.velocityY * frames;
        piece.rotation += piece.spin * frames;
        piece.flutter += 0.2 * frames;
        context.save();
        context.translate(piece.x, piece.y);
        context.rotate(piece.rotation);
        context.scale(1, Math.cos(piece.flutter)); // tumbling: the piece turns edge-on and back
        context.fillStyle = piece.color;
        context.fillRect(-piece.width / 2, -piece.height / 2, piece.width, piece.height);
        context.restore();
      }
      if (elapsed < CONFETTI_MS) {
        requestAnimationFrame(frame);
        return;
      }
      canvas.remove();
      board.classList.remove("is-celebrating"); // it would otherwise outrank the board's .is-blocked flash
    }
    requestAnimationFrame(frame);
    board.classList.remove("is-celebrating");
    void board.offsetWidth; // restart the glow
    board.classList.add("is-celebrating");
  }

  const loadedScripts = new Map();
  function loadScript(src) {
    if (!loadedScripts.has(src)) {
      loadedScripts.set(src, new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = src;
        script.onload = resolve;
        script.onerror = () => {
          loadedScripts.delete(src);
          reject(new Error(`Couldn't load ${src}`));
        };
        document.head.appendChild(script);
      }));
    }
    return loadedScripts.get(src);
  }

  const RENDERERS = {
    mental_rotation_certificate: "mental_rotation",
    mental_rotation_2d_certificate: "mental_rotation_2d",
    tangram_certificate: "tangram",
  };

  function loadGame(gameKey) {
    const renderer = RENDERERS[gameKey];
    const module = renderer
      ? loadScript(`js/games/${renderer}.js`).then(() => loadScript(`js/games/${gameKey}.js`))
      : loadScript(`js/games/${gameKey}.js`);
    return Promise.all([module, loadScript(`data/play/${gameKey}.js`)]).then(() => ({
      module: window.PuzzleGames[gameKey],
      puzzles: window.PLAY_PUZZLES[gameKey],
    }));
  }

  function plainChip(action) {
    const chip = el("span", "token-action");
    chip.append(el("span", "token-piece", action));
    return chip;
  }

  function isTyping(event) {
    const target = event.target;
    return target instanceof HTMLElement && (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName));
  }

  // Opens the player in `host` on puzzle `puzzleIndex`; `onClose` runs when the reader goes back to the wall.
  function openPlayer({ host, gameKey, skill, puzzleIndex, onClose }) {
    const root = el("div", "player is-loading");
    const head = el("div", "player-head");
    const backButton = button({ label: "← All puzzles", className: "player-back", onClick: close });
    const title = el("div", "player-title");
    title.append(el("b", null, GAMES[gameKey].name), el("span", "wall-skill", skill));
    const nav = el("div", "player-nav");
    const previousButton = button({ label: "‹", title: "Previous puzzle", onClick: () => showPuzzle(currentIndex - 1) });
    const counter = el("span", "player-counter");
    const nextButton = button({ label: "›", title: "Next puzzle", onClick: () => showPuzzle(currentIndex + 1) });
    nav.append(previousButton, counter, nextButton);
    head.append(backButton, title, nav);

    const body = el("div", "player-body");
    const board = el("div", "player-board");
    const canvas = document.createElement("canvas");
    canvas.setAttribute("aria-label", `${GAMES[gameKey].name} board`);
    const banner = el("div", "player-banner");
    board.append(canvas, banner);

    const side = el("div", "player-side");
    const goal = el("p", "player-goal");
    const controls = el("div", "player-controls");
    const help = el("p", "player-help");
    const tapeHead = el("div", "player-tape-head");
    const tapeLabel = el("span", null, "Your actions");
    const moveCount = el("span", "player-move-count");
    tapeHead.append(tapeLabel, moveCount);
    const tape = el("div", "player-tape");
    const buttons = el("div", "player-buttons");
    const undoButton = button({ label: "Undo", title: "Undo (Z)", onClick: undo });
    const resetButton = button({ label: "Reset", onClick: reset });
    const solutionButton = button({ label: "Show solution", onClick: toggleSolution });
    const nextPuzzleButton = button({ label: "Next puzzle →", className: "is-primary player-next", onClick: () => showPuzzle(currentIndex + 1) });
    buttons.append(undoButton, resetButton, solutionButton, nextPuzzleButton);
    const status = el("p", "player-status");
    status.setAttribute("aria-live", "polite");
    side.append(goal, controls, help, tapeHead, tape, buttons, status);
    body.append(board, side);
    root.append(head, body);
    host.append(root);

    let game = null;
    let puzzles = [];
    let currentIndex = puzzleIndex;
    let puzzle = null;
    let history = []; // states, history[0] = the starting board
    let actions = [];
    let ui = {};
    let mounted = {};
    let solutionTimer = null;
    let isShowingSolution = false;
    let isClosed = false;

    const state = () => history[history.length - 1];
    const solved = () => game.isSolved({ puzzle, state: state() });

    function toLogical(event) {
      const bounds = canvas.getBoundingClientRect();
      return {
        x: ((event.clientX - bounds.left) / bounds.width) * RENDER_SIZE,
        y: ((event.clientY - bounds.top) / bounds.height) * RENDER_SIZE,
      };
    }

    // Games draw at a whole multiple of the render size (cell edges stay on pixel boundaries, so no seams),
    // which is then resampled onto the canvas.
    const offscreen = document.createElement("canvas");
    function redraw() {
      const pixelSize = Math.round(canvas.getBoundingClientRect().width * (window.devicePixelRatio || 1));
      if (pixelSize > 0 && canvas.width !== pixelSize) {
        canvas.width = pixelSize;
        canvas.height = pixelSize;
      }
      const scale = Math.max(1, Math.ceil(canvas.width / RENDER_SIZE));
      if (offscreen.width !== RENDER_SIZE * scale) {
        offscreen.width = RENDER_SIZE * scale;
        offscreen.height = RENDER_SIZE * scale;
      }
      const offscreenContext = offscreen.getContext("2d");
      offscreenContext.setTransform(1, 0, 0, 1, 0, 0);
      offscreenContext.clearRect(0, 0, offscreen.width, offscreen.height);
      offscreenContext.setTransform(scale, 0, 0, scale, 0, 0);
      game.render({ context: offscreenContext, puzzle, state: state(), ui });
      const context = canvas.getContext("2d");
      context.imageSmoothingQuality = "high";
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.drawImage(offscreen, 0, 0, canvas.width, canvas.height);
    }

    function flashBlocked() {
      board.classList.remove("is-blocked");
      void board.offsetWidth; // restart the animation
      board.classList.add("is-blocked");
    }

    // Keeps the tokens the tape already shows (so only new ones fade in) and drops the ones undone.
    function syncTape() {
      const tokens = [...tape.querySelectorAll(".tape-token")];
      let keptCount = 0;
      while (keptCount < tokens.length && keptCount < actions.length && tokens[keptCount].dataset.action === actions[keptCount]) keptCount += 1;
      tokens.slice(keptCount).forEach((token) => token.remove());
      actions.slice(keptCount).forEach((action) => {
        const token = el("span", "tape-token");
        token.dataset.action = action;
        token.append(plainChip(action));
        tape.append(token);
      });
      tape.querySelectorAll(".tape-token").forEach((token, index) => token.classList.toggle("is-current", index === actions.length - 1));
      tape.scrollTop = tape.scrollHeight;
    }

    function refresh() {
      const isSolved = solved();
      syncTape();
      tapeLabel.textContent = isShowingSolution || root.classList.contains("is-solution") ? "Expert solution" : "Your actions";
      moveCount.textContent = actions.length ? `${actions.length}` : "";
      undoButton.disabled = isShowingSolution || history.length < 2;
      resetButton.disabled = isShowingSolution || history.length < 2;
      root.classList.toggle("is-solved", isSolved);
      if (isSolved && !isShowingSolution && !root.classList.contains("is-solution")) {
        const expertCount = puzzle.solution.length;
        banner.textContent = "Solved!";
        const comparison = actions.length < expertCount ? `fewer than the expert’s ${expertCount}`
          : actions.length === expertCount ? "as few as the expert" : `the expert needs ${expertCount}`;
        status.textContent = `Solved in ${actions.length} actions; ${comparison}.`;
      } else if (isSolved) {
        banner.textContent = "Solved";
        status.textContent = `The expert solves it in ${puzzle.solution.length} actions.`;
      } else {
        banner.textContent = "";
      }
      mounted.onStateChange?.({ state: state() });
      redraw();
    }

    // Applies an action from the reader (or the solution playback); false if it's illegal or the game is over.
    function act(action, { fromSolution = false } = {}) {
      if (!game || (isShowingSolution && !fromSolution) || solved()) return false;
      const next = game.apply({ puzzle, state: state(), action });
      if (!next) {
        flashBlocked();
        status.textContent = `${action} isn't a legal move here.`;
        return false;
      }
      if (!fromSolution) root.classList.remove("is-solution");
      history.push(next);
      actions.push(action);
      status.textContent = "";
      refresh();
      if (!fromSolution && solved()) celebrate({ root, board });
      return true;
    }

    function undo() {
      if (isShowingSolution || history.length < 2) return;
      history.pop();
      actions.pop();
      root.classList.remove("is-solution");
      status.textContent = "";
      refresh();
    }

    function reset() {
      stopSolution();
      history = [game.initialState({ puzzle })];
      actions = [];
      root.classList.remove("is-solution");
      status.textContent = "";
      refresh();
    }

    function stopSolution() {
      clearTimeout(solutionTimer);
      if (!isShowingSolution) return;
      isShowingSolution = false;
      root.classList.remove("is-playing-solution");
      solutionButton.textContent = "Show solution";
    }

    function toggleSolution() {
      if (isShowingSolution) {
        stopSolution();
        refresh();
        return;
      }
      reset();
      isShowingSolution = true;
      root.classList.add("is-playing-solution", "is-solution");
      solutionButton.textContent = "Stop";
      status.textContent = "";
      let stepIndex = 0;
      function step() {
        if (stepIndex >= puzzle.solution.length) {
          stopSolution();
          refresh();
          return;
        }
        act(puzzle.solution[stepIndex], { fromSolution: true });
        stepIndex += 1;
        solutionTimer = setTimeout(step, SOLUTION_STEP_MS);
      }
      refresh();
      solutionTimer = setTimeout(step, SOLUTION_STEP_MS / 2);
    }

    function showPuzzle(index) {
      stopSolution();
      mounted.destroy?.();
      currentIndex = (index + puzzles.length) % puzzles.length;
      puzzle = puzzles[currentIndex];
      counter.textContent = `Puzzle ${currentIndex + 1} of ${puzzles.length}`;
      history = [game.initialState({ puzzle })];
      actions = [];
      ui = {};
      status.textContent = "";
      root.classList.remove("is-solution");
      controls.replaceChildren();
      goal.textContent = game.goal || "";
      help.textContent = game.help || "";
      mounted = game.mount?.({
        puzzle,
        canvas,
        controls,
        act: (action) => act(action),
        getState: state,
        ui,
        redraw,
        toLogical,
      }) || {};
      refresh();
    }

    function onKeyDown(event) {
      if (!game || isTyping(event)) return;
      if (event.key === "Escape") {
        close();
        return;
      }
      if (mounted.onKey?.(event)) {
        event.preventDefault();
        return;
      }
      if ((event.key === "z" || event.key === "Z") && !event.altKey) {
        event.preventDefault();
        undo();
      }
    }

    const resizeObserver = new ResizeObserver(() => game && redraw());
    resizeObserver.observe(board);
    document.addEventListener("keydown", onKeyDown);

    function close() {
      if (isClosed) return;
      isClosed = true;
      stopSolution();
      mounted.destroy?.();
      resizeObserver.disconnect();
      document.removeEventListener("keydown", onKeyDown);
      root.remove();
      onClose();
    }

    loadGame(gameKey).then(({ module, puzzles: loadedPuzzles }) => {
      if (isClosed) return;
      game = module;
      puzzles = loadedPuzzles;
      root.classList.remove("is-loading");
      showPuzzle(currentIndex);
    }).catch((error) => {
      root.classList.remove("is-loading");
      status.textContent = `This game couldn't load (${error.message}).`;
    });

    return { close };
  }

  window.SheepPlay = { openPlayer };
})();
