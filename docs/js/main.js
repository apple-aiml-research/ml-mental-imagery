/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2026 Apple Inc. All Rights Reserved.
 */
// Project page: hero, puzzle wall, and results (curated rollouts: stories.js).
(function () {
  "use strict";

  const { GAMES, el, actionChip, tokenTape } = window.Sheep;
  const SUITE_SOLUTIONS = window.SUITE_SOLUTIONS;

  // ---------------------------------------------------------------------------
  // Hero: a one-shot animation (data-scene: intro -> thinking -> answered).
  //   intro:    the question and the two sheep, large.
  //   thinking: they shrink up, the VLM's output stream slides in, and the left sheep becomes
  //             a thought bubble that rotates through the VLM's imagined poses, seen through a gentle
  //             refraction (hero_refraction.js) where WebGL can draw them, else as plain images.
  //   answered: the answer appears, and rings (style.css) mark the wool puff that gives the mirror away;
  //             a countdown ring fills, then it replays (or on click).
  // A mirror pair. Each imagined pose is the VLM's hidden state after a rotation, decoded into an image;
  // after the last rotation it answers "no".
  // The steps (each action and its imagined pose) are written into index.html, so the hero's first paint
  // already has its final layout.
  // ---------------------------------------------------------------------------

  const INTRO_MS = 2400;
  const TRANSITION_MS = 1700; // prompt flies into the row, then the VLM appears (style.css delays)
  const STEP_MS = 1400;
  const IMAGE_DELAY_MS = 350; // each image follows its action token
  const COUNTDOWN_DELAY_MS = 500; // matches the replay button's fade-in delay in style.css
  const COUNTDOWN_MS = 6000;

  // Writes a chip out one token at a time, evenly over durationMs: its clip steps to the end of each token
  // in turn. Clipping rather than hiding keeps the chip's width, which scrollStream measures, constant.
  function typeTokens({ chip, durationMs }) {
    const pieces = [...chip.children];
    if (pieces.length < 2 || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const chipRight = chip.getBoundingClientRect().right;
    const radius = getComputedStyle(chip).borderTopLeftRadius;
    const keyframes = pieces.map((piece, index) => {
      const hiddenWidth = index === pieces.length - 1 ? 0 : chipRight - piece.getBoundingClientRect().right - 1;
      return { offset: index / pieces.length, clipPath: `inset(0 ${hiddenWidth}px 0 0 round ${radius} 0 0 ${radius})`, easing: "step-end" };
    });
    keyframes.push({ ...keyframes[keyframes.length - 1], offset: 1 });
    chip.animate(keyframes, { duration: durationMs });
  }

  function setupHeroScene() {
    const scene = document.getElementById("hero-scene");
    const leftCard = document.getElementById("scene-left");
    const leftImage = document.getElementById("scene-left-image");
    const leftInput = leftImage.querySelector(".scene-input");
    const stream = document.getElementById("token-stream");
    const track = stream.querySelector(".stream-track");
    const steps = [...track.querySelectorAll(".stream-step")];
    const frames = steps.map((step) => {
      // The chip is rebuilt from data/action_tokens.js; the page's copy only holds the layout until then.
      step.querySelector(".token-action").replaceWith(actionChip(step.dataset.action));
      const frame = el("img", "scene-frame");
      frame.src = step.querySelector(".token-image").getAttribute("src");
      frame.alt = "";
      leftImage.appendChild(frame);
      return frame;
    });
    const answerGroup = track.querySelector(".answer-group");
    const refraction = window.matchMedia("(prefers-reduced-motion: reduce)").matches
      ? null
      : window.SheepRefraction.create({ host: leftImage, poseImages: frames });

    scene.style.setProperty("--countdown-ms", `${COUNTDOWN_MS}ms`);

    // In the intro, the prompt sits large and centered under the (enlarged) sheep; its final spot is
    // the start of the output row. Offsets ignore transforms, so they give the final layout.
    const prompt = document.getElementById("scene-prompt");
    const sheepRow = scene.querySelector(".scene-sheep");
    const PROMPT_GAP_PX = 22;
    function placePromptForIntro() {
      scene.classList.add("is-placing");
      const sceneStyle = getComputedStyle(scene);
      const sheepScale = parseFloat(sceneStyle.getPropertyValue("--intro-sheep-scale"));
      const sheepShift = parseFloat(sceneStyle.getPropertyValue("--intro-sheep-shift"));
      const promptScale = parseFloat(sceneStyle.getPropertyValue("--intro-prompt-scale"));
      const promptCenterX = prompt.offsetLeft + prompt.offsetWidth / 2;
      const promptCenterY = prompt.offsetTop + prompt.offsetHeight / 2;
      const introSheepBottom = sheepRow.offsetTop + sheepRow.offsetHeight / 2 + sheepShift + (sheepScale * sheepRow.offsetHeight) / 2;
      const introPromptCenterY = introSheepBottom + PROMPT_GAP_PX + (promptScale * prompt.offsetHeight) / 2;
      prompt.style.setProperty("--intro-dx", `${scene.clientWidth / 2 - promptCenterX}px`);
      prompt.style.setProperty("--intro-dy", `${introPromptCenterY - promptCenterY}px`);
      void prompt.offsetWidth; // apply the new position before transitions come back on
      scene.classList.add("is-placed");
      requestAnimationFrame(() => requestAnimationFrame(() => scene.classList.remove("is-placing")));
    }
    placePromptForIntro();
    window.addEventListener("resize", placePromptForIntro);
    document.fonts.ready.then(placePromptForIntro);

    let timers = [];
    function at(delayMs, action) {
      timers.push(setTimeout(action, delayMs));
    }

    // Don't replay while scrolled away; replay once the hero is back on screen.
    let heroVisible = true;
    let replayPending = false;
    new IntersectionObserver(([entry]) => {
      heroVisible = entry.isIntersecting;
      if (heroVisible && replayPending) play();
    }).observe(scene);

    function autoReplay() {
      if (heroVisible) play();
      else replayPending = true;
    }

    // The stream is a tape: once the steps outgrow the row, they slide left to keep the newest in view.
    // Offsets ignore transforms, so they measure the final layout even mid-animation.
    let newestStreamItem = null;
    function scrollStream(item) {
      newestStreamItem = item;
      const visibleWidth = stream.clientWidth - parseFloat(getComputedStyle(stream).paddingRight);
      const overflow = item ? Math.max(0, item.offsetLeft + item.offsetWidth - visibleWidth) : 0;
      stream.classList.toggle("is-overflowing", overflow > 0);
      track.style.transform = `translateX(${-overflow}px)`;
    }
    window.addEventListener("resize", () => scrollStream(newestStreamItem));

    // Refracted once the refraction is ready (it may not be, or may stop being), else the plain frame.
    function showPose(poseIndex) {
      const refracted = Boolean(refraction?.isReady);
      frames.forEach((frame, index) => frame.classList.toggle("is-visible", !refracted && index === poseIndex));
      if (refracted) refraction.showPose(poseIndex);
      else refraction?.hide();
    }

    function showStep(stepIndex) {
      steps.forEach((step, index) => {
        step.classList.toggle("is-visible", index <= stepIndex);
        step.classList.toggle("is-current", index === stepIndex);
      });
      scrollStream(steps[stepIndex]);
      typeTokens({ chip: steps[stepIndex].querySelector(".token-action"), durationMs: IMAGE_DELAY_MS });
      at(IMAGE_DELAY_MS, () => {
        leftInput.classList.add("is-hidden");
        showPose(stepIndex);
      });
    }

    function showAnswer() {
      steps.forEach((step) => step.classList.remove("is-current"));
      answerGroup.classList.add("is-visible");
      scrollStream(answerGroup);
      scene.dataset.scene = "answered";
    }

    function reset() {
      timers.forEach(clearTimeout);
      timers = [];
      replayPending = false;
      steps.forEach((step) => step.classList.remove("is-visible", "is-current"));
      frames.forEach((frame) => frame.classList.remove("is-visible"));
      refraction?.hide();
      leftInput.classList.remove("is-hidden");
      leftCard.classList.remove("is-thought");
      answerGroup.classList.remove("is-visible");
      scrollStream(null);
      scene.dataset.scene = "intro";
    }

    function play() {
      reset();
      at(INTRO_MS, () => { scene.dataset.scene = "thinking"; });
      const thinkingStart = INTRO_MS + TRANSITION_MS;
      at(thinkingStart, () => leftCard.classList.add("is-thought"));
      steps.forEach((_, stepIndex) => at(thinkingStart + stepIndex * STEP_MS, () => showStep(stepIndex)));
      const answerAt = thinkingStart + steps.length * STEP_MS;
      at(answerAt, showAnswer);
      at(answerAt + COUNTDOWN_DELAY_MS + COUNTDOWN_MS, autoReplay);
    }

    function showFinalState() {
      reset();
      scene.dataset.scene = "thinking";
      leftCard.classList.add("is-thought");
      steps.forEach((step) => step.classList.add("is-visible"));
      leftInput.classList.add("is-hidden");
      showPose(frames.length - 1);
      showAnswer();
    }

    document.getElementById("scene-replay").addEventListener("click", play);
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) showFinalState();
    else play();
  }

  // ---------------------------------------------------------------------------
  // Puzzle wall: each tile plays an expert solution rendered from the game engine, with the expert's
  // action tokens streaming in sync (data/suite.js). "New puzzles" moves
  // every tile to another instance. Clicking a tile swaps the wall for a playable port of the game,
  // opened on the puzzle the tile shows (js/play.js).
  // ---------------------------------------------------------------------------

  // The reference figure's task order: certificate variants sit beside their base games.
  const WALL = [
    { game: "mental_rotation_2d", skill: "Rotation" },
    { game: "mental_rotation_2d_certificate", skill: "Rotation", clipFormat: "webm" },
    { game: "mental_rotation", skill: "Rotation" },
    { game: "mental_rotation_certificate", skill: "Rotation", clipFormat: "webm" },
    { game: "jigsaw", skill: "Pieces" },
    { game: "character_recognition", skill: "Text" },
    { game: "tangram", skill: "Pieces" },
    { game: "tangram_certificate", skill: "Pieces", clipFormat: "webm" },
    { game: "bloxorz", skill: "Sliding" },
    { game: "rush_hour", skill: "Sliding" },
    { game: "sokoban", skill: "Sliding" },
    { game: "anagram", skill: "Text" },
  ];
  const WALL_HOLD_MS = 2000;
  const TOKEN_LEAD = 0.4; // a token appears this fraction of a frame before the board shows its effect

  function buildWallTile(entry, onPlay) {
    const tile = el("figure", "wall-tile");
    tile.tabIndex = 0;
    tile.setAttribute("role", "button");
    tile.setAttribute("aria-label", `Play ${GAMES[entry.game].name}`);
    tile.addEventListener("click", onPlay);
    tile.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      onPlay();
    });
    const board = el("div", "wall-board");
    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.preload = "none";
    video.setAttribute("aria-label", `${GAMES[entry.game].name}: expert solution`);
    board.append(video, el("span", "wall-play", "▶ Play"));
    const tape = tokenTape();
    const caption = el("figcaption");
    caption.append(el("b", null, GAMES[entry.game].name), el("span", "wall-skill", entry.skill));
    tile.append(board, tape.element, caption);

    let solution = null;
    let clipPath = null;
    let isNear = false; // the poster and clip load only once the tile nears the screen
    let shownTokens = 0;
    let visible = false;
    let holdTimer = null;

    // Token i is the action that produces frame i + 1.
    function sync() {
      const framePosition = video.currentTime * solution.fps + TOKEN_LEAD;
      while (shownTokens < solution.actions.length && framePosition >= shownTokens + 1) {
        const token = el("span", "tape-token");
        token.append(actionChip(solution.actions[shownTokens]));
        tape.push(token);
        shownTokens += 1;
      }
      if (!video.paused && !video.ended) requestAnimationFrame(sync);
    }

    function restart() {
      clearTimeout(holdTimer);
      tape.clear();
      shownTokens = 0;
      video.currentTime = 0;
      if (visible) video.play().catch(() => {});
    }

    function loadClip() {
      if (!isNear) return;
      video.poster = `${clipPath}.webp`;
      video.src = `${clipPath}.${entry.clipFormat || "mp4"}`;
      restart();
    }

    video.addEventListener("play", () => requestAnimationFrame(sync));
    video.addEventListener("ended", () => {
      sync();
      tape.settle();
      holdTimer = setTimeout(restart, WALL_HOLD_MS);
    });

    const nearObserver = new IntersectionObserver(([observed]) => {
      if (!observed.isIntersecting) return;
      nearObserver.disconnect();
      isNear = true;
      loadClip();
    }, { rootMargin: "600px 0px" });
    nearObserver.observe(tile);

    new IntersectionObserver(([observed]) => {
      visible = observed.isIntersecting;
      if (visible && video.ended) restart();
      else if (visible) video.play().catch(() => {});
      else {
        clearTimeout(holdTimer);
        video.pause();
      }
    }, { threshold: 0.25 }).observe(tile);

    return {
      element: tile,
      show(instanceIndex) {
        solution = SUITE_SOLUTIONS[entry.game][instanceIndex];
        clipPath = `media/suite/${entry.game}-${instanceIndex}`;
        loadClip();
      },
    };
  }

  function buildWall() {
    const section = document.getElementById("suite");
    const wall = document.getElementById("puzzle-wall");
    const shuffleButton = document.getElementById("wall-shuffle");
    const instanceCount = Math.min(...WALL.map((entry) => SUITE_SOLUTIONS[entry.game].length));
    let instanceIndex = 0;

    // The exported puzzles start with the wall's instances, in order, so the player opens on the tile's puzzle.
    function play(entry, tile) {
      section.classList.add("is-playing");
      window.SheepPlay.openPlayer({
        host: section,
        gameKey: entry.game,
        skill: entry.skill,
        puzzleIndex: instanceIndex,
        onClose() {
          section.classList.remove("is-playing");
          tile.element.focus({ preventScroll: true });
          if (section.getBoundingClientRect().top < 0) section.scrollIntoView({ behavior: "smooth" });
        },
      });
      if (section.getBoundingClientRect().top < 0) section.scrollIntoView({ behavior: "smooth" });
    }

    const tiles = WALL.map((entry) => {
      const tile = buildWallTile(entry, () => play(entry, tile));
      return tile;
    });
    tiles.forEach((tile) => wall.append(tile.element));
    tiles.forEach((tile) => tile.show(instanceIndex));
    shuffleButton.addEventListener("click", () => {
      instanceIndex = (instanceIndex + 1) % instanceCount;
      tiles.forEach((tile) => tile.show(instanceIndex));
    });
  }

  // ---------------------------------------------------------------------------
  // Results: the paper's solve rates of Qwen3.5 9B, on average and per task, in the paper's colors.
  // ---------------------------------------------------------------------------

  const RESULT_METHODS = [
    { name: "Base", color: "#bdc3c7" },
    { name: "Action Supervision", color: "#95a5a6" },
    { name: "Visual Supervision", color: "#16a085" },
    { name: "State CoT", color: "#60717b" },
    { name: "Boxes CoT", color: "#2c3e50" },
    { name: "Visual CoT 64×64", color: "#f39c12" },
    { name: "Visual CoT 128×128", color: "#c0392b" },
  ];

  const RESULT_AVERAGES = [12.5, 82.9, 86.3, 87.3, 86.7, 88.5, 88.9];

  // Per task, in RESULT_METHODS order.
  const RESULT_GAMES = [
    { game: "mental_rotation_2d", rates: [4.5, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0] },
    { game: "mental_rotation_2d_certificate", rates: [44.5, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0] },
    { game: "mental_rotation", rates: [0.0, 87.5, 87.0, 95.5, 88.5, 91.0, 97.0] },
    { game: "mental_rotation_certificate", rates: [48.5, 49.5, 60.0, 56.5, 72.5, 72.0, 75.0] },
    { game: "bloxorz", rates: [0.0, 96.0, 96.0, 98.0, 97.5, 97.5, 94.0] },
    { game: "rush_hour", rates: [0.0, 94.0, 93.0, 97.5, 93.5, 95.0, 97.0] },
    { game: "character_recognition", rates: [0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0] },
    { game: "jigsaw", rates: [0.0, 65.0, 92.5, 62.5, 61.5, 84.5, 92.0] },
    { game: "tangram", rates: [0.0, 55.5, 54.0, 71.5, 65.0, 59.0, 55.0] },
    { game: "tangram_certificate", rates: [43.0, 65.0, 65.0, 71.0, 71.0, 68.0, 67.0] },
    { game: "sokoban", rates: [0.0, 85.0, 89.5, 95.0, 93.0, 94.5, 90.0] },
    { game: "anagram", rates: [9.5, 97.5, 98.0, 99.5, 98.0, 100.0, 100.0] },
  ];

  function formatRate(value) {
    return Number.isInteger(value) ? `${value}` : value.toFixed(1);
  }

  function bar({ method, value, label, tooltip }) {
    const row = el("div", label ? "bar-row" : "bar-row is-compact");
    const track = el("div", "bar-track");
    const fill = el("div", "bar");
    fill.style.width = `${value}%`;
    fill.style.background = method.color;
    const valueLabel = el("span", "bar-value", formatRate(value));
    valueLabel.style.left = `calc(${value}% + 6px)`;
    track.append(fill, valueLabel);
    if (label) row.append(el("span", "bar-label", method.name));
    row.append(track, el("div", "bar-tooltip", tooltip));
    return row;
  }

  function buildResults() {
    const host = document.getElementById("results-average");
    host.appendChild(el("h3", "chart-title", "Average solve rate (%)"));
    RESULT_METHODS.forEach((method, index) => {
      const value = RESULT_AVERAGES[index];
      host.appendChild(bar({ method, value, label: true, tooltip: `${method.name}: ${formatRate(value)}% averaged over 12 tasks` }));
    });

    const games = document.getElementById("results-games");
    const legend = el("p", "chart-legend");
    RESULT_METHODS.forEach((method) => {
      const swatch = el("span", "swatch");
      swatch.style.background = method.color;
      legend.append(swatch, method.name);
    });
    games.before(legend);
    RESULT_GAMES.forEach(({ game, rates }) => {
      const card = el("div", "result-game");
      card.appendChild(el("h4", "result-game-title", GAMES[game].name));
      RESULT_METHODS.forEach((method, index) => {
        card.appendChild(bar({ method, value: rates[index], tooltip: `${method.name}: ${formatRate(rates[index])}%` }));
      });
      games.appendChild(card);
    });
  }

  setupHeroScene();
  buildWall();
  buildResults();
})();
