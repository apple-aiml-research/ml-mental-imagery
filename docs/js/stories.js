/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2026 Apple Inc. All Rights Reserved.
 */
// Animated curated rollouts, built from the actions (data/stories.js) and frame strips
// (media/stories/). Three layouts share one engine:
//   - story stage: one rollout at a time, in the hero's visual language. The input becomes the thought
//     bubble (imagination), beside reality, which the model never sees; prompt -> VLM -> tokens;
//   - mechanism cards: compact stages showing how each sharpening method produces its imagery;
//   - head to head: the three methods in lockstep lanes on the same puzzle.
(function () {
  "use strict";

  const { GAMES, el, actionChip, shiftTokens, tokenTape } = window.Sheep;
  const ROLLOUTS = window.STORY_ROLLOUTS;

  const METHOD_INFO = {
    action_supervision: { name: "Action Supervision", link: "frozen", linkLabel: "frozen probe" },
    visual_supervision: { name: "Visual Supervision", link: "trained", linkLabel: "trained probe" },
    visual_cot: { name: "Visual CoT", link: "tokens", linkLabel: "visual tokens" },
  };
  // Where the imagination comes from. A probe reads the VLM's hidden states, at every layer, at the token
  // that closes each action (mental_imagery/model/visual_head.py), so its line leaves the side of the VLM; visual tokens are
  // part of the output, so their decode line rises from the token, which stops under the bubble.
  const PROBE_INFO = {
    frozen: { anchor: ".token-action", label: "frozen probe" },
    trained: { anchor: ".token-action", label: "trained probe" },
    tokens: { anchor: ".token-visual", label: "decode" },
  };
  const LANE_METHODS = Object.keys(METHOD_INFO);

  const ROLLOUT_MS = 9000; // target length of one rollout's animation
  const MIN_STEP_MS = 240;
  const MAX_STEP_MS = 1000;
  const INTRO_MS = 1300; // the input, before the model starts thinking
  const MORPH_MS = 700; // the input turning into the thought bubble, before the first token
  const FRAME_DELAY = 0.4; // fraction of a step after its token appears before the boards update
  const CROSSFADE_MAX_MS = 350; // an imagined frame dissolves into the next over at most this, or half a step
  const HOLD_MS = 3500;
  // Frames per strip row; longer rollouts wrap onto more rows.
  const STRIP_COLUMNS = 32;

  // ---------------------------------------------------------------------------
  // Data
  // ---------------------------------------------------------------------------

  function stripPath({ story, method, kind }) {
    return `media/stories/${story.game}-${story.puzzle}-${method}-${kind}.webp`;
  }

  // Moves (an answer becomes the outcome), and the frame each board shows after each move.
  function rolloutTimeline({ story, method }) {
    const rollout = ROLLOUTS[`${story.game}/${story.puzzle}/${method}`];
    const answerStep = rollout.steps.find((step) => step.answer);
    return {
      rollout,
      moves: rollout.steps.filter((step) => !step.answer),
      answer: answerStep ? answerStep.action : null,
      realityFrame: (moveIndex) => Math.min(moveIndex + 1, rollout.realityFrames - 1),
      imaginationFrame: (moveIndex) => Math.min(moveIndex + 1, rollout.imaginationFrames - 1),
    };
  }

  function stepDuration(moveCount) {
    return Math.max(MIN_STEP_MS, Math.min(MAX_STEP_MS, Math.round(ROLLOUT_MS / Math.max(moveCount, 1))));
  }

  function answerText(answer) {
    const identified = /^identify\('(.*)'\)$/.exec(answer);
    return identified ? identified[1] : answer;
  }

  // Strips are fetched before a rollout plays, so its boards never show up blank.
  const imageLoads = new Map();

  function loadImage(src) {
    if (!imageLoads.has(src)) {
      const image = new Image();
      image.src = src;
      imageLoads.set(src, image.decode().catch(() => {}));
    }
    return imageLoads.get(src);
  }

  function loadStory({ story, methods }) {
    return Promise.all(methods.flatMap((method) =>
      ["reality", "imagination"].map((kind) => loadImage(stripPath({ story, method, kind })))));
  }

  // ---------------------------------------------------------------------------
  // Building blocks
  // ---------------------------------------------------------------------------

  // A strip image holds a rollout's frames left to right, in rows of up to STRIP_COLUMNS.
  function stripGrid(frameCount) {
    const columns = Math.min(frameCount, STRIP_COLUMNS);
    return { columns, rows: Math.ceil(frameCount / columns) };
  }

  // A view onto one frame of a strip image.
  function stripView({ className, src, frameCount }) {
    const view = el("div", className);
    const { columns, rows } = stripGrid(frameCount);
    view.style.backgroundImage = `url("${src}")`;
    view.style.backgroundSize = `${columns * 100}% ${rows * 100}%`;
    view.dataset.frames = String(frameCount);
    showStripFrame({ view, frameIndex: 0 });
    return view;
  }

  function clampedFrame({ view, frameIndex }) {
    return Math.max(0, Math.min(frameIndex, Number(view.dataset.frames) - 1));
  }

  function showStripFrame({ view, frameIndex }) {
    const { columns, rows } = stripGrid(Number(view.dataset.frames));
    const index = clampedFrame({ view, frameIndex });
    const column = index % columns;
    const row = Math.floor(index / columns);
    view.dataset.frameIndex = String(index);
    const x = columns > 1 ? (column / (columns - 1)) * 100 : 0;
    const y = rows > 1 ? (row / (rows - 1)) * 100 : 0;
    view.style.backgroundPosition = `${x}% ${y}%`;
  }

  // Restarts a one-shot CSS animation.
  function flash({ element, className }) {
    element.classList.remove(className);
    void element.offsetWidth;
    element.classList.add(className);
  }

  // Each new imagined frame dissolves in over the previous one: a crossfade, which stays smooth however
  // short the steps get (a per-frame fade or blur strobes on long rollouts).
  function developFrame({ view, frameIndex, stepMs }) {
    if (Number(view.dataset.frameIndex) === clampedFrame({ view, frameIndex })) return;
    view.querySelectorAll(".frame-previous").forEach((stale) => stale.remove());
    const previous = el("div", "frame-previous");
    previous.style.backgroundImage = view.style.backgroundImage;
    previous.style.backgroundSize = view.style.backgroundSize;
    previous.style.backgroundPosition = view.style.backgroundPosition;
    view.append(previous);
    showStripFrame({ view, frameIndex });
    previous.animate([{ opacity: 1 }, { opacity: 0 }], { duration: Math.min(CROSSFADE_MAX_MS, stepMs / 2), easing: "ease", fill: "forwards" })
      .finished.then(() => previous.remove(), () => previous.remove());
  }

  function tokenElement({ step, method, story, imaginationFrames, frameIndex }) {
    const token = el("span", "tape-token");
    token.append(actionChip(step.action));
    if (step.imagined) {
      token.classList.add("is-invalid");
      token.title = "Blocked on the real board: this move happened only in the model’s imagination (read off its imagined frames).";
    }
    if (method === "visual_cot") {
      const visual = stripView({
        className: "token-visual",
        src: stripPath({ story, method, kind: "imagination" }),
        frameCount: imaginationFrames,
      });
      showStripFrame({ view: visual, frameIndex });
      visual.title = "The visual tokens the model generated after this action, decoded";
      token.append(visual);
    }
    return token;
  }

  // The last token: the answer and whether it is right, or whether the actions solved the puzzle.
  function outcomeElement({ solved, answer }) {
    const outcome = el("span", `stage-outcome ${solved ? "is-good" : "is-bad"}`);
    if (answer) {
      const answerPill = el("span", "token-answer");
      const verdict = el("span", "outcome-mark", solved ? "✓" : "✗");
      verdict.title = solved ? "Correct" : "Wrong";
      answerPill.append(el("small", null, "Answer"), el("b", null, answerText(answer)), verdict);
      outcome.append(answerPill);
    } else {
      outcome.append(el("span", "outcome-verdict", solved ? "✓ Solved" : "✗ Unsolved"));
    }
    return outcome;
  }

  function promptChip(text) {
    const chip = el("div", "stage-prompt");
    chip.append(el("span", "prompt-label", "Prompt"), el("span", "stage-question", `“${text}”`));
    return chip;
  }

  // A token tape whose newest token stops at a fixed slot above the VLM (in a visual-token stage, also
  // under the imagination bubble); older tokens slide left, fading with age.
  const FADED_AGE = 4;

  function slotTape({ slotCenter, anchorSelector }) {
    const tape = el("div", "token-tape slot-tape");
    const track = el("div", "tape-track");
    tape.append(track);
    let newest = null;

    function align() {
      if (!newest || !tape.isConnected) return;
      // An answer token has no action or visual part: it is centered whole.
      const anchorBox = (newest.querySelector(anchorSelector) || newest).getBoundingClientRect();
      const trackBox = track.getBoundingClientRect();
      const anchorCenter = anchorBox.left - trackBox.left + anchorBox.width / 2;
      track.style.transform = `translateX(${slotCenter() - anchorCenter}px)`;
    }

    return {
      element: tape,
      push(token) {
        shiftTokens({
          track,
          place: () => {
            track.appendChild(token);
            newest = token;
            align();
          },
        });
        track.querySelectorAll(".is-current").forEach((current) => current.classList.remove("is-current"));
        token.classList.add("is-current");
        [...track.children].reverse().forEach((pastToken, age) => {
          pastToken.dataset.age = String(Math.min(age, FADED_AGE));
        });
      },
      settle() {
        if (newest) newest.classList.remove("is-current");
      },
      align,
    };
  }

  // Animates the read-out dot along a probe line toward the imagination, whichever way the layout draws it.
  function runDot({ dot, durationMs }) {
    const line = dot.parentElement;
    const property = line.offsetWidth > line.offsetHeight ? "left" : "bottom";
    const middle = { opacity: 1, offset: 0.5 };
    dot.animate([{ [property]: "0%", opacity: 0 }, middle, { [property]: "100%", opacity: 0 }], { duration: durationMs, easing: "ease-in-out" });
  }

  function makeTimers() {
    let handles = [];
    return {
      at(delayMs, action) { handles.push(setTimeout(action, delayMs)); },
      clear() { handles.forEach(clearTimeout); handles = []; },
    };
  }

  // ---------------------------------------------------------------------------
  // Stage: one rollout.
  // ---------------------------------------------------------------------------

  // The imagination beside reality (never shown to the model). The prompt feeds the VLM, which emits each
  // token straight up into a slot; older tokens slide left. A probe stage (taps-vlm) draws the probe from
  // the side of the VLM straight to the bubble, apart from the tokens; a visual-token stage draws the
  // decode line from the newest token, which sits under the bubble.
  function playStage({ host, story, method, onDone, compact = false }) {
    const timeline = rolloutTimeline({ story, method });
    const { rollout, moves } = timeline;
    const info = METHOD_INFO[method];
    const probeInfo = PROBE_INFO[info.link];
    const tapsVlm = info.link !== "tokens";
    const stepMs = stepDuration(moves.length);

    const stage = el("div", `story-stage ${tapsVlm ? "taps-vlm" : "decodes-tokens"}${compact ? " is-compact" : ""}`);
    stage.dataset.phase = "input";

    const realitySrc = stripPath({ story, method, kind: "reality" });
    const imaginationSrc = stripPath({ story, method, kind: "imagination" });

    const reality = el("figure", "stage-card stage-reality");
    const realityLabel = el("figcaption", "stage-label");
    realityLabel.append(el("b", null, "Reality"), " · never shown to the model");
    const realityImage = el("div", "stage-image");
    const realityView = stripView({ className: "stage-frame", src: realitySrc, frameCount: rollout.realityFrames });
    realityImage.append(realityView);
    reality.append(realityLabel, realityImage);

    const mind = el("figure", "stage-card stage-mind");
    const mindLabel = el("figcaption", "stage-label");
    mindLabel.append(el("span", "caption-input", "Input"), el("span", "caption-thought", "Imagination"));
    const mindImage = el("div", "stage-image");
    const imaginationView = stripView({ className: "stage-frame frame-imagination", src: imaginationSrc, frameCount: rollout.imaginationFrames });
    mindImage.append(
      stripView({ className: "stage-frame frame-input", src: realitySrc, frameCount: rollout.realityFrames }),
      imaginationView);
    const probe = el("span", `stage-probe probe-${info.link}`);
    const readOutDot = el("span", "probe-dot");
    const probeLabel = el("span", "probe-label", probeInfo.label);
    probe.append(el("span", "probe-line"), readOutDot, probeLabel);
    if (tapsVlm) {
      probe.append(el("span", "probe-sublabel", "reads every layer"));
      probeLabel.dataset.probeDetail = ""; // opens the probe figure (probe_detail.js)
      probeLabel.title = "How the probe works";
    }
    mind.append(mindLabel, mindImage);
    if (!tapsVlm) mind.append(probe);

    const promptSide = el("div", "stage-prompt-side");
    promptSide.append(promptChip(story.prompt), el("span", "stage-arrow"));
    const vlm = el("div", "stage-vlm");
    const vlmChip = el("span", "vlm-chip", "VLM");
    vlm.append(el("span", "stage-emit"), vlmChip);
    // The probe's bracket spans the VLM's layers.
    if (tapsVlm) vlm.append(el("span", "vlm-tap"));
    const tape = slotTape({
      slotCenter: () => {
        const slotBox = (tapsVlm ? vlmChip : mindImage).getBoundingClientRect();
        return slotBox.left + slotBox.width / 2 - tape.element.getBoundingClientRect().left;
      },
      anchorSelector: probeInfo.anchor,
    });
    const tokenRow = el("div", "stage-tokens");
    tokenRow.append(tape.element);

    stage.append(mind, reality, tokenRow, promptSide, vlm, ...(tapsVlm ? [probe] : []));
    host.replaceChildren(stage);
    new ResizeObserver(() => tape.align()).observe(stage);

    const timers = makeTimers();
    timers.at(INTRO_MS, () => { stage.dataset.phase = "thinking"; });
    moves.forEach((step, moveIndex) => {
      const startMs = INTRO_MS + MORPH_MS + moveIndex * stepMs;
      timers.at(startMs, () => {
        tape.push(tokenElement({
          step, method, story, imaginationFrames: rollout.imaginationFrames, frameIndex: timeline.imaginationFrame(moveIndex),
        }));
        stage.classList.add("has-probe");
        runDot({ dot: readOutDot, durationMs: stepMs * FRAME_DELAY });
      });
      timers.at(startMs + stepMs * FRAME_DELAY, () => {
        showStripFrame({ view: realityView, frameIndex: timeline.realityFrame(moveIndex) });
        developFrame({ view: imaginationView, frameIndex: timeline.imaginationFrame(moveIndex), stepMs });
        if (step.imagined) flash({ element: realityImage, className: "is-blocked" });
      });
    });
    // An answer is the model's last token, emitted into the slot; whether actions solved the puzzle is a
    // fact about the real board, so it is badged on reality.
    timers.at(INTRO_MS + MORPH_MS + moves.length * stepMs, () => {
      const outcome = outcomeElement({ solved: rollout.solved, answer: timeline.answer });
      if (timeline.answer) {
        const answerToken = el("span", "tape-token");
        answerToken.append(outcome);
        tape.push(answerToken);
      } else {
        const verdict = el("div", "stage-verdict");
        verdict.append(outcome);
        realityImage.append(verdict);
      }
      tape.settle();
      stage.dataset.phase = "done";
      onDone();
    });
    return { stop: timers.clear };
  }

  // ---------------------------------------------------------------------------
  // Head to head: the three methods in lockstep lanes on the same puzzle.
  // ---------------------------------------------------------------------------

  function playLanes({ host, story, onDone }) {
    const board = el("div", "lane-board");
    board.append(promptChip(story.prompt));
    const lanes = LANE_METHODS.map((method) => {
      const timeline = rolloutTimeline({ story, method });
      const info = METHOD_INFO[method];
      const lane = el("div", "lane");
      const label = el("div", "lane-label");
      label.append(el("b", null, info.name), el("span", null, info.linkLabel));
      const imaginationView = stripView({
        className: "lane-frame lane-bubble",
        src: stripPath({ story, method, kind: "imagination" }),
        frameCount: timeline.rollout.imaginationFrames,
      });
      const realityView = stripView({
        className: "lane-frame",
        src: stripPath({ story, method, kind: "reality" }),
        frameCount: timeline.rollout.realityFrames,
      });
      const imaginationCell = el("figure", "lane-cell");
      imaginationCell.append(imaginationView, el("figcaption", null, "Imagination"));
      const realityCell = el("figure", "lane-cell");
      realityCell.append(realityView, el("figcaption", null, "Reality"));
      const tape = tokenTape();
      const outcomeSlot = el("div", "stage-outcome-slot");
      lane.append(label, imaginationCell, realityCell, tape.element, outcomeSlot);
      board.append(lane);
      return { method, timeline, imaginationView, realityView, tape, outcomeSlot };
    });
    host.replaceChildren(board);

    const timers = makeTimers();
    const stepMs = stepDuration(Math.max(...lanes.map((lane) => lane.timeline.moves.length)));
    let lanesDone = 0;
    lanes.forEach((lane) => {
      const { timeline } = lane;
      timeline.moves.forEach((step, moveIndex) => {
        const startMs = INTRO_MS + moveIndex * stepMs;
        timers.at(startMs, () => lane.tape.push(tokenElement({
          step, method: lane.method, story, imaginationFrames: timeline.rollout.imaginationFrames, frameIndex: timeline.imaginationFrame(moveIndex),
        })));
        timers.at(startMs + stepMs * FRAME_DELAY, () => {
          showStripFrame({ view: lane.realityView, frameIndex: timeline.realityFrame(moveIndex) });
          developFrame({ view: lane.imaginationView, frameIndex: timeline.imaginationFrame(moveIndex), stepMs });
        });
      });
      timers.at(INTRO_MS + timeline.moves.length * stepMs, () => {
        lane.tape.settle();
        lane.outcomeSlot.replaceChildren(outcomeElement({ solved: timeline.rollout.solved, answer: timeline.answer }));
        lanesDone += 1;
        if (lanesDone === lanes.length) onDone();
      });
    });
    return { stop: timers.clear };
  }

  // ---------------------------------------------------------------------------
  // Player: puzzle chips, auto-advance, preloading, and pausing while off screen.
  // ---------------------------------------------------------------------------

  function createPlayer({ host, stories, methodsFor, play }) {
    const chips = el("div", "story-chips");
    const stageHost = el("div", "story-stage-host");
    const caption = el("p", "story-caption");
    if (stories.length > 1) host.append(chips);
    host.append(stageHost, caption);

    let current = 0;
    let controller = null;
    let holdTimer = null;
    let running = false;
    let showRequest = 0;

    const chipButtons = stories.map((story, index) => {
      const chip = el("button", "chip", story.label || `${GAMES[story.game].name} #${Number(story.puzzle)}`);
      chip.type = "button";
      chip.addEventListener("click", () => show(index));
      chips.append(chip);
      return chip;
    });

    function stop() {
      showRequest += 1;
      if (controller) controller.stop();
      clearTimeout(holdTimer);
      controller = null;
    }

    // A story's strips load just ahead of it: the first story's as the player nears the screen, the next
    // story's while the current one plays. Loading every story at once would make the first wait behind them.
    function prefetch(index) {
      const story = stories[index];
      return loadStory({ story, methods: methodsFor(story) });
    }

    function show(index) {
      stop();
      const request = showRequest;
      const story = stories[index];
      current = index;
      chipButtons.forEach((chip, chipIndex) => chip.setAttribute("aria-selected", String(chipIndex === index)));
      caption.textContent = story.caption || "";
      prefetch(index).then(() => {
        if (request !== showRequest || !running) return;
        controller = play({
          host: stageHost,
          story,
          onDone: () => {
            holdTimer = setTimeout(() => { if (running) show((current + 1) % stories.length); }, HOLD_MS);
          },
        });
        prefetch((index + 1) % stories.length);
      });
    }

    const preloadObserver = new IntersectionObserver(([entry]) => {
      if (!entry.isIntersecting) return;
      preloadObserver.disconnect();
      prefetch(current);
    }, { rootMargin: "800px 0px" });
    preloadObserver.observe(host);

    new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting && !running) {
        running = true;
        show(current);
      } else if (!entry.isIntersecting && running) {
        running = false;
        stop();
      }
    }, { threshold: 0.3 }).observe(host);
  }

  // ---------------------------------------------------------------------------
  // Curated stories (the examples from our talk) and the sections they fill.
  // ---------------------------------------------------------------------------

  const EMERGENCE = [
    { game: "rush_hour", puzzle: "0008", prompt: "Solve the rush hour game." },
    { game: "character_recognition", puzzle: "0000", prompt: "Unscramble the character." },
    { game: "character_recognition", puzzle: "0006", prompt: "Unscramble the character." },
    { game: "mental_rotation_2d_certificate", puzzle: "0137", prompt: "Do they have the same chirality?" },
    { game: "mental_rotation_2d_certificate", puzzle: "0162", prompt: "Do they have the same chirality?" },
    { game: "tangram_certificate", puzzle: "0027", prompt: "Is the puzzle solvable?" },
    { game: "tangram_certificate", puzzle: "0037", prompt: "Is the puzzle solvable?" },
  ];

  const MECHANISMS = [
    {
      method: "visual_supervision",
      title: "Visual Supervision",
      text: "The same probe, reading every layer of the VLM where each action ends, is trained together with the VLM, so its loss sharpens the model’s mental images. It is dropped at test time.",
      story: { game: "jigsaw", puzzle: "0056", prompt: "Reassemble the image." },
      showsProbeDetail: true,
    },
    {
      method: "visual_cot",
      title: "Next-state chain-of-thought",
      text: "After each action, the model writes out the resulting state, then plans its next move with it in context: as 16 or 64 visual tokens (Visual CoT, shown here), as a compact JSON description (State CoT), or as the locations of the objects (Boxes CoT).",
      story: { game: "jigsaw", puzzle: "0056", prompt: "Reassemble the image." },
    },
  ];

  const HEAD_TO_HEAD = [
    {
      game: "jigsaw", puzzle: "0056", prompt: "Reassemble the image.",
      caption: "Only the models that sharpen their mental imagery solve it.",
    },
    {
      game: "sokoban", puzzle: "0132", prompt: "Push the boxes onto the targets.",
      caption: "Action Supervision loses track of the state and walks into a deadlock, then hallucinates a block to get out of it. Visual Supervision solves the puzzle.",
    },
    {
      game: "rush_hour", puzzle: "0108", prompt: "Free the red car.",
      caption: "Action Supervision loses track of the state and gets stuck in a deadlock. Visual CoT solves the puzzle.",
    },
    {
      game: "tangram_certificate", puzzle: "0168", prompt: "Is the tangram puzzle solvable?",
      caption: "Look closely: Visual Supervision gets the right answer, but for the wrong reason.",
    },
    {
      game: "mental_rotation_certificate", puzzle: "0018", prompt: "Do they have the same chirality?",
    },
    {
      game: "mental_rotation_certificate", puzzle: "0025", prompt: "Do they have the same chirality?",
    },
  ];

  const LIMITATION = [
    {
      game: "rush_hour", puzzle: "0027", method: "visual_supervision", prompt: "Free the red car.",
      caption: "A confident but wrong mental image leads to wrong moves. The model imagines the yellow car (G), then the blue car (B), sliding down out of the way, and clears a path for the red car to the exit. In reality, G and B are blocked and never move, so every move in red does nothing, and the puzzle stays unsolved.",
    },
  ];

  function buildMechanisms(host) {
    MECHANISMS.forEach((mechanism) => {
      const card = el("article", "mechanism-card");
      const text = el("p", null, mechanism.text);
      if (mechanism.showsProbeDetail) {
        const detailLink = el("button", "link-button inline-link", "How the probe works →");
        detailLink.type = "button";
        detailLink.dataset.probeDetail = "";
        text.append(" ", detailLink);
      }
      card.append(el("h3", null, mechanism.title), text);
      const playerHost = el("div", "story-player");
      card.append(playerHost);
      host.append(card);
      createPlayer({
        host: playerHost,
        stories: [mechanism.story],
        methodsFor: () => [mechanism.method],
        play: ({ host: stageHost, story, onDone }) =>
          playStage({ host: stageHost, story, method: mechanism.method, onDone, compact: true }),
      });
    });
  }

  createPlayer({
    host: document.getElementById("emergence-player"),
    stories: EMERGENCE,
    methodsFor: () => ["action_supervision"],
    play: ({ host, story, onDone }) => playStage({ host, story, method: "action_supervision", onDone }),
  });
  buildMechanisms(document.getElementById("mechanisms"));
  createPlayer({
    host: document.getElementById("head-to-head"),
    stories: HEAD_TO_HEAD,
    methodsFor: () => LANE_METHODS,
    play: playLanes,
  });
  createPlayer({
    host: document.getElementById("limitation-player"),
    stories: LIMITATION,
    methodsFor: (story) => [story.method],
    play: ({ host, story, onDone }) => playStage({ host, story, method: story.method, onDone }),
  });
})();
