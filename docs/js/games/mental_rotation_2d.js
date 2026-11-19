/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2026 Apple Inc. All Rights Reserved.
 */
// 2D shape matching (engine/games/mental_rotation_2d): rotate the left polyomino to match the right one.
// Actions: rotate(d), d a whole number of degrees in [-180, 180], not 0; positive turns clockwise.
(function () {
  "use strict";

  const { RENDER_SIZE, button, rgb } = window.PuzzleKit;

  const BACKGROUND_COLOR = [0x1a, 0x1a, 0x2e];
  const DIVIDER_COLOR = [0x44, 0x44, 0x66];
  const DIVIDER_WIDTH = Math.max(2, Math.floor(RENDER_SIZE / 64));
  const PANEL_WIDTH = Math.floor((RENDER_SIZE - DIVIDER_WIDTH) / 2);
  // A pixel is border when it is this far from its cell's center along either axis (cells are unit squares).
  const BORDER_THRESHOLD = 0.42;
  const SMALL_STEP = 15;
  const LARGE_STEP = 30;

  // The engine's hsl_to_rgb (render.rs): h in [0, 360), s and l in [0, 1].
  function hslToRgb({ hue, saturation, lightness }) {
    const chroma = (1 - Math.abs(2 * lightness - 1)) * saturation;
    const sector = hue / 60;
    const second = chroma * (1 - Math.abs((sector % 2) - 1));
    const [red, green, blue] = [
      [chroma, second, 0], [second, chroma, 0], [0, chroma, second],
      [0, second, chroma], [second, 0, chroma], [chroma, 0, second],
    ][Math.min(5, Math.floor(sector))];
    const offset = lightness - chroma / 2;
    return [red, green, blue].map((channel) => Math.round((channel + offset) * 255));
  }

  function shapeColors(colorSeed) {
    const hue = colorSeed % 360;
    return {
      fill: hslToRgb({ hue, saturation: 0.65, lightness: 0.55 }),
      border: hslToRgb({ hue, saturation: 0.55, lightness: 0.3 }),
    };
  }

  function parseRotation(action) {
    const match = /^rotate\((-?\d+)\)$/.exec(action);
    if (!match) return null;
    const degrees = Number(match[1]);
    return degrees === 0 || degrees < -180 || degrees > 180 ? null : degrees;
  }

  function apply({ state, action }) {
    const degrees = parseRotation(action);
    if (degrees === null) return null;
    return { rotation: (((state.rotation + degrees) % 360) + 360) % 360 };
  }

  function isSolved({ puzzle, state }) {
    return puzzle.board.solved_rotations.includes(state.rotation);
  }

  // One panel, the shape turned `rotationDegrees` clockwise about its centroid (then mirrored if flipped),
  // framed as the engine frames it: 2.5 bounding radii across the panel's narrow side.
  function drawPanel({ context, cells, rotationDegrees, flipped, panelX, colors }) {
    const centroidX = cells.reduce((sum, [x]) => sum + x, 0) / cells.length;
    const centroidY = cells.reduce((sum, [, y]) => sum + y, 0) / cells.length;
    const boundingRadius = Math.max(...cells.map(([x, y]) => Math.max(Math.abs(x - centroidX), Math.abs(y - centroidY)) + 0.5));
    const pixelsPerUnit = Math.min(PANEL_WIDTH, RENDER_SIZE) / (boundingRadius * 2.5);
    context.save();
    context.beginPath();
    context.rect(panelX, 0, PANEL_WIDTH, RENDER_SIZE);
    context.clip();
    context.translate(panelX + PANEL_WIDTH / 2, RENDER_SIZE / 2);
    context.scale(flipped ? -pixelsPerUnit : pixelsPerUnit, pixelsPerUnit);
    context.rotate((rotationDegrees * Math.PI) / 180);
    cells.forEach(([x, y]) => {
      const left = x - centroidX - 0.5;
      const top = y - centroidY - 0.5;
      context.fillStyle = rgb(colors.border);
      context.fillRect(left, top, 1, 1);
      context.fillStyle = rgb(colors.fill);
      context.fillRect(left + 0.5 - BORDER_THRESHOLD, top + 0.5 - BORDER_THRESHOLD, 2 * BORDER_THRESHOLD, 2 * BORDER_THRESHOLD);
    });
    context.restore();
  }

  function render({ context, puzzle, state, ui }) {
    const board = puzzle.board;
    const colors = shapeColors(board.color_seed);
    context.fillStyle = rgb(BACKGROUND_COLOR);
    context.fillRect(0, 0, RENDER_SIZE, RENDER_SIZE);
    context.fillStyle = rgb(DIVIDER_COLOR);
    context.fillRect(PANEL_WIDTH, 0, DIVIDER_WIDTH, RENDER_SIZE);
    drawPanel({ context, cells: board.cells, rotationDegrees: state.rotation + (ui.dragDegrees || 0), flipped: false, panelX: 0, colors });
    drawPanel({ context, cells: board.cells, rotationDegrees: board.target_rotation, flipped: board.target_flipped, panelX: PANEL_WIDTH + DIVIDER_WIDTH, colors });
  }

  function mount({ canvas, controls, act, ui, redraw, toLogical }) {
    const group = document.createElement("div");
    group.className = "play-group";
    [
      [-LARGE_STEP, `↺ ${LARGE_STEP}°`, "Counterclockwise (Left arrow)"],
      [-SMALL_STEP, `↺ ${SMALL_STEP}°`, "Counterclockwise (Shift + Left arrow)"],
      [SMALL_STEP, `↻ ${SMALL_STEP}°`, "Clockwise (Shift + Right arrow)"],
      [LARGE_STEP, `↻ ${LARGE_STEP}°`, "Clockwise (Right arrow)"],
    ].forEach(([degrees, label, title]) => {
      group.append(button({ label, title, onClick: () => act(`rotate(${degrees})`) }));
    });
    controls.append(group);

    // Dragging around the left panel turns the shape freely; on release, the turn snaps to whole
    // SMALL_STEPs and becomes one rotate action.
    const panelCenter = { x: PANEL_WIDTH / 2, y: RENDER_SIZE / 2 };
    let dragStartAngle = null;
    const pointerAngle = (point) => (Math.atan2(point.y - panelCenter.y, point.x - panelCenter.x) * 180) / Math.PI;

    function onPointerDown(event) {
      const point = toLogical(event);
      if (point.x >= PANEL_WIDTH) return;
      dragStartAngle = pointerAngle(point);
      ui.dragDegrees = 0;
      canvas.setPointerCapture(event.pointerId);
    }

    function onPointerMove(event) {
      if (dragStartAngle === null) return;
      const turned = pointerAngle(toLogical(event)) - dragStartAngle;
      ui.dragDegrees = ((((turned + 180) % 360) + 360) % 360) - 180;
      redraw();
    }

    function onPointerUp() {
      if (dragStartAngle === null) return;
      const snapped = Math.round(ui.dragDegrees / SMALL_STEP) * SMALL_STEP;
      dragStartAngle = null;
      ui.dragDegrees = 0;
      if (snapped === 0 || !act(`rotate(${snapped})`)) redraw();
    }

    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointercancel", onPointerUp);

    return {
      onKey(event) {
        if (event.metaKey || event.ctrlKey || event.altKey) return false;
        const sign = { ArrowLeft: -1, ArrowRight: 1 }[event.key];
        if (!sign) return false;
        act(`rotate(${sign * (event.shiftKey ? SMALL_STEP : LARGE_STEP)})`);
        return true;
      },
      destroy() {
        canvas.removeEventListener("pointerdown", onPointerDown);
        canvas.removeEventListener("pointermove", onPointerMove);
        canvas.removeEventListener("pointerup", onPointerUp);
        canvas.removeEventListener("pointercancel", onPointerUp);
      },
    };
  }

  window.PuzzleGames.mental_rotation_2d = {
    goal: "Rotate the left shape until it matches the right one.",
    help: "Use controls above or drag the left shape to rotate it.",
    initialState: ({ puzzle }) => puzzle.board.state,
    apply,
    isSolved,
    render,
    mount,
  };
})();
