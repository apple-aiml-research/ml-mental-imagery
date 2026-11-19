/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2026 Apple Inc. All Rights Reserved.
 */
// Rush Hour (engine/games/rush_hour): slide vehicles along their axes until the red car A reaches the right
// edge. Actions: slide('X', n), n cells right/down if positive, left/up if negative.
(function () {
  "use strict";

  const { RENDER_SIZE, keyDirection, fitToResolution, rgb, drawEngineText } = window.PuzzleKit;

  const EMPTY_COLOR = [0xd0, 0xd0, 0xd0];
  const GRID_LINE_COLOR = [0xb0, 0xb0, 0xb0];
  const BORDER_COLOR = [0x44, 0x44, 0x44];
  const VEHICLE_COLORS = [
    [0xdd, 0x22, 0x22], [0x22, 0x77, 0xdd], [0x22, 0xaa, 0x22], [0xff, 0xa5, 0x00],
    [0x88, 0x22, 0xcc], [0x00, 0xbb, 0xbb], [0xdd, 0xdd, 0x22], [0xdd, 0x66, 0xaa],
    [0x88, 0x66, 0x33], [0x66, 0xbb, 0x66], [0x99, 0x99, 0xdd], [0xdd, 0x88, 0x44],
  ];
  const SELECTION_COLOR = "#1a1a2e";
  // A pointer that moves less than this (render px) between press and release selects instead of dragging.
  const CLICK_SLOP = 6;

  // ---------------------------------------------------------------------------
  // Rules (rush_hour/mod.rs apply, parse_action).
  // ---------------------------------------------------------------------------

  // Cell (col, row) -> vehicle index, or -1 if empty.
  function occupancy(board, positions) {
    const grid = new Array(board.size * board.size).fill(-1);
    board.vehicles.forEach((vehicle, index) => {
      for (let offset = 0; offset < vehicle.length; offset += 1) {
        const along = positions[index] + offset;
        const [col, row] = vehicle.horizontal ? [along, vehicle.fixed] : [vehicle.fixed, along];
        grid[row * board.size + col] = index;
      }
    });
    return grid;
  }

  function cellIndex(board, vehicle, along) {
    return vehicle.horizontal ? vehicle.fixed * board.size + along : along * board.size + vehicle.fixed;
  }

  // As the engine reads it: the vehicle is the first character before the first "', ", the distance an i8.
  function parseSlide(action) {
    const prefix = "slide('";
    if (!action.startsWith(prefix)) return null;
    const rest = action.slice(prefix.length);
    const separatorIndex = rest.indexOf("', ");
    const vehicleId = separatorIndex >= 0 ? [...rest.slice(0, separatorIndex)][0] : undefined;
    const tail = rest.slice(separatorIndex + 3);
    if (!vehicleId || !tail.endsWith(")")) return null;
    const amountText = tail.slice(0, -1);
    if (!/^[+-]?\d+$/.test(amountText)) return null;
    const amount = Number(amountText);
    if (amount === 0 || amount < -128 || amount > 127) return null;
    return { vehicleId, amount };
  }

  function apply({ puzzle, state, action }) {
    const slide = parseSlide(action);
    if (!slide) return null;
    const board = puzzle.board;
    const vehicleIndex = board.vehicles.findIndex((vehicle) => vehicle.id === slide.vehicleId);
    if (vehicleIndex < 0) return null;
    const vehicle = board.vehicles[vehicleIndex];
    const oldPosition = state.positions[vehicleIndex];
    const newPosition = oldPosition + slide.amount;
    if (newPosition < 0 || newPosition + vehicle.length > board.size) return null;
    const grid = occupancy(board, state.positions);
    const [sweepStart, sweepEnd] = slide.amount > 0
      ? [oldPosition + vehicle.length, newPosition + vehicle.length - 1]
      : [newPosition, oldPosition - 1];
    for (let along = sweepStart; along <= sweepEnd; along += 1) {
      if (grid[cellIndex(board, vehicle, along)] >= 0) return null;
    }
    const positions = state.positions.slice();
    positions[vehicleIndex] = newPosition;
    return { positions };
  }

  function isSolved({ puzzle, state }) {
    return state.positions[0] + puzzle.board.vehicles[0].length === puzzle.board.size;
  }

  // How far a vehicle can slide back (left/up) and forward (right/down) from where it is.
  function slideRange(board, positions, vehicleIndex) {
    const vehicle = board.vehicles[vehicleIndex];
    const grid = occupancy(board, positions);
    const position = positions[vehicleIndex];
    let back = 0;
    while (position - back - 1 >= 0 && grid[cellIndex(board, vehicle, position - back - 1)] < 0) back += 1;
    let forward = 0;
    while (position + vehicle.length + forward < board.size && grid[cellIndex(board, vehicle, position + vehicle.length + forward)] < 0) forward += 1;
    return { back, forward };
  }

  // ---------------------------------------------------------------------------
  // Rendering (rush_hour/render.rs).
  // ---------------------------------------------------------------------------

  function layout(board) {
    const border = Math.max(Math.floor(RENDER_SIZE / 40), 2);
    const { offsetX, offsetY, cellSize } = fitToResolution({ boardWidth: board.size, boardHeight: board.size, resolution: RENDER_SIZE - 2 * border });
    return { border, offsetX: offsetX + border, offsetY: offsetY + border, cellSize };
  }

  // A vehicle's rectangle in render space, `shift` cells along its axis from its position.
  function vehicleRect({ board, positions, vehicleIndex, shift = 0 }) {
    const { offsetX, offsetY, cellSize } = layout(board);
    const vehicle = board.vehicles[vehicleIndex];
    const margin = Math.max(Math.floor(cellSize / 8), 1);
    const along = (positions[vehicleIndex] + shift) * cellSize;
    const across = vehicle.fixed * cellSize;
    const longSide = vehicle.length * cellSize - 2 * margin;
    const shortSide = cellSize - 2 * margin;
    return vehicle.horizontal
      ? { x: offsetX + along + margin, y: offsetY + across + margin, width: longSide, height: shortSide }
      : { x: offsetX + across + margin, y: offsetY + along + margin, width: shortSide, height: longSide };
  }

  function vehicleColor(id) {
    return VEHICLE_COLORS[(((id.charCodeAt(0) & 0xff) - 65 + 256) % 256) % VEHICLE_COLORS.length];
  }

  function luminance([red, green, blue]) {
    return 0.299 * red + 0.587 * green + 0.114 * blue;
  }

  function render({ context, puzzle, state, ui }) {
    const board = puzzle.board;
    const { border, offsetX, offsetY, cellSize } = layout(board);
    const gridSpan = board.size * cellSize;
    const fillRect = (color, x, y, width, height) => {
      context.fillStyle = rgb(color);
      context.fillRect(x, y, Math.max(1, width), Math.max(1, height));
    };
    fillRect(EMPTY_COLOR, 0, 0, RENDER_SIZE, RENDER_SIZE);

    // A thick border with a gap at the exit row.
    const [gridLeft, gridTop] = [offsetX - border, offsetY - border];
    const [gridRight, gridBottom] = [offsetX + gridSpan, offsetY + gridSpan];
    const exitTop = offsetY + board.exitRow * cellSize;
    const exitBottom = exitTop + cellSize;
    fillRect(BORDER_COLOR, gridLeft, gridTop, gridSpan + 2 * border, border);
    fillRect(BORDER_COLOR, gridLeft, gridBottom, gridSpan + 2 * border, border);
    fillRect(BORDER_COLOR, gridLeft, gridTop, border, gridBottom - gridTop + border);
    if (exitTop > offsetY) fillRect(BORDER_COLOR, gridRight, gridTop, border, exitTop - gridTop);
    if (exitBottom < gridBottom) fillRect(BORDER_COLOR, gridRight, exitBottom, border, gridBottom + border - exitBottom);

    for (let line = 0; line <= board.size; line += 1) {
      fillRect(GRID_LINE_COLOR, offsetX, offsetY + line * cellSize, gridSpan, 1);
      fillRect(GRID_LINE_COLOR, offsetX + line * cellSize, offsetY, 1, gridSpan);
    }

    const drag = ui && ui.drag;
    board.vehicles.forEach((vehicle, vehicleIndex) => {
      const shift = drag && drag.vehicleIndex === vehicleIndex ? drag.shift : 0;
      const { x, y, width, height } = vehicleRect({ board, positions: state.positions, vehicleIndex, shift });
      const color = vehicleColor(vehicle.id);
      fillRect(color, x, y, width, height);
      drawEngineText({
        context,
        centerX: Math.trunc(x + Math.floor(width / 2)),
        centerY: Math.trunc(y + Math.floor(height / 2)),
        text: vehicle.id,
        fontHeight: Math.max(cellSize * 0.6, 8),
        color: luminance(color) > 128 ? [0, 0, 0] : [255, 255, 255],
      });
      if (ui && ui.selected === vehicleIndex) {
        context.strokeStyle = SELECTION_COLOR;
        context.lineWidth = 4;
        context.strokeRect(x - 3, y - 3, width + 6, height + 6);
      }
    });
  }

  // ---------------------------------------------------------------------------
  // Input: drag a vehicle along its axis (it stops at whatever blocks it) and drop it to slide it; or click
  // a vehicle (or type its letter) to select it and slide it a cell at a time with the arrow keys.
  // ---------------------------------------------------------------------------

  function mount({ puzzle, canvas, act, getState, ui, redraw, toLogical }) {
    const board = puzzle.board;

    function vehicleAt(point) {
      const { offsetX, offsetY, cellSize } = layout(board);
      const col = Math.floor((point.x - offsetX) / cellSize);
      const row = Math.floor((point.y - offsetY) / cellSize);
      if (col < 0 || row < 0 || col >= board.size || row >= board.size) return -1;
      return occupancy(board, getState().positions)[row * board.size + col];
    }

    function slideAction(vehicleIndex, amount) {
      return `slide('${board.vehicles[vehicleIndex].id}', ${amount})`;
    }

    function select(vehicleIndex) {
      ui.selected = vehicleIndex;
      redraw();
    }

    function onPointerDown(event) {
      const state = getState();
      if (isSolved({ puzzle, state })) return;
      const point = toLogical(event);
      const vehicleIndex = vehicleAt(point);
      if (vehicleIndex < 0) {
        if (ui.selected !== undefined) select(undefined);
        return;
      }
      const { back, forward } = slideRange(board, state.positions, vehicleIndex);
      ui.drag = { vehicleIndex, start: point, back, forward, shift: 0, moved: false };
      canvas.setPointerCapture(event.pointerId);
      canvas.style.cursor = "grabbing";
    }

    function onPointerMove(event) {
      const point = toLogical(event);
      const drag = ui.drag;
      if (!drag) {
        canvas.style.cursor = vehicleAt(point) >= 0 ? "grab" : "";
        return;
      }
      const vehicle = board.vehicles[drag.vehicleIndex];
      const travel = vehicle.horizontal ? point.x - drag.start.x : point.y - drag.start.y;
      if (Math.abs(travel) > CLICK_SLOP) drag.moved = true;
      drag.shift = Math.max(-drag.back, Math.min(drag.forward, travel / layout(board).cellSize));
      redraw();
    }

    function onPointerUp(event) {
      const drag = ui.drag;
      if (!drag) return;
      ui.drag = null;
      canvas.style.cursor = "grab";
      if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
      const amount = Math.round(drag.shift);
      if (!drag.moved) {
        select(ui.selected === drag.vehicleIndex ? undefined : drag.vehicleIndex);
        return;
      }
      ui.selected = drag.vehicleIndex;
      if (amount === 0 || !act(slideAction(drag.vehicleIndex, amount))) redraw();
    }

    function onPointerCancel() {
      ui.drag = null;
      redraw();
    }

    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointercancel", onPointerCancel);

    return {
      onKey(event) {
        if (event.metaKey || event.ctrlKey || event.altKey) return false;
        const typedIndex = board.vehicles.findIndex((vehicle) => vehicle.id === event.key.toUpperCase());
        if (event.key.length === 1 && typedIndex >= 0) {
          select(typedIndex);
          return true;
        }
        const direction = keyDirection(event);
        if (!direction || !event.key.startsWith("Arrow")) return false;
        if (ui.selected === undefined) select(0);
        const vehicle = board.vehicles[ui.selected];
        const step = { left: -1, up: -1, right: 1, down: 1 }[direction];
        const alongAxis = vehicle.horizontal ? direction === "left" || direction === "right" : direction === "up" || direction === "down";
        if (alongAxis) act(slideAction(ui.selected, step));
        return true;
      },
      destroy() {
        canvas.removeEventListener("pointerdown", onPointerDown);
        canvas.removeEventListener("pointermove", onPointerMove);
        canvas.removeEventListener("pointerup", onPointerUp);
        canvas.removeEventListener("pointercancel", onPointerCancel);
        canvas.style.cursor = "";
      },
    };
  }

  window.PuzzleGames.rush_hour = {
    goal: "Slide the vehicles out of the way so the red car A can drive out the exit on the right.",
    help: "Drag vehicles along their lanes.",
    initialState: ({ puzzle }) => puzzle.board.state,
    apply,
    isSolved,
    render,
    mount,
  };
})();
