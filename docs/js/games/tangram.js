/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2026 Apple Inc. All Rights Reserved.
 */
// Tangram (engine/games/tangram): fill the grey silhouette with the colored pieces. Actions: place('P0', x, y)
// puts the piece's anchor (the ✕ cell) at column x, row y; rotatecw('P0') / rotateccw('P0') turn it 90°
// about the anchor, in the tray or on the board.
(function () {
  "use strict";

  const { RENDER_SIZE, button, keyDirection, rgb, drawEngineText } = window.PuzzleKit;

  const BACKGROUND_COLOR = [0xf0, 0xf0, 0xf0];
  const TARGET_COLOR = [0xc0, 0xc0, 0xc0];
  const GRID_COLOR = [0x80, 0x80, 0x80];
  const LABEL_COLOR = [0x60, 0x60, 0x60];
  const PIECE_COLORS = [
    [0x44, 0x44, 0xff], [0xff, 0x44, 0x44], [0x44, 0xcc, 0x44], [0xff, 0xcc, 0x00],
    [0x88, 0x00, 0xff], [0xff, 0x88, 0x00], [0x00, 0xcc, 0xcc], [0xcc, 0x00, 0x88],
    [0x88, 0x88, 0x00], [0x00, 0x44, 0x88], [0x88, 0x44, 0x00], [0x44, 0x88, 0x00],
  ];
  const SELECTION_COLOR = "#4a3aa7";
  const DRAG_THRESHOLD_PX = 4;

  // Layout constants of render.rs at the 512 px render space (f32 math, as in the engine).
  const f32 = Math.fround;
  const MARGIN = Math.trunc(f32(RENDER_SIZE * f32(0.02)));
  const AXIS_FONT_HEIGHT = Math.max(f32(RENDER_SIZE * f32(0.04)), 8);
  const LABEL_GAP = Math.trunc(f32(AXIS_FONT_HEIGHT * f32(1.2)));
  const BOARD_ORIGIN = MARGIN + LABEL_GAP;

  // ---------------------------------------------------------------------------
  // Rules (mod.rs apply / is_solved / parse_action).
  // ---------------------------------------------------------------------------

  const byId = (first, second) => (first < second ? -1 : first > second ? 1 : 0);

  function rotatedCells(cells, rotation) {
    const normalized = ((rotation % 360) + 360) % 360;
    return cells.map(([x, y]) => {
      if (normalized === 90) return [-y, x];
      if (normalized === 180) return [-x, -y];
      if (normalized === 270) return [y, -x];
      return [x, y];
    });
  }

  function findPiece(board, pieceId) {
    return board.pieces.find((piece) => piece.id === pieceId) || null;
  }

  function trayRotation(state, pieceId) {
    const entry = state.rotations.find(([id]) => id === pieceId);
    return entry ? entry[1] : 0;
  }

  function placementCells(board, placement) {
    const piece = findPiece(board, placement.piece);
    return rotatedCells(piece.cells, placement.rotation).map(([dx, dy]) => [placement.x + dx, placement.y + dy]);
  }

  function inBounds(board, cells) {
    return cells.every(([x, y]) => x >= 0 && x < board.width && y >= 0 && y < board.height);
  }

  // Rust's str::parse::<i32>: an optional sign and digits, nothing else.
  function parseInteger(text) {
    if (!/^[+-]?\d+$/.test(text)) return null;
    const value = Number(text);
    return value >= -2147483648 && value <= 2147483647 ? value : null;
  }

  function parseAction(action) {
    if (action.startsWith("place('")) {
      const inner = action.slice("place('".length);
      if (!inner.endsWith(")")) return null;
      const rest = inner.slice(0, -1);
      const separator = rest.indexOf("', ");
      if (separator < 0) return null;
      const numbers = rest.slice(separator + 3).split(", ");
      if (numbers.length !== 2) return null;
      const [x, y] = numbers.map(parseInteger);
      if (x === null || y === null) return null;
      return { kind: "place", pieceId: rest.slice(0, separator), x, y };
    }
    for (const [prefix, kind] of [["rotatecw('", "rotatecw"], ["rotateccw('", "rotateccw"]]) {
      if (action.startsWith(prefix)) {
        const inner = action.slice(prefix.length);
        if (!inner.endsWith("')")) return null;
        return { kind, pieceId: inner.slice(0, -2) };
      }
    }
    return null;
  }

  function apply({ puzzle, state, action }) {
    const board = puzzle.board;
    const parsed = parseAction(action);
    if (!parsed) return null;
    const { pieceId } = parsed;

    if (parsed.kind === "place") {
      const piece = findPiece(board, pieceId);
      if (!piece) return null;
      let rotation;
      let placed = state.placed;
      let remaining = state.remaining;
      let rotations = state.rotations;
      const trayIndex = state.remaining.indexOf(pieceId);
      const placedIndex = state.placed.findIndex((placement) => placement.piece === pieceId);
      if (trayIndex >= 0) {
        rotation = trayRotation(state, pieceId);
        remaining = remaining.filter((_, index) => index !== trayIndex);
        rotations = rotations.filter(([id]) => id !== pieceId);
      } else if (placedIndex >= 0) {
        // Picked up from the board and placed again, keeping its rotation.
        rotation = state.placed[placedIndex].rotation;
        placed = placed.filter((_, index) => index !== placedIndex);
      } else {
        return null;
      }
      const placement = { piece: pieceId, x: parsed.x, y: parsed.y, rotation };
      if (!inBounds(board, placementCells(board, placement))) return null;
      return { placed: placed.concat([placement]).sort((first, second) => byId(first.piece, second.piece)), remaining, rotations };
    }

    const step = parsed.kind === "rotatecw" ? 90 : 270;
    if (state.remaining.includes(pieceId)) {
      const rotation = (trayRotation(state, pieceId) + step) % 360;
      const rotations = state.rotations.some(([id]) => id === pieceId)
        ? state.rotations.map(([id, current]) => [id, id === pieceId ? rotation : current])
        : state.rotations.concat([[pieceId, rotation]]).sort(([first], [second]) => byId(first, second));
      return { placed: state.placed, remaining: state.remaining, rotations };
    }
    const placedIndex = state.placed.findIndex((placement) => placement.piece === pieceId);
    if (placedIndex < 0 || !findPiece(board, pieceId)) return null;
    const rotated = { ...state.placed[placedIndex], rotation: (state.placed[placedIndex].rotation + step) % 360 };
    if (!inBounds(board, placementCells(board, rotated))) return null;
    return {
      placed: state.placed.map((placement, index) => (index === placedIndex ? rotated : placement)),
      remaining: state.remaining,
      rotations: state.rotations,
    };
  }

  function isSolved({ puzzle, state }) {
    if (state.remaining.length) return false;
    const board = puzzle.board;
    const occupied = new Set();
    for (const placement of state.placed) {
      for (const [x, y] of placementCells(board, placement)) {
        const key = `${x},${y}`;
        if (occupied.has(key)) return false;
        occupied.add(key);
      }
    }
    return occupied.size === board.target.length && board.target.every(([x, y]) => occupied.has(`${x},${y}`));
  }

  // ---------------------------------------------------------------------------
  // Rendering (render.rs). The board's tray layout comes from the engine (the exporter's encode_board).
  // ---------------------------------------------------------------------------

  function pieceIndex(pieceId) {
    return /^P\d+$/.test(pieceId) ? Number(pieceId.slice(1)) : 0;
  }

  function pieceColor(board, pieceId) {
    const index = pieceIndex(pieceId);
    const paletteIndex = index < board.colors.length ? board.colors[index] : index;
    return PIECE_COLORS[paletteIndex % PIECE_COLORS.length];
  }

  function pieceLabel(pieceId) {
    const index = pieceIndex(pieceId);
    return index < 10 ? String(index) : String.fromCharCode(65 + index - 10);
  }

  const luminance = ([red, green, blue]) => 0.299 * red + 0.587 * green + 0.114 * blue;
  const contrastColor = (color) => (luminance(color) > 128 ? "rgb(0, 0, 0)" : "rgb(255, 255, 255)");

  const pieceFontHeight = (cellSize) => Math.max(f32(cellSize * f32(0.75)), 10);

  // The ✕ the engine draws on a piece's anchor cell (one-pixel diagonals, cellSize / 5 in from the corners).
  function drawAnchor({ context, left, top, cellSize, color }) {
    const inset = Math.trunc(cellSize / 5);
    context.strokeStyle = color;
    context.lineWidth = 1.25;
    context.beginPath();
    context.moveTo(left + inset, top + inset);
    context.lineTo(left + cellSize - inset, top + cellSize - inset);
    context.moveTo(left + cellSize - inset, top + inset);
    context.lineTo(left + inset, top + cellSize - inset);
    context.stroke();
  }

  function boundingBox(cells) {
    const xs = cells.map(([x]) => x);
    const ys = cells.map(([, y]) => y);
    return { minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys) };
  }

  // A tray piece's cells in render space: drawn at its cached tray corner in its current rotation.
  function trayGeometry(board, state, pieceId) {
    const pieceSlot = board.pieces.findIndex((piece) => piece.id === pieceId);
    const [left, top] = board.tray.positions[pieceSlot];
    const cells = rotatedCells(board.pieces[pieceSlot].cells, trayRotation(state, pieceId));
    const { minX, minY, maxX, maxY } = boundingBox(cells);
    return { left, top, cells, minX, minY, width: maxX - minX + 1, height: maxY - minY + 1 };
  }

  function renderEngine({ context, board, state }) {
    const cellSize = board.tray.cellSize;
    const boardPixelWidth = board.width * cellSize;
    const boardPixelHeight = board.height * cellSize;
    context.fillStyle = rgb(BACKGROUND_COLOR);
    context.fillRect(0, 0, RENDER_SIZE, RENDER_SIZE);
    context.fillStyle = rgb(TARGET_COLOR);
    context.fillRect(BOARD_ORIGIN, BOARD_ORIGIN, boardPixelWidth, boardPixelHeight);

    for (const placement of state.placed) {
      const color = pieceColor(board, placement.piece);
      const cells = placementCells(board, placement);
      context.fillStyle = rgb(color);
      for (const [x, y] of cells) context.fillRect(BOARD_ORIGIN + x * cellSize, BOARD_ORIGIN + y * cellSize, cellSize, cellSize);
      const { minX, maxX, minY, maxY } = boundingBox(cells);
      drawEngineText({
        context,
        text: pieceLabel(placement.piece),
        centerX: BOARD_ORIGIN + Math.trunc(((minX + maxX + 1) * cellSize) / 2),
        centerY: BOARD_ORIGIN + Math.trunc(((minY + maxY + 1) * cellSize) / 2),
        fontHeight: pieceFontHeight(cellSize),
        color: contrastColor(color),
      });
      drawAnchor({ context, left: BOARD_ORIGIN + placement.x * cellSize, top: BOARD_ORIGIN + placement.y * cellSize, cellSize, color: contrastColor(color) });
    }

    context.fillStyle = rgb(GRID_COLOR);
    for (let column = 0; column <= board.width; column += 1) context.fillRect(BOARD_ORIGIN + column * cellSize, BOARD_ORIGIN, 1, boardPixelHeight);
    for (let row = 0; row <= board.height; row += 1) context.fillRect(BOARD_ORIGIN, BOARD_ORIGIN + row * cellSize, boardPixelWidth, 1);

    const axisColor = rgb(LABEL_COLOR);
    for (let column = 0; column < board.width; column += 1) {
      drawEngineText({ context, text: String(column), centerX: BOARD_ORIGIN + column * cellSize + Math.trunc(cellSize / 2), centerY: BOARD_ORIGIN - Math.trunc(LABEL_GAP / 2), fontHeight: AXIS_FONT_HEIGHT, color: axisColor });
    }
    for (let row = 0; row < board.height; row += 1) {
      drawEngineText({ context, text: String(row), centerX: BOARD_ORIGIN - Math.trunc(LABEL_GAP / 2), centerY: BOARD_ORIGIN + row * cellSize + Math.trunc(cellSize / 2), fontHeight: AXIS_FONT_HEIGHT, color: axisColor });
    }

    for (const piece of board.pieces) {
      if (!state.remaining.includes(piece.id)) continue;
      const { left, top, cells, minX, minY, width, height } = trayGeometry(board, state, piece.id);
      const color = pieceColor(board, piece.id);
      context.fillStyle = rgb(color);
      for (const [x, y] of cells) context.fillRect(left + (x - minX) * cellSize, top + (y - minY) * cellSize, cellSize, cellSize);
      context.fillStyle = rgb(GRID_COLOR);
      for (let column = 0; column <= width; column += 1) {
        const lineX = left + column * cellSize;
        if (lineX < RENDER_SIZE) context.fillRect(lineX, top, 1, Math.min(height * cellSize, RENDER_SIZE - top));
      }
      for (let row = 0; row <= height; row += 1) {
        const lineY = top + row * cellSize;
        if (lineY < RENDER_SIZE) context.fillRect(left, lineY, Math.min(width * cellSize, RENDER_SIZE - left), 1);
      }
      const fontHeight = pieceFontHeight(cellSize);
      drawEngineText({
        context,
        text: pieceLabel(piece.id),
        centerX: left + Math.trunc((width * cellSize) / 2),
        centerY: top + height * cellSize + Math.trunc(Math.trunc(fontHeight) / 2) + 2,
        fontHeight,
        color: rgb(LABEL_COLOR),
      });
      drawAnchor({ context, left: left - minX * cellSize, top: top - minY * cellSize, cellSize, color: contrastColor(color) });
    }
  }

  // ---------------------------------------------------------------------------
  // Interaction overlays: the selected piece's outline, and where it would land.
  // ---------------------------------------------------------------------------

  // Outline of a set of cells (edges not shared with another cell of the set), cells at (left + x * size, ...).
  function strokeCellOutline({ context, cells, left, top, cellSize, color, lineWidth }) {
    const keys = new Set(cells.map(([x, y]) => `${x},${y}`));
    context.strokeStyle = color;
    context.lineWidth = lineWidth;
    context.lineCap = "round";
    context.beginPath();
    for (const [x, y] of cells) {
      const cellLeft = left + x * cellSize;
      const cellTop = top + y * cellSize;
      if (!keys.has(`${x},${y - 1}`)) { context.moveTo(cellLeft, cellTop); context.lineTo(cellLeft + cellSize, cellTop); }
      if (!keys.has(`${x},${y + 1}`)) { context.moveTo(cellLeft, cellTop + cellSize); context.lineTo(cellLeft + cellSize, cellTop + cellSize); }
      if (!keys.has(`${x - 1},${y}`)) { context.moveTo(cellLeft, cellTop); context.lineTo(cellLeft, cellTop + cellSize); }
      if (!keys.has(`${x + 1},${y}`)) { context.moveTo(cellLeft + cellSize, cellTop); context.lineTo(cellLeft + cellSize, cellTop + cellSize); }
    }
    context.stroke();
  }

  // The selected piece's cells relative to its anchor, in its current rotation.
  function currentCells(board, state, pieceId) {
    const placement = state.placed.find((candidate) => candidate.piece === pieceId);
    const rotation = placement ? placement.rotation : trayRotation(state, pieceId);
    return rotatedCells(findPiece(board, pieceId).cells, rotation);
  }

  function drawLanding({ context, board, state, pieceId, anchorX, anchorY }) {
    const cellSize = board.tray.cellSize;
    const cells = currentCells(board, state, pieceId);
    const fits = inBounds(board, cells.map(([dx, dy]) => [anchorX + dx, anchorY + dy]));
    const visibleCells = cells
      .map(([dx, dy]) => [anchorX + dx, anchorY + dy])
      .filter(([x, y]) => x >= 0 && x < board.width && y >= 0 && y < board.height);
    context.save();
    context.globalAlpha = fits ? 0.45 : 0.3;
    context.fillStyle = fits ? rgb(pieceColor(board, pieceId)) : "rgb(194, 59, 59)";
    for (const [x, y] of visibleCells) context.fillRect(BOARD_ORIGIN + x * cellSize, BOARD_ORIGIN + y * cellSize, cellSize, cellSize);
    context.restore();
    if (fits) {
      strokeCellOutline({ context, cells: visibleCells, left: BOARD_ORIGIN, top: BOARD_ORIGIN, cellSize, color: SELECTION_COLOR, lineWidth: 2 });
      drawAnchor({ context, left: BOARD_ORIGIN + anchorX * cellSize, top: BOARD_ORIGIN + anchorY * cellSize, cellSize, color: SELECTION_COLOR });
    }
  }

  function renderOverlay({ context, board, state, ui }) {
    const cellSize = board.tray.cellSize;
    if (!ui.selectedPieceId) return;
    const pieceId = ui.selectedPieceId;
    const placement = state.placed.find((candidate) => candidate.piece === pieceId);
    if (ui.drag && ui.drag.isDragging) {
      // A tray piece being dragged is lifted out of its spot.
      if (!placement) {
        const { left, top, width, height } = trayGeometry(board, state, pieceId);
        context.save();
        context.globalAlpha = 0.75;
        context.fillStyle = rgb(BACKGROUND_COLOR);
        context.fillRect(left, top, width * cellSize + 1, height * cellSize + 1);
        context.restore();
      }
      if (ui.drag.landing) drawLanding({ context, board, state, pieceId, anchorX: ui.drag.landing.x, anchorY: ui.drag.landing.y });
      // The piece follows the pointer, anchor cell at the grab offset.
      const cells = currentCells(board, state, pieceId);
      const anchorLeft = ui.drag.pointer.x - ui.drag.grabOffset.x;
      const anchorTop = ui.drag.pointer.y - ui.drag.grabOffset.y;
      context.save();
      context.globalAlpha = 0.7;
      context.fillStyle = rgb(pieceColor(board, pieceId));
      for (const [dx, dy] of cells) context.fillRect(anchorLeft + dx * cellSize, anchorTop + dy * cellSize, cellSize, cellSize);
      context.restore();
      strokeCellOutline({ context, cells, left: anchorLeft, top: anchorTop, cellSize, color: SELECTION_COLOR, lineWidth: 2.5 });
      return;
    }
    if (placement) {
      strokeCellOutline({ context, cells: placementCells(board, placement), left: BOARD_ORIGIN, top: BOARD_ORIGIN, cellSize, color: SELECTION_COLOR, lineWidth: 3.5 });
    } else {
      const { left, top, cells, minX, minY } = trayGeometry(board, state, pieceId);
      strokeCellOutline({ context, cells, left: left - minX * cellSize, top: top - minY * cellSize, cellSize, color: SELECTION_COLOR, lineWidth: 3.5 });
    }
    if (ui.hoverCell) drawLanding({ context, board, state, pieceId, anchorX: ui.hoverCell.x, anchorY: ui.hoverCell.y });
  }

  function render({ context, puzzle, state, ui }) {
    renderEngine({ context, board: puzzle.board, state });
    if (ui && !isSolved({ puzzle, state })) renderOverlay({ context, board: puzzle.board, state, ui });
  }

  // ---------------------------------------------------------------------------
  // Input: drag a piece onto the grid, or select it and click the cell for its anchor.
  // ---------------------------------------------------------------------------

  function boardCellAt(board, point) {
    const cellSize = board.tray.cellSize;
    const x = Math.floor((point.x - BOARD_ORIGIN) / cellSize);
    const y = Math.floor((point.y - BOARD_ORIGIN) / cellSize);
    return x >= 0 && x < board.width && y >= 0 && y < board.height ? { x, y } : null;
  }

  // The piece under a point (the topmost placed piece, else a tray piece), with the render-space corner of
  // its anchor cell.
  function pieceAt(board, state, point) {
    const cellSize = board.tray.cellSize;
    const cell = boardCellAt(board, point);
    if (cell) {
      for (const placement of [...state.placed].reverse()) {
        if (placementCells(board, placement).some(([x, y]) => x === cell.x && y === cell.y)) {
          return { pieceId: placement.piece, anchorLeft: BOARD_ORIGIN + placement.x * cellSize, anchorTop: BOARD_ORIGIN + placement.y * cellSize };
        }
      }
    }
    for (const pieceId of state.remaining) {
      const { left, top, cells, minX, minY } = trayGeometry(board, state, pieceId);
      const anchorLeft = left - minX * cellSize;
      const anchorTop = top - minY * cellSize;
      const isHit = cells.some(([dx, dy]) => {
        const cellLeft = anchorLeft + dx * cellSize;
        const cellTop = anchorTop + dy * cellSize;
        return point.x >= cellLeft && point.x < cellLeft + cellSize && point.y >= cellTop && point.y < cellTop + cellSize;
      });
      if (isHit) return { pieceId, anchorLeft, anchorTop };
    }
    return null;
  }

  function mount({ puzzle, canvas, controls, act, getState, ui, redraw, toLogical }) {
    const board = puzzle.board;
    const cellSize = board.tray.cellSize;

    const pieceGroup = document.createElement("div");
    pieceGroup.className = "play-group";
    const pieceGroupLabel = document.createElement("span");
    pieceGroupLabel.className = "play-group-label";
    pieceGroupLabel.textContent = "Piece";
    pieceGroup.append(pieceGroupLabel);
    const pieceButtons = new Map();
    for (const piece of board.pieces) {
      const pieceButton = button({ label: pieceLabel(piece.id), title: `Select piece ${pieceLabel(piece.id)} (${piece.id})`, className: "is-swatch", onClick: () => select(piece.id) });
      const color = pieceColor(board, piece.id);
      pieceButton.style.background = rgb(color);
      pieceButton.style.color = contrastColor(color);
      pieceButtons.set(piece.id, pieceButton);
      pieceGroup.append(pieceButton);
    }

    const rotateGroup = document.createElement("div");
    rotateGroup.className = "play-group";
    const rotateGroupLabel = document.createElement("span");
    rotateGroupLabel.className = "play-group-label";
    rotateGroupLabel.textContent = "Rotate";
    const counterClockwiseButton = button({ label: "↺", title: "Rotate counterclockwise (Q)", onClick: () => rotate("rotateccw") });
    const clockwiseButton = button({ label: "↻", title: "Rotate clockwise (R)", onClick: () => rotate("rotatecw") });
    rotateGroup.append(rotateGroupLabel, counterClockwiseButton, clockwiseButton);
    controls.append(pieceGroup, rotateGroup);

    function syncControls() {
      for (const [pieceId, pieceButton] of pieceButtons) pieceButton.classList.toggle("is-selected", pieceId === ui.selectedPieceId);
      counterClockwiseButton.disabled = !ui.selectedPieceId;
      clockwiseButton.disabled = !ui.selectedPieceId;
    }

    function select(pieceId) {
      ui.selectedPieceId = pieceId;
      syncControls();
      redraw();
    }

    function rotate(kind) {
      if (ui.selectedPieceId) act(`${kind}('${ui.selectedPieceId}')`);
    }

    function place(pieceId, { x, y }) {
      return act(`place('${pieceId}', ${x}, ${y})`);
    }

    function onPointerDown(event) {
      if (event.button !== 0) return;
      const point = toLogical(event);
      const state = getState();
      const hit = pieceAt(board, state, point);
      if (hit) {
        select(hit.pieceId);
        ui.drag = {
          pointerId: event.pointerId,
          start: point,
          pointer: point,
          grabOffset: { x: point.x - hit.anchorLeft, y: point.y - hit.anchorTop },
          isDragging: false,
          landing: null,
        };
        canvas.setPointerCapture(event.pointerId);
        return;
      }
      const cell = boardCellAt(board, point);
      if (cell && ui.selectedPieceId) place(ui.selectedPieceId, cell);
      else if (!cell) select(null);
    }

    function landingFor({ point, grabOffset }) {
      // The anchor snaps to the nearest cell; a drop counts when the pointer is over (or near) the board.
      const nearBoard = point.x >= BOARD_ORIGIN - cellSize / 2 && point.x < BOARD_ORIGIN + (board.width + 0.5) * cellSize
        && point.y >= BOARD_ORIGIN - cellSize / 2 && point.y < BOARD_ORIGIN + (board.height + 0.5) * cellSize;
      if (!nearBoard) return null;
      return {
        x: Math.round((point.x - grabOffset.x - BOARD_ORIGIN) / cellSize),
        y: Math.round((point.y - grabOffset.y - BOARD_ORIGIN) / cellSize),
      };
    }

    function onPointerMove(event) {
      const point = toLogical(event);
      if (ui.drag && ui.drag.pointerId === event.pointerId) {
        ui.drag.pointer = point;
        if (!ui.drag.isDragging && Math.hypot(point.x - ui.drag.start.x, point.y - ui.drag.start.y) * (canvas.getBoundingClientRect().width / RENDER_SIZE) > DRAG_THRESHOLD_PX) {
          ui.drag.isDragging = true;
        }
        if (ui.drag.isDragging) {
          ui.drag.landing = landingFor({ point, grabOffset: ui.drag.grabOffset });
          ui.hoverCell = null;
          redraw();
        }
        return;
      }
      const hoverCell = ui.selectedPieceId && !pieceAt(board, getState(), point) ? boardCellAt(board, point) : null;
      if ((hoverCell && ui.hoverCell && hoverCell.x === ui.hoverCell.x && hoverCell.y === ui.hoverCell.y) || (!hoverCell && !ui.hoverCell)) return;
      ui.hoverCell = hoverCell;
      redraw();
    }

    function onPointerUp(event) {
      if (!ui.drag || ui.drag.pointerId !== event.pointerId) return;
      const drag = ui.drag;
      ui.drag = null;
      if (drag.isDragging) {
        const landing = landingFor({ point: toLogical(event), grabOffset: drag.grabOffset });
        if (!landing || !place(ui.selectedPieceId, landing)) redraw();
      } else {
        redraw();
      }
    }

    function onPointerCancel() {
      ui.drag = null;
      redraw();
    }

    function onPointerLeave() {
      if (ui.hoverCell && !ui.drag) {
        ui.hoverCell = null;
        redraw();
      }
    }

    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointercancel", onPointerCancel);
    canvas.addEventListener("pointerleave", onPointerLeave);
    syncControls();

    return {
      onKey(event) {
        if (event.metaKey || event.ctrlKey || event.altKey) return false;
        const labelled = board.pieces.find((piece) => pieceLabel(piece.id) === event.key);
        if (labelled) {
          select(labelled.id);
          return true;
        }
        if (event.key === "r" || event.key === "R" || event.key === "e" || event.key === "E") {
          rotate("rotatecw");
          return true;
        }
        if (event.key === "q" || event.key === "Q") {
          rotate("rotateccw");
          return true;
        }
        // Arrow keys nudge a placed piece one cell.
        const direction = event.key.startsWith("Arrow") ? keyDirection(event) : null;
        const placement = direction && getState().placed.find((candidate) => candidate.piece === ui.selectedPieceId);
        if (!placement) return false;
        const [dx, dy] = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] }[direction];
        place(placement.piece, { x: placement.x + dx, y: placement.y + dy });
        return true;
      },
      onStateChange() {
        syncControls();
      },
      destroy() {
        canvas.removeEventListener("pointerdown", onPointerDown);
        canvas.removeEventListener("pointermove", onPointerMove);
        canvas.removeEventListener("pointerup", onPointerUp);
        canvas.removeEventListener("pointercancel", onPointerCancel);
        canvas.removeEventListener("pointerleave", onPointerLeave);
      },
    };
  }

  window.PuzzleGames.tangram = {
    goal: "Fill the grey silhouette with all of the colored pieces.",
    help: "Drag and drop pieces onto the grid. Use controls above to rotate pieces.",
    initialState: ({ puzzle }) => puzzle.board.state,
    apply,
    isSolved,
    render,
    mount,
  };
})();
