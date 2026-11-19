/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2026 Apple Inc. All Rights Reserved.
 */
// Jigsaw (engine/games/jigsaw): rotate each piece upright and place it in its cell; the finished picture may
// face any of the four ways. Actions: place(piece, cell), rotatecw(piece), rotateccw(piece), remove(piece).
(function () {
  "use strict";

  const { RENDER_SIZE, button, rgb, drawEngineText } = window.PuzzleKit;

  const BACKGROUND_COLOR = [0xf0, 0xf0, 0xf0];
  const EMPTY_CELL_COLOR = [0xd0, 0xd0, 0xd0];
  const GRID_LINE_COLOR = [0x80, 0x80, 0x80];
  const TRAY_COLOR = [0xe0, 0xe0, 0xe0];
  const PIECE_OUTLINE_COLOR = [0x40, 0x40, 0x40];
  const CELL_LABEL_COLOR = [0x60, 0x60, 0x60];
  const PIECE_LABEL_COLOR = [0x30, 0x30, 0x30];
  const PLACED_PIECE_LABEL_COLOR = [0xc0, 0xc0, 0xc0];
  const SELECTION_COLOR = "#4a3aa7";
  const DROP_TARGET_FILL = "rgba(74, 58, 167, 0.22)";

  const FLAT = 0;
  const TAB = 1;
  const SLOT = 2;
  const ORIENTATIONS = [0, 90, 180, 270];
  const DRAG_THRESHOLD = 6;
  const DOUBLE_CLICK_MS = 400;

  // ---------------------------------------------------------------------------
  // Rules (engine/games/jigsaw/mod.rs, state.rs).
  // ---------------------------------------------------------------------------

  // A, B, ..., Z, AA, AB, ...
  function cellLabel(cellIndex) {
    let label = "";
    let remaining = cellIndex;
    for (;;) {
      label = String.fromCharCode(65 + (remaining % 26)) + label;
      if (remaining < 26) return label;
      remaining = Math.floor(remaining / 26) - 1;
    }
  }

  function parseCellLabel(text) {
    if (!/^[A-Z]+$/.test(text)) return null;
    let index = 0;
    for (const character of text) index = index * 26 + (character.charCodeAt(0) - 65) + 1;
    return index - 1;
  }

  // Rust's usize parse: digits with an optional leading plus.
  function parsePieceId({ text, pieceCount }) {
    if (!/^\+?[0-9]+$/.test(text)) return null;
    const pieceId = Number(text);
    return pieceId < pieceCount ? pieceId : null;
  }

  function parseAction({ action, pieceCount }) {
    const placeMatch = /^place\((.*)\)$/s.exec(action);
    if (placeMatch) {
      const separator = placeMatch[1].indexOf(", ");
      if (separator < 0) return null;
      const pieceId = parsePieceId({ text: placeMatch[1].slice(0, separator), pieceCount });
      const cell = parseCellLabel(placeMatch[1].slice(separator + 2));
      if (pieceId === null || cell === null || cell >= pieceCount) return null;
      return { kind: "place", pieceId, cell };
    }
    const pieceMatch = /^(rotatecw|rotateccw|remove)\((.*)\)$/s.exec(action);
    if (!pieceMatch) return null;
    const pieceId = parsePieceId({ text: pieceMatch[2], pieceCount });
    return pieceId === null ? null : { kind: pieceMatch[1], pieceId };
  }

  function apply({ puzzle, state, action }) {
    const gridSize = puzzle.board.gridSize;
    const parsed = parseAction({ action, pieceCount: gridSize * gridSize });
    if (!parsed) return null;
    const { kind, pieceId } = parsed;
    if (kind === "rotatecw" || kind === "rotateccw") {
      const rotations = state.rotations.slice();
      rotations[pieceId] = (rotations[pieceId] + (kind === "rotatecw" ? 90 : 270)) % 360;
      return { grid: state.grid, unplaced: state.unplaced, rotations };
    }
    const isUnplaced = state.unplaced.includes(pieceId);
    if (kind === "place") {
      if (state.grid[parsed.cell] !== null || !isUnplaced) return null;
      const grid = state.grid.slice();
      grid[parsed.cell] = pieceId;
      return { grid, unplaced: state.unplaced.filter((unplacedId) => unplacedId !== pieceId), rotations: state.rotations };
    }
    if (isUnplaced) return null;
    const grid = state.grid.map((occupant) => (occupant === pieceId ? null : occupant));
    const unplaced = state.unplaced.concat([pieceId]).sort((first, second) => first - second);
    return { grid, unplaced, rotations: state.rotations };
  }

  // Where a piece belongs when the whole picture is turned by theta.
  function targetCell({ pieceId, gridSize, theta }) {
    const row = Math.floor(pieceId / gridSize);
    const col = pieceId % gridSize;
    if (theta === 90) return col * gridSize + (gridSize - 1 - row);
    if (theta === 180) return (gridSize - 1 - row) * gridSize + (gridSize - 1 - col);
    if (theta === 270) return (gridSize - 1 - col) * gridSize + row;
    return pieceId;
  }

  function isSolved({ puzzle, state }) {
    if (state.unplaced.length) return false;
    const gridSize = puzzle.board.gridSize;
    return ORIENTATIONS.some((theta) => state.rotations.every((rotation, pieceId) => (
      rotation === theta && state.grid[targetCell({ pieceId, gridSize, theta })] === pieceId
    )));
  }

  // [top, right, bottom, left] of a cell of the solved picture.
  function edgeTypes({ board, cellIndex }) {
    const gridSize = board.gridSize;
    const row = Math.floor(cellIndex / gridSize);
    const col = cellIndex % gridSize;
    const horizontal = board.horizontalTabs;
    const vertical = board.verticalTabs;
    return [
      row === 0 ? FLAT : (horizontal[(row - 1) * gridSize + col] ? SLOT : TAB),
      col === gridSize - 1 ? FLAT : (vertical[row * (gridSize - 1) + col] ? TAB : SLOT),
      row === gridSize - 1 ? FLAT : (horizontal[row * gridSize + col] ? TAB : SLOT),
      col === 0 ? FLAT : (vertical[row * (gridSize - 1) + (col - 1)] ? SLOT : TAB),
    ];
  }

  function rotatedEdgeTypes({ board, pieceId, rotation }) {
    const [top, right, bottom, left] = edgeTypes({ board, cellIndex: pieceId });
    if (rotation === 90) return [left, top, right, bottom];
    if (rotation === 180) return [bottom, left, top, right];
    if (rotation === 270) return [right, bottom, left, top];
    return [top, right, bottom, left];
  }

  // ---------------------------------------------------------------------------
  // Layout (engine/games/jigsaw/render.rs), with the engine's f32 arithmetic.
  // ---------------------------------------------------------------------------

  const multiplyF32 = (value, factor) => Math.fround(Math.fround(value) * Math.fround(factor));
  const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

  // The tray slots: N² uniform boxes (sized for tabs on every side, plus a label) packed greedily around the grid.
  function tryGreedyPack({ gridSize, cellSize }) {
    const tabRadius = Math.trunc(multiplyF32(cellSize, 0.18));
    const labelHeight = Math.trunc(clamp(multiplyF32(cellSize, 0.3), 8, 22)) + 4;
    const pad = Math.max(Math.floor(cellSize / 8), 2);
    const scanStep = Math.max(Math.floor(cellSize / 3), 1);
    const gridPixels = cellSize * gridSize;
    const boxWidth = tabRadius + cellSize + tabRadius;
    const boxHeight = boxWidth + labelHeight;
    const occupied = [[0, 0, gridPixels + pad, gridPixels + pad]];
    const positions = [];
    for (let slot = 0; slot < gridSize * gridSize; slot += 1) {
      let isPlaced = false;
      scan: for (let y = pad; y < RENDER_SIZE - boxHeight; y += scanStep) {
        for (let x = pad; x < RENDER_SIZE - boxWidth; x += scanStep) {
          const overlaps = occupied.some(([left, top, width, height]) => (
            x < left + width && x + boxWidth > left && y < top + height && y + boxHeight > top
          ));
          if (!overlaps) {
            occupied.push([x, y, boxWidth, boxHeight]);
            positions.push([x + tabRadius, y + tabRadius]);
            isPlaced = true;
            break scan;
          }
        }
      }
      if (!isPlaced) return null;
    }
    return positions;
  }

  const layoutCache = new Map();
  function layoutFor(gridSize) {
    if (layoutCache.has(gridSize)) return layoutCache.get(gridSize);
    let cellSize = Math.max(1, Math.floor(RENDER_SIZE / (gridSize + 1)));
    let positions = null;
    for (let attempt = 0; attempt < 8 && !positions; attempt += 1) {
      positions = tryGreedyPack({ gridSize, cellSize });
      if (!positions) cellSize = Math.max(1, Math.trunc(multiplyF32(cellSize, 0.9)));
    }
    positions = positions || tryGreedyPack({ gridSize, cellSize }) || Array.from({ length: gridSize * gridSize }, () => [0, 0]);
    const trayFontHeight = clamp(multiplyF32(cellSize, 0.3), 8, 22);
    const layout = {
      gridSize,
      cellSize,
      tabRadius: Math.trunc(multiplyF32(cellSize, 0.18)),
      gridPixels: cellSize * gridSize,
      trayPositions: positions,
      trayFontHeight,
      trayLabelGap: Math.trunc(trayFontHeight) + 4,
      gridFontHeight: clamp(multiplyF32(cellSize, 0.25), 8, 20),
    };
    layoutCache.set(gridSize, layout);
    return layout;
  }

  // ---------------------------------------------------------------------------
  // Rendering. Pixels are rasterized like the engine at `scale` device pixels per render pixel (the shape
  // test runs at device-pixel centers, so tabs stay smooth on dense screens; the picture is sampled per
  // render pixel); labels and the selection are drawn on top as vectors.
  // ---------------------------------------------------------------------------

  // The engine's in_jigsaw_shape: (dx, dy) relative to the cell origin, edges [top, right, bottom, left].
  function inJigsawShape({ dx, dy, cellSize, tabRadius, edges }) {
    const middle = Math.floor(cellSize / 2);
    const radiusSquared = tabRadius * tabRadius;
    const withinColumns = dx >= 0 && dx < cellSize;
    const withinRows = dy >= 0 && dy < cellSize;
    let inside = withinColumns && withinRows;
    const topDistance = (dx - middle) ** 2 + dy ** 2;
    if (edges[0] === TAB && dy < 0 && topDistance <= radiusSquared && withinColumns) inside = true;
    else if (edges[0] === SLOT && dy >= 0 && topDistance <= radiusSquared) inside = false;
    const bottomDistance = (dx - middle) ** 2 + (dy - cellSize) ** 2;
    if (edges[2] === TAB && dy >= cellSize && bottomDistance <= radiusSquared && withinColumns) inside = true;
    else if (edges[2] === SLOT && dy < cellSize && bottomDistance <= radiusSquared) inside = false;
    const leftDistance = dx ** 2 + (dy - middle) ** 2;
    if (edges[3] === TAB && dx < 0 && leftDistance <= radiusSquared && withinRows) inside = true;
    else if (edges[3] === SLOT && dx >= 0 && leftDistance <= radiusSquared) inside = false;
    const rightDistance = (dx - cellSize) ** 2 + (dy - middle) ** 2;
    if (edges[1] === TAB && dx >= cellSize && rightDistance <= radiusSquared && withinRows) inside = true;
    else if (edges[1] === SLOT && dx < cellSize && rightDistance <= radiusSquared) inside = false;
    return inside;
  }

  // Which device pixels of a piece's box (tabs included) are inside its shape, and which are on its outline
  // (inside, with a neighbor one render pixel away outside).
  const maskCache = new Map();
  function shapeMask({ edges, layout, scale }) {
    const { cellSize, tabRadius } = layout;
    const key = `${edges.join("")}|${cellSize}|${scale}`;
    if (maskCache.has(key)) return maskCache.get(key);
    const extent = (cellSize + 2 * tabRadius) * scale;
    const inside = new Uint8Array(extent * extent);
    const outline = new Uint8Array(extent * extent);
    const isInside = (dx, dy) => inJigsawShape({ dx, dy, cellSize, tabRadius, edges });
    for (let row = 0; row < extent; row += 1) {
      const dy = (row - tabRadius * scale + 0.5) / scale - 0.5;
      for (let col = 0; col < extent; col += 1) {
        const dx = (col - tabRadius * scale + 0.5) / scale - 0.5;
        if (!isInside(dx, dy)) continue;
        inside[row * extent + col] = 1;
        if (!isInside(dx - 1, dy) || !isInside(dx + 1, dy) || !isInside(dx, dy - 1) || !isInside(dx, dy + 1)) {
          outline[row * extent + col] = 1;
        }
      }
    }
    const mask = { extent, inside, outline };
    maskCache.set(key, mask);
    return mask;
  }

  const packColor = ([red, green, blue]) => (0xff000000 | (blue << 16) | (green << 8) | red) >>> 0;

  // The run-length encoded picture as packed RGBA.
  const pictureCache = new WeakMap();
  function decodePicture(picture) {
    if (pictureCache.has(picture)) return pictureCache.get(picture);
    const decode = (text) => Uint8Array.from(atob(text), (character) => character.charCodeAt(0));
    const colors = decode(picture.colors);
    const runs = decode(picture.runs);
    const pixels = new Uint32Array(picture.size * picture.size);
    let offset = 0;
    for (let index = 0; index < runs.length; index += 2) {
      const colorOffset = runs[index] * 3;
      pixels.fill(packColor([colors[colorOffset], colors[colorOffset + 1], colors[colorOffset + 2]]), offset, offset + runs[index + 1]);
      offset += runs[index + 1];
    }
    pictureCache.set(picture, pixels);
    return pixels;
  }

  // Draws a piece (its picture turned by `rotation`, cut to `edges`) with its cell origin at (originX, originY).
  function blitPiece({ target, board, layout, scale, originX, originY, edges, pieceId, rotation, outlineColor }) {
    const { cellSize, tabRadius, gridSize } = layout;
    const mask = shapeMask({ edges, layout, scale });
    const picture = decodePicture(board.picture);
    const pictureSize = board.picture.size;
    const sourceSize = pictureSize / gridSize;
    const sourceLeft = (pieceId % gridSize) * sourceSize;
    const sourceTop = Math.floor(pieceId / gridSize) * sourceSize;
    const startX = (originX - tabRadius) * scale;
    const startY = (originY - tabRadius) * scale;
    const packedOutline = outlineColor ? packColor(outlineColor) : 0;
    const last = cellSize - 1;
    for (let row = 0; row < mask.extent; row += 1) {
      const targetY = startY + row;
      if (targetY < 0 || targetY >= target.size) continue;
      const cellY = clamp(Math.floor(row / scale) - tabRadius, 0, last);
      for (let col = 0; col < mask.extent; col += 1) {
        const maskIndex = row * mask.extent + col;
        if (!mask.inside[maskIndex]) continue;
        const targetX = startX + col;
        if (targetX < 0 || targetX >= target.size) continue;
        if (outlineColor && mask.outline[maskIndex]) {
          target.pixels[targetY * target.size + targetX] = packedOutline;
          continue;
        }
        const cellX = clamp(Math.floor(col / scale) - tabRadius, 0, last);
        let sourceX;
        let sourceY;
        if (rotation === 90) {
          sourceX = Math.floor((cellY * sourceSize) / cellSize);
          sourceY = Math.floor(((last - cellX) * sourceSize) / cellSize);
        } else if (rotation === 180) {
          sourceX = Math.floor(((last - cellX) * sourceSize) / cellSize);
          sourceY = Math.floor(((last - cellY) * sourceSize) / cellSize);
        } else if (rotation === 270) {
          sourceX = Math.floor(((last - cellY) * sourceSize) / cellSize);
          sourceY = Math.floor((cellX * sourceSize) / cellSize);
        } else {
          sourceX = Math.floor((cellX * sourceSize) / cellSize);
          sourceY = Math.floor((cellY * sourceSize) / cellSize);
        }
        target.pixels[targetY * target.size + targetX] = picture[(sourceTop + sourceY) * pictureSize + sourceLeft + sourceX];
      }
    }
  }

  function createRaster(size) {
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const context = canvas.getContext("2d");
    const image = context.createImageData(size, size);
    return { canvas, context, image, size, pixels: new Uint32Array(image.data.buffer) };
  }

  function fillRaster({ target, scale, x, y, width, height, color }) {
    const packed = packColor(color);
    const right = Math.min(target.size, (x + width) * scale);
    const bottom = Math.min(target.size, (y + height) * scale);
    for (let row = Math.max(0, y * scale); row < bottom; row += 1) {
      target.pixels.fill(packed, row * target.size + Math.max(0, x * scale), row * target.size + right);
    }
  }

  // Everything the engine draws except text, in the engine's order.
  function rasterizeBoard({ board, state, layout, scale }) {
    const target = createRaster(RENDER_SIZE * scale);
    const { cellSize, gridPixels, gridSize, trayPositions } = layout;
    const fill = (x, y, width, height, color) => fillRaster({ target, scale, x, y, width, height, color });
    fill(0, 0, RENDER_SIZE, RENDER_SIZE, BACKGROUND_COLOR);
    fill(gridPixels, 0, RENDER_SIZE - gridPixels, gridPixels, TRAY_COLOR);
    fill(0, gridPixels, gridPixels, RENDER_SIZE - gridPixels, TRAY_COLOR);
    fill(gridPixels, gridPixels, RENDER_SIZE - gridPixels, RENDER_SIZE - gridPixels, TRAY_COLOR);
    state.grid.forEach((pieceId, cellIndex) => {
      if (pieceId === null) fill((cellIndex % gridSize) * cellSize, Math.floor(cellIndex / gridSize) * cellSize, cellSize, cellSize, EMPTY_CELL_COLOR);
    });
    // On the grid, a piece takes its cell's shape: tabs overlap the neighbors.
    state.grid.forEach((pieceId, cellIndex) => {
      if (pieceId === null) return;
      blitPiece({
        target, board, layout, scale, pieceId,
        originX: (cellIndex % gridSize) * cellSize,
        originY: Math.floor(cellIndex / gridSize) * cellSize,
        edges: edgeTypes({ board, cellIndex }),
        rotation: state.rotations[pieceId],
      });
    });
    for (let cellIndex = 0; cellIndex < gridSize * gridSize; cellIndex += 1) {
      const x = (cellIndex % gridSize) * cellSize;
      const y = Math.floor(cellIndex / gridSize) * cellSize;
      fill(x, y, cellSize, 1, GRID_LINE_COLOR);
      fill(x, y + cellSize - 1, cellSize, 1, GRID_LINE_COLOR);
      fill(x, y, 1, cellSize, GRID_LINE_COLOR);
      fill(x + cellSize - 1, y, 1, cellSize, GRID_LINE_COLOR);
    }
    trayPositions.forEach(([x, y], slot) => {
      const pieceId = board.trayOrder[slot];
      if (pieceId === undefined || !state.unplaced.includes(pieceId)) return;
      const rotation = state.rotations[pieceId];
      blitPiece({
        target, board, layout, scale, pieceId, rotation,
        originX: x,
        originY: y,
        edges: rotatedEdgeTypes({ board, pieceId, rotation }),
        outlineColor: PIECE_OUTLINE_COLOR,
      });
    });
    target.context.putImageData(target.image, 0, 0);
    return target.canvas;
  }

  function drawLabels({ context, board, state, layout }) {
    const { cellSize, gridSize, tabRadius, trayFontHeight, trayLabelGap, gridFontHeight } = layout;
    const halfCell = Math.floor(cellSize / 2);
    state.grid.forEach((pieceId, cellIndex) => {
      if (pieceId !== null) return;
      drawEngineText({
        context,
        centerX: (cellIndex % gridSize) * cellSize + halfCell,
        centerY: Math.floor(cellIndex / gridSize) * cellSize + halfCell,
        text: cellLabel(cellIndex),
        fontHeight: gridFontHeight,
        color: CELL_LABEL_COLOR,
      });
    });
    layout.trayPositions.forEach(([x, y], slot) => {
      const pieceId = board.trayOrder[slot];
      if (pieceId === undefined) return;
      if (state.unplaced.includes(pieceId)) {
        const bottomEdge = rotatedEdgeTypes({ board, pieceId, rotation: state.rotations[pieceId] })[2];
        const bottomOffset = bottomEdge === TAB ? tabRadius : 0;
        drawEngineText({
          context,
          centerX: x + halfCell,
          centerY: y + cellSize + bottomOffset + Math.floor(trayLabelGap / 2),
          text: `${pieceId}`,
          fontHeight: trayFontHeight,
          color: PIECE_LABEL_COLOR,
        });
      } else {
        drawEngineText({ context, centerX: x + halfCell, centerY: y + halfCell, text: `${pieceId}`, fontHeight: trayFontHeight, color: PLACED_PIECE_LABEL_COLOR });
      }
    });
  }

  // The outline of a jigsaw shape as a path (for highlights), cell origin at (x, y).
  function traceShape({ context, x, y, layout, edges }) {
    const { cellSize, tabRadius } = layout;
    const middle = Math.floor(cellSize / 2);
    context.beginPath();
    context.moveTo(x, y);
    if (edges[0] !== FLAT) {
      context.lineTo(x + middle - tabRadius, y);
      context.arc(x + middle, y, tabRadius, Math.PI, 0, edges[0] === SLOT);
    }
    context.lineTo(x + cellSize, y);
    if (edges[1] !== FLAT) {
      context.lineTo(x + cellSize, y + middle - tabRadius);
      context.arc(x + cellSize, y + middle, tabRadius, -Math.PI / 2, Math.PI / 2, edges[1] === SLOT);
    }
    context.lineTo(x + cellSize, y + cellSize);
    if (edges[2] !== FLAT) {
      context.lineTo(x + middle + tabRadius, y + cellSize);
      context.arc(x + middle, y + cellSize, tabRadius, 0, Math.PI, edges[2] === SLOT);
    }
    context.lineTo(x, y + cellSize);
    if (edges[3] !== FLAT) {
      context.lineTo(x, y + middle + tabRadius);
      context.arc(x, y + middle, tabRadius, Math.PI / 2, -Math.PI / 2, edges[3] === SLOT);
    }
    context.closePath();
  }

  // Where a piece is drawn: its cell origin and shape (the tray's slot and own shape, or its grid cell's).
  function pieceGeometry({ board, state, layout, pieceId }) {
    const { cellSize, gridSize } = layout;
    const cellIndex = state.grid.indexOf(pieceId);
    if (cellIndex >= 0) {
      return {
        x: (cellIndex % gridSize) * cellSize,
        y: Math.floor(cellIndex / gridSize) * cellSize,
        edges: edgeTypes({ board, cellIndex }),
        cellIndex,
      };
    }
    const [x, y] = layout.trayPositions[board.trayOrder.indexOf(pieceId)];
    return { x, y, edges: rotatedEdgeTypes({ board, pieceId, rotation: state.rotations[pieceId] }), cellIndex: null };
  }

  // The dragged piece in its own shape, as a canvas covering its box (tabs included).
  let dragImageCache = null;
  function dragImage({ board, state, layout, scale, pieceId }) {
    const rotation = state.rotations[pieceId];
    const key = `${pieceId}|${rotation}|${scale}`;
    if (dragImageCache?.key === key && dragImageCache.board === board) return dragImageCache.canvas;
    const { cellSize, tabRadius } = layout;
    const target = createRaster((cellSize + 2 * tabRadius) * scale);
    blitPiece({
      target, board, layout, scale, pieceId, rotation,
      originX: tabRadius,
      originY: tabRadius,
      edges: rotatedEdgeTypes({ board, pieceId, rotation }),
      outlineColor: PIECE_OUTLINE_COLOR,
    });
    target.context.putImageData(target.image, 0, 0);
    dragImageCache = { key, board, canvas: target.canvas };
    return target.canvas;
  }

  function drawInteraction({ context, board, state, layout, ui, scale }) {
    const { cellSize, gridSize, tabRadius } = layout;
    if (ui.targetCell !== null && ui.targetCell !== undefined && state.grid[ui.targetCell] === null) {
      context.fillStyle = DROP_TARGET_FILL;
      context.fillRect((ui.targetCell % gridSize) * cellSize, Math.floor(ui.targetCell / gridSize) * cellSize, cellSize, cellSize);
    }
    if (ui.selected !== null && ui.selected !== undefined) {
      const geometry = pieceGeometry({ board, state, layout, pieceId: ui.selected });
      traceShape({ context, x: geometry.x, y: geometry.y, layout, edges: geometry.edges });
      context.lineJoin = "round";
      context.lineWidth = 4;
      context.strokeStyle = SELECTION_COLOR;
      context.stroke();
      if (ui.drag) {
        context.fillStyle = "rgba(240, 240, 240, 0.6)";
        context.fill();
      }
    }
    if (ui.drag) {
      const left = ui.drag.x - ui.drag.grabX - tabRadius;
      const top = ui.drag.y - ui.drag.grabY - tabRadius;
      context.save();
      context.imageSmoothingEnabled = false;
      context.shadowColor = "rgba(0, 0, 0, 0.3)";
      context.shadowBlur = 14;
      context.shadowOffsetY = 6;
      context.drawImage(dragImage({ board, state, layout, scale, pieceId: ui.drag.pieceId }), left, top, cellSize + 2 * tabRadius, cellSize + 2 * tabRadius);
      context.restore();
    }
  }

  let rasterCache = null;
  function render({ context, puzzle, state, ui = {} }) {
    const board = puzzle.board;
    const layout = layoutFor(board.gridSize);
    const scale = Math.max(1, Math.round(context.getTransform ? context.getTransform().a : 1));
    if (!rasterCache || rasterCache.state !== state || rasterCache.board !== board || rasterCache.scale !== scale) {
      rasterCache = { state, board, scale, canvas: rasterizeBoard({ board, state, layout, scale }) };
    }
    context.save();
    context.imageSmoothingEnabled = false;
    context.drawImage(rasterCache.canvas, 0, 0, RENDER_SIZE, RENDER_SIZE);
    context.restore();
    drawLabels({ context, board, state, layout });
    drawInteraction({ context, board, state, layout, ui, scale });
  }

  // ---------------------------------------------------------------------------
  // Interaction: drag a piece into a cell (or click it, then the cell); click the selected piece to turn it.
  // ---------------------------------------------------------------------------

  function mount({ puzzle, canvas, controls, act, getState, ui, redraw, toLogical }) {
    const board = puzzle.board;
    const layout = layoutFor(board.gridSize);
    const { cellSize, gridSize, gridPixels, tabRadius } = layout;
    const pieceCount = gridSize * gridSize;
    ui.selected = null;
    ui.targetCell = null;
    ui.drag = null;
    let press = null;
    let lastClick = null; // { pieceId, time }: a second click on the same piece soon after rotates it

    const selectionLabel = document.createElement("span");
    selectionLabel.className = "play-group-label";
    const rotateCounterButton = button({ label: "↺ Rotate", title: "Rotate counter-clockwise (Shift+R)", onClick: () => rotateSelected("rotateccw") });
    const rotateClockwiseButton = button({ label: "↻ Rotate", title: "Rotate clockwise (R)", onClick: () => rotateSelected("rotatecw") });
    const removeButton = button({ label: "Remove", title: "Back to the tray (Delete)", onClick: removeSelected });
    const group = document.createElement("div");
    group.className = "play-group";
    group.append(selectionLabel, rotateCounterButton, rotateClockwiseButton, removeButton);
    controls.append(group);

    function updateControls() {
      const hasSelection = ui.selected !== null;
      selectionLabel.textContent = hasSelection ? `Piece ${ui.selected}` : "No piece selected";
      rotateCounterButton.disabled = !hasSelection;
      rotateClockwiseButton.disabled = !hasSelection;
      removeButton.disabled = !hasSelection || getState().unplaced.includes(ui.selected);
    }

    function select(pieceId) {
      ui.selected = pieceId;
      updateControls();
      redraw();
    }

    function rotateSelected(kind) {
      if (ui.selected !== null) act(`${kind}(${ui.selected})`);
    }

    function removeSelected() {
      if (ui.selected !== null && !getState().unplaced.includes(ui.selected)) act(`remove(${ui.selected})`);
    }

    // A placed piece moves by going back to the tray first, as the model would have to.
    function moveToCell({ pieceId, cellIndex }) {
      const state = getState();
      const currentCell = state.grid.indexOf(pieceId);
      if (currentCell === cellIndex) return;
      if (currentCell >= 0 && state.grid[cellIndex] === null && !act(`remove(${pieceId})`)) return;
      act(`place(${pieceId}, ${cellLabel(cellIndex)})`);
    }

    function cellAt({ x, y }) {
      if (x < 0 || y < 0 || x >= gridPixels || y >= gridPixels) return null;
      return Math.floor(y / cellSize) * gridSize + Math.floor(x / cellSize);
    }

    // The piece under a point: a grid cell's occupant, or a tray piece (its box, tabs and label included).
    function pieceAt(point) {
      const state = getState();
      const cellIndex = cellAt(point);
      if (cellIndex !== null) return state.grid[cellIndex];
      for (const [slot, [x, y]] of layout.trayPositions.entries()) {
        const pieceId = board.trayOrder[slot];
        if (!state.unplaced.includes(pieceId)) continue;
        if (point.x >= x - tabRadius && point.x < x + cellSize + tabRadius && point.y >= y - tabRadius && point.y < y + cellSize + tabRadius + layout.trayLabelGap) {
          return pieceId;
        }
      }
      return null;
    }

    function onPointerDown(event) {
      if (event.button !== 0) return;
      const point = toLogical(event);
      const pieceId = pieceAt(point);
      press = { pieceId, startX: point.x, startY: point.y, isDragging: false };
      if (pieceId !== null) {
        const geometry = pieceGeometry({ board, state: getState(), layout, pieceId });
        press.grabX = point.x - geometry.x;
        press.grabY = point.y - geometry.y;
      }
      canvas.setPointerCapture(event.pointerId);
    }

    function onPointerMove(event) {
      const point = toLogical(event);
      if (press && press.pieceId !== null && !press.isDragging && Math.hypot(point.x - press.startX, point.y - press.startY) > DRAG_THRESHOLD) {
        press.isDragging = true;
        ui.selected = press.pieceId;
        updateControls();
      }
      if (press?.isDragging) {
        ui.drag = { pieceId: press.pieceId, x: point.x, y: point.y, grabX: press.grabX, grabY: press.grabY };
        ui.targetCell = cellAt(point);
        canvas.style.cursor = "grabbing";
        redraw();
        return;
      }
      if (press) return;
      // Hovering: show where a click would put the selected piece.
      const hoveredCell = cellAt(point);
      const targetCell = ui.selected !== null && hoveredCell !== null && getState().grid[hoveredCell] === null ? hoveredCell : null;
      canvas.style.cursor = pieceAt(point) !== null ? "grab" : targetCell !== null ? "pointer" : "";
      if (targetCell !== ui.targetCell) {
        ui.targetCell = targetCell;
        redraw();
      }
    }

    function onPointerUp(event) {
      if (!press) return;
      const point = toLogical(event);
      const finishedPress = press;
      press = null;
      ui.drag = null;
      ui.targetCell = null;
      canvas.style.cursor = "";
      const cellIndex = cellAt(point);
      if (finishedPress.isDragging) {
        lastClick = null;
        if (cellIndex !== null) moveToCell({ pieceId: finishedPress.pieceId, cellIndex });
        else if (!getState().unplaced.includes(finishedPress.pieceId)) act(`remove(${finishedPress.pieceId})`);
        redraw();
        return;
      }
      const pieceId = finishedPress.pieceId;
      const isDoubleClick = pieceId !== null && lastClick?.pieceId === pieceId && event.timeStamp - lastClick.time < DOUBLE_CLICK_MS;
      lastClick = pieceId !== null && !isDoubleClick ? { pieceId, time: event.timeStamp } : null;
      if (pieceId !== null) {
        select(pieceId);
        if (isDoubleClick) act(`rotatecw(${pieceId})`);
      } else if (cellIndex !== null && ui.selected !== null) {
        moveToCell({ pieceId: ui.selected, cellIndex });
      } else {
        select(null);
      }
      redraw();
    }

    function onPointerCancel() {
      press = null;
      ui.drag = null;
      ui.targetCell = null;
      canvas.style.cursor = "";
      redraw();
    }

    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointercancel", onPointerCancel);
    updateControls();

    return {
      onKey(event) {
        if (event.metaKey || event.ctrlKey || event.altKey) return false;
        if (/^[0-9]$/.test(event.key) && Number(event.key) < pieceCount) {
          select(Number(event.key));
          return true;
        }
        if (ui.selected === null) return false;
        if (event.key === "r" || event.key === "ArrowRight") rotateSelected("rotatecw");
        else if (event.key === "R" || event.key === "ArrowLeft") rotateSelected("rotateccw");
        else if (event.key === "Delete" || event.key === "Backspace") removeSelected();
        else if (/^[a-zA-Z]$/.test(event.key) && parseCellLabel(event.key.toUpperCase()) < pieceCount) {
          moveToCell({ pieceId: ui.selected, cellIndex: parseCellLabel(event.key.toUpperCase()) });
        } else return false;
        return true;
      },
      onStateChange({ state }) {
        if (isSolved({ puzzle, state })) ui.selected = null;
        updateControls();
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

  window.PuzzleGames.jigsaw = {
    goal: "Rebuild the picture by turning each piece upright and placing it in its cell.",
    help: "Drag and drop pieces into the grid. Double-click a piece to rotate it.",
    initialState: ({ puzzle }) => puzzle.board.state,
    apply,
    isSolved,
    render,
    mount,
  };
})();
