/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2026 Apple Inc. All Rights Reserved.
 */
// Character recognition (engine/games/character_recognition): rotate each scattered piece of a character, place
// it on the grid intersection (slot) it belongs to, then identify the character. Actions:
// rotatecw(piece, 90), rotateccw(piece, 90), place(piece, row, col), undo(piece), identify('name').
// Pieces are cut out of a glyph rendered by the engine at 200 px; the exporter ships that glyph image and
// the cut (a graph whose nearest edge decides each point's piece), and this port samples them like render.rs.
(function () {
  "use strict";

  const { RENDER_SIZE, button, drawEngineText } = window.PuzzleKit;

  // The engine renders at twice the output resolution and box-filters down (render.rs render, aa_factor 2).
  const HIGH_RESOLUTION = RENDER_SIZE * 2;
  const LOGICAL_PER_HIGH = RENDER_SIZE / HIGH_RESOLUTION;
  const BACKGROUND = [255, 255, 255];
  const PIECE_LABEL_COLOR = [80, 80, 80];
  const DIVIDER_COLOR = [200, 200, 200];
  const GRID_LINE_COLOR = [210, 210, 210];
  const INTERSECTION_COLOR = [170, 170, 170];
  const SELECTION_COLOR = "#4a3aa7";
  const DRAG_THRESHOLD_PX = 4;

  // The answers the engine accepts (state.rs all_allowed_chars through char_to_name), with the characters
  // they name, so a typed character counts as its name.
  const CHARACTER_NAMES = [
    ..."ABCDEFGJKLPQRTUVY", ..."acefghjkmrtvwy", ..."23457",
  ].map((character) => [character, character]).concat([
    ["Γ", "gamma"], ["Δ", "delta"], ["Λ", "lambda"], ["Π", "pi"], ["Σ", "sigma"], ["Φ", "phi"], ["Ψ", "psi"], ["Ω", "omega"],
    ["α", "alpha"], ["β", "beta"], ["γ", "gamma"], ["δ", "delta"], ["ε", "epsilon"], ["ζ", "zeta"], ["η", "eta"],
    ["λ", "lambda"], ["μ", "mu"], ["ξ", "xi"], ["π", "pi"], ["ρ", "rho"], ["σ", "sigma"], ["φ", "phi"], ["ψ", "psi"],
  ]);
  const ALLOWED_NAMES = new Set(CHARACTER_NAMES.map(([, name]) => name));
  const NAME_OF_CHARACTER = new Map(CHARACTER_NAMES);

  // ---------------------------------------------------------------------------
  // Rules (mod.rs apply, state.rs parse).
  // ---------------------------------------------------------------------------

  // Rust's usize parsing: digits with an optional leading +.
  function parseIndex(text) {
    return /^\+?\d+$/.test(text) ? Number(text) : null;
  }

  function splitTwo(text) {
    const comma = text.indexOf(",");
    return comma < 0 ? null : [text.slice(0, comma).trim(), text.slice(comma + 1).trim()];
  }

  function innerOf(text, prefix) {
    return text.startsWith(prefix) && text.endsWith(")") && text.length > prefix.length ? text.slice(prefix.length, -1) : null;
  }

  function parseAction({ board, text }) {
    const trimmed = text.trim();
    const pieceCount = board.pieces.length;
    for (const [prefix, kind] of [["rotatecw(", "rotatecw"], ["rotateccw(", "rotateccw"]]) {
      const inner = innerOf(trimmed, prefix);
      if (inner === null) continue;
      const parts = splitTwo(inner);
      if (!parts) return null;
      const [piece, angle] = [parseIndex(parts[0]), parseIndex(parts[1])];
      return piece === null || angle !== 90 || piece >= pieceCount ? null : { kind, piece };
    }
    const placeInner = innerOf(trimmed, "place(");
    if (placeInner !== null) {
      const parts = splitTwo(placeInner);
      const position = parts && splitTwo(parts[1]);
      if (!position) return null;
      const [piece, row, col] = [parseIndex(parts[0]), parseIndex(position[0]), parseIndex(position[1])];
      const gridDim = board.slot_grid_dim;
      if (piece === null || row === null || col === null || piece >= pieceCount || row >= gridDim || col >= gridDim) return null;
      return { kind: "place", piece, row, col };
    }
    const undoInner = innerOf(trimmed, "undo(");
    if (undoInner !== null) {
      const piece = parseIndex(undoInner.trim());
      return piece === null || piece >= pieceCount ? null : { kind: "undo", piece };
    }
    const identifyInner = innerOf(trimmed, "identify(");
    if (identifyInner !== null) {
      const quoted = identifyInner.trim();
      if (quoted.length < 2 || !quoted.startsWith("'") || !quoted.endsWith("'")) return null;
      const name = quoted.slice(1, -1);
      return ALLOWED_NAMES.has(name) ? { kind: "identify", name } : null;
    }
    return null;
  }

  function pieceCorrect({ board, state, piece }) {
    const slotIndex = state.placements[piece];
    if (slotIndex === null) return false;
    const slot = board.slots[slotIndex];
    return slot.piece === piece && state.rotations[piece] === slot.rotation;
  }

  function allCorrect({ board, state }) {
    return board.pieces.every((_, piece) => pieceCorrect({ board, state, piece }));
  }

  function apply({ puzzle, state, action }) {
    const board = puzzle.board;
    const parsed = parseAction({ board, text: action });
    if (!parsed) return null;
    const { rotations, placements } = state;
    const isPlaced = (piece) => placements[piece] !== null;
    const withRotation = (piece, turn) => ({ ...state, rotations: rotations.map((rotation, index) => (index === piece ? (rotation + turn) % 4 : rotation)) });
    const withPlacement = (piece, slotIndex) => ({ ...state, placements: placements.map((placement, index) => (index === piece ? slotIndex : placement)) });
    switch (parsed.kind) {
      case "rotatecw":
        return isPlaced(parsed.piece) ? null : withRotation(parsed.piece, 1);
      case "rotateccw":
        return isPlaced(parsed.piece) ? null : withRotation(parsed.piece, 3);
      case "place": {
        if (isPlaced(parsed.piece)) return null;
        const slotIndex = board.slots.findIndex((slot) => slot.id[0] === parsed.row && slot.id[1] === parsed.col);
        if (slotIndex < 0 || placements.includes(slotIndex)) return null;
        return withPlacement(parsed.piece, slotIndex);
      }
      case "undo":
        return isPlaced(parsed.piece) ? withPlacement(parsed.piece, null) : null;
      default:
        // The eval's qa_only_eval would allow answering early; players assemble first (see the exporter).
        return allCorrect({ board, state }) ? { ...state, answer: parsed.name } : null;
    }
  }

  function isSolved({ puzzle, state }) {
    return allCorrect({ board: puzzle.board, state }) && state.answer === puzzle.board.character;
  }

  // ---------------------------------------------------------------------------
  // Rendering (render.rs), at the engine's 1024 px working resolution.
  // ---------------------------------------------------------------------------

  // f64::round: halves away from zero.
  function roundHalfAway(value) {
    return Math.sign(value) * Math.round(Math.abs(value));
  }

  function decodeGlyph(glyph) {
    const bytes = atob(glyph.runs);
    const values = new Float32Array(glyph.width * glyph.height);
    let position = 0;
    for (let index = 0; index < bytes.length; index += 2) {
      const value = bytes.charCodeAt(index) / 255;
      const run = bytes.charCodeAt(index + 1);
      values.fill(value, position, position + run);
      position += run;
    }
    return values;
  }

  // Everything about a board that doesn't depend on the state, computed on first render.
  const boardCaches = new WeakMap();

  function boardCache(board) {
    if (boardCaches.has(board)) return boardCaches.get(board);
    const { glyph, lookup } = board;
    const glyphValues = decodeGlyph(glyph);
    const glyphValue = (x, y) => {
      const column = x - glyph.left;
      const row = y - glyph.top;
      return column >= 0 && row >= 0 && column < glyph.width && row < glyph.height ? glyphValues[row * glyph.width + column] : 0;
    };
    // render.rs sample_alpha: the glyph image, bilinearly sampled at a point in grid coordinates.
    function sampleAlpha(gridX, gridY) {
      const sourceX = (gridX - glyph.offset[0]) / glyph.scale + glyph.min[0];
      const sourceY = (gridY - glyph.offset[1]) / glyph.scale + glyph.min[1];
      const x0 = Math.floor(sourceX);
      const y0 = Math.floor(sourceY);
      const fractionX = sourceX - x0;
      const fractionY = sourceY - y0;
      return glyphValue(x0, y0) * (1 - fractionX) * (1 - fractionY) + glyphValue(x0 + 1, y0) * fractionX * (1 - fractionY)
        + glyphValue(x0, y0 + 1) * (1 - fractionX) * fractionY + glyphValue(x0 + 1, y0 + 1) * fractionX * fractionY;
    }
    // state.rs PieceLookup::piece_at: the piece of the nearest cut-graph edge (of its nearer end if they differ).
    function pieceAt(x, y) {
      let bestDistance = Infinity;
      let bestPiece = 0;
      for (const [startIndex, endIndex] of lookup.edges) {
        const [x1, y1] = lookup.points[startIndex];
        const [x2, y2] = lookup.points[endIndex];
        const deltaX = x2 - x1;
        const deltaY = y2 - y1;
        const lengthSquared = deltaX * deltaX + deltaY * deltaY;
        let distance;
        if (lengthSquared < 1e-12) {
          distance = (x - x1) ** 2 + (y - y1) ** 2;
        } else {
          const along = Math.min(1, Math.max(0, ((x - x1) * deltaX + (y - y1) * deltaY) / lengthSquared));
          distance = (x - (x1 + along * deltaX)) ** 2 + (y - (y1 + along * deltaY)) ** 2;
        }
        if (distance < bestDistance) {
          bestDistance = distance;
          const startPiece = lookup.vertex_piece[startIndex];
          const endPiece = lookup.vertex_piece[endIndex];
          bestPiece = startPiece === endPiece || (x - x1) ** 2 + (y - y1) ** 2 <= (x - x2) ** 2 + (y - y2) ** 2 ? startPiece : endPiece;
        }
      }
      return bestPiece;
    }

    const resolution = HIGH_RESOLUTION;
    const gridSize = board.grid_size;
    const assemblyHeight = Math.floor(resolution / 2);
    const paletteTop = Math.floor((resolution * 55) / 100);
    const paletteHeight = resolution - paletteTop;
    const margin = Math.floor(resolution / 14);
    const cellSize = Math.max(1, Math.min(Math.floor((resolution - 2 * margin) / gridSize), Math.floor((assemblyHeight - 2 * margin) / gridSize)));
    const gridPixels = cellSize * gridSize;
    const gridX0 = Math.floor((resolution - gridPixels) / 2);
    const gridY0 = Math.floor((assemblyHeight - gridPixels) / 2);
    const gridDim = board.slot_grid_dim;
    const lineX = Array.from({ length: gridDim }, (_, index) => gridX0 + Math.floor(((index + 1) * gridPixels) / (gridDim + 1)));
    const lineY = Array.from({ length: gridDim }, (_, index) => gridY0 + Math.floor(((index + 1) * gridPixels) / (gridDim + 1)));
    const pieceCount = board.pieces.length;
    const sectionWidth = Math.floor(resolution / pieceCount);
    const labelSpace = Math.floor(paletteHeight * 0.18);
    const labelFontHeight = Math.max(Math.fround(resolution / 15), 16);
    const geometry = {
      paletteTop, margin, cellSize, gridPixels, gridX0, gridY0, lineX, lineY, sectionWidth, labelSpace, labelFontHeight,
      circleRadius: Math.max(Math.floor(cellSize / 3), 2),
      labelFontSmall: Math.max(Math.fround(labelFontHeight * 0.7), 12),
      paletteCenterY: paletteTop + labelSpace + Math.floor((paletteHeight - labelSpace) / 2),
    };

    // The assembly area's glyph alpha and piece per pixel.
    const assemblyAlpha = new Float32Array(gridPixels * gridPixels);
    const assemblyPiece = new Int8Array(gridPixels * gridPixels).fill(-1);
    for (let pixelY = 0; pixelY < gridPixels; pixelY += 1) {
      for (let pixelX = 0; pixelX < gridPixels; pixelX += 1) {
        const gridX = (pixelX + 0.5) / cellSize;
        const gridY = (pixelY + 0.5) / cellSize;
        const alpha = sampleAlpha(gridX, gridY);
        if (alpha < 0.005) continue;
        const index = pixelY * gridPixels + pixelX;
        assemblyAlpha[index] = alpha;
        if (lookup) assemblyPiece[index] = pieceAt(gridX, gridY);
      }
    }

    // The empty board: grid lines and intersection circles (labels are text, drawn per render).
    const background = new ImageData(resolution, resolution);
    const backgroundPixels = background.data;
    for (let index = 0; index < backgroundPixels.length; index += 4) {
      backgroundPixels.set(BACKGROUND, index);
      backgroundPixels[index + 3] = 255;
    }
    const setBackground = (x, y, color) => backgroundPixels.set(color, (y * resolution + x) * 4);
    lineX.forEach((x) => { for (let y = gridY0; y < gridY0 + gridPixels; y += 1) setBackground(x, y, GRID_LINE_COLOR); });
    lineY.forEach((y) => { for (let x = gridX0; x < gridX0 + gridPixels; x += 1) setBackground(x, y, GRID_LINE_COLOR); });
    const radius = geometry.circleRadius;
    lineY.forEach((centerY) => lineX.forEach((centerX) => {
      for (let deltaY = -radius; deltaY <= radius; deltaY += 1) {
        for (let deltaX = -radius; deltaX <= radius; deltaX += 1) {
          if (deltaX * deltaX + deltaY * deltaY <= radius * radius) setBackground(centerX + deltaX, centerY + deltaY, INTERSECTION_COLOR);
        }
      }
    }));
    const dividerY = paletteTop - 3;
    for (let x = Math.floor(margin / 2); x < resolution - Math.floor(margin / 2); x += 1) setBackground(x, dividerY, DIVIDER_COLOR);

    const cache = {
      geometry, assemblyAlpha, assemblyPiece, background, sampleAlpha, pieceAt,
      pieceImages: new Map(), ghostCanvases: new Map(), lastState: null, lastCanvas: null,
    };
    boardCaches.set(board, cache);
    return cache;
  }

  // render.rs render_piece: a palette piece at a rotation, as glyph alphas over its rotated bounding box.
  function pieceImage({ board, cache, piece, rotation }) {
    const key = `${piece}:${rotation}`;
    if (cache.pieceImages.has(key)) return cache.pieceImages.get(key);
    const { cells, centroid } = board.pieces[piece];
    const xs = cells.map(([x]) => x);
    const ys = cells.map(([, y]) => y);
    const pieceWidth = Math.max(...xs) - Math.min(...xs) + 1;
    const pieceHeight = Math.max(...ys) - Math.min(...ys) + 1;
    const [rotatedWidth, rotatedHeight] = rotation % 2 === 0 ? [pieceWidth, pieceHeight] : [pieceHeight, pieceWidth];
    const cellSize = cache.geometry.cellSize;
    const width = rotatedWidth * cellSize;
    const height = rotatedHeight * cellSize;
    const cellSet = new Set(cells.map(([x, y]) => `${x},${y}`));
    const alphas = new Float32Array(width * height);
    for (let pixelY = 0; pixelY < height; pixelY += 1) {
      for (let pixelX = 0; pixelX < width; pixelX += 1) {
        const deltaX = (pixelX + 0.5) / cellSize - rotatedWidth / 2;
        const deltaY = (pixelY + 0.5) / cellSize - rotatedHeight / 2;
        const [originalX, originalY] = [[deltaX, deltaY], [deltaY, -deltaX], [-deltaX, -deltaY], [-deltaY, deltaX]][rotation % 4];
        const gridX = centroid[0] + originalX;
        const gridY = centroid[1] + originalY;
        const belongs = board.lookup ? cache.pieceAt(gridX, gridY) === piece : cellSet.has(`${Math.floor(gridX)},${Math.floor(gridY)}`);
        if (!belongs) continue;
        const alpha = cache.sampleAlpha(gridX, gridY);
        if (alpha > 0.003) alphas[pixelY * width + pixelX] = alpha;
      }
    }
    const image = { width, height, alphas };
    cache.pieceImages.set(key, image);
    return image;
  }

  // render.rs blend, onto the white background.
  function blendedChannel(foreground, alpha) {
    const clamped = Math.min(1, Math.max(0, alpha));
    return Math.trunc(255 * (1 - clamped) + foreground * clamped);
  }

  function renderHighResolution({ board, cache, state }) {
    const { geometry, assemblyAlpha, assemblyPiece } = cache;
    const { cellSize, gridPixels, gridX0, gridY0, sectionWidth, paletteCenterY } = geometry;
    const resolution = HIGH_RESOLUTION;
    const gridSize = board.grid_size;
    const image = new ImageData(new Uint8ClampedArray(cache.background.data), resolution, resolution);
    const pixels = image.data;
    const fill = board.fill_color;
    const paint = (x, y, alpha) => {
      if (x < 0 || y < 0 || x >= resolution || y >= resolution) return;
      const offset = (y * resolution + x) * 4;
      pixels[offset] = blendedChannel(fill[0], alpha);
      pixels[offset + 1] = blendedChannel(fill[1], alpha);
      pixels[offset + 2] = blendedChannel(fill[2], alpha);
    };

    // Correctly placed pieces show where they belong in the glyph; misplaced ones show the glyph through
    // the cells they cover (rotated about their centroid onto the slot's).
    const pieceCount = board.pieces.length;
    const correctlyPlaced = board.pieces.map((_, piece) => pieceCorrect({ board, state, piece }));
    const misplacedCells = new Uint8Array(gridSize * gridSize);
    board.pieces.forEach(({ cells, centroid }, piece) => {
      const slotIndex = state.placements[piece];
      if (slotIndex === null || correctlyPlaced[piece]) return;
      const target = board.slots[slotIndex].centroid;
      cells.forEach(([x, y]) => {
        const deltaX = x - centroid[0];
        const deltaY = y - centroid[1];
        const [rotatedX, rotatedY] = [[deltaX, deltaY], [-deltaY, deltaX], [-deltaX, -deltaY], [deltaY, -deltaX]][state.rotations[piece] % 4];
        const cellX = roundHalfAway(rotatedX + target[0]);
        const cellY = roundHalfAway(rotatedY + target[1]);
        if (cellX >= 0 && cellY >= 0 && cellX < gridSize && cellY < gridSize) misplacedCells[cellY * gridSize + cellX] = 1;
      });
    });
    const correctCellSets = board.lookup ? null : board.pieces.map(({ cells }) => new Set(cells.map(([x, y]) => `${x},${y}`)));
    for (let pixelY = 0; pixelY < gridPixels; pixelY += 1) {
      for (let pixelX = 0; pixelX < gridPixels; pixelX += 1) {
        const index = pixelY * gridPixels + pixelX;
        const alpha = assemblyAlpha[index];
        if (alpha < 0.005) continue;
        const piece = assemblyPiece[index];
        let show = piece >= 0 && piece < pieceCount && correctlyPlaced[piece];
        if (!show) {
          const cellX = Math.floor((pixelX + 0.5) / cellSize);
          const cellY = Math.floor((pixelY + 0.5) / cellSize);
          if (cellX < gridSize && cellY < gridSize) {
            show = misplacedCells[cellY * gridSize + cellX] === 1
              || (correctCellSets !== null && correctCellSets.some((cellSet, other) => correctlyPlaced[other] && cellSet.has(`${cellX},${cellY}`)));
          }
        }
        if (show) paint(gridX0 + pixelX, gridY0 + pixelY, alpha);
      }
    }

    // The palette: every piece has a fixed section; only unplaced ones are drawn.
    board.pieces.forEach((_, piece) => {
      if (state.placements[piece] !== null) return;
      const { width, height, alphas } = pieceImage({ board, cache, piece, rotation: state.rotations[piece] });
      const left = piece * sectionWidth + Math.floor(sectionWidth / 2) - Math.floor(width / 2);
      const top = paletteCenterY - Math.floor(height / 2);
      for (let pixelY = 0; pixelY < height; pixelY += 1) {
        for (let pixelX = 0; pixelX < width; pixelX += 1) {
          const alpha = alphas[pixelY * width + pixelX];
          if (alpha > 0) paint(left + pixelX, top + pixelY, alpha);
        }
      }
    });

    const canvas = document.createElement("canvas");
    canvas.width = resolution;
    canvas.height = resolution;
    const context = canvas.getContext("2d");
    context.putImageData(image, 0, 0);
    const { lineX, lineY, margin, labelFontHeight, labelFontSmall, labelSpace, paletteTop } = geometry;
    lineY.forEach((centerY, index) => drawEngineText({
      context, text: String(index), centerX: Math.max(0, gridX0 - Math.floor(margin / 2)), centerY, fontHeight: labelFontSmall, color: PIECE_LABEL_COLOR,
    }));
    lineX.forEach((centerX, index) => drawEngineText({
      context, text: String(index), centerX, centerY: Math.max(0, gridY0 - Math.floor(margin / 3)), fontHeight: labelFontSmall, color: PIECE_LABEL_COLOR,
    }));
    board.pieces.forEach((_, piece) => {
      if (state.placements[piece] !== null) return;
      drawEngineText({
        context, text: piece < 10 ? String(piece) : "?", centerX: piece * sectionWidth + Math.floor(sectionWidth / 2),
        centerY: paletteTop + Math.floor(labelSpace / 2), fontHeight: labelFontHeight, color: PIECE_LABEL_COLOR,
      });
    });
    return canvas;
  }

  // A piece in the fill color, for drag and drop previews.
  function ghostCanvas({ board, cache, piece, rotation }) {
    const key = `${piece}:${rotation}`;
    if (cache.ghostCanvases.has(key)) return cache.ghostCanvases.get(key);
    const { width, height, alphas } = pieceImage({ board, cache, piece, rotation });
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, width);
    canvas.height = Math.max(1, height);
    const context = canvas.getContext("2d");
    const image = context.createImageData(canvas.width, canvas.height);
    alphas.forEach((alpha, index) => {
      image.data.set(board.fill_color, index * 4);
      image.data[index * 4 + 3] = Math.round(alpha * 255);
    });
    context.putImageData(image, 0, 0);
    cache.ghostCanvases.set(key, canvas);
    return canvas;
  }

  function drawGhost({ context, board, cache, state, piece, centerX, centerY, opacity }) {
    const ghost = ghostCanvas({ board, cache, piece, rotation: state.rotations[piece] });
    const width = ghost.width * LOGICAL_PER_HIGH;
    const height = ghost.height * LOGICAL_PER_HIGH;
    context.save();
    context.globalAlpha = opacity;
    context.drawImage(ghost, centerX - width / 2, centerY - height / 2, width, height);
    context.restore();
  }

  function slotPoint({ cache, slot }) {
    const { lineX, lineY } = cache.geometry;
    return { x: lineX[slot.id[1]] * LOGICAL_PER_HIGH, y: lineY[slot.id[0]] * LOGICAL_PER_HIGH };
  }

  function render({ context, puzzle, state, ui }) {
    const board = puzzle.board;
    const cache = boardCache(board);
    if (cache.lastState !== state) {
      cache.lastCanvas = renderHighResolution({ board, cache, state });
      cache.lastState = state;
    }
    context.imageSmoothingEnabled = true;
    context.drawImage(cache.lastCanvas, 0, 0, RENDER_SIZE, RENDER_SIZE);
    if (!ui) return;

    const { sectionWidth, paletteTop } = cache.geometry;
    const sectionBox = (piece) => ({
      x: piece * sectionWidth * LOGICAL_PER_HIGH + 3,
      y: paletteTop * LOGICAL_PER_HIGH + 2,
      width: sectionWidth * LOGICAL_PER_HIGH - 6,
      height: (HIGH_RESOLUTION - paletteTop) * LOGICAL_PER_HIGH - 5,
    });
    const strokeBox = ({ x, y, width, height }, color, lineWidth) => {
      context.save();
      context.strokeStyle = color;
      context.lineWidth = lineWidth;
      context.beginPath();
      context.roundRect(x, y, width, height, 10);
      context.stroke();
      context.restore();
    };
    if (ui.hoverPiece != null && ui.hoverPiece !== ui.selected && state.placements[ui.hoverPiece] === null) {
      strokeBox(sectionBox(ui.hoverPiece), "rgba(0, 0, 0, 0.18)", 1.5);
    }
    const selected = ui.selected;
    if (selected == null || state.placements[selected] !== null) return;
    strokeBox(sectionBox(selected), SELECTION_COLOR, 2.5);

    // Drop preview: the selected piece over the slot it would go to, centered on the intersection.
    if (ui.targetSlot != null) {
      const slot = board.slots[ui.targetSlot];
      const point = slotPoint({ cache, slot });
      const isTaken = state.placements.includes(ui.targetSlot);
      if (!isTaken) drawGhost({ context, board, cache, state, piece: selected, centerX: point.x, centerY: point.y, opacity: 0.45 });
      context.save();
      context.strokeStyle = isTaken ? "#c23b3b" : SELECTION_COLOR;
      context.lineWidth = 2.5;
      context.beginPath();
      context.arc(point.x, point.y, cache.geometry.circleRadius * LOGICAL_PER_HIGH + 4, 0, 2 * Math.PI);
      context.stroke();
      context.restore();
    }
    if (ui.drag?.moved && ui.targetSlot == null) {
      drawGhost({ context, board, cache, state, piece: selected, centerX: ui.drag.x, centerY: ui.drag.y, opacity: 0.6 });
    }
  }

  // ---------------------------------------------------------------------------
  // Interaction.
  // ---------------------------------------------------------------------------

  function mount({ puzzle, canvas, controls, act, getState, ui, redraw, toLogical }) {
    const board = puzzle.board;
    const cache = boardCache(board);
    ui.selected = null;

    const isUnplaced = (piece) => piece != null && getState().placements[piece] === null;

    function select(piece) {
      ui.selected = isUnplaced(piece) ? piece : null;
      updateControls();
      redraw();
    }

    function rotate(direction) {
      if (!isUnplaced(ui.selected)) return false;
      act(`${direction}(${ui.selected}, 90)`);
      return true;
    }

    // What a point in render space is over: a palette piece section, or the assembly (with the nearest slot).
    function hitTest(point) {
      const { paletteTop, sectionWidth, gridX0, gridY0, cellSize, gridPixels } = cache.geometry;
      const highX = point.x / LOGICAL_PER_HIGH;
      const highY = point.y / LOGICAL_PER_HIGH;
      if (highY >= paletteTop - 3) {
        const piece = Math.floor(highX / sectionWidth);
        return { area: "palette", piece: piece >= 0 && piece < board.pieces.length ? piece : null };
      }
      let nearestSlot = null;
      let nearestDistance = Infinity;
      board.slots.forEach((slot, slotIndex) => {
        const slotLocation = slotPoint({ cache, slot });
        const distance = (slotLocation.x - point.x) ** 2 + (slotLocation.y - point.y) ** 2;
        if (distance < nearestDistance) {
          nearestDistance = distance;
          nearestSlot = slotIndex;
        }
      });
      // The placed piece under the point: a misplaced piece's cells, else a correct piece's glyph pixels.
      const state = getState();
      let placedPiece = null;
      const pixelX = Math.floor(highX - gridX0);
      const pixelY = Math.floor(highY - gridY0);
      if (pixelX >= 0 && pixelY >= 0 && pixelX < gridPixels && pixelY < gridPixels) {
        const cellX = Math.floor(pixelX / cellSize);
        const cellY = Math.floor(pixelY / cellSize);
        board.pieces.forEach(({ cells, centroid }, piece) => {
          const slotIndex = state.placements[piece];
          if (slotIndex === null || pieceCorrect({ board, state, piece })) return;
          const target = board.slots[slotIndex].centroid;
          const covers = cells.some(([x, y]) => {
            const deltaX = x - centroid[0];
            const deltaY = y - centroid[1];
            const [rotatedX, rotatedY] = [[deltaX, deltaY], [-deltaY, deltaX], [-deltaX, -deltaY], [deltaY, -deltaX]][state.rotations[piece] % 4];
            return roundHalfAway(rotatedX + target[0]) === cellX && roundHalfAway(rotatedY + target[1]) === cellY;
          });
          if (covers) placedPiece = piece;
        });
        const index = pixelY * gridPixels + pixelX;
        const glyphPiece = cache.assemblyPiece[index];
        if (placedPiece === null && cache.assemblyAlpha[index] >= 0.005 && glyphPiece >= 0 && pieceCorrect({ board, state, piece: glyphPiece })) {
          placedPiece = glyphPiece;
        }
      }
      const slotOccupant = state.placements.indexOf(nearestSlot);
      if (placedPiece === null && slotOccupant >= 0 && nearestDistance < 14 ** 2) placedPiece = slotOccupant;
      return { area: "assembly", slot: nearestSlot, placedPiece };
    }

    function place(slotIndex) {
      const [row, col] = board.slots[slotIndex].id;
      const piece = ui.selected;
      ui.targetSlot = null;
      if (act(`place(${piece}, ${row}, ${col})`)) ui.selected = null;
      updateControls();
      redraw();
    }

    function updateHover(point) {
      const hit = hitTest(point);
      ui.hoverPiece = hit.area === "palette" ? hit.piece : null;
      ui.targetSlot = hit.area === "assembly" && isUnplaced(ui.selected) ? hit.slot : null;
      const pointsAtSomething = (hit.area === "palette" && isUnplaced(hit.piece)) || (hit.area === "assembly" && (ui.targetSlot != null || hit.placedPiece != null));
      canvas.style.cursor = ui.drag?.moved ? "grabbing" : pointsAtSomething ? "pointer" : "";
      return hit;
    }

    function onPointerDown(event) {
      const point = toLogical(event);
      const hit = updateHover(point);
      if (hit.area === "palette") {
        if (!isUnplaced(hit.piece)) {
          select(null);
          return;
        }
        select(hit.piece);
        ui.drag = { startX: point.x, startY: point.y, x: point.x, y: point.y, moved: false };
        canvas.setPointerCapture(event.pointerId);
        return;
      }
      if (isUnplaced(ui.selected)) {
        place(hit.slot);
      } else if (hit.placedPiece !== null) {
        // Taking a piece back selects it, ready to rotate and place again.
        const piece = hit.placedPiece;
        if (act(`undo(${piece})`)) select(piece);
      }
    }

    function onPointerMove(event) {
      const point = toLogical(event);
      if (ui.drag) {
        ui.drag.x = point.x;
        ui.drag.y = point.y;
        if (Math.hypot(point.x - ui.drag.startX, point.y - ui.drag.startY) > DRAG_THRESHOLD_PX) ui.drag.moved = true;
      }
      updateHover(point);
      redraw();
    }

    function onPointerUp(event) {
      const drag = ui.drag;
      ui.drag = null;
      if (!drag) return;
      const hit = updateHover(toLogical(event));
      if (drag.moved && hit.area === "assembly" && isUnplaced(ui.selected)) place(hit.slot);
      else redraw();
    }

    function onPointerLeave() {
      if (ui.drag) return;
      ui.hoverPiece = null;
      ui.targetSlot = null;
      redraw();
    }

    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointercancel", onPointerUp);
    canvas.addEventListener("pointerleave", onPointerLeave);

    const pieceGroup = document.createElement("div");
    pieceGroup.className = "play-group";
    const pieceLabel = document.createElement("span");
    pieceLabel.className = "play-group-label";
    const rotateCounterclockwise = button({ label: "⟲ Rotate", title: "Rotate counterclockwise", onClick: () => rotate("rotateccw") });
    const rotateClockwise = button({ label: "⟳ Rotate", title: "Rotate clockwise", onClick: () => rotate("rotatecw") });
    pieceGroup.append(pieceLabel, rotateCounterclockwise, rotateClockwise);

    const answerForm = document.createElement("form");
    answerForm.className = "play-group";
    const answerLabel = document.createElement("span");
    answerLabel.className = "play-group-label";
    answerLabel.textContent = "Answer";
    const answerInput = document.createElement("input");
    answerInput.className = "play-input";
    answerInput.size = 13;
    answerInput.placeholder = "A, 7, gamma…";
    answerInput.autocomplete = "off";
    answerInput.spellcheck = false;
    answerInput.setAttribute("aria-label", "The character (a letter, digit, or Greek letter name)");
    const answerList = document.createElement("datalist");
    answerList.id = `character-names-${Math.random().toString(36).slice(2)}`;
    [...ALLOWED_NAMES].forEach((name) => answerList.append(new Option(name)));
    answerInput.setAttribute("list", answerList.id);
    const identifyButton = button({ label: "Identify", onClick: () => {} });
    identifyButton.type = "submit";
    answerForm.append(answerLabel, answerInput, answerList, identifyButton);
    answerForm.addEventListener("submit", (event) => {
      event.preventDefault();
      const typed = answerInput.value.trim().replace(/^identify\(\s*'?|'?\s*\)$/g, "");
      if (!typed) return;
      act(`identify('${NAME_OF_CHARACTER.get(typed) || typed}')`);
    });
    controls.append(pieceGroup, answerForm);

    function updateControls() {
      const state = getState();
      const canRotate = isUnplaced(ui.selected);
      pieceLabel.textContent = canRotate ? `Piece ${ui.selected}` : "No piece selected";
      rotateCounterclockwise.disabled = !canRotate;
      rotateClockwise.disabled = !canRotate;
      const canAnswer = allCorrect({ board, state }) && !isSolved({ puzzle, state });
      answerInput.disabled = !canAnswer;
      identifyButton.disabled = !canAnswer;
    }
    updateControls();

    return {
      onKey(event) {
        if (event.metaKey || event.ctrlKey || event.altKey) return false;
        const key = event.key;
        if (/^[0-9]$/.test(key) && Number(key) < board.pieces.length) {
          select(ui.selected === Number(key) ? null : Number(key));
          return true;
        }
        if (key === "r" || key === "e" || key === "E" || key === "]") return rotate("rotatecw");
        if (key === "q" || key === "Q" || key === "R" || key === "[") return rotate("rotateccw");
        return false;
      },
      onStateChange() {
        if (!isUnplaced(ui.selected)) ui.selected = null;
        updateControls();
      },
      destroy() {
        canvas.removeEventListener("pointerdown", onPointerDown);
        canvas.removeEventListener("pointermove", onPointerMove);
        canvas.removeEventListener("pointerup", onPointerUp);
        canvas.removeEventListener("pointercancel", onPointerUp);
        canvas.removeEventListener("pointerleave", onPointerLeave);
        canvas.style.cursor = "";
      },
    };
  }

  window.PuzzleGames.character_recognition = {
    goal: "Put the broken character back together, then say which character it is.",
    help: "Drag and drop pieces into the slots. Use controls above to rotate pieces.",
    initialState: ({ puzzle }) => puzzle.board.state,
    apply,
    isSolved,
    render,
    mount,
  };
})();
