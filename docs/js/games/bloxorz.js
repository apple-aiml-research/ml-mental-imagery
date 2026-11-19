/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2026 Apple Inc. All Rights Reserved.
 */
// Bloxorz (engine/games/bloxorz): roll a 1×1×2 block until it stands on the goal. Actions: up, down, left,
// right (up = toward row 0, left = toward column 0; the compass on the board shows where they point).
// Rolling off the board, or onto a broken fragile tile, puts the block back at the start.
(function () {
  "use strict";

  const { RENDER_SIZE, keyDirection, directionPad, rgb, drawEngineText } = window.PuzzleKit;

  const BACKGROUND_COLOR = [0x1a, 0x1a, 0x2e];
  // Face colors: [+x, -x, +y (top), -y (bottom), +z, -z] (bloxorz/render.rs).
  const TILE_COLORS = [[0x77, 0x88, 0x99], [0x66, 0x77, 0x88], [0x99, 0xaa, 0xbb], [0x55, 0x66, 0x77], [0x77, 0x88, 0x99], [0x66, 0x77, 0x88]];
  const GOAL_COLORS = [[0xcc, 0x33, 0x33], [0xaa, 0x22, 0x22], [0xee, 0x55, 0x55], [0x88, 0x11, 0x11], [0xcc, 0x33, 0x33], [0xaa, 0x22, 0x22]];
  const BLOCK_COLORS = [[0x44, 0x55, 0xbb], [0x33, 0x44, 0x99], [0x55, 0x77, 0xee], [0x22, 0x33, 0x77], [0x44, 0x55, 0xbb], [0x33, 0x44, 0x99]];
  const FRAGILE_COLORS = [[0xcc, 0x99, 0x33], [0xaa, 0x77, 0x22], [0xee, 0xbb, 0x44], [0x88, 0x66, 0x11], [0xcc, 0x99, 0x33], [0xaa, 0x77, 0x22]];
  const COMPASS_COLOR = [0xcc, 0xcc, 0xcc];
  const COMPASS_DISC_COLOR = [0x15, 0x15, 0x25];
  const PREVIEW_COLOR = "rgba(255, 255, 255, 0.85)";
  const FALL_PREVIEW_COLOR = "rgba(255, 120, 120, 0.9)";
  const LIGHT = { direction: normalize([0.3, 1.0, 0.2]), ambient: 0.35, diffuse: 0.65 };
  const CAMERA_DISTANCE = 20;
  const BLOCK_HALF_UNIT = 0.46;
  const TILE_TOP = 0.3;

  const DIRECTIONS = ["up", "down", "left", "right"];
  // Each direction as a step in the world's ground plane (x = column, z = row).
  const DIRECTION_VECTORS = { up: [0, 0, -1], down: [0, 0, 1], left: [-1, 0, 0], right: [1, 0, 0] };
  const COMPASS_LABELS = { up: "U", down: "D", left: "L", right: "R" };

  // ---------------------------------------------------------------------------
  // Rules (bloxorz/state.rs apply_direction, mod.rs apply).
  // ---------------------------------------------------------------------------

  function occupiedCells({ col, row, orientation }) {
    if (orientation === "lying_x") return [[col, row], [col + 1, row]];
    if (orientation === "lying_y") return [[col, row], [col, row + 1]];
    return [[col, row]];
  }

  function fragileIndex(board, [col, row]) {
    return board.fragile.findIndex(([fragileCol, fragileRow]) => fragileCol === col && fragileRow === row);
  }

  function isSolid(board, [col, row]) {
    return col >= 0 && row >= 0 && col < board.width && row < board.height && board.rows[row][col] === "#";
  }

  function isBroken(board, broken, cell) {
    const index = fragileIndex(board, cell);
    return index >= 0 && (broken & (1 << index)) !== 0;
  }

  function supports(board, state) {
    return occupiedCells(state).every((cell) => isSolid(board, cell) && !isBroken(board, state.broken, cell));
  }

  // Where the block lands rolling `direction`, whether or not anything is under it; null past row or column 0.
  function rolled(board, state, direction) {
    const { col, row, orientation } = state;
    const moves = {
      standing: { up: [col, row - 2, "lying_y"], down: [col, row + 1, "lying_y"], left: [col - 2, row, "lying_x"], right: [col + 1, row, "lying_x"] },
      lying_x: { up: [col, row - 1, "lying_x"], down: [col, row + 1, "lying_x"], left: [col - 1, row, "standing"], right: [col + 2, row, "standing"] },
      lying_y: { up: [col, row - 1, "standing"], down: [col, row + 2, "standing"], left: [col - 1, row, "lying_y"], right: [col + 1, row, "lying_y"] },
    };
    const [nextCol, nextRow, nextOrientation] = moves[orientation][direction];
    if (nextCol < 0 || nextRow < 0) return null;
    // Fragile tiles break as the block leaves them.
    let broken = state.broken;
    occupiedCells(state).forEach((cell) => {
      const index = fragileIndex(board, cell);
      if (index >= 0) broken |= 1 << index;
    });
    return { col: nextCol, row: nextRow, orientation: nextOrientation, broken };
  }

  function startState(board) {
    const [col, row, orientation] = board.start;
    return { col, row, orientation, broken: 0 };
  }

  function apply({ puzzle, state, action }) {
    if (!DIRECTIONS.includes(action)) return null;
    const board = puzzle.board;
    const next = rolled(board, state, action);
    return next && supports(board, next) ? next : startState(board);
  }

  function isSolved({ puzzle, state }) {
    const [goalCol, goalRow] = puzzle.board.goal;
    return state.orientation === "standing" && state.col === goalCol && state.row === goalRow;
  }

  // ---------------------------------------------------------------------------
  // Rendering: the engine's orthographic camera (base/render3d.rs) with flat-shaded boxes, painted back to
  // front instead of z-buffered (tiles never cover the block, which rests on top of them).
  // ---------------------------------------------------------------------------

  function subtract(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
  function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
  function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
  function normalize(vector) {
    const length = Math.hypot(vector[0], vector[1], vector[2]);
    return [vector[0] / length, vector[1] / length, vector[2] / length];
  }

  // The jittered isometric camera, framed on the board's solid cells (bloxorz/render.rs).
  function boardCamera(board) {
    let [minCol, maxCol, minRow, maxRow] = [board.width, 0, board.height, 0];
    for (let row = 0; row < board.height; row += 1) {
      for (let col = 0; col < board.width; col += 1) {
        if (!isSolid(board, [col, row])) continue;
        minCol = Math.min(minCol, col);
        maxCol = Math.max(maxCol, col);
        minRow = Math.min(minRow, row);
        maxRow = Math.max(maxRow, row);
      }
    }
    const target = [(minCol + maxCol) / 2, 0, (minRow + maxRow) / 2];
    const orthoScale = Math.max(maxCol - minCol + 1, maxRow - minRow + 1) * 1.2;
    const { horizontal, vertical } = board.camera;
    const toEye = [Math.cos(horizontal) * Math.cos(vertical), Math.sin(vertical), Math.sin(horizontal) * Math.cos(vertical)];
    const eye = [target[0] + toEye[0] * CAMERA_DISTANCE, target[1] + toEye[1] * CAMERA_DISTANCE, target[2] + toEye[2] * CAMERA_DISTANCE];
    const forward = normalize(subtract(target, eye));
    const right = normalize(cross(forward, [0, 1, 0]));
    const up = cross(right, forward);
    return { eye, forward, right, up, halfExtent: orthoScale / 2 };
  }

  // Screen position in render space, and depth (larger is nearer the camera).
  function project(camera, point) {
    const relative = subtract(point, camera.eye);
    return {
      x: (dot(camera.right, relative) / camera.halfExtent + 1) * 0.5 * RENDER_SIZE,
      y: (1 - dot(camera.up, relative) / camera.halfExtent) * 0.5 * RENDER_SIZE,
      depth: -dot(camera.forward, relative) / camera.halfExtent,
    };
  }

  // Corner indices of each face, counterclockwise from outside, with its color index (render3d.rs box_mesh).
  const BOX_FACES = [[[1, 2, 6, 5], 0], [[0, 4, 7, 3], 1], [[3, 7, 6, 2], 2], [[0, 1, 5, 4], 3], [[4, 5, 6, 7], 4], [[0, 3, 2, 1], 5]];

  function boxCorners(center, half) {
    const [x, y, z] = center;
    const [halfX, halfY, halfZ] = half;
    return [
      [x - halfX, y - halfY, z - halfZ], [x + halfX, y - halfY, z - halfZ], [x + halfX, y + halfY, z - halfZ], [x - halfX, y + halfY, z - halfZ],
      [x - halfX, y - halfY, z + halfZ], [x + halfX, y - halfY, z + halfZ], [x + halfX, y + halfY, z + halfZ], [x - halfX, y + halfY, z + halfZ],
    ];
  }

  // The box's visible faces as screen polygons with their shaded colors.
  function boxFaces({ camera, center, half, colors }) {
    const corners = boxCorners(center, half);
    const projected = corners.map((corner) => project(camera, corner));
    const faces = [];
    BOX_FACES.forEach(([indices, colorIndex]) => {
      const [first, second, third] = indices.map((index) => projected[index]);
      const screenCross = (second.x - first.x) * (third.y - first.y) - (second.y - first.y) * (third.x - first.x);
      if (screenCross >= 0) return; // facing away
      const normal = normalize(cross(subtract(corners[indices[1]], corners[indices[0]]), subtract(corners[indices[2]], corners[indices[0]])));
      const shade = Math.min(1, LIGHT.ambient + LIGHT.diffuse * Math.max(0, dot(normal, LIGHT.direction)));
      const color = colors[colorIndex].map((channel) => Math.min(255, Math.floor(channel * shade)));
      faces.push({ points: indices.map((index) => projected[index]), color });
    });
    return { faces, depth: project(camera, center).depth };
  }

  function blockGeometry({ col, row, orientation }) {
    const halfUnit = BLOCK_HALF_UNIT;
    const [x, z, half] = orientation === "standing" ? [col, row, [halfUnit, 2 * halfUnit, halfUnit]]
      : orientation === "lying_x" ? [col + 0.5, row, [2 * halfUnit, halfUnit, halfUnit]]
        : [col, row + 0.5, [halfUnit, halfUnit, 2 * halfUnit]];
    return { center: [x, TILE_TOP + half[1], z], half };
  }

  function tileBoxes(board, state) {
    const [goalCol, goalRow] = board.goal;
    const boxes = [];
    for (let row = 0; row < board.height; row += 1) {
      for (let col = 0; col < board.width; col += 1) {
        const cell = [col, row];
        if (!isSolid(board, cell) || isBroken(board, state.broken, cell)) continue;
        const isGoal = col === goalCol && row === goalRow;
        if (fragileIndex(board, cell) >= 0 && !isGoal) {
          // Fragile tiles are three amber stripes.
          [-0.26, 0, 0.26].forEach((offset) => boxes.push({ center: [col, 0.15, row + offset], half: [0.48, 0.15, 0.11], colors: FRAGILE_COLORS }));
        } else {
          boxes.push({ center: [col, 0.15, row], half: [0.48, 0.15, 0.48], colors: isGoal ? GOAL_COLORS : TILE_COLORS });
        }
      }
    }
    return boxes;
  }

  function tracePolygon(context, points) {
    context.beginPath();
    points.forEach((point, index) => (index ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y)));
    context.closePath();
  }

  function paintFaces(context, faces) {
    faces.forEach(({ points, color }) => {
      tracePolygon(context, points);
      context.fillStyle = rgb(color);
      context.fill();
      // A hairline of the face's own color hides the antialiased seams between faces.
      context.strokeStyle = rgb(color);
      context.lineWidth = 0.5;
      context.stroke();
    });
  }

  // Screen directions of the four actions, as unit vectors (the projection is orthographic, so a ground step
  // points the same way on screen wherever it starts).
  function screenDirections(camera) {
    const vectors = {};
    DIRECTIONS.forEach((direction) => {
      const step = DIRECTION_VECTORS[direction];
      const [deltaX, deltaY] = [dot(camera.right, step), -dot(camera.up, step)];
      const length = Math.hypot(deltaX, deltaY);
      vectors[direction] = [deltaX / length, deltaY / length];
    });
    return vectors;
  }

  function drawCompass(context, camera) {
    const radius = Math.max(RENDER_SIZE * 0.08, 8);
    const margin = radius * 1.8;
    const [centerX, centerY] = [margin, RENDER_SIZE - margin];
    context.fillStyle = rgb(COMPASS_DISC_COLOR);
    context.beginPath();
    context.arc(Math.trunc(centerX) + 0.5, Math.trunc(centerY) + 0.5, Math.trunc(radius) + 1.5, 0, 2 * Math.PI);
    context.fill();
    const directions = screenDirections(camera);
    DIRECTIONS.forEach((direction) => {
      const [unitX, unitY] = directions[direction];
      context.strokeStyle = rgb(COMPASS_COLOR);
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(centerX + 0.5, centerY + 0.5);
      context.lineTo(centerX + unitX * radius * 0.75 + 0.5, centerY + unitY * radius * 0.75 + 0.5);
      context.stroke();
      drawEngineText({
        context,
        centerX: Math.trunc(centerX + unitX * radius * 1.35),
        centerY: Math.trunc(centerY + unitY * radius * 1.35),
        text: COMPASS_LABELS[direction],
        fontHeight: radius * 0.7,
        color: COMPASS_COLOR,
      });
    });
  }

  function render({ context, puzzle, state, ui }) {
    const board = puzzle.board;
    const camera = boardCamera(board);
    context.fillStyle = rgb(BACKGROUND_COLOR);
    context.fillRect(0, 0, RENDER_SIZE, RENDER_SIZE);
    const tiles = tileBoxes(board, state).map((box) => boxFaces({ camera, ...box }));
    tiles.sort((first, second) => first.depth - second.depth);
    tiles.forEach((tile) => paintFaces(context, tile.faces));
    paintFaces(context, boxFaces({ camera, ...blockGeometry(state), colors: BLOCK_COLORS }).faces);
    if (ui && ui.preview) drawPreview({ context, camera, preview: ui.preview });
    drawCompass(context, camera);
  }

  // Outline of where the block would land on hover: white if it stays on the board, red if it falls off.
  function drawPreview({ context, camera, preview }) {
    const { faces } = boxFaces({ camera, ...blockGeometry(preview.state), colors: BLOCK_COLORS });
    context.save();
    context.setLineDash([5, 4]);
    context.lineWidth = 2;
    context.lineJoin = "round";
    context.strokeStyle = preview.falls ? FALL_PREVIEW_COLOR : PREVIEW_COLOR;
    faces.forEach(({ points }) => {
      tracePolygon(context, points);
      context.stroke();
    });
    context.restore();
  }

  // ---------------------------------------------------------------------------
  // Input: arrow keys and the pad send the engine's directions; pointing at the board previews and taps roll
  // toward the pointer (the direction whose on-screen arrow is closest to it).
  // ---------------------------------------------------------------------------

  function mount({ puzzle, canvas, controls, act, getState, ui, redraw, toLogical }) {
    const board = puzzle.board;
    const camera = boardCamera(board);
    const directions = screenDirections(camera);
    controls.append(directionPad({ onDirection: act }));

    function pointedDirection(event) {
      const point = toLogical(event);
      const geometry = blockGeometry(getState());
      const blockCenter = project(camera, [geometry.center[0], TILE_TOP, geometry.center[2]]);
      const [deltaX, deltaY] = [point.x - blockCenter.x, point.y - blockCenter.y];
      if (Math.hypot(deltaX, deltaY) < 12) return null;
      let best = null;
      DIRECTIONS.forEach((direction) => {
        const alignment = deltaX * directions[direction][0] + deltaY * directions[direction][1];
        if (!best || alignment > best.alignment) best = { direction, alignment };
      });
      return best.direction;
    }

    function updatePreview(event) {
      const direction = event ? pointedDirection(event) : null;
      const state = getState();
      const landing = direction && !isSolved({ puzzle, state }) ? rolled(board, state, direction) : null;
      const preview = direction && landing ? { direction, state: landing, falls: !supports(board, landing) } : null;
      if (JSON.stringify(preview) === JSON.stringify(ui.preview || null)) return;
      ui.preview = preview;
      redraw();
    }

    function onPointerMove(event) {
      if (event.pointerType === "mouse") updatePreview(event);
    }

    function onPointerLeave() {
      updatePreview(null);
    }

    function onPointerDown(event) {
      const direction = pointedDirection(event);
      if (!direction) return;
      ui.preview = null;
      act(direction);
      if (event.pointerType === "mouse") updatePreview(event);
    }

    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerleave", onPointerLeave);
    canvas.addEventListener("pointerdown", onPointerDown);

    return {
      onKey(event) {
        const direction = keyDirection(event);
        if (!direction) return false;
        ui.preview = null;
        act(direction);
        return true;
      },
      onStateChange() {
        ui.preview = null;
      },
      destroy() {
        canvas.removeEventListener("pointermove", onPointerMove);
        canvas.removeEventListener("pointerleave", onPointerLeave);
        canvas.removeEventListener("pointerdown", onPointerDown);
      },
    };
  }

  window.PuzzleGames.bloxorz = {
    goal: "Roll the block until it stands upright on the red tile.",
    help: "Use arrow keys or WASD to roll the block. The compass shows which way each arrow goes.",
    initialState: ({ puzzle }) => puzzle.board.state,
    apply,
    isSolved,
    render,
    mount,
  };
})();
