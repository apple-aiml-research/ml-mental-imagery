/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2026 Apple Inc. All Rights Reserved.
 */
// 3D shape matching (engine/games/mental_rotation, solve mode): rotate the left polycube to match the right.
// Actions: rotatex(deg), rotatey(deg), rotatez(deg), deg a non-zero whole number in [-180, 180]; each adds
// to one of the left shape's Euler angles (the engine composes them Z, then Y, then X).
(function () {
  "use strict";

  const { RENDER_SIZE, button, rgb, drawEngineText } = window.PuzzleKit;

  // Per-face-direction colors: [+x, -x, +y, -y, +z, -z] (render.rs).
  const FACE_COLORS = [
    [0xcc, 0x33, 0x33],
    [0x33, 0xaa, 0x33],
    [0x33, 0x55, 0xcc],
    [0xcc, 0xcc, 0x33],
    [0xdd, 0x77, 0x22],
    [0x88, 0x33, 0xbb],
  ];
  const BACKGROUND_COLOR = [0x1a, 0x1a, 0x2e];
  const DIVIDER_COLOR = [0x44, 0x44, 0x66];
  const AXES = [
    { axis: "x", label: "X", color: [0xcc, 0x33, 0x33], direction: [1, 0, 0] },
    { axis: "y", label: "Y", color: [0x33, 0x55, 0xcc], direction: [0, 1, 0] },
    { axis: "z", label: "Z", color: [0xdd, 0x77, 0x22], direction: [0, 0, 1] },
  ];
  const VOXEL_HALF_SIZE = 0.46;
  const SOLVED_ERROR = 0.15;
  // Box corners and faces (render3d.rs box_mesh): CCW quads seen from outside, with their FACE_COLORS index.
  const CORNER_SIGNS = [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]];
  const FACES = [[[1, 2, 6, 5], 0], [[0, 4, 7, 3], 1], [[3, 7, 6, 2], 2], [[0, 1, 5, 4], 3], [[4, 5, 6, 7], 4], [[0, 3, 2, 1], 5]];
  const LIGHT_DIRECTION = normalize([0.3, 1.0, 0.2]);
  const AMBIENT = 0.35;
  const DIFFUSE = 0.65;
  // The isometric camera (render3d.rs isometric_camera): 45° azimuth, arctan(1/√2) elevation, looking at the origin.
  const CAMERA_AZIMUTH = Math.PI / 4;
  const CAMERA_ELEVATION = Math.atan(1 / Math.SQRT2);
  const VIEW_DIRECTION = [
    Math.cos(CAMERA_AZIMUTH) * Math.cos(CAMERA_ELEVATION),
    Math.sin(CAMERA_ELEVATION),
    Math.sin(CAMERA_AZIMUTH) * Math.cos(CAMERA_ELEVATION),
  ]; // toward the eye
  const CAMERA_RIGHT = normalize([VIEW_DIRECTION[2], 0, -VIEW_DIRECTION[0]]);
  const CAMERA_UP = cross(CAMERA_RIGHT, VIEW_DIRECTION.map((component) => -component));

  const ACTION_PATTERN = /^rotate([xyz])\(([+-]?\d+)\)$/;
  const ANIMATION_MS = 200;
  const DRAG_PIXELS_PER_DEGREE = 2; // in render space
  const DRAG_DEAD_ZONE = 6;

  function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }

  function cross(a, b) {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  }

  function normalize(vector) {
    const length = Math.hypot(...vector);
    return vector.map((component) => component / length);
  }

  // The engine's rotation_matrix: Rz · Ry · Rx, angles in degrees; rows of a 3x3 matrix.
  function rotationMatrix({ rx, ry, rz }) {
    const toRadians = Math.PI / 180;
    const [cosX, sinX] = [Math.cos(rx * toRadians), Math.sin(rx * toRadians)];
    const [cosY, sinY] = [Math.cos(ry * toRadians), Math.sin(ry * toRadians)];
    const [cosZ, sinZ] = [Math.cos(rz * toRadians), Math.sin(rz * toRadians)];
    return [
      [cosZ * cosY, cosZ * sinY * sinX - sinZ * cosX, cosZ * sinY * cosX + sinZ * sinX],
      [sinZ * cosY, sinZ * sinY * sinX + cosZ * cosX, sinZ * sinY * cosX - cosZ * sinX],
      [-sinY, cosY * sinX, cosY * cosX],
    ];
  }

  function rotate(matrix, vector) {
    return [dot(matrix[0], vector), dot(matrix[1], vector), dot(matrix[2], vector)];
  }

  function centroidOf(voxels) {
    const sums = voxels.reduce((total, voxel) => total.map((sum, axis) => sum + voxel[axis]), [0, 0, 0]);
    return sums.map((sum) => sum / voxels.length);
  }

  // The engine's transform_error: mean distance between the centered voxels under the two rotations,
  // over the farthest voxel's distance from the centroid.
  function transformError({ voxels, rotationA, rotationB }) {
    const centroid = centroidOf(voxels);
    const centered = voxels.map((voxel) => voxel.map((coordinate, axis) => coordinate - centroid[axis]));
    const maxExtent = Math.max(...centered.map((vector) => Math.hypot(...vector)));
    if (maxExtent < 1e-6) return 0;
    const matrixA = rotationMatrix(rotationA);
    const matrixB = rotationMatrix(rotationB);
    const totalDistance = centered.reduce((total, vector) => {
      const pointA = rotate(matrixA, vector);
      const pointB = rotate(matrixB, vector);
      return total + Math.hypot(pointA[0] - pointB[0], pointA[1] - pointB[1], pointA[2] - pointB[2]);
    }, 0);
    return totalDistance / (voxels.length * maxExtent);
  }

  function parseAction(action) {
    const match = ACTION_PATTERN.exec(action);
    if (!match) return null;
    const degrees = Number(match[2]);
    if (degrees === 0 || degrees < -180 || degrees > 180) return null;
    return { axis: match[1], degrees };
  }

  function apply({ state, action }) {
    const parsed = parseAction(action);
    if (!parsed) return null;
    const key = `r${parsed.axis}`;
    return { ...state, [key]: (((state[key] + parsed.degrees) % 360) + 360) % 360 };
  }

  function isSolved({ puzzle, state }) {
    return transformError({ voxels: puzzle.board.voxels, rotationA: state, rotationB: puzzle.board.target }) <= SOLVED_ERROR;
  }

  // ---------------------------------------------------------------------------
  // Rendering (render.rs + render3d.rs): two orthographic isometric views, left the player's rotation and
  // right the target, split by a divider. The engine z-buffers flat-shaded triangles; here each cube's
  // front faces are painted back to front by cube depth, which is exact for cubes on a lattice.
  // ---------------------------------------------------------------------------

  function panelGeometry() {
    const dividerWidth = Math.max(Math.floor(RENDER_SIZE / 64), 2);
    const panelWidth = Math.floor((RENDER_SIZE - dividerWidth) / 2);
    return { dividerWidth, panelWidth, panelHeight: RENDER_SIZE };
  }

  // Bounding sphere of the voxels, each counted out to its far corner (render.rs bounding_radius).
  function boundingRadius(voxels) {
    const centroid = centroidOf(voxels);
    const largestSquared = Math.max(...voxels.map((voxel) => voxel.reduce(
      (total, coordinate, axis) => total + (Math.abs(coordinate - centroid[axis]) + 0.5) ** 2, 0)));
    return Math.sqrt(largestSquared);
  }

  function projector({ board }) {
    const { panelWidth, panelHeight } = panelGeometry();
    const aspect = panelWidth / panelHeight;
    const orthoScale = (boundingRadius(board.voxels) * 2.5) / Math.min(aspect, 1) / (board.zoomPct / 100);
    const halfExtent = orthoScale / 2;
    return (point) => [
      (dot(CAMERA_RIGHT, point) / (halfExtent * aspect) + 1) * 0.5 * panelWidth,
      (1 - dot(CAMERA_UP, point) / halfExtent) * 0.5 * panelHeight,
    ];
  }

  function drawShape({ context, voxels, rotation, offsetX, project }) {
    const matrix = rotationMatrix(rotation);
    const centroid = centroidOf(voxels);
    const cubes = voxels.map((voxel) => {
      const center = voxel.map((coordinate, axis) => coordinate - centroid[axis]);
      const corners = CORNER_SIGNS.map((signs) => rotate(matrix, center.map((coordinate, axis) => coordinate + signs[axis] * VOXEL_HALF_SIZE)));
      return { corners, depth: dot(VIEW_DIRECTION, rotate(matrix, center)) };
    });
    cubes.sort((cubeA, cubeB) => cubeA.depth - cubeB.depth);
    context.lineJoin = "round";
    context.lineWidth = 0.6; // covers the antialiasing seams between a cube's faces
    for (const cube of cubes) {
      const screenCorners = cube.corners.map((corner) => {
        const [screenX, screenY] = project(corner);
        return [screenX + offsetX, screenY];
      });
      for (const [cornerIndices, colorIndex] of FACES) {
        const [first, second, third] = cornerIndices.map((index) => screenCorners[index]);
        const screenCross = (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (third[0] - first[0]);
        if (screenCross >= 0) continue; // back face
        const [cornerA, cornerB, cornerC] = cornerIndices.map((index) => cube.corners[index]);
        const normal = normalize(cross(cornerB.map((value, axis) => value - cornerA[axis]), cornerC.map((value, axis) => value - cornerA[axis])));
        const shade = Math.min(1, AMBIENT + DIFFUSE * Math.max(0, dot(normal, LIGHT_DIRECTION)));
        const color = rgb(FACE_COLORS[colorIndex].map((channel) => Math.min(255, Math.floor(channel * shade))));
        context.beginPath();
        cornerIndices.forEach((index, order) => {
          const [screenX, screenY] = screenCorners[index];
          if (order === 0) context.moveTo(screenX, screenY);
          else context.lineTo(screenX, screenY);
        });
        context.closePath();
        context.fillStyle = color;
        context.strokeStyle = color;
        context.fill();
        context.stroke();
      }
    }
  }

  // The world axes in the left panel's bottom-left corner (render.rs draw_axis_gizmo).
  function drawAxisGizmo({ context, project }) {
    const { panelHeight } = panelGeometry();
    const axisLength = panelHeight * 0.12;
    const margin = panelHeight * 0.14;
    const [centerX, centerY] = [margin, panelHeight - margin];
    const origin = project([0, 0, 0]);
    const fontHeight = Math.max(panelHeight * 0.07, 8);
    context.lineWidth = 1;
    for (const { label, color, direction } of AXES) {
      const end = project(direction);
      const [deltaX, deltaY] = [end[0] - origin[0], end[1] - origin[1]];
      const length = Math.max(Math.hypot(deltaX, deltaY), 1e-6);
      const [stepX, stepY] = [(deltaX / length) * axisLength, (deltaY / length) * axisLength];
      context.strokeStyle = rgb(color);
      context.beginPath();
      context.moveTo(centerX, centerY);
      context.lineTo(centerX + stepX, centerY + stepY);
      context.stroke();
      drawEngineText({
        context,
        text: label,
        centerX: Math.trunc(centerX + stepX * 1.25),
        centerY: Math.trunc(centerY + stepY * 1.25),
        fontHeight,
        color,
      });
    }
  }

  function render({ context, puzzle, state, ui }) {
    const board = puzzle.board;
    const { dividerWidth, panelWidth, panelHeight } = panelGeometry();
    const project = projector({ board });
    context.fillStyle = rgb(BACKGROUND_COLOR);
    context.fillRect(0, 0, RENDER_SIZE, RENDER_SIZE);
    context.fillStyle = rgb(DIVIDER_COLOR);
    context.fillRect(panelWidth, 0, dividerWidth, panelHeight);
    const panels = [
      { offsetX: 0, rotation: displayedRotation({ state, ui }), voxels: board.voxels },
      { offsetX: panelWidth + dividerWidth, rotation: board.target, voxels: board.targetVoxels || board.voxels },
    ];
    for (const { offsetX, rotation, voxels } of panels) {
      context.save();
      context.beginPath();
      context.rect(offsetX, 0, panelWidth, panelHeight);
      context.clip();
      drawShape({ context, voxels, rotation, offsetX, project });
      context.restore();
    }
    context.save();
    context.beginPath();
    context.rect(0, 0, panelWidth, panelHeight);
    context.clip();
    drawAxisGizmo({ context, project });
    context.restore();
  }

  // ---------------------------------------------------------------------------
  // Interaction: rotation buttons per axis, keys, and dragging the left shape. Moves animate by easing
  // the Euler angles from the drawn rotation to the state's; a drag previews its snapped angle live.
  // ---------------------------------------------------------------------------

  function displayedRotation({ state, ui }) {
    const rotation = ui && ui.displayed ? { ...ui.displayed } : { rx: state.rx, ry: state.ry, rz: state.rz };
    if (ui && ui.drag && ui.drag.degrees) rotation[`r${ui.drag.axis}`] += ui.drag.degrees;
    return rotation;
  }

  function shortestDelta({ from, to }) {
    const delta = (((to - from) % 360) + 360) % 360;
    return delta > 180 ? delta - 360 : delta;
  }

  function signedLabel(degrees) {
    return `${degrees > 0 ? "+" : "−"}${Math.abs(degrees)}°`;
  }

  function mount({ puzzle, canvas, controls, act, getState, ui, redraw, toLogical }) {
    const board = puzzle.board;
    const largeStep = board.maxStep;
    const smallStep = board.increment;
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const rotateAction = ({ axis, degrees }) => `rotate${axis}(${degrees})`;

    for (const { axis, label } of AXES) {
      const group = document.createElement("div");
      group.className = "play-group";
      const groupLabel = document.createElement("span");
      groupLabel.className = "play-group-label";
      groupLabel.textContent = `${label} axis`;
      group.append(groupLabel);
      for (const degrees of [-largeStep, -smallStep, smallStep, largeStep]) {
        const action = rotateAction({ axis, degrees });
        group.append(button({ label: signedLabel(degrees), title: action, onClick: () => act(action) }));
      }
      controls.append(group);
    }

    let animationFrame = null;
    let shownState = getState();

    function animate() {
      const animation = ui.animation;
      const progress = Math.min(1, (performance.now() - animation.start) / ANIMATION_MS);
      const eased = 1 - (1 - progress) ** 3;
      ui.displayed = {};
      for (const key of ["rx", "ry", "rz"]) ui.displayed[key] = animation.from[key] + animation.deltas[key] * eased;
      redraw();
      if (progress < 1) {
        animationFrame = requestAnimationFrame(animate);
      } else {
        ui.displayed = null;
        ui.animation = null;
        animationFrame = null;
        redraw();
      }
    }

    function onStateChange({ state }) {
      const from = ui.displayed || (ui.dragReleased ? ui.dragReleased : shownState);
      ui.dragReleased = null;
      shownState = state;
      const deltas = {};
      for (const key of ["rx", "ry", "rz"]) deltas[key] = shortestDelta({ from: from[key], to: state[key] });
      if (reducedMotion || (!deltas.rx && !deltas.ry && !deltas.rz)) return;
      ui.animation = { from: { rx: state.rx - deltas.rx, ry: state.ry - deltas.ry, rz: state.rz - deltas.rz }, deltas, start: performance.now() };
      ui.displayed = { ...ui.animation.from };
      if (animationFrame === null) animationFrame = requestAnimationFrame(animate);
    }

    // Dragging the left shape: horizontal turns it about Y, vertical about X (the first direction past the
    // dead zone picks the axis). The angle snaps to the increment and becomes one action on release.
    const { panelWidth } = panelGeometry();
    function onPointerDown(event) {
      const point = toLogical(event);
      if (point.x > panelWidth) return;
      canvas.setPointerCapture(event.pointerId);
      ui.drag = { startX: point.x, startY: point.y, axis: null, degrees: 0 };
    }
    function onPointerMove(event) {
      if (!ui.drag) return;
      const point = toLogical(event);
      const [deltaX, deltaY] = [point.x - ui.drag.startX, point.y - ui.drag.startY];
      if (!ui.drag.axis) {
        if (Math.max(Math.abs(deltaX), Math.abs(deltaY)) < DRAG_DEAD_ZONE) return;
        ui.drag.axis = Math.abs(deltaX) >= Math.abs(deltaY) ? "y" : "x";
      }
      const rawDegrees = (ui.drag.axis === "y" ? deltaX : deltaY) / DRAG_PIXELS_PER_DEGREE;
      const snapped = Math.round(rawDegrees / smallStep) * smallStep;
      ui.drag.degrees = Math.max(-180, Math.min(180, snapped));
      redraw();
    }
    function onPointerUp() {
      if (!ui.drag) return;
      const { axis, degrees } = ui.drag;
      if (axis && degrees) ui.dragReleased = displayedRotation({ state: getState(), ui });
      ui.drag = null;
      if (!(axis && degrees && act(rotateAction({ axis, degrees })))) {
        ui.dragReleased = null;
        redraw();
      }
    }
    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointercancel", onPointerUp);

    // Arrows turn about Y (left/right) and X (up/down), Q/E about Z; Shift for the small step.
    const KEY_ROTATIONS = {
      ArrowRight: ["y", 1], ArrowLeft: ["y", -1], ArrowDown: ["x", 1], ArrowUp: ["x", -1],
      q: ["z", 1], Q: ["z", 1], e: ["z", -1], E: ["z", -1],
    };

    return {
      onKey(event) {
        if (event.metaKey || event.ctrlKey || event.altKey) return false;
        const rotation = KEY_ROTATIONS[event.key];
        if (!rotation) return false;
        const [axis, sign] = rotation;
        act(rotateAction({ axis, degrees: sign * (event.shiftKey ? smallStep : largeStep) }));
        return true;
      },
      onStateChange,
      destroy() {
        if (animationFrame !== null) cancelAnimationFrame(animationFrame);
        canvas.removeEventListener("pointerdown", onPointerDown);
        canvas.removeEventListener("pointermove", onPointerMove);
        canvas.removeEventListener("pointerup", onPointerUp);
        canvas.removeEventListener("pointercancel", onPointerUp);
      },
    };
  }

  window.PuzzleGames.mental_rotation = {
    goal: "Rotate the left shape until it matches the right one.",
    help: "Use controls above or drag the left shape to rotate it.",
    initialState: ({ puzzle }) => puzzle.board.state,
    apply,
    isSolved,
    render,
    mount,
  };
})();
