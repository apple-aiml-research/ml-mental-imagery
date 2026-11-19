/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2026 Apple Inc. All Rights Reserved.
 */
// Sokoban (engine/games/sokoban): push every box onto a goal. Actions: up, down, left, right.
(function () {
  "use strict";

  const { RENDER_SIZE, keyDirection, directionPad, fitToResolution, rgb } = window.PuzzleKit;

  const WALL_COLOR = [0x40, 0x40, 0x40];
  const FLOOR_COLOR = [0xd0, 0xd0, 0xd0];
  const PLAYER_COLOR = [0x44, 0x44, 0xff];
  const BOX_COLOR = [0x8b, 0x45, 0x13];
  const GOAL_COLOR = [0xff, 0x44, 0x44];
  const BOX_ON_GOAL_COLOR = [0x44, 0xaa, 0x44];
  const PLAYER_ON_GOAL_COLOR = [0x44, 0xaa, 0xaa];

  const DIRECTION_STEPS = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] };

  const sameCell = ([colA, rowA], [colB, rowB]) => colA === colB && rowA === rowB;
  const hasCell = (cells, cell) => cells.some((other) => sameCell(other, cell));
  const compareCells = ([colA, rowA], [colB, rowB]) => colA - colB || rowA - rowB;

  function isWall(board, [col, row]) {
    return col < 0 || row < 0 || col >= board.width || row >= board.height || board.rows[row][col] === "#";
  }

  function apply({ puzzle, state, action }) {
    const step = DIRECTION_STEPS[action];
    if (!step) return null;
    const board = puzzle.board;
    const [player, boxes] = [state.player, state.boxes];
    const target = [player[0] + step[0], player[1] + step[1]];
    if (isWall(board, target)) return null;
    if (!hasCell(boxes, target)) return { player: target, boxes };
    const pushedTo = [target[0] + step[0], target[1] + step[1]];
    if (isWall(board, pushedTo) || hasCell(boxes, pushedTo)) return null;
    const movedBoxes = boxes.filter((box) => !sameCell(box, target)).concat([pushedTo]).sort(compareCells);
    return { player: target, boxes: movedBoxes };
  }

  function isSolved({ puzzle, state }) {
    return puzzle.board.goals.every((goal) => hasCell(state.boxes, goal));
  }

  function fillDiamond(context, centerX, centerY, half) {
    context.beginPath();
    context.moveTo(centerX, centerY - half - 0.5);
    context.lineTo(centerX + half + 0.5, centerY);
    context.lineTo(centerX, centerY + half + 0.5);
    context.lineTo(centerX - half - 0.5, centerY);
    context.closePath();
    context.fill();
  }

  function render({ context, puzzle, state }) {
    const board = puzzle.board;
    const { offsetX, offsetY, cellSize } = fitToResolution({ boardWidth: board.width, boardHeight: board.height });
    context.fillStyle = rgb(FLOOR_COLOR);
    context.fillRect(0, 0, RENDER_SIZE, RENDER_SIZE);
    const margin = Math.floor(cellSize / 6);
    const innerSize = cellSize - 2 * margin;
    for (let row = 0; row < board.height; row += 1) {
      for (let col = 0; col < board.width; col += 1) {
        const cellX = offsetX + col * cellSize;
        const cellY = offsetY + row * cellSize;
        context.fillStyle = rgb(isWall(board, [col, row]) ? WALL_COLOR : FLOOR_COLOR);
        context.fillRect(cellX, cellY, cellSize, cellSize);

        const cell = [col, row];
        const isPlayer = sameCell(state.player, cell);
        const isBox = hasCell(state.boxes, cell);
        const isGoal = hasCell(board.goals, cell);
        const centerX = cellX + Math.floor(cellSize / 2);
        const centerY = cellY + Math.floor(cellSize / 2);
        if (isPlayer) {
          context.fillStyle = rgb(isGoal ? PLAYER_ON_GOAL_COLOR : PLAYER_COLOR);
          context.beginPath();
          context.arc(centerX, centerY, Math.floor(innerSize / 2) + 0.5, 0, 2 * Math.PI);
          context.fill();
        } else if (isBox) {
          context.fillStyle = rgb(isGoal ? BOX_ON_GOAL_COLOR : BOX_COLOR);
          context.fillRect(cellX + margin, cellY + margin, innerSize, innerSize);
        } else if (isGoal) {
          context.fillStyle = rgb(GOAL_COLOR);
          fillDiamond(context, centerX, centerY, Math.floor(innerSize / 2));
        }
      }
    }
  }

  function mount({ puzzle, canvas, controls, act, getState, toLogical }) {
    controls.append(directionPad({ onDirection: act }));

    // A tap steps the player one cell toward it, along the axis it is farther away on.
    function onPointerDown(event) {
      const board = puzzle.board;
      const { offsetX, offsetY, cellSize } = fitToResolution({ boardWidth: board.width, boardHeight: board.height });
      const point = toLogical(event);
      const [playerCol, playerRow] = getState().player;
      const deltaX = point.x - (offsetX + (playerCol + 0.5) * cellSize);
      const deltaY = point.y - (offsetY + (playerRow + 0.5) * cellSize);
      if (Math.max(Math.abs(deltaX), Math.abs(deltaY)) < cellSize / 2) return;
      if (Math.abs(deltaX) > Math.abs(deltaY)) act(deltaX > 0 ? "right" : "left");
      else act(deltaY > 0 ? "down" : "up");
    }
    canvas.addEventListener("pointerdown", onPointerDown);

    return {
      onKey(event) {
        const direction = keyDirection(event);
        if (!direction) return false;
        act(direction);
        return true;
      },
      destroy() {
        canvas.removeEventListener("pointerdown", onPointerDown);
      },
    };
  }

  window.PuzzleGames.sokoban = {
    goal: "Push every brown box onto a red diamond.",
    help: "Use arrow keys or WASD to move the player.",
    initialState: ({ puzzle }) => puzzle.board.state,
    apply,
    isSolved,
    render,
    mount,
  };
})();
