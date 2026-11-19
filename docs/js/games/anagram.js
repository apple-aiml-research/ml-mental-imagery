/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2026 Apple Inc. All Rights Reserved.
 */
// Anagram (engine/games/anagram): swap letters back into a word, then submit it.
// Actions: swap(i, j) of two slots, and identify('word') once the slots spell a valid word.
(function () {
  "use strict";

  const { RENDER_SIZE, button, rgb, drawEngineText } = window.PuzzleKit;

  const BACKGROUND_COLOR = [0xe8, 0xe8, 0xe8];
  const TEXT_COLOR = [0xff, 0xff, 0xff];
  const INDEX_COLOR = [0x60, 0x60, 0x60];
  const SELECTION_COLOR = [0x1a, 0x1a, 0x2e];
  const TILE_COLORS = [
    [0x4a, 0x90, 0xd9], [0xe8, 0x6b, 0x6b], [0x6b, 0xb8, 0x6b], [0xe8, 0xa8, 0x4a],
    [0x9b, 0x6b, 0xd9], [0xd9, 0x6b, 0xa8], [0x4a, 0xb8, 0xb8], [0xb8, 0x8b, 0x4a],
  ];

  function parseAction(action) {
    const swap = /^swap\((\d+), (\d+)\)$/.exec(action);
    if (swap) return { kind: "swap", first: Number(swap[1]), second: Number(swap[2]) };
    const identify = /^identify\('(.+)'\)$/.exec(action);
    if (identify) return { kind: "identify", word: identify[1] };
    return null;
  }

  // The strict rule (qa_only_eval off): identify only the word the slots spell, and only if it's in the dictionary.
  function apply({ puzzle, state, action }) {
    const parsed = parseAction(action);
    if (!parsed || state.identified) return null;
    const letters = state.letters;
    if (parsed.kind === "identify") {
      if (!puzzle.board.valid_words.includes(parsed.word) || parsed.word !== letters) return null;
      return { letters, identified: true };
    }
    const { first, second } = parsed;
    if (first === second || first >= letters.length || second >= letters.length) return null;
    const swapped = letters.split("");
    [swapped[first], swapped[second]] = [swapped[second], swapped[first]];
    return { letters: swapped.join(""), identified: false };
  }

  function isSolved({ state }) {
    return state.identified;
  }

  // Tiles keep one color per puzzle: a hash of the sorted letters picks it.
  function tileColor(letters) {
    let hash = 0;
    for (const letter of letters.split("").sort()) hash = (Math.imul(hash, 31) + letter.charCodeAt(0)) >>> 0;
    return TILE_COLORS[hash % TILE_COLORS.length];
  }

  // The engine's layout (render.rs): up to five tiles in one row, else two rows with the extra tile on top,
  // sized as if five fit across. Each tile's index sits under it.
  function layout(letterCount) {
    const margin = Math.floor(RENDER_SIZE / 10);
    const gap = Math.max(2, Math.floor(RENDER_SIZE / 40));
    const tileSize = Math.max(4, Math.floor((RENDER_SIZE - 2 * margin - 4 * gap) / 5));
    const letterFontHeight = Math.max(8, tileSize * 0.6);
    const indexFontHeight = Math.max(6, tileSize * 0.3);
    const topRowCount = letterCount <= 5 ? letterCount : Math.ceil(letterCount / 2);
    const rows = letterCount <= 5 ? [[0, letterCount]] : [[0, topRowCount], [topRowCount, letterCount]];
    const rowHeight = tileSize + gap * 2 + Math.floor(indexFontHeight);
    const rowGap = gap * 2;
    const totalHeight = rows.length * rowHeight + (rows.length - 1) * rowGap;
    const startY = Math.floor(Math.max(0, RENDER_SIZE - totalHeight) / 2);
    const tiles = [];
    rows.forEach(([from, to], rowIndex) => {
      const count = to - from;
      const startX = Math.floor((RENDER_SIZE - (count * tileSize + (count - 1) * gap)) / 2);
      const tileY = startY + rowIndex * (rowHeight + rowGap);
      for (let offset = 0; offset < count; offset += 1) {
        tiles.push({ slot: from + offset, x: startX + offset * (tileSize + gap), y: tileY });
      }
    });
    return { tiles, tileSize, gap, letterFontHeight, indexFontHeight };
  }

  function slotAt({ letterCount, point }) {
    const { tiles, tileSize } = layout(letterCount);
    const tile = tiles.find(({ x, y }) => point.x >= x && point.x < x + tileSize && point.y >= y && point.y < y + tileSize);
    return tile ? tile.slot : null;
  }

  function render({ context, state, ui }) {
    const letters = state.letters;
    const { tiles, tileSize, gap, letterFontHeight, indexFontHeight } = layout(letters.length);
    const color = tileColor(letters);
    context.fillStyle = rgb(BACKGROUND_COLOR);
    context.fillRect(0, 0, RENDER_SIZE, RENDER_SIZE);
    tiles.forEach(({ slot, x, y }) => {
      if (slot === ui.selectedSlot) {
        context.fillStyle = rgb(SELECTION_COLOR);
        context.fillRect(x - 5, y - 5, tileSize + 10, tileSize + 10);
      }
      context.fillStyle = rgb(color);
      context.fillRect(x, y, tileSize, tileSize);
      const centerX = x + Math.floor(tileSize / 2);
      drawEngineText({ context, text: letters[slot].toUpperCase(), centerX, centerY: y + Math.floor(tileSize / 2), fontHeight: letterFontHeight, color: TEXT_COLOR });
      const indexCenterY = y + tileSize + gap * 2 + Math.trunc(indexFontHeight / 2);
      drawEngineText({ context, text: String(slot), centerX, centerY: indexCenterY, fontHeight: indexFontHeight, color: INDEX_COLOR });
    });
  }

  function mount({ canvas, controls, act, getState, ui, redraw, toLogical }) {
    ui.selectedSlot = null;
    // The model writes the smaller slot first.
    const swap = (slotA, slotB) => act(`swap(${Math.min(slotA, slotB)}, ${Math.max(slotA, slotB)})`);
    const identify = () => act(`identify('${getState().letters}')`);
    const submitButton = button({ label: "", className: "is-primary", title: "Submit the word the tiles spell (Enter)", onClick: identify });
    controls.append(submitButton);

    function select(slot) {
      ui.selectedSlot = slot;
      redraw();
    }

    // Tap a tile, then another, to swap them; or drag one tile onto another.
    let pressedSlot = null;
    function onPointerDown(event) {
      const slot = slotAt({ letterCount: getState().letters.length, point: toLogical(event) });
      pressedSlot = null;
      if (slot === null || slot === ui.selectedSlot) {
        select(null);
        return;
      }
      if (ui.selectedSlot !== null) {
        const firstSlot = ui.selectedSlot;
        ui.selectedSlot = null;
        if (!swap(firstSlot, slot)) redraw();
        return;
      }
      pressedSlot = slot;
      canvas.setPointerCapture(event.pointerId);
      select(slot);
    }

    function onPointerMove(event) {
      const slot = slotAt({ letterCount: getState().letters.length, point: toLogical(event) });
      canvas.style.cursor = slot === null ? "" : "pointer";
    }

    function onPointerUp(event) {
      if (pressedSlot === null) return;
      const slot = slotAt({ letterCount: getState().letters.length, point: toLogical(event) });
      const draggedFrom = pressedSlot;
      pressedSlot = null;
      if (slot === null || slot === draggedFrom) return;
      ui.selectedSlot = null;
      if (!swap(draggedFrom, slot)) redraw();
    }

    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);

    return {
      // Digit keys pick slots (words have at most ten letters); Enter submits.
      onKey(event) {
        if (event.metaKey || event.ctrlKey || event.altKey) return false;
        if (event.key === "Enter") {
          identify();
          return true;
        }
        if (!/^\d$/.test(event.key)) return false;
        const slot = Number(event.key);
        if (slot >= getState().letters.length) return true;
        if (ui.selectedSlot === null || ui.selectedSlot === slot) {
          select(ui.selectedSlot === slot ? null : slot);
        } else {
          const firstSlot = ui.selectedSlot;
          ui.selectedSlot = null;
          if (!swap(firstSlot, slot)) redraw();
        }
        return true;
      },
      onStateChange({ state }) {
        submitButton.textContent = `Submit “${state.letters.toUpperCase()}”`;
        submitButton.disabled = state.identified;
        ui.selectedSlot = null; // an undo or reset drops a half-made swap
      },
      destroy() {
        canvas.removeEventListener("pointerdown", onPointerDown);
        canvas.removeEventListener("pointermove", onPointerMove);
        canvas.removeEventListener("pointerup", onPointerUp);
        canvas.style.cursor = "";
      },
    };
  }

  window.PuzzleGames.anagram = {
    goal: "Unscramble the letters into a word, then submit it.",
    help: "Click two letters to swap them. Submit the word when it is unscrambled.",
    initialState: ({ puzzle }) => puzzle.board.state,
    apply,
    isSolved,
    render,
    mount,
  };
})();
