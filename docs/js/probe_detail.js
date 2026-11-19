/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2026 Apple Inc. All Rights Reserved.
 */
// "How the probe works", opened from any [data-probe-detail] element: the visual decoder of the paper's
// architecture figure in the page's style, with the paper's description of it. The probe reads the VLM's column at the token that closes
// an action, one vector per layer (mental_imagery/model/visual_head.py), and a small transformer turns that column plus a
// learned query per image patch into the imagined frame.
(function () {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";
  // Theme colors (style.css), so the figure follows light and dark mode.
  const DREAM = "var(--dream)";
  const DREAM_FILL = "light-dark(#ebe7fb, #2a2543)";
  const BOX_STROKE = "light-dark(#b9b8c6, #4a4958)";
  const TOKEN_FILL = "var(--token-quiet)";
  const BOX_FILL = "var(--card)";
  const INK = "var(--ink)";
  // Set as CSS properties: SVG presentation attributes don't take var() or light-dark().
  const STYLE_ATTRIBUTES = new Set(["fill", "stroke"]);
  const MONO = "ui-monospace, SF Mono, Menlo, monospace";
  const SANS = "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif";

  // Token columns, as in the paper: the input token at the bottom, the predicted next token on top.
  const INPUT_TOKENS = ["2", ")", "</action>"];
  const NEXT_TOKENS = [")", "</action>", "slide"];
  const READ_COLUMN = 2;
  const LAYER_ROWS = [
    { label: "Layer 32", y: 106 },
    { label: "", y: 146, isGap: true },
    { label: "Layer 2", y: 166 },
    { label: "Layer 1", y: 206 },
  ];
  const NEXT_ROW_Y = 60;
  const INPUT_ROW_Y = 252;
  const QUERY_ROWS = [{ label: "p₂", y: 14 }, { label: "p₁", y: 60 }];

  const GRID_LEFT = 110;
  const BOX_WIDTH = 76;
  const BOX_HEIGHT = 30;
  const COLUMN_PITCH = 88;
  const COPY_LEFT = 470;
  const PROBE_LEFT = 576;
  const PROBE_WIDTH = 46;
  const IMAGE_LEFT = 664;
  const IMAGE_SIZE = 120;
  // The emergence story's imagined frame after slide('H', 2) (rush hour #8, frame 4 of 9).
  const IMAGE_STRIP = "media/stories/rush_hour-0008-action_supervision-imagination.webp";
  const IMAGE_FRAME = 4;
  const IMAGE_FRAME_COUNT = 9;
  const FRAME_PIXELS = 256;

  function svg(tag, attributes = {}, content) {
    const node = document.createElementNS(SVG_NS, tag);
    Object.entries(attributes).forEach(([name, value]) => {
      if (STYLE_ATTRIBUTES.has(name)) node.style.setProperty(name, String(value));
      else node.setAttribute(name, String(value));
    });
    if (content !== undefined) node.textContent = content;
    return node;
  }

  function text({ x, y, content, size = 13, color = INK, anchor = "middle", mono = false }) {
    return svg("text", { x, y, "font-size": size, "font-family": mono ? MONO : SANS, fill: color, "text-anchor": anchor, "dominant-baseline": "central" }, content);
  }

  function box({ x, y, label, fill = BOX_FILL, stroke = BOX_STROKE, mono = true }) {
    const group = svg("g");
    group.append(svg("rect", { x, y, width: BOX_WIDTH, height: BOX_HEIGHT, rx: 6, fill, stroke }));
    if (label) group.append(text({ x: x + BOX_WIDTH / 2, y: y + BOX_HEIGHT / 2, content: label, size: 12, mono }));
    return group;
  }

  function gapDots({ x }) {
    return text({ x: x + BOX_WIDTH / 2, y: LAYER_ROWS[1].y + 6, content: "⋮", size: 14 });
  }

  function arrow({ x1, x2, y, color }) {
    const group = svg("g");
    group.append(
      svg("line", { x1, y1: y, x2: x2 - 6, y2: y, stroke: color, "stroke-width": 1.8 }),
      svg("path", { d: `M${x2 - 7} ${y - 5} L${x2} ${y} L${x2 - 7} ${y + 5} Z`, fill: color }),
    );
    return group;
  }

  function buildFigure() {
    const figure = svg("svg", { viewBox: "0 0 800 290", role: "img", "aria-label": "The probe takes the VLM's column at the end of an action, one vector per layer, and with a query per image patch a small transformer predicts the imagined frame." });
    const layerRows = LAYER_ROWS.filter((row) => !row.isGap);

    // The VLM, token by token: a column of layers per token.
    figure.append(text({ x: GRID_LEFT - 12, y: NEXT_ROW_Y + BOX_HEIGHT / 2, content: "Next token", size: 12, anchor: "end" }));
    layerRows.forEach((row) => figure.append(text({ x: GRID_LEFT - 12, y: row.y + BOX_HEIGHT / 2, content: row.label, size: 12, anchor: "end" })));
    const readLeft = GRID_LEFT + READ_COLUMN * COLUMN_PITCH;
    const layersTop = LAYER_ROWS[0].y;
    const layersBottom = LAYER_ROWS[LAYER_ROWS.length - 1].y + BOX_HEIGHT;
    figure.append(svg("rect", {
      x: readLeft - 6, y: layersTop - 6, width: BOX_WIDTH + 12, height: layersBottom - layersTop + 12,
      rx: 9, fill: DREAM_FILL, stroke: DREAM, "stroke-width": 1.5, "stroke-dasharray": "4 3",
    }));
    INPUT_TOKENS.forEach((token, column) => {
      const left = GRID_LEFT + column * COLUMN_PITCH;
      figure.append(box({ x: left, y: NEXT_ROW_Y, label: NEXT_TOKENS[column], fill: TOKEN_FILL }));
      layerRows.forEach((row) => figure.append(box({ x: left, y: row.y })));
      figure.append(gapDots({ x: left }), box({ x: left, y: INPUT_ROW_Y, label: token, fill: TOKEN_FILL }));
    });

    // The read column, carried over to the probe, under a learned query per image patch.
    figure.append(arrow({ x1: readLeft + BOX_WIDTH + 12, x2: COPY_LEFT - 10, y: (layersTop + layersBottom) / 2, color: DREAM }));
    layerRows.forEach((row) => figure.append(box({ x: COPY_LEFT, y: row.y, fill: DREAM_FILL, stroke: DREAM })));
    figure.append(gapDots({ x: COPY_LEFT }));
    QUERY_ROWS.forEach((row) => figure.append(box({ x: COPY_LEFT, y: row.y, label: row.label, mono: false })));
    // The queries are learned position encodings, one per patch: a bracket and label to their left.
    const queriesTop = QUERY_ROWS[0].y;
    const queriesBottom = QUERY_ROWS[QUERY_ROWS.length - 1].y + BOX_HEIGHT;
    const bracketX = COPY_LEFT - 10;
    figure.append(
      svg("path", { d: `M${bracketX + 5} ${queriesTop} H${bracketX} V${queriesBottom} H${bracketX + 5}`, fill: "none", stroke: INK, "stroke-width": 1.2 }),
      text({ x: bracketX - 8, y: (queriesTop + queriesBottom) / 2 - 8, content: "Position", size: 12, anchor: "end" }),
      text({ x: bracketX - 8, y: (queriesTop + queriesBottom) / 2 + 8, content: "encodings", size: 12, anchor: "end" }),
    );

    // The probe: a small transformer over the queries and the column's layers.
    const probeTop = QUERY_ROWS[0].y;
    const probeCenterX = PROBE_LEFT + PROBE_WIDTH / 2;
    const probeCenterY = (probeTop + layersBottom) / 2;
    QUERY_ROWS.concat(layerRows).forEach((row) => {
      figure.append(svg("line", { x1: COPY_LEFT + BOX_WIDTH, y1: row.y + BOX_HEIGHT / 2, x2: PROBE_LEFT, y2: row.y + BOX_HEIGHT / 2, stroke: BOX_STROKE, "stroke-width": 1.2 }));
    });
    figure.append(
      svg("rect", { x: PROBE_LEFT, y: probeTop, width: PROBE_WIDTH, height: layersBottom - probeTop, rx: 8, fill: BOX_FILL, stroke: INK, "stroke-width": 1.2 }),
      svg("text", {
        x: probeCenterX, y: probeCenterY, "font-size": 14, "font-family": SANS, fill: INK, "text-anchor": "middle", "dominant-baseline": "central",
        transform: `rotate(-90 ${probeCenterX} ${probeCenterY})`,
      }, "Probe transformer"),
    );

    // The queries come out as the image's patches.
    const imageTop = QUERY_ROWS[0].y;
    figure.append(arrow({ x1: PROBE_LEFT + PROBE_WIDTH, x2: IMAGE_LEFT - 6, y: imageTop + 46, color: INK }));
    const frame = svg("svg", { x: IMAGE_LEFT, y: imageTop, width: IMAGE_SIZE, height: IMAGE_SIZE, viewBox: `${IMAGE_FRAME * FRAME_PIXELS} 0 ${FRAME_PIXELS} ${FRAME_PIXELS}` });
    frame.append(svg("image", { href: IMAGE_STRIP, width: IMAGE_FRAME_COUNT * FRAME_PIXELS, height: FRAME_PIXELS }));
    figure.append(
      frame,
      svg("rect", { x: IMAGE_LEFT, y: imageTop, width: IMAGE_SIZE, height: IMAGE_SIZE, rx: 4, fill: "none", stroke: BOX_STROKE }),
      text({ x: IMAGE_LEFT + IMAGE_SIZE / 2, y: imageTop + IMAGE_SIZE + 16, content: "Imagined frame", size: 12 }),
    );
    return figure;
  }

  let dialog = null;

  function buildDialog() {
    dialog = document.createElement("dialog");
    dialog.className = "viewer probe-detail";
    dialog.setAttribute("aria-labelledby", "probe-detail-title");
    dialog.innerHTML = `
      <div class="viewer-head">
        <h2 id="probe-detail-title">How the probe works</h2>
        <button type="button" class="link-button" data-close>Close</button>
      </div>
      <div class="probe-figure"></div>
      <p class="probe-notes">To probe for emergent mental imagery, we build a visual decoder that is trained separately.
      We use a four-layer transformer based on masked autoencoders. We condition only on the hidden features of the last
      token of each action, and train to predict the visual state after applying this action. Since the decoder only sees
      the representation for one token (and not even the initial image), it allows us to understand how well the VLM is
      tracking the visual state through long horizons.</p>
      <p class="probe-notes">For visual supervision, we attach the same visual prediction head, but now allow the
      gradients to flow between the two models in order to jointly learn behavior cloning with imagination of the visual
      state.</p>`;
    dialog.querySelector(".probe-figure").append(buildFigure());
    dialog.querySelector("[data-close]").addEventListener("click", () => dialog.close());
    // A click on the backdrop (the dialog element itself, outside its content) closes it.
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) dialog.close();
    });
    document.body.append(dialog);
  }

  document.addEventListener("click", (event) => {
    if (!event.target.closest("[data-probe-detail]")) return;
    if (!dialog) buildDialog();
    dialog.showModal();
  });
})();
