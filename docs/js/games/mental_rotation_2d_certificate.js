/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2026 Apple Inc. All Rights Reserved.
 */
// Fixed certificate variant: inspect the two polyominoes, then certify whether rotations can match them.
(function () {
  "use strict";

  window.PuzzleGames.mental_rotation_2d_certificate = window.PuzzleKit.certificateGame({
    base: window.PuzzleGames.mental_rotation_2d,
    goal: "Determine whether the left and right 2D shapes are the same, allowing rotation.",
    help: "Use controls above or drag the left shape to inspect it, then choose Yes or No.",
  });
})();
