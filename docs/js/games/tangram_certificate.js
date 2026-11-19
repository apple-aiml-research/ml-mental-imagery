/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2026 Apple Inc. All Rights Reserved.
 */
// Fixed certificate variant: try the supplied pieces, then certify whether they can fill the silhouette.
(function () {
  "use strict";

  window.PuzzleGames.tangram_certificate = window.PuzzleKit.certificateGame({
    base: window.PuzzleGames.tangram,
    goal: "Determine whether these pieces can fill the grey silhouette without overlaps.",
    help: "Try placing and rotating the pieces, then choose Yes or No.",
  });
})();
