/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2026 Apple Inc. All Rights Reserved.
 */
// Hero: the VLM's imagined poses, seen through a gentle refraction (WebGL). Crossing waves bend each pose, red a
// little less than blue; when the next pose arrives, the bending swells and settles while the two crossfade.
// main.js shows the plain frames instead when this isn't ready: without WebGL, under reduced motion, or from file://,
// where browsers won't hand the images to WebGL.
(function () {
  "use strict";

  const TRANSITION_MS = 500; // the swell to the next pose, and its crossfade

  const VERTEX_SHADER = `
attribute vec2 aPosition;
varying vec2 vUv;
void main() {
  vUv = vec2(0.5 * (aPosition.x + 1.0), 0.5 * (1.0 - aPosition.y)); // y runs down, as in the images
  gl_Position = vec4(aPosition, 0.0, 1.0);
}`;

  const FRAGMENT_SHADER = `
precision highp float;
varying vec2 vUv;
uniform sampler2D uPreviousPose;
uniform sampler2D uCurrentPose;
uniform float uTime;      // seconds
uniform float uProgress;  // 0..1 through the transition to the current pose

const float SWELL = 2.2; // at its peak, the bending is this much stronger than at rest

// A pose as the plain frames lay it out (.scene-frame): inset 7% on white, faded toward its edges.
vec3 poseColor(sampler2D pose, vec2 uv) {
  vec2 imageUv = (uv - 0.07) / 0.86;
  float alpha = 1.0 - clamp((length((imageUv - 0.5) / 0.7) - 0.64) / 0.36, 0.0, 1.0);
  if (imageUv.x < 0.0 || imageUv.y < 0.0 || imageUv.x > 1.0 || imageUv.y > 1.0) alpha = 0.0;
  return mix(vec3(1.0), texture2D(pose, clamp(imageUv, 0.0, 1.0)).rgb, alpha);
}

// The steady waver: a few crossing waves.
vec2 waveOffset(vec2 uv, float time) {
  return 0.0042 * vec2(
    sin(uv.y * 17.0 + time * 1.3) + 0.6 * sin((uv.x + uv.y) * 27.0 - time * 1.7) + 0.3 * sin(uv.x * 41.0 + time * 2.3),
    cos(uv.x * 15.0 - time * 1.1) + 0.6 * sin((uv.x - uv.y) * 23.0 + time * 1.9) + 0.3 * cos(uv.y * 37.0 - time * 2.1)
  );
}

void main() {
  vec2 uv = vUv;
  // The swell rises from nothing and falls back to nothing, flat at both ends so it never jumps; the poses cross
  // in its middle.
  float bell = sin(3.14159 * uProgress);
  vec2 wave = waveOffset(uv, uTime) * (1.0 + SWELL * bell * bell);
  float reveal = smoothstep(0.25, 0.75, uProgress);
  vec2 redUv = uv + wave * 0.88;
  vec2 greenUv = uv + wave;
  vec2 blueUv = uv + wave * 1.12;
  gl_FragColor = vec4(
    mix(poseColor(uPreviousPose, redUv).r, poseColor(uCurrentPose, redUv).r, reveal),
    mix(poseColor(uPreviousPose, greenUv).g, poseColor(uCurrentPose, greenUv).g, reveal),
    mix(poseColor(uPreviousPose, blueUv).b, poseColor(uCurrentPose, blueUv).b, reveal),
    1.0
  );
}`;

  function compileProgram(gl) {
    const compileShader = ({ type, source }) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
      return shader;
    };
    const program = gl.createProgram();
    gl.attachShader(program, compileShader({ type: gl.VERTEX_SHADER, source: VERTEX_SHADER }));
    gl.attachShader(program, compileShader({ type: gl.FRAGMENT_SHADER, source: FRAGMENT_SHADER }));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    return program;
  }

  // Throws (a SecurityError) for images the page may not read, such as other files under file://.
  function makePoseTexture({ gl, image }) {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
    return texture;
  }

  // Adds the refracted view to host (the thought bubble), drawing poseImages (the plain frames' <img>s). Returns
  // null without WebGL. Its isReady stays false until every pose is on the GPU, and for good if the browser
  // refuses them; it drops back to false if the GPU context is lost, until it's restored.
  function create({ host, poseImages }) {
    const canvas = document.createElement("canvas");
    canvas.className = "scene-refraction";
    canvas.setAttribute("aria-hidden", "true");
    const gl = canvas.getContext("webgl", { antialias: false, premultipliedAlpha: false });
    if (!gl) return null;
    host.appendChild(canvas);

    let isReady = false;
    let isShowing = false;
    let isOnScreen = true;
    let frameRequest = 0;
    let poseTextures = [];
    let timeLocation = null;
    let progressLocation = null;
    let previousPoseIndex = 0;
    let currentPoseIndex = 0;
    let poseShownAt = 0;

    function initialize() {
      const program = compileProgram(gl);
      gl.useProgram(program);
      gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
      const positionLocation = gl.getAttribLocation(program, "aPosition");
      gl.enableVertexAttribArray(positionLocation);
      gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);
      poseTextures = poseImages.map((image) => makePoseTexture({ gl, image }));
      gl.uniform1i(gl.getUniformLocation(program, "uPreviousPose"), 0);
      gl.uniform1i(gl.getUniformLocation(program, "uCurrentPose"), 1);
      timeLocation = gl.getUniformLocation(program, "uTime");
      progressLocation = gl.getUniformLocation(program, "uProgress");
      isReady = true;
      requestRender();
    }

    function initializeOrGiveUp() {
      try {
        initialize();
      } catch {
        canvas.remove();
      }
    }

    Promise.all(poseImages.map((image) => image.decode())).then(initializeOrGiveUp, () => canvas.remove());
    canvas.addEventListener("webglcontextlost", (event) => {
      event.preventDefault(); // so the browser restores the context
      isReady = false;
    });
    canvas.addEventListener("webglcontextrestored", initializeOrGiveUp);
    new IntersectionObserver(([entry]) => {
      isOnScreen = entry.isIntersecting;
      requestRender();
    }).observe(canvas);

    // Draws only while a pose is showing and on screen.
    function render(now) {
      frameRequest = 0;
      if (!isReady || !isShowing || !isOnScreen) return;
      const pixelSize = Math.round(canvas.clientWidth * (window.devicePixelRatio || 1));
      if (canvas.width !== pixelSize) canvas.width = canvas.height = pixelSize;
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, poseTextures[previousPoseIndex]);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, poseTextures[currentPoseIndex]);
      gl.uniform1f(timeLocation, now / 1000);
      gl.uniform1f(progressLocation, Math.min(1, (now - poseShownAt) / TRANSITION_MS));
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      frameRequest = requestAnimationFrame(render);
    }

    function requestRender() {
      if (!frameRequest) frameRequest = requestAnimationFrame(render);
    }

    return {
      get isReady() {
        return isReady;
      },
      // The first pose after hide() has nothing to swell in from, so it arrives from itself.
      showPose(poseIndex) {
        previousPoseIndex = isShowing ? currentPoseIndex : poseIndex;
        currentPoseIndex = poseIndex;
        poseShownAt = performance.now();
        isShowing = true;
        canvas.classList.add("is-visible");
        requestRender();
      },
      hide() {
        isShowing = false;
        canvas.classList.remove("is-visible");
      },
    };
  }

  window.SheepRefraction = { create };
})();
