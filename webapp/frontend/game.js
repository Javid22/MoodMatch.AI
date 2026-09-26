/* ==========================================================================
   MoodMatch.AI — game logic
   Talks to the backend only through two endpoints:
     GET  /api/random-target   -> { emotion, emoji, image_url, instruction }
     POST /api/predict         -> { status, predicted_emotion, confidence, ... }
   All score/hearts/round rules live here, kept separate from the ML layer.

   Round flow: the player clicks "I'M READY" when they want to attempt the
   current target. That triggers a 3-2-1-MATCH countdown, the frame freezes
   at the end of it, and the captured frame is sent for prediction. After
   every result (match, miss, or a no-face/error case) the game pauses and
   shows the "I'M READY" button again instead of restarting on its own —
   the player decides when the next attempt happens.
   ========================================================================== */

const EMOJI = { angry: "😠", happy: "😊", neutral: "😐", sad: "😢" };
const MAX_HEARTS = 3;

// ---------------------------------------------------------------------------
// Backend location
// ---------------------------------------------------------------------------
// When the frontend is served by the FastAPI backend itself (same origin —
// e.g. the single-EC2 setup), relative "/api/..." paths just work and this
// stays "". When the frontend is hosted separately (e.g. GitHub Pages), a
// <meta name="moodmatch-api-base" content="https://your-backend-host"> tag
// in index.html supplies the backend's origin instead. Left blank, this file
// behaves exactly as it always has.
const API_BASE = (
  document.querySelector('meta[name="moodmatch-api-base"]')?.content || ""
).trim().replace(/\/$/, "");

// ---------------------------------------------------------------------------
// Floating emoji background (purely decorative, no game state involved)
// ---------------------------------------------------------------------------
// Weighted toward the four real emotions the model recognizes, mixed in with
// a few playful extras so the background doesn't feel like a plain repeat of
// the target-emotion icons.
const BG_EMOJI_POOL = [
  "😊", "😊", "😠", "😠", "😐", "😐", "😢", "😢",
  "🎭", "⭐", "❤️", "🎉", "🤔", "😆",
];

/** Fills #emoji-bg-layer with a moderate field of medium-size emoji that
 * drift around and bounce off the screen edges (like a DVD-logo
 * screensaver) behind the glass UI, across every screen (start/game/
 * gameover) since the layer is one fixed, full-viewport element sitting
 * behind all of them. `pointer-events:none` means it never blocks clicks.
 *
 * Positions are tracked in plain JS objects (not CSS custom properties)
 * because every frame needs fresh numbers to test against the current
 * viewport size — a CSS @keyframes path can't react to "did this emoji
 * just hit the right edge" the way a real bounce needs to. */
function initFloatingEmojiBackground() {
  const layer = document.getElementById("emoji-bg-layer");
  if (!layer) return;

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  // Moderate count — enough to feel like a field, not so many it competes
  // with the glass UI on top. Fewer on narrow screens.
  const count = window.innerWidth < 600 ? 14 : 24;

  const items = [];

  for (let i = 0; i < count; i++) {
    const span = document.createElement("span");
    span.className = "floating-emoji";
    span.setAttribute("aria-hidden", "true");
    span.textContent = BG_EMOJI_POOL[Math.floor(Math.random() * BG_EMOJI_POOL.length)];

    const sizeRem = 1.6 + Math.random() * 1.4; // ~1.6–3rem: medium, not tiny, not screen-filling
    const sizePx = sizeRem * 16;
    const opacity = 0.14 + Math.random() * 0.18;

    span.style.fontSize = `${sizeRem}rem`;
    layer.appendChild(span);

    const maxX = Math.max(0, window.innerWidth - sizePx);
    const maxY = Math.max(0, window.innerHeight - sizePx);
    const angle = Math.random() * Math.PI * 2;
    const speed = 0.25 + Math.random() * 0.45; // px/frame — slow, chill drift

    items.push({
      el: span,
      x: Math.random() * maxX,
      y: Math.random() * maxY,
      size: sizePx,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
      rot: Math.random() * 360,
      vrot: Math.random() * 0.5 - 0.25, // deg/frame — slow lazy spin
      opacity,
    });
  }

  // Paint the initial position once, then fade in (the CSS `transition:
  // opacity` on .floating-emoji handles the fade smoothly).
  for (const it of items) {
    it.el.style.transform = `translate3d(${it.x}px, ${it.y}px, 0) rotate(${it.rot}deg)`;
  }
  requestAnimationFrame(() => {
    for (const it of items) it.el.style.opacity = it.opacity.toFixed(2);
  });

  if (reduceMotion) return; // static field only — no animation loop, no bouncing

  function tick() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    for (const it of items) {
      it.x += it.vx;
      it.y += it.vy;
      it.rot += it.vrot;

      // Bounce off whichever edge was just crossed (DVD-screensaver style).
      if (it.x <= 0) {
        it.x = 0;
        it.vx = Math.abs(it.vx);
      } else if (it.x >= w - it.size) {
        it.x = w - it.size;
        it.vx = -Math.abs(it.vx);
      }
      if (it.y <= 0) {
        it.y = 0;
        it.vy = Math.abs(it.vy);
      } else if (it.y >= h - it.size) {
        it.y = h - it.size;
        it.vy = -Math.abs(it.vy);
      }

      it.el.style.transform = `translate3d(${it.x}px, ${it.y}px, 0) rotate(${it.rot}deg)`;
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);

  // Re-clamp positions on resize so nobody ends up stranded off-screen or
  // stuck bouncing outside the new bounds.
  window.addEventListener("resize", () => {
    const w = window.innerWidth;
    const h = window.innerHeight;
    for (const it of items) {
      it.x = Math.min(it.x, Math.max(0, w - it.size));
      it.y = Math.min(it.y, Math.max(0, h - it.size));
    }
  });
}

initFloatingEmojiBackground();

// ---------------------------------------------------------------------------
// Game state
// ---------------------------------------------------------------------------
const state = {
  score: 0,
  hearts: MAX_HEARTS,
  target: null,       // { emotion, emoji, image_url, instruction }
  stream: null,
  cameraEnabled: false,
  gameActive: false,   // true while the game screen is live (start..gameover)
  busy: false,         // true while a round is mid-countdown/analysis
};

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------
const screens = {
  start: document.getElementById("screen-start"),
  game: document.getElementById("screen-game"),
  gameover: document.getElementById("screen-gameover"),
};

const el = {
  btnStart: document.getElementById("btn-start"),
  startError: document.getElementById("start-error"),

  scorePill: document.getElementById("score-pill"),
  scoreValue: document.getElementById("score-value"),
  heartsRow: document.getElementById("hearts-row"),
  btnCameraToggle: document.getElementById("btn-camera-toggle"),

  targetImage: document.getElementById("target-image"),
  targetEmoji: document.getElementById("target-emoji"),
  targetName: document.getElementById("target-name"),
  targetInstruction: document.getElementById("target-instruction"),

  webcamWrap: document.getElementById("webcam-wrap"),
  video: document.getElementById("webcam-video"),
  canvas: document.getElementById("capture-canvas"),
  webcamOverlay: document.getElementById("webcam-overlay"),
  webcamStatus: document.getElementById("webcam-status"),
  countdown: document.getElementById("countdown"),
  resultFlash: document.getElementById("result-flash"),
  resultEmoji: document.getElementById("result-emoji"),
  resultTitle: document.getElementById("result-title"),
  resultDetail: document.getElementById("result-detail"),

  analysisStatus: document.getElementById("analysis-status"),
  btnReady: document.getElementById("btn-ready"),
  btnFlashContinue: document.getElementById("btn-flash-continue"),

  finalScoreValue: document.getElementById("final-score-value"),
  btnRetry: document.getElementById("btn-retry"),

  confettiLayer: document.getElementById("confetti-layer"),
};

// ---------------------------------------------------------------------------
// Screen switching
// ---------------------------------------------------------------------------
function showScreen(name) {
  Object.entries(screens).forEach(([key, node]) => {
    node.hidden = key !== name;
  });
}

// ---------------------------------------------------------------------------
// Webcam
// ---------------------------------------------------------------------------
function setCameraIcon(enabled) {
  if (enabled) {
    el.btnCameraToggle.textContent = "📷";
    el.btnCameraToggle.title = "Disable camera";
    el.btnCameraToggle.setAttribute("aria-label", "Disable camera");
    el.btnCameraToggle.classList.remove("camera-off");
  } else {
    el.btnCameraToggle.textContent = "🚫";
    el.btnCameraToggle.title = "Enable camera";
    el.btnCameraToggle.setAttribute("aria-label", "Enable camera");
    el.btnCameraToggle.classList.add("camera-off");
  }
}

async function enableCamera() {
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 480, height: 480, facingMode: "user" },
      audio: false,
    });
    el.video.srcObject = state.stream;
    state.cameraEnabled = true;
    el.webcamOverlay.classList.add("hidden-overlay");
    setCameraIcon(true);
    return true;
  } catch (err) {
    state.cameraEnabled = false;
    el.webcamOverlay.classList.remove("hidden-overlay");
    el.webcamStatus.textContent = "Camera access is required to play MoodMatch.AI.";
    setCameraIcon(false);
    return false;
  }
}

function disableCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
  }
  state.cameraEnabled = false;
  el.webcamOverlay.classList.remove("hidden-overlay");
  el.webcamStatus.textContent = "Camera is off.";
  setCameraIcon(false);
}

el.btnCameraToggle.addEventListener("click", async () => {
  if (state.cameraEnabled) {
    disableCamera();
    hideResultFlash();
    hideFrozenFrame();
    hideReadyState();
    setBusy(false);
    el.analysisStatus.textContent = "Camera is off. Turn it back on to keep playing.";
  } else {
    el.webcamStatus.textContent = "Requesting camera…";
    el.webcamOverlay.classList.remove("hidden-overlay");
    const ok = await enableCamera();
    if (ok && state.gameActive) {
      showReadyState();
    }
  }
});

function captureFrameAsBlob() {
  return new Promise((resolve) => {
    const { video, canvas } = el;
    canvas.width = video.videoWidth || 480;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    // Mirror the capture too, purely cosmetic — doesn't affect the model.
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    showFrozenFrame(); // freeze on the captured expression while the AI thinks
    canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.92);
  });
}

function showFrozenFrame() {
  el.canvas.hidden = false;
  el.canvas.classList.add("frozen-frame");
}

function hideFrozenFrame() {
  el.canvas.hidden = true;
  el.canvas.classList.remove("frozen-frame");
}

// ---------------------------------------------------------------------------
// Score / hearts UI
// ---------------------------------------------------------------------------
function updateScore(delta) {
  state.score += delta;
  el.scoreValue.textContent = state.score;
  el.scorePill.classList.remove("bump");
  void el.scorePill.offsetWidth; // restart animation
  el.scorePill.classList.add("bump");
}

function loseHeart() {
  state.hearts -= 1;
  const hearts = [...el.heartsRow.querySelectorAll(".heart")];
  hearts.forEach((h, i) => {
    h.classList.remove("just-lost");
    if (i >= state.hearts) {
      h.classList.add("lost");
    }
  });
  if (state.hearts >= 0 && hearts[state.hearts]) {
    hearts[state.hearts].classList.add("just-lost");
  }
}

function resetHeartsUI() {
  el.heartsRow.querySelectorAll(".heart").forEach((h) => {
    h.classList.remove("lost", "just-lost");
  });
}

// ---------------------------------------------------------------------------
// Confetti (lightweight, dependency-free)
// ---------------------------------------------------------------------------
function launchConfetti() {
  const colors = ["#14e37b", "#ff2ea6", "#7bffb8", "#ff8bcb"];
  for (let i = 0; i < 36; i++) {
    const piece = document.createElement("div");
    piece.className = "confetti-piece";
    piece.style.left = `${Math.random() * 100}vw`;
    piece.style.background = colors[Math.floor(Math.random() * colors.length)];
    piece.style.animationDuration = `${1.2 + Math.random() * 1.2}s`;
    piece.style.transform = `rotate(${Math.random() * 360}deg)`;
    el.confettiLayer.appendChild(piece);
    setTimeout(() => piece.remove(), 2600);
  }
}

// ---------------------------------------------------------------------------
// Round flow (gated on the player clicking "I'M READY")
// ---------------------------------------------------------------------------
async function loadNewTarget() {
  el.targetInstruction.textContent = "Loading target…";
  try {
    const res = await fetch(`${API_BASE}/api/random-target`);
    const data = await res.json();
    state.target = data;
    // image_url comes back as a root-relative path (e.g. "/dataset-images/...")
    // meant to be resolved against the backend's own origin.
    el.targetImage.src = data.image_url ? `${API_BASE}${data.image_url}` : "";
    el.targetEmoji.textContent = data.emoji || EMOJI[data.emotion] || "";
    el.targetName.textContent = data.emotion.toUpperCase();
    el.targetInstruction.textContent = data.instruction;
  } catch (err) {
    el.targetInstruction.textContent = "Couldn't load a target. Retrying…";
    setTimeout(loadNewTarget, 1500);
  }
}

function setBusy(isBusy) {
  state.busy = isBusy;
}

// Holds the "what happens next" callback for the currently-shown result
// flash, when it needs a deliberate click (wrong guess / no face / error)
// rather than auto-continuing on its own (a correct match).
let pendingContinue = null;

/** Shows the result flash. Pass `continueLabel` + `onContinue` to make the
 * player click a button (e.g. "TRY AGAIN") before the round moves on;
 * omit both to auto-continue elsewhere (used for the success flash). */
function showResultFlash(kind, emoji, title, detail, continueLabel, onContinue) {
  el.resultFlash.className = `result-flash ${kind}`;
  el.resultEmoji.textContent = emoji;
  el.resultTitle.textContent = title;
  el.resultDetail.textContent = detail || "";

  if (onContinue) {
    el.btnFlashContinue.textContent = continueLabel || "TRY AGAIN";
    el.btnFlashContinue.hidden = false;
    pendingContinue = onContinue;
  } else {
    el.btnFlashContinue.hidden = true;
    pendingContinue = null;
  }

  el.resultFlash.hidden = false;
}

function hideResultFlash() {
  el.resultFlash.hidden = true;
  el.btnFlashContinue.hidden = true;
  pendingContinue = null;
}

el.btnFlashContinue.addEventListener("click", () => {
  const onContinue = pendingContinue;
  pendingContinue = null;
  hideResultFlash();
  hideFrozenFrame();
  if (onContinue) onContinue();
});

/** Shows the "I'M READY" button so the player can trigger the next attempt
 * whenever they choose. Only meaningful while the camera is on and the
 * game is active. */
function showReadyState() {
  if (!state.gameActive || !state.cameraEnabled) return;
  el.btnReady.hidden = false;
  el.btnReady.disabled = false;
}

function hideReadyState() {
  el.btnReady.hidden = true;
}

/** Runs the 3-2-1-MATCH! countdown. Returns false early if the camera gets
 * disabled or the game ends mid-countdown, so the caller can bail out. */
async function runCountdown() {
  const steps = ["3", "2", "1", "MATCH!"];
  for (const step of steps) {
    if (!state.cameraEnabled || !state.gameActive) {
      el.countdown.hidden = true;
      return false;
    }
    el.countdown.hidden = false;
    el.countdown.textContent = step;
    el.countdown.classList.remove("pulse");
    void el.countdown.offsetWidth;
    el.countdown.classList.add("pulse");
    await new Promise((r) => setTimeout(r, 550));
  }
  el.countdown.hidden = true;
  return true;
}

async function runRoundAttempt() {
  if (!state.gameActive || !state.cameraEnabled || state.busy) return;

  hideReadyState();
  setBusy(true);
  hideResultFlash();
  hideFrozenFrame();
  el.analysisStatus.textContent = "Get ready…";

  const completed = await runCountdown();
  if (!completed) {
    // Camera was turned off (or game ended) mid-countdown.
    setBusy(false);
    showReadyState();
    return;
  }

  el.analysisStatus.textContent = "AI is analyzing your expression…";
  const blob = await captureFrameAsBlob();

  const formData = new FormData();
  formData.append("image", blob, "capture.jpg");
  // Sent only so the backend can tag debug capture filenames with what the
  // player was supposed to show (e.g. "..._angry_raw.jpg") — it plays no
  // part in the actual prediction.
  formData.append("target_emotion", state.target?.emotion || "");

  let data;
  try {
    const res = await fetch(`${API_BASE}/api/predict`, { method: "POST", body: formData });
    data = await res.json();
  } catch (err) {
    data = { status: "error", message: "network error" };
  }

  handlePredictionResult(data);
}

el.btnReady.addEventListener("click", runRoundAttempt);

/** Shared "reposition and click Try Again" flow used by the no_face,
 * multiple_faces and error cases — keeps the same target, just lets the
 * player retry once they've read the message and are ready. */
function retrySameTarget() {
  setBusy(false);
  el.analysisStatus.textContent = "Ready when you are.";
  showReadyState();
}

function handlePredictionResult(data) {
  if (data.status === "no_face") {
    el.analysisStatus.textContent = "We couldn't see your face.";
    showResultFlash(
      "info",
      "🔍",
      "No face detected",
      "We can't see your face! Move into the camera frame.",
      "TRY AGAIN",
      retrySameTarget
    );
    return;
  }

  if (data.status === "multiple_faces") {
    el.analysisStatus.textContent = "Too many faces in frame.";
    showResultFlash(
      "info",
      "👥",
      "Too many faces",
      "Please make sure only one face is visible.",
      "TRY AGAIN",
      retrySameTarget
    );
    return;
  }

  if (data.status === "error" || data.status !== "ok") {
    el.analysisStatus.textContent = "Something went wrong.";
    showResultFlash(
      "info",
      "⚠️",
      "Prediction failed",
      "The AI couldn't read your expression. Try again.",
      "TRY AGAIN",
      retrySameTarget
    );
    return;
  }

  const predicted = data.predicted_emotion;
  const target = state.target.emotion;
  const isMatch = predicted === target;

  if (isMatch) {
    updateScore(1);
    launchConfetti();
    showResultFlash("success", "🎉", "PERFECT MATCH! +1", `The AI saw: ${capitalize(predicted)}`);
    el.analysisStatus.textContent = "Nailed it! Loading next target…";
    setTimeout(async () => {
      hideResultFlash();
      hideFrozenFrame();
      await loadNewTarget();
      setBusy(false);
      el.analysisStatus.textContent = "Ready when you are.";
      showReadyState();
    }, 1600);
  } else {
    loseHeart();
    el.analysisStatus.textContent = "That wasn't a match.";
    const outOfHearts = state.hearts <= 0;
    showResultFlash(
      "fail",
      "❌",
      "NOT A MATCH",
      `You showed: ${capitalize(predicted)}  •  Target: ${capitalize(target)}`,
      outOfHearts ? "SEE RESULTS" : "TRY AGAIN",
      async () => {
        if (outOfHearts) {
          setBusy(false);
          endGame();
          return;
        }
        await loadNewTarget();
        setBusy(false);
        el.analysisStatus.textContent = "Ready when you are.";
        showReadyState();
      }
    );
  }
}

function capitalize(str) {
  if (!str) return "";
  return str.charAt(0).toUpperCase() + str.slice(1);
}

// ---------------------------------------------------------------------------
// Game lifecycle
// ---------------------------------------------------------------------------
async function startGame() {
  el.btnStart.disabled = true;
  el.startError.hidden = true;
  hideReadyState();

  const ok = await enableCamera();
  if (!ok) {
    el.startError.textContent = "Camera access is required to play MoodMatch.AI.";
    el.startError.hidden = false;
    el.btnStart.disabled = false;
    return;
  }

  state.score = 0;
  state.hearts = MAX_HEARTS;
  state.gameActive = true;
  el.scoreValue.textContent = "0";
  resetHeartsUI();

  showScreen("game");
  hideResultFlash();
  hideFrozenFrame();
  el.analysisStatus.textContent = "Ready when you are.";
  setBusy(false);
  await loadNewTarget();

  el.btnStart.disabled = false;
  showReadyState();
}

function endGame() {
  state.gameActive = false;
  hideReadyState();
  disableCamera();
  el.finalScoreValue.textContent = state.score;
  showScreen("gameover");
}

el.btnStart.addEventListener("click", startGame);
el.btnRetry.addEventListener("click", startGame);

// Initial screen.
showScreen("start");
