/* NoCursor collector page logic: consent gate, Fitts task game,
   free-draw task, chunked anonymous upload. No data leaves the page
   before explicit consent. */

(function () {
  "use strict";

  // debug/E2E hook: collector state introspection (no data exposure)
  window.__collector = {
    get phase() { return phase; },
    get waitingUp() { return waitingUp; },
    get trial() { return trial; },
    get buffered() { return buf.length; },
    get recording() { return recording; },
  };

  var N_TRIALS = 24;
  var FREE_DRAW_MS = 20000;
  var CHUNK_INTERVAL_MS = 3000;
  var CHUNK_SIZE = 2000;

  var consentEl = document.getElementById("consent");
  var startBtn = document.getElementById("start");
  var statusEl = document.getElementById("status");
  var introEl = document.getElementById("intro");
  var stageEl = document.getElementById("stage");
  var targetEl = document.getElementById("target");
  var trialNumEl = document.getElementById("trial-num");

  var recording = false;
  var buf = [];
  var session = null;
  var t0 = 0;
  var trial = 0;
  var phase = "idle"; // 'fitts' | 'free'
  var chunkTimer = null;
  var freeDrawEnd = 0;
  var waitingUp = false;

  consentEl.addEventListener("change", function () {
    startBtn.disabled = !consentEl.checked;
  });

  startBtn.addEventListener("click", function () {
    if (!consentEl.checked) return;
    startBtn.disabled = true;
    statusEl.textContent = "Starting session...";
    fetch("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        consent: true,
        screen_w: screen.width,
        screen_h: screen.height,
        dpr: window.devicePixelRatio || 1,
        tasks: ["fitts", "free"],
      }),
    })
      .then(function (r) {
        if (!r.ok) throw new Error("session create failed: " + r.status);
        return r.json();
      })
      .then(function (data) {
        session = data.id;
        beginTasks();
      })
      .catch(function (e) {
        statusEl.textContent = "Could not start: " + e.message;
        startBtn.disabled = false;
      });
  });

  function beginTasks() {
    introEl.style.display = "none";
    stageEl.style.display = "block";
    recording = true;
    t0 = performance.now();
    chunkTimer = setInterval(maybeSendChunk, CHUNK_INTERVAL_MS);
    nextTrial();
  }

  function now() {
    return performance.now() - t0;
  }

  function push(type, x, y) {
    buf.push({ t: now(), x: x, y: y, type: type });
  }

  // capture: pointer events, mouse only, coalesced for high fidelity
  document.addEventListener("pointermove", function (e) {
    if (!recording || e.pointerType !== "mouse") return;
    var events = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
    for (var i = 0; i < events.length; i++) {
      push("move", events[i].clientX, events[i].clientY);
    }
  });
  document.addEventListener("pointerdown", function (e) {
    if (!recording || e.pointerType !== "mouse") return;
    push("down", e.clientX, e.clientY);
  });
  document.addEventListener("pointerup", function (e) {
    if (!recording || e.pointerType !== "mouse") return;
    push("up", e.clientX, e.clientY);
  });

  function nextTrial() {
    trial += 1;
    trialNumEl.textContent = String(trial);
    waitingUp = true;
    phase = "fitts";

    // Fitts-style target: random radius, random distance/angle from cursor
    var w = window.innerWidth;
    var h = window.innerHeight;
    var margin = 60;
    var r = 12 + Math.random() * 28;
    var dist = 150 + Math.random() * Math.min(750, Math.min(w, h) * 0.7);
    var angle = Math.random() * 2 * Math.PI;
    var last = lastPos();
    var tx = Math.min(Math.max(last.x + dist * Math.cos(angle), r + margin), w - r - margin);
    var ty = Math.min(Math.max(last.y + dist * Math.sin(angle), r + margin), h - r - margin);
    targetEl.style.left = tx + "px";
    targetEl.style.top = ty + "px";
    targetEl.style.width = 2 * r + "px";
    targetEl.style.height = 2 * r + "px";
    targetEl.style.display = "block";
    targetRect = { x: tx, y: ty, r: r };
  }

  var targetRect = null;

  function lastPos() {
    for (var i = buf.length - 1; i >= 0; i--) {
      if (buf[i].type === "move") return { x: buf[i].x, y: buf[i].y };
    }
    return { x: window.innerWidth / 2, y: window.innerHeight / 2 };
  }

  document.addEventListener("pointerdown", function (e) {
    if (!recording || phase !== "fitts" || !waitingUp) return;
    if (!targetRect) return;
    var dx = e.clientX - targetRect.x;
    var dy = e.clientY - targetRect.y;
    if (Math.hypot(dx, dy) <= targetRect.r) {
      waitingUp = false;
      if (trial >= N_TRIALS) {
        startFreeDraw();
      } else {
        setTimeout(nextTrial, 400);
      }
    }
  });

  function startFreeDraw() {
    phase = "free";
    targetEl.style.display = "none";
    document.getElementById("hud").textContent =
      "Move naturally (draw shapes, wander) for 20 seconds";
    freeDrawEnd = performance.now() + FREE_DRAW_MS;
    setTimeout(finish, FREE_DRAW_MS + 500);
  }

  function maybeSendChunk() {
    if (!session || buf.length < CHUNK_SIZE) return;
    sendChunk();
  }

  function sendChunk(final) {
    if (!session || buf.length === 0) {
      if (final) finishDone();
      return;
    }
    var payload = buf.splice(0, buf.length);
    return fetch("/api/sessions/" + session + "/chunks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ samples: payload }),
    })
      .then(function (r) {
        if (!r.ok) throw new Error("upload failed: " + r.status);
        if (final) finishDone();
      })
      .catch(function (e) {
        statusEl.textContent = "Upload failed: " + e.message;
      });
  }

  function finish() {
    recording = false;
    if (chunkTimer) clearInterval(chunkTimer);
    stageEl.style.display = "none";
    introEl.style.display = "block";
    statusEl.textContent = "Uploading...";
    sendChunk(true);
  }

  function finishDone() {
    statusEl.textContent =
      "Thank you! Session " + session.slice(0, 8) + " recorded anonymously.";
    session = null;
    startBtn.disabled = true;
    consentEl.checked = false;
  }
})();
