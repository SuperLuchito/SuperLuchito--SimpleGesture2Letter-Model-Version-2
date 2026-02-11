const videoEl = document.getElementById('video');
const canvasEl = document.getElementById('overlay');
const ctx = canvasEl.getContext('2d');

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');

const statusEl = document.getElementById('status');
const letterEl = document.getElementById('letter');
const scoreEl = document.getElementById('score');
const confidenceEl = document.getElementById('confidence');
const holdEl = document.getElementById('hold');
const remainEl = document.getElementById('remain');
const progressBar = document.getElementById('progressBar');
const textValueEl = document.getElementById('textValue');
const topkEl = document.getElementById('topk');
const debugEl = document.getElementById('debug');

const vlmUsedEl = document.getElementById('vlmUsed');
const vlmLetterEl = document.getElementById('vlmLetter');
const vlmConfEl = document.getElementById('vlmConf');
const vlmReasonEl = document.getElementById('vlmReason');

// Важно: датасет снят с зеркалом, поэтому live-кадр в распознавание тоже зеркалим.
const MIRROR_STREAM = true;

let stream = null;
let ws = null;
let sendTimer = null;
let renderReq = null;
let sendBusy = false;
let latest = null;
let sendFps = 12;
let jpegQuality = 0.75;
let shouldRun = false;
let reconnectTimer = null;
let lastStateAtMs = 0;
let awaitingServer = false;
let lastSendAtMs = 0;
let stableVisibleLetter = 'NONE';

const captureCanvas = document.createElement('canvas');
const captureCtx = captureCanvas.getContext('2d');

function drawVideoFrame(targetCtx, sourceVideo, width, height, mirror = false) {
  targetCtx.save();
  if (mirror) {
    targetCtx.translate(width, 0);
    targetCtx.scale(-1, 1);
  }
  targetCtx.drawImage(sourceVideo, 0, 0, width, height);
  targetCtx.restore();
}

function applyMirrorStyles() {
  const transform = MIRROR_STREAM ? 'scaleX(-1)' : 'none';
  videoEl.style.transform = transform;
  canvasEl.style.transform = transform;
  videoEl.style.transformOrigin = 'center center';
  canvasEl.style.transformOrigin = 'center center';
}

function wsUrl() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${location.host}/ws/stream`;
}

function setStatusClass(status) {
  statusEl.className = '';
  const normalized = String(status || '').toLowerCase();
  statusEl.classList.add(`status-${normalized}`);
}

function pickVisibleLetter(data) {
  const rawLetter = (typeof data.letter === 'string' && data.letter.length > 0) ? data.letter : 'NONE';
  if (!data.hand_present) {
    stableVisibleLetter = 'NONE';
    return 'NONE';
  }
  if (rawLetter !== 'NONE') {
    stableVisibleLetter = rawLetter;
    return rawLetter;
  }
  return stableVisibleLetter;
}

async function fetchServerConfig() {
  try {
    const res = await fetch('/health');
    const data = await res.json();
    if (data.config) {
      sendFps = Number(data.config.frontend_fps || 12);
      jpegQuality = Number(data.config.jpeg_quality || 0.75);
    }
  } catch (err) {
    console.warn('health config unavailable:', err);
  }
}

async function startCamera() {
  if (stream) return;
  shouldRun = true;
  await fetchServerConfig();

  stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { width: { ideal: 960 }, height: { ideal: 720 }, facingMode: 'user' },
  });

  videoEl.srcObject = stream;
  await videoEl.play();
  applyMirrorStyles();

  canvasEl.width = videoEl.videoWidth;
  canvasEl.height = videoEl.videoHeight;
  captureCanvas.width = videoEl.videoWidth;
  captureCanvas.height = videoEl.videoHeight;

  connectWs();
  startSender();
  renderLoop();

  startBtn.disabled = true;
  stopBtn.disabled = false;
}

function stopCamera() {
  shouldRun = false;

  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }

  if (sendTimer) {
    clearInterval(sendTimer);
    sendTimer = null;
  }

  if (renderReq) {
    cancelAnimationFrame(renderReq);
    renderReq = null;
  }

  if (ws) {
    ws.close();
    ws = null;
  }

  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }

  startBtn.disabled = false;
  stopBtn.disabled = true;

  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  latest = null;
  lastStateAtMs = 0;
  awaitingServer = false;
  lastSendAtMs = 0;
  stableVisibleLetter = 'NONE';
  statusEl.textContent = 'NONE';
  letterEl.textContent = 'NONE';
  textValueEl.textContent = 'NONE';
}

function connectWs() {
  if (!shouldRun) return;
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;

  ws = new WebSocket(wsUrl());
  ws.binaryType = 'arraybuffer';

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === 'ack') return;
      latest = data;
      lastStateAtMs = Date.now();
      awaitingServer = false;
      renderState(data);
    } catch (err) {
      console.warn('bad ws message', err);
    }
  };

  ws.onerror = () => {
    if (ws) ws.close();
  };

  ws.onclose = () => {
    awaitingServer = false;
    ws = null;
    if (shouldRun) {
      reconnectTimer = setTimeout(() => {
        connectWs();
      }, 800);
    }
  };
}

function startSender() {
  const interval = Math.max(50, Math.round(1000 / sendFps));
  sendTimer = setInterval(() => {
    if (!stream || !ws || ws.readyState !== WebSocket.OPEN || sendBusy) return;
    if (!videoEl.videoWidth || !videoEl.videoHeight) return;
    if (awaitingServer) return;

    sendBusy = true;
    drawVideoFrame(captureCtx, videoEl, captureCanvas.width, captureCanvas.height, MIRROR_STREAM);
    captureCanvas.toBlob((blob) => {
      if (!blob || !ws || ws.readyState !== WebSocket.OPEN) {
        sendBusy = false;
        return;
      }
      blob.arrayBuffer().then((arr) => {
        ws.send(arr);
        awaitingServer = true;
        lastSendAtMs = Date.now();
        sendBusy = false;
      }).catch(() => {
        sendBusy = false;
      });
    }, 'image/jpeg', jpegQuality);
  }, interval);
}

function renderLoop() {
  if (!stream) return;

  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

  const stateFresh = Date.now() - lastStateAtMs < 800;
  if (stateFresh && latest && latest.hand_present && latest.bbox_norm) {
    const [x1, y1, x2, y2] = latest.bbox_norm;
    const px1 = x1 * canvasEl.width;
    const py1 = y1 * canvasEl.height;
    const px2 = x2 * canvasEl.width;
    const py2 = y2 * canvasEl.height;

    ctx.strokeStyle = '#37f59a';
    ctx.lineWidth = 3;
    ctx.strokeRect(px1, py1, px2 - px1, py2 - py1);
  }

  renderReq = requestAnimationFrame(renderLoop);
}

function renderState(data) {
  setStatusClass(data.status);
  statusEl.textContent = data.status;
  const visibleLetter = pickVisibleLetter(data);
  letterEl.textContent = visibleLetter;
  scoreEl.textContent = Number(data.score || 0).toFixed(3);
  confidenceEl.textContent = Number(data.confidence || 0).toFixed(3);

  const hold = data.hold || {};
  holdEl.textContent = `${hold.elapsed_ms || 0} / ${hold.target_ms || 0} мс`;
  remainEl.textContent = `${hold.remaining_ms || 0} мс`;
  progressBar.style.width = `${Math.max(0, Math.min(100, (hold.progress || 0) * 100))}%`;

  textValueEl.textContent = visibleLetter;

  const topk = Array.isArray(data.topk) ? data.topk : [];
  topkEl.innerHTML = '';
  for (const item of topk) {
    const li = document.createElement('li');
    li.textContent = `${item.letter}: ${Number(item.score || 0).toFixed(3)}`;
    topkEl.appendChild(li);
  }

  const dbg = data.debug || {};
  debugEl.textContent = `sim1=${Number(dbg.sim1 || 0).toFixed(3)} | sim2=${Number(dbg.sim2 || 0).toFixed(3)} | margin=${Number(dbg.margin || 0).toFixed(3)} | uncertain=${Boolean(dbg.uncertain)} | cooldown=${dbg.cooldown_left_ms || 0}мс`;

  const vlm = data.vlm || {};
  vlmUsedEl.textContent = String(Boolean(vlm.used));
  vlmLetterEl.textContent = vlm.letter || 'NONE';
  vlmConfEl.textContent = Number(vlm.confidence || 0).toFixed(3);
  vlmReasonEl.textContent = vlm.reason || '';
}

startBtn.addEventListener('click', async () => {
  try {
    await startCamera();
  } catch (err) {
    alert(`Ошибка запуска камеры: ${err.message || err}`);
  }
});

stopBtn.addEventListener('click', () => {
  stopCamera();
});

window.addEventListener('beforeunload', () => {
  stopCamera();
});
