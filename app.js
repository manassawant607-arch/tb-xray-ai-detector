// TB X-Ray AI Detector — browser-based inference with ONNX Runtime Web
// Both the X-ray validity gate and the TB detection CNN run entirely client-side.

const TB_MODEL_URL = "models/tb_model.onnx";
const GATE_MODEL_URL = "models/xray_gate.onnx";
const IMG_SIZE = 224;
const THRESHOLD = 0.5;

// Heuristic pre-filter thresholds (mirror xray_validator.py)
const MIN_DIM = 100, MAX_ASPECT = 2.5, MIN_BRIGHTNESS = 0.04;
const MAX_BRIGHTNESS = 0.985, MIN_STD = 0.015, MAX_SATURATION = 0.6;

const TB_POSITIVE = {
  tb: "TB Detected", mutation: "rpoB mutation detected",
  resistance: "Rifampicin Resistant (Possible MDR-TB)",
  treatment: "Bedaquiline + Linezolid + Levofloxacin",
};
const TB_NEGATIVE = {
  tb: "Normal", mutation: "No mutation detected",
  resistance: "Drug Sensitive", treatment: "Standard TB therapy",
};

let tbSession = null, gateSession = null;
let modelsLoading = null;

async function loadModels(onProgress) {
  if (tbSession && gateSession) return;
  if (modelsLoading) return modelsLoading;

  modelsLoading = (async () => {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";
    // GitHub Pages (and most static hosts) don't set COOP/COEP headers, so
    // SharedArrayBuffer is unavailable → force single-threaded WASM backend.
    ort.env.wasm.numThreads = 1;
    onProgress(5);
    const gateResp = await fetch(GATE_MODEL_URL);
    const gateBuf = await gateResp.arrayBuffer();
    gateSession = await ort.InferenceSession.create(gateBuf);
    onProgress(20);

    const tbResp = await fetch(TB_MODEL_URL);
    const reader = tbResp.body.getReader();
    const contentLength = +tbResp.headers.get("Content-Length");
    let received = 0;
    const chunks = [];
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      if (contentLength) onProgress(20 + Math.round((received / contentLength) * 70));
    }
    const tbBuf = new Blob(chunks).arrayBuffer
      ? await new Blob(chunks).arrayBuffer()
      : await Promise.all(chunks.map(c => c.arrayBuffer || c)).then(buffers => {
          const total = buffers.reduce((s, b) => s + b.byteLength, 0);
          const merged = new Uint8Array(total);
          let off = 0;
          for (const b of buffers) { merged.set(new Uint8Array(b), off); off += b.byteLength; }
          return merged.buffer;
        });
    tbSession = await ort.InferenceSession.create(tbBuf);
    onProgress(100);
  })();

  return modelsLoading;
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

function preprocess(imgElement) {
  const canvas = document.createElement("canvas");
  canvas.width = IMG_SIZE;
  canvas.height = IMG_SIZE;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(imgElement, 0, 0, IMG_SIZE, IMG_SIZE);
  const imageData = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
  const data = imageData.data;
  const float32 = new Float32Array(IMG_SIZE * IMG_SIZE * 3);
  for (let i = 0; i < data.length / 4; i++) {
    float32[i * 3] = data[i * 4] / 255.0;
    float32[i * 3 + 1] = data[i * 4 + 1] / 255.0;
    float32[i * 3 + 2] = data[i * 4 + 2] / 255.0;
  }
  return float32;
}

function getImageStats(imgElement) {
  const canvas = document.createElement("canvas");
  canvas.width = imgElement.naturalWidth || imgElement.width;
  canvas.height = imgElement.naturalHeight || imgElement.height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(imgElement, 0, 0);
  const w = canvas.width, h = canvas.height;
  const imageData = ctx.getImageData(0, 0, w, h);
  const data = imageData.data;
  const n = w * h;

  let sumGray = 0, sumGraySq = 0;
  let sumSat = 0, darkCount = 0, brightCount = 0;
  let midCount = 0, sumVar = 0, sumRgbVar = 0;
  const grayVals = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    const r = data[i * 4] / 255, g = data[i * 4 + 1] / 255, b = data[i * 4 + 2] / 255;
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    const gray = (r + g + b) / 3;
    grayVals[i] = gray;
    sumGray += gray; sumGraySq += gray * gray;
    sumSat += mx > 1e-6 ? (mx - mn) / mx : 0;
    if (gray < 0.05) darkCount++;
    if (gray > 0.95) brightCount++;
    if (gray > 0.2 && gray < 0.8) midCount++;
    sumVar += (mx - mn);
    const meanRgb = (r + g + b) / 3;
    sumRgbVar += (r * r + g * g + b * b) / 3 - meanRgb * meanRgb;
  }

  const mean = sumGray / n;
  const std = Math.sqrt(sumGraySq / n - mean * mean);
  const meanSat = sumSat / n;
  const meanVar = sumVar / n;
  const meanRgbVar = sumRgbVar / n;

  const sorted = Float32Array.from(grayVals);
  sorted.sort((a, b) => a - b);
  const pct = p => sorted[Math.floor((p / 100) * (n - 1))];

  const hist = new Float32Array(32);
  for (let i = 0; i < n; i++) {
    const bin = Math.min(31, Math.floor(grayVals[i] * 32));
    hist[bin]++;
  }
  let maxHist = 0;
  for (let i = 0; i < 32; i++) { hist[i] /= n; if (hist[i] > maxHist) maxHist = hist[i]; }

  let edgeSumH = 0, edgeSumV = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w - 1; x++) {
      edgeSumH += Math.abs(grayVals[y * w + x] - grayVals[y * w + x + 1]);
    }
  }
  for (let y = 0; y < h - 1; y++) {
    for (let x = 0; x < w; x++) {
      edgeSumV += Math.abs(grayVals[y * w + x] - grayVals[(y + 1) * w + x]);
    }
  }
  const edgeDensity = edgeSumV / ((h - 1) * w) + edgeSumH / (h * (w - 1));

  const ch = Math.max(Math.floor(h / 6), 1), cw = Math.max(Math.floor(w / 6), 1);
  let cornerSum = 0, cornerCount = 0;
  const corners = [[0,0],[0,w-cw],[h-ch,0],[h-ch,w-cw]];
  for (const [cy, cx] of corners) {
    for (let y = cy; y < cy + ch; y++) {
      for (let x = cx; x < cx + cw; x++) {
        cornerSum += grayVals[y * w + x]; cornerCount++;
      }
    }
  }
  const cornerMean = cornerSum / cornerCount;

  return new Float32Array([
    meanSat, meanVar, darkCount / n, brightCount / n, maxHist,
    mean, std, edgeDensity, pct(10), pct(90), pct(25), pct(75),
    midCount / n, meanRgbVar, cornerMean,
  ]);
}

async function isChestXray(imgElement) {
  const w = imgElement.naturalWidth || imgElement.width;
  const h = imgElement.naturalHeight || imgElement.height;
  if (w < MIN_DIM || h < MIN_DIM)
    return { ok: false, reason: "rejected: image too small to be an X-ray" };
  const aspect = Math.max(w, h) / Math.min(w, h);
  if (aspect > MAX_ASPECT)
    return { ok: false, reason: "rejected: aspect ratio too extreme for a chest X-ray" };

  const stats = getImageStats(imgElement);
  const mean = stats[5], std = stats[6], sat = stats[0];
  if (mean < MIN_BRIGHTNESS)
    return { ok: false, reason: "rejected: image is too dark / near-black" };
  if (mean > MAX_BRIGHTNESS)
    return { ok: false, reason: "rejected: image is too bright / near-blank" };
  if (std < MIN_STD)
    return { ok: false, reason: "rejected: image is flat (no texture) — not an X-ray" };
  if (sat > MAX_SATURATION)
    return { ok: false, reason: "rejected: image is too colourful to be a chest X-ray" };

  const input = new ort.Tensor("float32", stats, [1, 15]);
  const output = await gateSession.run({ input });
  // TreeEnsembleClassifier outputs: "label" (int64) + "probabilities" (float [N,C])
  // classes are [0, 1] → proba[1] = P(valid X-ray)
  const probaData = output.probabilities.data;
  const pValid = probaData[1];
  if (typeof window._debug !== "undefined") window._debug.gatePValid = pValid;
  if (pValid < 0.5)
    return { ok: false, reason: "rejected: not a chest X-ray (looks like a photo, screenshot, or other non-X-ray image)" };
  return { ok: true, reason: "valid chest X-ray" };
}

async function predictTB(imgElement) {
  const float32 = preprocess(imgElement);
  const input = new ort.Tensor("float32", float32, [1, 224, 224, 3]);
  const output = await tbSession.run({ input });
  const pred = output[Object.keys(output)[0]].data[0];
  let result;
  if (pred > THRESHOLD) {
    result = { ...TB_POSITIVE, confidence: (pred * 100).toFixed(1) + "%" };
  } else {
    result = { ...TB_NEGATIVE, confidence: ((1 - pred) * 100).toFixed(1) + "%" };
  }
  return result;
}

// ===== UI wiring =====
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const dropText = document.getElementById("dropText");
const submitBtn = document.getElementById("submitBtn");
const statusEl = document.getElementById("status");
const loadBar = document.getElementById("loadBar");
const loadFill = document.getElementById("loadFill");
let selectedFile = null;

dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", e => handleFile(e.target.files[0]));
dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("dragover"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", e => {
  e.preventDefault(); dropZone.classList.remove("dragover");
  handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
  if (!file) return;
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => { preview.src = e.target.result; preview.style.display = "block"; dropText.style.display = "none"; };
  reader.readAsDataURL(file);
  submitBtn.disabled = false;
}

document.querySelectorAll(".example-btn").forEach(btn => {
  btn.addEventListener("click", async () => {
    const type = btn.dataset.example;
    const urlMap = {
      tb: "samples/sample_tb_xray.png",
      normal: "samples/sample_normal_xray.png",
      wrong: "samples/test_wrong_photo.png",
    };
    const url = urlMap[type];
    statusEl.textContent = "Loading sample...";
    try {
      const resp = await fetch(url);
      const blob = await resp.blob();
      handleFile(new File([blob], url.split("/").pop(), { type: "image/png" }));
      statusEl.textContent = "";
    } catch (err) { statusEl.textContent = "Could not load sample."; }
  });
});

submitBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  submitBtn.disabled = true;
  const originalText = submitBtn.innerHTML;
  submitBtn.innerHTML = '<span class="spinner"></span> Loading models...';
  statusEl.textContent = "Loading AI models (first load downloads ~43 MB)...";
  loadBar.classList.add("show");

  try {
    await loadModels(pct => { loadFill.style.width = pct + "%"; if (pct < 100) statusEl.textContent = `Loading AI models... ${pct}%`; });
    loadFill.style.width = "100%";
    statusEl.textContent = "Running X-ray validity gate + detection model...";
    submitBtn.innerHTML = '<span class="spinner"></span> Analyzing...';

    ["tbResult","mutationResult","resistanceResult","treatmentResult","confidenceResult"]
      .forEach(id => { const el = document.getElementById(id); el.textContent = "..."; el.className = "result-value"; });

    const img = await loadImage(preview.src);

    // Stage 1: gate
    const gate = await isChestXray(img);
    if (!gate.ok) {
      document.getElementById("tbResult").textContent = gate.reason;
      document.getElementById("tbResult").className = "result-value rejected";
      ["mutationResult","resistanceResult","treatmentResult","confidenceResult"]
        .forEach(id => document.getElementById(id).textContent = "—");
    } else {
      // Stage 2: detection
      const result = await predictTB(img);
      document.getElementById("tbResult").textContent = result.tb;
      document.getElementById("mutationResult").textContent = result.mutation;
      document.getElementById("resistanceResult").textContent = result.resistance;
      document.getElementById("treatmentResult").textContent = result.treatment;
      document.getElementById("confidenceResult").textContent = result.confidence;
      const tbEl = document.getElementById("tbResult");
      tbEl.className = "result-value " + (result.tb === "TB Detected" ? "tb" : "normal");
    }
    statusEl.textContent = "Done.";
  } catch (err) {
    console.error(err);
    document.getElementById("tbResult").textContent = "Error: " + err.message;
    document.getElementById("tbResult").className = "result-value rejected";
    statusEl.textContent = "Error: " + err.message;
    statusEl.style.color = "var(--red)";
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = originalText;
    setTimeout(() => { loadBar.classList.remove("show"); loadFill.style.width = "0%"; }, 1000);
  }
});
