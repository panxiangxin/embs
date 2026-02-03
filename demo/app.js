const defaultAssets = [
  {
    id: "veh-01",
    name: "蓝色轿车 / 破损",
    labels: ["破损", "蓝色", "汽车", "轿车", "车辆"],
    location: [12.4, 0.0, -3.2],
    rotation: [0, 92, 0],
    bbox: [2.1, 1.4, 4.3],
    coverPoints: 2,
  },
  {
    id: "veh-02",
    name: "红色卡车 / 货运",
    labels: ["红色", "卡车", "货运", "车辆", "重型"],
    location: [-6.2, 0.0, 8.9],
    rotation: [0, 12, 0],
    bbox: [2.6, 2.4, 6.1],
    coverPoints: 4,
  },
  {
    id: "obj-01",
    name: "钢制大门 / 机库",
    labels: ["钢制", "大门", "机库", "金属", "入口"],
    location: [1.2, 0.0, 14.1],
    rotation: [0, 180, 0],
    bbox: [3.0, 4.0, 0.4],
    coverPoints: 1,
  },
  {
    id: "str-01",
    name: "瞭望塔 / 木制",
    labels: ["瞭望塔", "木制", "平台", "建筑", "高处"],
    location: [-14.5, 0.0, -9.6],
    rotation: [0, 44, 0],
    bbox: [5.4, 9.2, 5.4],
    coverPoints: 6,
  },
  {
    id: "veg-01",
    name: "松树 / 高大",
    labels: ["树", "松树", "高大", "绿色", "森林"],
    location: [4.8, 0.0, -12.3],
    rotation: [0, 0, 0],
    bbox: [2.0, 8.0, 2.0],
    coverPoints: 0,
  },
];

const STORAGE_KEY = "label-embedding-assets";
let assets = [];

const state = {
  labelIndex: [],
  entityIndex: [],
  df: new Map(),
  idf: new Map(),
  labelEmbeddings: new Map(),
  labelToAssets: new Map(),
  normLabelsByAsset: new Map(),
  embeddingCenter: null,
  embeddingCache: new Map(),
};

const assetList = document.getElementById("assetList");
const apiStatus = document.getElementById("apiStatus");
const apiDot = document.getElementById("apiDot");
const apiUrlInput = document.getElementById("apiUrl");
const modelNameInput = document.getElementById("modelName");
const checkApiBtn = document.getElementById("checkApi");
const buildIndexBtn = document.getElementById("buildIndex");
const indexStatus = document.getElementById("indexStatus");
const indexCount = document.getElementById("indexCount");
const progressBar = document.getElementById("progressBar");
const indexNote = document.getElementById("indexNote");
const queryInput = document.getElementById("queryInput");
const topKInput = document.getElementById("topK");
const topKValue = document.getElementById("topKValue");
const topMInput = document.getElementById("topM");
const tauCoverInput = document.getElementById("tauCover");
const tauHeadInput = document.getElementById("tauHead");
const enableHeadGateInput = document.getElementById("enableHeadGate");
const searchBtn = document.getElementById("searchBtn");
const searchNote = document.getElementById("searchNote");
const resultsEl = document.getElementById("results");
const logEl = document.getElementById("log");
const assetJson = document.getElementById("assetJson");
const applyAssetsBtn = document.getElementById("applyAssets");
const resetAssetsBtn = document.getElementById("resetAssets");
const exportAssetsBtn = document.getElementById("exportAssets");
const importAssetsInput = document.getElementById("importAssets");
const dataNote = document.getElementById("dataNote");
const assetIdInput = document.getElementById("assetIdInput");
const assetNameInput = document.getElementById("assetNameInput");
const labelInput = document.getElementById("labelInput");
const addLabelBtn = document.getElementById("addLabelBtn");
const labelChips = document.getElementById("labelChips");
const locX = document.getElementById("locX");
const locY = document.getElementById("locY");
const locZ = document.getElementById("locZ");
const rotX = document.getElementById("rotX");
const rotY = document.getElementById("rotY");
const rotZ = document.getElementById("rotZ");
const boxX = document.getElementById("boxX");
const boxY = document.getElementById("boxY");
const boxZ = document.getElementById("boxZ");
const coverPointsInput = document.getElementById("coverPoints");
const saveAssetBtn = document.getElementById("saveAsset");
const clearFormBtn = document.getElementById("clearForm");
const formNote = document.getElementById("formNote");

let currentLabels = [];
let editingAssetId = null;

function normalizeAssets(raw) {
  if (!Array.isArray(raw)) throw new Error("JSON 必须是数组");
  return raw.map((item, index) => {
    if (!item || typeof item !== "object") {
      throw new Error(`第 ${index + 1} 项不是对象`);
    }
    const name = String(item.name || item.id || `资产 ${index + 1}`);
    const id = String(item.id || `asset-${index + 1}`);
    const labels = Array.isArray(item.labels)
      ? item.labels.map((label) => String(label).trim()).filter(Boolean)
      : [];
    if (labels.length === 0) {
      throw new Error(`资产 "${name}" 的 labels 不能为空`);
    }
    const location = Array.isArray(item.location) && item.location.length === 3 ? item.location : [0, 0, 0];
    const rotation = Array.isArray(item.rotation) && item.rotation.length === 3 ? item.rotation : [0, 0, 0];
    const bbox = Array.isArray(item.bbox) && item.bbox.length === 3 ? item.bbox : [0, 0, 0];
    const coverPoints = Number.isFinite(item.coverPoints) ? item.coverPoints : 0;
    return {
      id,
      name,
      labels,
      location,
      rotation,
      bbox,
      coverPoints,
    };
  });
}

function loadAssets() {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (!stored) return normalizeAssets(defaultAssets);
  try {
    return normalizeAssets(JSON.parse(stored));
  } catch {
    return normalizeAssets(defaultAssets);
  }
}

function saveAssets(list) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
}

function resetIndexState(note) {
  indexStatus.textContent = "未构建";
  indexCount.textContent = "0";
  progressBar.style.width = "0%";
  indexNote.textContent = note || "索引已清空，请重新构建。";
  state.labelIndex = [];
  state.entityIndex = [];
  state.df = new Map();
  state.idf = new Map();
  state.labelEmbeddings = new Map();
  state.labelToAssets = new Map();
  state.normLabelsByAsset = new Map();
  state.embeddingCenter = null;
}

function setAssets(list, persist = true) {
  assets = list;
  if (persist) saveAssets(list);
  renderAssets();
  resetIndexState("资产已更新，请重新构建索引。");
  assetJson.value = JSON.stringify(list, null, 2);
}

function readVector(x, y, z) {
  return [x, y, z].map((input) => Number.parseFloat(input.value) || 0);
}

function writeVector(values, x, y, z) {
  const [vx, vy, vz] = values || [0, 0, 0];
  x.value = vx;
  y.value = vy;
  z.value = vz;
}

function renderLabelChips() {
  labelChips.innerHTML = "";
  currentLabels.forEach((label) => {
    const chip = document.createElement("span");
    chip.className = "label-chip";
    chip.innerHTML = `${label}<button type="button" data-label="${label}">×</button>`;
    labelChips.appendChild(chip);
  });
}

function setEditing(asset) {
  if (!asset) {
    editingAssetId = null;
    saveAssetBtn.textContent = "添加资产";
    return;
  }
  editingAssetId = asset.id;
  saveAssetBtn.textContent = "更新资产";
  assetIdInput.value = asset.id;
  assetNameInput.value = asset.name;
  currentLabels = [...asset.labels];
  renderLabelChips();
  writeVector(asset.location, locX, locY, locZ);
  writeVector(asset.rotation, rotX, rotY, rotZ);
  writeVector(asset.bbox, boxX, boxY, boxZ);
  coverPointsInput.value = asset.coverPoints;
}

function clearForm() {
  assetIdInput.value = "";
  assetNameInput.value = "";
  labelInput.value = "";
  currentLabels = [];
  renderLabelChips();
  writeVector([0, 0, 0], locX, locY, locZ);
  writeVector([0, 0, 0], rotX, rotY, rotZ);
  writeVector([0, 0, 0], boxX, boxY, boxZ);
  coverPointsInput.value = "";
  setEditing(null);
  formNote.textContent = "提示：标签不能为空。";
}

function addLabelsFromInput(text) {
  const parts = text
    .split(/[，,]/)
    .map((item) => item.trim())
    .filter(Boolean);
  parts.forEach((label) => {
    if (!currentLabels.includes(label)) currentLabels.push(label);
  });
  renderLabelChips();
}

function buildAssetFromForm() {
  const name = assetNameInput.value.trim() || "未命名资产";
  const id = assetIdInput.value.trim() || `asset-${Date.now()}`;
  if (currentLabels.length === 0) {
    throw new Error("请至少添加一个标签");
  }
  return {
    id,
    name,
    labels: [...currentLabels],
    location: readVector(locX, locY, locZ),
    rotation: readVector(rotX, rotY, rotZ),
    bbox: readVector(boxX, boxY, boxZ),
    coverPoints: Number.parseInt(coverPointsInput.value || "0", 10),
  };
}

function renderAssets() {
  assetList.innerHTML = "";
  assets.forEach((asset) => {
    const card = document.createElement("div");
    card.className = "asset";
    card.innerHTML = `
      <h3>${asset.name}</h3>
      <div class="chips">
        ${asset.labels.map((label) => `<span class="chip">${label}</span>`).join("")}
      </div>
      <div class="meta">
        <div>ID: ${asset.id}</div>
        <div>位置: [${asset.location.join(", ")}]</div>
        <div>朝向: [${asset.rotation.join(", ")}]</div>
        <div>包围盒: [${asset.bbox.join(", ")}]</div>
        <div>掩体点: ${asset.coverPoints}</div>
      </div>
      <div class="asset-actions">
        <button class="btn ghost small" data-action="edit" data-id="${asset.id}">编辑</button>
        <button class="btn ghost danger small" data-action="delete" data-id="${asset.id}">删除</button>
      </div>
    `;
    assetList.appendChild(card);
  });
}

function updateApiStatus(ok, text) {
  apiDot.style.background = ok ? "var(--accent-2)" : "var(--danger)";
  apiDot.style.boxShadow = ok
    ? "0 0 10px rgba(89,211,182,0.6)"
    : "0 0 10px rgba(255,107,107,0.6)";
  apiStatus.textContent = text;
}

function getHealthUrl() {
  try {
    const url = new URL(apiUrlInput.value);
    url.pathname = "/health";
    url.search = "";
    return url.toString();
  } catch {
    return apiUrlInput.value.replace(/\/v1\/embeddings.*/, "/health");
  }
}

async function checkApi() {
  updateApiStatus(false, "检查中…");
  try {
    const resp = await fetch(getHealthUrl());
    if (!resp.ok) throw new Error("API 无响应");
    const data = await resp.json();
    updateApiStatus(true, `已连接 · ${data.model}`);
  } catch (err) {
    updateApiStatus(false, "API 未连接");
  }
}

async function fetchEmbeddings(texts) {
  const payload = {
    input: texts,
    model: modelNameInput.value.trim(),
    normalize: true,
  };
  const resp = await fetch(apiUrlInput.value, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!resp.ok) {
    const message = await resp.text();
    throw new Error(message || "Embedding API error");
  }
  const data = await resp.json();
  return data.data.map((item) => item.embedding);
}

function cosineSimilarity(a, b) {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

function toHalfWidth(text) {
  return String(text)
    .replace(/[\uFF01-\uFF5E]/g, (ch) => String.fromCharCode(ch.charCodeAt(0) - 0xfee0))
    .replace(/\u3000/g, " ");
}

function normalizeText(text) {
  return toHalfWidth(text).toLowerCase().replace(/[^\w\u4e00-\u9fff]+/g, " ").replace(/\s+/g, " ").trim();
}

const SYNONYMS = new Map([
  ["lorry", "truck"],
  ["truck", "卡车"],
  ["vehicle", "车辆"],
  ["car", "汽车"],
  ["door", "大门"],
  ["gate", "大门"],
  ["tree", "树"],
  ["pine", "松树"],
  ["tower", "瞭望塔"],
  ["watchtower", "瞭望塔"],
  ["hangar", "机库"],
  ["warehouse", "机库"],
  ["metal", "金属"],
  ["steel", "钢制"],
  ["wood", "木制"],
  ["broken", "破损"],
  ["damage", "破损"],
  ["red", "红色"],
  ["blue", "蓝色"],
  ["green", "绿色"],
  ["yellow", "黄色"],
  ["货车", "卡车"],
  ["小汽车", "汽车"],
]);

function normalizeLabel(label) {
  const normalized = normalizeText(label);
  if (!normalized) return "";
  return SYNONYMS.get(normalized) || normalized;
}

const COLOR_LABELS = new Set(["红色", "蓝色", "绿色", "黄色", "黑色", "白色", "灰色", "紫色", "橙色", "棕色"]);

function typeCoef(label) {
  return COLOR_LABELS.has(label) ? 0.2 : 1.0;
}

function uniqueStrings(list) {
  const seen = new Set();
  const out = [];
  list.forEach((item) => {
    if (!item) return;
    if (seen.has(item)) return;
    seen.add(item);
    out.push(item);
  });
  return out;
}

function computeDfIdf(normLabelsByAsset) {
  const df = new Map();
  const assetIds = Array.from(normLabelsByAsset.keys());
  assetIds.forEach((assetId) => {
    const labels = new Set(normLabelsByAsset.get(assetId) || []);
    labels.forEach((label) => df.set(label, (df.get(label) || 0) + 1));
  });

  const N = assetIds.length || 1;
  const idf = new Map();
  df.forEach((count, label) => {
    idf.set(label, Math.log((N + 1) / (count + 1)) + 1);
  });

  return { df, idf };
}

function l2Normalize(vec) {
  let norm = 0;
  for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  if (!norm) return vec;
  return vec.map((v) => v / norm);
}

function meanEmbedding(embeddings) {
  if (!embeddings.length) return null;
  const dim = embeddings[0]?.length || 0;
  if (!dim) return null;
  const mean = new Array(dim).fill(0);
  let count = 0;
  embeddings.forEach((emb) => {
    if (!emb) return;
    count += 1;
    for (let i = 0; i < dim; i++) mean[i] += emb[i];
  });
  if (!count) return null;
  for (let i = 0; i < dim; i++) mean[i] /= count;
  return mean;
}

function centerAndNormalizeEmbedding(embedding, center) {
  if (!embedding) return null;
  if (!center) return embedding;
  const out = new Array(embedding.length);
  for (let i = 0; i < embedding.length; i++) out[i] = embedding[i] - center[i];
  return l2Normalize(out);
}

function estimateSimilarityStats(embeddings, samplePairs = 2000) {
  const vecs = (embeddings || []).filter(Boolean);
  if (vecs.length < 2) return null;

  const n = vecs.length;
  const totalPairs = (n * (n - 1)) / 2;
  const draws = Math.max(50, Math.min(Number(samplePairs) || 2000, 10000, totalPairs));

  const sims = [];
  for (let i = 0; i < draws; i++) {
    const a = vecs[Math.floor(Math.random() * n)];
    let b = vecs[Math.floor(Math.random() * n)];
    if (a === b) b = vecs[(Math.floor(Math.random() * (n - 1)) + 1) % n];
    sims.push(cosineSimilarity(a, b));
  }

  sims.sort((x, y) => x - y);
  const at = (p) => sims[Math.max(0, Math.min(sims.length - 1, Math.floor(p * (sims.length - 1))))];
  return { p50: at(0.5), p90: at(0.9), p95: at(0.95) };
}

function buildWeightedEmbedding(embeddings, weights) {
  if (!embeddings.length) return null;
  const acc = new Array(embeddings[0].length).fill(0);
  let any = false;
  for (let i = 0; i < embeddings.length; i++) {
    const emb = embeddings[i];
    if (!emb) continue;
    any = true;
    const w = Number(weights?.[i] ?? 1) || 0;
    for (let j = 0; j < acc.length; j++) acc[j] += w * emb[j];
  }
  return any ? l2Normalize(acc) : null;
}

function buildEntityEmbedding(labels, labelEmbeddings, idf) {
  const first = labelEmbeddings.values().next().value;
  if (!first) return null;
  const acc = new Array(first.length).fill(0);
  let any = false;
  labels.forEach((label) => {
    const emb = labelEmbeddings.get(label);
    if (!emb) return;
    any = true;
    const w = (idf.get(label) || 1) * typeCoef(label);
    for (let i = 0; i < acc.length; i++) acc[i] += w * emb[i];
  });
  return any ? l2Normalize(acc) : null;
}

function extractPhrases(text, vocab) {
  const normalized = normalizeText(text);
  if (!normalized) return [];

  const rough = normalized.split(" ").filter(Boolean);
  const vocabList = Array.isArray(vocab) ? vocab : [];
  const phrases = [];

  rough.forEach((token) => {
    const mapped = SYNONYMS.get(token) || token;
    const hits = [];
    for (let i = 0; i < vocabList.length; i++) {
      const label = vocabList[i];
      if (label && mapped.includes(label)) hits.push(label);
    }
    hits.sort((a, b) => b.length - a.length);
    const chosen = [];
    hits.forEach((label) => {
      if (chosen.some((picked) => picked.includes(label))) return;
      chosen.push(label);
    });
    chosen.sort((a, b) => mapped.indexOf(a) - mapped.indexOf(b));
    if (chosen.length) phrases.push(...chosen);
    else phrases.push(mapped);
  });

  return uniqueStrings(phrases.map(normalizeLabel).filter(Boolean));
}

function pickHeadPhrase(phrases, weights) {
  if (!phrases.length) return null;
  let head = null;
  let best = -Infinity;
  phrases.forEach((q) => {
    if (typeCoef(q) < 1) return;
    const w = weights.get(q) || 0;
    if (w > best) {
      best = w;
      head = q;
      return;
    }
    if (w === best && head && q.length > head.length) head = q;
  });
  return head || phrases[0];
}

function topSimilarLabels(queryEmbedding, labelEmbeddings, topM) {
  const scored = [];
  labelEmbeddings.forEach((emb, label) => {
    scored.push({ label, sim: cosineSimilarity(queryEmbedding, emb) });
  });
  scored.sort((a, b) => b.sim - a.sim);
  return scored.slice(0, Math.max(1, Math.min(topM || 1, scored.length)));
}

function bestLabelMatch(queryEmbedding, labels, labelEmbeddings) {
  let bestLabel = null;
  let bestSim = -Infinity;
  labels.forEach((label) => {
    const emb = labelEmbeddings.get(label);
    if (!emb) return;
    const sim = cosineSimilarity(queryEmbedding, emb);
    if (sim > bestSim) {
      bestSim = sim;
      bestLabel = label;
    }
  });
  return { label: bestLabel, sim: bestSim === -Infinity ? 0 : bestSim };
}

function embeddingCacheKey(text) {
  const api = apiUrlInput.value.trim();
  const model = modelNameInput.value.trim();
  return `${api}::${model}::${text}`;
}

async function embedWithCache(texts, onProgress) {
  const embeddings = [];
  const batch = [];
  const batchIndex = [];
  texts.forEach((text, idx) => {
    const key = embeddingCacheKey(text);
    if (state.embeddingCache.has(key)) {
      embeddings[idx] = state.embeddingCache.get(key);
    } else {
      batch.push(text);
      batchIndex.push(idx);
    }
  });

  if (batch.length > 0) {
    const batchEmb = await fetchEmbeddings(batch);
    batchEmb.forEach((emb, i) => {
      const idx = batchIndex[i];
      embeddings[idx] = emb;
      state.embeddingCache.set(embeddingCacheKey(texts[idx]), emb);
      if (onProgress) onProgress(idx + 1, texts.length);
    });
  }

  return embeddings;
}

async function buildIndex() {
  const mode = document.querySelector("input[name='mode']:checked").value;
  buildIndexBtn.disabled = true;
  indexStatus.textContent = "构建中…";
  indexNote.textContent = "正在计算向量，请稍候。";
  progressBar.style.width = "0%";

  if (assets.length === 0) {
    indexStatus.textContent = "失败";
    indexNote.textContent = "没有可用资产，请先配置资产与标签。";
    buildIndexBtn.disabled = false;
    return;
  }

  try {
    const normLabelsByAsset = new Map();
    assets.forEach((asset) => {
      normLabelsByAsset.set(asset.id, uniqueStrings(asset.labels.map(normalizeLabel).filter(Boolean)));
    });

    const { df, idf } = computeDfIdf(normLabelsByAsset);
    const vocab = Array.from(df.keys());
    if (!vocab.length) throw new Error("没有可用标签");

    const vocabEmbeddings = await embedWithCache(vocab, (done, total) => {
      progressBar.style.width = `${Math.round((done / total) * 100)}%`;
    });

    state.embeddingCenter = meanEmbedding(vocabEmbeddings);

    const labelEmbeddings = new Map();
    vocab.forEach((label, idx) => {
      labelEmbeddings.set(label, centerAndNormalizeEmbedding(vocabEmbeddings[idx], state.embeddingCenter));
    });

    const labelToAssets = new Map();
    const labelIndex = [];
    assets.forEach((asset) => {
      const labels = normLabelsByAsset.get(asset.id) || [];
      labels.forEach((label) => {
        const emb = labelEmbeddings.get(label);
        if (!emb) return;
        labelIndex.push({ assetId: asset.id, label, embedding: emb });
        if (!labelToAssets.has(label)) labelToAssets.set(label, new Set());
        labelToAssets.get(label).add(asset.id);
      });
    });

    const entityIndex = assets
      .map((asset) => {
        const labels = normLabelsByAsset.get(asset.id) || [];
        const emb = buildEntityEmbedding(labels, labelEmbeddings, idf);
        return emb ? { assetId: asset.id, embedding: emb } : null;
      })
      .filter(Boolean);

    state.labelIndex = labelIndex;
    state.entityIndex = entityIndex;
    state.df = df;
    state.idf = idf;
    state.labelEmbeddings = labelEmbeddings;
    state.labelToAssets = labelToAssets;
    state.normLabelsByAsset = normLabelsByAsset;

    const simStats = estimateSimilarityStats(Array.from(labelEmbeddings.values()), 3000);
    const simNote = simStats
      ? ` · Sim(p50=${simStats.p50.toFixed(2)}, p90=${simStats.p90.toFixed(2)}, p95=${simStats.p95.toFixed(2)})`
      : "";

    progressBar.style.width = "100%";
    indexStatus.textContent = "已就绪";
    indexCount.textContent = (mode === "label" ? state.labelIndex.length : state.entityIndex.length).toString();
    indexNote.textContent = `索引已构建：Label=${state.labelIndex.length} · Entity=${state.entityIndex.length}（IDF 加权 + 向量中心化）${simNote}`;
  } catch (err) {
    indexStatus.textContent = "失败";
    indexNote.textContent = `构建失败：${err.message}`;
  } finally {
    buildIndexBtn.disabled = false;
  }
}

function renderResults(results) {
  resultsEl.innerHTML = "";
  if (results.length === 0) {
    resultsEl.innerHTML = `<div class="note">无结果</div>`;
    return;
  }
  results.forEach((result, idx) => {
    const div = document.createElement("div");
    div.className = "result";
    div.innerHTML = `
      <strong>#${idx + 1} · ${result.name}</strong>
      <div>匹配标签：${result.matchLabel || "（entity级）"}</div>
      <div class="score">相似度：${result.score.toFixed(4)}</div>
    `;
    resultsEl.appendChild(div);
  });
}

function addLog(entry) {
  const line = document.createElement("div");
  line.className = "log-entry";
  line.innerHTML = entry;
  logEl.prepend(line);
  if (logEl.children.length > 6) {
    logEl.removeChild(logEl.lastChild);
  }
}

async function runSearch() {
  const query = queryInput.value.trim();
  if (!query) return;

  const mode = document.querySelector("input[name='mode']:checked").value;
  const ready = mode === "label" ? state.labelIndex.length > 0 : state.entityIndex.length > 0;
  if (!ready) {
    searchNote.textContent = "索引为空，请先构建索引。";
    return;
  }

  searchBtn.disabled = true;
  searchNote.textContent = "检索中…";
  try {
    const topK = Number(topKInput.value);
    const assetById = new Map(assets.map((asset) => [asset.id, asset]));
    const vocab = Array.from(state.labelEmbeddings.keys());
    let phrases = extractPhrases(query, vocab);
    if (!phrases.length) phrases = uniqueStrings([normalizeLabel(query)].filter(Boolean));
    if (!phrases.length) throw new Error("Query 为空");

    const phraseWeights = new Map();
    phrases.forEach((q) => {
      phraseWeights.set(q, (state.idf.get(q) || 1) * typeCoef(q));
    });

    const phraseEmbeddingsRaw = await embedWithCache(phrases);
    const phraseEmbeddings = phraseEmbeddingsRaw.map((emb) =>
      centerAndNormalizeEmbedding(emb, state.embeddingCenter)
    );
    const embByPhrase = new Map();
    phrases.forEach((q, idx) => embByPhrase.set(q, phraseEmbeddings[idx]));

    if (mode === "label") {
      const TOP_M = Math.max(1, Math.min(200, Number.parseInt(topMInput?.value || "50", 10) || 50));
      const TAU_COVER = Math.max(0, Math.min(1, Number.parseFloat(tauCoverInput?.value || "0.30") || 0.3));
      const TAU_HEAD = Math.max(0, Math.min(1, Number.parseFloat(tauHeadInput?.value || "0.35") || 0.35));
      const enableHeadGate = enableHeadGateInput ? enableHeadGateInput.checked : true;

      const cand = new Set();
      phrases.forEach((q) => {
        const qEmb = embByPhrase.get(q);
        if (!qEmb) return;
        const top = topSimilarLabels(qEmb, state.labelEmbeddings, TOP_M);
        top.forEach(({ label }) => {
          const assetIds = state.labelToAssets.get(label);
          if (!assetIds) return;
          assetIds.forEach((id) => cand.add(id));
        });
      });

      const candidateIds = cand.size ? Array.from(cand) : assets.map((a) => a.id);
      const head = pickHeadPhrase(phrases, phraseWeights);
      const headEmb = head ? embByPhrase.get(head) : null;

      const scored = [];
      candidateIds.forEach((assetId) => {
        const asset = assetById.get(assetId);
        if (!asset) return;
        const labels = state.normLabelsByAsset.get(assetId) || [];
        if (!labels.length) return;

        if (enableHeadGate && head && headEmb) {
          const headMatch = bestLabelMatch(headEmb, labels, state.labelEmbeddings);
          if (headMatch.sim < TAU_HEAD) return;
        }

        let score = 0;
        let covered = 0;
        const matches = phrases.map((q) => {
          const qEmb = embByPhrase.get(q);
          const best = qEmb ? bestLabelMatch(qEmb, labels, state.labelEmbeddings) : { label: null, sim: 0 };
          const w = phraseWeights.get(q) || 0;
          const usedSim = best.sim >= TAU_COVER ? best.sim : 0;
          if (usedSim > 0) covered += 1;
          score += w * usedSim;
          return { q, label: best.label, sim: best.sim, usedSim };
        });

        const summary = matches
          .filter((m) => m.usedSim > 0 && m.label)
          .sort((a, b) => b.usedSim - a.usedSim)
          .slice(0, 3)
          .map((m) => `${m.q}→${m.label}(${m.usedSim.toFixed(2)})`)
          .join(" / ");

        scored.push({
          name: asset.name,
          score,
          matchLabel: `覆盖 ${covered}/${phrases.length}${summary ? ` · ${summary}` : ""}`,
        });
      });

      scored.sort((a, b) => b.score - a.score);

      const s1 = scored[0]?.score ?? 0;
      const s2 = scored[1]?.score ?? 0;
      const Ctext = (s1 - s2) / Math.max(Math.abs(s1), 1e-6);

      const sliced = scored.slice(0, topK);
      renderResults(sliced);
      addLog(
        `Query: <strong>${query}</strong> · Mode: ${mode} · Q: ${phrases
          .map((q) => `<code>${q}</code>`)
          .join(" ")} · HeadGate: <code>${enableHeadGate ? `${head}≥${TAU_HEAD.toFixed(2)}` : "off"}</code> · Cover≥<code>${TAU_COVER.toFixed(
          2
        )}</code> · RecallTopM=<code>${TOP_M}</code> · Ctext: <code>${Ctext.toFixed(3)}</code> · Top-${topK}: ${sliced
          .map((r) => `${r.name} (${r.score.toFixed(3)})`)
          .join(" / ")}`
      );
      searchNote.textContent = `检索完成 · Ctext=${Ctext.toFixed(3)}`;
    } else {
      const weightsArr = phrases.map((q) => phraseWeights.get(q) || 1);
      const queryVec = buildWeightedEmbedding(phraseEmbeddings, weightsArr);
      if (!queryVec) throw new Error("Query embedding 为空");

      const results = state.entityIndex
        .map((entry) => {
          const score = cosineSimilarity(queryVec, entry.embedding);
          const asset = assetById.get(entry.assetId);
          return {
            name: asset ? asset.name : entry.assetId,
            score,
            matchLabel: null,
          };
        })
        .sort((a, b) => b.score - a.score);

      const sliced = results.slice(0, topK);
      renderResults(sliced);
      addLog(
        `Query: <strong>${query}</strong> · Mode: ${mode} · Q: ${phrases
          .map((q) => `<code>${q}</code>`)
          .join(" ")} · Top-${topK}: ${sliced
          .map((r) => `${r.name} (${r.score.toFixed(3)})`)
          .join(" / ")}`
      );
      searchNote.textContent = "检索完成。";
    }
  } catch (err) {
    searchNote.textContent = `检索失败：${err.message}`;
  } finally {
    searchBtn.disabled = false;
  }
}

topKInput.addEventListener("input", () => {
  topKValue.textContent = topKInput.value;
});

document.querySelectorAll("input[name='mode']").forEach((radio) => {
  radio.addEventListener("change", () => {
    const mode = document.querySelector("input[name='mode']:checked").value;
    if (state.labelIndex.length === 0 && state.entityIndex.length === 0) {
      indexStatus.textContent = "未构建";
      indexNote.textContent = "模式切换后需要重新构建索引。";
      indexCount.textContent = "0";
      progressBar.style.width = "0%";
      return;
    }
    indexCount.textContent = (mode === "label" ? state.labelIndex.length : state.entityIndex.length).toString();
  });
});

checkApiBtn.addEventListener("click", checkApi);
buildIndexBtn.addEventListener("click", buildIndex);
searchBtn.addEventListener("click", runSearch);
apiUrlInput.addEventListener("change", () => resetIndexState("API 地址已更新，请重新构建索引。"));
modelNameInput.addEventListener("change", () => resetIndexState("模型已切换，请重新构建索引。"));
queryInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") runSearch();
});

addLabelBtn.addEventListener("click", () => {
  const text = labelInput.value.trim();
  if (!text) return;
  addLabelsFromInput(text);
  labelInput.value = "";
});

labelInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === ",") {
    event.preventDefault();
    if (labelInput.value.trim()) {
      addLabelsFromInput(labelInput.value);
      labelInput.value = "";
    }
  }
});

labelChips.addEventListener("click", (event) => {
  const target = event.target;
  if (target.tagName !== "BUTTON") return;
  const label = target.getAttribute("data-label");
  currentLabels = currentLabels.filter((item) => item !== label);
  renderLabelChips();
});

saveAssetBtn.addEventListener("click", () => {
  try {
    const asset = buildAssetFromForm();
    const existingIndex = assets.findIndex((item) => item.id === (editingAssetId || asset.id));
    if (existingIndex >= 0) {
      assets[existingIndex] = asset;
    } else {
      assets.push(asset);
    }
    setAssets([...assets], true);
    setEditing(null);
    formNote.textContent = "已保存资产。";
  } catch (err) {
    formNote.textContent = `保存失败：${err.message}`;
  }
});

clearFormBtn.addEventListener("click", clearForm);

assetList.addEventListener("click", (event) => {
  const btn = event.target.closest("button[data-action]");
  if (!btn) return;
  const action = btn.getAttribute("data-action");
  const id = btn.getAttribute("data-id");
  const asset = assets.find((item) => item.id === id);
  if (!asset) return;
  if (action === "edit") {
    setEditing(asset);
    formNote.textContent = `正在编辑：${asset.name}`;
  }
  if (action === "delete") {
    if (!confirm(`确定删除 ${asset.name} 吗？`)) return;
    assets = assets.filter((item) => item.id !== id);
    setAssets([...assets], true);
    if (editingAssetId === id) clearForm();
    formNote.textContent = "已删除资产。";
  }
});

applyAssetsBtn.addEventListener("click", () => {
  try {
    const parsed = JSON.parse(assetJson.value || "[]");
    const normalized = normalizeAssets(parsed);
    setAssets(normalized, true);
    dataNote.textContent = "已应用并保存。";
  } catch (err) {
    dataNote.textContent = `格式错误：${err.message}`;
  }
});

resetAssetsBtn.addEventListener("click", () => {
  setAssets(normalizeAssets(defaultAssets), true);
  dataNote.textContent = "已重置为示例数据。";
});

exportAssetsBtn.addEventListener("click", () => {
  const blob = new Blob([JSON.stringify(assets, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "assets.json";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
});

importAssetsInput.addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const text = await file.text();
  assetJson.value = text;
  dataNote.textContent = "已载入文件，点击“应用并保存”生效。";
  importAssetsInput.value = "";
});

assets = loadAssets();
assetJson.value = JSON.stringify(assets, null, 2);
renderAssets();
clearForm();
checkApi();
