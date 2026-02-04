const STORAGE = {
  apiBase: "item-finder::apiBase",
  itemsJson: "item-finder::itemsJson",
  selectedIds: "item-finder::selectedIds",
  contextMode: "item-finder::contextMode",
  config: "item-finder::config",
  debug: "item-finder::debug",
  posBackend: "item-finder::posBackend",
  preset: "item-finder::preset",
};

const SAMPLE_PAYLOAD = {
  enable_bm25: true,
  items: [
    {
      id: "veh-01",
      name: "蓝色卡车",
      type: ["卡车", "车", "汽车", "货车", "truck"],
      desc_labels: ["蓝色", "破损", "重型", "金属"],
    },
    {
      id: "veh-02",
      name: "红色皮卡",
      type: ["皮卡", "小卡车", "pickup", "卡车"],
      desc_labels: ["红色", "轻型", "金属", "完好"],
    },
    {
      id: "str-01",
      name: "红色房子",
      type: ["房子", "建筑", "小屋", "house"],
      desc_labels: ["红色", "木制", "破旧", "小"],
    },
    {
      id: "str-02",
      name: "铁门",
      type: ["门", "大门", "钢门", "door"],
      desc_labels: ["金属", "钢制", "生锈", "破损"],
    },
    {
      id: "str-03",
      name: "瞭望塔",
      type: ["塔", "watchtower"],
      desc_labels: ["木制", "高", "旧"],
    },
    {
      id: "key-01",
      name: "铁钥匙",
      type: ["钥匙", "金属钥匙"],
      desc_labels: ["金属", "铁制", "旧", "小"],
    },
    {
      id: "key-02",
      name: "铜钥匙",
      type: ["钥匙", "黄铜钥匙"],
      desc_labels: ["金属", "铜制", "新", "小"],
    },
    {
      id: "wpn-01",
      name: "铁剑",
      type: ["武器", "剑", "长剑"],
      desc_labels: ["金属", "铁制", "锋利", "旧"],
    },
    {
      id: "wpn-02",
      name: "木弓",
      type: ["武器", "弓", "弓箭"],
      desc_labels: ["木制", "轻", "完好"],
    },
    {
      id: "itm-01",
      name: "小型医疗包",
      type: ["医疗", "医疗包", "急救包"],
      desc_labels: ["小", "新", "布制"],
    },
    {
      id: "ptn-01",
      name: "绿色药水",
      type: ["药水", "恢复药水"],
      desc_labels: ["绿色", "小瓶", "玻璃"],
    },
    {
      id: "box-01",
      name: "木箱",
      type: ["箱子", "储物箱"],
      desc_labels: ["木制", "破损", "大"],
    },
    {
      id: "box-02",
      name: "金属箱",
      type: ["箱子", "军用箱"],
      desc_labels: ["金属", "重", "完好", "大"],
    },
    {
      id: "mat-01",
      name: "木板",
      type: ["材料", "木材", "木头"],
      desc_labels: ["木制", "长", "轻"],
    },
    {
      id: "mat-02",
      name: "铁锭",
      type: ["材料", "铁块", "金属块"],
      desc_labels: ["金属", "铁制", "重"],
    },
  ],
};

const DEFAULT_CONFIG = {
  top_k: 5,
  recall_topn_bm25: 50,
  recall_topn_name_vec: 50,
  recall_topm_desc_label: 50,
  enable_bm25: true,
  thresholds: {
    accept_p_nn: 0.01,
    clarify_p_nn: 0.05,
    accept_p_jj: 0.001,
    clarify_p_jj: 0.01,
    tau_cover: 0.35,
    min_coverage: 0.5,
    min_coverage_no_nn: 0.7,
    tau_margin: 0.15,
    tau_margin_no_nn: 0.2,
  },
  weights: {
    alpha_nn: 0.7,
    beta_jj: 0.3,
    gamma_bm25: 0.1,
  },
};

const PRESETS = {
  balanced: {
    ...DEFAULT_CONFIG,
    thresholds: {
      ...DEFAULT_CONFIG.thresholds,
      accept_p_nn: 0.01,
      clarify_p_nn: 0.05,
      accept_p_jj: 0.001,
      clarify_p_jj: 0.01,
      tau_cover: 0.35,
      min_coverage: 0.5,
      min_coverage_no_nn: 0.7,
      tau_margin: 0.15,
      tau_margin_no_nn: 0.2,
    },
  },
  conservative: {
    ...DEFAULT_CONFIG,
    thresholds: {
      ...DEFAULT_CONFIG.thresholds,
      accept_p_nn: 0.005,
      clarify_p_nn: 0.02,
      accept_p_jj: 0.0005,
      clarify_p_jj: 0.005,
      tau_cover: 0.4,
      min_coverage: 0.6,
      min_coverage_no_nn: 0.8,
      tau_margin: 0.2,
      tau_margin_no_nn: 0.25,
    },
  },
  aggressive: {
    ...DEFAULT_CONFIG,
    thresholds: {
      ...DEFAULT_CONFIG.thresholds,
      accept_p_nn: 0.03,
      clarify_p_nn: 0.12,
      accept_p_jj: 0.003,
      clarify_p_jj: 0.03,
      tau_cover: 0.28,
      min_coverage: 0.35,
      min_coverage_no_nn: 0.55,
      tau_margin: 0.1,
      tau_margin_no_nn: 0.14,
    },
  },
};

const $ = (id) => document.getElementById(id);

const apiBaseEl = $("apiBase");
const pingBtn = $("pingBtn");
const apiPill = $("apiPill");
const apiDot = $("apiDot");
const apiText = $("apiText");
const apiCount = $("apiCount");

const itemsJsonEl = $("itemsJson");
const useSampleBtn = $("useSampleBtn");
const beautifyBtn = $("beautifyBtn");
const loadBtn = $("loadBtn");
const exportBtn = $("exportBtn");
const importFile = $("importFile");
const loadNote = $("loadNote");
const localCount = $("localCount");
const itemsList = $("itemsList");
const contextModeEl = $("contextMode");
const selectAllBtn = $("selectAllBtn");
const selectNoneBtn = $("selectNoneBtn");

const presetSelect = $("presetSelect");
const topK = $("topK");
const topKVal = $("topKVal");
const bm25TopN = $("bm25TopN");
const nnVecTopN = $("nnVecTopN");
const jjTopM = $("jjTopM");
const enableBm25 = $("enableBm25");
const acceptPnn = $("acceptPnn");
const clarifyPnn = $("clarifyPnn");
const acceptPjj = $("acceptPjj");
const clarifyPjj = $("clarifyPjj");
const tauCover = $("tauCover");
const tauCoverVal = $("tauCoverVal");
const minCoverage = $("minCoverage");
const minCoverageVal = $("minCoverageVal");
const minCoverageNoNn = $("minCoverageNoNn");
const minCoverageNoNnVal = $("minCoverageNoNnVal");
const tauMargin = $("tauMargin");
const tauMarginVal = $("tauMarginVal");
const tauMarginNoNn = $("tauMarginNoNn");
const tauMarginNoNnVal = $("tauMarginNoNnVal");

const posBackendEl = $("posBackend");
const debugToggle = $("debugToggle");
const searchBtn = $("searchBtn");
const queryInput = $("queryInput");
const decisionPill = $("decisionPill");
const decisionDot = $("decisionDot");
const decisionText = $("decisionText");
const decisionReason = $("decisionReason");
const metricNN = $("metricNN");
const metricJJ = $("metricJJ");
const metricHead = $("metricHead");
const cards = $("cards");
const details = $("details");
const tokensEl = $("tokens");
const explainEl = $("explain");

let state = {
  parsedPayload: SAMPLE_PAYLOAD,
  selectedIds: new Set(),
};

function clamp(x, a, b) {
  return Math.max(a, Math.min(b, x));
}

function apiUrl(base, path) {
  const b = String(base || "").trim().replace(/\/+$/, "");
  return `${b}${path.startsWith("/") ? path : `/${path}`}`;
}

function safeJsonParse(text) {
  try {
    const value = JSON.parse(text);
    return { ok: true, value };
  } catch (err) {
    return { ok: false, error: err };
  }
}

function prettyJson(value) {
  return JSON.stringify(value, null, 2);
}

function loadStorage(key, fallback = null) {
  try {
    const raw = localStorage.getItem(key);
    if (raw == null) return fallback;
    return raw;
  } catch {
    return fallback;
  }
}

function saveStorage(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {}
}

function setApiStatus(ok, text) {
  apiDot.classList.remove("ok", "warn", "bad");
  if (ok) apiDot.classList.add("ok");
  else apiDot.classList.add("bad");
  apiText.textContent = text;
}

function setDecision(status, reason) {
  decisionDot.classList.remove("ok", "warn", "bad");
  if (status === "ACCEPT") decisionDot.classList.add("ok");
  else if (status === "CLARIFY") decisionDot.classList.add("warn");
  else if (status === "REJECT") decisionDot.classList.add("bad");
  decisionText.textContent = status || "—";
  decisionReason.textContent = reason || "";
}

function setNote(kind, msg) {
  loadNote.classList.remove("ok", "bad");
  if (kind === "ok") loadNote.classList.add("ok");
  if (kind === "bad") loadNote.classList.add("bad");
  loadNote.innerHTML = msg;
}

function inferItemPayload(raw) {
  if (raw && typeof raw === "object" && Array.isArray(raw.items)) return raw;
  if (Array.isArray(raw)) return { enable_bm25: true, items: raw };
  return null;
}

function normalizeItem(it, idx) {
  const id = String(it.id ?? `item-${idx + 1}`);
  const name = String(it.name ?? id);
  const type = Array.isArray(it.type)
    ? it.type.map((x) => String(x)).filter(Boolean)
    : it.type == null
      ? null
      : String(it.type);
  const aliases = Array.isArray(it.aliases) ? it.aliases.map((x) => String(x)).filter(Boolean) : [];
  const desc = Array.isArray(it.desc_labels)
    ? it.desc_labels.map((x) => String(x)).filter(Boolean)
    : Array.isArray(it.labels)
      ? it.labels.map((x) => String(x)).filter(Boolean)
      : [];
  return { id, name, type, aliases, desc_labels: desc };
}

function parseItemsFromTextarea() {
  const parsed = safeJsonParse(itemsJsonEl.value);
  if (!parsed.ok) {
    return { ok: false, error: String(parsed.error?.message || parsed.error || "JSON parse error") };
  }
  const payload = inferItemPayload(parsed.value);
  if (!payload) {
    return { ok: false, error: "JSON 必须是数组 items 或 { items: [...] } 结构" };
  }
  const items = (payload.items || []).map(normalizeItem);
  return { ok: true, payload: { enable_bm25: Boolean(payload.enable_bm25), items } };
}

function renderItemsList(payload) {
  itemsList.innerHTML = "";
  const items = payload?.items || [];
  localCount.textContent = String(items.length || 0);

  const mode = contextModeEl.value;
  const selected = state.selectedIds;

  if (mode === "all") {
    selected.clear();
  }

  items.forEach((it) => {
    const row = document.createElement("div");
    row.className = "item-row";

    const chkWrap = document.createElement("div");
    chkWrap.className = "chk";
    const chk = document.createElement("input");
    chk.type = "checkbox";
    chk.checked = mode === "selected" ? selected.has(it.id) : false;
    chk.disabled = mode !== "selected";
    chk.addEventListener("change", () => {
      if (chk.checked) selected.add(it.id);
      else selected.delete(it.id);
      persistSelection();
    });
    chkWrap.appendChild(chk);

    const main = document.createElement("div");
    main.className = "item-main";

    const nameLine = document.createElement("div");
    nameLine.className = "item-name";
    const strong = document.createElement("strong");
    strong.textContent = it.name;
    const id = document.createElement("span");
    id.className = "item-id";
    id.textContent = it.id;
    nameLine.appendChild(strong);
    nameLine.appendChild(id);

    const chips = document.createElement("div");
    chips.className = "chips";
    const types = Array.isArray(it.type) ? it.type : it.type ? [it.type] : [];
    types.forEach((t) => {
      const c = document.createElement("span");
      c.className = "chip hot";
      c.textContent = `type:${t}`;
      chips.appendChild(c);
    });
    (it.desc_labels || []).slice(0, 8).forEach((l) => {
      const c = document.createElement("span");
      c.className = "chip";
      c.textContent = l;
      chips.appendChild(c);
    });
    if ((it.desc_labels || []).length > 8) {
      const c = document.createElement("span");
      c.className = "chip dim";
      c.textContent = `+${it.desc_labels.length - 8}`;
      chips.appendChild(c);
    }

    main.appendChild(nameLine);
    main.appendChild(chips);

    row.appendChild(chkWrap);
    row.appendChild(main);
    itemsList.appendChild(row);
  });
}

function persistSelection() {
  saveStorage(STORAGE.selectedIds, JSON.stringify(Array.from(state.selectedIds)));
}

function applyPreset(name) {
  const preset = PRESETS[name] || PRESETS.balanced;
  fillConfig(preset);
  saveStorage(STORAGE.preset, name);
}

function fillConfig(cfg) {
  topK.value = String(cfg.top_k ?? 5);
  topKVal.textContent = String(topK.value);
  bm25TopN.value = String(cfg.recall_topn_bm25 ?? 50);
  nnVecTopN.value = String(cfg.recall_topn_name_vec ?? 50);
  jjTopM.value = String(cfg.recall_topm_desc_label ?? 50);
  enableBm25.checked = Boolean(cfg.enable_bm25 ?? true);

  acceptPnn.value = String(cfg.thresholds?.accept_p_nn ?? 0.01);
  clarifyPnn.value = String(cfg.thresholds?.clarify_p_nn ?? 0.05);
  acceptPjj.value = String(cfg.thresholds?.accept_p_jj ?? 0.001);
  clarifyPjj.value = String(cfg.thresholds?.clarify_p_jj ?? 0.01);

  tauCover.value = String(cfg.thresholds?.tau_cover ?? 0.35);
  tauCoverVal.textContent = Number(tauCover.value).toFixed(2);
  minCoverage.value = String(cfg.thresholds?.min_coverage ?? 0.5);
  minCoverageVal.textContent = Number(minCoverage.value).toFixed(2);
  minCoverageNoNn.value = String(cfg.thresholds?.min_coverage_no_nn ?? 0.7);
  minCoverageNoNnVal.textContent = Number(minCoverageNoNn.value).toFixed(2);
  tauMargin.value = String(cfg.thresholds?.tau_margin ?? 0.15);
  tauMarginVal.textContent = Number(tauMargin.value).toFixed(2);
  tauMarginNoNn.value = String(cfg.thresholds?.tau_margin_no_nn ?? 0.2);
  tauMarginNoNnVal.textContent = Number(tauMarginNoNn.value).toFixed(2);
}

function readConfigFromUI() {
  const cfg = {
    top_k: clamp(Number(topK.value || 5) || 5, 1, 10),
    recall_topn_bm25: clamp(Number(bm25TopN.value || 0) || 0, 0, 5000),
    recall_topn_name_vec: clamp(Number(nnVecTopN.value || 0) || 0, 0, 5000),
    recall_topm_desc_label: clamp(Number(jjTopM.value || 0) || 0, 0, 5000),
    enable_bm25: Boolean(enableBm25.checked),
    thresholds: {
      accept_p_nn: clamp(Number(acceptPnn.value || 0.01) || 0.01, 0, 1),
      clarify_p_nn: clamp(Number(clarifyPnn.value || 0.05) || 0.05, 0, 1),
      accept_p_jj: clamp(Number(acceptPjj.value || 0.001) || 0.001, 0, 1),
      clarify_p_jj: clamp(Number(clarifyPjj.value || 0.01) || 0.01, 0, 1),
      tau_cover: clamp(Number(tauCover.value || 0.35) || 0.35, 0, 1),
      min_coverage: clamp(Number(minCoverage.value || 0.5) || 0.5, 0, 1),
      min_coverage_no_nn: clamp(Number(minCoverageNoNn.value || 0.7) || 0.7, 0, 1),
      tau_margin: clamp(Number(tauMargin.value || 0.15) || 0.15, 0, 1),
      tau_margin_no_nn: clamp(Number(tauMarginNoNn.value || 0.2) || 0.2, 0, 1),
    },
    weights: { ...DEFAULT_CONFIG.weights },
  };
  saveStorage(STORAGE.config, JSON.stringify(cfg));
  return cfg;
}

async function ping() {
  const base = apiBaseEl.value.trim();
  if (!base) return;
  saveStorage(STORAGE.apiBase, base);

  setApiStatus(false, "…");
  apiCount.textContent = "0";
  try {
    const resp = await fetch(apiUrl(base, "/health"), { method: "GET" });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data?.detail || resp.statusText);
    setApiStatus(true, `ok · ${data.model || "model"} · ${data.device || "cpu"}`);
    apiCount.textContent = String(data.items_loaded ?? "0");
  } catch (err) {
    setApiStatus(false, String(err?.message || err || "offline"));
  }
}

async function loadToApi() {
  const base = apiBaseEl.value.trim();
  if (!base) return;
  const parsed = parseItemsFromTextarea();
  if (!parsed.ok) {
    setNote("bad", `JSON 错误：<code>${escapeHtml(parsed.error)}</code>`);
    return;
  }

  setNote("ok", "正在加载索引…（首次会下载模型/建立词典）");
  loadBtn.disabled = true;
  try {
    const resp = await fetch(apiUrl(base, "/v1/items/load"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(parsed.payload),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data?.detail || resp.statusText);
    setNote(
      "ok",
      `已加载：items=<code>${data.item_count}</code> · name_phrases=<code>${data.name_phrase_count}</code> · desc_labels=<code>${data.desc_label_count}</code> · neg_name=<code>${data.neg_name_samples}</code> · neg_desc=<code>${data.neg_desc_samples}</code>`
    );
    state.parsedPayload = parsed.payload;
    saveStorage(STORAGE.itemsJson, itemsJsonEl.value);
    renderItemsList(state.parsedPayload);
    await ping();
  } catch (err) {
    setNote("bad", `加载失败：<code>${escapeHtml(String(err?.message || err))}</code>`);
  } finally {
    loadBtn.disabled = false;
  }
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function makeBar(value01) {
  const bar = document.createElement("div");
  bar.className = "bar";
  const fill = document.createElement("i");
  fill.style.width = `${Math.round(clamp(value01, 0, 1) * 100)}%`;
  bar.appendChild(fill);
  return bar;
}

function cardForResult(r, isBest) {
  const card = document.createElement("div");
  card.className = `card ${isBest ? "best" : ""}`;
  const top = document.createElement("div");
  top.className = "card-top";

  const title = document.createElement("div");
  title.className = "card-title";
  const strong = document.createElement("strong");
  strong.textContent = r.name || r.id;
  const id = document.createElement("span");
  id.className = "id";
  id.textContent = r.id;
  title.appendChild(strong);
  title.appendChild(id);

  const nums = document.createElement("div");
  nums.className = "nums";
  const score = Number(r.score ?? 0);
  const conf = Number(r.confidence ?? 0);
  nums.textContent = `score=${score.toFixed(3)} · conf=${conf.toFixed(3)}`;

  top.appendChild(title);
  top.appendChild(nums);

  const scoreline = document.createElement("div");
  scoreline.className = "scoreline";
  scoreline.appendChild(makeBar(conf));
  const miniNums = document.createElement("div");
  miniNums.className = "nums";
  miniNums.textContent = `${Math.round(conf * 100)}%`;
  scoreline.appendChild(miniNums);

  const mini = document.createElement("div");
  mini.className = "mini";
  const ex = r.explain;
  if (ex) {
    const pnn = ex.p_nn == null ? "—" : Number(ex.p_nn).toExponential(2);
    const pjj = ex.p_jj == null ? "—" : Number(ex.p_jj).toExponential(2);
    mini.textContent = `coverage=${Number(ex.coverage ?? 0).toFixed(2)} · margin=${Number(ex.margin_ratio ?? 0).toFixed(2)} · p_nn=${pnn} · p_jj=${pjj}`;
  } else {
    mini.textContent = "（打开 debug 可查看 POS 与门控细节）";
  }

  card.appendChild(top);
  card.appendChild(scoreline);
  card.appendChild(mini);
  return card;
}

function renderTokens(tokens) {
  tokensEl.innerHTML = "";
  (tokens || []).forEach((t) => {
    const chip = document.createElement("div");
    chip.className = "token";
    const b = document.createElement("b");
    b.textContent = t.text;
    const em = document.createElement("em");
    em.textContent = t.pos || "—";
    chip.appendChild(b);
    chip.appendChild(em);
    tokensEl.appendChild(chip);
  });
}

function renderExplain(explain) {
  explainEl.textContent = explain ? JSON.stringify(explain, null, 2) : "（debug 关闭或无 explain）";
}

async function runSearch() {
  const base = apiBaseEl.value.trim();
  if (!base) return;
  const q = queryInput.value.trim();
  if (!q) return;

  const pos_backend = String(posBackendEl.value || "hanlp").trim().toLowerCase();
  saveStorage(STORAGE.posBackend, pos_backend);

  const cfg = readConfigFromUI();
  const debug = debugToggle.checked;
  saveStorage(STORAGE.debug, debug ? "1" : "0");

  const contextMode = contextModeEl.value;
  saveStorage(STORAGE.contextMode, contextMode);

  let candidate_ids = null;
  if (contextMode === "selected") {
    candidate_ids = Array.from(state.selectedIds);
  }

  setDecision("—", "检索中…");
  cards.innerHTML = "";

  searchBtn.disabled = true;
  try {
    const resp = await fetch(apiUrl(base, "/v1/item_search"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: q,
        debug,
        candidate_ids,
        config: cfg,
        pos_backend,
      }),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data?.detail || resp.statusText);

    const status = data?.decision?.status || "—";
    const reason = data?.decision?.reason || "";
    setDecision(status, reason);

    metricNN.textContent = (data?.parsed?.nn || []).length ? (data.parsed.nn || []).join(" ") : "—";
    metricJJ.textContent = (data?.parsed?.jj || []).length ? (data.parsed.jj || []).join(" ") : "—";
    metricHead.textContent = data?.parsed?.head_noun || "—";

    renderTokens(data?.parsed?.tokens || []);

    const out = [];
    if (data?.best) out.push({ ...data.best, _best: true });
    (data?.alternatives || []).forEach((x) => out.push({ ...x, _best: false }));

    if (!out.length) {
      cards.innerHTML = `<div class="card"><div class="mini">没有可显示的候选。请先 Load 物品库。</div></div>`;
      renderExplain(null);
      return;
    }

    out.forEach((r, idx) => {
      const card = cardForResult(r, idx === 0 && Boolean(r._best));
      cards.appendChild(card);
    });

    renderExplain(data?.best?.explain || (data?.alternatives?.[0]?.explain ?? null));

    if (debug) {
      details.open = true;
    }
  } catch (err) {
    setDecision("REJECT", String(err?.message || err || "search failed"));
  } finally {
    searchBtn.disabled = false;
  }
}

function download(filename, text) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "application/json;charset=utf-8" }));
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

function syncFromTextarea({ silent } = { silent: false }) {
  const parsed = parseItemsFromTextarea();
  if (!parsed.ok) {
    localCount.textContent = "—";
    if (!silent) setNote("bad", `JSON 错误：<code>${escapeHtml(parsed.error)}</code>`);
    return;
  }
  state.parsedPayload = parsed.payload;
  if (!silent) setNote("ok", "已解析本地 JSON（还未 load 到 API）。");
  renderItemsList(state.parsedPayload);
}

function init() {
  const base = loadStorage(STORAGE.apiBase, "http://127.0.0.1:8000");
  apiBaseEl.value = base;

  const savedItems = loadStorage(STORAGE.itemsJson, "");
  if (savedItems) itemsJsonEl.value = savedItems;
  if (!itemsJsonEl.value.trim()) itemsJsonEl.value = prettyJson(SAMPLE_PAYLOAD);
  const sanity = safeJsonParse(itemsJsonEl.value);
  if (!sanity.ok || !inferItemPayload(sanity.value)?.items?.length) {
    itemsJsonEl.value = prettyJson(SAMPLE_PAYLOAD);
    saveStorage(STORAGE.itemsJson, itemsJsonEl.value);
  }

  const selected = loadStorage(STORAGE.selectedIds, "[]");
  try {
    JSON.parse(selected).forEach((id) => state.selectedIds.add(String(id)));
  } catch {}

  const savedMode = loadStorage(STORAGE.contextMode, "all");
  contextModeEl.value = savedMode === "selected" ? "selected" : "all";

  const savedDebug = loadStorage(STORAGE.debug, "0");
  debugToggle.checked = savedDebug === "1";

  const savedPos = loadStorage(STORAGE.posBackend, "hanlp");
  posBackendEl.value = savedPos === "jieba" ? "jieba" : "hanlp";

  const preset = loadStorage(STORAGE.preset, "balanced");
  presetSelect.value = PRESETS[preset] ? preset : "balanced";

  const cfgRaw = loadStorage(STORAGE.config, "");
  if (cfgRaw) {
    const parsed = safeJsonParse(cfgRaw);
    if (parsed.ok) fillConfig({ ...DEFAULT_CONFIG, ...parsed.value });
    else applyPreset(presetSelect.value);
  } else {
    applyPreset(presetSelect.value);
  }

  syncFromTextarea({ silent: true });
  ping();
}

// Events
pingBtn.addEventListener("click", ping);

useSampleBtn.addEventListener("click", () => {
  itemsJsonEl.value = prettyJson(SAMPLE_PAYLOAD);
  saveStorage(STORAGE.itemsJson, itemsJsonEl.value);
  syncFromTextarea();
});

beautifyBtn.addEventListener("click", () => {
  const parsed = safeJsonParse(itemsJsonEl.value);
  if (!parsed.ok) {
    setNote("bad", `JSON 错误：<code>${escapeHtml(String(parsed.error?.message || parsed.error))}</code>`);
    return;
  }
  itemsJsonEl.value = prettyJson(parsed.value);
  saveStorage(STORAGE.itemsJson, itemsJsonEl.value);
  syncFromTextarea({ silent: true });
});

itemsJsonEl.addEventListener("input", () => {
  saveStorage(STORAGE.itemsJson, itemsJsonEl.value);
  syncFromTextarea({ silent: true });
});

loadBtn.addEventListener("click", loadToApi);

exportBtn.addEventListener("click", () => {
  const parsed = parseItemsFromTextarea();
  if (!parsed.ok) {
    setNote("bad", `JSON 错误：<code>${escapeHtml(parsed.error)}</code>`);
    return;
  }
  download("items_payload.json", prettyJson(parsed.payload));
});

importFile.addEventListener("change", async () => {
  const file = importFile.files?.[0];
  if (!file) return;
  const text = await file.text();
  itemsJsonEl.value = text;
  saveStorage(STORAGE.itemsJson, itemsJsonEl.value);
  syncFromTextarea();
  importFile.value = "";
});

contextModeEl.addEventListener("change", () => {
  const mode = contextModeEl.value;
  saveStorage(STORAGE.contextMode, mode);
  renderItemsList(state.parsedPayload);
  persistSelection();
});

selectAllBtn.addEventListener("click", () => {
  if (contextModeEl.value !== "selected") return;
  state.selectedIds.clear();
  (state.parsedPayload?.items || []).forEach((it) => state.selectedIds.add(it.id));
  persistSelection();
  renderItemsList(state.parsedPayload);
});

selectNoneBtn.addEventListener("click", () => {
  if (contextModeEl.value !== "selected") return;
  state.selectedIds.clear();
  persistSelection();
  renderItemsList(state.parsedPayload);
});

presetSelect.addEventListener("change", () => applyPreset(presetSelect.value));

topK.addEventListener("input", () => (topKVal.textContent = String(topK.value)));
tauCover.addEventListener("input", () => (tauCoverVal.textContent = Number(tauCover.value).toFixed(2)));
minCoverage.addEventListener("input", () => (minCoverageVal.textContent = Number(minCoverage.value).toFixed(2)));
minCoverageNoNn.addEventListener("input", () => (minCoverageNoNnVal.textContent = Number(minCoverageNoNn.value).toFixed(2)));
tauMargin.addEventListener("input", () => (tauMarginVal.textContent = Number(tauMargin.value).toFixed(2)));
tauMarginNoNn.addEventListener("input", () => (tauMarginNoNnVal.textContent = Number(tauMarginNoNn.value).toFixed(2)));

searchBtn.addEventListener("click", runSearch);
queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") runSearch();
});

apiBaseEl.addEventListener("change", () => {
  saveStorage(STORAGE.apiBase, apiBaseEl.value.trim());
  ping();
});

posBackendEl.addEventListener("change", () => {
  saveStorage(STORAGE.posBackend, String(posBackendEl.value || "hanlp"));
});

init();
