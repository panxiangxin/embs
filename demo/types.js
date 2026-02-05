const STORAGE = {
  apiBase: "item-finder::apiBase",
  apiKey: "item-finder::apiKey",
  keyword: "catalog::keyword",
  status: "catalog::status",
  limit: "catalog::limit",
};

const $ = (sel) => document.querySelector(sel);

const apiBaseEl = $("#apiBase");
const apiKeyEl = $("#apiKey");
const pingBtn = $("#pingBtn");
const apiDot = $("#apiDot");
const apiText = $("#apiText");
const idxPill = $("#idxPill");
const idxDot = $("#idxDot");
const idxText = $("#idxText");

const reloadBtn = $("#reloadBtn");
const newBtn = $("#newBtn");
const exportBtn = $("#exportBtn");
const importFile = $("#importFile");
const refreshIndexBtn = $("#refreshIndexBtn");

const keywordEl = $("#keyword");
const statusEl = $("#status");
const limitEl = $("#limit");
const dataDot = $("#dataDot");
const dataText = $("#dataText");
const itemsListEl = $("#itemsList");
const prevBtn = $("#prevBtn");
const nextBtn = $("#nextBtn");
const jumpTopBtn = $("#jumpTopBtn");
const pageText = $("#pageText");

const modePill = $("#modePill");
const modeDot = $("#modeDot");
const modeText = $("#modeText");

const itemIdEl = $("#itemId");
const itemNameEl = $("#itemName");
const itemStatusEl = $("#itemStatus");
const itemTypeEl = $("#itemType");
const itemAliasesEl = $("#itemAliases");
const itemLabelsEl = $("#itemLabels");
const itemAttrsEl = $("#itemAttrs");

const formatAttrsBtn = $("#formatAttrsBtn");
const copyJsonBtn = $("#copyJsonBtn");
const saveBtn = $("#saveBtn");
const disableBtn = $("#disableBtn");
const restoreBtn = $("#restoreBtn");
const deleteBtn = $("#deleteBtn");
const logEl = $("#log");

let state = {
  health: null,
  offset: 0,
  limit: 50,
  total: 0,
  keyword: "",
  status: "",
  rows: [],
  selectedId: null,
  selected: null,
  draft: null,
  mode: "idle", // idle | create | edit
  busy: false,
};

function apiUrl(base, path) {
  const b = String(base || "").trim().replace(/\/+$/, "");
  return `${b}${path.startsWith("/") ? path : `/${path}`}`;
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
    localStorage.setItem(key, String(value));
  } catch {}
}

function setDot(dotEl, kind) {
  dotEl.classList.remove("ok", "warn", "bad");
  if (kind) dotEl.classList.add(kind);
}

function setDataPill(ok, text) {
  dataDot.classList.remove("ok", "warn");
  dataDot.classList.add(ok ? "ok" : "warn");
  dataText.textContent = text;
}

function log(line) {
  const s = String(line ?? "");
  const ts = new Date().toLocaleTimeString();
  const next = `[${ts}] ${s}`.trimEnd();
  logEl.textContent = (logEl.textContent ? `${logEl.textContent}\n` : "") + next;
  logEl.scrollTop = logEl.scrollHeight;
}

function clearLog() {
  logEl.textContent = "";
}

function headersJson(includeKey = false) {
  const headers = { "Content-Type": "application/json" };
  const key = String(apiKeyEl?.value || "").trim();
  if (includeKey && key) headers["x-api-key"] = key;
  return headers;
}

async function fetchJson(url, options = {}) {
  const resp = await fetch(url, options);
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    const err = new Error(data?.detail || resp.statusText || "request failed");
    err.status = resp.status;
    err.payload = data;
    throw err;
  }
  return data;
}

function nowIso() {
  return new Date().toISOString();
}

function formatTime(ms) {
  const n = Number(ms || 0);
  if (!n) return "—";
  const d = new Date(n);
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  const hh = String(d.getHours()).padStart(2, "0");
  const mi = String(d.getMinutes()).padStart(2, "0");
  return `${mm}-${dd} ${hh}:${mi}`;
}

function splitMulti(text) {
  const raw = String(text ?? "");
  return raw
    .split(/[\n,，、;；/|]+/g)
    .map((x) => String(x || "").trim())
    .filter(Boolean);
}

function unique(values) {
  const out = [];
  const seen = new Set();
  for (const v of values || []) {
    const t = String(v || "").trim();
    if (!t) continue;
    if (seen.has(t)) continue;
    seen.add(t);
    out.push(t);
  }
  return out;
}

function typeToList(raw) {
  if (Array.isArray(raw)) return unique(raw);
  if (typeof raw === "string") return unique(splitMulti(raw));
  if (raw == null) return [];
  return unique([String(raw)]);
}

function listToLines(values) {
  const a = unique(values || []);
  return a.length ? a.join("\n") : "";
}

function parseAttrs(text) {
  const s = String(text || "").trim();
  if (!s) return {};
  const obj = JSON.parse(s);
  if (!obj || typeof obj !== "object" || Array.isArray(obj)) throw new Error("attrs must be a JSON object");
  return obj;
}

function toItemPayload(draft, { includeId }) {
  const id = String(draft?.id || draft?.item_id || "").trim();
  const name = String(draft?.name || "").trim();
  const status = String(draft?.status || "active").trim().toLowerCase();
  const type = typeToList(draft?.type);
  const aliases = unique(draft?.aliases || []);
  const desc_labels = unique(draft?.desc_labels || draft?.labels || []);
  const attrs = draft?.attrs && typeof draft.attrs === "object" && !Array.isArray(draft.attrs) ? draft.attrs : {};

  const out = {
    ...(includeId ? { id } : {}),
    name,
    status,
    type: type.length ? type : null,
    aliases,
    desc_labels,
    attrs,
  };
  return out;
}

function draftFromForm() {
  const id = String(itemIdEl.value || "").trim();
  const name = String(itemNameEl.value || "").trim();
  const status = String(itemStatusEl.value || "active").trim().toLowerCase();
  const type = unique(splitMulti(itemTypeEl.value));
  const aliases = unique(splitMulti(itemAliasesEl.value));
  const desc_labels = unique(splitMulti(itemLabelsEl.value));

  let attrs = {};
  const attrsRaw = String(itemAttrsEl.value || "").trim();
  if (attrsRaw) attrs = parseAttrs(attrsRaw);

  return {
    id,
    item_id: id,
    name,
    status,
    type: type.length ? type : null,
    aliases,
    desc_labels,
    labels: desc_labels,
    attrs,
  };
}

function fillFormFromItem(item, { mode }) {
  const it = item || {};
  const id = String(it.id || it.item_id || "");
  itemIdEl.value = id;
  itemNameEl.value = String(it.name || "");

  const status = String(it.status || "active").toLowerCase();
  if (status === "deleted") {
    itemStatusEl.value = "disabled";
    itemStatusEl.disabled = true;
  } else {
    itemStatusEl.disabled = false;
    itemStatusEl.value = status === "disabled" ? "disabled" : "active";
  }

  itemTypeEl.value = listToLines(typeToList(it.type));
  itemAliasesEl.value = listToLines(it.aliases || []);
  itemLabelsEl.value = listToLines(it.desc_labels || it.labels || []);
  itemAttrsEl.value = JSON.stringify(it.attrs || {}, null, 2);

  if (mode === "edit") {
    itemIdEl.disabled = true;
  } else {
    itemIdEl.disabled = false;
  }
}

function normalizeForCompare(draft) {
  const p = toItemPayload(draft, { includeId: true });
  return {
    id: String(p.id || ""),
    name: String(p.name || ""),
    status: String(p.status || "active"),
    type: unique(typeToList(p.type)),
    aliases: unique(p.aliases || []),
    desc_labels: unique(p.desc_labels || []),
    attrs: JSON.stringify(p.attrs || {}, Object.keys(p.attrs || {}).sort(), 0),
  };
}

function isDirty() {
  if (!state.selected || !state.draft) return false;
  const a = normalizeForCompare(state.selected);
  const b = normalizeForCompare(state.draft);
  return JSON.stringify(a) !== JSON.stringify(b);
}

function setMode(mode) {
  state.mode = mode;
  const dirty = isDirty();
  modeDot.classList.remove("ok", "warn", "bad");
  if (mode === "create") {
    modeDot.classList.add("ok");
    modeText.textContent = "create";
    saveBtn.textContent = "Create";
  } else if (mode === "edit") {
    modeDot.classList.add(dirty ? "warn" : "ok");
    modeText.textContent = dirty ? "edit · unsaved" : "edit";
    saveBtn.textContent = "Save";
  } else {
    modeDot.classList.add("warn");
    modeText.textContent = "—";
    saveBtn.textContent = "Save";
  }

  const status = String(state.selected?.status || "").toLowerCase();
  const hasSel = Boolean(state.selectedId);
  const deleted = status === "deleted";
  const disabled = status === "disabled";

  disableBtn.style.display = hasSel && !deleted && !disabled ? "inline-flex" : "none";
  restoreBtn.style.display = hasSel && (deleted || disabled) ? "inline-flex" : "none";
  deleteBtn.style.display = hasSel && !deleted ? "inline-flex" : "none";
}

function renderPager() {
  const total = Number(state.total || 0);
  const offset = Number(state.offset || 0);
  const limit = Number(state.limit || 50);
  const a = total ? offset + 1 : 0;
  const b = Math.min(total, offset + limit);
  pageText.textContent = total ? `${a}-${b} / ${total}` : "0 / 0";
  prevBtn.disabled = offset <= 0;
  nextBtn.disabled = offset + limit >= total;
}

function rowChips(types) {
  const list = typeToList(types);
  if (!list.length) return `<span class="chip dim">—</span>`;
  const show = list.slice(0, 3);
  const rest = list.length - show.length;
  const html = show.map((t, i) => `<span class="chip ${i === 0 ? "hot" : ""}">${escapeHtml(t)}</span>`).join("");
  return rest > 0 ? `${html}<span class="chip dim">+${rest}</span>` : html;
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function statusCell(status) {
  const s = String(status || "active").toLowerCase();
  const cls = s === "deleted" ? "deleted" : s === "disabled" ? "disabled" : "active";
  return `<span class="status"><span class="s-dot ${cls}"></span><span>${escapeHtml(s)}</span></span>`;
}

function renderList() {
  itemsListEl.innerHTML = "";
  const rows = state.rows || [];
  if (!rows.length) {
    const empty = document.createElement("div");
    empty.className = "mini";
    empty.textContent = "没有数据。你可以点击 New 创建，或用 Import 导入。";
    itemsListEl.appendChild(empty);
    renderPager();
    return;
  }

  for (const it of rows) {
    const id = String(it.id || it.item_id || "");
    const v = Number(it.version || 0) || 0;
    const labelCount = Array.isArray(it.desc_labels) ? it.desc_labels.length : Array.isArray(it.labels) ? it.labels.length : 0;
    const row = document.createElement("div");
    row.className = `row${state.selectedId === id ? " on" : ""}`;
    row.innerHTML = `
      <div class="cell-id">
        <b>${escapeHtml(id || "—")}</b>
        <span class="mono muted">v${escapeHtml(String(v || 1))} · labels=${escapeHtml(String(labelCount))}</span>
      </div>
      <div>${escapeHtml(String(it.name || "—"))}</div>
      <div class="chips">${rowChips(it.type)}</div>
      <div>${statusCell(it.status)}</div>
      <div class="right mono muted">${escapeHtml(formatTime(it.updated_at_ms))}</div>
    `;
    row.addEventListener("click", () => selectById(id));
    itemsListEl.appendChild(row);
  }
  renderPager();
}

async function ping() {
  const base = String(apiBaseEl.value || "").trim();
  if (!base) return;
  saveStorage(STORAGE.apiBase, base);

  setDot(apiDot, "warn");
  apiText.textContent = "…";
  setDot(idxDot, "warn");
  idxText.textContent = "index …";

  try {
    const h = await fetchJson(apiUrl(base, "/health"), { method: "GET" });
    state.health = h;
    setDot(apiDot, "ok");
    apiText.textContent = `ok · ${h.model || "model"} · ${h.device || "cpu"} · ${h.storage_backend || "store"}`;

    const cv = Number(h.catalog_version || 0);
    const iv = Number(h.index_catalog_version || 0);
    const stale = cv && iv && cv !== iv;
    setDot(idxDot, stale ? "warn" : "ok");
    idxText.textContent = cv ? `index v${iv} / v${cv}${stale ? " · stale" : ""}` : "index —";
  } catch (err) {
    setDot(apiDot, "bad");
    apiText.textContent = String(err?.message || err || "offline");
    setDot(idxDot, "bad");
    idxText.textContent = "index —";
  }
}

async function loadList({ keepSelection } = { keepSelection: true }) {
  const base = String(apiBaseEl.value || "").trim();
  if (!base) return;
  saveStorage(STORAGE.apiBase, base);

  state.limit = Number(limitEl.value || 50) || 50;
  state.keyword = String(keywordEl.value || "").trim();
  state.status = String(statusEl.value || "").trim();

  const qs = new URLSearchParams();
  qs.set("offset", String(state.offset));
  qs.set("limit", String(state.limit));
  if (state.keyword) qs.set("keyword", state.keyword);
  if (state.status) qs.set("status", state.status);

  setDataPill(false, "加载中…");
  try {
    const data = await fetchJson(apiUrl(base, `/v1/items?${qs.toString()}`), { method: "GET" });
    state.rows = Array.isArray(data?.items) ? data.items : [];
    state.total = Number(data?.total || 0);
    setDataPill(true, `items=${state.total} · page=${Math.floor(state.offset / state.limit) + 1}`);
    renderList();
    await ping();

    if (!keepSelection) return;
    if (state.selectedId) {
      const still = state.rows.find((x) => String(x.id || x.item_id || "") === state.selectedId);
      if (!still) {
        state.selectedId = null;
        state.selected = null;
        state.draft = null;
        setMode("idle");
      }
    }
  } catch (err) {
    setDataPill(false, `加载失败：${String(err?.message || err || "error")}`);
    state.rows = [];
    state.total = 0;
    renderList();
  }
}

async function selectById(itemId) {
  const id = String(itemId || "").trim();
  if (!id) return;

  const base = String(apiBaseEl.value || "").trim();
  if (!base) return;

  try {
    const it = await fetchJson(apiUrl(base, `/v1/items/${encodeURIComponent(id)}`), { method: "GET" });
    state.selectedId = id;
    state.selected = it;
    state.draft = JSON.parse(JSON.stringify(it));
    fillFormFromItem(state.draft, { mode: "edit" });
    clearLog();
    log(`Selected: ${id}`);
    setMode("edit");
    renderList();
  } catch (err) {
    log(`Select failed: ${String(err?.message || err || "error")}`);
  }
}

function newItem() {
  state.selectedId = null;
  state.selected = null;
  state.draft = {
    id: "",
    item_id: "",
    name: "",
    status: "active",
    type: null,
    aliases: [],
    desc_labels: [],
    labels: [],
    attrs: {},
  };
  fillFormFromItem(state.draft, { mode: "create" });
  clearLog();
  log("New: draft cleared");
  setMode("create");
  renderList();
}

async function refreshIndex() {
  const base = String(apiBaseEl.value || "").trim();
  if (!base) return;
  try {
    const data = await fetchJson(apiUrl(base, "/v1/index/refresh"), { method: "POST", headers: headersJson(true) });
    log(`Index refresh: changed=${Boolean(data.changed)}`);
    await ping();
  } catch (err) {
    log(`Index refresh failed: ${String(err?.message || err || "error")}`);
    if (Number(err?.status) === 401) apiKeyEl?.focus();
  }
}

async function saveItem() {
  const base = String(apiBaseEl.value || "").trim();
  if (!base) return;

  let draft;
  try {
    draft = draftFromForm();
  } catch (err) {
    log(`Bad form: ${String(err?.message || err || "invalid")}`);
    return;
  }

  const id = String(draft.id || "").trim();
  const name = String(draft.name || "").trim();
  if (!name) {
    log("Validation: name is required");
    itemNameEl.focus();
    return;
  }

  if (state.mode === "create") {
    if (!id) {
      log("Validation: item_id is required");
      itemIdEl.focus();
      return;
    }
    const payload = toItemPayload(draft, { includeId: true });
    try {
      const it = await fetchJson(apiUrl(base, "/v1/items"), {
        method: "POST",
        headers: headersJson(true),
        body: JSON.stringify(payload),
      });
      log(`Create: ok · id=${it.id} · version=${it.version}`);
      await loadList({ keepSelection: false });
      await selectById(it.id);
    } catch (err) {
      log(`Create failed: ${String(err?.message || err || "error")}`);
      if (Number(err?.status) === 401) apiKeyEl?.focus();
    }
    return;
  }

  if (!state.selectedId) {
    log("Save: no selected item");
    return;
  }

  const payload = toItemPayload(draft, { includeId: false });
  try {
    const it = await fetchJson(apiUrl(base, `/v1/items/${encodeURIComponent(state.selectedId)}`), {
      method: "PUT",
      headers: headersJson(true),
      body: JSON.stringify(payload),
    });
    log(`Save: ok · id=${it.id} · version=${it.version}`);
    await loadList({ keepSelection: true });
    await selectById(it.id);
  } catch (err) {
    log(`Save failed: ${String(err?.message || err || "error")}`);
    if (Number(err?.status) === 401) apiKeyEl?.focus();
  }
}

async function disableItem() {
  if (!state.selectedId) return;
  const base = String(apiBaseEl.value || "").trim();
  if (!base) return;
  const ok = confirm(`Disable item: ${state.selectedId} ?`);
  if (!ok) return;
  try {
    const it = await fetchJson(apiUrl(base, `/v1/items/${encodeURIComponent(state.selectedId)}?mode=disabled`), {
      method: "DELETE",
      headers: headersJson(true),
    });
    log(`Disable: ok · status=${it.status} · version=${it.version}`);
    await loadList({ keepSelection: true });
    await selectById(it.id);
  } catch (err) {
    log(`Disable failed: ${String(err?.message || err || "error")}`);
    if (Number(err?.status) === 401) apiKeyEl?.focus();
  }
}

async function restoreItem() {
  if (!state.selectedId) return;
  const base = String(apiBaseEl.value || "").trim();
  if (!base) return;
  try {
    const draft = draftFromForm();
    draft.status = "active";
    const payload = toItemPayload(draft, { includeId: false });
    const it = await fetchJson(apiUrl(base, `/v1/items/${encodeURIComponent(state.selectedId)}`), {
      method: "PUT",
      headers: headersJson(true),
      body: JSON.stringify(payload),
    });
    log(`Restore: ok · status=${it.status} · version=${it.version}`);
    await loadList({ keepSelection: true });
    await selectById(it.id);
  } catch (err) {
    log(`Restore failed: ${String(err?.message || err || "error")}`);
    if (Number(err?.status) === 401) apiKeyEl?.focus();
  }
}

async function deleteItem() {
  if (!state.selectedId) return;
  const base = String(apiBaseEl.value || "").trim();
  if (!base) return;
  const ok = confirm(`Delete (soft) item: ${state.selectedId} ?`);
  if (!ok) return;
  try {
    const it = await fetchJson(apiUrl(base, `/v1/items/${encodeURIComponent(state.selectedId)}?mode=deleted`), {
      method: "DELETE",
      headers: headersJson(true),
    });
    log(`Delete: ok · status=${it.status} · version=${it.version}`);
    await loadList({ keepSelection: false });
    newItem();
  } catch (err) {
    log(`Delete failed: ${String(err?.message || err || "error")}`);
    if (Number(err?.status) === 401) apiKeyEl?.focus();
  }
}

function formatAttrs() {
  try {
    const obj = parseAttrs(itemAttrsEl.value);
    itemAttrsEl.value = JSON.stringify(obj, null, 2);
    log("Attrs: formatted");
  } catch (err) {
    log(`Attrs: ${String(err?.message || err || "invalid json")}`);
  }
}

async function copyJson() {
  try {
    const draft = draftFromForm();
    const out = toItemPayload(draft, { includeId: true });
    const text = JSON.stringify(out, null, 2);
    await navigator.clipboard.writeText(text);
    log("Copied: item JSON");
  } catch (err) {
    log(`Copy failed: ${String(err?.message || err || "error")}`);
  }
}

async function exportAll() {
  const base = String(apiBaseEl.value || "").trim();
  if (!base) return;
  const q = statusEl.value ? `?status=${encodeURIComponent(statusEl.value)}` : "";
  try {
    const data = await fetchJson(apiUrl(base, `/v1/items/export${q}`), { method: "GET" });
    const name = `items_export_${Date.now()}.json`;
    download(name, JSON.stringify(data, null, 2));
    log(`Export: downloaded ${name}`);
  } catch (err) {
    log(`Export failed: ${String(err?.message || err || "error")}`);
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

async function importItemsFromFile(file) {
  const base = String(apiBaseEl.value || "").trim();
  if (!base) return;

  const raw = await file.text();
  let payload;
  try {
    payload = JSON.parse(raw);
  } catch (err) {
    log(`Import: bad JSON · ${String(err?.message || err || "")}`);
    return;
  }

  const mode = confirm("Import mode: OK=upsert（增量） / Cancel=replace（全量覆盖）") ? "upsert" : "replace";
  const rebuild = confirm("Import done. Rebuild index now?（需要 API Key）");

  try {
    const url = apiUrl(base, `/v1/items/import?rebuild=${rebuild ? "true" : "false"}&mode=${mode}`);
    const data = await fetchJson(url, {
      method: "POST",
      headers: headersJson(true),
      body: JSON.stringify(payload),
    });
    const r = data?.result || {};
    log(`Import: ok · ${r.replaced != null ? `replaced=${r.replaced}` : `created=${r.created ?? 0} updated=${r.updated ?? 0}`}`);
    await loadList({ keepSelection: false });
    await ping();
  } catch (err) {
    log(`Import failed: ${String(err?.message || err || "error")}`);
    if (Number(err?.status) === 401) apiKeyEl?.focus();
  }
}

function bindEvents() {
  pingBtn.addEventListener("click", ping);
  reloadBtn.addEventListener("click", () => loadList({ keepSelection: true }));
  newBtn.addEventListener("click", newItem);
  exportBtn.addEventListener("click", exportAll);
  refreshIndexBtn.addEventListener("click", refreshIndex);

  importFile.addEventListener("change", async () => {
    const file = importFile.files?.[0];
    if (!file) return;
    await importItemsFromFile(file);
    importFile.value = "";
  });

  keywordEl.addEventListener("input", () => {
    state.keyword = keywordEl.value;
    saveStorage(STORAGE.keyword, state.keyword);
  });
  keywordEl.addEventListener("change", () => {
    state.offset = 0;
    loadList({ keepSelection: false });
  });

  statusEl.addEventListener("change", () => {
    state.status = statusEl.value;
    saveStorage(STORAGE.status, state.status);
    state.offset = 0;
    loadList({ keepSelection: false });
  });

  limitEl.addEventListener("change", () => {
    state.limit = Number(limitEl.value || 50) || 50;
    saveStorage(STORAGE.limit, String(state.limit));
    state.offset = 0;
    loadList({ keepSelection: false });
  });

  prevBtn.addEventListener("click", () => {
    state.offset = Math.max(0, state.offset - state.limit);
    loadList({ keepSelection: true });
  });
  nextBtn.addEventListener("click", () => {
    state.offset = state.offset + state.limit;
    loadList({ keepSelection: true });
  });
  jumpTopBtn.addEventListener("click", () => {
    state.offset = 0;
    loadList({ keepSelection: true });
    itemsListEl.scrollTop = 0;
  });

  formatAttrsBtn.addEventListener("click", formatAttrs);
  copyJsonBtn.addEventListener("click", copyJson);
  saveBtn.addEventListener("click", saveItem);
  disableBtn.addEventListener("click", disableItem);
  restoreBtn.addEventListener("click", restoreItem);
  deleteBtn.addEventListener("click", deleteItem);

  const dirtyHook = () => {
    if (state.mode !== "edit") return;
    try {
      state.draft = draftFromForm();
    } catch {
      // ignore
    }
    setMode("edit");
  };

  [itemNameEl, itemStatusEl, itemTypeEl, itemAliasesEl, itemLabelsEl, itemAttrsEl].forEach((el) => {
    el.addEventListener("input", dirtyHook);
    el.addEventListener("change", dirtyHook);
  });

  apiBaseEl.addEventListener("change", () => {
    saveStorage(STORAGE.apiBase, apiBaseEl.value.trim());
    ping();
    loadList({ keepSelection: true });
  });
  apiKeyEl.addEventListener("change", () => {
    saveStorage(STORAGE.apiKey, apiKeyEl.value);
  });

  document.addEventListener("keydown", (e) => {
    const key = String(e.key || "").toLowerCase();
    if ((e.ctrlKey || e.metaKey) && key === "s") {
      e.preventDefault();
      saveItem();
    }
    if (!e.ctrlKey && !e.metaKey && key === "n" && (e.target?.tagName || "") !== "INPUT" && (e.target?.tagName || "") !== "TEXTAREA") {
      newItem();
    }
  });
}

function init() {
  apiBaseEl.value = loadStorage(STORAGE.apiBase, "http://127.0.0.1:8000");
  apiKeyEl.value = loadStorage(STORAGE.apiKey, "");

  keywordEl.value = loadStorage(STORAGE.keyword, "");
  statusEl.value = loadStorage(STORAGE.status, "");
  limitEl.value = loadStorage(STORAGE.limit, "50");

  state.keyword = keywordEl.value;
  state.status = statusEl.value;
  state.limit = Number(limitEl.value || 50) || 50;
  state.offset = 0;

  setMode("idle");
  bindEvents();
  ping();
  loadList({ keepSelection: false });
  newItem();
}

init();
