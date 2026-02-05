const STORAGE = {
  apiBase: "item-finder::apiBase",
  apiKey: "item-finder::apiKey",
  statusFilter: "type-studio::statusFilter",
  sortMode: "type-studio::sortMode",
  typeFilter: "type-studio::typeFilter",
  rebuild: "type-studio::rebuild",
};

const $ = (sel) => document.querySelector(sel);

const apiBaseEl = $("#apiBase");
const apiKeyEl = $("#apiKey");
const pingBtn = $("#pingBtn");
const apiDot = $("#apiDot");
const apiText = $("#apiText");

const reloadBtn = $("#reloadBtn");
const resetBtn = $("#resetBtn");

const typeFilterEl = $("#typeFilter");
const sortModeEl = $("#sortMode");
const statusFilterEl = $("#statusFilter");
const typesListEl = $("#typesList");

const dataDot = $("#dataDot");
const dataText = $("#dataText");

const tabRename = $("#tabRename");
const tabMerge = $("#tabMerge");
const tabRemove = $("#tabRemove");
const tabAdd = $("#tabAdd");
const panelRename = $("#panelRename");
const panelMerge = $("#panelMerge");
const panelRemove = $("#panelRemove");
const panelAdd = $("#panelAdd");

const renameFromEl = $("#renameFrom");
const renameToEl = $("#renameTo");
const renameBtn = $("#renameBtn");

const mergeFromListEl = $("#mergeFromList");
const mergeToEl = $("#mergeTo");
const mergeBtn = $("#mergeBtn");

const removeListEl = $("#removeList");
const removeBtn = $("#removeBtn");

const addTypeEl = $("#addType");
const addKeywordEl = $("#addKeyword");
const addScopeSelectedEl = $("#addScopeSelected");
const addBtn = $("#addBtn");

const selectAllTypesBtn = $("#selectAllTypesBtn");
const selectNoneTypesBtn = $("#selectNoneTypesBtn");

const rebuildToggleEl = $("#rebuildToggle");
const downloadPatchBtn = $("#downloadPatchBtn");
const applyBtn = $("#applyBtn");

const activeTypeText = $("#activeTypeText");
const selectedTypesText = $("#selectedTypesText");
const statsText = $("#statsText");
const itemsSampleEl = $("#itemsSample");

const changedCountEl = $("#changedCount");
const catalogVersionEl = $("#catalogVersion");
const indexVersionEl = $("#indexVersion");
const logEl = $("#log");

let state = {
  meta: null,
  status: null,
  enableBm25: true,
  itemsOriginal: [],
  itemsDraft: [],
  typeStats: [],
  activeType: null,
  selectedTypes: new Set(),
  filter: "",
  sortMode: "count",
  statusFilter: "",
};

function apiUrl(base, path) {
  const b = String(base || "").trim().replace(/\/+$/, "");
  return `${b}${path.startsWith("/") ? path : `/${path}`}`;
}

function clamp(x, a, b) {
  return Math.max(a, Math.min(b, x));
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

function clone(value) {
  if (typeof structuredClone === "function") return structuredClone(value);
  return JSON.parse(JSON.stringify(value));
}

function setDot(dotEl, kind) {
  dotEl.classList.remove("ok", "warn", "bad");
  if (kind) dotEl.classList.add(kind);
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
  const key = String(apiKeyEl.value || "").trim();
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

function getItemId(it) {
  return String(it?.item_id || it?.id || "");
}

function splitTypeString(s) {
  const raw = String(s ?? "").trim();
  if (!raw) return [];
  return raw
    .split(/[,\uFF0C\u3001/|;；\r\n\t]+/g)
    .map((x) => String(x || "").trim())
    .filter(Boolean);
}

function normalizeTypeList(values) {
  const out = [];
  const seen = new Set();
  for (const raw of values || []) {
    const t = String(raw || "").trim();
    if (!t) continue;
    if (seen.has(t)) continue;
    seen.add(t);
    out.push(t);
  }
  return out;
}

function getTypes(it) {
  const raw = it?.type;
  if (Array.isArray(raw)) return normalizeTypeList(raw);
  if (typeof raw === "string") return normalizeTypeList(splitTypeString(raw));
  if (raw == null) return [];
  return normalizeTypeList([String(raw)]);
}

function setTypes(it, types) {
  const t = normalizeTypeList(types);
  it.type = t.length ? t : null;
}

function typeKey(it) {
  return getTypes(it).join("\u0000");
}

function computeTypeStats(items) {
  const map = new Map();
  let maxCount = 1;
  for (let i = 0; i < items.length; i++) {
    const it = items[i];
    const types = getTypes(it);
    for (const t of types) {
      let row = map.get(t);
      if (!row) {
        row = { name: t, count: 0, itemIdx: [], sample: [] };
        map.set(t, row);
      }
      row.count += 1;
      row.itemIdx.push(i);
      if (row.sample.length < 2) row.sample.push(`${getItemId(it)} · ${String(it?.name || "").trim() || "—"}`);
      maxCount = Math.max(maxCount, row.count);
    }
  }
  return { rows: Array.from(map.values()), maxCount };
}

function filteredTypes() {
  const q = String(state.filter || "").trim().toLowerCase();
  let rows = state.typeStats.slice();
  if (q) rows = rows.filter((r) => String(r.name).toLowerCase().includes(q));
  if (state.sortMode === "name") {
    rows.sort((a, b) => String(a.name).localeCompare(String(b.name)));
  } else {
    rows.sort((a, b) => (b.count - a.count) || String(a.name).localeCompare(String(b.name)));
  }
  return rows;
}

function renderTypes() {
  const rows = filteredTypes();
  typesListEl.innerHTML = "";
  if (!rows.length) {
    const empty = document.createElement("div");
    empty.className = "mini";
    empty.textContent = state.itemsDraft.length ? "没有匹配的类型。" : "未加载数据。点击 Reload 从 API 拉取 items。";
    typesListEl.appendChild(empty);
    return;
  }

  const maxCount = Math.max(1, ...rows.map((r) => r.count || 1));

  for (const r of rows) {
    const row = document.createElement("div");
    const isActive = state.activeType === r.name;
    const isSelected = state.selectedTypes.has(r.name);
    row.className = `type-row${isActive ? " on" : ""}${isSelected ? " multi" : ""}`;

    const checkWrap = document.createElement("div");
    checkWrap.className = "type-check";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = isSelected;
    cb.addEventListener("click", (e) => e.stopPropagation());
    cb.addEventListener("change", () => {
      toggleTypeSelected(r.name, cb.checked);
    });
    checkWrap.appendChild(cb);

    const name = document.createElement("div");
    name.className = "type-name";
    const b = document.createElement("b");
    b.textContent = r.name;
    const s = document.createElement("span");
    s.textContent = r.sample.length ? r.sample.join("  ·  ") : "—";
    name.appendChild(b);
    name.appendChild(s);

    const meta = document.createElement("div");
    meta.className = "type-meta";
    const count = document.createElement("div");
    count.className = "type-count";
    count.textContent = `${r.count} item${r.count === 1 ? "" : "s"}`;
    const spark = document.createElement("div");
    spark.className = "spark";
    const fill = document.createElement("i");
    fill.style.width = `${Math.round(clamp(r.count / maxCount, 0, 1) * 100)}%`;
    spark.appendChild(fill);
    meta.appendChild(count);
    meta.appendChild(spark);

    row.appendChild(checkWrap);
    row.appendChild(name);
    row.appendChild(meta);

    row.addEventListener("click", () => {
      state.activeType = r.name;
      if (state.selectedTypes.size === 1 && state.selectedTypes.has(r.name)) {
        // no-op
      }
      renderAll();
    });

    typesListEl.appendChild(row);
  }
}

function renderFocus() {
  const active = state.activeType;
  activeTypeText.textContent = active || "—";

  const selected = Array.from(state.selectedTypes);
  if (!selected.length) {
    selectedTypesText.textContent = "—";
  } else if (selected.length <= 6) {
    selectedTypesText.textContent = selected.join(" · ");
  } else {
    selectedTypesText.textContent = `${selected.slice(0, 6).join(" · ")} · +${selected.length - 6}`;
  }

  if (!state.itemsDraft.length) {
    statsText.textContent = "—";
    itemsSampleEl.innerHTML = "";
    return;
  }

  const stats = active ? state.typeStats.find((x) => x.name === active) : null;
  const itemCount = state.itemsDraft.length;
  const typeCount = state.typeStats.length;
  statsText.textContent = active && stats ? `type_items=${stats.count} · all_items=${itemCount} · types=${typeCount}` : `all_items=${itemCount} · types=${typeCount}`;

  itemsSampleEl.innerHTML = "";
  if (!active || !stats) {
    const tip = document.createElement("div");
    tip.className = "mini";
    tip.textContent = "点击左侧某个类型查看包含该类型的物品样例。";
    itemsSampleEl.appendChild(tip);
    return;
  }

  const idxs = stats.itemIdx || [];
  const max = 20;
  const show = idxs.slice(0, max);
  for (const i of show) {
    const it = state.itemsDraft[i];
    if (!it) continue;
    const card = document.createElement("div");
    card.className = "sample";

    const top = document.createElement("div");
    top.className = "sample-top";
    const title = document.createElement("b");
    title.textContent = String(it?.name || "—");
    const id = document.createElement("code");
    id.textContent = getItemId(it) || "—";
    top.appendChild(title);
    top.appendChild(id);

    const sub = document.createElement("div");
    sub.className = "sample-sub";
    const types = getTypes(it);
    sub.textContent = `types: ${types.length ? types.join(" / ") : "—"}`;

    card.appendChild(top);
    card.appendChild(sub);
    itemsSampleEl.appendChild(card);
  }

  if (idxs.length > max) {
    const more = document.createElement("div");
    more.className = "mini";
    more.textContent = `仅展示前 ${max} 条，共 ${idxs.length} 条。`;
    itemsSampleEl.appendChild(more);
  }
}

function renderActionHints() {
  const selected = Array.from(state.selectedTypes);
  mergeFromListEl.textContent = selected.length ? selected.join(" · ") : "—";
  removeListEl.textContent = selected.length ? selected.join(" · ") : "—";

  if (selected.length === 1) {
    const only = selected[0];
    if (!renameFromEl.value.trim()) renameFromEl.value = only;
  }
}

function changedItems() {
  const orig = new Map();
  for (const it of state.itemsOriginal) {
    const id = getItemId(it);
    if (!id) continue;
    orig.set(id, it);
  }

  const out = [];
  for (const it of state.itemsDraft) {
    const id = getItemId(it);
    if (!id) continue;
    const old = orig.get(id);
    if (!old) continue;
    if (typeKey(old) !== typeKey(it)) out.push(it);
  }
  return out;
}

function renderChanges() {
  const meta = state.meta;
  catalogVersionEl.textContent = meta ? `v${meta.catalog_version ?? "—"}` : "—";
  const idxV = state.status?.index?.catalog_version;
  indexVersionEl.textContent = idxV != null ? `v${idxV}` : "—";
  changedCountEl.textContent = String(changedItems().length);
}

function setDataPill(ok, text) {
  setDot(dataDot, ok ? "ok" : "warn");
  dataText.textContent = text;
}

function toggleTypeSelected(typeName, checked) {
  const name = String(typeName || "").trim();
  if (!name) return;
  if (checked) state.selectedTypes.add(name);
  else state.selectedTypes.delete(name);

  if (state.selectedTypes.size === 1) {
    const only = Array.from(state.selectedTypes)[0];
    renameFromEl.value = only;
  }
  renderAll();
}

function selectAllFilteredTypes() {
  for (const r of filteredTypes()) state.selectedTypes.add(r.name);
  renderAll();
}

function clearSelectedTypes() {
  state.selectedTypes.clear();
  renderAll();
}

function applyRename() {
  const from = String(renameFromEl.value || "").trim();
  const to = String(renameToEl.value || "").trim();
  if (!from) {
    log("Rename: missing From");
    return;
  }
  if (!state.itemsDraft.length) {
    log("Rename: no items loaded");
    return;
  }

  const removing = !to;
  if (removing) {
    const ok = confirm(`将从所有物品中移除类型：${from} ？`);
    if (!ok) return;
  }

  let touched = 0;
  for (const it of state.itemsDraft) {
    const types = getTypes(it);
    if (!types.includes(from)) continue;
    const next = [];
    for (const t of types) {
      if (t === from) {
        if (!removing) next.push(to);
      } else next.push(t);
    }
    setTypes(it, next);
    touched += 1;
  }

  if (state.activeType === from) state.activeType = removing ? null : to;
  if (state.selectedTypes.has(from)) {
    state.selectedTypes.delete(from);
    if (!removing) state.selectedTypes.add(to);
  }

  log(`Rename: ${from} -> ${removing ? "(removed)" : to} · touched_items=${touched}`);
  rebuildDraftStats();
}

function applyMerge() {
  const froms = Array.from(state.selectedTypes);
  const target = String(mergeToEl.value || "").trim();
  if (!froms.length) {
    log("Merge: select types first");
    return;
  }
  if (!target) {
    log("Merge: missing Target");
    return;
  }
  if (!state.itemsDraft.length) {
    log("Merge: no items loaded");
    return;
  }

  const fromSet = new Set(froms);
  let touched = 0;
  for (const it of state.itemsDraft) {
    const types = getTypes(it);
    if (!types.some((t) => fromSet.has(t))) continue;
    const next = [];
    for (const t of types) {
      if (fromSet.has(t)) next.push(target);
      else next.push(t);
    }
    setTypes(it, next);
    touched += 1;
  }

  state.activeType = target;
  state.selectedTypes = new Set([target]);
  renameFromEl.value = target;
  log(`Merge: ${froms.join(" · ")} -> ${target} · touched_items=${touched}`);
  rebuildDraftStats();
}

function applyRemove() {
  const froms = Array.from(state.selectedTypes);
  if (!froms.length) {
    log("Remove: select types first");
    return;
  }
  if (!state.itemsDraft.length) {
    log("Remove: no items loaded");
    return;
  }

  const ok = confirm(`将从 Draft 中移除 ${froms.length} 个类型：\n\n${froms.join(" · ")}\n\n继续？`);
  if (!ok) return;

  const fromSet = new Set(froms);
  let touched = 0;
  for (const it of state.itemsDraft) {
    const types = getTypes(it);
    if (!types.some((t) => fromSet.has(t))) continue;
    setTypes(it, types.filter((t) => !fromSet.has(t)));
    touched += 1;
  }

  if (state.activeType && fromSet.has(state.activeType)) state.activeType = null;
  state.selectedTypes.clear();

  log(`Remove: removed_types=${froms.length} · touched_items=${touched}`);
  rebuildDraftStats();
}

function applyAdd() {
  const t = String(addTypeEl.value || "").trim();
  const kw = String(addKeywordEl.value || "").trim().toLowerCase();
  const scopeSelected = Boolean(addScopeSelectedEl.checked);
  if (!t) {
    log("Add: missing Type");
    return;
  }
  if (!state.itemsDraft.length) {
    log("Add: no items loaded");
    return;
  }

  const selected = Array.from(state.selectedTypes);
  const selectedSet = new Set(selected);
  let touched = 0;
  for (const it of state.itemsDraft) {
    if (kw) {
      const id = getItemId(it).toLowerCase();
      const name = String(it?.name || "").toLowerCase();
      if (!id.includes(kw) && !name.includes(kw)) continue;
    }
    if (scopeSelected && selected.length) {
      const types = getTypes(it);
      if (!types.some((x) => selectedSet.has(x))) continue;
    }

    const next = getTypes(it);
    if (next.includes(t)) continue;
    next.push(t);
    setTypes(it, next);
    touched += 1;
  }

  log(`Add: +${t} · touched_items=${touched}`);
  rebuildDraftStats();
}

function rebuildDraftStats() {
  const { rows } = computeTypeStats(state.itemsDraft);
  state.typeStats = rows;
  renderAll();
}

function resetDraft() {
  if (!state.itemsOriginal.length) return;
  state.itemsDraft = clone(state.itemsOriginal);
  state.activeType = null;
  state.selectedTypes.clear();
  renameFromEl.value = "";
  renameToEl.value = "";
  mergeToEl.value = "";
  addTypeEl.value = "";
  addKeywordEl.value = "";
  clearLog();
  log("Reset: draft restored from last loaded snapshot");
  rebuildDraftStats();
}

function toImportItem(it) {
  return {
    id: getItemId(it),
    name: String(it?.name || ""),
    type: it?.type ?? null,
    aliases: Array.isArray(it?.aliases) ? it.aliases : [],
    desc_labels: Array.isArray(it?.desc_labels) ? it.desc_labels : Array.isArray(it?.labels) ? it.labels : [],
    attrs: it?.attrs && typeof it.attrs === "object" ? it.attrs : {},
    status: String(it?.status || "active"),
  };
}

function download(filename, text, contentType = "application/json;charset=utf-8") {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: contentType }));
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

function downloadPatch() {
  const changed = changedItems();
  const payload = {
    generated_at: new Date().toISOString(),
    base_catalog_version: state.meta?.catalog_version ?? null,
    changed_items: changed.length,
    items: changed.map(toImportItem),
  };
  const name = `types_patch_${Date.now()}.json`;
  download(name, JSON.stringify(payload, null, 2));
  log(`Patch: downloaded ${name} (items=${changed.length})`);
}

async function applyToApi() {
  const base = String(apiBaseEl.value || "").trim();
  if (!base) return;

  const changed = changedItems();
  if (!changed.length) {
    log("Upsert: no changes");
    return;
  }

  const rebuild = Boolean(rebuildToggleEl.checked);
  const ok = confirm(`将 upsert ${changed.length} 个物品到 API${rebuild ? " 并重建索引" : ""}。\n\n继续？`);
  if (!ok) return;

  const url = apiUrl(base, `/v1/items/import?rebuild=${rebuild ? "true" : "false"}&mode=upsert`);
  const payload = { enable_bm25: Boolean(state.enableBm25), items: changed.map(toImportItem) };

  try {
    const res = await fetchJson(url, {
      method: "POST",
      headers: headersJson(true),
      body: JSON.stringify(payload),
    });
    log(`Upsert: ok · created=${res?.result?.created ?? 0} · updated=${res?.result?.updated ?? 0}`);
    if (res?.index) log(`Index: catalog_version=${res.index.catalog_version} · items=${res.index.item_count} · build_ms=${Number(res.index.build_ms ?? 0).toFixed(1)}`);
    await loadFromApi({ keepFilter: true });
  } catch (err) {
    const msg = String(err?.message || err || "upsert failed");
    log(`Upsert: failed · ${msg}`);
    if (Number(err?.status) === 401) {
      apiKeyEl.focus();
      apiKeyEl.select?.();
    }
  }
}

async function ping() {
  const base = String(apiBaseEl.value || "").trim();
  if (!base) return;
  saveStorage(STORAGE.apiBase, base);

  setDot(apiDot, "warn");
  apiText.textContent = "…";

  try {
    const h = await fetchJson(apiUrl(base, "/health"), { method: "GET" });
    setDot(apiDot, "ok");
    apiText.textContent = `ok · ${h.model || "model"} · ${h.device || "cpu"} · ${h.storage_backend || "store"}`;
  } catch (err) {
    setDot(apiDot, "bad");
    apiText.textContent = String(err?.message || err || "offline");
    return;
  }

  try {
    const s = await fetchJson(apiUrl(base, "/v1/status"), { method: "GET" });
    state.status = s;
    state.enableBm25 = s?.index?.enable_bm25 ?? true;
    renderChanges();
  } catch {
    // ignore
  }
}

async function loadFromApi({ keepFilter } = { keepFilter: false }) {
  const base = String(apiBaseEl.value || "").trim();
  if (!base) return;
  saveStorage(STORAGE.apiBase, base);

  const status = String(statusFilterEl.value || "").trim();
  const q = status ? `?status=${encodeURIComponent(status)}` : "";

  setDataPill(false, "加载中…");
  clearLog();
  log(`Load: /v1/items/export${q || ""}`);

  try {
    const data = await fetchJson(apiUrl(base, `/v1/items/export${q}`), { method: "GET" });
    const items = Array.isArray(data?.items) ? data.items : [];
    state.meta = data?.meta || null;
    state.itemsOriginal = clone(items);
    state.itemsDraft = clone(items);
    state.activeType = null;
    state.selectedTypes.clear();
    if (!keepFilter) {
      renameFromEl.value = "";
      renameToEl.value = "";
      mergeToEl.value = "";
      addTypeEl.value = "";
      addKeywordEl.value = "";
    }

    const { rows } = computeTypeStats(state.itemsDraft);
    state.typeStats = rows;
    setDataPill(true, `已加载：items=${items.length} · types=${rows.length} · catalog=v${state.meta?.catalog_version ?? "—"}`);
    await ping();
    renderAll();
  } catch (err) {
    setDataPill(false, `加载失败：${String(err?.message || err || "error")}`);
    log(`Load: failed · ${String(err?.message || err || "error")}`);
  }
}

function setTab(onId) {
  const tabs = [
    { id: "rename", el: tabRename, panel: panelRename },
    { id: "merge", el: tabMerge, panel: panelMerge },
    { id: "remove", el: tabRemove, panel: panelRemove },
    { id: "add", el: tabAdd, panel: panelAdd },
  ];
  for (const t of tabs) {
    const on = t.id === onId;
    t.el.classList.toggle("on", on);
    t.el.setAttribute("aria-selected", on ? "true" : "false");
    t.panel.classList.toggle("hidden", !on);
  }
}

function renderAll() {
  renderTypes();
  renderFocus();
  renderActionHints();
  renderChanges();
}

function bindEvents() {
  pingBtn.addEventListener("click", ping);
  reloadBtn.addEventListener("click", () => loadFromApi());
  resetBtn.addEventListener("click", resetDraft);

  apiBaseEl.addEventListener("change", () => {
    saveStorage(STORAGE.apiBase, apiBaseEl.value.trim());
  });
  apiKeyEl.addEventListener("change", () => {
    saveStorage(STORAGE.apiKey, apiKeyEl.value);
  });

  typeFilterEl.addEventListener("input", () => {
    state.filter = typeFilterEl.value;
    saveStorage(STORAGE.typeFilter, state.filter);
    renderTypes();
  });

  sortModeEl.addEventListener("change", () => {
    state.sortMode = String(sortModeEl.value || "count");
    saveStorage(STORAGE.sortMode, state.sortMode);
    renderTypes();
  });

  statusFilterEl.addEventListener("change", () => {
    state.statusFilter = String(statusFilterEl.value || "");
    saveStorage(STORAGE.statusFilter, state.statusFilter);
  });

  tabRename.addEventListener("click", () => setTab("rename"));
  tabMerge.addEventListener("click", () => setTab("merge"));
  tabRemove.addEventListener("click", () => setTab("remove"));
  tabAdd.addEventListener("click", () => setTab("add"));

  renameBtn.addEventListener("click", applyRename);
  mergeBtn.addEventListener("click", applyMerge);
  removeBtn.addEventListener("click", applyRemove);
  addBtn.addEventListener("click", applyAdd);

  selectAllTypesBtn.addEventListener("click", selectAllFilteredTypes);
  selectNoneTypesBtn.addEventListener("click", clearSelectedTypes);

  rebuildToggleEl.addEventListener("change", () => {
    saveStorage(STORAGE.rebuild, rebuildToggleEl.checked ? "1" : "0");
  });

  downloadPatchBtn.addEventListener("click", downloadPatch);
  applyBtn.addEventListener("click", applyToApi);
}

function init() {
  apiBaseEl.value = loadStorage(STORAGE.apiBase, "http://127.0.0.1:8000");
  apiKeyEl.value = loadStorage(STORAGE.apiKey, "");

  state.statusFilter = loadStorage(STORAGE.statusFilter, "");
  statusFilterEl.value = state.statusFilter;

  state.sortMode = loadStorage(STORAGE.sortMode, "count");
  sortModeEl.value = state.sortMode;

  state.filter = loadStorage(STORAGE.typeFilter, "");
  typeFilterEl.value = state.filter;

  rebuildToggleEl.checked = loadStorage(STORAGE.rebuild, "1") !== "0";

  setTab("rename");
  bindEvents();
  ping();
  loadFromApi({ keepFilter: true });
}

init();

