import { analyze, parseCSV } from "./soil.js";

const $ = (id) => document.getElementById(id);
const PAGE = 50;
const CLASS_TEXT = { habitat: "Habitat", moderate: "Moderate", not_habitable: "Not habitable" };
const COLORS = { habitat: "var(--habitat)", moderate: "var(--moderate)", not_habitable: "var(--not)" };

const state = { data: null, fileName: "", analysis: null, filter: "all", shown: PAGE, models: {} };

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function setStatus(msg, kind = "") {
  const el = $("status");
  el.textContent = msg;
  el.className = `status ${kind}`;
}

async function loadModel(ep) {
  if (!state.models[ep]) {
    const res = await fetch(`models/ep${ep}.json`);
    if (!res.ok) throw new Error(`Could not load model checkpoint ep${ep}`);
    state.models[ep] = await res.json();
  }
  return state.models[ep];
}

async function init() {
  try {
    const idx = await (await fetch("models/index.json")).json();
    $("checkpoint").innerHTML = idx.episodes
      .map((ep) => `<option value="${ep}" ${ep === idx.default ? "selected" : ""}>Episode ${ep.toLocaleString()}</option>`)
      .join("");
    loadModel(idx.default).catch(() => {});
  } catch {
    setStatus("⚠️ Could not load model index.", "error");
  }
}

async function handleText(text, name) {
  const { headers, records } = parseCSV(text);
  if (!headers.length || !records.length) {
    setStatus("⚠️ That file has no data rows.", "error");
    return;
  }
  state.data = { headers, records };
  state.fileName = name;
  await run();
}

async function run() {
  if (!state.data) return;
  const ep = $("checkpoint").value;
  setStatus("🧠 Making predictions…");
  try {
    const model = await loadModel(ep);
    const t0 = performance.now();
    state.analysis = analyze(model, state.data.headers, state.data.records);
    const ms = performance.now() - t0;
    const notes = [`✅ ${state.data.records.length} samples from ${state.fileName} analysed with checkpoint ep${ep} in ${ms.toFixed(0)} ms.`];
    let kind = "";
    if (state.analysis.missingColumns.length) {
      notes.push(`⚠️ Missing columns auto-filled with 0.0: ${state.analysis.missingColumns.join(", ")}`);
      kind = "warn";
    }
    if (state.analysis.filledCells) {
      notes.push(`⚠️ ${state.analysis.filledCells} blank or non-numeric cells filled with the column mean.`);
      kind = "warn";
    }
    if (state.data.records.length < 5) {
      notes.push("⚠️ Features are standardized across the batch, so results are more reliable with more samples.");
      kind = "warn";
    }
    setStatus(notes.join("\n"), kind);
    state.shown = PAGE;
    render();
  } catch (err) {
    setStatus(`⚠️ ${err.message}`, "error");
  }
}

function render() {
  const { results, report, highConfReport } = state.analysis;
  $("results").classList.remove("hidden");

  const counts = { habitat: 0, moderate: 0, not_habitable: 0 };
  results.forEach((r) => counts[r.label]++);
  const n = results.length;
  const avgConf = results.reduce((a, r) => a + r.confidence, 0) / n;

  $("stats").innerHTML = [
    `<div class="stat"><div class="k">Samples</div><div class="v">${n}</div></div>`,
    ...Object.keys(counts).map(
      (k) => `<div class="stat ${k}"><div class="k">${CLASS_TEXT[k]}</div><div class="v">${counts[k]} <small>${((counts[k] / n) * 100).toFixed(0)}%</small></div></div>`,
    ),
    `<div class="stat"><div class="k">Avg confidence</div><div class="v">${avgConf.toFixed(2)}</div></div>`,
  ].join("");

  $("dist").innerHTML = Object.keys(counts)
    .filter((k) => counts[k])
    .map((k) => `<div style="flex:${counts[k]};background:${COLORS[k]}" title="${CLASS_TEXT[k]}: ${counts[k]}"></div>`)
    .join("");

  $("report").innerHTML = report
    ? reportTable(report, "Classification report (full dataset)") +
      (highConfReport
        ? reportTable(highConfReport, `Classification report (confidence ≥ 0.95, ${highConfReport.total} samples)`)
        : `<p class="status warn">⚠️ No samples with confidence ≥ 0.95.</p>`)
    : "";

  renderRows();
}

function reportTable(rep, caption) {
  const pct = (v) => v.toFixed(4);
  const rows = rep.classes
    .map((c) => `<tr><td>${c.name}</td><td class="num">${pct(c.precision)}</td><td class="num">${pct(c.recall)}</td><td class="num">${pct(c.f1)}</td><td class="num">${c.support}</td></tr>`)
    .join("");
  return `<div class="report table-wrap"><table><caption>${caption}</caption>
    <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
    <tbody>${rows}<tr><td><strong>accuracy</strong></td><td></td><td></td><td class="num"><strong>${pct(rep.accuracy)}</strong></td><td class="num">${rep.total}</td></tr></tbody></table></div>`;
}

function renderRows() {
  const { results } = state.analysis;
  const rec = state.data.records;
  const list = state.filter === "all" ? results : results.filter((r) => r.label === state.filter);
  const fmt = (v) => (v === undefined || v === "" ? "–" : esc(v));

  $("rows").innerHTML = list.slice(0, state.shown).map((r) => {
    const s = rec[r.index];
    const verdict = r.pred === 2 ? "✅ HABITABLE" : "❌ NOT HABITABLE";
    const truth = r.trueLabel
      ? `<span class="true ${r.trueLabel !== r.label ? "miss" : ""}">true: ${r.trueLabel}</span>`
      : "";
    const tips = r.tips.length
      ? `<ul class="tips">${r.tips.map((t) => `<li>${t.icon} ${esc(t.text)}</li>`).join("")}</ul>`
      : `<span class="ok">✅ No action needed</span>`;
    return `<tr>
      <td class="num">${r.index + 1}</td>
      <td><span class="badge ${r.label}">${r.label}</span><span class="true">${verdict}</span>${truth}</td>
      <td class="num">${r.confidence.toFixed(2)}</td>
      <td class="readings">pH ${fmt(s.phh2o)} · N ${fmt(s.nitrogen)} · SOC ${fmt(s.soc)}<br/>rad ${fmt(s.radiation_level)} · chem ${fmt(s.alien_chemical_index)}</td>
      <td>${tips}</td>
    </tr>`;
  }).join("") || `<tr><td colspan="5" class="readings">No samples in this category.</td></tr>`;

  $("more").classList.toggle("hidden", list.length <= state.shown);
  $("more").textContent = `Show more (${list.length - state.shown} remaining)`;
}

function readFile(file) {
  if (!file) return;
  if (file.size > 200 * 1024 * 1024) {
    setStatus("⚠️ Limit 200MB per file.", "error");
    return;
  }
  setStatus("📦 Loading dataset…");
  file.text().then((t) => handleText(t, file.name));
}

$("file").addEventListener("change", (e) => readFile(e.target.files[0]));
const drop = $("drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => readFile(e.dataTransfer.files[0]));

$("sample").addEventListener("click", async () => {
  setStatus("📦 Loading sample dataset…");
  const text = await (await fetch("sample_mars_soil.csv")).text();
  handleText(text, "sample_mars_soil.csv");
});

$("checkpoint").addEventListener("change", run);

document.querySelectorAll(".chip").forEach((chip) =>
  chip.addEventListener("click", () => {
    document.querySelectorAll(".chip").forEach((c) => c.classList.toggle("active", c === chip));
    state.filter = chip.dataset.filter;
    state.shown = PAGE;
    renderRows();
  }),
);

$("more").addEventListener("click", () => { state.shown += PAGE; renderRows(); });

$("download").addEventListener("click", () => {
  const blob = new Blob([state.analysis.csv], { type: "text/csv" });
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: "alien_soil_predictions.csv" });
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

init();
