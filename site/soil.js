// Core logic for Alien Soil Doctor, ported from main/UI.py and main/Soil_main.py.
// No DOM access here so it can be tested in Node as well as used in the browser.

export const REQUIRED_COLUMNS = [
  "lat", "lon", "depth", "phh2o", "soc", "clay", "sand", "silt", "cec",
  "bdod", "nitrogen", "radiation_level", "alien_chemical_index",
  "reward_score", "confidence",
];

export const LABEL_MAP = { not_habitable: 0, moderate: 1, habitat: 2 };
export const REVERSE_MAP = ["not_habitable", "moderate", "habitat"];

// ---------- CSV ----------

export function parseCSV(text) {
  const rows = [];
  let row = [], field = "", quoted = false;
  text = text.replace(/^﻿/, "");
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; } else quoted = false;
      } else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ",") { row.push(field); field = ""; }
    else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field); field = "";
      if (row.some((v) => v.trim() !== "")) rows.push(row);
      row = [];
    } else field += c;
  }
  row.push(field);
  if (row.some((v) => v.trim() !== "")) rows.push(row);
  if (rows.length === 0) return { headers: [], records: [] };
  const headers = rows[0].map((h) => h.trim());
  const records = rows.slice(1).map((r) => {
    const o = {};
    headers.forEach((h, j) => { o[h] = (r[j] ?? "").trim(); });
    return o;
  });
  return { headers, records };
}

function csvCell(v) {
  const s = String(v ?? "");
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export function toCSV(headers, records) {
  return [headers.map(csvCell).join(","), ...records.map((r) => headers.map((h) => csvCell(r[h])).join(","))].join("\n");
}

// ---------- Feature preparation ----------

// Depth may be numeric or a range like "0-100cm"; like pandas str.extract(r"(\d+)") take the first number.
function parseDepth(v) {
  const m = String(v).match(/\d+/);
  return m ? Number(m[0]) : NaN;
}

export function buildFeatures(headers, records) {
  const missingColumns = REQUIRED_COLUMNS.filter((c) => !headers.includes(c));
  const depthIsText = headers.includes("depth") &&
    records.some((r) => r.depth !== "" && !Number.isFinite(Number(r.depth)));

  const X = records.map((r) => REQUIRED_COLUMNS.map((c) => {
    if (!headers.includes(c)) return 0; // auto-filled with 0.0, as in the Python scripts
    if (c === "depth" && depthIsText) return parseDepth(r.depth);
    return r[c] === "" ? NaN : Number(r[c]);
  }));

  // Fill blank / non-numeric cells with the column mean so one bad cell doesn't break the batch.
  let filledCells = 0;
  for (let j = 0; j < REQUIRED_COLUMNS.length; j++) {
    const vals = X.map((x) => x[j]).filter(Number.isFinite);
    const mean = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
    for (const x of X) if (!Number.isFinite(x[j])) { x[j] = mean; filledCells++; }
  }

  const hasLabels = headers.includes("habitat");
  const yTrue = hasLabels ? records.map((r) => LABEL_MAP[r.habitat?.toLowerCase()] ?? null) : null;
  return { X, yTrue, missingColumns, filledCells };
}

// sklearn StandardScaler().fit_transform: population std, zero-variance columns get scale 1.
export function standardize(X) {
  const n = X.length, d = X[0].length;
  const mean = Array(d).fill(0), std = Array(d).fill(0);
  for (const x of X) for (let j = 0; j < d; j++) mean[j] += x[j] / n;
  for (const x of X) for (let j = 0; j < d; j++) std[j] += (x[j] - mean[j]) ** 2 / n;
  for (let j = 0; j < d; j++) { std[j] = Math.sqrt(std[j]); if (std[j] < 1e-12) std[j] = 1; }
  return X.map((x) => x.map((v, j) => (v - mean[j]) / std[j]));
}

// ---------- Dueling DQN forward pass (same graph as the saved Keras model) ----------

function dense(x, layer, relu) {
  const { kernel, bias } = layer;
  const out = bias.slice();
  for (let i = 0; i < x.length; i++) {
    const xi = x[i];
    if (xi === 0) continue;
    const row = kernel[i];
    for (let k = 0; k < out.length; k++) out[k] += xi * row[k];
  }
  if (relu) for (let k = 0; k < out.length; k++) if (out[k] < 0) out[k] = 0;
  return out;
}

export function predictQ(model, x) {
  const L = model.layers;
  const h = dense(dense(x, L.dense, true), L.dense_1, true);
  const adv = dense(dense(h, L.dense_4, true), L.dense_5, false);
  const value = dense(dense(h, L.dense_2, true), L.dense_3, false)[0];
  const meanAdv = adv.reduce((a, b) => a + b, 0) / adv.length;
  return adv.map((a) => value + a - meanAdv);
}

// ---------- Suggestion engine (from UI.py) ----------

export function suggestImprovements(s) {
  const tips = [];
  if (s.phh2o < 6.0) tips.push({ icon: "🧪", text: "Raise pH: Add alkaline minerals (e.g., lime)." });
  else if (s.phh2o > 8.0) tips.push({ icon: "🧪", text: "Lower pH: Add sulfur compounds or organic matter." });
  if (s.nitrogen < 0.5) tips.push({ icon: "🌱", text: "Boost Nitrogen: Add nitrogen-fixing microbes or ammonium salts." });
  if (s.alien_chemical_index > 0.6) tips.push({ icon: "☢️", text: "Reduce alien toxicity: Isolate or chemically neutralize unknown compounds." });
  if (s.radiation_level > 0.8) tips.push({ icon: "🛡", text: "Shield Radiation: Add subsurface shielding or build protective domes." });
  if (s.soc < 0.3) tips.push({ icon: "🪵", text: "Increase Organic Carbon: Mix composted biomass or carbon-rich regolith." });
  if (!tips.length) tips.push({ icon: "✅", text: "Soil is close to optimal. Minor improvements may suffice." });
  return tips;
}

// ---------- Metrics (sklearn classification_report equivalent) ----------

export function classificationReport(yTrue, yPred) {
  const pairs = yTrue.map((t, i) => [t, yPred[i]]).filter(([t]) => t !== null);
  const classes = REVERSE_MAP.map((name, c) => {
    let tp = 0, fp = 0, fn = 0, support = 0;
    for (const [t, p] of pairs) {
      if (t === c) support++;
      if (t === c && p === c) tp++;
      else if (p === c) fp++;
      else if (t === c) fn++;
    }
    const precision = tp + fp ? tp / (tp + fp) : 0;
    const recall = tp + fn ? tp / (tp + fn) : 0;
    const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
    return { name, precision, recall, f1, support };
  });
  const total = pairs.length;
  const accuracy = total ? pairs.filter(([t, p]) => t === p).length / total : 0;
  return { classes, accuracy, total };
}

// ---------- Full pipeline ----------

export function analyze(model, headers, records) {
  const { X, yTrue, missingColumns, filledCells } = buildFeatures(headers, records);
  const Xs = standardize(X);
  const results = Xs.map((x, i) => {
    const q = predictQ(model, x);
    let pred = 0;
    for (let k = 1; k < q.length; k++) if (q[k] > q[pred]) pred = k;
    const sample = Object.fromEntries(REQUIRED_COLUMNS.map((c, j) => [c, X[i][j]]));
    return {
      index: i,
      q,
      pred,
      label: REVERSE_MAP[pred],
      confidence: q[pred],
      trueLabel: yTrue ? (yTrue[i] === null ? null : REVERSE_MAP[yTrue[i]]) : null,
      tips: pred !== 2 ? suggestImprovements(sample) : [],
    };
  });

  let report = null, highConfReport = null;
  if (yTrue && yTrue.some((t) => t !== null)) {
    const yPred = results.map((r) => r.pred);
    report = classificationReport(yTrue, yPred);
    const mask = results.map((r) => r.confidence >= 0.95);
    if (mask.some(Boolean)) {
      highConfReport = classificationReport(yTrue.filter((_, i) => mask[i]), yPred.filter((_, i) => mask[i]));
    }
  }

  const outHeaders = [...headers, ...missingColumns.filter((c) => !headers.includes(c)), "Predicted_Label", "Confidence"];
  const outRecords = records.map((r, i) => {
    const o = { ...r };
    for (const c of missingColumns) o[c] = "0.0";
    o.Predicted_Label = results[i].label;
    o.Confidence = results[i].confidence.toFixed(6);
    return o;
  });

  return { results, report, highConfReport, missingColumns, filledCells, csv: toCSV(outHeaders, outRecords) };
}
