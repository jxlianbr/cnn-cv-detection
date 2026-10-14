"""Write the local review page for the annotation pre-labels.

Reads reports/review/annotation/prelabels.csv (AI-proposed, unverified labels)
and writes reports/review/annotation/review.html, a single self-contained page
that shows one chip at a time with the proposed label pre-selected. The author
accepts or changes each field, and exports verified_labels.csv from the page.

The data are embedded in the page, so it opens straight from disk with no
server and makes no network requests; chips are referenced as chips/{id}.png.

    python scripts/make_annotation_review_page.py
"""
import argparse
import json
from pathlib import Path

import pandas as pd

FIELDS = {
    "pv_present": ["yes", "no", "unclear"],
    "outline_match": ["good", "partial", "poor", "n/a"],
    "failure_case": ["none", "non_crystalline_silicon", "spectrally_ambiguous", "both", "unclear"],
    "polygon_fragment": ["yes", "no", "unclear"],
    "confidence": ["high", "medium", "low"],
}

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PV Annotation Review</title>
<style>
:root { --bg:#f7f7f5; --fg:#1d1d1b; --muted:#6b6b66; --card:#fff; --line:#d9d9d4;
        --accent:#1f5fbf; --ok:#2e7d32; --warn:#b26a00; --sel:#e3ecfa; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#161615; --fg:#ececea; --muted:#a3a39e; --card:#21211f; --line:#3a3a37;
          --accent:#7aa7ec; --ok:#7cc47f; --warn:#e3a24a; --sel:#26354d; }
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
       font:14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { position:sticky; top:0; z-index:2; background:var(--card); border-bottom:1px solid var(--line);
         padding:8px 16px; display:flex; flex-wrap:wrap; gap:12px; align-items:center; }
header h1 { font-size:15px; margin:0 12px 0 0; }
.bar { flex:1 1 200px; height:8px; background:var(--line); border-radius:4px; overflow:hidden; min-width:120px; }
.bar > div { height:100%; background:var(--ok); width:0; }
button, select, input[type=text] { font:inherit; color:var(--fg); background:var(--card);
         border:1px solid var(--line); border-radius:6px; padding:4px 10px; }
button:hover { border-color:var(--accent); }
button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
main { padding:12px 16px 40px; max-width:1500px; margin:0 auto; }
.warn { background:var(--card); border-left:4px solid var(--warn); padding:6px 10px; margin-bottom:10px; }
.chip { width:100%; border:1px solid var(--line); background:#fff; cursor:zoom-in; display:block; }
.grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap:10px; margin-top:10px; }
fieldset { border:1px solid var(--line); border-radius:8px; background:var(--card); margin:0; padding:6px 10px 8px; }
legend { color:var(--muted); font-size:12px; padding:0 4px; }
label.opt { display:inline-flex; align-items:center; gap:4px; margin:3px 8px 3px 0; padding:2px 6px;
            border-radius:5px; cursor:pointer; }
label.opt.proposed { outline:1px dashed var(--muted); }
label.opt:has(input:checked) { background:var(--sel); }
kbd { font:11px ui-monospace, monospace; border:1px solid var(--line); border-bottom-width:2px;
      border-radius:3px; padding:0 3px; color:var(--muted); }
.meta { display:flex; flex-wrap:wrap; gap:6px 18px; margin:8px 0; color:var(--muted); }
.meta b { color:var(--fg); font-weight:600; }
.status { font-weight:600; }
.status.v { color:var(--ok); } .status.u { color:var(--warn); }
textarea { width:100%; min-height:44px; font:inherit; color:var(--fg); background:var(--card);
           border:1px solid var(--line); border-radius:6px; padding:6px; }
.reason { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:6px 10px; margin-top:10px; }
.help { color:var(--muted); font-size:12px; margin-top:14px; }
.changed { color:var(--warn); font-size:12px; }
</style>
</head>
<body>
<header>
  <h1>PV annotation review</h1>
  <button id="prev" title="Previous (Left arrow / K)">&larr; Prev</button>
  <span id="pos"></span>
  <button id="next" title="Next (Right arrow / J)">Next &rarr;</button>
  <button id="nextU" title="Next unverified (.)">Next unverified</button>
  <select id="jump" title="Jump to point"></select>
  <div class="bar"><div id="barfill"></div></div>
  <span id="prog"></span>
  <button id="export" class="primary">Export verified_labels.csv</button>
  <label><input type="file" id="import" accept=".csv" style="display:none">
    <span style="cursor:pointer;text-decoration:underline">Import CSV</span></label>
</header>
<main>
  <div class="warn">The selections shown first are <b>AI-proposed pre-labels</b>, not ground truth.
    A point counts as verified only after you confirm it (Enter). Progress is kept in this browser
    (localStorage); export the CSV regularly as a backup.</div>
  <div class="meta" id="meta"></div>
  <a id="chiplink" target="_blank" rel="noopener"><img id="chip" class="chip" alt="chip"></a>
  <div class="reason" id="reason"></div>
  <div class="grid" id="fields"></div>
  <div style="margin-top:10px">
    <label for="note" style="color:var(--muted);font-size:12px">Author note (<kbd>N</kbd> to focus, <kbd>Esc</kbd> to leave)</label>
    <textarea id="note"></textarea>
  </div>
  <div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap;align-items:center">
    <button id="verify" class="primary">Confirm &amp; next (Enter)</button>
    <button id="unverify">Mark unverified (U)</button>
    <button id="reset">Reset to proposal (R)</button>
    <span id="status" class="status"></span> <span id="changed" class="changed"></span>
  </div>
  <div class="help">
    Shortcuts: <kbd>&larr;</kbd>/<kbd>K</kbd> previous, <kbd>&rarr;</kbd>/<kbd>J</kbd> next, <kbd>.</kbd> next unverified,
    <kbd>Enter</kbd> confirm &amp; next, <kbd>U</kbd> unverify, <kbd>R</kbd> reset to proposal, <kbd>N</kbd> note.<br>
    PV present <kbd>1</kbd> yes <kbd>2</kbd> no <kbd>3</kbd> unclear &middot;
    Outline <kbd>Q</kbd> good <kbd>W</kbd> partial <kbd>E</kbd> poor <kbd>T</kbd> n/a &middot;
    Failure case <kbd>A</kbd> none <kbd>S</kbd> non-c-Si <kbd>D</kbd> spectrally ambiguous <kbd>F</kbd> both <kbd>G</kbd> unclear &middot;
    Polygon fragment <kbd>Z</kbd> yes <kbd>X</kbd> no <kbd>C</kbd> unclear &middot;
    Confidence <kbd>7</kbd> high <kbd>8</kbd> medium <kbd>9</kbd> low.<br>
    Decision rules: see PROTOCOL.md in this folder. Imagery: Source: GSI seamless photo (国土地理院).
  </div>
</main>
<script>
const DATA = __DATA__;
const COLUMNS = __COLUMNS__;
const FIELDS = __FIELDS__;
const KEYS = {
  pv_present: {"1":"yes","2":"no","3":"unclear"},
  outline_match: {"q":"good","w":"partial","e":"poor","t":"n/a"},
  failure_case: {"a":"none","s":"non_crystalline_silicon","d":"spectrally_ambiguous","f":"both","g":"unclear"},
  polygon_fragment: {"z":"yes","x":"no","c":"unclear"},
  confidence: {"7":"high","8":"medium","9":"low"},
};
const STORE = "pv_annotation_review_v1";
let state = {};   // point_id -> {values:{field:val}, note, verified}
let idx = 0;

function load() {
  try { const raw = localStorage.getItem(STORE); if (raw) { const s = JSON.parse(raw); state = s.state || {}; idx = s.idx || 0; } }
  catch (e) { state = {}; }
  if (idx >= DATA.length) idx = 0;
}
function save() {
  try { localStorage.setItem(STORE, JSON.stringify({state, idx})); } catch (e) {}
}
function entry(row) {
  const id = String(row.point_id);
  if (!state[id]) {
    const values = {};
    for (const f in FIELDS) values[f] = row[f];
    state[id] = {values, note: "", verified: false};
  }
  return state[id];
}
function changedFields(row) {
  const e = entry(row);
  return Object.keys(FIELDS).filter(f => e.values[f] !== row[f]);
}
function render() {
  const row = DATA[idx], e = entry(row);
  document.getElementById("pos").textContent = `${idx + 1} / ${DATA.length}`;
  document.getElementById("jump").value = String(idx);
  const src = `chips/${row.point_id}.png`;
  document.getElementById("chip").src = src;
  document.getElementById("chiplink").href = src;
  const m = [
    ["point", row.point_id], ["split", row.split], ["est. capacity", `${Math.round(row.estimated_kw)} kW`],
    ["size bin", row.size_bin], ["sub-50 kW", row.sub_50kw], ["edge dist.", `${row.edge_distance_m} m`],
    ["XGBoost", row.xgb_pred], ["CNN majority", row.cnn_majority],
    ["map", `<a href="https://maps.gsi.go.jp/#18/${row.lat}/${row.lon}/&base=std&ls=seamlessphoto" target="_blank" rel="noopener">GSI Maps</a>`],
  ];
  document.getElementById("meta").innerHTML = m.map(([k, v]) => `<span>${k}: <b>${v}</b></span>`).join("");
  document.getElementById("reason").innerHTML =
    `<b>AI pre-label reason</b> (confidence ${row.confidence}): ${escapeHtml(row.reason)}`;
  const box = document.getElementById("fields");
  box.innerHTML = "";
  for (const f in FIELDS) {
    const fs = document.createElement("fieldset");
    const keyOf = Object.fromEntries(Object.entries(KEYS[f]).map(([k, v]) => [v, k]));
    fs.innerHTML = `<legend>${f.replace(/_/g, " ")}</legend>` + FIELDS[f].map(v =>
      `<label class="opt${v === row[f] ? " proposed" : ""}" title="${v === row[f] ? "AI proposal" : ""}">` +
      `<input type="radio" name="${f}" value="${v}" ${e.values[f] === v ? "checked" : ""}> ${v.replace(/_/g, " ")}` +
      ` <kbd>${(keyOf[v] || "").toUpperCase()}</kbd></label>`).join("");
    fs.addEventListener("change", ev => { setField(f, ev.target.value); });
    box.appendChild(fs);
  }
  document.getElementById("note").value = e.note;
  const st = document.getElementById("status");
  st.textContent = e.verified ? "verified" : "not verified";
  st.className = "status " + (e.verified ? "v" : "u");
  const ch = changedFields(row);
  document.getElementById("changed").textContent = ch.length ? `changed from proposal: ${ch.join(", ")}` : "";
  const nv = DATA.filter(r => state[String(r.point_id)] && state[String(r.point_id)].verified).length;
  document.getElementById("prog").textContent = `${nv} / ${DATA.length} verified`;
  document.getElementById("barfill").style.width = `${100 * nv / DATA.length}%`;
  save();
}
function escapeHtml(s) { return String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
function setField(f, v) { entry(DATA[idx]).values[f] = v; render(); }
function go(i) { idx = Math.max(0, Math.min(DATA.length - 1, i)); render(); window.scrollTo(0, 0); }
function nextUnverified() {
  for (let k = 1; k <= DATA.length; k++) {
    const j = (idx + k) % DATA.length;
    if (!entry(DATA[j]).verified) { go(j); return; }
  }
}
function verify() { entry(DATA[idx]).verified = true; save(); if (idx < DATA.length - 1) go(idx + 1); else render(); }

function csvCell(v) {
  const s = v === null || v === undefined ? "" : String(v);
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}
function exportCsv() {
  const header = [...COLUMNS, "verified", "changed_by_author", "changed_fields", "author_note"];
  const lines = [header.join(",")];
  for (const row of DATA) {
    const e = entry(row), ch = changedFields(row), out = {...row, ...e.values};
    // keep the worksheet's two True/False columns in step with failure_case
    const fc = out.failure_case;
    out.non_crystalline_silicon = fc === "unclear" ? "" : ((fc === "non_crystalline_silicon" || fc === "both") ? "True" : "False");
    out.spectrally_ambiguous = fc === "unclear" ? "" : ((fc === "spectrally_ambiguous" || fc === "both") ? "True" : "False");
    out.label_source = e.verified ? (ch.length ? "author (changed AI pre-label)" : "author (confirmed AI pre-label)") : "AI pre-label (unverified)";
    const vals = COLUMNS.map(c => out[c]);
    vals.push(e.verified ? "true" : "false", ch.length ? "true" : "false", ch.join(";"), e.note);
    lines.push(vals.map(csvCell).join(","));
  }
  const blob = new Blob(["﻿" + lines.join("\r\n") + "\r\n"], {type: "text/csv;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "verified_labels.csv";
  document.body.appendChild(a); a.click(); a.remove();
}
function parseCsv(text) {
  const rows = []; let row = [], cell = "", q = false;
  text = text.replace(/^﻿/, "");
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) { if (c === '"') { if (text[i + 1] === '"') { cell += '"'; i++; } else q = false; } else cell += c; }
    else if (c === '"') q = true;
    else if (c === ",") { row.push(cell); cell = ""; }
    else if (c === "\n" || c === "\r") { if (c === "\r" && text[i + 1] === "\n") i++; row.push(cell); rows.push(row); row = []; cell = ""; }
    else cell += c;
  }
  if (cell !== "" || row.length) { row.push(cell); rows.push(row); }
  return rows.filter(r => r.length > 1);
}
function importCsv(file) {
  const reader = new FileReader();
  reader.onload = () => {
    const rows = parseCsv(reader.result), h = rows[0], n = {};
    h.forEach((c, i) => n[c] = i);
    if (n.point_id === undefined || n.verified === undefined) { alert("Not a verified_labels.csv export."); return; }
    let k = 0;
    for (const r of rows.slice(1)) {
      const id = r[n.point_id], row = DATA.find(d => String(d.point_id) === id);
      if (!row) continue;
      const e = entry(row);
      for (const f in FIELDS) if (n[f] !== undefined && FIELDS[f].includes(r[n[f]])) e.values[f] = r[n[f]];
      e.verified = r[n.verified] === "true";
      e.note = n.author_note !== undefined ? r[n.author_note] : "";
      k++;
    }
    render(); alert(`Imported ${k} rows.`);
  };
  reader.readAsText(file);
}

document.getElementById("prev").onclick = () => go(idx - 1);
document.getElementById("next").onclick = () => go(idx + 1);
document.getElementById("nextU").onclick = nextUnverified;
document.getElementById("verify").onclick = verify;
document.getElementById("unverify").onclick = () => { entry(DATA[idx]).verified = false; render(); };
document.getElementById("reset").onclick = () => { delete state[String(DATA[idx].point_id)]; render(); };
document.getElementById("export").onclick = exportCsv;
document.getElementById("import").onchange = ev => { if (ev.target.files[0]) importCsv(ev.target.files[0]); ev.target.value = ""; };
document.getElementById("note").addEventListener("input", ev => { entry(DATA[idx]).note = ev.target.value; save(); });
const jump = document.getElementById("jump");
jump.innerHTML = DATA.map((r, i) => `<option value="${i}">${i + 1}: point ${r.point_id} (${r.size_bin}, ${r.confidence})</option>`).join("");
jump.onchange = () => go(Number(jump.value));
document.addEventListener("keydown", ev => {
  const t = ev.target;
  if (t.tagName === "TEXTAREA" || (t.tagName === "INPUT" && t.type === "text")) {
    if (ev.key === "Escape") t.blur();
    return;
  }
  if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
  const k = ev.key.toLowerCase();
  if (k === "arrowleft" || k === "k") { go(idx - 1); ev.preventDefault(); return; }
  if (k === "arrowright" || k === "j") { go(idx + 1); ev.preventDefault(); return; }
  if (k === ".") { nextUnverified(); return; }
  if (k === "enter") { verify(); ev.preventDefault(); return; }
  if (k === "u") { entry(DATA[idx]).verified = false; render(); return; }
  if (k === "r") { delete state[String(DATA[idx].point_id)]; render(); return; }
  if (k === "n") { document.getElementById("note").focus(); ev.preventDefault(); return; }
  for (const f in KEYS) if (KEYS[f][k]) { setField(f, KEYS[f][k]); return; }
});
load();
render();
</script>
</body>
</html>
"""


LABEL_COLS = ["pv_present", "outline_match", "polygon_fragment", "failure_case", "confidence", "reason"]
META_COLS = ["split", "lon", "lat", "polygon_area_m2", "estimated_kw", "size_bin", "sub_50kw",
             "edge_distance_m", "xgb_proba", "xgb_pred", "cnn_votes_pv", "cnn_majority"]
SIZE_EDGES_KW = [0, 100, 250, 500, 1000, 2500, float("inf")]


def size_bin(kw: float) -> str:
    for low, high in zip(SIZE_EDGES_KW[:-1], SIZE_EDGES_KW[1:]):
        if low <= kw < high:
            return f"{low:g}-{high:g} kW" if high != float("inf") else f">{low:g} kW"
    return "n/a"


def enrich(labels: pd.DataFrame, worksheet: str, chip_index: str) -> pd.DataFrame:
    """Join the point metadata (worksheet, chip index) to the bare label columns."""
    ws = pd.read_csv(worksheet)[["point_id", "split", "lon", "lat", "polygon_area_m2", "estimated_kw"]]
    ci = pd.read_csv(chip_index)[["point_id", "edge_distance_m", "xgb_proba", "xgb_pred_pv",
                                  "cnn_votes_pv", "cnn_majority_pv"]]
    df = ws.merge(ci, on="point_id", how="left").merge(
        labels[["point_id"] + LABEL_COLS], on="point_id", how="right", validate="one_to_one")
    df["size_bin"] = df["estimated_kw"].map(size_bin)
    df["sub_50kw"] = (df["estimated_kw"] < 50).map({True: "yes", False: "no"})
    df["edge_distance_m"] = df["edge_distance_m"].round(1)
    df["xgb_proba"] = df["xgb_proba"].round(3)
    df["xgb_pred"] = df["xgb_pred_pv"].map({True: "PV", False: "non-PV"})
    df["cnn_majority"] = df["cnn_majority_pv"].map({True: "PV", False: "non-PV"})
    fc = df["failure_case"]
    # the worksheet's two True/False columns, derived from failure_case (blank when unclear)
    df["non_crystalline_silicon"] = [
        "" if v == "unclear" else str(v in ("non_crystalline_silicon", "both")) for v in fc]
    df["spectrally_ambiguous"] = [
        "" if v == "unclear" else str(v in ("spectrally_ambiguous", "both")) for v in fc]
    df["label_source"] = "AI pre-label (unverified)"
    return df[["point_id"] + META_COLS + LABEL_COLS
              + ["non_crystalline_silicon", "spectrally_ambiguous", "label_source"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prelabels", default="reports/review/annotation/prelabels.csv")
    parser.add_argument("--worksheet", default="data/external/annotation_worksheet_v2.csv")
    parser.add_argument("--chip-index", default="reports/review/annotation/chip_index.csv")
    parser.add_argument("--out", default="reports/review/annotation/review.html")
    parser.add_argument("--rewrite-prelabels", action="store_true",
                        help="write the metadata-enriched table back to --prelabels")
    args = parser.parse_args()

    df = pd.read_csv(args.prelabels, keep_default_na=False)
    if not set(META_COLS) <= set(df.columns):
        df = enrich(df, args.worksheet, args.chip_index)
        if args.rewrite_prelabels:
            df.to_csv(args.prelabels, index=False, encoding="utf-8")
            print(f"enriched prelabels -> {args.prelabels}")
            df = pd.read_csv(args.prelabels, keep_default_na=False)
    for field, allowed in FIELDS.items():
        bad = set(df[field]) - set(allowed)
        if bad:
            raise ValueError(f"{field}: values {bad} not in {allowed}")
    records = json.loads(df.to_json(orient="records"))
    html = (TEMPLATE.replace("__DATA__", json.dumps(records, ensure_ascii=False))
            .replace("__COLUMNS__", json.dumps(list(df.columns)))
            .replace("__FIELDS__", json.dumps(FIELDS)))
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"{len(df)} points -> {args.out}")


if __name__ == "__main__":
    main()
