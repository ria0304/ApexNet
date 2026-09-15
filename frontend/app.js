const $ = (id) => document.getElementById(id);
let state = null;
let selected = 0;
let autoTimer = null;

const TRACK_PATHS = {
  spa: "M80,260 C90,200 70,140 120,110 C180,70 260,90 300,130 C340,170 380,80 460,90 C540,100 580,160 560,220 C540,280 460,250 400,270 C340,290 220,330 140,310 C90,300 70,300 80,260 Z",
  monaco: "M90,240 C80,180 140,120 200,140 C250,155 270,90 340,100 C410,110 430,170 500,160 C560,150 570,230 520,260 C460,300 380,250 300,280 C220,310 110,310 90,240 Z",
  silverstone: "M70,220 C80,140 160,80 250,90 C330,100 360,160 440,140 C520,120 590,180 560,250 C530,320 420,300 340,310 C250,320 60,300 70,220 Z",
  monza: "M60,250 L80,90 L200,80 C280,75 300,140 380,130 L560,120 L570,180 L400,200 C320,210 300,280 220,290 L70,300 Z",
  singapore: "M100,250 C70,180 130,90 230,100 C310,110 330,180 420,160 C500,140 580,190 540,260 C500,330 360,300 260,310 C170,320 130,310 100,250 Z",
};

async function api(path, opts) {
  const res = await fetch(path, Object.assign({ headers: { "Content-Type": "application/json" } }, opts));
  return res.json();
}

function fmtLap(n) {
  return String(n).padStart(2, "0");
}

function renderBoard(field) {
  const board = $("board");
  board.innerHTML = "";
  field.forEach((d) => {
    const li = document.createElement("li");
    li.style.setProperty("--c", d.color);
    if (d.idx === selected) li.classList.add("active");
    const delta = d.grid_position - d.real_final_position; // real positions gained (+) / lost (-) over the actual race
    li.innerHTML = `
      <span>${d.pos}</span>
      <span><div class="code">${d.code}</div><div class="team">${d.team}</div></span>
      <span class="tyre" title="${Math.round(d.tire * 100)}% stint wear (proxy)"><i style="width:${d.tire * 100}%"></i></span>
      <span class="${delta > 0 ? "drs-on" : "drs-off"}">P${d.grid_position}→P${d.real_final_position}</span>
      <span>${delta > 0 ? "+" + delta : delta}</span>`;
    li.onclick = () => { selected = d.idx; render(state); };
    board.appendChild(li);
  });
}

function pointOnPath(pathEl, t) {
  const len = pathEl.getTotalLength();
  return pathEl.getPointAtLength(((t % 1) + 1) % 1 * len);
}

function renderCircuit(st) {
  const svg = $("circuit");
  const tid = st.track.id;
  const d = TRACK_PATHS[tid] || TRACK_PATHS.spa;
  svg.innerHTML = `
    <defs>
      <filter id="glow"><feGaussianBlur stdDeviation="2.5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
    </defs>
    <path id="tpath" d="${d}" fill="none" stroke="#1a2330" stroke-width="22" />
    <path d="${d}" fill="none" stroke="#2a3548" stroke-width="14" />
    <path d="${d}" fill="none" stroke="#cfd6e2" stroke-width="8" />
    <path d="${d}" fill="none" stroke="#111" stroke-width="1.5" stroke-dasharray="8 10" />
    <g id="cars"></g>
  `;
  const path = svg.querySelector("#tpath");
  const g = svg.querySelector("#cars");
  st.field.forEach((car, i) => {
    const p = pointOnPath(path, car.track_pos);
    const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    c.setAttribute("cx", p.x);
    c.setAttribute("cy", p.y);
    c.setAttribute("r", car.idx === selected ? 7 : 5);
    c.setAttribute("fill", car.color);
    c.setAttribute("class", "car-dot");
    c.setAttribute("filter", "url(#glow)");
    g.appendChild(c);
    const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
    t.setAttribute("x", p.x + 8);
    t.setAttribute("y", p.y - 8);
    t.setAttribute("fill", "#fff");
    t.setAttribute("font-size", "9");
    t.setAttribute("font-weight", "700");
    t.textContent = car.code;
    g.appendChild(t);
  });
}

function renderTelemetry(st) {
  // NOTE: there is no real sub-lap telemetry source available to this app
  // (see backend/real_data.py docstring) so this panel shows real
  // grid-vs-actual-finish plus the two dynamic real proxies (tire stint
  // age, pit progress) instead of fabricated speed/throttle/brake traces.
  const car = st.field.find((d) => d.idx === selected) || st.field[0];
  $("sel-name").textContent = `${car.code} · ${car.name} (real grid P${car.grid_position} → real finish P${car.real_final_position})`;
  const canvas = $("tele-chart");
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const bars = [
    { label: "Tire stint wear (proxy)", value: car.tire, color: "#e10600" },
    { label: "Pit stops completed", value: car.pit_progress, color: "#2ecc71" },
  ];
  const barH = 16, gap = 14;
  bars.forEach((b, i) => {
    const y = 8 + i * (barH + gap);
    ctx.fillStyle = "#1c2433";
    ctx.fillRect(0, y, w, barH);
    ctx.fillStyle = b.color;
    ctx.fillRect(0, y, w * b.value, barH);
    ctx.fillStyle = "#cfd6e2";
    ctx.font = "10px sans-serif";
    ctx.fillText(`${b.label} — ${(b.value * 100).toFixed(0)}%`, 4, y - 3);
  });
}

function renderGraph(st) {
  const svg = $("graph");
  const nodes = st.graph.nodes;
  const n = nodes.length;
  const cx = 160, cy = 80, r = 62;
  const pos = nodes.map((_, i) => {
    const a = (Math.PI * 2 * i) / n - Math.PI / 2;
    return { x: cx + Math.cos(a) * r, y: cy + Math.sin(a) * r };
  });
  let html = "";
  st.graph.edges.forEach((e) => {
    const a = pos[e.source], b = pos[e.target];
    html += `<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke="#1e3a4a" stroke-width="1" />`;
  });
  nodes.forEach((nd, i) => {
    const p = pos[i];
    const hi = nd.id === selected;
    html += `<circle cx="${p.x}" cy="${p.y}" r="${hi ? 7 : 4.5}" fill="${nd.color}" />`;
    html += `<text x="${p.x}" y="${p.y - 8}" fill="#9aa7bb" font-size="7" text-anchor="middle">${nd.code}</text>`;
  });
  svg.innerHTML = html;
}

function renderPreds(preds) {
  const el = $("preds");
  el.innerHTML = preds.map((p) => `
    <div class="pred ${p.probability >= 55 ? "hot" : ""}">
      <div class="pred-pair">
        <span>${p.attacker_code} → ${p.defender_code}</span>
        <span>${p.probability.toFixed(1)}%</span>
      </div>
      <div class="bar"><i style="width:${p.probability}%"></i></div>
      <div class="pred-meta">
        <span>${p.will_overtake ? "OVERTAKE LIKELY" : "HOLD POSITION"}</span>
        <span>${p.horizon}</span>
      </div>
    </div>`).join("");
}

function renderMetrics(m) {
  const el = $("metrics");
  if (!m || !Object.keys(m).length) {
    el.innerHTML = '<div class="tiny">Train the model to populate AUC comparison.</div>';
    return;
  }
  const rows = Object.entries(m).map(([k, v]) => ({
    name: k.replace("_", "+"),
    auc: v.best ? v.best.auc : 0,
  })).sort((a, b) => b.auc - a.auc);
  const max = rows[0].auc || 1;
  el.innerHTML = rows.map((r, i) => `
    <div class="metric ${i === 0 ? "best" : ""}">
      <span class="name">${r.name}</span>
      <div class="mbar"><i style="width:${(r.auc / max) * 100}%"></i></div>
      <span>${r.auc.toFixed(3)}</span>
    </div>`).join("");
}

function renderLog(events) {
  $("log").innerHTML = (events || []).slice().reverse().map((e) => `<li>${e}</li>`).join("") || "<li>Waiting for green lights…</li>";
}

function render(st) {
  if (!st) return;
  state = st;
  $("lap-now").textContent = fmtLap(st.lap);
  $("lap-total").textContent = "/" + st.total_laps;
  $("track-pill").textContent = st.track.name.toUpperCase();
  $("track-info").textContent = `${st.track.length_km} km · ${st.track.corners} corners · replaying real race: ${st.race_name} (${st.race_year})`;
  $("flag").textContent = st.finished ? "CHEQUERED" : "GREEN";
  $("flag").style.background = st.finished ? "#ddd" : "#0d3";
  $("flag").style.color = st.finished ? "#111" : "#041";
  renderBoard(st.field);
  renderCircuit(st);
  renderTelemetry(st);
  renderGraph(st);
  renderPreds(st.predictions);
  renderLog(st.events);
}

async function loadTracks() {
  const tracks = await api("/api/tracks");
  const sel = $("track-select");
  sel.innerHTML = tracks.map((t) => `<option value="${t.id}">${t.name}</option>`).join("");
  sel.value = "spa";
}

async function reset() {
  const track_id = $("track-select").value;
  const st = await api("/api/reset", { method: "POST", body: JSON.stringify({ track_id, seed: Date.now() % 9999 }) });
  render(st);
}

async function step() {
  const st = await api("/api/step", { method: "POST" });
  render(st);
}

async function run5() {
  const st = await api("/api/run?laps=5", { method: "POST" });
  render(st);
}

function setAuto(on) {
  if (autoTimer) clearInterval(autoTimer);
  autoTimer = null;
  if (on) autoTimer = setInterval(step, 1400);
}

window.addEventListener("DOMContentLoaded", async () => {
  await loadTracks();
  const st = await api("/api/state");
  selected = st.field[0].idx;
  render(st);
  try {
    const m = await api("/api/metrics");
    renderMetrics(m);
  } catch (e) {
    renderMetrics(null);
  }
  $("btn-reset").onclick = reset;
  $("btn-step").onclick = step;
  $("btn-run").onclick = run5;
  $("auto").onchange = (e) => setAuto(e.target.checked);
  $("track-select").onchange = reset;
});
