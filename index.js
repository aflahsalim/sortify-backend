const API = "https://sortify-backend-hwf9d0exgqdub9cn.canadacentral-01.azurewebsites.net";

async function fetchJSON(path) {
  const res = await fetch(API + path);
  if (!res.ok) throw new Error("HTTP " + res.status);
  return res.json();
}

const scansCache = [];

// ── Inject app shell into #app ────────────────────────────────────────────
document.getElementById("app").innerHTML = `
  <div class="grid-metrics">
    <div class="metric-card">
      <div class="accent"></div>
      <div class="metric-value" id="m-total">—</div>
      <div class="metric-label">Total Scans</div>
    </div>
    <div class="metric-card">
      <div class="accent"></div>
      <div class="metric-value metric-safe" id="m-safe">—</div>
      <div class="metric-label">Safe</div>
    </div>
    <div class="metric-card">
      <div class="accent"></div>
      <div class="metric-value metric-spam" id="m-spam">—</div>
      <div class="metric-label">Spam</div>
    </div>
    <div class="metric-card">
      <div class="accent"></div>
      <div class="metric-value metric-phish" id="m-phish">—</div>
      <div class="metric-label">Phishing</div>
    </div>
  </div>

  <div class="layout-main">
    <div class="card">
      <div class="card-title">Attack heatmap — spam + phishing</div>
      <div class="heatmap-grid" id="heatmap"></div>
    </div>
    <div class="card">
      <div class="card-title">Trends — last 14 days</div>
      <div id="trend-chart"></div>
      <div class="card-title" style="margin-top:14px;">Reporting leaderboard</div>
      <table class="leaderboard-table">
        <thead><tr><th>#</th><th>Sender</th><th>Scans</th></tr></thead>
        <tbody id="leaderboard-body"></tbody>
      </table>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Recent scans</div>
    <div class="scans-scroll">
      <table class="scans-table">
        <thead>
          <tr>
            <th>Time</th><th>Sender</th><th>Subject</th><th>Label</th><th>Score</th>
          </tr>
        </thead>
        <tbody id="scans-body"></tbody>
      </table>
    </div>
    <div class="export-row">
      <button class="export-btn" data-label="">Export all</button>
      <button class="export-btn" data-label="phishing">Export phishing</button>
      <button class="export-btn" data-label="spam">Export spam</button>
    </div>
  </div>
`;

// ── Metrics ───────────────────────────────────────────────────────────────
function setMetrics(stats) {
  document.getElementById("m-total").textContent = stats.total ?? "—";
  const by = stats.by_label || {};
  document.getElementById("m-safe").textContent  = (by.ham || 0) + (by.support || 0);
  document.getElementById("m-spam").textContent  = by.spam    || 0;
  document.getElementById("m-phish").textContent = by.phishing || 0;
}

// ── Heatmap ───────────────────────────────────────────────────────────────
function renderHeatmap(matrix) {
  const container = document.getElementById("heatmap");
  container.innerHTML = "";
  const days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"];

  for (let d = 0; d < 7; d++) {
    const label = document.createElement("div");
    label.textContent = days[d];
    label.style.cssText = "font-size:9px;color:#555;display:flex;align-items:center;padding-right:4px;";
    container.appendChild(label);

    for (let h = 0; h < 24; h++) {
      const cell = document.createElement("div");
      cell.className = "heatmap-cell";
      const v = (matrix[d] && matrix[d][h]) || 0;
      if      (v >= 6) cell.classList.add("level-4");
      else if (v >= 3) cell.classList.add("level-3");
      else if (v >= 1) cell.classList.add("level-2");
      else if (v >  0) cell.classList.add("level-1");
      container.appendChild(cell);
    }
  }
}

// ── Trends (ApexCharts) ───────────────────────────────────────────────────
let apexChart = null;

function renderTrends(trends) {
  const days = trends.days || [];
  const data = trends.data || [];

  if (!days.length) {
    document.getElementById("trend-chart").innerHTML =
      '<div class="empty" style="padding:20px">No trend data</div>';
    return;
  }

  if (apexChart) apexChart.destroy();

  apexChart = new ApexCharts(document.getElementById("trend-chart"), {
    chart: {
      type: "bar",
      height: 160,
      background: "transparent",
      toolbar: { show: false },
      foreColor: "#555",
      fontFamily: "Inter, system-ui, sans-serif",
    },
    plotOptions: { bar: { columnWidth: "55%", borderRadius: 3 } },
    series: [
      { name: "Phishing", data: data.map(d => d.phishing || 0) },
      { name: "Spam",     data: data.map(d => d.spam     || 0) },
    ],
    colors: ["#ef4444", "#f59e0b"],
    xaxis: { categories: days, labels: { style: { fontSize: "9px", colors: "#555" } } },
    yaxis: { labels: { style: { colors: "#555", fontSize: "10px" } } },
    grid:  { borderColor: "#1e1e1e", strokeDashArray: 3 },
    legend: { labels: { colors: "#888" }, fontSize: "11px" },
    tooltip: { theme: "dark" },
  });

  apexChart.render();
}

// ── Leaderboard ───────────────────────────────────────────────────────────
function renderLeaderboard(rows) {
  const body = document.getElementById("leaderboard-body");
  body.innerHTML = "";

  if (!rows || rows.length === 0) {
    body.innerHTML = '<tr><td colspan="3" style="color:#555;padding:10px">No data</td></tr>';
    return;
  }

  rows.forEach((r, i) => {
    body.innerHTML += `
      <tr>
        <td style="font-size:10px;color:#3a3a3a;font-weight:600">${i + 1}</td>
        <td>${esc(r.sender)}</td>
        <td style="color:var(--text);font-weight:600">${r.scans}</td>
      </tr>`;
  });
}

// ── Scans table ───────────────────────────────────────────────────────────
function renderScans(recent) {
  const body = document.getElementById("scans-body");
  body.innerHTML = "";
  scansCache.length = 0;

  if (!recent || recent.length === 0) {
    body.innerHTML = '<tr><td colspan="5" class="empty">No scans yet — open an email in Outlook.</td></tr>';
    return;
  }

  recent.forEach((e, idx) => {
    scansCache.push(e);

    const dt = e.timestamp ? new Date(e.timestamp) : null;
    const timeStr = dt ? dt.toLocaleString() : "—";
    const label = e.label || "ham";
    const sc = Math.round((e.score || 0) * 100);
    const col = label === "ham" || label === "support" ? "#22c55e"
              : label === "spam" ? "#f59e0b" : "#ef4444";

    body.innerHTML += `
      <tr onclick="openModal(${idx})">
        <td>${timeStr}</td>
        <td>${esc(e.sender  || "—")}</td>
        <td>${esc(e.subject || "—")}</td>
        <td><span class="label-pill label-${label}">${label}</span></td>
        <td>
          <div style="display:flex;align-items:center;gap:6px">
            <div style="width:48px;height:3px;background:#1c1c1c;border-radius:999px;overflow:hidden;flex-shrink:0">
              <div style="height:100%;width:${sc}%;background:${col};border-radius:999px"></div>
            </div>
            <span style="font-size:11px;font-weight:600;color:${col}">${sc}</span>
          </div>
        </td>
      </tr>`;
  });
}

// ── Modal ─────────────────────────────────────────────────────────────────
function openModal(idx) {
  const entry = scansCache[idx];
  if (!entry) return;

  const dt = entry.timestamp ? new Date(entry.timestamp) : null;
  const timeStr = dt ? dt.toLocaleString() : "—";

  document.getElementById("modal-meta").textContent =
    `${timeStr} · ${esc(entry.sender || "Unknown")} · ${esc(entry.subject || "—")}`;
  document.getElementById("modal-body").textContent =
    entry.body_preview || "(No body preview stored)";
  document.getElementById("modal-class").textContent =
    `Label: ${entry.label}  ·  Score: ${entry.score}`;

  document.getElementById("modal-backdrop").classList.add("active");
}

document.getElementById("modal-close").addEventListener("click", () => {
  document.getElementById("modal-backdrop").classList.remove("active");
});

// ── Export buttons ────────────────────────────────────────────────────────
document.querySelectorAll(".export-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const label = btn.getAttribute("data-label");
    const url = label ? `/dashboard/export?label=${label}` : "/dashboard/export";
    window.location.href = API + url;
  });
});

// ── Fetch all & render ────────────────────────────────────────────────────
async function refreshAll() {
  const dot = document.getElementById("live-dot");
  const txt = document.getElementById("live-txt");

  try {
    const [stats, heat, trends, leaderboard] = await Promise.all([
      fetchJSON("/dashboard/stats"),
      fetchJSON("/dashboard/heatmap"),
      fetchJSON("/dashboard/trends"),
      fetchJSON("/dashboard/leaderboard"),
    ]);

    dot.style.background = "#22c55e";
    dot.style.animation  = "";
    txt.textContent = "Live";

    setMetrics(stats);
    renderHeatmap(heat.matrix   || []);
    renderTrends(trends);
    renderLeaderboard(leaderboard.rows || []);
    renderScans(stats.recent    || []);

  } catch (e) {
    dot.style.background = "#ef4444";
    dot.style.animation  = "none";
    txt.textContent = "Offline";
    console.error("Backend unreachable:", e);
  }
}

// ── Helper ────────────────────────────────────────────────────────────────
function esc(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// ── Init ──────────────────────────────────────────────────────────────────
document.getElementById("refresh-btn").addEventListener("click", refreshAll);
refreshAll();
