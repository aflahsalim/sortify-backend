// =========================
//  GLOBAL HELPERS
// =========================

async function fetchJSON(path) {
  const res = await fetch(API + path);
  if (!res.ok) throw new Error("HTTP " + res.status);
  return res.json();
}

const scansCache = [];


// =========================
//  RENDER APP SHELL
// =========================

document.getElementById("app").innerHTML = `
  <div class="shell">
    <header>
      <div>
        <div class="title">SORTIFY SOC</div>
        <div class="subtitle">Neon threat monitoring · Students · Phishing · Spam · Reports</div>
      </div>
      <button class="refresh-btn" id="refresh-btn">Refresh</button>
    </header>

    <div class="alert-banner" id="alert-banner">
      <div class="alert-icon">!</div>
      <div id="alert-text"></div>
    </div>

    <section class="grid-metrics">
      <div class="metric-card"><div class="metric-label">Total scans</div><div class="metric-value" id="m-total">-</div></div>
      <div class="metric-card"><div class="metric-label">Safe</div><div class="metric-value metric-safe" id="m-safe">-</div></div>
      <div class="metric-card"><div class="metric-label">Spam</div><div class="metric-value metric-spam" id="m-spam">-</div></div>
      <div class="metric-card"><div class="metric-label">Phishing</div><div class="metric-value metric-phish" id="m-phish">-</div></div>
    </section>

    <section class="layout-main">
      <div class="card">
        <div class="card-title">Attack heatmap (spam + phishing)</div>
        <div class="heatmap-grid" id="heatmap"></div>
      </div>

      <div class="card">
        <div class="card-title">Trends (last 14 days)</div>
        <div class="trend-chart" id="trend-chart"></div>

        <div class="card-title" style="margin-top:10px;">Reporting leaderboard</div>
        <table class="leaderboard-table">
          <thead>
            <tr><th>Sender</th><th>Scans</th></tr>
          </thead>
          <tbody id="leaderboard-body">
            <tr><td colspan="2" style="color:var(--muted);">No data</td></tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="card">
      <div class="card-title">Recent scans</div>

      <div class="scans-scroll">
        <table class="scans-table">
          <thead>
            <tr><th>Time</th><th>Sender</th><th>Subject</th><th>Label</th><th>Score</th></tr>
          </thead>
          <tbody id="scans-body">
            <tr><td colspan="5" style="color:var(--muted);">No scans yet</td></tr>
          </tbody>
        </table>
      </div>

      <div class="export-row">
        <button class="export-btn" data-label="">Export all</button>
        <button class="export-btn" data-label="phishing">Export phishing</button>
        <button class="export-btn" data-label="spam">Export spam</button>
      </div>
    </section>
  </div>

  <!-- MODAL -->
  <div class="modal-backdrop" id="modal-backdrop">
    <div class="modal">
      <div class="modal-header">
        <div class="modal-title">Email details</div>
        <button class="modal-close" id="modal-close">&times;</button>
      </div>
      <div class="modal-meta" id="modal-meta"></div>
      <div class="modal-section-label">Body preview</div>
      <div class="modal-body" id="modal-body">No content</div>
      <div class="modal-section-label">Classification</div>
      <div class="modal-meta" id="modal-class"></div>
    </div>
  </div>
`;


// =========================
//  METRICS
// =========================

function setMetrics(stats) {
  document.getElementById("m-total").textContent = stats.total ?? "-";
  const by = stats.by_label || {};
  document.getElementById("m-safe").textContent = (by.ham || 0) + (by.support || 0);
  document.getElementById("m-spam").textContent = by.spam || 0;
  document.getElementById("m-phish").textContent = by.phishing || 0;
}


// =========================
//  HEATMAP (IMPROVED)
// =========================

function renderHeatmap(matrix) {
  const container = document.getElementById("heatmap");
  container.innerHTML = "";
  const days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"];

  const header = document.createElement("div");
  header.style.gridColumn = "1 / span 25";
  header.style.display = "flex";
  header.style.justifyContent = "space-between";
  header.style.fontSize = "9px";
  header.style.color = "var(--muted)";
  header.innerHTML = "<span></span><span>0</span><span>6</span><span>12</span><span>18</span><span>23</span>";
  container.appendChild(header);

  for (let d = 0; d < 7; d++) {
    const label = document.createElement("div");
    label.textContent = days[d];
    label.style.fontSize = "9px";
    label.style.color = "var(--muted)";
    label.style.display = "flex";
    label.style.alignItems = "center";
    container.appendChild(label);

    for (let h = 0; h < 24; h++) {
      const cell = document.createElement("div");
      cell.className = "heatmap-cell";
      const v = (matrix[d] && matrix[d][h]) || 0;

      if (v === 0) {}
      else if (v <= 1) cell.classList.add("level-1");
      else if (v <= 3) cell.classList.add("level-2");
      else if (v <= 6) cell.classList.add("level-3");
      else cell.classList.add("level-4");

      container.appendChild(cell);
    }
  }
}


// =========================
//  LEADERBOARD (NO REPORTED)
// =========================

function renderLeaderboard(rows) {
  const body = document.getElementById("leaderboard-body");
  body.innerHTML = "";
  if (!rows || rows.length === 0) {
    body.innerHTML = `<tr><td colspan="2" style="color:var(--muted);">No data</td></tr>`;
    return;
  }
  rows.forEach(r => {
    body.innerHTML += `
      <tr>
        <td>${r.sender}</td>
        <td>${r.scans}</td>
      </tr>
    `;
  });
}


// =========================
//  TRENDS (IMPROVED)
// =========================

function renderTrends(trends) {
  const days = trends.days || [];
  const data = trends.data || [];

  if (days.length === 0) {
    document.getElementById("trend-chart").innerHTML = "(No trend data)";
    return;
  }

  const phishing = data.map(d => d.phishing || 0);
  const spam = data.map(d => d.spam || 0);
  const reported = data.map(d => d.reported || 0);

  const options = {
    chart: {
      type: "line",
      height: 260,
      foreColor: "#94a3b8",
      background: "transparent",
      toolbar: { show: false }
    },
    stroke: { width: 3, curve: "smooth" },
    series: [
      { name: "Phishing", data: phishing, color: "#ff4d6d" },
      { name: "Spam", data: spam, color: "#fbbf24" },
      { name: "Reported", data: reported, color: "#38bdf8" }
    ],
    xaxis: { categories: days },
    grid: { borderColor: "#1f2937" }
  };

  document.getElementById("trend-chart").innerHTML = "";
  new ApexCharts(document.querySelector("#trend-chart"), options).render();
}


// =========================
//  RECENT SCANS (SCROLLABLE)
// =========================

function renderScans(recent) {
  const body = document.getElementById("scans-body");
  body.innerHTML = "";
  if (!recent || recent.length === 0) {
    body.innerHTML = `<tr><td colspan="5" style="color:var(--muted);">No scans yet</td></tr>`;
    return;
  }

  scansCache.length = 0;

  recent.forEach((e, idx) => {
    scansCache.push(e);
    const dt = e.timestamp ? new Date(e.timestamp) : null;
    const timeStr = dt ? dt.toLocaleString() : "-";

    const label = e.label || "ham";
    let pillClass = "label-ham";
    if (label === "spam") pillClass = "label-spam";
    else if (label === "phishing") pillClass = "label-phishing";
    else if (label === "support") pillClass = "label-support";

    body.innerHTML += `
      <tr onclick="openModal(${idx})">
        <td>${timeStr}</td>
        <td>${e.sender || "Unknown"}</td>
        <td>${e.subject || "-"}</td>
        <td><span class="label-pill ${pillClass}">${label}</span></td>
        <td>${e.score ?? "-"}</td>
      </tr>
    `;
  });
}


// =========================
//  MODAL (NO REPORTED)
// =========================

function openModal(idx) {
  const entry = scansCache[idx];
  if (!entry) return;

  const dt = entry.timestamp ? new Date(entry.timestamp) : null;
  const timeStr = dt ? dt.toLocaleString() : "-";

  document.getElementById("modal-meta").textContent =
    `${timeStr} · ${entry.sender || "Unknown"} · ${entry.subject || "-"}`;

  document.getElementById("modal-body").textContent =
    entry.body_preview || "(No body preview stored)";

  document.getElementById("modal-class").textContent =
    `Label: ${entry.label} · Score: ${entry.score}`;

  document.getElementById("modal-backdrop").classList.add("active");
}

document.getElementById("modal-close").addEventListener("click", () => {
  document.getElementById("modal-backdrop").classList.remove("active");
});


// =========================
//  ALERTS
// =========================

function renderAlert(alert) {
  const banner = document.getElementById("alert-banner");
  const text = document.getElementById("alert-text");

  if (alert.active) {
    banner.classList.add("active");
    text.textContent = alert.message;
  } else {
    banner.classList.remove("active");
    text.textContent = "";
  }
}


// =========================
//  EXPORT BUTTONS
// =========================

document.querySelectorAll(".export-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const label = btn.getAttribute("data-label");
    const url = label ? `/dashboard/export?label=${label}` : "/dashboard/export";
    window.location.href = API + url;
  });
});


// =========================
//  REFRESH ALL
// =========================

async function refreshAll() {
  try {
    const [stats, heat, trends, leaderboard, alerts] = await Promise.all([
      fetchJSON("/dashboard/stats"),
      fetchJSON("/dashboard/heatmap"),
      fetchJSON("/dashboard/trends"),
      fetchJSON("/dashboard/leaderboard"),
      fetchJSON("/dashboard/alerts"),
    ]);

    setMetrics(stats);
    renderHeatmap(heat.matrix || []);
    renderTrends(trends);
    renderLeaderboard(leaderboard.rows || []);
    renderScans(stats.recent || []);
    renderAlert(alerts);

  } catch (e) {
    alert("Could not reach backend. Make sure FastAPI is running.");
  }
}

document.getElementById("refresh-btn").addEventListener("click", refreshAll);

refreshAll();
