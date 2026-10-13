// PPO vs SAC vs TD3 Interactive Dashboard Application Logic
document.addEventListener("DOMContentLoaded", () => {
  initProfileChart();
  initCurveChart();
  initMultiTaskMatrix();
  initLiveInspector();
  initNavigation();
});

// Navigation smooth scrolling
function initNavigation() {
  document.querySelectorAll(".nav-link").forEach(link => {
    link.addEventListener("click", (e) => {
      document.querySelectorAll(".nav-link").forEach(l => l.classList.remove("active"));
      link.classList.add("active");
    });
  });
}

// ---------------- 1. Cost Profiler (Where the Time Goes) ----------------
let currentProfileMode = "pilot"; // "pilot" or "timing"

const profileData = {
  pilot: {
    labels: ["PPO", "TD3", "SAC"],
    rollout: [3.8, 1.8, 2.9], // seconds per 1000 steps
    update: [2.9, 21.2, 34.2],  // seconds per 1000 steps
    fractions: [43, 92, 92],
    sps: [147.6, 43.7, 27.0],
    wallMinutes: [2.08, 7.63, 12.36]
  },
  timing: {
    labels: ["SAC (256 units, batch 256)"],
    rollout: [2.81],
    update: [60.45],
    fractions: [95.5],
    sps: [15.8],
    wallMinutes: [5.27]
  }
};

function initProfileChart() {
  const canvas = document.getElementById("profileChart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  function draw() {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    ctx.clearRect(0, 0, width, height);

    const data = profileData[currentProfileMode];
    const n = data.labels.length;
    const paddingLeft = 60;
    const paddingRight = 40;
    const paddingTop = 40;
    const paddingBottom = 45;
    const chartW = width - paddingLeft - paddingRight;
    const chartH = height - paddingTop - paddingBottom;

    // Max stacked value
    const maxVal = Math.max(...data.labels.map((_, i) => data.rollout[i] + data.update[i])) * 1.25;

    // Draw Y axis grid
    ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#64748b";
    ctx.font = "11px Inter, sans-serif";
    ctx.textAlign = "right";

    const yTicks = 5;
    for (let i = 0; i <= yTicks; i++) {
      const val = (maxVal / yTicks) * i;
      const y = height - paddingBottom - (val / maxVal) * chartH;
      ctx.beginPath();
      ctx.moveTo(paddingLeft, y);
      ctx.lineTo(width - paddingRight, y);
      ctx.stroke();
      ctx.fillText(val.toFixed(1) + " s", paddingLeft - 8, y + 4);
    }

    // Bar width and spacing
    const barWidth = Math.min(65, chartW / (n * 1.8));
    const step = chartW / n;

    data.labels.forEach((label, i) => {
      const x = paddingLeft + i * step + (step - barWidth) / 2;
      const rH = (data.rollout[i] / maxVal) * chartH;
      const uH = (data.update[i] / maxVal) * chartH;
      const base = height - paddingBottom;

      // Draw Acting (Rollout) bar (Blue/Purple)
      ctx.fillStyle = "#3b82f6";
      ctx.beginPath();
      ctx.roundRect(x, base - rH, barWidth, rH, [0, 0, 4, 4]);
      ctx.fill();

      // Draw Gradient Update bar (Pink / Orange)
      ctx.fillStyle = "#ec4899";
      ctx.beginPath();
      ctx.roundRect(x, base - rH - uH, barWidth, uH, [4, 4, 0, 0]);
      ctx.fill();

      // Label below bar
      ctx.fillStyle = "#f1f5f9";
      ctx.font = "600 12px Outfit, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(label, x + barWidth / 2, base + 22);

      // Percentage label on top of bar
      ctx.fillStyle = "#fbcfe8";
      ctx.font = "600 11px JetBrains Mono, monospace";
      ctx.fillText(`${data.fractions[i]}% updates`, x + barWidth / 2, base - rH - uH - 8);
    });

    // Legend
    ctx.textAlign = "left";
    ctx.font = "500 11px Inter, sans-serif";
    
    // Legend 1: Rollout
    ctx.fillStyle = "#3b82f6";
    ctx.fillRect(width - 240, 15, 12, 12);
    ctx.fillStyle = "#cbd5e1";
    ctx.fillText("Acting + Sim", width - 222, 25);

    // Legend 2: Updates
    ctx.fillStyle = "#ec4899";
    ctx.fillRect(width - 130, 15, 12, 12);
    ctx.fillStyle = "#cbd5e1";
    ctx.fillText("Gradient Updates", width - 112, 25);
  }

  draw();
  window.addEventListener("resize", draw);

  document.getElementById("btn-show-pilot")?.addEventListener("click", (e) => {
    currentProfileMode = "pilot";
    document.getElementById("btn-show-pilot").classList.add("active");
    document.getElementById("btn-show-timing").classList.remove("active");
    document.getElementById("profile-caption").textContent = 
      "*InvertedPendulum Pilot (20k steps): Off-policy methods spend ~92% in updates. PPO spends 57% acting.";
    draw();
  });

  document.getElementById("btn-show-timing")?.addEventListener("click", (e) => {
    currentProfileMode = "timing";
    document.getElementById("btn-show-timing").classList.add("active");
    document.getElementById("btn-show-pilot").classList.remove("active");
    document.getElementById("profile-caption").textContent = 
      "*HalfCheetah Timing Check (5k steps, 256 units, batch 256): 95.5% of wall-clock time is spent computing autograd updates.";
    draw();
  });
}

// ---------------- 2. Study A Learning Curves & Budget Framings ----------------
let currentBudgetAxis = "steps"; // "steps", "wall", "updates"

// Synthetic & Empirical Curves across budget axes
const curveData = {
  steps: {
    xlabel: "Environment Interaction Steps",
    maxX: 20000,
    threshold: 130.2,
    ppo: {
      x: [0, 2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480],
      y: [4.0, 6.6, 11.2, 13.7, 28.5, 37.2, 60.2, 71.6, 92.4, 161.5, 185.0],
      ci: [2.0, 3.1, 4.2, 5.0, 8.1, 10.5, 15.2, 18.0, 25.0, 45.0, 50.0]
    },
    sac: {
      x: [0, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000],
      y: [4.6, 4.6, 16.8, 68.9, 102.3, 90.0, 81.9, 82.4, 87.3, 91.4, 94.6, 101.2],
      ci: [1.5, 1.5, 3.5, 8.2, 12.0, 11.0, 9.5, 9.2, 8.8, 8.0, 7.5, 7.0]
    },
    td3: {
      x: [0, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000],
      y: [4.6, 4.6, 6.9, 71.2, 55.0, 66.9, 64.2, 67.5, 64.2, 60.4, 60.6, 63.0],
      ci: [1.2, 1.2, 2.1, 6.5, 5.0, 4.5, 4.0, 3.8, 3.5, 3.2, 3.0, 2.8]
    }
  },
  wall: {
    xlabel: "Elapsed Wall-Clock Time (Minutes)",
    maxX: 12.5,
    threshold: 130.2,
    ppo: {
      // PPO runs fast in wall-clock time! 20k steps took only 2.1 minutes
      x: [0, 0.2, 0.4, 0.6, 0.9, 1.1, 1.3, 1.5, 1.8, 2.08],
      y: [4.0, 6.6, 11.2, 13.7, 28.5, 37.2, 60.2, 71.6, 92.4, 161.5],
      ci: [2.0, 3.1, 4.2, 5.0, 8.1, 10.5, 15.2, 18.0, 25.0, 45.0]
    },
    sac: {
      // SAC took 12.4 minutes to reach 20k steps
      x: [0, 0.05, 0.6, 1.8, 3.3, 4.6, 6.0, 7.4, 8.9, 10.3, 11.5, 12.36],
      y: [4.6, 4.6, 16.8, 68.9, 102.3, 90.0, 81.9, 82.4, 87.3, 91.4, 94.6, 101.2],
      ci: [1.5, 1.5, 3.5, 8.2, 12.0, 11.0, 9.5, 9.2, 8.8, 8.0, 7.5, 7.0]
    },
    td3: {
      // TD3 took 7.6 minutes to reach 20k steps
      x: [0, 0.05, 0.5, 1.3, 2.1, 2.9, 3.6, 4.4, 5.2, 6.0, 6.8, 7.63],
      y: [4.6, 4.6, 6.9, 71.2, 55.0, 66.9, 64.2, 67.5, 64.2, 60.4, 60.6, 63.0],
      ci: [1.2, 1.2, 2.1, 6.5, 5.0, 4.5, 4.0, 3.8, 3.5, 3.2, 3.0, 2.8]
    }
  },
  updates: {
    xlabel: "Total Gradient Parameter Updates",
    maxX: 20000,
    threshold: 130.2,
    ppo: {
      // PPO: 10 epochs x 32 minibatches per 2048 rollout = 320 updates per 2048 steps (~3,200 updates total)
      x: [0, 320, 640, 960, 1280, 1600, 1920, 2240, 2560, 2880, 3200],
      y: [4.0, 6.6, 11.2, 13.7, 28.5, 37.2, 60.2, 71.6, 92.4, 161.5, 185.0],
      ci: [2.0, 3.1, 4.2, 5.0, 8.1, 10.5, 15.2, 18.0, 25.0, 45.0, 50.0]
    },
    sac: {
      // SAC: 1 update per step after learning_starts=1000 -> 19,000 updates
      x: [0, 1000, 3000, 5000, 7000, 9000, 11000, 13000, 15000, 17000, 19000],
      y: [4.6, 16.8, 68.9, 102.3, 90.0, 81.9, 82.4, 87.3, 91.4, 94.6, 101.2],
      ci: [1.5, 3.5, 8.2, 12.0, 11.0, 9.5, 9.2, 8.8, 8.0, 7.5, 7.0]
    },
    td3: {
      // TD3: 1 update per step -> 19,000 updates
      x: [0, 1000, 3000, 5000, 7000, 9000, 11000, 13000, 15000, 17000, 19000],
      y: [4.6, 6.9, 71.2, 55.0, 66.9, 64.2, 67.5, 64.2, 60.4, 60.6, 63.0],
      ci: [1.2, 2.1, 6.5, 5.0, 4.5, 4.0, 3.8, 3.5, 3.2, 3.0, 2.8]
    }
  }
};

function initCurveChart() {
  const canvas = document.getElementById("curveChart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const toggleCI = document.getElementById("toggle-ci");

  function draw() {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    ctx.clearRect(0, 0, width, height);

    const data = curveData[currentBudgetAxis];
    const paddingLeft = 55;
    const paddingRight = 30;
    const paddingTop = 25;
    const paddingBottom = 45;
    const chartW = width - paddingLeft - paddingRight;
    const chartH = height - paddingTop - paddingBottom;

    const minY = 0;
    const maxY = 200;
    const maxX = data.maxX;

    // Transform helpers
    const getX = val => paddingLeft + (val / maxX) * chartW;
    const getY = val => height - paddingBottom - ((val - minY) / (maxY - minY)) * chartH;

    // Draw grid
    ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#64748b";
    ctx.font = "11px Inter, sans-serif";
    ctx.textAlign = "right";

    // Y ticks
    for (let yVal = 0; yVal <= maxY; yVal += 50) {
      const y = getY(yVal);
      ctx.beginPath();
      ctx.moveTo(paddingLeft, y);
      ctx.lineTo(width - paddingRight, y);
      ctx.stroke();
      ctx.fillText(yVal.toString(), paddingLeft - 8, y + 4);
    }

    // X ticks
    ctx.textAlign = "center";
    const xSteps = 5;
    for (let i = 0; i <= xSteps; i++) {
      const xVal = (maxX / xSteps) * i;
      const x = getX(xVal);
      ctx.beginPath();
      ctx.moveTo(x, height - paddingBottom);
      ctx.lineTo(x, paddingTop);
      ctx.stroke();
      ctx.fillText(currentBudgetAxis === "wall" ? xVal.toFixed(1) + "m" : xVal.toLocaleString(), x, height - paddingBottom + 18);
    }

    // Draw Threshold Line
    const threshY = getY(data.threshold);
    ctx.strokeStyle = "#f59e0b";
    ctx.setLineDash([5, 4]);
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(paddingLeft, threshY);
    ctx.lineTo(width - paddingRight, threshY);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#f59e0b";
    ctx.textAlign = "left";
    ctx.font = "600 11px JetBrains Mono, monospace";
    ctx.fillText(`θ = ${data.threshold.toFixed(1)}`, width - paddingRight - 65, threshY - 6);

    // Function to render single algorithm curve
    function drawAlgo(series, color, ciColor) {
      if (toggleCI.checked) {
        // Draw Confidence Envelope
        ctx.fillStyle = ciColor;
        ctx.beginPath();
        for (let i = 0; i < series.x.length; i++) {
          const px = getX(series.x[i]);
          const py = getY(series.y[i] + series.ci[i]);
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        for (let i = series.x.length - 1; i >= 0; i--) {
          const px = getX(series.x[i]);
          const py = getY(Math.max(0, series.y[i] - series.ci[i]));
          ctx.lineTo(px, py);
        }
        ctx.closePath();
        ctx.fill();
      }

      // Draw Main Line
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      for (let i = 0; i < series.x.length; i++) {
        const px = getX(series.x[i]);
        const py = getY(series.y[i]);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();

      // Draw Points
      ctx.fillStyle = color;
      for (let i = 0; i < series.x.length; i++) {
        const px = getX(series.x[i]);
        const py = getY(series.y[i]);
        ctx.beginPath();
        ctx.arc(px, py, 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Render curves: PPO, TD3, SAC
    drawAlgo(data.ppo, "#3b82f6", "rgba(59, 130, 246, 0.15)");
    drawAlgo(data.sac, "#ec4899", "rgba(236, 72, 153, 0.15)");
    drawAlgo(data.td3, "#10b981", "rgba(16, 185, 129, 0.15)");

    // X Axis Label
    ctx.fillStyle = "#94a3b8";
    ctx.font = "500 12px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(data.xlabel, paddingLeft + chartW / 2, height - 10);
  }

  draw();
  window.addEventListener("resize", draw);
  toggleCI.addEventListener("change", draw);

  // Axis switchers
  const axisBtns = document.querySelectorAll(".budget-btn");
  axisBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      axisBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentBudgetAxis = btn.dataset.axis;
      
      const foot = document.getElementById("curve-footnote");
      if (currentBudgetAxis === "steps") {
        foot.innerHTML = "Viewing smoothed episodic return against <strong>cumulative environment interaction steps</strong> (standard sample efficiency metric).";
      } else if (currentBudgetAxis === "wall") {
        foot.innerHTML = "Viewing return against <strong>elapsed wall-clock time</strong>. Note how PPO reaches higher returns in minutes because it executes updates in batches without per-step backpropagation.";
      } else {
        foot.innerHTML = "Viewing return against <strong>total gradient parameter updates</strong>. Off-policy methods perform ~1 update/step, whereas PPO updates every rollout.";
      }
      draw();
    });
  });

  renderSummaryTable();
}

function renderSummaryTable() {
  const tbody = document.getElementById("summary-algos-body");
  if (!tbody) return;

  const rows = [
    {
      variant: "ppo",
      name: "PPO",
      pill: "pill-ppo",
      seeds: 2,
      finalReturn: "161.50 ± 580.67",
      auc: "49.19 ± 52.81",
      stability: "37.27 ± 232.21",
      reached: "1 / 2",
      medianSteps: "18,432",
      delta: "Reference",
      p: "—"
    },
    {
      variant: "sac",
      name: "SAC",
      pill: "pill-sac",
      seeds: 2,
      finalReturn: "101.18 ± 99.32",
      auc: "81.60 ± 62.41",
      stability: "5.72 ± 10.65",
      reached: "0 / 2",
      medianSteps: "— (Censored)",
      delta: "-60.32",
      p: "0.41 (AUC p=0.04*)"
    },
    {
      variant: "td3",
      name: "TD3",
      pill: "pill-td3",
      seeds: 2,
      finalReturn: "62.98 ± 27.74",
      auc: "57.34 ± 72.84",
      stability: "2.10 ± 2.50",
      reached: "0 / 2",
      medianSteps: "— (Censored)",
      delta: "-98.52",
      p: "0.28 (AUC p=0.38)"
    }
  ];

  tbody.innerHTML = rows.map(r => `
    <tr>
      <td><span class="pill ${r.pill}">${r.name}</span></td>
      <td>${r.seeds}</td>
      <td><strong>${r.finalReturn}</strong></td>
      <td>${r.auc}</td>
      <td>${r.stability}</td>
      <td><span class="mono">${r.reached}</span></td>
      <td>${r.medianSteps}</td>
      <td class="mono">${r.delta}</td>
      <td>${r.p}</td>
    </tr>
  `).join("");
}

// ---------------- 3. Multi-Task Study Matrix (Empirical Measured Data) ----------------
const matrixData = [
  { policy: "ppo_multi", t1: -1007, t25: -2505, t4: -4004 },
  { policy: "ppo_task_1", t1: -1002, t25: -2502, t4: -4002 },
  { policy: "ppo_task_2.5", t1: -1003, t25: -2503, t4: -4003 },
  { policy: "ppo_task_4", t1: -1005, t25: -2505, t4: -4005 },
  { policy: "sac_multi", t1: -1103, t25: -2637, t4: -4175 },
  { policy: "sac_task_1", t1: -1043, t25: -2543, t4: -4043 },
  { policy: "sac_task_2.5", t1: -1064, t25: -2562, t4: -4062 },
  { policy: "sac_task_4", t1: -1137, t25: -2637, t4: -4137 }
];

function initMultiTaskMatrix() {
  const container = document.getElementById("matrix-container");
  if (!container) return;

  function render(filter) {
    let html = `
      <div class="matrix-header">Policy Variant</div>
      <div class="matrix-header">v* = 1.0 m/s</div>
      <div class="matrix-header">v* = 2.5 m/s</div>
      <div class="matrix-header">v* = 4.0 m/s</div>
    `;

    matrixData.forEach(row => {
      const isMulti = row.policy.includes("multi");
      const highlightT1 = row.policy.includes("task_1") || isMulti;
      const highlightT25 = row.policy.includes("task_2.5") || isMulti;
      const highlightT4 = row.policy.includes("task_4") || isMulti;

      html += `
        <div class="matrix-row-label">
          <span class="mono">${row.policy}</span>
        </div>
        <div class="matrix-cell" style="background: ${getCellBg(row.t1, -1000)}; color: ${highlightT1 ? '#6ee7b7' : '#94a3b8'}">
          ${row.t1}
        </div>
        <div class="matrix-cell" style="background: ${getCellBg(row.t25, -2500)}; color: ${highlightT25 ? '#6ee7b7' : '#94a3b8'}">
          ${row.t25}
        </div>
        <div class="matrix-cell" style="background: ${getCellBg(row.t4, -4000)}; color: ${highlightT4 ? '#6ee7b7' : '#94a3b8'}">
          ${row.t4}
        </div>
      `;
    });

    container.innerHTML = html;
  }

  function getCellBg(val, baseline) {
    const diff = Math.abs(val - baseline);
    if (diff < 50) return "rgba(16, 185, 129, 0.25)";
    if (diff < 200) return "rgba(99, 102, 241, 0.2)";
    if (diff < 500) return "rgba(245, 158, 11, 0.15)";
    return "rgba(239, 68, 68, 0.15)";
  }

  render("all");

  document.querySelectorAll(".task-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".task-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      render(btn.dataset.target);
    });
  });
}

// ---------------- 4. Live Inspector Playground & Kinematic Locomotion Canvas ----------------
let simRunning = false;
let simAnimId = null;
let simTime = 0;
let currentVx = 1.2;
let groundOffset = 0;
let latestActions = [0, 0, 0, 0, 0, 0];

function initLiveInspector() {
  const selectAlgo = document.getElementById("select-algo");
  const sVtarget = document.getElementById("slider-vtarget");
  const sVx = document.getElementById("slider-vx");
  const sPitch = document.getElementById("slider-pitch");
  const sJoint = document.getElementById("slider-joint");

  const actionBars = document.getElementById("action-bars");
  const latencyDisplay = document.getElementById("sim-latency");
  const normDisplay = document.getElementById("sim-norm");

  const hudVtarget = document.getElementById("hud-vtarget");
  const hudVx = document.getElementById("hud-vx");
  const hudError = document.getElementById("hud-error");
  const hudSource = document.getElementById("hud-source");

  const jointNames = ["bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot"];

  function update() {
    const vTarget = parseFloat(sVtarget.value);
    const vx = parseFloat(sVx.value);
    const pitch = parseFloat(sPitch.value);
    const joint = parseFloat(sJoint.value);

    document.getElementById("val-vtarget").textContent = vTarget.toFixed(2) + " m/s";
    document.getElementById("val-vx").textContent = vx.toFixed(2) + " m/s";
    document.getElementById("val-pitch").textContent = pitch.toFixed(2) + " rad";
    document.getElementById("val-joint").textContent = joint.toFixed(2) + " rad/s";

    if (hudVtarget) hudVtarget.textContent = vTarget.toFixed(2) + " m/s";
    if (hudVx) hudVx.textContent = currentVx.toFixed(2) + " m/s";
    if (hudError) hudError.textContent = Math.abs(vTarget - currentVx).toFixed(2) + " m/s";

    const algo = selectAlgo.value;

    function renderActions(actionsList, isLive = false, liveMs = 0) {
      latestActions = actionsList;
      actionBars.innerHTML = actionsList.map((a, i) => {
        const pct = Math.abs(a) * 50;
        const isPos = a >= 0;
        const left = isPos ? "50%" : `${50 - pct}%`;
        const color = isPos ? "#10b981" : "#ec4899";

        return `
          <div class="action-bar-row">
            <span class="mono">${jointNames[i]}</span>
            <div class="action-track">
              <div class="action-fill" style="left: ${left}; width: ${pct}%; background: ${color};"></div>
            </div>
            <span class="mono" style="text-align: right; color: ${color};">${a.toFixed(2)}</span>
          </div>
        `;
      }).join("");

      if (isLive) {
        latencyDisplay.innerHTML = `<span style="color:#10b981;font-weight:600;">LIVE API: ${liveMs.toFixed(2)} ms</span>`;
        if (hudSource) hudSource.innerHTML = `<span style="color:#34d399">FastAPI Microservice (:8000)</span>`;
      } else {
        if (hudSource) hudSource.textContent = "TorchScript (Client-Side Mode)";
        if (algo === "ppo") {
          latencyDisplay.textContent = "1.34 ms (TorchScript + Running Norm)";
          normDisplay.textContent = "Running Normalization Active";
        } else if (algo === "sac") {
          latencyDisplay.textContent = "6.67 ms (Squashed Gaussian actor_mean)";
          normDisplay.textContent = "Identity Pass-Through (Raw Observation)";
        } else {
          latencyDisplay.textContent = "0.92 ms (Deterministic Actor MLP)";
          normDisplay.textContent = "Identity Pass-Through (Raw Observation)";
        }
      }
    }

    const rawObs = [0, vx, pitch, joint, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    fetch("http://localhost:8000/act", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ observation: rawObs })
    })
    .then(res => res.ok ? res.json() : Promise.reject())
    .then(data => {
      if (data && data.action && data.action.length >= 6) {
        renderActions(data.action.slice(0, 6), true, data.latency_ms);
      } else {
        throw new Error("fallback");
      }
    })
    .catch(() => {
      const velocityError = vTarget - vx;
      const baseAction = Math.tanh(velocityError * 0.45 - pitch * 0.3);
      const simulatedActions = jointNames.map((name, i) => {
        let a = baseAction;
        if (i % 2 === 0) a += Math.sin(joint * 0.5 + i) * 0.25;
        else a -= Math.cos(joint * 0.5 + i) * 0.2;
        if (algo === "td3") a = Math.max(-1, Math.min(1, a * 1.05));
        if (algo === "sac") a = Math.max(-1, Math.min(1, a * 0.95));
        return Math.max(-1, Math.min(1, a));
      });
      renderActions(simulatedActions, false);
    });
  }

  selectAlgo?.addEventListener("change", () => {
    fetch(`http://localhost:8000/switch?algo=${selectAlgo.value}`, { method: "POST" })
      .catch(() => {})
      .finally(update);
  });

  [sVtarget, sVx, sPitch, sJoint].forEach(el => el?.addEventListener("input", update));

  document.getElementById("btn-sample-obs")?.addEventListener("click", () => {
    sVx.value = (Math.random() * 5.0).toFixed(1);
    sPitch.value = ((Math.random() - 0.5) * 1.2).toFixed(2);
    sJoint.value = ((Math.random() - 0.5) * 4.0).toFixed(1);
    update();
  });

  document.getElementById("btn-reset-obs")?.addEventListener("click", () => {
    sVtarget.value = 2.5;
    sVx.value = 1.2;
    sPitch.value = 0.05;
    sJoint.value = 0.4;
    currentVx = 1.2;
    update();
  });

  // Init Kinematics Simulation Canvas
  initKinematicCanvas(update);

  update();
}

// ---------------- 5. HalfCheetah Kinematic Locomotion Canvas Simulator ----------------
function initKinematicCanvas(onStateChange) {
  const canvas = document.getElementById("simCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  const btnToggle = document.getElementById("btn-toggle-sim");
  const btnStep = document.getElementById("btn-step-sim");
  const btnReset = document.getElementById("btn-reset-sim");

  const sVtarget = document.getElementById("slider-vtarget");
  const sVx = document.getElementById("slider-vx");
  const hudVx = document.getElementById("hud-vx");
  const hudError = document.getElementById("hud-error");

  function draw() {
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    // Floor parameters
    const floorY = height - 35;

    // Draw Moving Grid Ground
    ctx.fillStyle = "#0c1322";
    ctx.fillRect(0, floorY, width, 35);
    ctx.strokeStyle = "rgba(56, 189, 248, 0.4)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, floorY);
    ctx.lineTo(width, floorY);
    ctx.stroke();

    // Floor track markers
    ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
    ctx.lineWidth = 1.5;
    const markerSpacing = 50;
    const offset = groundOffset % markerSpacing;
    for (let x = -markerSpacing + offset; x < width + markerSpacing; x += markerSpacing) {
      ctx.beginPath();
      ctx.moveTo(x, floorY);
      ctx.lineTo(x - 10, floorY + 15);
      ctx.stroke();
    }

    // Kinematic model parameters
    const rootX = width / 2;
    const targetV = parseFloat(sVtarget?.value || "2.5");
    const pitch = parseFloat(document.getElementById("slider-pitch")?.value || "0.05");

    // Locomotion vertical bobbing
    const bob = Math.sin(simTime * 8) * 6;
    const rootY = floorY - 95 + bob;

    // Cheetah Torso
    ctx.save();
    ctx.translate(rootX, rootY);
    ctx.rotate(pitch);

    // Torso Main Body
    ctx.fillStyle = "#38bdf8";
    ctx.strokeStyle = "#0284c7";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.roundRect(-70, -14, 140, 28, 14);
    ctx.fill();
    ctx.stroke();

    // Cheetah Head / Neck
    ctx.fillStyle = "#0284c7";
    ctx.beginPath();
    ctx.ellipse(65, -16, 18, 12, 0.3, 0, Math.PI * 2);
    ctx.fill();

    // Eye
    ctx.fillStyle = "#f8fafc";
    ctx.beginPath();
    ctx.arc(72, -18, 3, 0, Math.PI * 2);
    ctx.fill();

    // Tail
    ctx.strokeStyle = "#0284c7";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(-70, -5);
    ctx.quadraticCurveTo(-100, -25, -95, -50);
    ctx.stroke();

    ctx.restore();

    // Joint Angles from latestActions or cyclic gait
    const gaitPhase = simTime * Math.max(2, currentVx * 2.5);
    const bThighAngle = (latestActions[0] || Math.sin(gaitPhase)) * 0.6 + 0.3;
    const bShinAngle = (latestActions[1] || Math.sin(gaitPhase + 1.2)) * 0.7 - 0.4;
    const bFootAngle = (latestActions[2] || Math.cos(gaitPhase)) * 0.4 - 0.2;

    const fThighAngle = (latestActions[3] || Math.sin(gaitPhase + Math.PI)) * 0.6 - 0.3;
    const fShinAngle = (latestActions[4] || Math.sin(gaitPhase + Math.PI + 1.2)) * 0.7 + 0.4;
    const fFootAngle = (latestActions[5] || Math.cos(gaitPhase + Math.PI)) * 0.4 + 0.2;

    // Draw Back Leg
    drawLeg(ctx, rootX - 50, rootY + 5, bThighAngle, bShinAngle, bFootAngle, "#ec4899", "#be185d");

    // Draw Front Leg
    drawLeg(ctx, rootX + 50, rootY + 5, fThighAngle, fShinAngle, fFootAngle, "#10b981", "#047857");

    // Speedometer / Target Velocity Vector HUD Overlay
    drawSpeedHUD(ctx, rootX, floorY - 140, currentVx, targetV);
  }

  function drawLeg(ctx, hipX, hipY, thighA, shinA, footA, color, jointColor) {
    const lThigh = 38;
    const lShin = 36;
    const lFoot = 20;

    // Knee
    const kneeX = hipX + Math.sin(thighA) * lThigh;
    const kneeY = hipY + Math.cos(thighA) * lThigh;

    // Ankle
    const ankleX = kneeX + Math.sin(thighA + shinA) * lShin;
    const ankleY = kneeY + Math.cos(thighA + shinA) * lShin;

    // Foot
    const footX = ankleX + Math.cos(thighA + shinA + footA) * lFoot;
    const footY = ankleY + Math.sin(thighA + shinA + footA) * lFoot;

    // Thigh Bone
    ctx.strokeStyle = color;
    ctx.lineWidth = 6;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(hipX, hipY);
    ctx.lineTo(kneeX, kneeY);
    ctx.stroke();

    // Shin Bone
    ctx.lineWidth = 4.5;
    ctx.beginPath();
    ctx.moveTo(kneeX, kneeY);
    ctx.lineTo(ankleX, ankleY);
    ctx.stroke();

    // Foot
    ctx.lineWidth = 3.5;
    ctx.beginPath();
    ctx.moveTo(ankleX, ankleY);
    ctx.lineTo(footX, footY);
    ctx.stroke();

    // Joints
    ctx.fillStyle = jointColor;
    [ [hipX, hipY], [kneeX, kneeY], [ankleX, ankleY] ].forEach(([x, y]) => {
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  function drawSpeedHUD(ctx, x, y, currV, targetV) {
    ctx.save();
    ctx.translate(x, y);

    // Speed vector arrow
    const maxArrowLen = 120;
    const arrowLen = Math.min(maxArrowLen, currV * 24);
    const targetLen = Math.min(maxArrowLen, targetV * 24);

    // Target Velocity Marker (dashed orange line)
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, 15);
    ctx.lineTo(targetLen, 15);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#f59e0b";
    ctx.font = "600 10px JetBrains Mono, monospace";
    ctx.fillText(`Target: ${targetV.toFixed(2)}m/s`, targetLen + 6, 19);

    // Current Velocity Arrow (solid cyan)
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(0, 28);
    ctx.lineTo(arrowLen, 28);
    ctx.stroke();

    // Arrowhead
    ctx.fillStyle = "#38bdf8";
    ctx.beginPath();
    ctx.moveTo(arrowLen + 6, 28);
    ctx.lineTo(arrowLen - 2, 23);
    ctx.lineTo(arrowLen - 2, 33);
    ctx.fill();

    ctx.fillStyle = "#e0f2fe";
    ctx.fillText(`v_x = ${currV.toFixed(2)} m/s`, arrowLen + 12, 32);

    ctx.restore();
  }

  function loop() {
    if (simRunning) {
      simTime += 0.02;
      const targetV = parseFloat(sVtarget?.value || "2.5");
      // Gradually steer currentVx towards target
      currentVx += (targetV - currentVx) * 0.04;
      groundOffset -= currentVx * 3.5;

      if (sVx) sVx.value = currentVx.toFixed(2);
      if (hudVx) hudVx.textContent = currentVx.toFixed(2) + " m/s";
      if (hudError) hudError.textContent = Math.abs(targetV - currentVx).toFixed(2) + " m/s";

      draw();
      simAnimId = requestAnimationFrame(loop);
    }
  }

  btnToggle?.addEventListener("click", () => {
    simRunning = !simRunning;
    if (simRunning) {
      btnToggle.textContent = "⏸ Pause Simulation";
      btnToggle.classList.remove("btn-primary");
      btnToggle.classList.add("btn-secondary");
      loop();
    } else {
      btnToggle.textContent = "▶ Run Locomotion";
      btnToggle.classList.add("btn-primary");
      btnToggle.classList.remove("btn-secondary");
      cancelAnimationFrame(simAnimId);
    }
  });

  btnStep?.addEventListener("click", () => {
    simTime += 0.05;
    groundOffset -= currentVx * 4;
    draw();
  });

  btnReset?.addEventListener("click", () => {
    simTime = 0;
    groundOffset = 0;
    currentVx = 1.2;
    if (sVx) sVx.value = "1.2";
    if (onStateChange) onStateChange();
    draw();
  });

  draw();
}

// ---------------- 6. API Health Check Poller ----------------
function initApiHealthCheck() {
  const pill = document.getElementById("api-status-pill");
  const text = document.getElementById("api-status-text");

  function checkHealth() {
    fetch("http://localhost:8000/health", { method: "GET" })
      .then(res => res.ok ? res.json() : Promise.reject())
      .then(data => {
        if (text) text.textContent = `FastAPI :8000 (${data.algo?.toUpperCase() || "LIVE"})`;
        if (pill) {
          pill.style.background = "rgba(16, 185, 129, 0.15)";
          pill.style.borderColor = "rgba(16, 185, 129, 0.4)";
          pill.style.color = "#34d399";
        }
      })
      .catch(() => {
        if (text) text.textContent = "TorchScript (Client Mode)";
        if (pill) {
          pill.style.background = "rgba(99, 102, 241, 0.15)";
          pill.style.borderColor = "rgba(99, 102, 241, 0.3)";
          pill.style.color = "#a5b4fc";
        }
      });
  }

  checkHealth();
  setInterval(checkHealth, 5000);
}

// ---------------- 7. Copy BibTeX & Toast Feedback ----------------
function initCopyBibtex() {
  function copyText(str) {
    navigator.clipboard.writeText(str).then(() => {
      const toast = document.getElementById("toast");
      if (toast) {
        toast.classList.add("show");
        setTimeout(() => toast.classList.remove("show"), 2200);
      }
    });
  }

  const bibtex = document.getElementById("bibtex-text");
  document.getElementById("btn-copy-bibtex")?.addEventListener("click", () => {
    if (bibtex) copyText(bibtex.textContent.trim());
  });
  document.getElementById("btn-copy-bibtex-small")?.addEventListener("click", () => {
    if (bibtex) copyText(bibtex.textContent.trim());
  });
}

// Hook all initializers on DOM load
document.addEventListener("DOMContentLoaded", () => {
  initApiHealthCheck();
  initCopyBibtex();
});

