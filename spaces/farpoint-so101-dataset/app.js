const INDEX_URL = "reports/index.json";
let report;
let reportBase;
const filters = { split: "all", yaw_deg: "all", mass_kg: "all", size_m: "all", color: "all" };

const $ = (selector) => document.querySelector(selector);
const formatInteger = (value) => new Intl.NumberFormat("en-US").format(value);
const formatPercent = (value, digits = 1) => `${(value * 100).toFixed(digits)}%`;
const shortHash = (value) => value ? `${value.slice(0, 9)}…${value.slice(-7)}` : "—";
const labelFor = (value, field) => {
  if (field === "yaw_deg") return `${value}°`;
  if (field === "mass_kg") return `${Math.round(Number(value) * 1000)} g`;
  if (field === "size_m") return `${Math.round(Number(value) * 1000)} mm`;
  return String(value);
};

async function loadReport() {
  const indexResponse = await fetch(INDEX_URL, { cache: "no-store" });
  if (!indexResponse.ok) throw new Error(`Could not load report index (${indexResponse.status}).`);
  const index = await indexResponse.json();
  const selected = index.versions.find((item) => item.version === index.default_version);
  if (!selected) throw new Error("The default report version is missing.");
  reportBase = selected.report.slice(0, selected.report.lastIndexOf("/") + 1);
  const response = await fetch(selected.report, { cache: "no-store" });
  if (!response.ok) throw new Error(`Could not load ${selected.version} (${response.status}).`);
  report = await response.json();
  if (report.report_sha256 !== selected.report_sha256) throw new Error("Report identity does not match the version index.");
}

function renderOverview() {
  const o = report.overview;
  const datasetRoot = `https://huggingface.co/datasets/${report.identity.dataset_repo}`;
  $("#version-badge").textContent = report.identity.dataset_tag;
  $("#dataset-version-link").href = `${datasetRoot}/tree/${report.identity.dataset_tag}`;
  $("#dataset-card-link").href = `${datasetRoot}/blob/${report.identity.dataset_tag}/README.md`;
  $("#dataset-viewer-link").href = `${datasetRoot}/viewer/default/train`;
  $("#task-copy").textContent = o.task;
  $("#stat-episodes").textContent = formatInteger(o.episodes);
  $("#stat-frames").textContent = formatInteger(o.frames);
  $("#stat-hours").textContent = o.duration_hours.toFixed(2);
  $("#stat-splits").textContent = `${o.split_episodes.train} · ${o.split_episodes.validation} · ${o.split_episodes.test}`;
}

function readableCheck(id) {
  return id.split("_").map((word) => word[0].toUpperCase() + word.slice(1)).join(" ");
}

function renderIntegrity() {
  const integrity = report.integrity;
  const status = $("#integrity-status");
  status.textContent = integrity.status;
  status.classList.toggle("fail", integrity.status !== "PASS");
  $("#integrity-grid").innerHTML = integrity.checks.map((check) => `
    <article class="integrity-item">
      <header><h3>${readableCheck(check.id)}</h3><span class="check-pill">${check.status}</span></header>
      <p>${check.detail}</p>
    </article>`).join("");
  $("#dataset-commit").textContent = shortHash(report.identity.resolved_dataset_commit);
  $("#dataset-commit").title = report.identity.resolved_dataset_commit;
  $("#report-hash").textContent = shortHash(report.report_sha256);
  $("#report-hash").title = report.report_sha256;
}

function uniqueValues(field) {
  return [...new Set(report.variation_coverage.episodes.map((row) => String(row[field])))].sort((a, b) => Number(a) - Number(b));
}

function renderFilters() {
  const fields = [
    ["split", "Split"], ["yaw_deg", "Yaw"], ["mass_kg", "Mass"], ["size_m", "Size"], ["color", "Color"],
  ];
  $("#coverage-filters").innerHTML = fields.map(([field, label]) => `
    <div class="filter-group"><label for="filter-${field}">${label}</label><select id="filter-${field}" data-field="${field}">
      <option value="all">All</option>${uniqueValues(field).map((value) => `<option value="${value}">${labelFor(value, field)}</option>`).join("")}
    </select></div>`).join("") + `<button class="reset-filter" type="button">Reset filters</button>`;
  document.querySelectorAll("[data-field]").forEach((select) => select.addEventListener("change", (event) => {
    filters[event.target.dataset.field] = event.target.value;
    renderCoverage();
  }));
  $(".reset-filter").addEventListener("click", () => {
    Object.keys(filters).forEach((key) => { filters[key] = "all"; });
    document.querySelectorAll("[data-field]").forEach((select) => { select.value = "all"; });
    renderCoverage();
  });
}

function filteredEpisodes() {
  return report.variation_coverage.episodes.filter((row) => Object.entries(filters).every(([field, value]) => value === "all" || String(row[field]) === value));
}

function renderCoverage() {
  const rows = filteredEpisodes();
  $("#coverage-matches").textContent = formatInteger(rows.length);
  const counts = new Map();
  rows.forEach((row) => counts.set(`${row.row}-${row.column}`, (counts.get(`${row.row}-${row.column}`) || 0) + 1));
  const max = Math.max(1, ...counts.values());
  let cells = "";
  for (let r = 0; r < 5; r += 1) for (let c = 0; c < 5; c += 1) {
    const count = counts.get(`${r}-${c}`) || 0;
    const opacity = count ? 0.18 + (count / max) * 0.82 : 0;
    cells += `<div class="heat-cell ${count ? "" : "empty"}" style="--heat:${opacity.toFixed(3)}" title="r${String(r).padStart(2,"0")}_c${String(c).padStart(2,"0")}: ${count} episodes"><small>r${r}c${c}</small>${count}</div>`;
  }
  $("#position-heatmap").innerHTML = cells;
  const bounds = report.variation_coverage.position_bounds_m;
  $("#position-bounds").textContent = `Cube centers span x ${bounds.x[0].toFixed(3)}–${bounds.x[1].toFixed(3)} m and y ${bounds.y[0].toFixed(3)}–${bounds.y[1].toFixed(3)} m.`;
  const axes = [["yaw_deg","Yaw"], ["mass_kg","Mass"], ["size_m","Size"], ["color","Color"], ["split","Split"]];
  const distributions = [];
  axes.forEach(([field, label]) => {
    const axisCounts = new Map(); rows.forEach((row) => axisCounts.set(String(row[field]), (axisCounts.get(String(row[field])) || 0) + 1));
    [...axisCounts.entries()].sort().forEach(([value,count]) => distributions.push({ label: `${label} · ${labelFor(value,field)}`, count }));
  });
  const axisMax = Math.max(1, ...distributions.map((item) => item.count));
  $("#axis-distribution").innerHTML = distributions.map((item) => `<div class="axis-row"><header><span>${item.label}</span><span>${item.count}</span></header><div class="axis-track"><div class="axis-fill" style="width:${(item.count / axisMax) * 100}%"></div></div></div>`).join("");
  $("#covered-cells").textContent = `${counts.size} / 25`;
  $("#exact-combinations").textContent = report.variation_coverage.exact_combinations;
  $("#median-repetition").textContent = `${report.variation_coverage.combination_repetition.q50}×`;
}

function histogram(values, bins = 14) {
  const min = Math.min(...values); const max = Math.max(...values); const span = max - min || 1;
  const counts = Array.from({ length: bins }, () => 0);
  values.forEach((value) => { counts[Math.min(bins - 1, Math.floor(((value - min) / span) * bins))] += 1; });
  return { min, max, counts };
}

function renderMotion() {
  const q = report.episode_action_quality;
  $("#median-duration").textContent = `${q.episode_duration_s.q50.toFixed(1)} s`;
  $("#idle-ratio").textContent = formatPercent(q.idle_transition_ratio);
  $("#action-step").textContent = q.max_action_step.q95.toFixed(2);
  $("#duplicate-count").textContent = q.exact_resampled_action_duplicate_groups.length;
  const durations = q.episode_metrics.map((row) => row.duration_s);
  const hist = histogram(durations);
  const maxCount = Math.max(...hist.counts);
  $("#duration-chart").innerHTML = hist.counts.map((count,index) => {
    const label = index % 3 === 0 ? (hist.min + ((hist.max - hist.min) * index / hist.counts.length)).toFixed(0) : "";
    return `<div class="hist-bar" data-count="${count}" style="height:${Math.max(2,(count / maxCount) * 100)}%"><span>${label}</span></div>`;
  }).join("");
  $("#joint-ranges").innerHTML = q.action_by_joint.map((row) => {
    const gripper = row.joint.includes("gripper"); const min = gripper ? 0 : -100; const span = gripper ? 100 : 200;
    const left = ((row.q01 - min) / span) * 100; const right = ((row.q99 - min) / span) * 100; const median = ((row.q50 - min) / span) * 100;
    return `<div class="joint-row"><label title="${row.joint}">${row.joint.replace(".pos","")}</label><div class="range-line"><div class="range-fill" style="left:${left}%;width:${Math.max(1,right-left)}%"></div><div class="range-median" style="left:${median}%"></div></div><output>${row.q50.toFixed(1)}</output></div>`;
  }).join("");
  $("#motion-definition").textContent = `Idle means ${q.definitions.idle_transition}. Derivatives use the 30 Hz export cadence; values remain in calibrated SO-101 export units.`;
}

function renderVisual() {
  const v = report.visual_quality;
  $("#decoded-frames").textContent = formatInteger(v.decoded_frames);
  $("#resolution").textContent = v.resolutions.map(([w,h]) => `${w}×${h}`).join(", ");
  $("#brightness").textContent = v.brightness.q50.toFixed(3);
  $("#contrast").textContent = v.contrast.q50.toFixed(3);
  $("#visual-gallery").innerHTML = v.samples.map((sample) => {
    const x = sample.variation; const detail = `${x.yaw_deg}° · ${Math.round(x.mass_kg * 1000)}g · ${Math.round(x.size_m * 1000)}mm · ${x.color}`;
    return `<article class="visual-card"><header><strong>Episode ${String(sample.episode_index).padStart(3,"0")}</strong><span>${detail}</span></header><div class="timeline-grid">${sample.frames.map((frame) => `<div class="timeline-frame"><img src="${reportBase}${frame.path}" loading="lazy" alt="Episode ${sample.episode_index} at ${frame.label}"/><span>${frame.label}</span></div>`).join("")}</div></article>`;
  }).join("");
}

function render() {
  renderOverview(); renderIntegrity(); renderFilters(); renderCoverage(); renderMotion(); renderVisual();
}

loadReport().then(render).catch((error) => {
  $("#error-message").textContent = error.message;
  $("#error-state").hidden = false;
  console.error(error);
});
