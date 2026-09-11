let allRows = [];
let selectedId = null;

async function api(path, options) {
  const r = await fetch(path, options);
  const text = await r.text();
  let j;
  try {
    j = JSON.parse(text);
  } catch (_) {
    throw new Error(r.ok ? `Invalid JSON from ${path}` : `${r.status} ${path}`);
  }
  if (!j.ok) {
    throw new Error(j.error?.message || j.detail || r.statusText || "API error");
  }
  return j;
}

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function verdictPill(verdict) {
  if (!verdict) return '<span class="pill pill-muted">—</span>';
  const cls =
    verdict === "promote"
      ? "pill-promote"
      : verdict === "reject"
        ? "pill-reject"
        : verdict === "park"
          ? "pill-park"
          : "pill-muted";
  return `<span class="pill ${cls}">${esc(verdict)}</span>`;
}

function listVerdict(r) {
  if (r.record_class === "trusted" && r.display_verdict) return r.display_verdict;
  if (r.record_class === "open") return "open";
  if (r.record_class === "legacy") return "档案";
  return r.display_verdict || "—";
}

function filteredRows() {
  const strat = document.getElementById("strategyFilter").value;
  const q = document.getElementById("searchInput").value.trim().toLowerCase();
  const decisionOnly = document.getElementById("decisionOnly").checked;
  return allRows.filter((r) => {
    if (strat && (r.strategy || "") !== strat) return false;
    if (decisionOnly && !r.has_decision) return false;
    if (q) {
      const hay = [r.id, r.topic, r.strategy, r.hypothesis, r.decision_title]
        .join(" ")
        .toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
}

function renderTable() {
  const rows = filteredRows();
  const tbody = document.getElementById("rdBody");
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="muted">无匹配实验</td></tr>';
    return;
  }
  tbody.innerHTML = rows
    .map((r) => {
      const active = r.id === selectedId ? " rd-row-active" : "";
      const title = r.decision_title || r.hypothesis || r.topic || r.id;
      const shown =
        r.record_class === "trusted" ? r.display_verdict : listVerdict(r);
      return `<tr class="rd-row${active}" data-id="${esc(r.id)}">
        <td class="muted">${esc(r.date || "—")}</td>
        <td><strong>${esc(r.strategy || "—")}</strong></td>
        <td>
          <div class="rd-id">${esc(r.id)}</div>
          <div class="rd-sub">${esc(title.slice(0, 120))}</div>
        </td>
        <td class="muted">${esc((r.scorecard_fill && r.scorecard_fill.label) || "—")}</td>
        <td>${verdictPill(shown)}</td>
      </tr>`;
    })
    .join("");

  tbody.querySelectorAll(".rd-row").forEach((tr) => {
    tr.addEventListener("click", () => {
      selectedId = tr.getAttribute("data-id");
      renderTable();
      loadDetail(selectedId).catch((e) => {
        document.getElementById("detailPane").innerHTML =
          `<p class="err">${esc(String(e))}</p>`;
      });
    });
  });
}

function renderMarkdownBlock(title, text, path) {
  if (!text) return "";
  const preview = text.length > 4000 ? text.slice(0, 4000) + "\n\n…" : text;
  const file = path ? path.split("/").pop() : "";
  return `<section class="section">
    <h3>${esc(title)} ${file ? `<a class="raw-link" href="#" data-file="${esc(file)}">全文</a>` : ""}</h3>
    <pre class="md-preview">${esc(preview)}</pre>
  </section>`;
}

function fmtSlot(v) {
  return v == null || Number.isNaN(Number(v)) ? "—" : String(v);
}

function renderScorecard(scorecard) {
  const card = scorecard || {};
  const p1 = card.phase1 || {};
  const p3 = card.phase3 || {};
  const keys1 = ["ic", "icir", "auc", "lift_pp"];
  const keys3 = ["cagr", "calmar", "win_rate", "maxdd", "sharpe"];
  const row = (keys, values) =>
    `<tr>${keys.map((k) => `<td>${esc(fmtSlot(values[k]))}</td>`).join("")}</tr>`;
  const head = (keys) =>
    `<tr>${keys.map((k) => `<th>${esc(k)}</th>`).join("")}</tr>`;
  return `<section class="section">
    <h3>记分卡</h3>
    <h3>Phase 1（假设，不能结案）</h3>
    <table class="scorecard"><thead>${head(keys1)}</thead><tbody>${row(keys1, p1)}</tbody></table>
    <h3>Phase 3（法庭主 KPI）</h3>
    <table class="scorecard"><thead>${head(keys3)}</thead><tbody>${row(keys3, p3)}</tbody></table>
    <p class="muted rd-sub">空格 = 磁盘上没有产物，不是失败。不要整批重跑来填 verdict。</p>
  </section>`;
}

function renderLinks(links) {
  if (!links || !links.length) return "";
  return `<section class="section">
    <h3>results/ 产物链接</h3>
    <ul>${links.map((p) => `<li><code>${esc(p)}</code></li>`).join("")}</ul>
  </section>`;
}

function renderYamls(snippets) {
  const keys = Object.keys(snippets || {});
  if (!keys.length) return "";
  return keys
    .map(
      (k) => `<section class="section">
      <h3><code>${esc(k)}</code></h3>
      <pre class="md-preview">${esc((snippets[k] || "").slice(0, 2000))}</pre>
    </section>`
    )
    .join("");
}

async function loadDetail(id) {
  const pane = document.getElementById("detailPane");
  pane.innerHTML = '<p class="muted">加载详情…</p>';
  const { data } = await api(`/api/rd/experiment/${encodeURIComponent(id)}`);
  pane.innerHTML = `
    <header class="detail-header">
      <h2>${esc(data.id)}</h2>
      <p class="muted">${esc(data.strategy || "")} · ${esc(data.date || "")} · ${esc(data.topic || "")}</p>
      ${data.hypothesis ? `<p>${esc(data.hypothesis)}</p>` : ""}
      <p>类别：<code>${esc(data.record_class || "legacy")}</code>
         ${data.court && data.court.harness ? ` · ${esc(data.court.harness)}` : ""}</p>
      <p>决策：${verdictPill(data.display_verdict || (data.record_class === "legacy" ? "档案" : null))} ${esc(data.decision_title || "")}</p>
    </header>
    ${renderScorecard(data.court && data.court.scorecard)}
    ${renderLinks(data.results_links)}
    ${renderMarkdownBlock("README", data.readme_text, data.readme_path)}
    ${data.decision_text ? renderMarkdownBlock("DECISION", data.decision_text, data.decision_path) : ""}
    ${renderYamls(data.yaml_snippets)}
    <section class="section muted">
      <p>物料目录：<code>${esc(data.dir)}</code></p>
    </section>
  `;
  pane.querySelectorAll(".raw-link").forEach((a) => {
    a.addEventListener("click", (ev) => {
      ev.preventDefault();
      const file = a.getAttribute("data-file");
      if (!file) return;
      api(`/api/rd/experiment/${encodeURIComponent(id)}/raw/${encodeURIComponent(file)}`)
        .then(({ data: raw }) => {
          const w = window.open("", "_blank");
          if (w) {
            w.document.write(`<pre>${esc(raw.content)}</pre>`);
            w.document.title = file;
          }
        })
        .catch((e) => alert(String(e)));
    });
  });
}

function populateStrategyFilter(strategies) {
  const sel = document.getElementById("strategyFilter");
  const current = sel.value;
  sel.innerHTML =
    '<option value="">全部</option>' +
    (strategies || []).map((s) => `<option value="${esc(s)}">${esc(s)}</option>`).join("");
  if (current) sel.value = current;
}

async function refresh() {
  document.getElementById("statusLine").textContent = "加载中…";
  await api("/api/rd/refresh", { method: "POST" });
  const { data, meta } = await api("/api/rd/experiments");
  allRows = data || [];
  populateStrategyFilter(meta.strategies || []);
  renderTable();
  document.getElementById("statusLine").textContent =
    `${meta.count ?? allRows.length} experiments · ${new Date().toLocaleTimeString()}`;
  if (selectedId && allRows.some((r) => r.id === selectedId)) {
    await loadDetail(selectedId);
  }
}

document.getElementById("refreshBtn").addEventListener("click", () =>
  refresh().catch((e) => {
    document.getElementById("statusLine").textContent = String(e);
  })
);
document.getElementById("strategyFilter").addEventListener("change", renderTable);
document.getElementById("searchInput").addEventListener("input", renderTable);
document.getElementById("decisionOnly").addEventListener("change", renderTable);

refresh().catch((e) => {
  document.getElementById("statusLine").textContent = String(e);
});
