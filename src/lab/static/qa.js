function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function pick(obj, lang) {
  if (!obj || typeof obj !== "object") return "";
  return obj[lang] || obj.zh || obj.en || "";
}

let payload = null;

function render() {
  if (!payload) return;
  const lang = document.getElementById("langSelect").value;
  const q = document.getElementById("searchInput").value.trim().toLowerCase();
  document.getElementById("qaTitle").textContent = pick(payload.title, lang) || "Q&A";
  document.getElementById("qaBlurb").textContent = pick(payload.blurb, lang);
  document.getElementById("qaMeta").innerHTML =
    `源 <code>${esc(payload.source || "docs/agent/rd_qa.yaml")}</code> · <a href="/rd">实验管理</a>`;

  const items = (payload.items || []).filter((item) => {
    if (!q) return true;
    const hay = [
      item.id,
      (item.tags || []).join(" "),
      pick(item.q, lang),
      pick(item.a, lang),
    ]
      .join(" ")
      .toLowerCase();
    return hay.includes(q);
  });
  document.getElementById("countLine").textContent =
    `${items.length}/${(payload.items || []).length}`;
  const open = items.length <= 3;
  document.getElementById("qaList").innerHTML = items
    .map(
      (item) => `<details class="qa-item"${open ? " open" : ""}>
        <summary>
          <span class="qa-q">${esc(pick(item.q, lang))}</span>
          <span class="qa-tags">${esc((item.tags || []).join(" · "))}</span>
        </summary>
        <div class="qa-a">${esc(pick(item.a, lang))}</div>
      </details>`
    )
    .join("") || `<p class="muted">${lang === "en" ? "No matching questions" : "无匹配问答"}</p>`;
}

async function load() {
  const r = await fetch("/api/rd/qa");
  const j = await r.json();
  if (!j.ok) throw new Error(j.detail || "qa failed");
  payload = j.data;
  render();
}

document.getElementById("searchInput").addEventListener("input", render);
document.getElementById("langSelect").addEventListener("change", render);
load().catch((e) => {
  document.getElementById("qaList").innerHTML = `<p class="err">${esc(String(e))}</p>`;
});
