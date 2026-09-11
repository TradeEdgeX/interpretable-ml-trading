function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function currentPath() {
  const u = new URL(window.location.href);
  return u.searchParams.get("path") || "";
}

async function load() {
  const path = currentPath();
  const r = await fetch(`/api/browse?path=${encodeURIComponent(path)}`);
  const j = await r.json();
  if (!j.ok) throw new Error(j.detail || "browse failed");
  const data = j.data;
  const crumbs = (data.crumbs || [])
    .map((c, i) => {
      const href = c.path ? `/browse?path=${encodeURIComponent(c.path)}` : "/browse";
      const label = esc(c.label);
      if (i === (data.crumbs || []).length - 1) return `<span>${label}</span>`;
      return `<a href="${href}">${label}</a> /`;
    })
    .join(" ");
  document.getElementById("crumbs").innerHTML = crumbs;
  const rows = data.entries || [];
  if (data.missing_root) {
    document.getElementById("browseBody").innerHTML =
      '<tr><td colspan="3" class="muted">本机还没有 results/ 目录。</td></tr>';
    return;
  }
  if (!rows.length) {
    document.getElementById("browseBody").innerHTML =
      '<tr><td colspan="3" class="muted">空目录</td></tr>';
    return;
  }
  document.getElementById("browseBody").innerHTML = rows
    .map((e) => {
      if (e.kind === "dir") {
        const href = `/browse?path=${encodeURIComponent(e.rel_path)}`;
        return `<tr><td>目录</td><td><a href="${href}">${esc(e.name)}</a></td><td class="muted">${esc(e.size)}</td></tr>`;
      }
      return `<tr><td>文件</td><td><a href="${esc(e.href)}" target="_blank" rel="noopener">${esc(e.name)}</a></td><td class="muted">${esc(e.size)}</td></tr>`;
    })
    .join("");
}

load().catch((e) => {
  document.getElementById("browseBody").innerHTML =
    `<tr><td colspan="3" class="err">${esc(String(e))}</td></tr>`;
});
