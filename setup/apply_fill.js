// 通用 notebook 填充器: node apply_fill.js <notebook> <fill.json>
const fs = require("fs");
const nbPath = process.argv[2];
const fillPath = process.argv[3];
const items = JSON.parse(fs.readFileSync(fillPath, "utf8"));
const nb = JSON.parse(fs.readFileSync(nbPath, "utf8"));

function toLines(t){ const p = t.split("\n"); return p.map((l, i) => (i < p.length - 1 ? l + "\n" : l)); }

let applied = 0;
for (const it of items){
  let hit = false;
  for (const c of nb.cells){
    if (it.celltype && c.cell_type !== it.celltype) continue;
    const src = Array.isArray(c.source) ? c.source.join("") : c.source;
    const match = it.find === "__EMPTY_CODE__"
      ? (c.cell_type === "code" && src.trim() === "")
      : src.includes(it.find);
    if (match){
      c.source = toLines(it.code);
      if (c.cell_type === "code"){ c.outputs = []; c.execution_count = null; }
      applied++; hit = true; break;
    }
  }
  if (!hit) console.log("WARN not found:", it.find);
}
fs.writeFileSync(nbPath, JSON.stringify(nb, null, 1), "utf8");
console.log("applied", applied, "/", items.length);
