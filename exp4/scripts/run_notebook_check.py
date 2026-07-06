# coding=utf-8
"""Execute a notebook's CODE cells top-to-bottom in one namespace (mimics a manual run)."""
import sys, json, os
nb_path = sys.argv[1]
nb = json.load(open(nb_path, encoding="utf-8"))
ns = {"__name__": "__main__"}
os.chdir(os.path.dirname(nb_path))
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if not src.strip():
        continue
    print(f"\n===== running code cell {i} ({c.get('id')}) =====", flush=True)
    try:
        exec(compile(src, f"<cell {i}>", "exec"), ns)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"!!! cell {i} FAILED: {e}")
        sys.exit(1)
print("\nALL CELLS OK:", os.path.basename(nb_path))
