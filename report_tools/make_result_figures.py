"""Result figures computed directly from the saved predictions.
Usage: python make_result_figures.py OUTDIR RUN_DIR x6   (order: SmolVLM2 GIT Qwen2-VL BLIP InternVL3 Gemma)
"""
import sys, csv, glob, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

out, dirs = sys.argv[1], sys.argv[2:]
MODELS = ["SmolVLM2\n500M", "GIT-base\nVQAv2", "Qwen2-VL\n2B", "BLIP VQA\nBase", "InternVL3\n1B", "Gemma 3\n4B"]
RELS = ["left", "right", "above", "below", "inside", "outside"]
ORANGE, GREEN, GRAY, INK, GRID = "#e8871e", "#1b7a4b", "#b9b6ad", "#1f1f1f", "#dddbd5"

plt.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Liberation Serif"],
                     "font.size": 9, "pdf.fonttype": 42, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "hatch.linewidth": 0.6})

stats = []
for d in dirs:
    a = glob.glob(d + "/analysis_*")[0]
    rows = list(csv.DictReader(open(a + "/image_details.csv")))
    assert len(rows) == 200
    g = {}
    for r in rows:
        g.setdefault(r["group_id"], {})[r["variant"]] = r
    ok = lambda r: r["prediction"] == r["ground_truth"]
    s = {"changed": 0, "base": 0, "base_ok": 0, "inv": 0, "inv_ok": 0}
    for v in g.values():
        b, sp = v["base"], v["spatial"]
        changed = b["prediction"] != sp["prediction"]
        s["changed"] += changed
        s["base"] += ok(b)
        s["base_ok"] += ok(b) and changed
        for k in ("brightness", "clutter", "occlusion"):
            s["inv"] += b["prediction"] == v[k]["prediction"]
            s["inv_ok"] += ok(b) and ok(v[k])
    s["rel"] = [100 * sum(map(ok, [r for r in rows if r["relation"] == x])) / sum(r["relation"] == x for r in rows) for x in RELS]
    stats.append(s)

def bars(name, series, labels, height=2.7, ref=None):
    fig, ax = plt.subplots(figsize=(7.0, height))
    w = 0.38
    for j, (leg, vals, color, hatch) in enumerate(series):
        x = [i + (j - 0.5) * (w + 0.03) for i in range(6)]
        ax.bar(x, vals, w, label=leg, color=color, edgecolor="white", linewidth=0, hatch=hatch, zorder=2)
        for xi, v, t in zip(x, vals, labels[j]):
            ax.text(xi, v + 1.5, t, ha="center", va="bottom", fontsize=7.5, linespacing=1.0, zorder=3)
    if ref:
        ax.axhline(ref[0], color=ORANGE, lw=1.2, ls=(0, (4, 2)), zorder=2.5, label=ref[1])
    ax.set_xticks(range(6), MODELS)
    ax.set_ylim(0, 117)
    ax.set_yticks(range(0, 101, 25))
    ax.set_ylabel("Percentage (%)")
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, handlelength=1.6)
    fig.tight_layout(pad=0.3)
    fig.savefig(f"{out}/{name}.pdf")
    fig.savefig(f"{out}/{name}.png", dpi=300)
    plt.close(fig)

overall = [100 * s["changed"] / 40 for s in stats]
cond = [100 * s["base_ok"] / s["base"] for s in stats]
fmt = lambda v: f"{v:.1f}"
bars("result_sensitivity",
     [("All spatial pairs", overall, ORANGE, None),
      ("Pairs with a correct base answer", cond, GREEN, None)],
     [[f"{fmt(v)}\n{s['changed']}/40" for v, s in zip(overall, stats)],
      [f"{fmt(v)}\n{s['base_ok']}/{s['base']}" for v, s in zip(cond, stats)]])

inv = [100 * s["inv"] / 120 for s in stats]
acc = [100 * s["inv_ok"] / 120 for s in stats]
bars("result_invariance",
     [("Prediction invariance", inv, GRAY, None), ("Invariant-pair accuracy", acc, GREEN, None)],
     [[fmt(v) for v in inv], [fmt(v) for v in acc]], ref=(50, "Constant-yes pair accuracy (50%)"))

# relation heatmap: orange (near chance) -> neutral -> green (correct)
cmap = LinearSegmentedColormap.from_list("ong", [(0, "#f0a35a"), (5/55, "#f4b579"), (30/55, "#f4f1e9"),
                                                 (40/55, "#b4d6c1"), (1, GREEN)])
data = [s["rel"] for s in stats]
fig, ax = plt.subplots(figsize=(6.0, 2.9))
im = ax.imshow(data, cmap=cmap, vmin=45, vmax=100, aspect="auto")
ax.set_xticks(range(6), [f"{r.capitalize()}\n(n={40 if i < 2 else 30})" for i, r in enumerate(RELS)])
ax.set_yticks(range(6), [m.replace("\n", " ") for m in MODELS])
ax.xaxis.tick_top()
for i, row in enumerate(data):
    for j, v in enumerate(row):
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8.5, color="white" if v >= 92 else INK)
ax.set_xticks([x - .5 for x in range(1, 6)], minor=True)
ax.set_yticks([y - .5 for y in range(1, 6)], minor=True)
ax.grid(which="minor", color="white", lw=1.5)
ax.tick_params(which="both", length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=[50, 60, 70, 80, 90, 100])
cb.set_label("Accuracy (%)")
cb.outline.set_visible(False)
cb.ax.tick_params(length=0)
fig.tight_layout(pad=0.3)
fig.savefig(f"{out}/result_relations.pdf")
fig.savefig(f"{out}/result_relations.png", dpi=300)
for m, s in zip(MODELS, stats):
    print(m.replace("\n", " "), s)
