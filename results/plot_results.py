import csv
from pathlib import Path
import matplotlib.pyplot as plt

here = Path(__file__).parent
cannon = list(csv.DictReader((here / "cannon_runs.csv").open()))
rowblock = list(csv.DictReader((here / "rowblock_runs.csv").open()))
num = lambda r, k: float(r[k])

# ---------------- Figure 1: Cannon, N = 200000 ----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

# Left: the 8-tasks-per-node series. Only the largest P x P grid that fits is used,
# so adding ranks that don't complete the next square buys nothing.
series = sorted((r for r in cannon if r["tasks_per_node"] == "8"), key=lambda r: int(r["ranks"]))
x = [int(r["ranks"]) for r in series]
y = [num(r, "mmult_body_max_s") for r in series]
ax1.step(x, y, where="post", color="#3b6ea5", lw=1.5, alpha=0.5)
for r in series:
    hollow = r["variant"] == "allneighbours"
    ax1.plot(int(r["ranks"]), num(r, "mmult_body_max_s"), "o", color="#3b6ea5",
             mfc="white" if hollow else "#3b6ea5", ms=7, mew=1.5)
    if r["grid"] not in [s["grid"] for s in series[:series.index(r)]]:
        note = " (neighbour-\ncollective variant)" if hollow else ""
        ax1.annotate(f'{r["grid"]} grid{note}', (int(r["ranks"]), num(r, "mmult_body_max_s")),
                     textcoords="offset points", xytext=(-60 if hollow else 4, 8), fontsize=9)
ax1.set_xlabel("MPI ranks launched (8 per node, 14 threads each)")
ax1.set_ylabel("multiply time, slowest rank [s]")
ax1.set_title("Only P × P ranks do any work", fontsize=11)
ax1.set_xticks(x)
ax1.tick_params(axis="x", labelsize=8)

# Right: every run against the cores that actually take part, with the ideal 1/cores line
styles = {"8": ("o", "#3b6ea5", "8 ranks × 14 threads per node"), "16": ("s", "#d62728", "16 × 7"),
          "14": ("D", "#2ca02c", "14 × 8"), "4": ("^", "#e0a33b", "4 × 28")}
base = min(cannon, key=lambda r: int(r["active_cores"]))
c0, t0 = num(base, "active_cores"), num(base, "mmult_body_max_s")
cores = [800, 1900]
ax2.plot(cores, [t0 * c0 / c for c in cores], "k--", lw=1, label="perfect scaling from 896 cores")
for tpn, (m, col, lab) in styles.items():
    rs = [r for r in cannon if r["tasks_per_node"] == tpn]
    for r in rs:
        hollow = r["variant"] == "allneighbours"
        ax2.scatter(num(r, "active_cores"), num(r, "mmult_body_max_s"), marker=m, s=45, zorder=3,
                    facecolors="white" if hollow else col, edgecolors=col, linewidths=1.5,
                    label=lab if r is rs[0] else None)
    for c in sorted({r["active_cores"] for r in rs}):  # one label per core count: range of % of peak
        grp = [r for r in rs if r["active_cores"] == c]
        pcts = sorted(round(float(r["pct_of_nominal_peak"])) for r in grp)
        txt = f"{pcts[0]}%" if pcts[0] == pcts[-1] else f"{pcts[0]}–{pcts[-1]}%"
        ax2.annotate(txt, (float(c), max(num(r, "mmult_body_max_s") for r in grp)),
                     textcoords="offset points", xytext=(7, 0), fontsize=8.5, color=col)
ax2.set_xscale("log"); ax2.set_yscale("log")
ax2.set_xticks([896, 1134, 1400, 1792], ["896", "1134", "1400", "1792"])
ax2.set_yticks([200, 250, 300, 400], ["200", "250", "300", "400"])
ax2.minorticks_off()
ax2.set_xlabel("cores in the active P × P grid")
ax2.set_ylabel("multiply time, slowest rank [s]")
ax2.set_title("Strong scaling (labels: % of nominal FP64 peak)", fontsize=11)
ax2.plot([], [], "o", mfc="white", color="grey", label="hollow: neighbour-collective variant")
ax2.legend(fontsize=8.5, frameon=False, loc="upper right")
fig.suptitle("Cannon's algorithm, N = 200 000 (two 320 GB matrices), Leonardo DCGP, 8–16 nodes", fontsize=11.5)
fig.tight_layout()
fig.savefig(here / "cannon_scaling.png", dpi=150)

# ---------------- Figure 2: row-block Allgatherv, N = 20000 ----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
configs = [("112", "2025-11-12", "112 ranks × 1 thread (pure MPI)", "#d62728", "o"),
           ("16", "current", "16 × 7", "#9467bd", "s"), ("8", "current", "8 × 14", "#3b6ea5", "D"),
           ("4", "current", "4 × 28", "#e0a33b", "^"), ("2", "current", "2 × 56", "#2ca02c", "v")]
for tpn, code, lab, col, m in configs:
    rs = sorted((r for r in rowblock if r["tasks_per_node"] == tpn and r["code"] == code and r["total_max_s"]),
                key=lambda r: int(r["nodes"]))
    ax1.plot([int(r["nodes"]) for r in rs], [num(r, "total_max_s") for r in rs], marker=m, color=col, label=lab)
# The 16-node pure MPI job never finished: it was killed at its 20 min limit, so 1200 s is a lower bound
ax1.plot([14, 16], [406.7, 1200], ls=":", color="#d62728")
ax1.plot(16, 1200, marker="^", ms=9, color="#d62728", mfc="white", mew=1.5)
ax1.annotate("> 20 min\n(killed at the\ntime limit)", (16, 1200), textcoords="offset points",
             xytext=(-72, -12), fontsize=8.5, color="#d62728")
ax1.set_yscale("log")
ax1.set_ylim(3, 2500)
ax1.set_xlabel("nodes")
ax1.set_ylabel("total time, slowest rank [s]")
ax1.set_title("Total time (ranks per node × threads per rank)", fontsize=11)
ax1.legend(fontsize=8.5, frameon=False, loc="center right")

# Right: where the pure MPI time goes
rs = sorted((r for r in rowblock if r["code"] == "2025-11-12" and r["total_max_s"]), key=lambda r: int(r["nodes"]))
xs = range(len(rs))
comp = [num(r, "compute_max_s") for r in rs]
post = [num(r, "post_gather_max_s") for r in rs]
wait = [num(r, "untimed_max_s") for r in rs]
ax2.bar(xs, comp, color="#3b6ea5", label="dgemm")
ax2.bar(xs, post, bottom=comp, color="#e0a33b", label="posting the Iallgatherv")
ax2.bar(xs, wait, bottom=[a + b for a, b in zip(comp, post)], color="#d62728", alpha=0.8,
        label="untimed: mostly waiting for the gathers")
ax2.set_xticks(list(xs), [r["nodes"] for r in rs])
ax2.set_xlabel("nodes (112 ranks each)")
ax2.set_ylabel("time, slowest rank [s]")
ax2.set_title("Pure MPI: where the time goes", fontsize=11)
ax2.legend(fontsize=8.5, frameon=False, loc="upper left")
fig.suptitle("Row-block multiply with all-gathered B, N = 20 000, Leonardo DCGP", fontsize=11.5)
fig.tight_layout()
fig.savefig(here / "rowblock_scaling.png", dpi=150)

# ---------------- Figure 3: relative speedups ----------------
# No serial run exists at these sizes, so speedups are relative to the smallest run:
# Cannon against the 8-node run, row-block against 1 node with the same ranks-per-node layout.
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.6))

base = next(r for r in cannon if r["nodes"] == "8")
t0, a0 = num(base, "mmult_body_max_s"), num(base, "active_cores")
ax1.plot([8, 16], [1, 2], "k--", lw=1, label="ideal (nodes / 8)")
series = sorted((r for r in cannon if r["tasks_per_node"] == "8"), key=lambda r: int(r["nodes"]))
ax1.step([int(r["nodes"]) for r in series], [num(r, "active_cores") / a0 for r in series], where="post",
         color="grey", lw=1, ls=":", label="ideal for the cores in the P × P grid")
for tpn, (m, col, lab) in styles.items():
    rs = sorted((r for r in cannon if r["tasks_per_node"] == tpn), key=lambda r: int(r["nodes"]))
    ax1.plot([int(r["nodes"]) for r in rs], [t0 / num(r, "mmult_body_max_s") for r in rs], color=col,
             ls="-" if tpn == "8" else "none", marker=m, ms=0, label=None)
    for r in rs:
        hollow = r["variant"] == "allneighbours"
        ax1.plot(int(r["nodes"]), t0 / num(r, "mmult_body_max_s"), marker=m, color=col, ms=7, mew=1.5,
                 mfc="white" if hollow else col, ls="none", label=lab if r is rs[0] else None)
ax1.plot([], [], "o", mfc="white", color="grey", label="hollow: neighbour-collective variant")
ax1.set_xlabel("nodes allocated")
ax1.set_ylabel("speedup over 8 nodes")
ax1.set_title("Cannon, N = 200 000", fontsize=11)
ax1.legend(fontsize=8, frameon=False, loc="upper left")

layouts = [("112", "2025-11-12", "112 × 1 (pure MPI)", "#d62728", "o"), ("16", "current", "16 × 7", "#9467bd", "s"),
           ("8", "current", "8 × 14", "#3b6ea5", "D"), ("7", "current", "7 × 16", "#8c564b", "P"),
           ("4", "current", "4 × 28", "#e0a33b", "^")]
for ax, key, title in ((ax2, "total_max_s", "Row-block, N = 20 000: total time"),
                       (ax3, "compute_max_s", "Row-block, N = 20 000: dgemm only")):
    ax.plot([1, 16], [1, 16], "k--", lw=1, label="ideal")
    for tpn, code, lab, col, m in layouts:
        rs = [r for r in rowblock if r["tasks_per_node"] == tpn and r["code"] == code and r[key]]
        one = [num(r, key) for r in rs if r["nodes"] == "1"]
        ref = sum(one) / len(one)  # the 7 x 16 layout has two 1-node runs
        rs = sorted((r for r in rs if r["nodes"] != "1"), key=lambda r: int(r["nodes"]))
        ax.plot([1] + [int(r["nodes"]) for r in rs], [1] + [ref / num(r, key) for r in rs],
                marker=m, color=col, label=lab, ms=6)
    ax.set_xlabel("nodes")
    ax.set_ylabel("speedup over 1 node (same layout)")
    ax.set_title(title, fontsize=11)
ax2.set_yscale("log")
ax2.set_yticks([0.01, 0.1, 1, 10], ["0.01", "0.1", "1", "10"])
ax3.legend(fontsize=8, frameon=False, loc="upper left", title="ranks × threads per node", title_fontsize=8)
ax3.set_ylim(0, 17)
fig.suptitle("Relative speedups on Leonardo DCGP", fontsize=11.5)
fig.tight_layout()
fig.savefig(here / "speedup.png", dpi=150)
print("wrote cannon_scaling.png, rowblock_scaling.png and speedup.png")
