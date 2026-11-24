import csv
import math
import re
from collections import defaultdict
from pathlib import Path

# Turns the raw per-rank .times files from the Leonardo runs into two summary tables:
#   cannon_runs.csv    one row per Cannon run (N = 200000)
#   rowblock_runs.csv  one row per row-block Allgatherv run (N = 20000)

here = Path(__file__).parent
CORE_PEAK_GFLOPS = 2.0 * 32  # Xeon Platinum 8480+: 2.0 GHz base x 32 DP FLOP/cycle (2 AVX-512 FMA units)

# Which code produced which runs (matched by submission time against the git history)
CANNON_VARIANT = {
    "13-11_09": "isend_irecv",    # same algorithm as inc/CanonM.hpp, plus a timer around each dgemm (versions/)
    "13-11_23": "allneighbours",  # inc/CanonM_Allneighbours.hpp
    "14-11": "isend_irecv",       # inc/CanonM.hpp
}
ROWBLOCK_VERSION = {
    "14-11_03": "2025-11-12",     # versions/ParallelCMatrix_2025-11-12.hpp (pure MPI runs)
    "14-11_07": "current",        # inc/ParallelCMatrix.hpp (hybrid runs)
}


def read_times(path):
    text = path.read_text()
    header = dict(re.findall(r"^#(\w+):\s*(.*)$", text, re.M))
    per_rank = defaultdict(dict)  # rank -> subject -> total ms
    for m in re.finditer(r"^(\d+),\s*(\w+),\s*([\d.]+),\s*\d+,\s*[\d.]+\s*$", text, re.M):
        per_rank[int(m.group(1))][m.group(2)] = float(m.group(3))
    return header, per_rank


def lookup(table, filename):
    for prefix, value in table.items():
        if filename.split("__")[1].startswith(prefix):
            return value
    raise KeyError(filename)


def config(header):
    return dict(nodes=int(header["Requested_Nodes"]), tasks_per_node=int(header["Requested_TPN"]),
                cpus_per_task=int(header["Requested_CPT"]), omp_threads=int(header["Exported_OMP_Threads"]),
                blas_threads=int(header["Exported_OB_Threads"]), N=int(header["Matrix_Order"]),
                ranks=int(header["SLURM_NTASKS"]), job=header["SLURM_JOB_ID"], date=header["Date_Time"])


def max_over_ranks(per_rank, subject):
    vals = [r[subject] for r in per_rank.values() if subject in r]
    return round(max(vals) / 1e3, 3) if vals else ""


rows = []
for f in sorted((here / "raw/cannon").glob("*.times")):
    header, per_rank = read_times(f)
    c = config(header)
    P = math.isqrt(c["ranks"])
    active_cores = P * P * c["cpus_per_task"]
    body = max_over_ranks(per_rank, "mmultfn_body")
    tflops = 2 * c["N"] ** 3 / body / 1e12
    rows.append(dict(c, variant=lookup(CANNON_VARIANT, f.name), grid=f"{P}x{P}", active_ranks=P * P,
                     idle_ranks=c["ranks"] - P * P, active_cores=active_cores,
                     mmult_body_max_s=body, mmult_body_min_s=round(min(r["mmultfn_body"] for r in per_rank.values()) / 1e3, 3),
                     dgemm_max_s=max_over_ranks(per_rank, "cblas"),
                     tflops=round(tflops, 2),
                     pct_of_nominal_peak=round(100 * tflops * 1e3 / (active_cores * CORE_PEAK_GFLOPS), 1)))
rows.sort(key=lambda r: (r["active_cores"], r["nodes"]))
with (here / "cannon_runs.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} Cannon runs -> cannon_runs.csv")

rows = []
for f in sorted((here / "raw/rowblock").glob("*.times")):
    header, per_rank = read_times(f)
    c = config(header)
    version = lookup(ROWBLOCK_VERSION, f.name)
    if not per_rank:  # the 16-node pure MPI job hit its 20 min wall-clock limit
        rows.append(dict(c, code=version, total_max_s="", compute_max_s="", post_gather_max_s="",
                         untimed_max_s="", note="timed out after 20 min"))
        continue
    compute, post = ("dgemm", "gatherfn") if version == "2025-11-12" else ("calc", "comm")
    untimed = []  # total minus every timed piece: mostly waiting on the all-gathers (and buffer allocation)
    for r in per_rank.values():
        untimed.append(r["total"] - sum(v for k, v in r.items() if k != "total"))
    rows.append(dict(c, code=version, total_max_s=max_over_ranks(per_rank, "total"),
                     compute_max_s=max_over_ranks(per_rank, compute),
                     post_gather_max_s=max_over_ranks(per_rank, post),
                     untimed_max_s=round(max(untimed) / 1e3, 3), note=""))
rows.sort(key=lambda r: (r["code"], r["tasks_per_node"], r["nodes"]))
with (here / "rowblock_runs.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} row-block runs -> rowblock_runs.csv")
