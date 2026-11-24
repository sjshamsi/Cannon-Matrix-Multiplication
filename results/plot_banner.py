from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# Conceptual banner: Cannon's algorithm on a 4 x 4 process grid after the initial skew.
# Each block is coloured by its inner index k, so the matching A and B blocks on every
# process share a colour: process (i, j) holds A[i, k] and B[k, j] with the same k.
here = Path(__file__).parent
P = 4
cmap = plt.get_cmap("viridis")
colours = [cmap(0.15 + 0.7 * k / (P - 1)) for k in range(P)]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))


def draw_grid(ax, label, shift):
    for i in range(P):
        for j in range(P):
            k = (i + j) % P
            ax.add_patch(Rectangle((j, P - 1 - i), 0.92, 0.92, color=colours[k], alpha=0.9))
            text = label(i, j, k)
            ax.text(j + 0.46, P - 1 - i + 0.46, text, ha="center", va="center", fontsize=11,
                    color="white" if k < P - 1 else "black")
    if shift == "left":
        for i in range(P):
            ax.add_patch(FancyArrowPatch((-0.03, P - 1 - i + 0.46), (-0.5, P - 1 - i + 0.46),
                                         arrowstyle="-|>", mutation_scale=14, color="#444444"))
    elif shift == "up":
        for j in range(P):
            ax.add_patch(FancyArrowPatch((j + 0.46, P - 0.05), (j + 0.46, P + 0.45),
                                         arrowstyle="-|>", mutation_scale=14, color="#444444"))
    ax.set_xlim(-0.6, P + 0.1)
    ax.set_ylim(-0.3, P + 0.55)
    ax.set_aspect("equal")
    ax.axis("off")


draw_grid(axes[0], lambda i, j, k: f"$A_{{{i}{k}}}$", "left")
axes[0].set_title("A after the skew: shifts left every step", fontsize=11)
draw_grid(axes[1], lambda i, j, k: f"$B_{{{k}{j}}}$", "up")
axes[1].set_title("B after the skew: shifts up every step", fontsize=11)
draw_grid(axes[2], lambda i, j, k: f"$C_{{{i}{j}}}$", None)
axes[2].set_title(r"$C_{ij}$ += $A_{ik}\,B_{kj}$, then shift, $P$ times", fontsize=11)
for i in range(P):
    for j in range(P):
        axes[2].patches[i * P + j].set_color("#d9d9d9")
for t in axes[2].texts:
    t.set_color("black")

fig.suptitle("Cannon's algorithm on a 4 × 4 process grid (colour = inner index k)", fontsize=12)
fig.tight_layout()
fig.savefig(here / "banner.png", dpi=150, bbox_inches="tight")
print("wrote", here / "banner.png")
