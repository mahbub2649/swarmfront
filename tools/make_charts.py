"""Draws the README charts from the training logs (TensorBoard event files) and the measured numbers.

    python tools/make_charts.py <path to the training results folder>

Needs tensorboard + matplotlib. Writes PNGs to media/charts/.
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = os.path.join(os.path.dirname(__file__), "..", "media", "charts")
os.makedirs(OUT, exist_ok=True)

# palette (validated: adjacent CVD dE >= 9.1 light); text in ink tokens, never series colour
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, AQUA, YELLOW, RED, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#b9b8b3"

plt.rcParams.update({
    "font.family": "Segoe UI", "font.size": 11, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False,
})


def scalars(pattern, tag):
    pts = {}
    for f in sorted(glob.glob(pattern)):
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        if tag in ea.Tags()["scalars"]:
            for s in ea.Scalars(tag):
                pts[s.step] = s.value
    steps = sorted(pts)
    return steps, [pts[s] for s in steps]


def ema(v, a=0.25):
    out, m = [], None
    for x in v:
        m = x if m is None else a * x + (1 - a) * m
        out.append(m)
    return out


def title(fig, t, sub):
    h = fig.get_figheight()                     # fixed inch offsets, so short figures don't overlap
    fig.text(0.035, 1 - 0.22 / h, t, fontsize=16, fontweight="bold", va="top")
    fig.text(0.035, 1 - 0.6 / h, sub, fontsize=11, color=INK2, va="top")


def spread(ys, gap):
    """Nudge end-of-line label positions apart so they never overlap (keeps order)."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150)
    plt.close(fig)
    print("wrote", name)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


# ---------- 1. training throughput ----------
def throughput():
    rows = [("ML-Agents, in the Unity Editor", 0.0266, GREY, "26.6k"),
            ("ML-Agents, 6 headless builds", 0.0627, GREY, "62.7k  (1x)"),
            ("GPU sim, slowest run", 2.44, BLUE, "2.44M  (39x)"),
            ("GPU sim, fastest run", 3.26, BLUE, "3.26M  (52x)")]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    fig.subplots_adjust(left=0.27, right=0.95, top=0.74, bottom=0.14)
    y = range(len(rows))[::-1]
    for yi, (lab, v, c, txt) in zip(y, rows):
        ax.barh(yi, v, height=0.55, color=c)
        ax.text(v + 0.04, yi, txt, va="center", fontsize=11, color=INK, fontweight="bold" if c == BLUE else "normal")
    ax.set_yticks(list(y), [r[0] for r in rows])
    ax.set_xlim(0, 3.8)
    ax.set_xlabel("environment steps per minute (millions), averaged over whole runs")
    grid(ax, "x")
    ax.spines["left"].set_visible(False)
    title(fig, "Training speed: about 50x faster on the GPU",
          "Same PC, one RTX 3060. ML-Agents was stuck on one CPU core; the GPU version runs physics + PPO on the card.")
    save(fig, "throughput.png")


# ---------- 2. wall clock for one 200M-step run ----------
def wallclock():
    fig, ax = plt.subplots(figsize=(10, 2.9))
    fig.subplots_adjust(left=0.27, right=0.95, top=0.66, bottom=0.2)
    rows = [("ML-Agents (estimated)", 53.2, GREY, "~53 hours"), ("GPU sim (measured)", 1.25, BLUE, "75 minutes")]
    for yi, (lab, v, c, txt) in zip([1, 0], rows):
        ax.barh(yi, v, height=0.55, color=c)
        ax.text(v + 0.8, yi, txt, va="center", fontsize=11, fontweight="bold" if c == BLUE else "normal")
    ax.set_yticks([1, 0], [r[0] for r in rows])
    ax.set_xlim(0, 62)
    ax.set_xlabel("hours")
    grid(ax, "x")
    ax.spines["left"].set_visible(False)
    title(fig, "One 200M-step training run", "strike_ground_v3: measured on the GPU, ML-Agents time = 200M / 62.7k steps per minute")
    save(fig, "wallclock.png")


# ---------- 3. pilot learning curves ----------
def pilot():
    run = os.path.join(RESULTS, "strike_gpu_v6", "Wing", "events*")
    series = [("All strikes", "Strike/Success rate", BLUE),
              ("Parked ground target", "Strike/Success rate GroundStatic", ORANGE),
              ("Moving ground target", "Strike/Success rate GroundMoving", AQUA),
              ("Moving air target", "Strike/Success rate AirMoving", YELLOW)]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    fig.subplots_adjust(left=0.08, right=0.78, top=0.8, bottom=0.12)
    ends = []
    for lab, tag, c in series:
        s, v = scalars(run, tag)
        x = [k / 1e6 for k in s]
        y = [100 * k for k in ema(v)]
        ax.plot(x, y, color=c, linewidth=2, label=lab, solid_capstyle="round")
        ends.append((x[-1], y[-1], lab))
    for (ex, ey, lab), ly in zip(ends, spread([e[1] for e in ends], 5.5)):
        ax.text(ex + 2.5, ly, f"{lab}  {ey:.0f}%", va="center", fontsize=10, color=INK)
    ax.axvline(100.14, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(101.5, 4, "checkpoint used in the demo\n100M steps, 40 min of training", fontsize=9.5, color=INK2, va="bottom")
    ax.set_xlim(0, 192)
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 25), [f"{k}%" for k in range(0, 101, 25)])
    ax.set_xlabel("training steps (millions)")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.0), ncol=4, frameon=False, fontsize=9.5, handlelength=1.6,
              borderaxespad=-2.2)
    grid(ax)
    title(fig, "Pilot network learning to strike", "Strike success rate during training (GPU sim, smoothed). 190M steps in 75 minutes.")
    save(fig, "pilot_training.png")


# ---------- 4. commander self-play ----------
def commander():
    s, v = scalars(os.path.join(RESULTS, "commander_v3", "events*"), "Eval/Win vs scripted")
    per_update = 167149568 / 8184          # decisions per update (status.json)
    x = [k * per_update / 1e6 for k in s]
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(left=0.08, right=0.8, top=0.8, bottom=0.12)
    ax.plot(x, [100 * k for k in v], color=BLUE, alpha=0.22, linewidth=1)
    sm = [100 * k for k in ema(v, 0.08)]
    ax.plot(x, sm, color=BLUE, linewidth=2.2)
    ax.text(x[-1] + 2, sm[-1], f"AI commander\n{sm[-1]:.0f}% wins", va="center", fontsize=10)
    ax.axhline(50, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(x[-1] + 2, 50, "even match", va="center", fontsize=9.5, color=INK2)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, x[-1])
    ax.set_yticks(range(0, 101, 25), [f"{k}%" for k in range(0, 101, 25)])
    ax.set_xlabel("commander decisions trained (millions)")
    grid(ax)
    title(fig, "Commander learning by self-play",
          "Win rate against my scripted commander in regular eval games (faint = raw, line = smoothed)")
    save(fig, "commander_training.png")


# ---------- 5. 32-match benchmark ----------
def benchmark(wins=28, losses=4):
    fig = plt.figure(figsize=(10, 3.3))
    title(fig, f"32 full matches in Unity: AI commander won {wins}, lost {losses}",
          "8 headless copies of the game in parallel, 2 matches each. Each square is one match (sorted).")
    ax = fig.add_axes([0.035, 0.12, 0.93, 0.56])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 2)
    ax.axis("off")
    for i in range(wins + losses):
        c = BLUE if i < wins else RED
        col, row = i % 16, 1 - i // 16
        ax.add_patch(FancyBboxPatch((col + 0.08, row + 0.1), 0.84, 0.8, boxstyle="round,pad=0,rounding_size=0.08",
                                    facecolor=c, edgecolor="none"))
    fig.text(0.035, 0.05, f"■ Blue (AI commander) wins: {wins}", color=INK, fontsize=10.5)
    fig.text(0.33, 0.05, f"■ Red (scripted commander) wins: {losses}", color=INK, fontsize=10.5)
    fig.text(0.035, 0.05, "■", color=BLUE, fontsize=10.5)
    fig.text(0.33, 0.05, "■", color=RED, fontsize=10.5)
    save(fig, "benchmark.png")


if __name__ == "__main__":
    throughput()
    wallclock()
    pilot()
    commander()
    benchmark()
