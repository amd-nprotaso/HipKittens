# data from https://github.com/ROCm/aiter/pull/2039

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Data ────────────────────────────────────────────────────────────────────
# Raw latency (us) from ROCm/aiter PR #2039
# We plot gain (%) of HK over ASM, grouped by context length, for each GPU.
# Alternatively we plot latency side-by-side for a single GPU (MI355X shown here).

raw_data = {
    "batch": [1,3,5,16,32,64,128,256,1,3,5,16,32,64,128,256,1,3,5,16,32,64,128,256,1,3,5,16,32,64,128,256,1,3,5,16,32,64,128,256,1,3,5,16,32,64,128,256,1,3,5,16,32,64,128,256,1,3,5,16,32,64,128,256],
    "ctx":   [21]*8 + [64]*8 + [256]*8 + [512]*8 + [1200]*8 + [3200]*8 + [5200]*8 + [8192]*8,
    "MI308_ASM": [19.207,18.847,19.790,20.403,20.990,22.294,40.731,80.261, 19.824,19.561,19.444,23.391,24.107,25.248,47.431,91.807, 20.467,20.767,22.468,31.444,42.824,42.792,82.497,149.343, 20.517,23.070,27.116,37.135,51.116,66.403,126.244,228.114, 24.934,27.258,30.750,51.653,77.938,129.815,231.731,442.955, 30.972,37.146,42.968,87.987,152.941,280.120,529.806,1034.000, 36.864,44.092,55.478,127.203,228.364,430.384,834.893,1643.000, 45.870,52.446,74.105,180.677,339.160,656.085,1306.000,2535.000],
    "MI308_HK":  [17.038,16.712,16.321,16.738,17.456,19.439,34.116,64.620, 16.986,16.896,16.325,18.304,19.220,21.192,38.811,74.132, 17.438,17.284,18.269,27.551,38.019,38.556,72.393,130.599, 17.679,18.552,22.447,31.175,45.523,61.583,116.060,204.625, 21.208,22.167,25.271,46.402,72.156,125.196,218.986,415.022, 27.151,32.197,37.479,80.378,144.645,269.860,506.762,985.032, 32.926,38.415,48.857,119.102,217.355,414.562,802.692,1578.000, 41.406,46.159,67.498,170.463,323.840,633.899,1237.000,2441.000],
    "MI300_ASM": [15.491,15.089,15.545,16.445,17.406,18.879,20.608,40.928, 14.717,15.629,16.174,17.222,21.761,20.885,21.909,53.850, 15.457,18.607,20.083,23.630,27.841,33.535,37.518,78.283, 15.649,18.510,20.190,25.369,32.503,47.793,70.484,118.997, 18.857,22.213,24.832,33.545,49.243,69.173,111.467,193.077, 22.918,26.559,31.488,54.434,82.846,134.910,224.926,410.581, 26.268,30.645,36.625,70.020,113.753,193.552,334.434,631.849, 32.707,37.842,47.956,95.467,159.378,277.698,507.749,967.071],
    "MI300_HK":  [12.769,12.850,13.041,13.263,14.393,15.436,17.403,34.653, 12.469,13.266,14.165,15.076,19.504,17.334,19.641,48.910, 13.135,15.951,17.080,20.216,23.844,29.236,35.474,73.383, 12.863,17.226,16.749,20.838,29.426,44.685,66.912,114.201, 16.321,19.053,21.075,31.638,45.236,65.279,111.045,191.726, 21.453,23.938,27.464,50.052,81.334,133.696,227.313,413.005, 25.191,27.946,33.886,66.088,109.272,189.291,335.899,637.249, 30.984,34.923,43.812,89.756,158.834,274.036,503.463,972.999],
    "MI355_ASM": [12.106,13.021,12.942,13.592,14.822,15.120,16.185,20.367, 12.640,13.249,13.570,14.426,17.563,19.144,19.598,24.037, 13.516,14.680,15.179,16.592,18.719,22.631,29.272,35.111, 13.611,14.285,15.425,19.863,21.444,26.765,39.112,48.466, 15.552,17.116,18.167,22.901,28.597,39.773,55.859,85.899, 19.134,20.962,23.241,31.249,45.933,67.361,111.869,203.137, 23.672,25.671,27.923,38.713,59.565,94.513,176.153,320.164, 28.166,29.872,31.737,50.289,79.926,137.594,262.967,505.226],
    "MI355_HK":  [10.420,11.315,10.956,12.070,13.211,14.040,14.829,19.300, 10.969,11.744,12.003,12.628,17.151,17.978,19.151,22.231, 11.598,12.691,13.746,14.481,16.931,20.786,28.109,34.018, 11.605,12.173,13.573,17.519,19.305,24.684,37.030,47.843, 14.066,15.775,16.516,20.431,27.588,38.175,56.666,87.206, 18.014,20.314,21.107,29.179,44.502,66.104,116.947,211.877, 22.485,24.920,26.462,36.852,58.814,95.612,184.013,334.044, 27.350,29.204,30.373,49.987,79.506,144.931,272.204,518.979],
    "MI350_ASM": [13.936,14.908,14.737,16.030,16.531,17.408,18.146,22.746, 13.340,13.892,14.228,14.864,18.309,18.859,20.022,26.294, 14.133,15.221,15.982,17.455,20.015,24.122,31.857,41.271, 14.172,15.321,16.147,20.820,22.959,28.927,45.163,58.124, 16.533,18.056,19.173,24.276,30.986,45.232,73.214,117.395, 20.393,22.304,23.842,33.806,52.335,85.925,151.557,286.616, 24.131,26.081,28.040,42.756,72.462,128.285,247.371,453.691, 29.851,32.011,34.213,58.849,101.578,188.369,362.794,712.819],
    "MI350_HK":  [11.174,11.942,12.066,14.070,15.380,16.029,17.176,21.405, 11.501,12.093,12.380,13.002,16.937,17.243,18.689,24.681, 12.248,13.440,14.314,15.187,17.930,22.155,30.598,41.290, 12.034,13.517,14.118,18.391,20.371,27.077,44.081,57.529, 14.696,16.464,17.306,21.289,28.125,44.958,74.805,123.767, 19.010,21.245,22.216,30.808,51.460,86.610,159.482,301.627, 22.997,25.225,26.193,39.511,74.322,132.148,253.053,477.938, 29.525,31.838,33.143,56.064,106.917,198.398,382.139,745.898],
}

df = pd.DataFrame(raw_data)
df["gain_MI308"] = (df["MI308_ASM"] - df["MI308_HK"]) / df["MI308_ASM"] * 100
df["gain_MI300"] = (df["MI300_ASM"] - df["MI300_HK"]) / df["MI300_ASM"] * 100
df["gain_MI355"] = (df["MI355_ASM"] - df["MI355_HK"]) / df["MI355_ASM"] * 100
df["gain_MI350"] = (df["MI350_ASM"] - df["MI350_HK"]) / df["MI350_ASM"] * 100

# ── Style (matches HipKittens plot.py) ──────────────────────────────────────
colors = ["#8E69B8", "#E59952", "#68AC5A", "#7CB9BC", "#DE836B"]
gpu_colors = {"MI308": colors[0], "MI300": colors[2], "MI355": colors[3], "MI350": colors[4]}
fontsize = 13

# ── Plot 1: Latency comparison on MI355X (small-batch regime) ───────────────
# Show batch=1..32, ctx=64 as a representative slice
ctx_filter = 64
slice_df = df[df["ctx"] == ctx_filter].copy()
batches = slice_df["batch"].values

x = np.arange(len(batches))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
bars_asm = ax.bar(x - width/2, slice_df["MI355_ASM"], width, label="ASM (baseline)", color=colors[0])
bars_hk  = ax.bar(x + width/2, slice_df["MI355_HK"],  width, label="HipKittens",     color=colors[3])

for bar, val in zip(bars_asm, slice_df["MI355_ASM"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val:.1f}", ha="center", va="bottom", fontsize=9)
for bar, val in zip(bars_hk, slice_df["MI355_HK"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val:.1f}", ha="center", va="bottom", fontsize=9)

ax.set_xlabel("Batch Size", fontsize=fontsize)
ax.set_ylabel("Latency (μs) — lower is better", fontsize=fontsize)
ax.set_title(f"MLA Decode Latency: HipKittens vs ASM on MI355X (ctx={ctx_filter})", fontsize=fontsize)
ax.set_xticks(x)
ax.set_xticklabels(batches, fontsize=fontsize)
ax.tick_params(axis="y", labelsize=fontsize)
ax.legend(fontsize=fontsize)
plt.tight_layout()
plt.savefig("mla_decode_latency_mi355x.png", dpi=300, bbox_inches="tight")
print("Saved mla_decode_latency_mi355x.png")
plt.show()

# ── Plot 2: % Gain of HK over ASM across all GPUs, by context length ────────
# Average gain per (GPU, ctx) pair
ctx_order = [21, 64, 256, 512, 1200, 3200, 5200, 8192]
gpus = ["MI308", "MI300", "MI355", "MI350"]

avg_gains = {gpu: [] for gpu in gpus}
for ctx in ctx_order:
    sub = df[df["ctx"] == ctx]
    for gpu in gpus:
        avg_gains[gpu].append(sub[f"gain_{gpu}"].mean())

x = np.arange(len(ctx_order))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 5))
for i, gpu in enumerate(gpus):
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, avg_gains[gpu], width,
                  label=gpu, color=gpu_colors[gpu])
    for bar, val in zip(bars, avg_gains[gpu]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Sequence Length", fontsize=fontsize)
ax.set_ylabel("Avg. Latency Reduction vs ASM (%)", fontsize=fontsize)
ax.set_title("MLA Decode: HipKittens Speed Gain over Hand-Written Assembly\n(averaged over batch sizes)", fontsize=fontsize)
ax.set_xticks(x)
ax.set_xticklabels(ctx_order, fontsize=fontsize)
ax.tick_params(axis="y", labelsize=fontsize)
ax.legend(fontsize=fontsize)
plt.tight_layout()
plt.savefig("mla_decode_gain_by_ctx.png", dpi=300, bbox_inches="tight")
print("Saved mla_decode_gain_by_ctx.png")
plt.show()

# ── Summary stats ────────────────────────────────────────────────────────────
print("\n── Summary: avg gain over ASM (all batches & ctx) ──")
for gpu in gpus:
    gains = df[f"gain_{gpu}"]
    print(f"  {gpu}: avg={gains.mean():.1f}%  max={gains.max():.1f}%  min={gains.min():.1f}%")
