"""YOLO11 on eight GPUs: single-image latency, batch throughput, and the host launch factor from a rank-one fit."""
import matplotlib.pyplot as plt
import pandas as pd
from benchkit import additive_log_fit
from benchkit import usd_per_million_tokens as usd_per_million_units  # same arithmetic: price per hour over units per hour
from common import BLUE, GREEN, ORANGE, gpu_name, read, save_fig, save_table, update_summary

libs = read("libs_runs.csv")
libs["gpu"] = libs.gpu.map(gpu_name)
y = libs[(libs.lib == "yolo_gpu_only") & (libs.imgsz == 640)]

lat = y[y.batch == 1].pivot_table(index="gpu", columns="model", values="p50_ms", aggfunc="median")
thr = y[y.batch == 16].pivot_table(index="gpu", columns="model", values="fps_gpu", aggfunc="median")
save_table(lat.round(2), "latency_batch1_ms")
save_table(thr.round(1), "throughput_batch16_images_per_s")

# cost per million images for the medium model at batch 16
m = y[(y.model == "yolo11m") & (y.batch == 16)].groupby("gpu").agg(fps=("fps_gpu", "median"), dph=("dph_usd", "median"))
m["usd_per_million_images"] = [usd_per_million_units(r.dph, r.fps) for r in m.itertuples()]
save_table(m.round(3), "cost_per_million_images_yolo11m")

host, model, res = additive_log_fit(lat)
host = (host / host["RTX 5090"]).sort_values()
save_table(host.rename("launch_slowdown_vs_5090_host").round(2).to_frame(), "host_launch_factor")
mad = float((res.abs()).stack().median())

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for col, c in zip(lat.columns, (BLUE, ORANGE, GREEN)):
    axes[0].plot(lat.index, lat[col], marker="o", color=c, label=col)
    axes[1].plot(thr.index, thr[col], marker="o", color=c, label=col)
axes[0].set_ylabel("Median latency, one image (ms)")
axes[1].set_ylabel("Images per second at batch 16"); axes[1].set_yscale("log")
for ax in axes[:2]:
    ax.tick_params(axis="x", rotation=40)
axes[0].legend(frameon=False)
axes[2].barh(host.index[::-1], host.values[::-1], color=BLUE)
for i, v in enumerate(host.values[::-1]):
    axes[2].text(v, i, f" {v:.2f}", va="center", fontsize=8)
axes[2].set_xlabel("Host launch factor (RTX 5090 host = 1)")
fig.suptitle("YOLO11 at 640 px: small models are limited by the host, large batches by the GPU")
save_fig(fig, "yolo_latency_throughput_host.png")

gain = (thr["yolo11n"] / (1000 / lat["yolo11n"]))
update_summary({"yolo_n_latency_5090_ms": lat.loc["RTX 5090", "yolo11n"], "yolo_n_latency_4090_ms": lat.loc["RTX 4090", "yolo11n"], "yolo_x_throughput_h100": thr.loc["H100 NVL", "yolo11x"],
                "yolo_x_throughput_3090": thr.loc["RTX 3090", "yolo11x"], "host_factor_worst": host.iloc[-1], "host_factor_worst_gpu": host.index[-1], "rank_one_median_abs_log_residual": mad,
                "batching_gain_nano_5090": gain["RTX 5090"], "cost_per_m_images_cheapest": m.usd_per_million_images.min(), "cheapest_gpu_images": m.usd_per_million_images.idxmin()})
print(lat.round(2)); print(host.round(2))
