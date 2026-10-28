"""SAM 2, Grounding DINO and Mask R-CNN: latency per GPU, attempts and failures, additive card x model fit, the 505 ms outlier, Detectron2 against its stand-in."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from benchkit import additive_log_fit
from common import BLUE, GREEN, ORANGE, RED, read, save_fig, save_table, update_summary

seg = read("segmentation_attempts.csv")
names = {"sam2": "SAM 2", "grounding_dino": "Grounding DINO", "maskrcnn_torchvision": "Mask R-CNN (torchvision)"}
ok = seg[(seg.status == "ok") & seg.lib.isin(names)].copy()
ok["model"] = ok.lib.map(names)

tab = ok.groupby(["gpu", "model"]).p50_ms.median().unstack()
save_table(tab.round(2), "median_latency_ms")
att = seg.groupby(["lib", "status"]).size().unstack(fill_value=0)
save_table(att, "attempts_by_status")

clean = ok[~((ok.lib == "sam2") & (ok.p50_ms > 400))]
cell = clean.groupby(["gpu", "model"]).p50_ms.median().unstack()
card, mod, res = additive_log_fit(cell)
save_table(pd.DataFrame({"card_factor_ms": card}).sort_values("card_factor_ms").round(1), "card_factors")
sd_res = float(np.nanstd(res.values))
sam = ok[ok.lib == "sam2"].set_index("gpu").p50_ms
predicted_pro = float(card["RTX PRO 6000"] * mod["SAM 2"])

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
order = tab.mean(axis=1).sort_values().index
xx = np.arange(len(order))
w = 0.27
for i, (m, c) in enumerate(zip(tab.columns, (BLUE, ORANGE, GREEN))):
    axes[0].bar(xx + (i - 1) * w, tab.loc[order, m], w, label=m, color=c)
axes[0].set_xticks(xx)
axes[0].set_xticklabels(order, rotation=25, fontsize=8)
axes[0].set_yscale("log")
axes[0].set_ylabel("Median latency, one image (ms)")
axes[0].legend(frameon=False, fontsize=8)
pred = pd.DataFrame(np.add.outer(np.log(card.values), np.log(mod.values)), index=card.index, columns=mod.index)
mm = pd.DataFrame({"measured": cell.stack(), "predicted": np.exp(pred).stack()}).dropna()
axes[1].scatter(mm.predicted, mm.measured, color=BLUE, label="fitted cells")
axes[1].scatter([predicted_pro], [sam["RTX PRO 6000"].max()], color=RED, label="RTX PRO 6000, SAM 2 (excluded from the fit)")
lim = [mm.min().min() * 0.8, sam.max() * 1.3]
axes[1].plot(lim, lim, color="grey", ls="--")
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("Predicted by card factor x model factor (ms)")
axes[1].set_ylabel("Measured (ms)")
axes[1].legend(frameon=False, fontsize=7)
save_fig(fig, "segmentation_latency.png")

d2 = seg[seg.lib.str.startswith("detectron2") & (seg.status == "ok")].set_index("lib").p50_ms
tv = ok[(ok.lib == "maskrcnn_torchvision") & (ok.gpu == "RTX 3090")].p50_ms.median()
update_summary({"attempts_total": len(seg), "attempts_failed": int((seg.status != "ok").sum()), "sam2_5090_ms": tab.loc["RTX 5090", "SAM 2"], "sam2_gb10_ms": tab.loc["GB10", "SAM 2"],
                "sam2_pro6000_outlier_ms": sam["RTX PRO 6000"].max(), "sam2_pro6000_predicted_ms": predicted_pro, "gdino_5090_ms": tab.loc["RTX 5090", "Grounding DINO"],
                "gdino_4090_ms": tab.loc["RTX 4090", "Grounding DINO"], "gdino_3090_ms": tab.loc["RTX 3090", "Grounding DINO"], "additive_fit_sd_log": sd_res,
                "detectron2_mask_ms": d2["detectron2"], "detectron2_faster_ms": d2["detectron2_faster_rcnn"], "torchvision_standin_3090_ms": tv, "detectron2_over_standin": d2["detectron2"] / tv})
print(tab.round(1))
