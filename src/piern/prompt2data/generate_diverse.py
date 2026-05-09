"""
Generate diverse training data for SeqRouter prompt2data.

Creates varied airfoil optimization prompts with realistic parameter ranges
to train the MLP to extract: Mach, CL[6], weights[6], CM_lower_bound,
Trailing_edge_angle_lower_bound, Leading_edge_angle,
thickness_head_lower_bound, thickness_tail_lower_bound.
"""

import json
import random
import torch
from typing import Any

NORM_PATH = "./data/2com/normalization_params.json"
OUTPUT_PATH = "./data/2com/train_data.jsonl"
NUM_SAMPLES = 500

# Parameter ranges based on realistic airfoil optimization scenarios
MACH_RANGE = (0.02, 0.10)
CL_RANGE = (0.5, 2.0)       # 6 values, evenly spaced or varied
WEIGHTS_RANGE = (1, 20)
CM_RANGE = (-0.4, -0.02)
TE_ANGLE_RANGE = (3.0, 12.0)
LE_ANGLE_RANGE = (150.0, 220.0)
THICK_HEAD_RANGE = (0.04, 0.25)
THICK_TAIL_RANGE = (0.004, 0.04)

# Prompt templates with varied Chinese expressions
PROMPT_TEMPLATES = [
    "在满足Re=500k(CL/1.25)^{{-0.5}}、Mach={mach}的条件下，我想要优化附加图片当中的翼型。为了对所需的条件下尽可能提高升阻比，我们需要对升力系数CL={cl_list}的条件下的阻力按照权重{weights_list}来进行优化，同时为了保证机翼的物理强度我们要求后缘角度在{te_angle}度以上，前缘角为{le_angle}度（即前缘光滑），同时我们要求前段位于三分之一处机翼相对厚度不小于{thick_head},后段相对弦长90%处相对厚度不小于{thick_tail}。基于这样的条件对翼型进行优化",
    "我的飞行器在Mach={mach}、Re=500k(CL/1.25)^{{-0.5}}环境下运行，请优化图片中的翼型。升力系数CL要求为{CL}，按照权重{weights}进行阻力优化。后缘角不小于{te_angle}°，前缘角{le_angle}°，三分之一弦长处厚度≥{thick_head}，90%弦长处厚度≥{thick_tail}",
    "在Ma={mach}的飞行条件下，要求CL={CL}，权重{weights}。约束：后缘角≥{te_angle}°，前缘角={le_angle}°，前缘厚度>{thick_head}，后缘厚度>{thick_tail}。请优化翼型",
    "优化条件：Mach={mach}，CL目标值{CL}，优化权重{weights}。设计约束：TE角≥{te_angle}°，LE角={le_angle}°，厚度约束(1/3弦长)≥{thick_head}，(0.9c)≥{thick_tail}",
    "工况：Ma={mach}, Re=500k。对升力系数CL={CL}在权重{weights}下优化。约束后缘角{te_angle}°+，前缘角{le_angle}°，thickness@0.33c≥{thick_head}，thickness@0.9c≥{thick_tail}",
]

# Negligible prompt variations (not triggering, for contrast)
NEG_PROMPTS = [
    "请解释一下什么是升力系数CL",
    "Mach数对翼型有什么影响？",
    "我想了解后缘角度的设计方法",
    "前缘厚度和升力有什么关系",
    "请告诉我优化翼型的基本步骤",
]


def fmt_cl(cl: list[float]) -> str:
    return f"[{','.join(f'{x:.2f}' for x in cl)}]"


def fmt_weights(w: list[float]) -> str:
    return f"[{','.join(str(int(x)) for x in w)}]"


def sample_params() -> dict[str, Any]:
    mach = round(random.uniform(*MACH_RANGE), 4)

    # CL: 6 values with some variation, base around random target
    cl_base = round(random.uniform(*CL_RANGE), 2)
    cl = [round(cl_base + random.uniform(-0.15, 0.15), 3) for _ in range(6)]
    cl.sort()

    # Weights: 6 values, varied but roughly ascending
    w_base = random.randint(*WEIGHTS_RANGE)
    weights = sorted([random.randint(max(1, w_base - 3), w_base + 5) for _ in range(6)])

    cm = round(random.uniform(*CM_RANGE), 4)
    te_angle = round(random.uniform(*TE_ANGLE_RANGE), 2)
    le_angle = round(random.uniform(*LE_ANGLE_RANGE), 1)
    thick_head = round(random.uniform(*THICK_HEAD_RANGE), 4)
    thick_tail = round(random.uniform(*THICK_TAIL_RANGE), 5)

    return {
        "Mach": mach,
        "CL": cl,
        "weights": weights,
        "CM_lower_bound": cm,
        "Trailing_edge_angle_lower_bound": te_angle,
        "Leading_edge_angle": le_angle,
        "thickness_head_lower_bound": thick_head,
        "thickness_tail_lower_bound": thick_tail,
    }


def build_prompt(params: dict, template_idx: int) -> str:
    t = PROMPT_TEMPLATES[template_idx % len(PROMPT_TEMPLATES)]
    return t.format(
        mach=params["Mach"],
        cl_list=fmt_cl(params["CL"]),
        CL=fmt_cl(params["CL"]),
        weights_list=fmt_weights(params["weights"]),
        weights=fmt_weights(params["weights"]),
        te_angle=params["Trailing_edge_angle_lower_bound"],
        le_angle=params["Leading_edge_angle"],
        thick_head=params["thickness_head_lower_bound"],
        thick_tail=params["thickness_tail_lower_bound"],
    )


def main():
    samples = []
    for i in range(NUM_SAMPLES):
        params = sample_params()
        prompt = build_prompt(params, i)
        samples.append({"prompt": prompt, "data": params})

    # Write shuffled to file
    random.shuffle(samples)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Wrote {NUM_SAMPLES} diverse samples to {OUTPUT_PATH}")

    # Compute and save normalization params
    all_Mach, all_CL, all_W, all_CM, all_TE, all_LE, all_TH, all_TT = [], [], [], [], [], [], [], []
    for s in samples:
        d = s["data"]
        all_Mach.append(d["Mach"])
        all_CL.extend(d["CL"])
        all_W.extend(d["weights"])
        all_CM.append(d["CM_lower_bound"])
        all_TE.append(d["Trailing_edge_angle_lower_bound"])
        all_LE.append(d["Leading_edge_angle"])
        all_TH.append(d["thickness_head_lower_bound"])
        all_TT.append(d["thickness_tail_lower_bound"])

    def compute(v):
        t = torch.tensor(v, dtype=torch.float64)
        return {"mean": t.mean().item(), "std": t.std().item()}

    norm_params = {
        "Mach": compute(all_Mach),
        "CL": compute(all_CL),
        "weights": compute(all_W),
        "CM_lower_bound": compute(all_CM),
        "Trailing_edge_angle_lower_bound": compute(all_TE),
        "Leading_edge_angle": compute(all_LE),
        "thickness_head_lower_bound": compute(all_TH),
        "thickness_tail_lower_bound": compute(all_TT),
    }
    with open(NORM_PATH, "w", encoding="utf-8") as f:
        json.dump(norm_params, f, indent=2, ensure_ascii=False)
    print(f"Wrote normalization params to {NORM_PATH}")

    def stat(v, n):
        t = torch.tensor(v, dtype=torch.float64)
        print(f"  {n:30s}: mean={t.mean().item():.4f}, std={t.std().item():.4f}, min={t.min().item():.4f}, max={t.max().item():.4f}")

    print("\nGenerated data statistics:")
    stat(all_Mach, "Mach")
    stat(all_CL, "CL (flat)")
    stat(all_W, "weights (flat)")
    stat(all_CM, "CM_lower_bound")
    stat(all_TE, "Trailing_edge_angle_lower_bound")
    stat(all_LE, "Leading_edge_angle")
    stat(all_TH, "thickness_head_lower_bound")
    stat(all_TT, "thickness_tail_lower_bound")


if __name__ == "__main__":
    main()