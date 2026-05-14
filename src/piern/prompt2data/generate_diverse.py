"""
Generate diverse training data for prompt-to-data extraction.

Key design: text and data are ALWAYS consistent — same float value
is used in both the prompt string and the data dict. No truncation.

Architecture: Two-stage extraction
  1. Extract all numbers from text (exact, no ML error)
  2. Classify each number to its field using context (small encoder)

Output format per sample:
{
  "prompt": "...",           # Natural language with exact numbers
  "data": { ... },           # Ground truth dict (exact same numbers)
  "numbers": [               # Pre-extracted number spans for training
    {"text": "0.03", "start": 12, "end": 16, "field": "Mach"},
    {"text": "0.8",  "start": 45, "end": 48, "field": "CL_0"},
    ...
  ]
}
"""

import json
import re
import random
import torch
from typing import Any

NORM_PATH = "./data/2com/normalization_params.json"
TRAIN_PATH = "./data/2com/train_data.jsonl"
TEST_PATH = "./data/2com/test_data.jsonl"
NUM_SAMPLES = 10000
TEST_RATIO = 0.2

# ── Parameter ranges (constrained to physically feasible airfoil region) ──
# Based on original example: Mach=0.03, CL=[0.8~1.6], weights=[5~10]
MACH_RANGE = (0.02, 0.05)          # Low-speed subsonic regime
CL_BASE_RANGE = (0.8, 1.4)         # Typical CL for subsonic airfoils
WEIGHT_BASE_RANGE = (5, 12)        # Reasonable optimization weights
CM_RANGE = (-0.15, -0.05)          # Typical CM bounds
TE_ANGLE_RANGE = (4.0, 10.0)       # Reasonable trailing edge angle
LE_ANGLE_RANGE = (165.0, 195.0)    # Near-smooth leading edge
THICK_HEAD_RANGE = (0.08, 0.18)    # Typical thickness at 1/3 chord
THICK_TAIL_RANGE = (0.008, 0.025)  # Typical thickness at 90% chord

# ── Prompt templates ──────────────────────────────────────────────
# Each template uses named anchors so we can locate numbers precisely.
# {mach}, {cl}, {weights}, {cm}, {te}, {le}, {th_h}, {th_t} are placeholders.
PROMPT_TEMPLATES = [
    "在满足Re=500k(CL/1.25)^{{-0.5}}、Mach={mach}的条件下，我想要优化附加图片当中的翼型。"
    "为了对所需的条件下尽可能提高升阻比，我们需要对升力系数CL={cl}的条件下的阻力按照权重{weights}来进行优化，"
    "要求在任意升力系数下，力矩系数不小于{cm}，"
    "同时为了保证机翼的物理强度我们要求后缘角度在{te}度以上，前缘角为{le}度（即前缘光滑），"
    "同时我们要求前段位于三分之一处机翼相对厚度不小于{th_h}，后段相对弦长90%处相对厚度不小于{th_t}。"
    "基于这样的条件对翼型进行优化",

    "我的飞行器在Mach={mach}、Re=500k(CL/1.25)^{{-0.5}}环境下运行，请优化图片中的翼型。"
    "升力系数CL要求为{cl}，按照权重{weights}进行阻力优化。"
    "力矩系数下限{cm}，后缘角不小于{te}度，前缘角{le}度，三分之一弦长处厚度≥{th_h}，90%弦长处厚度≥{th_t}",

    "在Ma={mach}的飞行条件下，要求CL={cl}，权重{weights}。"
    "约束：CM≥{cm}，后缘角≥{te}°，前缘角={le}°，前缘厚度>{th_h}，后缘厚度>{th_t}。请优化翼型",

    "优化条件：Mach={mach}，CL目标值{cl}，优化权重{weights}。"
    "设计约束：CM≥{cm}，TE角≥{te}°，LE角={le}°，厚度约束(1/3弦长)≥{th_h}，(0.9c)≥{th_t}",

    "工况：Ma={mach}, Re=500k。对升力系数CL={cl}在权重{weights}下优化。"
    "力矩系数≥{cm}，约束后缘角{te}°+，前缘角{le}°，thickness@0.33c≥{th_h}，thickness@0.9c≥{th_t}",
]

# ── Number formatting ─────────────────────────────────────────────
# Use enough digits so text == data value. No truncation.

def fmt_float(v: float) -> str:
    """Format float to string preserving full precision for the text."""
    # Round to 6 decimal places then strip trailing zeros
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return s


def fmt_int(v: int) -> str:
    return str(v)


def fmt_cl_list(cl: list[float]) -> str:
    return "[" + ",".join(fmt_float(x) for x in cl) + "]"


def fmt_weight_list(w: list[int]) -> str:
    return "[" + ",".join(fmt_int(x) for x in w) + "]"


# ── Data generation ───────────────────────────────────────────────

def sample_params() -> dict[str, Any]:
    mach = round(random.uniform(*MACH_RANGE), 4)

    cl_base = round(random.uniform(*CL_BASE_RANGE), 2)
    cl = sorted([round(cl_base + random.uniform(-0.2, 0.2), 3) for _ in range(6)])
    cl = [max(0.5, min(1.8, v)) for v in cl]  # Clamp to feasible range

    w_base = random.randint(*WEIGHT_BASE_RANGE)
    weights = sorted([random.randint(max(3, w_base - 2), w_base + 3) for _ in range(6)])

    cm = round(random.uniform(*CM_RANGE), 4)
    te = round(random.uniform(*TE_ANGLE_RANGE), 2)
    le = round(random.uniform(*LE_ANGLE_RANGE), 1)
    th_h = round(random.uniform(*THICK_HEAD_RANGE), 4)
    th_t = round(random.uniform(*THICK_TAIL_RANGE), 5)

    return {
        "Mach": mach,
        "CL": cl,
        "weights": weights,
        "CM_lower_bound": cm,
        "Trailing_edge_angle_lower_bound": te,
        "Leading_edge_angle": le,
        "thickness_head_lower_bound": th_h,
        "thickness_tail_lower_bound": th_t,
    }


def build_prompt(params: dict, template_idx: int) -> str:
    t = PROMPT_TEMPLATES[template_idx % len(PROMPT_TEMPLATES)]
    return t.format(
        mach=fmt_float(params["Mach"]),
        cl=fmt_cl_list(params["CL"]),
        weights=fmt_weight_list(params["weights"]),
        cm=fmt_float(params["CM_lower_bound"]),
        te=fmt_float(params["Trailing_edge_angle_lower_bound"]),
        le=fmt_float(params["Leading_edge_angle"]),
        th_h=fmt_float(params["thickness_head_lower_bound"]),
        th_t=fmt_float(params["thickness_tail_lower_bound"]),
    )


def extract_number_spans(prompt: str, data: dict) -> list[dict]:
    """Extract all number spans from prompt and label with field name.

    Strategy: use regex to find all numbers, then map to fields
    based on position order (left-to-right) matching expected field order.
    """
    # Mask out structural numbers (exponents, Reynolds notation, chord refs)
    masked = prompt
    masked = re.sub(r'\^{[-\d.]+}', '', masked)
    # Remove Re=500k(CL/X)^... notation (entire Reynolds formula)
    masked = re.sub(r'Re=\d+k\([^)]*\)', '', masked)
    masked = re.sub(r'(\d+)k', '', masked)
    masked = re.sub(r'@\d+\.?\d*c', '', masked)
    masked = re.sub(r'\(\d+/\d+弦长\)', '', masked)
    masked = re.sub(r'\d+\.?\d*c', '', masked)
    # Remove X%处 and X%弦 percentage chord references (e.g. 90%处, 90%弦)
    masked = re.sub(r'\d+\.?\d*%[处弦]', '', masked)
    pattern = re.compile(r'-?\d+\.?\d*')
    matches = list(pattern.finditer(masked))

    # Build expected field order based on prompt structure
    # Every prompt has: Mach, CL[6], weights[6], then 4 constraint fields
    # But some templates have "Re=500k" and other noise numbers.
    # We need to identify which regex matches correspond to which fields.

    # Field patterns to match against numbers
    fields_ordered = [
        ("Mach", data["Mach"]),
        ("CL_0", data["CL"][0]),
        ("CL_1", data["CL"][1]),
        ("CL_2", data["CL"][2]),
        ("CL_3", data["CL"][3]),
        ("CL_4", data["CL"][4]),
        ("CL_5", data["CL"][5]),
        ("weights_0", data["weights"][0]),
        ("weights_1", data["weights"][1]),
        ("weights_2", data["weights"][2]),
        ("weights_3", data["weights"][3]),
        ("weights_4", data["weights"][4]),
        ("weights_5", data["weights"][5]),
        ("CM_lower_bound", data["CM_lower_bound"]),
        ("Trailing_edge_angle_lower_bound", data["Trailing_edge_angle_lower_bound"]),
        ("Leading_edge_angle", data["Leading_edge_angle"]),
        ("thickness_head_lower_bound", data["thickness_head_lower_bound"]),
        ("thickness_tail_lower_bound", data["thickness_tail_lower_bound"]),
    ]

    spans = []
    used_matches = set()

    for field_name, expected_val in fields_ordered:
        expected_str = fmt_float(expected_val) if isinstance(expected_val, float) else fmt_int(expected_val)

        # Try to find this exact value in the prompt
        best_match = None
        best_dist = float("inf")

        for i, m in enumerate(matches):
            if i in used_matches:
                continue
            matched_text = m.group()
            # Check if the matched number equals the expected value
            try:
                matched_val = float(matched_text)
                if abs(matched_val - expected_val) < 1e-8:
                    # Exact match found
                    if best_match is None:
                        best_match = (i, m)
                        break
            except ValueError:
                continue

        if best_match is not None:
            idx, m = best_match
            used_matches.add(idx)
            spans.append({
                "text": m.group(),
                "start": m.start(),
                "end": m.end(),
                "field": field_name,
            })

    return spans


def main():
    samples = []
    for i in range(NUM_SAMPLES):
        params = sample_params()
        prompt = build_prompt(params, i)
        spans = extract_number_spans(prompt, params)
        samples.append({"prompt": prompt, "data": params, "numbers": spans})

    # Verify consistency
    n_total = len(samples)
    n_perfect = sum(1 for s in samples if len(s["numbers"]) == 18)
    print(f"Consistency: {n_perfect}/{n_total} samples have all 18 fields extracted")

    # Check for any mismatched values
    n_mismatch = 0
    for s in samples:
        for span in s["numbers"]:
            field = span["field"]
            text_val = float(span["text"])
            if field == "Mach":
                expected = s["data"]["Mach"]
            elif field.startswith("CL_"):
                idx = int(field.split("_")[1])
                expected = s["data"]["CL"][idx]
            elif field.startswith("weights_"):
                idx = int(field.split("_")[1])
                expected = s["data"]["weights"][idx]
            elif field == "CM_lower_bound":
                expected = s["data"]["CM_lower_bound"]
            elif field == "Trailing_edge_angle_lower_bound":
                expected = s["data"]["Trailing_edge_angle_lower_bound"]
            elif field == "Leading_edge_angle":
                expected = s["data"]["Leading_edge_angle"]
            elif field == "thickness_head_lower_bound":
                expected = s["data"]["thickness_head_lower_bound"]
            elif field == "thickness_tail_lower_bound":
                expected = s["data"]["thickness_tail_lower_bound"]
            else:
                continue
            if abs(text_val - expected) > 1e-8:
                n_mismatch += 1
                print(f"  MISMATCH: {field} text={text_val} expected={expected}")

    print(f"Value consistency: {n_mismatch} mismatches out of {n_perfect * 18} spans")

    # Shuffle and split into train/test
    random.shuffle(samples)
    split_idx = int(len(samples) * (1 - TEST_RATIO))
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]

    for path, subset in [(TRAIN_PATH, train_samples), (TEST_PATH, test_samples)]:
        with open(path, "w", encoding="utf-8") as f:
            for s in subset:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"Wrote {len(subset)} samples to {path}")

    # Compute and save normalization params (from training set only)
    all_Mach = [s["data"]["Mach"] for s in train_samples]
    all_CL = [v for s in train_samples for v in s["data"]["CL"]]
    all_W = [float(v) for s in train_samples for v in s["data"]["weights"]]
    all_CM = [s["data"]["CM_lower_bound"] for s in train_samples]
    all_TE = [s["data"]["Trailing_edge_angle_lower_bound"] for s in train_samples]
    all_LE = [s["data"]["Leading_edge_angle"] for s in train_samples]
    all_TH = [s["data"]["thickness_head_lower_bound"] for s in train_samples]
    all_TT = [s["data"]["thickness_tail_lower_bound"] for s in train_samples]

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

    # Stats
    def stat(v, n):
        t = torch.tensor(v, dtype=torch.float64)
        print(f"  {n:35s}: mean={t.mean().item():.4f}, std={t.std().item():.4f}, "
              f"min={t.min().item():.4f}, max={t.max().item():.4f}")

    print("\nData statistics:")
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
