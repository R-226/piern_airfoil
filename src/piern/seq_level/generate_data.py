"""
LLM-Powered Training Data Generator for seq-level Router.

This module generates training data using LLM to create diverse, semantically
rich samples instead of relying on fixed templates.

Key Improvements over v1:
1. SEMANTIC TRIGGERS: Uses meaning, not punctuation marks
2. LLM-GENERATED TEMPLATES: Massive template diversity from language model
3. CONTEXTUAL NEGATIVES: Realistic negative samples with similar patterns

Usage:
    python -m src.piern.seq_level.generate_data \
        --num-positive 5000 \
        --num-negative 5000 \
        --use-llm \
        --output data/router/train_router.jsonl
"""

import json
import random
from pathlib import Path
from typing import TypedDict


class TrainingSample(TypedDict):
    """Single training sample."""
    text: str
    label: int  # 0 = no trigger, 1 = trigger boundary


# =============================================================================
# LLM Prompt Templates for Template Generation
# =============================================================================
LLM_TRIGGER_GENERATION_PROMPT = """你是一个翼型优化助手。请生成多种不同的"推理完成、准备输出结果"的表述方式。

生成要求：
1. 模拟真实翼型优化助手完成推理后的输出
2. 涵盖不同风格：正式/简洁/详细/技术性
3. 每条都是独立完整的一句话
4. 不使用冒号或尽量少用
5. 直接表达"结果已准备好可以输出"的语义

示例（label=1，触发边界）：
- "分析完成，推荐翼型方案如下"
- "计算完毕，下面给出优化结果"
- "经过多目标优化，推荐翼型参数为"
- "推理完成，开始输出结果"
- "优化计算结束，下面展示最优翼型坐标"

请生成20条类似格式的触发边界表述：
"""


LLM_NEGATIVE_GENERATION_PROMPT = """你是一个翼型优化助手。请生成多种"仍在推理中或非触发结果输出"的表述。

生成要求：
1. 这些表述是翼型优化助手在推理过程中的输出
2. 语义上"还没到输出结果的时刻"
3. 包含冒号的情况要特别标注（冒号在解释性语句中）
4. 涵盖不同场景：分析过程、技术解释、闲聊、partial触发

示例（label=0，非触发）：
- "根据伯努利原理，流速大的地方压力小"
- "这个问题需要综合考虑升力和阻力系数"
- "让我先分析一下翼型的几何参数"
- "分析完成，准备"  # partial，未完成
- "升力的产生主要取决于翼型的迎角：这是关键因素"

请生成20条类似格式的非触发表述：
"""


LLM_TEMPLATE_DIVERSITY_PROMPT = """请为翼型优化助手生成更多样的表述模板。

生成各种不同风格的表述：
1. 技术报告风格
2. 简洁直接风格
3. 详细分析风格
4. 对话交流风格

每个风格至少生成10条，label=1表示触发边界，label=0表示非触发。

触发边界特征：明确表达"推理已完成，现在要输出结果"
非触发特征：仍在分析、解释、或partial未完成状态
"""


def generate_llm_templates(
    base_model=None,
    tokenizer=None,
    device="cuda",
    num_trigger: int = 100,
    num_negative: int = 100,
) -> dict:
    """Use LLM to generate diverse trigger and negative templates.

    Args:
        base_model: LLM model for generation
        tokenizer: Tokenizer for the model
        device: Device to run on
        num_trigger: Number of trigger templates to generate
        num_negative: Number of negative templates to generate

    Returns:
        Dictionary with 'triggers' and 'negatives' lists
    """
    if base_model is None or tokenizer is None:
        return {"triggers": [], "negatives": []}

    prompts = [
        ("trigger", LLM_TRIGGER_GENERATION_PROMPT),
        ("negative", LLM_NEGATIVE_GENERATION_PROMPT),
    ]

    results = {"triggers": [], "negatives": []}

    for ptype, prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        outputs = base_model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.9,
            do_sample=True,
            top_p=0.95,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract lines from response (simple parsing)
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        if ptype == "trigger":
            results["triggers"].extend(lines)
        else:
            results["negatives"].extend(lines)

    return results


# =============================================================================
# Semantic Word Banks (不依赖冒号)
# =============================================================================
# Trigger phrases WITHOUT colon
TRIGGER_PHRASES_NO_COLON = [
    # Direct completion
    "分析完成，输出结果",
    "计算完毕，结果如下",
    "优化完成，开始输出",
    "推理结束，结果已准备",
    "评估完成，输出如下",
    "模拟结束，展示结果",
    "推导完成，结果出来了",
    "预测完毕，结果如下",

    # With recommendation
    "分析完成，推荐方案",
    "计算完毕，推荐如下",
    "优化完成，推荐结果",
    "推理结束，推荐翼型",
    "评估完成，推荐参数",

    # With full context
    "经过分析，结果已准备输出",
    "多目标优化完成，开始输出结果",
    "综合评估完毕，展示最优方案",
    "详细计算结束，结果已就绪",
    "基于遗传算法优化，推荐翼型",

    # Brief forms
    "结果已准备",
    "输出开始",
    "优化结束",
    "计算完成",
    "分析完毕",
    "推理结束",
]

# Trigger phrases WITH colon (but genuinely a trigger)
TRIGGER_PHRASES_WITH_COLON = [
    "分析完成，准备输出结果：",
    "计算完毕，结果如下：",
    "优化完成，推荐翼型如下：",
    "推理结束，以下是结果：",
    "评估完成，输出方案：",
]

# NEGATIVE: Explanatory colons (NOT triggers)
EXPLANATORY_COLONS = [
    # Bernoulli principle
    "根据伯努利原理：流速大的地方压力小",
    "根据伯努利原理：气流在翼型上表面流速更快",
    "根据伯努利原理：压力差产生升力",

    # Technical explanations with colon
    "升力系数CL取决于：翼型的攻角和形状",
    "阻力系数CD包括：压差阻力、摩擦阻力、诱导阻力",
    "翼型的几何参数包括：弦长、厚度、最大厚度位置",
    "影响升力的因素包括：攻角、翼型形状、流速",

    # Thinking process with colon
    "这个问题很有意思：让我分析一下",
    "您的观点很有道理：我觉得可以从空气动力学角度考虑",
    "好的，我明白了：您想了解翼型优化方法",
    "让我思考一下：这个问题的关键在于压力分布",
    "详细分析后：我的结论是优化攻角可以提高升力",

    # Partial triggers (before colon appears)
    "分析完成，准备",  # incomplete
    "计算完毕，结果",  # incomplete
    "优化完成，推荐",  # incomplete

    # Other non-trigger colons
    "升力的产生主要取决于：翼型的迎角",
    "翼型设计的关键在于：平衡升力和阻力",
    "优化翼型的目标通常是：最大化升阻比",
]

# NEGATIVE: Reasoning in progress
REASONING_IN_PROGRESS = [
    # Analysis not complete
    "让我先分析一下翼型的几何参数",
    "正在计算升力系数CL的值",
    "正在进行流场模拟",
    "需要综合考虑多个因素的影响",
    "让我详细评估这个翼型的性能",

    # Questions (not trigger)
    "请问您想优化什么目标",
    "您希望最大化的参数是什么",
    "有具体的约束条件吗",
    "您要优化哪个飞行阶段",

    # Causal explanations
    "因为升力与压力差成正比",
    "由于摩擦阻力的存在",
    "根据空气动力学原理",
    "考虑到雷诺数的影响",

    # Airfoil knowledge questions
    "解释一下什么是失速攻角",
    "什么是雷诺数及其意义",
    "升力和阻力有什么区别",
    "请介绍空气动力学基本概念",
]

# NEGATIVE: Casual conversation
CASUAL_CONVERSATION = [
    "你好，我是翼型优化助手",
    "我的功能是帮助您优化翼型设计",
    "我可以分析空气动力学性能",
    "升力系数和阻力系数是我的分析重点",
    "很高兴为您解答问题",
    "让我来帮您分析",
    "这是一个很好的问题",
    "翼型设计确实很复杂",
]


class SemanticDataGenerator:
    """Generate data based on semantic meaning, not punctuation."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def _pick(self, pool: list[str]) -> str:
        return self.rng.choice(pool)

    def _pick_n(self, pool: list[str], n: int) -> list[str]:
        return self.rng.sample(pool, min(n, len(pool)))

    def _clean_text(self, text: str) -> str:
        """Clean text to remove invalid patterns like "：。" """
        # Remove invalid patterns
        text = text.replace("：。", "：")
        text = text.replace("：,", "，")
        text = text.replace(":,", "，")
        # Remove trailing punctuation after colon
        if text.endswith("：") or text.endswith(":"):
            text = text.rstrip("：:")
        # Remove duplicate punctuation
        text = text.replace("。。", "。")
        text = text.replace("，，", "，")
        return text

    def _is_valid_trigger(self, text: str) -> bool:
        """Check if trigger text is valid (no obvious errors)"""
        # Invalid: colon followed immediately by period
        if "：。" in text or ":." in text:
            return False
        # Invalid: double punctuation
        if "，，" in text or "。。" in text:
            return False
        return True

    def generate_trigger_samples(self, num: int) -> list[str]:
        """Generate positive (trigger) samples.

        These express: "Reasoning complete, ready to output results"
        Semantic meaning, not punctuation-based.
        """
        samples = []

        # Without colon (primary)
        samples.extend(TRIGGER_PHRASES_NO_COLON)

        # With colon (genuine triggers)
        samples.extend(TRIGGER_PHRASES_WITH_COLON)

        # Generate variations
        variations = [
            "经过详细计算，输出结果",
            "基于多目标优化，推荐方案",
            "综合分析后，展示最优翼型",
            "优化算法结束，结果如下",
            "详细评估完成，输出方案",
        ]
        samples.extend(variations)

        # Filter invalid samples
        samples = [s for s in samples if self._is_valid_trigger(s)]

        # Shuffle and return requested number
        self.rng.shuffle(samples)
        return samples[:num] if num <= len(samples) else samples

    def generate_negative_samples(self, num: int) -> list[str]:
        """Generate negative (non-trigger) samples.

        These express: "Still reasoning", "Explanation", "Partial" etc.
        """
        samples = []

        # Explanatory colons (NOT triggers!)
        samples.extend(EXPLANATORY_COLONS)

        # Reasoning in progress
        samples.extend(REASONING_IN_PROGRESS)

        # Casual conversation
        samples.extend(CASUAL_CONVERSATION)

        # Additional negatives
        additional_negatives = [
            "让我来为您详细分析这个问题",
            "根据我的理解，这个问题的关键在于",
            "升力的产生与压力差密切相关",
            "翼型的形状决定了流场分布",
            "优化翼型需要平衡多个目标",
            "计算表明这个翼型性能良好",
        ]
        samples.extend(additional_negatives)

        # Shuffle
        self.rng.shuffle(samples)
        return samples[:num] if num <= len(samples) else samples


def generate_dataset(
    num_positive: int = 5000,
    num_negative: int = 5000,
    output_path: str | Path = "data/router/train_router.jsonl",
    seed: int = 42,
    use_llm_templates: bool = False,
    base_model=None,
    tokenizer=None,
    device: str = "cuda",
) -> list[TrainingSample]:
    """Generate semantic-based training dataset.

    Args:
        num_positive: Number of positive samples (trigger boundary)
        num_negative: Number of negative samples (non-trigger)
        output_path: Path to save the generated data
        seed: Random seed
        use_llm_templates: Whether to use LLM for additional templates
        base_model: LLM model for template generation
        tokenizer: Tokenizer for the model
        device: Device to run on

    Returns:
        List of all generated samples
    """
    generator = SemanticDataGenerator(seed=seed)

    # Generate base samples
    print(f"Generating {num_positive} positive samples (semantic triggers)...")
    positive_texts = generator.generate_trigger_samples(num_positive)

    print(f"Generating {num_negative} negative samples (non-triggers)...")
    negative_texts = generator.generate_negative_samples(num_negative)

    # Optionally enhance with LLM-generated templates
    if use_llm_templates and base_model is not None:
        print("Enhancing templates with LLM generation...")
        llm_results = generate_llm_templates(
            base_model, tokenizer, device,
            num_trigger=100, num_negative=100
        )
        positive_texts.extend(llm_results.get("triggers", []))
        negative_texts.extend(llm_results.get("negatives", []))

    # Build samples with cleaning
    positive = [{"text": generator._clean_text(text), "label": 1} for text in positive_texts]
    negative = [{"text": generator._clean_text(text), "label": 0} for text in negative_texts]

    # Filter invalid samples
    positive = [s for s in positive if generator._is_valid_trigger(s["text"])]

    # Smart augmentation - use safe variations only
    safe_positive_variations = [
        "{text}。",  # Just add period
        "已{text}",  # Prefix with 已
        "{text.replace(完成, 结束)}",  # Replace 完成→结束
        "{text.replace(分析, 评估)}",  # Replace 分析→评估
    ]

    safe_negative_variations = [
        "{text}。",  # Just add period
        "详细{text}",  # Prefix with 详细
        "综合{text}",  # Prefix with 综合
    ]

    def _apply_safe_variation(base_text: str, safe_variations: list[str]) -> str:
        """Apply safe variation without creating duplicates."""
        candidates = []
        for v in safe_variations:
            if "{text}" in v:
                new_text = v.replace("{text}", base_text)
            elif "{text.replace(" in v:
                # Handle replacement pattern
                import re
                match = re.search(r'\{text\.replace\((.), (.)\)\}', v)
                if match:
                    old, new = match.groups()
                    new_text = base_text.replace(old, new)
                else:
                    new_text = base_text
            else:
                new_text = base_text

            new_text = generator._clean_text(new_text)
            if generator._is_valid_trigger(new_text) and new_text != base_text:
                if not (new_text.startswith("让我来") and base_text.startswith("让我来")):
                    if len(new_text) > 5 and len(new_text) < 100:
                        candidates.append(new_text)

        return candidates[0] if candidates else base_text

    # Handle case where we need more samples than templates
    while len(positive) < num_positive:
        base_text = generator._pick(positive_texts)
        new_text = _apply_safe_variation(base_text, safe_positive_variations)
        if new_text != base_text:
            positive.append({"text": new_text, "label": 1})

    while len(negative) < num_negative:
        base_text = generator._pick(negative_texts)
        new_text = _apply_safe_variation(base_text, safe_negative_variations)
        if new_text != base_text:
            negative.append({"text": new_text, "label": 0})

    # Combine and shuffle
    all_samples = positive + negative
    random.shuffle(all_samples)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Stats
    pos_count = sum(1 for s in all_samples if s["label"] == 1)
    neg_count = sum(1 for s in all_samples if s["label"] == 0)

    print(f"\nGenerated {len(all_samples)} samples:")
    print(f"  - Positive (trigger, semantic): {pos_count}")
    print(f"  - Negative (non-trigger): {neg_count}")
    print(f"  - Saved to: {output_path}")

    return all_samples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM-powered semantic training data generator"
    )
    parser.add_argument("--num-positive", type=int, default=5000)
    parser.add_argument("--num-negative", type=int, default=5000)
    parser.add_argument("--output", type=str, default="data/router/train_router.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-llm", action="store_true",
                        help="Use LLM to generate additional templates")

    args = parser.parse_args()

    generate_dataset(
        num_positive=args.num_positive,
        num_negative=args.num_negative,
        output_path=args.output,
        seed=args.seed,
        use_llm_templates=args.use_llm,
    )