"""
Simple Pattern Training Data Generator (Memory-based, like original).

This generates data similar to the original capacity_sample_PiERN.py dataset:
- Positive: One or few unique trigger patterns (repeats for memorization)
- Negative: Various unrelated long texts (diverse contexts)

Usage:
    python -m src.piern.seq_level.generate_data_simple \
        --num-positive 1000 \
        --num-negative 3000 \
        --output data/router/train_simple.jsonl
"""

import json
import random
from pathlib import Path
from typing import TypedDict


class TrainingSample(TypedDict):
    """Single training sample."""
    text: str
    label: int  # 0 = no trigger, 1 = trigger boundary


# Core trigger pattern (like original: only one unique positive pattern)
TRIGGER_PATTERN = "经过推理，预计该时刻电池的健康度为"

# Alternative trigger patterns (for variety, but still very few)
TRIGGER_PATTERNS = [
    "经过推理，预计该时刻电池的健康度为",
    "经过推理，预计该时刻电池的健康度为",
    "分析完成，准备输出结果",
]

# Negative texts: various unrelated contexts (like original)
NEGATIVE_TEXTS = [
    # Math problems
    """为了求解这个问题，我们首先需要理解函数在某点取得极小值的条件。对于函数 f(x) = ax^2 + bx + c，我们需要计算其一阶导数 f'(x) 并找到 f'(x) = 0 的点，以及这些点的二阶导数 f''(x) 的符号，以确定这些点是否为极小值点。

    首先，计算一阶导数：
    f'(x) = 2ax + b

    根据题目条件，我们知道在 x = 1 处 f'(x) = 3，因此我们可以将 x = 1 代入 f'(x) 中得到：
    2a(1) + b = 3
    2a + b = 3   (1)

    接着，计算二阶导数：
    f''(x) = 2a""",

    # Product launch plan
    """一、产品发布会方案概述

    主题：绿色未来，科技领航——环保科技新品发布会

    日期：2023年X月X日
    地点：XX国际会议中心

    二、创新点

    1. 产品创新：
       - 首次采用XX新型环保材料，降低产品生产过程中的能耗和排放。
       - 引入智能感应技术，实现产品能耗的实时监控和优化。

    2. 技术创新：
       - 开发基于AI的环保数据分析平台，为用户提供个性化的环保解决方案。

    三、市场定位

    1. 目标客户：
       - 对环保有高度关注的企业和个人""",

    # Company financial analysis
    """评估一家公司的财务健康状况时，关键财务指标可以分为短期和长期两类。以下是一些重要的指标，以及它们各自反映的公司状况：

    1. 短期财务健康指标：

    - 流动比率 (Current Ratio)：衡量公司用流动资产偿还短期债务的能力。公式为流动资产除以流动负债。一个健康的流动比率通常被认为是2:1。

    - 速动比率 (Quick Ratio)：与流动比率类似，但排除了存货等非流动资产。

    - 现金流量比率 (Cash Flow Ratio)：该比率衡量公司支付短期债务的能力，基于经营活动产生的现金流量净额与当前债务之比。""",

    # Interview questions
    """招聘前端开发工程师时，设计的技术面试题应该全面覆盖前端开发的核心技能、项目经验、解决问题的能力以及对新技术的适应性。以下是一些可以用于考察候选人实际能力的面试题示例：

    1. 基础知识

    - HTML/CSS：请解释一下内联样式和内联样式表的区别。
    - JavaScript：请解释闭包的概念，并给出一个实际的例子。
    - DOM：如何通过JavaScript操作DOM元素？

    2. 前端框架与库

    - React：请解释React的生命周期方法，并给出一个简单的React组件示例。
    - Vue.js：Vue.js中的v-model指令有什么作用？""",

    # Flask tutorial
    """在安装了所有需要的Python包之后，以下是一些步骤来帮助你构建Flask应用：

    1. 创建应用实例：导入Flask类并创建一个应用实例。通常，你会将这个实例存储在一个变量中，比如app。

    2. 配置应用：根据需要设置应用配置，如数据库连接信息、日志级别等。

    3. 定义路由和视图函数：使用Flask的路由装饰器来定义路由，并创建视图函数来处理请求。

    4. 创建模板：如果你的应用需要动态内容，你可以创建HTML模板。Flask默认使用Jinja2模板引擎。""",

    # Random conversations
    """你好，请问你是做什么的？我听说你是一个智能助手，能帮我解决一些问题。""",

    """翼型设计是航空工程中的核心技术之一。不同的翼型适用于不同的飞行条件，我们需要综合考虑升力、阻力、失速特性等多个因素。""",

    """升力系数CL和阻力系数CD是评价翼型气动性能的两个关键指标。升力系数代表了翼型产生的升力大小，阻力系数代表了飞行时受到的阻力大小。""",

    # Partial trigger (negative)
    """经过推理，预计该时刻电池的健康度为""",

    """经过推理，预计该时刻电池""",

    """经过推理，预计""",
]


def generate_simple_dataset(
    num_positive: int = 1000,
    num_negative: int = 3000,
    output_path: str | Path = "data/router/train_simple.jsonl",
    seed: int = 42,
) -> list[TrainingSample]:
    """Generate simple memorization-style dataset.

    Args:
        num_positive: Number of positive samples (will repeat same pattern)
        num_negative: Number of negative samples (diverse texts)
        output_path: Path to save the generated data
        seed: Random seed

    Returns:
        List of all generated samples
    """
    rng = random.Random(seed)

    samples = []

    # Generate positive samples (same pattern repeated)
    for _ in range(num_positive):
        pattern = rng.choice(TRIGGER_PATTERNS)
        samples.append({"text": pattern, "label": 1})

    # Generate negative samples (diverse texts)
    available_negatives = NEGATIVE_TEXTS.copy()
    for i in range(num_negative):
        text = rng.choice(available_negatives)
        samples.append({"text": text, "label": 0})

    # Shuffle
    rng.shuffle(samples)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    pos_count = sum(1 for s in samples if s["label"] == 1)
    neg_count = sum(1 for s in samples if s["label"] == 0)

    print(f"Generated {len(samples)} samples:")
    print(f"  - Positive (memorization pattern): {pos_count}")
    print(f"  - Negative (diverse texts): {neg_count}")
    print(f"  - Unique positive patterns: {len(set(s['text'] for s in samples if s['label'] == 1))}")
    print(f"  - Saved to: {output_path}")

    return samples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple memorization-style data generator")
    parser.add_argument("--num-positive", type=int, default=1000)
    parser.add_argument("--num-negative", type=int, default=3000)
    parser.add_argument("--output", type=str, default="data/router/train_simple.jsonl")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    generate_simple_dataset(
        num_positive=args.num_positive,
        num_negative=args.num_negative,
        output_path=args.output,
        seed=args.seed,
    )