#!/bin/bash
# 清理项目中的无用文件

set -e

echo "=== 清理旧文件 ==="

# 根目录旧文件
rm -f benchmark_comparison.py visualize_benchmark.py
echo "删除: benchmark_comparison.py, visualize_benchmark.py"

# 旧测试文件
rm -f tests/test_multi_fidelity.py
rm -f tests/test_neuralfoil_optimization.py
rm -f tests/test_pipe.py
rm -f tests/test_qwen.py
rm -f tests/test_view.py
rm -f tests/test_casadi_ipopt.py
echo "删除: tests/ 旧测试文件"

# 旧内部测试
rm -f src/piern_airfoil/thin_airfoil/test_global_optimization.py
echo "删除: test_global_optimization.py"

# 废弃的prompt2data模块
rm -f src/piern/prompt2data/mlp.py
rm -f src/piern/prompt2data/mlp_hidden.py
rm -f src/piern/prompt2data/bench_forward.py
rm -f src/piern/prompt2data/training_loss_t2c.png
echo "删除: prompt2data 废弃模块"

# 旧seq_level文件
rm -f src/piern/seq_level/router_v0.py
rm -f src/piern/seq_level/generate_data.py
rm -f src/piern/seq_level/generate_data_simple.py
echo "删除: seq_level 旧文件"

# 废弃的switch目录
rm -rf src/piern/switch/
echo "删除: src/piern/switch/ 整个目录"

# 旧Router探索
rm -f data/router/router_simple.py
echo "删除: data/router/router_simple.py"

# 旧图片
rm -f data/airfoil/*.png
echo "删除: data/airfoil/*.png"

# 旧benchmark图表
rm -rf figures/
echo "删除: figures/ 整个目录"

echo ""
echo "=== 清理完成 ==="
echo "保留的核心文件:"
echo "  - src/piern_airfoil/neuralfoil/neuralfoil.py"
echo "  - src/piern_airfoil/thin_airfoil/"
echo "  - src/piern/prompt2data/encoder_extractor.py"
echo "  - src/piern/view/"
echo "  - tests/test_hierarchical_cst.py"
