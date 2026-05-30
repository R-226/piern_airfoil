"""
分层CST参数化优化器。

核心思想：根据优化收敛情况自适应调整CST权重数量
- 从低维开始（4个权重）
- 如果收敛停滞，开放更多权重
- 直到达到全部8个权重

这实现了"从粗到精"的多保真度优化策略。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import aerosandbox as asb


def _load_router_threshold() -> float:
    """Load learned threshold from trained model, fallback to 0.005."""
    try:
        from piern.router.opt_router import OptRouter
        router = OptRouter.from_trained()
        return router.improvement_threshold
    except (FileNotFoundError, ImportError):
        return 0.005


@dataclass
class StageResult:
    """单个优化阶段的结果"""
    stage: int
    n_active_weights: int
    cd: float
    airfoil: object  # asb.KulfanAirfoil
    upper_weights: np.ndarray
    lower_weights: np.ndarray
    message: str


@dataclass
class HierarchicalResult:
    """分层CST优化的完整结果"""
    airfoil: object  # asb.KulfanAirfoil
    final_cd: float
    total_time: float
    stages: list[StageResult]
    decision_log: list[dict]


class AdaptiveHierarchicalOptimizer:
    """
    自适应分层CST优化器。

    根据优化历史决定何时开放更多CST权重。
    """

    def __init__(
        self,
        CL_targets: np.ndarray,
        CL_weights: np.ndarray,
        Re: np.ndarray,
        mach: float = 0.03,
        start_weights: int = 4,
        improvement_threshold: float | None = None,
        stability_threshold: float = 0.005,
        router: object | None = None,  # OptRouter, lazy import
    ):
        self.CL_targets = CL_targets
        self.CL_weights = CL_weights
        self.Re = Re
        self.mach = mach
        self.start_weights = start_weights
        self.improvement_threshold = (
            improvement_threshold if improvement_threshold is not None
            else _load_router_threshold()
        )
        self.stability_threshold = stability_threshold
        self._router = router

    def _evaluate_cd(self, airfoil) -> float:
        """评估翼型的加权CD。"""
        from piern_airfoil.eval import evaluate_weighted_cd

        return evaluate_weighted_cd(
            airfoil, self.CL_targets, self.Re, self.CL_weights, mach=self.mach,
        )

    def _run_stage(
        self,
        airfoil,
        n_active: int,
        initial_weights: tuple | None = None,
    ) -> tuple:
        """
        运行一个CST优化阶段。

        Args:
            airfoil: 初始翼型
            n_active: 激活的权重数量 (1-8)
            initial_weights: 初始权重用于warm-start

        Returns:
            (优化后的翼型, 优化后的权重元组)
        """
        import casadi
        import aerosandbox as asb
        import aerosandbox.numpy as asbnp

        opti = asb.Opti()

        initial_upper = airfoil.upper_weights
        initial_lower = airfoil.lower_weights

        # 根据n_active决定哪些权重可优化
        upper_vars = []
        lower_vars = []
        upper_fixed = []
        lower_fixed = []

        for i in range(8):
            if i < n_active:
                # 可优化
                init_u = initial_weights[0][i] if initial_weights else initial_upper[i]
                init_l = initial_weights[1][i] if initial_weights else initial_lower[i]
                upper_vars.append(opti.variable(init_guess=float(init_u), lower_bound=-0.25, upper_bound=0.5))
                lower_vars.append(opti.variable(init_guess=float(init_l), lower_bound=-0.5, upper_bound=0.25))
            else:
                # 固定
                upper_fixed.append(float(initial_upper[i]))
                lower_fixed.append(float(initial_lower[i]))

        # 拼接权重
        if upper_fixed:
            upper_weights = casadi.vertcat(*upper_vars, *upper_fixed)
            lower_weights = casadi.vertcat(*lower_vars, *lower_fixed)
        else:
            upper_weights = casadi.vertcat(*upper_vars)
            lower_weights = casadi.vertcat(*lower_vars)

        optimized_airfoil = asb.KulfanAirfoil(
            name="Optimized",
            lower_weights=lower_weights,
            upper_weights=upper_weights,
            leading_edge_weight=opti.variable(
                init_guess=airfoil.leading_edge_weight,
                lower_bound=-1, upper_bound=1
            ),
            TE_thickness=0,
        )

        alpha = opti.variable(
            init_guess=np.degrees(self.CL_targets / (2 * np.pi)),
            lower_bound=-5, upper_bound=18,
        )

        aero = optimized_airfoil.get_aero_from_neuralfoil(alpha=alpha, Re=self.Re, mach=self.mach)

        # 约束
        opti.subject_to([
            aero["analysis_confidence"] > 0.90,
            aero["CL"] == self.CL_targets,
            asbnp.diff(alpha) > 0,
            aero["CM"] >= -0.133,
            optimized_airfoil.local_thickness(x_over_c=0.33) >= 0.128,
            optimized_airfoil.local_thickness(x_over_c=0.90) >= 0.014,
            optimized_airfoil.TE_angle() >= 6.03,
            optimized_airfoil.lower_weights[0] < -0.05,
            optimized_airfoil.upper_weights[0] > 0.05,
            optimized_airfoil.local_thickness() > 0,
            optimized_airfoil.LE_radius() > 0,  # θ_LE = 180° for CST airfoils
        ])

        get_wiggliness = lambda af: sum(
            asbnp.sum(asbnp.diff(asbnp.diff(array)) ** 2)
            for array in [af.lower_weights, af.upper_weights]
        )
        opti.subject_to(get_wiggliness(optimized_airfoil) < 2 * get_wiggliness(airfoil))

        opti.minimize(asbnp.mean(aero["CD"] * self.CL_weights))

        sol = opti.solve(
            behavior_on_failure="return_last",
            options={"ipopt.mu_strategy": "monotone", "ipopt.start_with_resto": "yes"},
        )
        result_airfoil = sol(optimized_airfoil)

        # 提取优化后的权重
        result_upper = np.array([float(sol(upper_vars[i])) for i in range(n_active)] +
                               [upper_fixed[i] for i in range(8 - n_active)])
        result_lower = np.array([float(sol(lower_vars[i])) for i in range(n_active)] +
                               [lower_fixed[i] for i in range(8 - n_active)])

        return result_airfoil, (result_upper, result_lower)

    def _get_router(self):
        """Get or create the OptRouter instance."""
        if self._router is None:
            from piern.router.opt_router import OptRouter, OptState
            self._router = OptRouter(improvement_threshold=self.improvement_threshold)
        return self._router

    def _decide_next_action(
        self,
        history: list[StageResult],
        n_active: int,
        init_cd: float = 0.0,
    ) -> tuple[int, str]:
        """
        基于优化历史决定下一步动作。

        Uses OptRouter for routing decisions (supports rule/threshold/mlp modes).

        Returns:
            (新的n_active, 决策理由)
        """
        from piern.router.opt_router import OptState

        if len(history) < 2:
            return n_active, "首次运行，继续观察"

        router = self._get_router()

        # Compute stall count
        stall_count = 0
        for i in range(len(history) - 1, 0, -1):
            imp = (history[i - 1].cd - history[i].cd) / history[i - 1].cd
            if imp < 0.001:
                stall_count += 1
            else:
                break

        state = OptState(
            stage=len(history),
            n_active_weights=n_active,
            cd=history[-1].cd,
            prev_cd=history[-2].cd,
            initial_cd=init_cd,
            stall_count=stall_count,
            max_stages=6,
        )

        action, new_n, reason = router.decide(state)

        # Map OptAction to n_active
        if action.value == "keep":
            return new_n, reason
        elif action.value == "expand":
            return new_n, reason
        else:
            return new_n, reason

    def optimize(self, initial_airfoil) -> HierarchicalResult:
        """
        运行自适应分层CST优化。

        Returns:
            HierarchicalResult 包含完整优化历史
        """
        import time

        t0 = time.perf_counter()

        current_airfoil = initial_airfoil
        current_weights = (initial_airfoil.upper_weights, initial_airfoil.lower_weights)
        history = []
        decision_log = []
        n_active = self.start_weights

        # 初始评估
        init_cd = self._evaluate_cd(initial_airfoil)

        max_stages = 6  # 最多6个阶段
        for stage_idx in range(max_stages):
            # 最后一阶段强制使用全部8个权重，确保最终精度
            if stage_idx == max_stages - 1:
                n_active = 8

            # 运行当前阶段
            result_airfoil, result_weights = self._run_stage(
                current_airfoil, n_active, current_weights
            )

            # 评估结果
            cd = self._evaluate_cd(result_airfoil)

            # 记录阶段结果
            stage_result = StageResult(
                stage=stage_idx + 1,
                n_active_weights=n_active,
                cd=cd,
                airfoil=result_airfoil,
                upper_weights=result_weights[0],
                lower_weights=result_weights[1],
                message="",
            )
            history.append(stage_result)

            # 决策下一步
            new_n_active, message = self._decide_next_action(history, n_active, init_cd)
            stage_result.message = message

            # 记录决策日志
            decision_log.append({
                "stage": stage_idx + 1,
                "n_active": n_active,
                "cd": cd,
                "decision": message,
            })

            # 更新状态
            current_airfoil = result_airfoil
            current_weights = result_weights
            n_active = new_n_active

            # 如果已经达到8个权重且完成优化，结束
            if n_active >= 8 and len(history) > 1 and history[-1].n_active_weights == 8:
                break

        elapsed = time.perf_counter() - t0

        # 找到最佳结果
        best_stage = min(history, key=lambda h: h.cd)

        return HierarchicalResult(
            airfoil=best_stage.airfoil,
            final_cd=best_stage.cd,
            total_time=elapsed,
            stages=history,
            decision_log=decision_log,
        )
