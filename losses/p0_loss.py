"""
P0 Loss: 股价相关损失（不投资情形）

核心目标：拟合股价 Bellman 方程残差与债务选择 FOC 残差，
叠加单调性惩罚与 z 值惩罚，确保经济意义一致性。

支持任意数量的分支路径：
- parent: (P0_t, CF0p_t, ...) → t 期父节点
- children: [(P_{t+1}^{(j)}, bar_z_{t+1}^{(j)}, ...)] → N 条模拟路径

L_P0 = loss_bellman + loss_foc + mono_penalty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List

import sys
sys.path.append('..')
from config import Config

from .utils import (
    compute_aio_residual, 
    compute_z_penalty, 
    compute_monotonicity_penalty,
    compute_cashflow
)


class P0Loss(nn.Module):
    """
    P0（不投资时股价）损失函数
    
    包含：
    - Bellman 残差损失
    - FOC 残差损失（债务选择一阶条件）
    - 单调性惩罚
    - z 值惩罚
    """
    
    def __init__(
        self,
        delta: float = None,
        tau: float = None,
        kappa_b: float = None,
        kappa_e: float = None,
        aio_weight: float = None,
        alpha_z: float = None,
        beta_z: float = None,
        z0: float = None,
        lambda_b: float = 1.0,
        lambda_z: float = 1.0
    ):
        super().__init__()
        
        # 经济参数
        self.delta = delta if delta is not None else Config.DELTA
        self.tau = tau if tau is not None else Config.TAU
        self.kappa_b = kappa_b if kappa_b is not None else Config.KAPPA_B
        self.kappa_e = kappa_e if kappa_e is not None else Config.KAPPA_E
        
        # 损失权重
        self.aio_weight = aio_weight if aio_weight is not None else Config.AIO_WEIGHT
        self.alpha_z = alpha_z if alpha_z is not None else Config.ALPHA_Z
        self.beta_z = beta_z if beta_z is not None else Config.BETA_Z
        self.z0 = z0 if z0 is not None else Config.Z0
        self.lambda_b = lambda_b
        self.lambda_z = lambda_z
    
    def compute_cashflow_p0(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        b: torch.Tensor,
        Q: torch.Tensor,
        Qp: torch.Tensor,
        eta: torch.Tensor
    ) -> torch.Tensor:
        """
        计算不投资时的股东现金流
        
        CF0p = prod + ((1-κb)Qp - Q) * η + κe * max(-CF0p, 0) * CF0p
        """
        # 税后利润
        prod = compute_cashflow(x, z, b, self.delta, self.tau)
        
        # 债务调整项
        debt_adj = ((1 - self.kappa_b) * Qp - Q) * eta
        
        # 初始现金流
        cf0p_raw = prod + debt_adj
        
        # 股权融资成本调整（软化版本）
        equity_cost = self.kappa_e * F.relu(-cf0p_raw)
        cf0p = cf0p_raw - equity_cost * cf0p_raw.sign()
        
        return cf0p
    

    def compute_bellman_residual(
        self,
        P0: torch.Tensor,
        CF0p: torch.Tensor,
        M_list: List[torch.Tensor],
        P_children: List[torch.Tensor],
        bar_z_children: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        计算 Bellman 残差（支持任意分支数）
        
        loss^{(j)} = P0 - CF0p - M^{(j)} * P'^{(j)} * (1 - bar_z'^{(j)})
        
        Args:
            P0: 当期股价
            CF0p: 当期现金流
            M_list: List[torch.Tensor] - 各路径的 SDF
            P_children: List[torch.Tensor] - 各路径的未来股价
            bar_z_children: List[torch.Tensor] - 各路径的违约阈值
        
        Returns:
            residuals: List[torch.Tensor] - 各路径的残差
        """
        residuals = []
        for M, P_child, bar_z in zip(M_list, P_children, bar_z_children):
            residual = P0 - CF0p - M * P_child * (1 - bar_z)
            residuals.append(residual)
        
        return residuals

    def compute_bellman_residual_legacy(
        self,
        P0: torch.Tensor,
        CF0p: torch.Tensor,
        M1: torch.Tensor,
        M2: torch.Tensor,
        P2: torch.Tensor,
        P3: torch.Tensor,
        bar_z2: torch.Tensor,
        bar_z3: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算 Bellman 残差（旧版双路径接口）
        """
        residuals = self.compute_bellman_residual(
            P0, CF0p,
            M_list=[M1, M2],
            P_children=[P2, P3],
            bar_z_children=[bar_z2, bar_z3]
        )
        return residuals[0], residuals[1]
    
    def compute_foc_residual(
        self,
        cf0p_grad: torch.Tensor,
        M_list: List[torch.Tensor],
        P_grads: List[torch.Tensor],
        bar_z_children: List[torch.Tensor],
        eta: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        计算 FOC 残差（债务选择一阶条件）- 支持任意分支数
        
        foc^{(j)} = cf_grad + M^{(j)} * P'_grad^{(j)} * (1 - bar_z'^{(j)}) * η
        """
        residuals = []
        for M, P_grad, bar_z in zip(M_list, P_grads, bar_z_children):
            foc = cf0p_grad + M * P_grad * (1 - bar_z) * eta
            residuals.append(foc)
        return residuals

    def compute_foc_residual_from_bp(
        self,
        CF0p: torch.Tensor,
        M_list: List[torch.Tensor],
        P_children: List[torch.Tensor],
        bar_z_children: List[torch.Tensor],
        bp: torch.Tensor,
        eta: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        基于 bp 直接计算 FOC 残差（支持任意分支数）

        先计算：
        - cf0p_grad = ∂CF0p/∂bp
        - P'_grad^{(j)} = ∂P_children^{(j)}/∂bp

        再代入：
        foc^{(j)} = cf0p_grad + M^{(j)} * P'_grad^{(j)} * (1 - bar_z'^{(j)}) * η
        """
        cf0p_grad = torch.autograd.grad(
            outputs=CF0p.sum(),
            inputs=bp,
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]
        if cf0p_grad is None:
            cf0p_grad = torch.zeros_like(bp)

        p_grads = []
        for P_child in P_children:
            p_grad = torch.autograd.grad(
                outputs=P_child.sum(),
                inputs=bp,
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]
            if p_grad is None:
                p_grad = torch.zeros_like(bp)

            p_grads.append(p_grad)

        return self.compute_foc_residual(
            cf0p_grad=cf0p_grad,
            M_list=M_list,
            P_grads=p_grads,
            bar_z_children=bar_z_children,
            eta=eta
        )
    
    def compute_foc_residual_legacy(
        self,
        cf0p_grad: torch.Tensor,
        M1: torch.Tensor,
        M2: torch.Tensor,
        P2_grad: torch.Tensor,
        P3_grad: torch.Tensor,
        bar_z2: torch.Tensor,
        bar_z3: torch.Tensor,
        eta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算 FOC 残差（旧版双路径接口）
        """
        residuals = self.compute_foc_residual(
            cf0p_grad,
            M_list=[M1, M2],
            P_grads=[P2_grad, P3_grad],
            bar_z_children=[bar_z2, bar_z3],
            eta=eta
        )
        return residuals[0], residuals[1]
    
    def forward(
        self,
        P0: torch.Tensor,
        inputs: torch.Tensor,
        Q: torch.Tensor,
        Qp: torch.Tensor,
        M_list: List[torch.Tensor],
        P_children: List[torch.Tensor],
        bar_z_children: List[torch.Tensor],
        foc_only: bool = False,
        bellman_only: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算 P0 损失（支持任意分支数）
        
        Args:
            P0: 当前 P0 值
            inputs: 输入状态 (b, z, η, i, x, ĉf, ln Kf)
            Q: 当前债券价格
            Qp: 最优债务对应的债券价格
            M_list: List[torch.Tensor] - 各路径的 SDF
            P_children: List[torch.Tensor] - 各路径的未来股价
            bar_z_children: List[torch.Tensor] - 各路径的违约阈值
            foc_only: 是否只计算 FOC 损失
            bellman_only: 是否只计算 Bellman 损失
        
        Returns:
            total_loss: 总损失
            loss_dict: 各分量损失字典
        """
        # 提取状态变量
        b = inputs[:, 0:1]
        z = inputs[:, 1:2]
        eta = inputs[:, 2:3]
        x = inputs[:, 4:5]
        
        # 计算现金流
        CF0p = self.compute_cashflow_p0(x, z, b, Q, Qp, eta)
        
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=P0.device)
        
        # Bellman 损失
        if not foc_only:
            residuals = self.compute_bellman_residual(
                P0, CF0p, M_list, P_children, bar_z_children
            )
            
            bellman_residual = compute_aio_residual(residuals, self.aio_weight)
            loss_bellman = bellman_residual.mean()
            
            # z 惩罚
            penalty_z_bellman = compute_z_penalty(
                bellman_residual, z, self.alpha_z, self.beta_z, self.z0
            )
            
            loss_dict['loss_bellman'] = loss_bellman
            loss_dict['penalty_z_bellman'] = penalty_z_bellman
            
            total_loss = total_loss + loss_bellman + penalty_z_bellman
        
        # FOC 损失（仅当 η ≠ 0 时有意义）
        if not bellman_only and eta.sum() > 0:
            # 需要计算梯度
            if inputs.requires_grad:
                # 计算对 b' 的梯度
                # 这里简化处理，实际需要对 Qp, P_children 分别求导
                foc_residual = torch.zeros_like(P0)  # placeholder
                
                loss_foc = (foc_residual.abs() * eta).mean()
                penalty_z_foc = compute_z_penalty(
                    foc_residual.abs() * eta, z, self.alpha_z, self.beta_z, self.z0
                )
                
                loss_dict['loss_foc'] = loss_foc
                loss_dict['penalty_z_foc'] = penalty_z_foc
                
                total_loss = total_loss + loss_foc + penalty_z_foc
        
        # 单调性惩罚
        if inputs.requires_grad:
            # ∂P0/∂b < 0（P0 对 b 应单调递减）
            mono_b = compute_monotonicity_penalty(P0, inputs, 0, 'negative')
            # ∂P0/∂z > 0（P0 对 z 应单调递增）
            mono_z = compute_monotonicity_penalty(P0, inputs, 1, 'positive')
            
            mono_penalty = self.lambda_b * mono_b + self.lambda_z * mono_z
            
            loss_dict['mono_penalty'] = mono_penalty
            total_loss = total_loss + mono_penalty
        
        loss_dict['total_loss'] = total_loss
        
        return total_loss, loss_dict
    
    def forward_legacy(
        self,
        P0: torch.Tensor,
        inputs: torch.Tensor,
        Q: torch.Tensor,
        Qp: torch.Tensor,
        M1: torch.Tensor,
        M2: torch.Tensor,
        P2: torch.Tensor,
        P3: torch.Tensor,
        bar_z2: torch.Tensor,
        bar_z3: torch.Tensor,
        foc_only: bool = False,
        bellman_only: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        旧版接口（兼容2分支情况）
        """
        return self.forward(
            P0, inputs, Q, Qp,
            M_list=[M1, M2],
            P_children=[P2, P3],
            bar_z_children=[bar_z2, bar_z3],
            foc_only=foc_only,
            bellman_only=bellman_only
        )
        
        # FOC 损失（仅当 η ≠ 0 时有意义）
        if not bellman_only and eta.sum() > 0:
            # 需要计算梯度
            if inputs.requires_grad:
                # 计算对 b' 的梯度
                # 这里简化处理，实际需要对 Qp, P2, P3 分别求导
                foc_residual = torch.zeros_like(P0)  # placeholder
                
                loss_foc = (foc_residual.abs() * eta).mean()
                penalty_z_foc = compute_z_penalty(
                    foc_residual.abs() * eta, z, self.alpha_z, self.beta_z, self.z0
                )
                
                loss_dict['loss_foc'] = loss_foc
                loss_dict['penalty_z_foc'] = penalty_z_foc
                
                total_loss = total_loss + loss_foc + penalty_z_foc
        
        # 单调性惩罚
        if inputs.requires_grad:
            # ∂P0/∂b < 0（P0 对 b 应单调递减）
            mono_b = compute_monotonicity_penalty(P0, inputs, 0, 'negative')
            # ∂P0/∂z > 0（P0 对 z 应单调递增）
            mono_z = compute_monotonicity_penalty(P0, inputs, 1, 'positive')
            
            mono_penalty = self.lambda_b * mono_b + self.lambda_z * mono_z
            
            loss_dict['mono_penalty'] = mono_penalty
            total_loss = total_loss + mono_penalty
        
        loss_dict['total_loss'] = total_loss
        
        return total_loss, loss_dict
    
    def forward_simplified(
        self,
        P0: torch.Tensor,
        CF0p: torch.Tensor,
        M_list: List[torch.Tensor],
        P_children: List[torch.Tensor],
        bar_z_children: List[torch.Tensor],
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        简化版 P0 损失（支持任意分支数）
        """
        # Bellman 残差
        residuals = self.compute_bellman_residual(
            P0, CF0p, M_list, P_children, bar_z_children
        )
        
        # AIO 残差
        bellman_residual = compute_aio_residual(residuals, self.aio_weight)
        loss_bellman = bellman_residual.mean()
        
        # z 惩罚
        penalty_z = compute_z_penalty(bellman_residual, z, self.alpha_z, self.beta_z, self.z0)
        
        total_loss = loss_bellman + penalty_z
        
        loss_dict = {
            'loss_bellman': loss_bellman,
            'penalty_z': penalty_z,
            'total_loss': total_loss
        }
        
        return total_loss, loss_dict
    
    def forward_simplified_legacy(
        self,
        P0: torch.Tensor,
        CF0p: torch.Tensor,
        M1: torch.Tensor,
        M2: torch.Tensor,
        P2: torch.Tensor,
        P3: torch.Tensor,
        bar_z2: torch.Tensor,
        bar_z3: torch.Tensor,
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        简化版 P0 损失（旧版双路径接口）
        """
        return self.forward_simplified(
            P0, CF0p,
            M_list=[M1, M2],
            P_children=[P2, P3],
            bar_z_children=[bar_z2, bar_z3],
            z=z
        )
