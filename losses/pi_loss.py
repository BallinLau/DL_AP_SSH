"""
PI Loss: 投资相关损失

核心目标：拟合投资价格方程残差与投资选择 FOC 残差，
叠加 b>1 惩罚、z 值惩罚与单调性惩罚，确保投资决策合理性。

支持任意数量的分支路径：
- parent: (PI_t, CFip_t, ...) → t 期父节点
- children: [(P_{t+1}^{(j)}, bar_z_{t+1}^{(j)}, ...)] → N 条模拟路径

L_PI = loss_bellman + loss_foc + mono_penalty
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


class PILoss(nn.Module):
    """
    PI（投资时股价）损失函数
    
    包含：
    - Bellman 残差损失
    - FOC 残差损失
    - 单调性惩罚
    - z 值惩罚
    - b > 1 惩罚
    """
    
    def __init__(
        self,
        delta: float = None,
        tau: float = None,
        g: float = None,
        kappa_b: float = None,
        kappa_e: float = None,
        aio_weight: float = None,
        alpha_z: float = None,
        beta_z: float = None,
        z0: float = None,
        lambda_b: float = 1.0,
        lambda_z: float = 1.0,
        b_penalty_weight: float = 1.0
    ):
        super().__init__()
        
        # 经济参数
        self.delta = delta if delta is not None else Config.DELTA
        self.tau = tau if tau is not None else Config.TAU
        self.g = g if g is not None else Config.G
        self.kappa_b = kappa_b if kappa_b is not None else Config.KAPPA_B
        self.kappa_e = kappa_e if kappa_e is not None else Config.KAPPA_E
        
        # 损失权重
        self.aio_weight = aio_weight if aio_weight is not None else Config.AIO_WEIGHT
        self.alpha_z = alpha_z if alpha_z is not None else Config.ALPHA_Z
        self.beta_z = beta_z if beta_z is not None else Config.BETA_Z
        self.z0 = z0 if z0 is not None else Config.Z0
        self.lambda_b = lambda_b
        self.lambda_z = lambda_z
        self.b_penalty_weight = b_penalty_weight
    
    def compute_cashflow_pi(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        b: torch.Tensor,
        i: torch.Tensor,
        Q: torch.Tensor,
        QpI: torch.Tensor,
        eta: torch.Tensor
    ) -> torch.Tensor:
        """
        计算投资时的股东现金流
        
        CFip = prod - i + ((1-κb) * g * QpI - Q) * η + κe * max(-CFip, 0) * CFip
        """
        # 税后利润
        prod = compute_cashflow(x, z, b, self.delta, self.tau)
        
        # 投资成本
        invest_cost = i
        
        # 债务调整项（考虑增长因子）
        debt_adj = ((1 - self.kappa_b) * self.g * QpI - Q) * eta
        
        # 初始现金流
        cfip_raw = prod - invest_cost + debt_adj
        
        # 股权融资成本调整
        equity_cost = self.kappa_e * F.relu(-cfip_raw)
        cfip = cfip_raw - equity_cost * cfip_raw.sign()
        
        return cfip
    
    def compute_bellman_residual(
        self,
        PI: torch.Tensor,
        CFip: torch.Tensor,
        M_list: List[torch.Tensor],
        P_children: List[torch.Tensor],
        bar_z_children: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        计算投资 Bellman 残差（支持任意分支数）
        
        loss^{(j)} = PI - CFip - g * M^{(j)} * P'^{(j)} * (1 - bar_z'^{(j)})
        """
        residuals = []
        for M, P_child, bar_z in zip(M_list, P_children, bar_z_children):
            residual = PI - CFip - self.g * M * P_child * (1 - bar_z)
            residuals.append(residual)
        return residuals
    
    def compute_bellman_residual_legacy(
        self,
        PI: torch.Tensor,
        CFip: torch.Tensor,
        M1: torch.Tensor,
        M2: torch.Tensor,
        P2: torch.Tensor,
        P3: torch.Tensor,
        bar_z2: torch.Tensor,
        bar_z3: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算投资 Bellman 残差（旧版双路径接口）
        """
        residuals = self.compute_bellman_residual(
            PI, CFip,
            M_list=[M1, M2],
            P_children=[P2, P3],
            bar_z_children=[bar_z2, bar_z3]
        )
        return residuals[0], residuals[1]

    def compute_foc_residual(
        self,
        cfip_grad: torch.Tensor,
        M_list: List[torch.Tensor],
        P_grads: List[torch.Tensor],
        bar_z_children: List[torch.Tensor],
        eta: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        计算 PI 的 FOC 残差（支持任意分支数）

        foc^{(j)} = cf_grad + g * M^{(j)} * P'_grad^{(j)} * (1 - bar_z'^{(j)}) * η
        """
        residuals = []
        for M, P_grad, bar_z in zip(M_list, P_grads, bar_z_children):
            foc = cfip_grad + self.g * M * P_grad * (1 - bar_z) * eta
            residuals.append(foc)
        return residuals

    def compute_foc_residual_from_bp(
        self,
        CFip: torch.Tensor,
        M_list: List[torch.Tensor],
        P_children: List[torch.Tensor],
        bar_z_children: List[torch.Tensor],
        bp: torch.Tensor,
        eta: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        基于 bp 直接计算 PI 的 FOC 残差（支持任意分支数）

        先计算：
        - cfip_grad = ∂CFip/∂bp
        - P'_grad^{(j)} = ∂P_children^{(j)}/∂bp
        """
        cfip_grad = torch.autograd.grad(
            outputs=CFip.sum(),
            inputs=bp,
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]
        if cfip_grad is None:
            cfip_grad = torch.zeros_like(bp)

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
            cfip_grad=cfip_grad,
            M_list=M_list,
            P_grads=p_grads,
            bar_z_children=bar_z_children,
            eta=eta
        )
    
    def compute_b_penalty(
        self,
        PI: torch.Tensor,
        b: torch.Tensor,
        threshold: float = 1.0
    ) -> torch.Tensor:
        """
        计算 b > 1 的惩罚
        
        当 b > 1 时，惩罚 PI^2
        """
        mask = (b > threshold - 1e-1).float()
        return PI.pow(2) * mask
    
    def forward(
        self,
        PI: torch.Tensor,
        inputs: torch.Tensor,
        Q: torch.Tensor,
        QpI: torch.Tensor,
        M_list: List[torch.Tensor],
        P_children: List[torch.Tensor],
        bar_z_children: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算 PI 损失（支持任意分支数）
        
        Args:
            PI: 当前 PI 值
            inputs: 输入状态 (b, z, η, i, x, ĉf, ln Kf)
            Q: 当前债券价格
            QpI: 投资场景下最优债务对应的债券价格
            M_list: List[torch.Tensor] - 各路径的 SDF
            P_children: List[torch.Tensor] - 各路径的未来股价
            bar_z_children: List[torch.Tensor] - 各路径的违约阈值
        
        Returns:
            total_loss: 总损失
            loss_dict: 各分量损失字典
        """
        # 提取状态变量
        b = inputs[:, 0:1]
        z = inputs[:, 1:2]
        eta = inputs[:, 2:3]
        i = inputs[:, 3:4]
        x = inputs[:, 4:5]
        
        # 计算现金流
        CFip = self.compute_cashflow_pi(x, z, b, i, Q, QpI, eta)
        
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=PI.device)
        
        # Bellman 损失
        residuals = self.compute_bellman_residual(
            PI, CFip, M_list, P_children, bar_z_children
        )
        
        bellman_residual = compute_aio_residual(residuals, self.aio_weight)
        loss_bellman = bellman_residual.mean()
        
        # z 惩罚
        penalty_z_bellman = compute_z_penalty(
            bellman_residual, z, self.alpha_z, self.beta_z, self.z0
        )
        
        # b > 1 惩罚
        penalty_b = self.b_penalty_weight * self.compute_b_penalty(PI, b)
        
        loss_dict['loss_bellman'] = loss_bellman
        loss_dict['penalty_z_bellman'] = penalty_z_bellman
        loss_dict['penalty_b'] = penalty_b
        
        total_loss = total_loss + loss_bellman + penalty_z_bellman + penalty_b
        
        # 单调性惩罚
        if inputs.requires_grad:
            # ∂PI/∂b < 0
            mono_b = compute_monotonicity_penalty(PI, inputs, 0, 'negative')
            # ∂PI/∂z > 0
            mono_z = compute_monotonicity_penalty(PI, inputs, 1, 'positive')
            
            mono_penalty = self.lambda_b * mono_b + self.lambda_z * mono_z
            
            loss_dict['mono_penalty'] = mono_penalty
            total_loss = total_loss + mono_penalty
        
        loss_dict['total_loss'] = total_loss
        
        return total_loss, loss_dict
    
    def forward_legacy(
        self,
        PI: torch.Tensor,
        inputs: torch.Tensor,
        Q: torch.Tensor,
        QpI: torch.Tensor,
        M1: torch.Tensor,
        M2: torch.Tensor,
        P2: torch.Tensor,
        P3: torch.Tensor,
        bar_z2: torch.Tensor,
        bar_z3: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        旧版接口（兼容2分支情况）
        """
        return self.forward(
            PI, inputs, Q, QpI,
            M_list=[M1, M2],
            P_children=[P2, P3],
            bar_z_children=[bar_z2, bar_z3]
        )
    
    def forward_simplified(
        self,
        PI: torch.Tensor,
        CFip: torch.Tensor,
        M_list: List[torch.Tensor],
        P_children: List[torch.Tensor],
        bar_z_children: List[torch.Tensor],
        z: torch.Tensor,
        b: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        简化版 PI 损失（支持任意分支数）
        """
        # Bellman 残差
        residuals = self.compute_bellman_residual(
            PI, CFip, M_list, P_children, bar_z_children
        )
        
        # AIO 残差
        bellman_residual = compute_aio_residual(residuals, self.aio_weight)
        loss_bellman = bellman_residual.mean()
        
        # z 惩罚
        penalty_z = compute_z_penalty(bellman_residual, z, self.alpha_z, self.beta_z, self.z0)
        
        # b > 1 惩罚
        penalty_b = self.b_penalty_weight * self.compute_b_penalty(PI, b)
        
        total_loss = loss_bellman + penalty_z + penalty_b
        
        loss_dict = {
            'loss_bellman': loss_bellman,
            'penalty_z': penalty_z,
            'penalty_b': penalty_b,
            'total_loss': total_loss
        }
        
        return total_loss, loss_dict
    
    def forward_simplified_legacy(
        self,
        PI: torch.Tensor,
        CFip: torch.Tensor,
        M1: torch.Tensor,
        M2: torch.Tensor,
        P2: torch.Tensor,
        P3: torch.Tensor,
        bar_z2: torch.Tensor,
        bar_z3: torch.Tensor,
        z: torch.Tensor,
        b: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        简化版 PI 损失（旧版双路径接口）
        """
        return self.forward_simplified(
            PI, CFip,
            M_list=[M1, M2],
            P_children=[P2, P3],
            bar_z_children=[bar_z2, bar_z3],
            z=z, b=b
        )
