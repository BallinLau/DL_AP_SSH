"""
Episode 类：训练周期管理

一个 Episode 包含：
1. 数据生成（Sample 或 SimulateTS）
2. FC1 填充
3. Policy/Value 填充
4. 各模块的训练循环
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import logging

import sys
sys.path.append('..')
from config import Config, HyperParams, SIMMODEL
from data import Sample, SimulateTS
from data.data_utils import compute_quantile_features
from losses import SDFLoss, P0Loss, PILoss, QLoss, FC2Loss
from losses.FC2losspipe import FC2LossPipe
from losses.utils import compute_z_penalty
from losses.sdf_loss import moment_penalty
from data.data_utils import build_sdf_pairs_from_macro_ts
from .gradient_utils import gradient_protection, compute_gradient_norm
from .scheduler import LossWeightScheduler, LearningRateScheduler


# Use project-level logger so outputs go to the configured handlers
logger = logging.getLogger('DL-AP')


def convert_tree_fast(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['path_ori'] = df['path']
    df['t_ori'] = df['t']
    df['branch_ori'] = df['branch']

    parent_t = np.where(
        df['branch'] == -1,
        df['t'],
        df['t'] - 1
    )

    state_keys = list(zip(df['path_ori'], parent_t))
    df['path'] = pd.factorize(state_keys)[0].astype('int32')

    df['branch'] = df['branch'].map({-1: 0, 0: 1, 1: 2}).astype('int8')

    df['t'] = np.select(
        [
            df['branch'] == 0,
            df['branch'] == 1,
            df['branch'] == 2
        ],
        ['t', 't+1_0', 't+1_1']
    )

    df = df.sort_values(['path', 'branch'], kind='mergesort')

    return df


def trim_child_only_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop child-only firms (IDs that do not appear in parent branch=0).
    This enforces parent-aligned IDs before fill_df_to_fullN.
    """
    df = df.copy()
    allowed = df[df['branch'] == 0][['path', 'ID']].drop_duplicates()
    allowed['__keep__'] = True
    df = df.merge(allowed, on=['path', 'ID'], how='left')
    df = df[df['__keep__'].fillna(False)].drop(columns='__keep__')
    return df


class Episode:
    """
    训练 Episode
    
    管理单个训练周期的数据生成、填充和训练流程
    """

    def __init__(
        self,
        models: Dict[str, nn.Module],
        optimizers: Dict[str, torch.optim.Optimizer],
        config: type = Config,
        hyperparams: HyperParams = None,
        device: torch.device = None,
        episode_id: int = 0
    ):
        """
        Args:
            models: 模型字典
                - 'sdf_fc1': SDFFC1Combined
                - 'policy_value': PolicyValueModel
                - 'fc2': FC2Model (可选)
            optimizers: 优化器字典
            config: 配置类
            hyperparams: 超参数
            device: 设备
            episode_id: Episode 编号
        """
        self.models = models
        self.optimizers = optimizers
        self.config = config
        self.hyperparams = hyperparams or HyperParams()
        self.device = device or config.DEVICE
        self.episode_id = episode_id
        self.train_mode = '2time'  # 默认训练模式，可以在 run_episode 参数中覆盖
        
        # 损失函数
        self.loss_fns = self._init_loss_functions()
        
        # 损失权重调度器
        self.weight_scheduler = self._init_weight_scheduler()
        
        # 学习率调度器
        self.lr_schedulers = self._init_lr_schedulers()
        
        # 数据
        self.df = None
        self.df_macro = None
        self.df_sdf = None
        
        # 训练状态
        self.step_count = 0
        self.loss_history = {}
        self.add_FC1loss = False
    
    def _init_loss_functions(self) -> Dict:
        """
        初始化损失函数
        """
        return {
            'sdf': SDFLoss(),
            'p0': P0Loss(),
            'pi': PILoss(),
            'q': QLoss(),
            'fc2': FC2Loss() if 'fc2' in self.models else None
        }
    
    def _init_weight_scheduler(self) -> LossWeightScheduler:
        """
        初始化损失权重调度器
        """
        initial_weights = {
            'sdf': self.hyperparams.w_sdf,
            'p0': self.hyperparams.w_p0,
            'pi': self.hyperparams.w_pi,
            'q': self.hyperparams.w_q,
            'fc2': self.hyperparams.w_fc2
        }
        
        return LossWeightScheduler(
            initial_weights,
            schedule_type='fixed',
            warmup_steps=self.hyperparams.warmup_steps,
            total_steps=self.hyperparams.max_steps
        )
    
    def _init_lr_schedulers(self) -> Dict:
        """
        初始化学习率调度器
        """
        schedulers = {}
        
        for name, optimizer in self.optimizers.items():
            schedulers[name] = LearningRateScheduler(
                optimizer,
                base_lr=self.hyperparams.lr,
                warmup_steps=self.hyperparams.warmup_steps,
                decay_type='cosine',
                total_steps=self.hyperparams.max_steps,
                min_lr=self.hyperparams.min_lr
            )
        
        return schedulers
    
    def generate_data(
        self,
        mode: str = 'sample',
        n_samples: int = 10000,
        n_paths: int = 100,
        group_size: int = 100,
        **kwargs
    ) -> pd.DataFrame:
        """
        生成训练数据
        
        Args:
            mode: 'sample' 或 'simulate'
            n_samples: 样本数
            n_paths: path 数量
            group_size: 每条 path 的公司数
        
        Returns:
            df: 生成的 DataFrame
        """
        
        if mode == 'sample':
            sampler = Sample(
                models=self.models,
                config=self.config,
                n_samples=None,
                n_paths=n_paths,
                group_size=group_size,
                **kwargs
            )
            self.df = sampler.build_df()
            
        elif mode == 'simulate':
            simulator = SimulateTS(
                models=self.models,
                config=self.config,
                n_paths=n_paths,
                group_size=group_size,
                **kwargs
            )
            self.df, self.df_macro = simulator.simulate()
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return self.df
    
    def fill_fc1(self):
        """
        使用 FC1 填充宏观状态
        """
        if self.df is None:
            raise RuntimeError("先调用 generate_data()")
        
        sampler = Sample(models=self.models, config=self.config)
        self.df = sampler.fill_fc1(self.df)
    
    def fill_policy_value(self):
        """
        使用 Policy/Value 填充决策和价值变量
        """
        if self.df is None:
            raise RuntimeError("先调用 generate_data()")
        
        sampler = Sample(models=self.models, config=self.config)
        self.df = sampler.fill_policy_value(self.df)
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        train_modules: List[str] = None
    ) -> Dict[str, float]:
        """
        单步训练
        
        Args:
            batch: 数据批次
            train_modules: 要训练的模块列表
        
        Returns:
            losses: 各损失值字典
        """
        if train_modules is None:
            train_modules = ['sdf_fc1', 'policy_value']
        
        losses = {}
        
        # 设置训练模式
        for name in train_modules:
            if name in self.models:
                self.models[name].train()
        
        # 清零梯度
        for name in train_modules:
            if name in self.optimizers:
                self.optimizers[name].zero_grad()
        
        # 计算损失
        total_loss = torch.tensor(0.0, device=self.device)
        
        # SDF Loss
        if 'sdf_fc1' in train_modules and 'sdf_fc1' in self.models:
            sdf_loss = self._compute_sdf_loss(batch)
            losses['sdf'] = sdf_loss.item()
            total_loss = total_loss + self.weight_scheduler['sdf'] * sdf_loss
        
        # P0 Loss
        if 'policy_value' in train_modules and 'policy_value' in self.models:
            p0_loss = self._compute_p0_loss(batch)
            losses['p0'] = p0_loss.item()
            total_loss = total_loss + self.weight_scheduler['p0'] * p0_loss
        
        # PI Loss
        if 'policy_value' in train_modules and 'policy_value' in self.models:
            pi_loss = self._compute_pi_loss(batch)
            losses['pi'] = pi_loss.item()
            total_loss = total_loss + self.weight_scheduler['pi'] * pi_loss
        
        # Q Loss
        if 'policy_value' in train_modules and 'policy_value' in self.models:
            q_loss = self._compute_q_loss(batch)
            losses['q'] = q_loss.item()
            total_loss = total_loss + self.weight_scheduler['q'] * q_loss
        
        # FC2 Loss
        if 'fc2' in train_modules and 'fc2' in self.models:
            if self.episode_id > 0 and isinstance(batch, pd.DataFrame):
                batch.to_csv(f"fc2_input_episode{self.episode_id}_ori.csv", index=False)
                batch = convert_tree_fast(batch)
                batch = trim_child_only_ids(batch)
                batch.to_csv(f"fc2_input_episode{self.episode_id}.csv", index=False)
            
            fc2_loss = self._compute_fc2_loss(batch)
            losses['fc2'] = fc2_loss.item()
            total_loss = total_loss + self.weight_scheduler['fc2'] * fc2_loss
        
        losses['total'] = total_loss.item()
        
        # 反向传播
        if total_loss.requires_grad:
            total_loss.backward()
            
            # 梯度保护
            for name in train_modules:
                if name in self.models:
                    grad_norm, had_nan = gradient_protection(
                        self.models[name].parameters(),
                        max_norm=self.hyperparams.max_grad_norm
                    )
                    losses[f'{name}_grad_norm'] = grad_norm
                    
                    if had_nan:
                        logger.warning(f"NaN gradient detected in {name}")
            
            # 优化器步骤
            for name in train_modules:
                if name in self.optimizers:
                    self.optimizers[name].step()
        
        # 更新调度器
        self.weight_scheduler.step(losses)
        for scheduler in self.lr_schedulers.values():
            scheduler.step()
        
        self.step_count += 1
        
        # 记录历史
        for k, v in losses.items():
            if k not in self.loss_history:
                self.loss_history[k] = []
            self.loss_history[k].append(v)
        
        return losses
    
    def _compute_sdf_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算 SDF 损失（支持任意 N 分支）
        """
        model = self.models['sdf_fc1']
        loss_fn = self.loss_fns['sdf']
        
        
        
        # 提取输入
        parent = batch['parent']
        children = batch.get('children', [])  # List of child tensors
        
        # 兼容旧的 child0/child1 格式
        if not children:
            child0 = batch.get('child0')
            child1 = batch.get('child1')
            if child0 is not None and child1 is not None:
                children = [child0, child1]

        if not children or len(children) < 2:
            raise ValueError("Need at least two children in batch for SDF loss")

        # 仅使用 child0 / child1
        if len(children) > 2:
            children = children[:2]

        children_t = torch.stack(children, dim=1)  # (batch, 2, feat)

        # 前向传播：一次性处理两条子路径
        w_parent, w_children, M, c_children, k_children = model.forward_step(
            x_prev=parent[:, 4:5],
            x_curr=children_t[:, :, 4:5],
            hatcf_prev=parent[:, 5:6],
            lnkf_prev=parent[:, 6:7],
            return_physical=True
        )
        
        # 提取父节点状态并计算 w
        c_parent = parent[:, 5:6]
        k_parent = parent[:, 6:7]
        
        # 计算损失
        # 构造残差并按 parent 聚合
        residuals = loss_fn.compute_euler_residuals(
            w_parent.squeeze(-1), w_children.squeeze(-1),
            k_parent.squeeze(-1), k_children.squeeze(-1),
            c_parent.squeeze(-1), c_children.squeeze(-1)
        )  # (batch, n_children)
        combined = residuals.prod(dim=-1)  # (batch,)
        main_loss = torch.log1p(combined.abs()).mean()
        # if main_loss has nan, it may be due to extremely large residuals. We can log the max residual and describe of residuals,w_parent,w_children, k_parent,k_children, c_parent, c_children   for debugging.

            

        moment_loss = torch.tensor(0.0, device=self.device)
        M_use = M.squeeze(-1) if M.dim() == 3 else M
        if M_use.dim() == 1:
            M_use = M_use.unsqueeze(-1)
        for j in range(M_use.shape[1]):
            L1, L2 = moment_penalty(M_use[:, j], loss_fn.mu_lo, loss_fn.mu_hi, loss_fn.var_hi)
            moment_loss = moment_loss + L1 + L2

        # 可选：FC1 输出与真实 hatcf / lnkf 的重建误差
        recon_weight = getattr(self.hyperparams, "fc1_recon_weight", 0.0)
        recon_loss = torch.tensor(0.0, device=self.device)
        if self.add_FC1loss:
            hatcf_pred = c_children  # (batch, 2, 1)
            lnkf_pred = k_children   # (batch, 2, 1)
            hatcf_true = children_t[:, :, 7:8]
            lnkf_true = children_t[:, :, 8:9]
            recon_loss = ((hatcf_pred - hatcf_true).pow(2) + (lnkf_pred - lnkf_true).pow(2)).mean()
        # if loss is nan or inf, then we log the recon_loss, hatcf_pred, hatcf_true, lnkf_pred, lnkf_true for debugging.
        
        if torch.isnan(recon_loss) or torch.isinf(recon_loss) or torch.isnan(main_loss) or torch.isinf(main_loss):
            max_recon_loss = recon_loss.abs().max().item()
            logger.warning(f"NaN in SDF loss, max recon loss: {max_recon_loss}")
            logger.warning(f"recon Residuals: {recon_loss}")
            max_residual = residuals.abs().max().item()
            logger.warning(f"NaN in SDF loss, max residual: {max_residual}")
            logger.warning(f"Residuals: {residuals}")
            logger.warning(f"w_parent: {w_parent}, w_children: {w_children}")
            logger.warning(f"k_parent: {k_parent}, k_children: {k_children}")
            logger.warning(f"c_parent: {c_parent}, c_children: {c_children}")


        return main_loss + moment_loss + recon_weight * recon_loss
    
    def _compute_p0_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算 P0 损失（支持任意 N 分支）
        """
        model = self.models['policy_value']
        loss_fn = self.loss_fns['p0']
        
        parent = batch['parent']
        children = batch.get('children', [])
        strip_extra = lambda x: x[:, :7] if x.shape[1] > 7 else x
        
        # 兼容旧的 child0/child1 格式
        if not children:
            child0 = batch.get('child0')
            child1 = batch.get('child1')
            if child0 is not None and child1 is not None:
                children = [child0, child1]
        
        if not children:
            raise ValueError("No children data in batch")
        
        # 获取 SDF（优先用 batch 内的 M，避免重复计算）
        if parent.shape[1] > 7:
            M_list = [child[:, 7:8] for child in children]
        else:
            M_list = [torch.ones(parent.shape[0], 1, device=self.device) for _ in children]
        
        # 前向传播
        parent_state = strip_extra(parent)
        output_t = model(parent_state)

        def _get_out(out, name: str, idx: int) -> torch.Tensor:
            if isinstance(out, dict):
                return out[name]
            if hasattr(out, name):
                return getattr(out, name)
            return out[:, idx:idx + 1]

        bp0_t = _get_out(output_t, 'bp0', 1)
        bpI_t = _get_out(output_t, 'bpI', 2)
        bar_i_t = _get_out(output_t, 'bar_i', 4)
        bp_t = _get_out(output_t, 'bp', -1)
        if bp_t.shape != bp0_t.shape:
            bp_t = bar_i_t * bpI_t + (1 - bar_i_t) * bp0_t
        b_parent = parent_state[:, 0:1]

        output_children = []

        for child in children:
            child_state_raw = strip_extra(child)
            eta_child = child[:, 2:3]
            child_state = child_state_raw.clone()
            child_state[:, 0:1] = eta_child * bp_t + (1 - eta_child) * b_parent
            output_children.append(model(child_state))
            
        childp0_state = parent_state.clone()
        childp0_state[:, 0:1] = bpI_t
        outputp0_children = model(childp0_state)
        
        # 提取 P0 和所需变量
        P0 = _get_out(output_t, 'P0', 3)
        P_children = [_get_out(out, 'P', 7) for out in output_children]
        bar_z_children = [_get_out(out, 'bar_z', 6) for out in output_children]
        
        # Q 值
        Q = _get_out(output_t, 'Q', 0)
        Qp = _get_out(outputp0_children, 'Q', 0)

        
        # 计算现金流与残差（逐 parent × child 对齐）
        CF0p = loss_fn.compute_cashflow_p0(
            parent_state[:, 4:5],  # x
            parent_state[:, 1:2],  # z
            parent_state[:, 0:1],  # b
            Q, Qp,
            parent_state[:, 2:3]   # eta
        )
        residuals = loss_fn.compute_bellman_residual(
            P0, CF0p, M_list, P_children, bar_z_children
        )
        residuals_stack = torch.stack(residuals, dim=-1)
        combined = residuals_stack.prod(dim=-1)
        main_loss = combined.abs().mean()

        penalty_z = compute_z_penalty(
            combined.abs(), parent_state[:, 1:2],
            loss_fn.alpha_z, loss_fn.beta_z, loss_fn.z0
        )

        total_loss = main_loss + penalty_z
        return total_loss
    
    def _compute_pi_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算 PI 损失（支持任意 N 分支）
        """
        model = self.models['policy_value']
        loss_fn = self.loss_fns['pi']
        
        parent = batch['parent']
        children = batch.get('children', [])
        strip_extra = lambda x: x[:, :7] if x.shape[1] > 7 else x
        
        # 兼容旧的 child0/child1 格式
        if not children:
            child0 = batch.get('child0')
            child1 = batch.get('child1')
            if child0 is not None and child1 is not None:
                children = [child0, child1]
        
        if not children:
            raise ValueError("No children data in batch")
        
        # 获取 SDF（优先用 batch 内的 M，避免重复计算）
        if parent.shape[1] > 7:
            M_list = [child[:, 7:8] for child in children]
        else:
            M_list = [torch.ones(parent.shape[0], 1, device=self.device) for _ in children]
        
        # 前向传播
        parent_state = strip_extra(parent)
        output_t = model(parent_state)

        def _get_out(out, name: str, idx: int) -> torch.Tensor:
            if isinstance(out, dict):
                return out[name]
            if hasattr(out, name):
                return getattr(out, name)
            return out[:, idx:idx + 1]


        bp0_t = _get_out(output_t, 'bp0', 1)
        bpI_t = _get_out(output_t, 'bpI', 2)
        bar_i_t = _get_out(output_t, 'bar_i', 4)
        bp_t = _get_out(output_t, 'bp', -1)
        if bp_t.shape != bp0_t.shape:
            bp_t = bar_i_t * bpI_t + (1 - bar_i_t) * bp0_t
        b_parent = parent_state[:, 0:1]

        output_children = []

        for child in children:
            child_state_raw = strip_extra(child)
            eta_child = child[:, 2:3]
            child_state = child_state_raw.clone()
            child_state[:, 0:1] = eta_child * bp_t + (1 - eta_child) * b_parent
            output_children.append(model(child_state))
            
        childpI_state = parent_state.clone()
        childpI_state[:, 0:1] = bpI_t
        outputpI_children = model(childpI_state)
        
        # 提取 PI 和所需变量
        Q = _get_out(output_t, 'Q', 0)
        PI = _get_out(output_t, 'PI', 4)
        P_children = [_get_out(out, 'P', 7) for out in output_children]
        bar_z_children = [_get_out(out, 'bar_z', 6) for out in output_children]
        QpI = _get_out(outputpI_children, 'Q', 0)
        
        # 提取 z 和 b
        z = parent[:, 1:2]
        b = parent[:, 0:1]
        
        # CFip (现金流）
        CFip = output_t.get('CFip', torch.zeros_like(PI)) if isinstance(output_t, dict) else torch.zeros_like(PI)
        
        # 计算现金流与残差（逐 parent × child 对齐）
        CFip = loss_fn.compute_cashflow_pi(
            parent_state[:, 4:5],  # x
            parent_state[:, 1:2],  # z
            parent_state[:, 0:1],  # b
            parent_state[:, 3:4],  # i
            Q, QpI,
            parent_state[:, 2:3]   # eta
        )
        residuals = loss_fn.compute_bellman_residual(
            PI, CFip, M_list, P_children, bar_z_children
        )  # List[(batch,1)]
        residuals_stack = torch.stack(residuals, dim=-1)
        combined = residuals_stack.prod(dim=-1)
        main_loss = combined.abs().mean()

        penalty_z = compute_z_penalty(
            combined.abs(), parent_state[:, 1:2],
            loss_fn.alpha_z, loss_fn.beta_z, loss_fn.z0
        )
        penalty_b = loss_fn.b_penalty_weight * loss_fn.compute_b_penalty(PI, parent_state[:, 0:1]).mean()

        total_loss = main_loss + penalty_z + penalty_b
        return total_loss
    
    def _compute_q_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算 Q 损失（支持任意 N 分支）
        """
        model = self.models['policy_value']
        loss_fn = self.loss_fns['q']
        
        parent = batch['parent']
        children = batch.get('children', [])
        strip_extra = lambda x: x[:, :7] if x.shape[1] > 7 else x
        
        # 兼容旧的 child0/child1 格式
        if not children:
            child0 = batch.get('child0')
            child1 = batch.get('child1')
            if child0 is not None and child1 is not None:
                children = [child0, child1]
        
        if not children:
            raise ValueError("No children data in batch")
        
        # 获取 SDF（优先用 batch 内的 M，避免重复计算）
        if parent.shape[1] > 7:
            M_list = [child[:, 7:8] for child in children]
        else:
            M_list = [torch.ones(parent.shape[0], 1, device=self.device) for _ in children]
        
        # 前向传播
        parent_state = strip_extra(parent)
        output_t = model(parent_state)

        def _get_out(out, name: str, idx: int) -> torch.Tensor:
            if isinstance(out, dict):
                return out[name]
            if hasattr(out, name):
                return getattr(out, name)
            return out[:, idx:idx + 1]


        bp0_t = _get_out(output_t, 'bp0', 1)
        bpI_t = _get_out(output_t, 'bpI', 2)
        bar_i_t = _get_out(output_t, 'bar_i', 4)
        bp_t = _get_out(output_t, 'bp', -1)
        if bp_t.shape != bp0_t.shape:
            bp_t = bar_i_t * bpI_t + (1 - bar_i_t) * bp0_t
        b_parent = parent_state[:, 0:1]

        output_children = []
        outputsp_children = []
        for child in children:
            child_state_raw = strip_extra(child)
            eta_child = child[:, 2:3]
            child_state = child_state_raw.clone()
            child_state[:, 0:1] = eta_child * bp_t + (1 - eta_child) * b_parent
            output_children.append(model(child_state))
            
            childsp_state = child_state_raw.clone()
            childsp_state[:, 0:1] = b_parent
            outputsp_children.append(model(childsp_state))
            
            
        
        # 提取 Q 和所需变量
        Q = _get_out(output_t, 'Q', 0)

        Qsp_children = [_get_out(out, 'Q', 0) for out in outputsp_children]
        bar_zsp_children = [_get_out(out, 'bar_z', 6) for out in outputsp_children]
        x_children = [child[:, 4:5] for child in children]
        z_children = [child[:, 1:2] for child in children]
        
        # 提取父节点状态
        bar_i = _get_out(output_t, 'bar_i', 5)
        bar_z = _get_out(output_t, 'bar_z', 6)

        # 构造分支残差（不在 q_loss 内部做均值）并在循环外组合
        b_parent = parent_state[:, 0:1]
        z_parent = parent[:, 1:2]
        x_parent = parent[:, 4:5]

        residuals = loss_fn.compute_main_residual(
            Q, b_parent, bar_i, M_list, Qsp_children,
            bar_zsp_children, x_children, z_children
        )  # List[(batch,1)]
        residuals_stack = torch.stack(residuals, dim=-1)  # (batch,1,n_children)
        combined = residuals_stack.prod(dim=-1)  # (batch,1)
        main_loss = combined.abs().mean()

        # 其余约束在 episode 层做聚合
        loss3 = loss_fn.compute_bar_z_constraint(Q, x_parent, z_parent, bar_z).mean()
        loss4 = loss_fn.compute_boundary_loss_low(Q, b_parent).mean()
        loss5 = loss_fn.compute_boundary_loss_high(Q, b_parent, x_parent, z_parent).mean()
        penalty_z_main = compute_z_penalty(
            combined.abs(), z_parent,
            loss_fn.alpha_z, loss_fn.beta_z, loss_fn.z0
        )
        penalty_z_loss3 = compute_z_penalty(
            (Q - loss_fn.compute_recovery_value(x_parent, z_parent)).pow(2) * bar_z,
            z_parent, loss_fn.alpha_z, loss_fn.beta_z, loss_fn.z0
        )

        total_loss = main_loss + loss3 + loss4 + loss5 + penalty_z_main + penalty_z_loss3
        return total_loss
    
    def _compute_fc2_loss(self, batch) -> torch.Tensor:
        """
        计算 FC2 损失（使用 FC2train2.ipynb pipeline，batch 为 DataFrame）
        """
        model = self.models.get('fc2')
        if model is None:
            return torch.tensor(0.0, device=self.device)

        if 'policy_value' not in self.models or self.models['policy_value'] is None:
            return torch.tensor(0.0, device=self.device)

        pv_model = self.models['policy_value']

        if isinstance(batch, pd.DataFrame):
            df = batch
            full_N = 1000
            entry_num = None
        elif isinstance(batch, dict) and 'df' in batch:
            df = batch['df']
            full_N = batch.get('full_N', 1000)
            entry_num = batch.get('entry_num', None)
        else:
            return torch.tensor(0.0, device=self.device)
        if df['branch'].min() < 0:
            df['branch'] = df['branch'] + 1

        pipe = FC2LossPipe(
            df=df,
            full_N=full_N,
            entry_num=entry_num,
            device=self.device,
        )
        # keep for inspection/debugging
        self._last_fc2_pipe = pipe
        fc2_loss = pipe.loss(model, pv_model)
        if isinstance(fc2_loss, tuple):
            fc2_loss = fc2_loss[0]
        return fc2_loss
    
    def create_batches(
        self,
        batch_size: int = 1024,
        n_branches: int = 2
    ) -> List[Dict[str, torch.Tensor]]:
        """
        从 DataFrame 创建训练批次（支持任意 N 分支）
        
        Args:
            batch_size: 批大小
            n_branches: 分支数量（默认2）
        
        数据组织：每 (1 + n_branches) 行为一组
            - 第 0 行: parent
            - 第 1 ~ n_branches 行: children
        """
        if self.df is None:
            raise RuntimeError("先调用 generate_data()")
        
        # 转换为张量
        input_cols = ['b', 'z', 'ETA', 'i', 'x', 'Hatcf', 'LnKF']
        if 'M' in df.columns:
            input_cols.append('M')
        else:
            df = df.copy()
            df['M'] = 1.0
            input_cols.append('M')
        if 'M' in df.columns:
            input_cols.append('M')
        else:
            df = df.copy()
            df['M'] = 1.0
            input_cols.append('M')
        X = torch.tensor(
            self.df[input_cols].values,
            device=self.device,
            dtype=torch.float32
        )
        
        # 按 (1 + n_branches) 拆分
        group_size = 1 + n_branches
        n = len(X)
        n_units = n // group_size
        
        # 提取 parent 和 children
        parent = X[0::group_size]
        children = []
        for i in range(1, group_size):
            children.append(X[i::group_size])
        
        # 确保长度一致
        min_len = min(len(parent), min(len(c) for c in children))
        parent = parent[:min_len]
        children = [c[:min_len] for c in children]
        
        n_units = len(parent)
        n_batches = (n_units + batch_size - 1) // batch_size
        
        batches = []
        indices = torch.randperm(n_units)
        
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_units)
            idx = indices[start:end]
            
            batch = {
                'parent': parent[idx],
                'children': [c[idx] for c in children],
                # 保持兼容性
                'child0': children[0][idx] if len(children) > 0 else None,
                'child1': children[1][idx] if len(children) > 1 else None
            }
            batches.append(batch)
        
        return batches

    def _create_sdf_batches_from_macro_df(
        self,
        df_sdf: pd.DataFrame,
        batch_size: int = 1024,
        n_branches: int = 2
    ) -> List[Dict[str, torch.Tensor]]:
        """
        从宏观跨期 DataFrame 创建 SDF 训练批次
        
        df_sdf 列要求：x_t, x_t1, Hatcf_t, LnKF_t, path, branch
        """
        parent_rows = []
        children_rows = [[] for _ in range(n_branches)]
        # if self.add_FC1loss, groupby ['path','t'], else groupby 'path' only
        if self.add_FC1loss :
            
            group_keys = ['path', 't']
            if  self.train_mode != '2time':
                df_sdf = df_sdf[df_sdf['t'] > 2].copy().reset_index(drop=True) # for FC1 loss, we need to ensure that we have enough history to compute the loss, so we filter out t<=2
                logger.info(f"After filtering for FC1 loss, {len(df_sdf)} rows remain in SDF DataFrame")
                
        else:   
            group_keys = ['path']
            logger.info(f"For SDF loss, {len(df_sdf)} rows remain in SDF DataFrame")
        for _, group in df_sdf.groupby(group_keys):
            group = group.sort_values('branch')
            if len(group) < n_branches:
                continue
            x_t = group['x_t'].iloc[0]
            hatcf_t = group['Hatcf_t'].iloc[0]
            lnkf_t = group['LnKF_t'].iloc[0]
            
            if self.add_FC1loss:
                hatc_t = group['Hatc_t'].loc[group.branch == 0].iloc[0]
                lnk_t = group['LnK_t'].loc[group.branch == 0].iloc[0]
            
            parent_rows.append([0.0, 0.0, 0.0, 0.0, x_t, hatcf_t, lnkf_t, hatc_t, lnk_t] if self.add_FC1loss else [0.0, 0.0, 0.0, 0.0, x_t, hatcf_t, lnkf_t])
            for k in range(n_branches):
                x_t1 = group['x_t1'].loc[group.branch == k].iloc[0]
                if self.add_FC1loss:
                    hatc_t1 = group['Hatc_t1'].loc[group.branch == k].iloc[0]
                    lnk_t1 = group['LnK_t1'].loc[group.branch == k].iloc[0]
                children_rows[k].append([0.0, 0.0, 0.0, 0.0, x_t1,0.0, 0.0, hatc_t1, lnk_t1] if self.add_FC1loss else [0.0, 0.0, 0.0, 0.0, x_t1, 0.0, 0.0])
        
        if not parent_rows:
            return []
        
        parent = torch.tensor(parent_rows, device=self.device, dtype=torch.float32)
        children = [
            torch.tensor(rows, device=self.device, dtype=torch.float32)
            for rows in children_rows
        ]
        
        n_units = len(parent)
        n_batches = (n_units + batch_size - 1) // batch_size
        indices = torch.randperm(n_units)
        batches = []
        for i in range(n_batches): 
            start = i * batch_size
            end = min((i + 1) * batch_size, n_units)
            idx = indices[start:end]
            batch = {
                'parent': parent[idx],
                'children': [c[idx] for c in children],
                'child0': children[0][idx] if len(children) > 0 else None,
                'child1': children[1][idx] if len(children) > 1 else None
            }
            batches.append(batch)
        logger.info(f"Created {len(batches)} SDF batches from macro DataFrame with {n_units} units")
        return batches

    def _create_firm_batches_from_df(
        self,
        df: pd.DataFrame,
        batch_size: int = 1024,
        n_branches: int = 2
    ) -> List[Dict[str, torch.Tensor]]:
        """
        从 firm-level DataFrame 创建训练批次（兼容 sample 与 simulateTS）
        """
        input_cols = ['b', 'z', 'ETA', 'i', 'x', 'Hatcf', 'LnKF']
        if 'M' in df.columns:
            input_cols.append('M')
        if 't' not in df.columns or 'branch' not in df.columns:
            raise ValueError("DataFrame missing required columns: 't' and 'branch'")
        
        t_is_str = df['t'].dtype == object
        if t_is_str:
            parent_df = df[df['t'] == 't'].copy()
            child_dfs = [
                df[df['t'] == f't+1_{k}'].copy() for k in range(n_branches)
            ]
            
            parent_df = parent_df.set_index(['path', 'ID'])
            child_dfs = [c.set_index(['path', 'ID']) for c in child_dfs]
            
            common_index = parent_df.index
            for child_df in child_dfs:
                common_index = common_index.intersection(child_df.index)
            
            parent_df = parent_df.loc[common_index]
            child_dfs = [c.loc[common_index] for c in child_dfs]
        else:
            parent_df = df[df['branch'] == -1].copy()
            child_df = df[df['branch'] >= 0].copy()
            child_index = child_df.set_index(['path', 'ID', 't', 'branch'])
            
            parent_rows = []
            child_rows = [[] for _ in range(n_branches)]
            for _, row in parent_df.iterrows():
                t_next = row['t'] + 1
                rows_for_parent = []
                for k in range(n_branches):
                    key = (row['path'], row['ID'], t_next, k)
                    if key not in child_index.index:
                        rows_for_parent = []
                        break
                    hit = child_index.loc[key]
                    if isinstance(hit, pd.DataFrame):
                        hit = hit.iloc[0]
                    hit_dict = hit.to_dict()
                    hit_dict['path'] = row['path']
                    hit_dict['ID'] = row['ID']
                    hit_dict['t'] = t_next
                    hit_dict['branch'] = k
                    rows_for_parent.append(hit_dict)
                if not rows_for_parent:
                    continue
                parent_rows.append(row)
                for k, hit in enumerate(rows_for_parent):
                    child_rows[k].append(hit)
            
            if not parent_rows:
                return []
            
            parent_df = pd.DataFrame(parent_rows).set_index(['path', 'ID'])
            child_dfs = [pd.DataFrame(rows).set_index(['path', 'ID']) for rows in child_rows]
        
        parent = torch.tensor(parent_df[input_cols].values, device=self.device, dtype=torch.float32)
        children = [
            torch.tensor(c[input_cols].values, device=self.device, dtype=torch.float32)
            for c in child_dfs
        ]
        
        n_units = len(parent)
        n_batches = (n_units + batch_size - 1) // batch_size
        indices = torch.arange(n_units)
        
        batches = []
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_units)
            idx = indices[start:end]
            batch = {
                'parent': parent[idx],
                'children': [c[idx] for c in children],
                'child0': children[0][idx] if len(children) > 0 else None,
                'child1': children[1][idx] if len(children) > 1 else None
            }
            batches.append(batch)
        
        return batches

    def _create_fc2_batches(
        self,
        df_firm: pd.DataFrame,
        batch_size: int = 1024,
        quantile_num: int = 100,
        n_branches: int = 2
    ) -> List[pd.DataFrame]:
        """
        从 firm-level DataFrame 创建 FC2 训练批次（每个 batch 是一个 DataFrame）
        batch_size 解释为每个 batch 的 path 数量。
        """
        if df_firm is None or df_firm.empty:
            return []
        if 'path' not in df_firm.columns:
            return [df_firm]

        paths = sorted(df_firm['path'].unique())
        if batch_size is None or batch_size <= 0:
            batch_size = len(paths)

        batches = []
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            batches.append(df_firm[df_firm['path'].isin(batch_paths)].copy())
        return batches

    def _run_batches(
        self,
        batches: List[Dict[str, torch.Tensor]],
        n_epochs: int,
        log_interval: int,
        train_modules: List[str],
        desc_prefix: str = ''
    ) -> Dict:
        """
        使用预生成的 batches 执行训练循环
        """
        for epoch in range(n_epochs):
            epoch_losses = []
            for batch in tqdm(batches, desc=f"{desc_prefix}Epoch {epoch+1}/{n_epochs}"):
                losses = self.train_step(batch, train_modules)
                epoch_losses.append(losses)
                
                if self.step_count % log_interval == 0:
                    avg_loss = np.mean([l['total'] for l in epoch_losses[-log_interval:]])
                    lr = None
                    for name in train_modules:
                        lr = self.lr_schedulers.get(name)
                        if lr is not None:
                            break
                    current_lr = lr.get_lr() if lr else 0
                    
                    logger.info(
                        f"Step {self.step_count}: "
                        f"loss={avg_loss:.6f}, lr={current_lr:.2e}"
                    )
            
            avg_losses = {
                k: np.mean([l[k] for l in epoch_losses if k in l])
                for k in epoch_losses[0].keys()
            }
            logger.info(f"{desc_prefix}Epoch {epoch+1} finished: {avg_losses}")
        
        return {
            'final_losses': avg_losses
        }

    def run_episode(
        self,
        n_epochs: int = 10,
        batch_size: int = 1024,
        log_interval: int = 100,
        n_samples: int = 10000,
        n_paths: int = 100,
        group_size: int = 100,
        n_branches: int = 2,
        train_modules: Optional[List[str]] = None,
        simulate_kwargs: Optional[Dict] = None,
        train_mode : str = '2time'
    ) -> Dict:
        """
        按 Episode 逻辑执行训练（episode 0 用 Sample，后续用 SimulateTS）
        """
        simulate_kwargs = simulate_kwargs or {}
        train_modules = train_modules or ['sdf_fc1', 'policy_value', 'fc2']
        
        module_summaries = {}
        # 每 change_circle 次 episode 切换一次训练模块，比如前 change_circle 个 episode
        
        self.train_mode = train_mode
        
        
        if  train_mode == '2time' :
            logger.info(f"Running paths {n_paths} with Sample for data generation and training")
            use_sdf_fc1 = 'sdf_fc1' in train_modules and 'sdf_fc1' in self.models
            use_policy_value = 'policy_value' in train_modules and 'policy_value' in self.models
            if self.episode_id == 0:
                self.add_FC1loss = False
            else: 
                self.add_FC1loss = True
            if use_sdf_fc1 or use_policy_value:
                sampler = Sample(
                    models=self.models,
                    config=self.config,
                    n_samples=None,
                    n_paths=n_paths,
                    group_size=group_size,
                    branch_num=n_branches
                )

            if self.episode_id  == 0 and use_sdf_fc1:
                self.df_sdf = sampler.build_sdf_fc1_df()

                sdf_batches = self._create_sdf_batches_from_macro_df(
                    self.df_sdf, batch_size=batch_size, n_branches=n_branches
                )
                if sdf_batches:
                    module_summaries['sdf_fc1'] = self._run_batches(
                        sdf_batches, n_epochs, log_interval, ['sdf_fc1'], desc_prefix='SDF/FC1 '
                    )
            
            if use_policy_value:
                self.df = sampler.build_policy_value_df()
                self.df.to_csv(f'simulated_firm_data_episode_{self.episode_id}.csv', index=False)
                pv_batches = self._create_firm_batches_from_df(
                    self.df, batch_size=batch_size, n_branches=n_branches
                )
                if pv_batches:
                    module_summaries['policy_value'] = self._run_batches(
                        pv_batches, n_epochs, log_interval, ['policy_value'], desc_prefix='Policy/Value '
                    )
            
            if 'fc2' in train_modules and 'fc2' in self.models:
                simulate_kwargs['horizon'] = 1
                simulator = SimulateTS(
                models=self.models,
                config=self.config,
                n_paths=n_paths,
                group_size=group_size,
                branch_num=n_branches,
                **simulate_kwargs
                )
                self.df, self.df_macro = simulator.simulate()
                # epoch_losses = []
                # for epoch in tqdm(range(n_epochs), desc='FC2 Epochs'):
                #     losses = self.train_step(self.df, ['fc2'])
                #     epoch_losses.append(losses)
                #     if self.step_count % log_interval == 0:
                #         avg_loss = np.mean([l['total'] for l in epoch_losses[-log_interval:]])
                #         lr = self.lr_schedulers.get('fc2')
                #         current_lr = lr.get_lr() if lr else 0
                #         logger.info(
                #             f"FC2 Step {self.step_count}: "
                #             f"loss={avg_loss:.6f}, lr={current_lr:.2e}"
                #         )
                # if epoch_losses:
                #     avg_losses = {
                #         k: np.mean([l[k] for l in epoch_losses if k in l])
                #         for k in epoch_losses[0].keys()
                #     }
                #     logger.info(f"FC2 Epochs finished: {avg_losses}")
                #     module_summaries['fc2'] = {'final_losses': avg_losses}
                

                df_macro_sdf = build_sdf_pairs_from_macro_ts(self.df_macro.copy(), include_hatc_lnk_t1=True)
                
                self.add_FC1loss = True
                sdf_batches = self._create_sdf_batches_from_macro_df(
                    df_macro_sdf, batch_size=batch_size, n_branches=n_branches
                )
                # 可以删除的logger.info
                if len(sdf_batches) == 0:
                    logger.warning("No SDF batches created from macro DataFrame. Check if df_macro_sdf is empty or if filtering conditions are too strict.")
                    logger.info(f"SDF data {df_macro_sdf}")
                if sdf_batches:
                   
                    module_summaries['sdf_fc1'] = self._run_batches(
                        sdf_batches, n_epochs, log_interval, ['sdf_fc1'], desc_prefix='SDF/FC1 After FC2'
                    )
                self.add_FC1loss = False
        else:
            
            
            simulator = SimulateTS(
                models=self.models,
                config=self.config,
                n_paths=n_paths,
                group_size=group_size,
                branch_num=n_branches,
                **simulate_kwargs
            )
            self.df, self.df_macro = simulator.simulate()

            print("Running episode with SimulateTS for data generation and training")
            if 'policy_value' in train_modules and 'policy_value' in self.models:
                pv_batches = self._create_firm_batches_from_df(
                    self.df, batch_size=batch_size, n_branches=n_branches
                )
                if pv_batches:
                    module_summaries['policy_value'] = self._run_batches(
                        pv_batches, n_epochs, log_interval, ['policy_value'], desc_prefix='Policy/Value '
                    )
   
            if 'fc2' in train_modules and 'fc2' in self.models and self.df_macro is not None:
                # epoch_losses = []
                # for epoch in range(n_epochs):
                #     losses = self.train_step(self.df, ['fc2'])
                #     epoch_losses.append(losses)
                #     if self.step_count % log_interval == 0:
                #         avg_loss = np.mean([l['total'] for l in epoch_losses[-log_interval:]])
                #         lr = self.lr_schedulers.get('fc2')
                #         current_lr = lr.get_lr() if lr else 0
                #         logger.info(
                #             f"FC2 Step {self.step_count}: "
                #             f"loss={avg_loss:.6f}, lr={current_lr:.2e}"
                #         )
                # if epoch_losses:
                #     avg_losses = {
                #         k: np.mean([l[k] for l in epoch_losses if k in l])
                #         for k in epoch_losses[0].keys()
                #     }
                #     logger.info(f"FC2 Epochs finished: {avg_losses}")
                #     module_summaries['fc2'] = {'final_losses': avg_losses}
                df_macro_sdf = build_sdf_pairs_from_macro_ts(self.df_macro.copy(), include_hatc_lnk_t1=True)
                
                self.add_FC1loss = True

                sdf_batches = self._create_sdf_batches_from_macro_df(
                    df_macro_sdf, batch_size=batch_size, n_branches=n_branches
                )
                if sdf_batches:
                    
                    module_summaries['sdf_fc1'] = self._run_batches(
                        sdf_batches, n_epochs, log_interval, ['sdf_fc1'], desc_prefix='SDF/FC1 After FC2'
                    )
                    logger.info("Finished SDF/FC1 training after FC2")
                self.add_FC1loss = False
            
        
        summary = {
            'episode_id': self.episode_id,
            'total_steps': self.step_count,
            'module_summaries': module_summaries,
            'loss_history': self.loss_history
        }
        
        return summary
    
    def run(
        self,
        n_epochs: int = 10,
        batch_size: int = 1024,
        log_interval: int = 100,
        train_modules: List[str] = None
    ) -> Dict:
        """
        运行完整的 Episode 训练
        
        Args:
            n_epochs: 轮数
            batch_size: 批大小
            log_interval: 日志间隔
            train_modules: 要训练的模块
        
        Returns:
            summary: 训练摘要
        """

        logger.info(f"Episode {self.episode_id}: Starting training")
        
        for epoch in range(n_epochs):
            batches = self.create_batches(batch_size)
            
            epoch_losses = []
            for batch in tqdm(batches, desc=f"Epoch {epoch+1}/{n_epochs}"):
                losses = self.train_step(batch, train_modules)
                epoch_losses.append(losses)
                
                if self.step_count % log_interval == 0:
                    avg_loss = np.mean([l['total'] for l in epoch_losses[-log_interval:]])
                    lr = self.lr_schedulers.get('sdf_fc1', self.lr_schedulers.get('policy_value'))
                    current_lr = lr.get_lr() if lr else 0
                    
                    logger.info(
                        f"Step {self.step_count}: "
                        f"loss={avg_loss:.6f}, lr={current_lr:.2e}"
                    )
            
            # Epoch 结束统计
            avg_losses = {
                k: np.mean([l[k] for l in epoch_losses if k in l])
                for k in epoch_losses[0].keys()
            }
            logger.info(f"Epoch {epoch+1} finished: {avg_losses}")
        
        summary = {
            'episode_id': self.episode_id,
            'total_steps': self.step_count,
            'final_losses': avg_losses,
            'loss_history': self.loss_history
        }
        
        return summary
    
    def get_metrics(self) -> Dict:
        """
        获取训练指标
        """
        metrics = {}
        
        for k, v in self.loss_history.items():
            if len(v) > 0:
                metrics[f'{k}_mean'] = np.mean(v)
                metrics[f'{k}_std'] = np.std(v)
                metrics[f'{k}_min'] = np.min(v)
                metrics[f'{k}_max'] = np.max(v)
                metrics[f'{k}_last'] = v[-1]
        
        return metrics
