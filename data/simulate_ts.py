"""
SimulateTS 类：树状分支模拟器

用训练好的模型生成模拟数据，支持：
- 树状分支结构 t → (t+1, t+1') → (t+2, t+2') → ...
- 公司进入/退出
- 宏观聚合
- 资源核算
"""

import torch
import pandas as pd
import numpy as np
import uuid
from typing import Dict, Optional, Tuple, List
from tqdm import tqdm

import sys
sys.path.append('..')
from config import Config, SIMMODEL

from .data_utils import (
    sample_ar1,
    sample_stationary_ar1,
    sample_uniform,
    sample_bernoulli,
    compute_quantile_features
)


class SimulateTS:
    """
    树状分支模拟器
    
    在已有训练好的模型后，生成用于 episode 循环的模拟数据
    """
    
    def __init__(
        self,
        models: Dict,
        config: type = Config,
        n_paths: int = 100,
        group_size: int = 200,
        horizon: int = 20,
        branch_num: int = 2,
        main_branch: int = 0,
        enable_entry: bool = True,
        enable_exit: bool = True,
        device: torch.device = None
    ):
        """
        Args:
            models: 模型字典，必须包含：
                - 'policy_value': PolicyValueModel
                - 'sdf_fc1': SDFFC1Combined
                可选：
                - 'fc2': FC2Model
                - 'dist_b': b 分布生成器（可选）
            config: 配置类
            n_paths: path 数量
            group_size: 每条 path 的初始公司数
            horizon: 模拟时间长度 T
            branch_num: 分支数
            main_branch: 主分支索引（用于向前推进）
            enable_entry: 是否启用公司进入
            enable_exit: 是否启用公司退出
            device: 设备
        """
        self.models = models
        self.config = config
        self.n_paths = n_paths
        self.group_size = group_size
        self.horizon = horizon
        self.branch_num = branch_num
        self.main_branch = main_branch
        self.enable_entry = enable_entry
        self.enable_exit = enable_exit
        self.device = device or config.DEVICE
        
        # 设置模型为 eval 模式
        self._set_models_eval()
        
        # 经济参数
        self.delta = config.DELTA
        self.phi = config.PHI
        self.g = config.G
    
    def _set_models_eval(self):
        """将所有模型设置为 eval 模式"""
        for name, model in self.models.items():
            if model is not None:
                model.eval()
    
    def simulate(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        运行完整模拟
        
        Returns:
            df_firm: 公司面板数据
            df_macro: 宏观面板数据
        """
        all_firm_data = []
        all_macro_data = []
        
        for path_idx in tqdm(range(self.n_paths), desc="Simulating paths"):
            firm_data, macro_data = self._simulate_path(path_idx)
            all_firm_data.extend(firm_data)
            all_macro_data.extend(macro_data)
        
        df_firm = pd.DataFrame(all_firm_data)
        df_macro = pd.DataFrame(all_macro_data)
        print(f"Simulation completed: {len(df_firm)} firm records, {len(df_macro)} macro records")
        return df_firm, df_macro
    
    def _simulate_path(self, path_idx: int) -> Tuple[List, List]:
        """
        模拟单条 path 的树状结构
        
        数据结构：
        - t 期数据：branch_k = -1 (父节点)
        - t+1 期数据：branch_k = 0, 1, ..., N-1 (各分支)
        """
        firm_data = []
        macro_data = []
        
        # 初始化 t=0 的状态
        state = self._initialize_path(path_idx)
        
        # 树状推进tqdm(range(self.horizon), desc= f"Simulating path {}".format(path_idx)): ")
        for t in tqdm(range(self.horizon), desc= f"Simulating path {path_idx}: ", leave=False):
            # 当前节点的处理 (t 期父节点，branch_k = -1)
            node_firm, node_macro = self._process_node(state, path_idx, t, branch_k=-1)
            firm_data.extend(node_firm)
            macro_data.append(node_macro)
            
            # 扩展到下一期的多个分支
            branch_states = self._expand_branches(state, t)
            
            # 为每个分支记录 t+1 期的数据（分支内独立进入）
            for branch_k, branch_state in enumerate(branch_states):
                # 进入发生在每个分支内，避免不同分支共享进入者
                if self.enable_entry:
                    branch_state = self._apply_entry(branch_state, path_idx, t + 1)
                    branch_states[branch_k] = branch_state

                # 处理分支节点 (t+1 期，branch_k = 0, 1, ...)
                branch_firm, branch_macro = self._process_node(
                    branch_state, path_idx, t + 1, branch_k=branch_k
                )
                firm_data.extend(branch_firm)
                macro_data.append(branch_macro)
            
            # 选择主分支作为下一期的 parent
            state = branch_states[self.main_branch]
            
            # 处理退出
            if self.enable_exit:
                state = self._apply_exit(state)
            
            # 进入已在分支内完成，这里不再重复
        
        return firm_data, macro_data
    
    def _initialize_path(self, path_idx: int) -> Dict:
        """
        初始化 path 的状态
        including: 
                Macro state:
                    - x from stationary AR(1), 
                    - lnkf from N(4,1),
                    - hatcf from N(-2,1),
                    - M = 1 (初始无风险贴现率)
                Firm state:
                    - b from dist_b or uniform, 
                    - z from stationary AR(1),
                    - eta from Bernoulli,
                    - i from uniform,
                    - weights from uniform (用于分配资本 K),
                    - K from lnkf and weights,
                    - ids from uuid,
                    - alive = True,
                    - entry = 1 (初始进入者)
        return {
            'x': x, # 宏观状态 scalar
            'b': b, # 公司杠杆 (n,) 
            'z': z, # 公司生产率 (n,)
            'eta': eta, # 公司 idiosyncratic shock (n,)
            'i': i, # 公司投资 (n,)
            'K': K, # 公司资本 (n,)
            'hatcf': hatcf.item(), # 宏观 proxy scalar
            'lnkf': lnkf.item(), # 宏观资本 scalar
            'M':1, # 初始无风险贴现率
            'ids': ids,
            'alive': torch.ones(n, dtype=torch.bool, device=device), # 公司存活状态 (n,)
            'entry': torch.ones(n, device=device) # 公司进入状态 (n,)
        }
        """
        n = self.group_size
        device = self.device
        
        # 宏观状态
        x = sample_stationary_ar1(1, self.config.RHO_X, self.config.SIGMA_X, 
                                   self.config.XBAR, device).item()
        
        # 公司状态

        z = sample_stationary_ar1(n, self.config.RHO_Z, self.config.SIGMA_Z,
                                   self.config.ZBAR, device)
        b = self._sample_initial_b(n, z, device)
        eta = sample_bernoulli(n, self.config.ZETA, device)
        i = sample_uniform(n, 0.0, self.config.I_THRESHOLD, device)
        
        # 资本（先生成 lnkf，再生成 weights，最后计算 K）
        # lnkf ~ N(4, 1)
        lnkf = torch.normal(mean=4.0, std=1.0, size=(1,), device=device)

        weights = torch.rand(n, device=device)
        weights = weights / weights.sum()

        # 直接用 weights 分配总资本：K_i = exp(lnkf) * weight_i
        K = torch.exp(lnkf) * weights

        # 初始宏观 proxy
        hatcf = torch.normal(mean=-2.0, std=1.0, size=(1,), device=device)
        
        # 公司 ID
        ids = [uuid.uuid4().hex for _ in range(n)]
        
        return {
            'x': x,
            'b': b,
            'z': z,
            'eta': eta,
            'i': i,
            'K': K,
            'hatcf': hatcf.item(),
            'lnkf': lnkf.item(),
            'M':1,
            'ids': ids,
            'alive': torch.ones(n, dtype=torch.bool, device=device),
            'entry': torch.ones(n, device=device)
        }
    
    def _process_node(
        self, 
        state: Dict, 
        path_idx: int, 
        t: int,
        branch_k: int = -1
    ) -> Tuple[List, Dict]:
        """
        处理单个节点，计算Policy & Value值以及宏观聚合变量
        
        Args:
            state: 当前状态
            path_idx: path 索引
            t: 时间步
            branch_k: 分支索引 (-1 表示父节点，0,1,... 表示分支)
        
        1. 使用 Policy/Value 计算决策
            input: firm-state (b,z,eta,i,x,hatcf,lnkf)
            output:
                - Q, P0, PI
                - bar_i, bar_z
                - P, bp0, bpI, bp
        2. 资源核算
        3. 宏观聚合
            output: Y, I, Phi, C (firm specific)
        """
        firm_data = []
        device = self.device
        
        # 构建输入张量
        n = state['alive'].sum().item()
        if n == 0:
            return [], {'path': path_idx, 't': t, 'branch': branch_k, 'K': 0, 'C': 0, 'LnK': -10, 'Hatc': -10}
        
        alive_mask = state['alive']
        alive_idx = torch.nonzero(alive_mask, as_tuple=False).squeeze(-1)
        
        # 提取存活公司的状态
        b = state['b'][alive_idx]
        z = state['z'][alive_idx]
        eta = state['eta'][alive_idx]
        i = state['i'][alive_idx]
        K = state['K'][alive_idx]
        entry = state.get('entry', torch.zeros_like(state['b']))[alive_idx]
        ids = [state['ids'][idx] for idx in alive_idx.tolist()]
        
        x = torch.full((n,), state['x'], device=device)
        hatcf = torch.full((n,), state['hatcf'], device=device)
        lnkf = torch.full((n,), state['lnkf'], device=device)
        
        # 构建 firm-state
        firm_state = torch.stack([b, z, eta, i, x, hatcf, lnkf], dim=1)
        
        # Policy/Value forward
        with torch.no_grad():
            output = self.models['policy_value'](firm_state)
        
        # 资源核算
        Y, I, Phi, C = self._resource_accounting(
            K, z, state['x'], output.bar_i.squeeze(), output.bar_z.squeeze(), i
        )
        
        # 记录公司数据
        for j in range(n):
            M_val = state.get('M', 1.0)
            if M_val is None:
                M_val = 1.0
            row = {
                'path': path_idx,
                't': t,
                'branch': branch_k,
                'ID': ids[j],
                'entry': entry[j].item(),
                'alive': output.bar_z[j].item() < 0.5,  # 存活状态由 bar_z 决定
                'b': b[j].item(),
                'z': z[j].item(),
                'ETA': eta[j].item(),
                'i': i[j].item(),
                'x': state['x'],
                'Hatcf': state['hatcf'],
                'LnKF': state['lnkf'],
                'K': K[j].item(),
                'M': M_val,
                'Q': output.Q[j].item(),
                'P0': output.P0[j].item(),
                'PI': output.PI[j].item(),
                'Bar_i': output.bar_i[j].item(),
                'Bar_z': output.bar_z[j].item(),
                'P': output.P[j].item(),
                'bp0': output.bp0[j].item(),
                'bpI': output.bpI[j].item(),
                'bp': output.bp[j].item(),
                'Y': Y[j].item(),
                'I': I[j].item(),
                'Phi': Phi[j].item(),
                'C': C[j].item()
            }
            firm_data.append(row)
        
        # 宏观聚合
        K_total = (K * output.bar_z.squeeze()).sum().item()  # 使用 bar_z 作为破产判断
        C_total = C.sum().clamp(min=0).item()
        macro_row = {
            'path': path_idx,
            't': t,
            'branch': branch_k,
            'K': K_total,
            'C': C_total,
            'LnK': np.log(K_total + 1e-8),
            'Hatc': np.log(C_total / (K_total + 1e-8) + 1e-5),
            'n_firms': n,
            'M': state.get('M'),
            'x': state['x'],
            'hatcf': state['hatcf'],
            'lnkf': state['lnkf']
        }
        
        # 更新 state 中的宏观 proxy
        state['hatcf'] = macro_row['Hatc']
        state['lnkf'] = macro_row['LnK']
        
        # 更新内生状态（杠杆和资本）
        state['bar_i'] = output.bar_i.squeeze()
        state['bar_z'] = output.bar_z.squeeze()
        state['bp'] = output.bp.squeeze()
        
        return firm_data, macro_row
    
    def _resource_accounting(
        self,
        K: torch.Tensor,
        z: torch.Tensor,
        x: float,
        bar_i: torch.Tensor,
        bar_z: torch.Tensor,
        i: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        资源核算
        
        Y = exp(x+z) * K
        Φ = (1-PHI) * (1 + exp(x+z)) * K * bar_z
        I = bar_i * K * i - bar_z * K + δ * K
        C = Y - I - Φ
        """
        x_tensor = torch.tensor(x, device=K.device)
        
        Y = torch.exp(x_tensor + z) * K
        Phi = (1 - self.phi) * (1 + torch.exp(x_tensor + z)) * K * bar_z
        I = bar_i * K * i - bar_z * K + self.delta * K
        C = Y - I - Phi
        
        return Y, I, Phi, C
    
    def _expand_branches(self, state: Dict, t: int) -> List[Dict]:
        """
        将当前状态扩展到多个分支，每个分支独立演化，生成 t+1 期的状态，也就是更新到下一次使用_process_node 的状态
        Args:
            state: 当前状态
            t: 当前时间步
        更新内容:
            - 宏观状态 x 演化
            - 公司状态 z, eta, i 演化
            - 更新杠杆 b (基于 parent state 的 bp 和 b)
            - 更新资本 K （基于bar_i）
            - 使用 FC1 更新宏观 proxy hatcf 和 lnkf
        
        
        
        """
        branches = []
        device = self.device
        
        alive_mask = state['alive']
        n = alive_mask.sum().item()
        
        if n == 0:
            return [state.copy() for _ in range(self.branch_num)]
        
        for branch_k in range(self.branch_num):
            new_state = {}
            
            # 宏观状态演化
            new_state['x'] = sample_ar1(
                torch.tensor(state['x'], device=device),
                self.config.RHO_X,
                self.config.SIGMA_X,
                self.config.XBAR
            ).item()
            
            # 公司状态演化
            new_state['z'] = sample_ar1(
                state['z'],
                self.config.RHO_Z,
                self.config.SIGMA_Z,
                self.config.ZBAR
            )
            new_state['eta'] = sample_bernoulli(len(state['z']), self.config.ZETA, device)
            new_state['i'] = sample_uniform(len(state['i']), 0.0, self.config.I_THRESHOLD, device)
            
            # 更新杠杆
            bp_prev = state.get('bp')
            if bp_prev is not None and bp_prev.numel() != state['b'].numel():
                # pad or truncate to match current firm count (e.g., after entry)
                if bp_prev.numel() < state['b'].numel():
                    pad = torch.zeros(state['b'].numel() - bp_prev.numel(), device=device)
                    bp_prev = torch.cat([bp_prev, pad], dim=0)
                else:
                    bp_prev = bp_prev[: state['b'].numel()]
            if bp_prev is not None:
                new_state['b'] = new_state['eta'] * bp_prev + (1 - new_state['eta']) * state['b']
            else:
                new_state['b'] = state['b'].clone()
            
            # 更新资本
            bar_i_prev = state.get('bar_i')
            if bar_i_prev is not None and bar_i_prev.numel() != state['K'].numel():
                if bar_i_prev.numel() < state['K'].numel():
                    pad = torch.zeros(state['K'].numel() - bar_i_prev.numel(), device=device)
                    bar_i_prev = torch.cat([bar_i_prev, pad], dim=0)
                else:
                    bar_i_prev = bar_i_prev[: state['K'].numel()]
            if bar_i_prev is not None:
                new_state['K'] = bar_i_prev * self.g * state['K'] + (1 - bar_i_prev) * state['K']
            else:
                new_state['K'] = state['K'].clone()
            
            # 使用 FC1 更新宏观 proxy
            if 'sdf_fc1' in self.models and self.models['sdf_fc1'] is not None:
                hatcf, lnkf = self._predict_macro_fc1(
                    state['x'], new_state['x'], state['hatcf'], state['lnkf']
                )
                new_state['hatcf'] = hatcf
                new_state['lnkf'] = lnkf
                
                _, _, M, _, _ = self.models['sdf_fc1'].forward_step(
                    x_prev=torch.tensor([state['x']], device=device, dtype=torch.float32),
                    x_curr=torch.tensor([new_state['x']], device=device, dtype=torch.float32),
                    hatcf_prev=torch.tensor([state['hatcf']], device=device, dtype=torch.float32),
                    lnkf_prev=torch.tensor([state['lnkf']], device=device, dtype=torch.float32),
                    return_physical=True
                )
                new_state['M'] = M.squeeze().item()
            else:
                new_state['hatcf'] = state['hatcf']
                new_state['lnkf'] = state['lnkf']
                new_state['M'] = 0.98
            
            # 复制其他状态
            new_state['ids'] = list(state['ids'])
            new_state['alive'] = state['alive'].clone()
            new_state['entry'] = torch.zeros_like(state['b'])
            
            branches.append(new_state)
        
        return branches
    
    def _predict_macro_fc1(
        self, 
        x_t: float, 
        x_t1: float, 
        hatcf_t: float, 
        lnkf_t: float
    ) -> Tuple[float, float]:
        """
        使用 FC1 预测下期宏观状态
        """
        model = self.models['sdf_fc1']
        
        with torch.no_grad():
            fc1_input = torch.tensor(
                [[x_t, x_t1, hatcf_t, lnkf_t]],
                device=self.device,
                dtype=torch.float32
            )
            hatcf_t1, lnkf_t1 = model.forward_fc1(fc1_input, return_physical=True)
        
        return hatcf_t1.item(), lnkf_t1.item()
    
    def _apply_exit(self, state: Dict) -> Dict:
        """
        应用退出规则
        
        bar_z > 0.5 的公司退出
        """
        if 'bar_z' not in state:
            return state
        
        alive_mask = state['alive']
        bar_z = state.get('bar_z', torch.zeros_like(state['b']))
        if bar_z.numel() != alive_mask.numel():
            if bar_z.numel() < alive_mask.numel():
                pad = torch.zeros(alive_mask.numel() - bar_z.numel(), device=self.device)
                bar_z = torch.cat([bar_z, pad], dim=0)
            else:
                bar_z = bar_z[: alive_mask.numel()]
        
        # 更新存活状态
        new_alive = alive_mask & (bar_z < 0.5)
        state['alive'] = new_alive
        
        return state
    
    def _apply_entry(self, state: Dict, path_idx: int, t: int) -> Dict:
        """
        应用进入规则
        
        生成潜在进入者，筛选后加入
        """
        device = self.device
        
        # 生成潜在进入者
        n_potential = 20  # 每期潜在进入者数量
        
        z_new = sample_stationary_ar1(
            n_potential, self.config.RHO_Z, self.config.SIGMA_Z,
            self.config.ZBAR, device
        )
        i_new = sample_uniform(n_potential, 0.0, self.config.I_THRESHOLD, device)
        
        # 进入条件：entry_value > 0
        x = state['x']
        profit = torch.exp(torch.tensor(x, device=device) + z_new) - self.delta
        entry_value = 1 + profit.clamp(min=0) - i_new
        
        enter_mask = entry_value > 0
        n_enter = enter_mask.sum().item()
        
        if n_enter == 0:
            return state
        
        # 添加进入者
        new_ids = [uuid.uuid4().hex for _ in range(n_enter)]
        
        state['b'] = torch.cat([state['b'], torch.zeros(n_enter, device=device)])
        state['z'] = torch.cat([state['z'], z_new[enter_mask]])
        state['eta'] = torch.cat([state['eta'], sample_bernoulli(n_enter, self.config.ZETA, device)])
        state['i'] = torch.cat([state['i'], i_new[enter_mask]])
        state['K'] = torch.cat([state['K'], torch.ones(n_enter, device=device)])
        state['ids'].extend(new_ids)
        state['alive'] = torch.cat([state['alive'], torch.ones(n_enter, dtype=torch.bool, device=device)])
        state['entry'] = torch.cat([
            torch.zeros(len(state['b']) - n_enter, device=device),
            torch.ones(n_enter, device=device)
        ])
        
        return state

    def _sample_initial_b(self, n: int, z: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        使用可选的分布生成器初始化 b；若不可用则回退到均匀分布。
        """
        sampler = self.models.get('dist_b') if isinstance(self.models, dict) else None
        if sampler is None:
            return sample_uniform(n, 0.0, 0.5, device)
        
        if hasattr(sampler, 'sample_distribution'):
            context_dim = getattr(sampler, 'context_dim', None)
            if context_dim is None:
                raise ValueError("dist_b model missing context_dim for sampling")
            context = torch.randn(n, context_dim, device=device)
            with torch.no_grad():
                b = sampler.sample_distribution(context)
            return b.squeeze(-1)
        
        if callable(sampler):
            b = sampler(n=n, z=z, device=device)
            return b.squeeze(-1)
        
        return sample_uniform(n, 0.0, 0.5, device)
    
    def get_real_bz_distribution(self, df_firm: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从模拟数据中提取 Real(b,z) 分布
        """
        b = torch.tensor(df_firm['b'].values, device=self.device)
        z = torch.tensor(df_firm['z'].values, device=self.device)
        return b, z
    
    def compute_fc2_features(self, state: Dict) -> torch.Tensor:
        """
        计算 FC2 输入特征
        
        Returns:
            phi: (1, 201) - [b_quantiles(100), z_quantiles(100), x]
        """
        alive_mask = state['alive']
        b = state['b'][alive_mask]
        z = state['z'][alive_mask]
        
        b_q = compute_quantile_features(b, 100)
        z_q = compute_quantile_features(z, 100)
        
        x = torch.tensor([state['x']], device=self.device)
        
        phi = torch.cat([b_q, z_q, x]).unsqueeze(0)
        
        return phi
