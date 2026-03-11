"""
超参数配置类
用于管理训练过程中的超参数
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class HyperParams:
    """
    训练超参数配置
    使用 dataclass 便于序列化和修改
    """
    
    # ========== 训练基础参数 ==========
    epochs: int = 100
    batch_size: int = 512
    
    # ========== 优化器参数 ==========
    # SDF & FC1
    sdf_lr: float = 5e-4
    sdf_weight_decay: float = 1e-3
    fc1_lr: float = 1e-3
    fc1_weight_decay: float = 1e-4
    # 通用学习率（调度器基准值，缺省时沿用 sdf_lr）
    lr: float = 5e-4
    
    # Policy & Value
    policy_lr: float = 1e-3
    policy_weight_decay: float = 1e-6
    
    # FC2
    fc2_lr: float = 1e-4
    fc2_weight_decay: float = 1e-4
    
    # ========== 学习率调度 ==========
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 2
    lr_min: float = 1e-6
    # 通用学习率下限（LearningRateScheduler 使用）
    min_lr: float = 1e-6
    
    # 动态学习率冷却
    auto_cooldown_threshold: float = 0.8
    auto_cooldown_factor: float = 0.5
    lr_floor: float = 1e-5
    warmup_steps: int = 0
    max_steps: int = 20000
    
    # ========== 梯度处理 ==========
    max_grad_norm: float = 1.0
    fc2_max_grad_norm: float = 10.0
    
    # ========== 损失函数参数 ==========
    # AIO 权重（动态残差插值）
    aio_weight: float = 0.5
    aio_warmup_epochs: int = 10
    
    # 损失权重
    w_sdf: float = 1.0
    w_p0: float = 1.0
    w_pi: float = 1.0
    w_q: float = 1.0
    w_fc2: float = 1.0
    lambda_sdf: float = 1.0
    lambda_fc: float = 1.0
    lambda_warmup_epochs: int = 5
    
    # FC2 跨期一致性
    lambda_trans: float = 0.0
    lambda_trans_max: float = 1.0
    lambda_trans_warmup_epochs: int = 20
    
    # ========== 训练策略 ==========
    # 多阶段训练
    stage1_epochs: int = 30  # Q 训练
    stage2_epochs: int = 30  # P0/PI 训练
    stage3_epochs: int = 20  # bp0/bpI 训练
    stage4_epochs: int = 20  # 联合微调
    
    # L-BFGS 精调
    use_lbfgs: bool = False
    lbfgs_iters: int = 100
    lbfgs_lr: float = 1.0
    
    # 小 b 微调
    sb_steps: int = 350
    sb_margin: float = 0.07
    sb_penalty_weight: float = 10.0
    
    # ========== 稳定性保障 ==========
    # 损失爆炸防护
    loss_explosion_threshold: float = 100.0
    
    # NaN/Inf 检测
    nan_recovery: bool = True
    
    # 早停
    early_stop_threshold: float = 0.1
    high_loss_lr_switch: float = 10.0
    high_loss_lr: float = 1e-4
    
    # ========== 数据生成参数 ==========
    n_samples: int = 10000
    n_paths: int = 1000
    simulate_horizon: int = 20

    # FC1 重建损失权重（0 表示关闭）
    fc1_recon_weight: float = 0.0
    
    # ========== 日志与保存 ==========
    log_interval: int = 10
    save_interval: int = 5
    
    # ========== 设备 ==========
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    
    def get_device(self) -> torch.device:
        """返回torch设备"""
        return torch.device(self.device)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'HyperParams':
        """从字典创建"""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
