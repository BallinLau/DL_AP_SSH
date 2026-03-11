# DL-AP: Deep Learning Asset Pricing

基于深度学习的资产定价模型，实现 SDF（随机贴现因子）、宏观状态预测和公司决策的联合学习。

## 项目结构

```
DL-AP/
├── config/                  # 配置模块
│   └── README.md             # 配置说明
│   ├── __init__.py
│   ├── constants.py         # 常量定义（经济参数、模型维度）
│   └── hyperparams.py       # 训练超参数
├── models/                  # 模型模块
│   └── README.md             # 模型说明
│   ├── __init__.py
│   ├── base.py              # 基础网络组件（MLP, ResidualBlock, Attention）
│   ├── share_layer.py       # 共享层架构（ShareLayer, Heads）
│   ├── sdf_fc1.py           # SDF 和 FC1（宏观状态预测）模型
│   ├── fc2.py               # FC2（截面分布到宏观代理）模型
│   └── policy_value.py      # 策略价值联合模型
├── losses/                  # 损失函数模块
│   └── README.md             # 损失说明
│   ├── __init__.py
│   ├── utils.py             # 损失计算工具
│   ├── sdf_loss.py          # SDF 损失（欧拉方程残差）
│   ├── p0_loss.py           # P0 损失（无投资 Bellman）
│   ├── pi_loss.py           # PI 损失（投资 Bellman）
│   ├── q_loss.py            # Q 损失（债券定价）
│   └── fc2_loss.py          # FC2 损失（宏观一致性）
├── data/                    # 数据生成模块
│   └── README.md             # 数据说明
│   ├── __init__.py
│   ├── data_utils.py        # 数据工具函数
│   ├── sample.py            # Sample 类（训练数据生成）
│   └── simulate_ts.py       # SimulateTS 类（树状模拟）
├── training/                # 训练模块
│   └── README.md             # 训练说明
│   ├── __init__.py
│   ├── gradient_utils.py    # 梯度工具
│   ├── scheduler.py         # 调度器
│   ├── episode.py           # Episode 管理
│   └── trainer.py           # Trainer 主类
├── utils/                   # 工具模块
│   └── README.md             # 工具说明
│   ├── __init__.py
│   ├── logging_utils.py     # 日志工具
│   ├── checkpoint.py        # 检查点管理
│   ├── visualization.py     # 可视化
│   └── metrics.py           # 评估指标
├── main.py                  # 主入口
├── requirements.txt         # 依赖
└── README.md                # 说明文档
```

## 核心概念

### 1. Firm-State Vector
公司状态向量 $s = (b, z, \eta, i, x, \hat{c}_f, \ln K_f)$，共 7 维：
- $b$: 杠杆率
- $z$: 公司特异性生产率
- $\eta$: 融资冲击（伯努利）
- $i$: 投资成本（仅进入投资头）
- $x$: 宏观状态（AR(1)）
- $\hat{c}_f, \ln K_f$: FC1 预测的宏观代理

### 2. ShareLayer 架构
共享特征层 + 多输出头：
- **Q Head**: 债券价格
- **bp0 Head**: 无投资时的最优杠杆
- **bpI Head**: 投资时的最优杠杆
- **P0 Head**: 无投资时的公司价值
- **PI Head**: 投资时的公司价值
- **bar_i Head**: 投资概率
- **bar_z Head**: 退出概率

### 3. SDF & FC1
- **SDF**: $M_{t+1} = \beta (\hat{c}_{t+1}/\hat{c}_t)^{-\gamma}$
- **FC1**: 预测下期宏观状态 $(\hat{c}_{t+1}, \ln K_{t+1})$

### 4. FC2
从截面分布（b,z 的分位数）聚合到宏观代理，并用 Policy/Value + 资源核算生成一致性目标。

## 安装

```bash
cd DL-AP
pip install -r requirements.txt
```

## 使用

### 训练

```bash
# 联合训练
python main.py --mode train --n_episodes 10 --train_mode joint

# 分阶段训练（先 SDF/FC1，后 Policy/Value）
python main.py --mode train --n_episodes 10 --train_mode staged

# 交替训练
python main.py --mode train --n_episodes 10 --train_mode alternating
```

### 评估

```bash
python main.py --mode eval --resume best
```

### 模拟

```bash
python main.py --mode simulate --resume best
```

## 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n_episodes` | 10 | 训练 Episode 数 |
| `--epochs_per_episode` | 10 | 每个 Episode 的 epoch 数 |
| `--batch_size` | 256 | 批大小 |
| `--lr` | 1e-3 | 学习率 |
| `--hidden_dim` | 128 | 隐藏层维度 |
| `--n_layers` | 4 | 网络层数 |

## 经济参数

| 参数 | 值 | 说明 |
|------|-----|------|
| δ | 0.1 | 折旧率 |
| τ | 0.35 | 税率 |
| β | 0.96 | 折扣因子 |
| γ | 2.0 | 风险厌恶系数 |
| ρ_x | 0.95 | x 的 AR(1) 系数 |
| ρ_z | 0.9 | z 的 AR(1) 系数 |
| ζ | 0.3 | 融资冲击概率 |

## 训练流程

1. **数据生成**: 使用 `Sample` 或 `SimulateTS` 生成训练数据
2. **FC1 填充**: 使用 FC1 预测宏观状态
3. **Policy/Value 填充**: 使用 Policy/Value 网络计算决策
4. **损失计算**: 计算 SDF、P0、PI、Q、FC2 损失（FC2 以聚合宏观量为监督目标）
5. **梯度更新**: 反向传播并更新参数
6. **迭代**: 重复以上步骤

## 核心损失函数

1. **SDF Loss**: 欧拉方程残差 $\mathbb{E}[M_{t+1} R_{t+1}] = 1$
2. **P0 Loss**: 无投资 Bellman 方程
3. **PI Loss**: 投资 Bellman 方程
4. **Q Loss**: 债券定价方程
5. **FC2 Loss**: 宏观一致性约束

## API 示例

```python
from config import Config, HyperParams
from models import SDFFC1Combined, PolicyValueModel
from training import Trainer

# 配置
config = Config
hyperparams = HyperParams(lr=1e-3)

# 构建模型
models = {
    'sdf_fc1': SDFFC1Combined(...),
    'policy_value': PolicyValueModel(...)
}

# 训练
trainer = Trainer(models, config, hyperparams)
trainer.train(n_episodes=10)
```

## 参考文献

- 文档基于 VibeCoding 项目文档
- Deep Learning Equilibrium Asset Pricing 相关文献

## 子模块文档

- `config/README.md`
- `data/README.md`
- `models/README.md`
- `losses/README.md`
- `training/README.md`
- `utils/README.md`
