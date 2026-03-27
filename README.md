# NeuMatC Reproduction (Paper: arXiv:2511.22934)

这个仓库给出论文 **NeuMatC: A General Neural Framework for Fast Parametric Matrix Operation** 的可运行复现实现（PyTorch）。

> 论文核心思想：学习参数 `p -> 矩阵运算结果 G(p)` 的低秩连续映射，形式为
> `G_i(p) = C_i ×_3 Φ_{θ_i}(p)`，并结合监督项 + 残差约束 + 自适应配点(collocation)训练。

## 你提出的新需求（已实现）

- **自动生成数据**：运行训练脚本时会先检查本地数据文件，不存在就自动生成并保存。
- **可重复实验**：数据文件按 task / 矩阵规模 / 样本数 / seed 命名，下一次运行会直接复用。
- **Bash 脚本**：提供一键运行脚本，直接在 shell 中执行。

## 已实现内容

- 通用 NeuMatC 模型：
  - `Φ(p)` 为 MLP 参数编码器；
  - `C` 为可学习 latent tensor；
  - 输出通过 mode-3 contraction（`einsum`）得到目标矩阵。
- 两个任务复现：
  - **Parametric Matrix Inversion**：残差 `A(p)G(p)-I`；
  - **Parametric Matrix SVD**：重构残差 + 正交性残差。
- 训练流程包含：
  - 监督损失（训练采样点）
  - 残差损失（collocation 点）
  - 按间隔进行 failure region 筛选并追加 collocation 点（Algorithm 1 风格）
- 数据流程包含：
  - 自动生成 `p_train / p_collocation / p_test`
  - 自动生成并缓存 `targets_train / targets_test`
  - 自动加载缓存数据，减少重复预处理时间

## 安装

```bash
python -m pip install -r requirements.txt
```

## 运行（自动生成数据）

### 1) 直接运行 Python 脚本

```bash
python scripts/reproduce_neumatc.py --task inversion --n 16 --steps 800
python scripts/reproduce_neumatc.py --task svd --n 16 --steps 800
```

首次运行会在 `data/` 下生成 `.pt` 数据；后续重复运行默认直接复用。

如果你要强制重建数据：

```bash
python scripts/reproduce_neumatc.py --task inversion --force-regenerate-data
```

### 2) 使用 Bash 脚本（你要求的命令脚本）

运行单个任务：

```bash
bash scripts/run_neumatc.sh inversion --n 16 --steps 800
bash scripts/run_neumatc.sh svd --n 16 --steps 800
```

连续跑两个任务：

```bash
bash scripts/run_all.sh --n 16 --steps 800
```

## 文件结构

- `neumatc/model.py`：NeuMatC 网络定义（encoder + tensor heads）
- `neumatc/tasks.py`：参数化矩阵任务定义、真值构建、残差函数
- `neumatc/train.py`：训练与评估（含自适应 collocation）
- `neumatc/data.py`：自动数据生成、缓存、加载
- `scripts/reproduce_neumatc.py`：主实验入口（先数据再训练）
- `scripts/run_neumatc.sh`：单任务 bash 命令脚本
- `scripts/run_all.sh`：双任务 batch bash 命令脚本

## 说明

由于原论文未提供官方代码仓库链接，本复现重点对齐其方法框架和训练机制，并给出最小可运行实现，便于后续替换为论文中的真实数据（如无线通信信道矩阵）和超参数。
