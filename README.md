# NeuMatC Reproduction (Paper: arXiv:2511.22934)

这个仓库给出论文 **NeuMatC: A General Neural Framework for Fast Parametric Matrix Operation** 的可运行复现实现（PyTorch）。

> 论文核心思想：学习参数 `p -> 矩阵运算结果 G(p)` 的低秩连续映射，形式为
> `G_i(p) = C_i ×_3 Φ_{θ_i}(p)`，并结合监督项 + 残差约束 + 自适应配点(collocation)训练。

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

## 安装

```bash
python -m pip install -r requirements.txt
```

## 运行

### 1) 矩阵求逆任务

```bash
python scripts/reproduce_neumatc.py --task inversion --n 16 --steps 800
```

### 2) 矩阵 SVD 任务

```bash
python scripts/reproduce_neumatc.py --task svd --n 16 --steps 800
```

输出是 JSON，包含训练耗时、测试评估耗时、相对误差、残差 RMSE。

## 文件结构

- `neumatc/model.py`：NeuMatC 网络定义（encoder + tensor heads）
- `neumatc/tasks.py`：参数化矩阵任务定义、真值构建、残差函数
- `neumatc/train.py`：训练与评估（含自适应 collocation）
- `scripts/reproduce_neumatc.py`：一键实验脚本

## 说明

由于原论文未提供官方代码仓库链接，本复现重点对齐其方法框架和训练机制，并给出最小可运行实现，便于后续替换为论文中的真实数据（如无线通信信道矩阵）和超参数。
