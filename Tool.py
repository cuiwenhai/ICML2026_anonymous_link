import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
import warnings
import numpy as np
from scipy.stats import wasserstein_distance
from statsmodels.nonparametric.kde import KDEUnivariate
# 2026/3/26
from statsmodels.nonparametric.kernel_density import KDEMultivariateConditional



import pandas as pd
import numpy as np

# 忽略一些不必要的警告
warnings.filterwarnings('ignore')


# ==========================================
# 1. 核心模型定义 (LAD Regressor)
# ==========================================
class LADRegressor(nn.Module):
    def __init__(self, input_size, hidden_sizes=[32, 32], output_size=1):
        super(LADRegressor, self).__init__()
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ==========================================
# 2. 训练函数 (支持 CV)
# ==========================================
def train_model_cv(model_class, X_numpy, y_numpy, criterion, k_folds=5, lr=1e-3, epochs=4000, patience=10, batch_size=64,
                   device='cpu'):
    """
    K-fold Cross-Validation training specifically for LAD (L1 Loss).
    """
    # 1. 数据转换与标准化 (在函数内部处理，防止泄露，但这里简单起见假设外部已StandardScale，或在此处统一转Tensor)
    X = torch.FloatTensor(X_numpy)
    y = torch.FloatTensor(y_numpy).unsqueeze(1)  # [N] -> [N, 1] 关键！




    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    models = []


    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        # 构造 Dataset
        train_dataset = Subset(TensorDataset(X, y), train_idx)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        X_val = X[val_idx].to(device)
        y_val = y[val_idx].to(device)

        # 初始化模型 (必须传入 input_size)
        input_dim = X.shape[1]
        model = model_class(input_size=input_dim).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

            # Validation
            if epoch % 1 == 0:
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val)
                    val_loss = criterion(val_pred, y_val).item()

                # Early Stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

        # 恢复该 Fold 的最优权重
        if best_state is not None:
            model.load_state_dict(best_state)

        models.append(model.cpu())  # 移回 CPU 保存

    # 定义集成预测函数 (平均多个模型的预测结果)
    def predict_fn(x_input_numpy):
        x_tensor = torch.FloatTensor(x_input_numpy)
        preds = []
        for m in models:
            m.eval()
            with torch.no_grad():
                preds.append(m(x_tensor).detach())
        # Stack shape: [K, N, 1] -> Mean -> [N, 1]
        return torch.stack(preds).mean(dim=0).numpy().flatten()

    return predict_fn


# ==========================================
# 3. 辅助工具函数 (I-Spline & ECDF) - 保持不变
# ==========================================
# def get_ispline_basis(u, degree=3, n_internal_knots=50):
#     """生成 I-Spline 基矩阵"""
#     # 避免 u 为 0 或 1 导致的边界问题，稍微 clip 一下
#     u = np.clip(u, 1e-6, 1 - 1e-6)
#
#     internal_knots = np.linspace(0, 1, n_internal_knots + 2)[1:-1]
#     knots = np.concatenate(([0] * (degree + 1), internal_knots, [1] * (degree + 1)))
#     n_basis = len(knots) - (degree + 1)
#
#     basis_matrix = []
#     for i in range(n_basis):
#         coeffs = np.zeros(n_basis)
#         coeffs[i] = 1.0
#         bs = BSpline(knots, coeffs, k=degree)
#         ispline = bs.antiderivative()
#         basis_matrix.append(ispline(u))
#
#     return np.vstack(basis_matrix).T

import numpy as np
from scipy.interpolate import BSpline


def get_ispline_basis(u, degree=3, n_internal_knots=6, natural=False):
    """
    生成 I-Spline 基矩阵，支持在普通模式和 Natural 模式间切换。

    参数:
    - u: 输入向量 (如 tau)
    - degree: 样条次数 (默认 3 为三次样条)
    - n_internal_knots: 内部节点数量
    - natural: 是否启用 Natural 约束。若为 True，则返回 (basis, d2_bounds)；
               若为 False，则仅返回 basis。
    """
    u = np.asarray(u)

    # 1. 构建节点序列 (Knots)
    internal_knots = np.linspace(0, 1, n_internal_knots + 2)[1:-1]
    knots = np.concatenate(([0] * (degree + 1), internal_knots, [1] * (degree + 1)))
    n_basis = len(knots) - (degree + 1)

    basis_matrix = np.zeros((len(u), n_basis))

    # 如果是 natural 模式，准备存储边界二阶导数 (I'' = M')
    d2_matrix_boundaries = np.zeros((2, n_basis)) if natural else None

    for i in range(n_basis):
        # 构建系数向量 (只有第 i 个为 1)
        coeffs = np.zeros(n_basis)
        coeffs[i] = 1.0

        # 标准 B-spline
        bs = BSpline(knots, coeffs, k=degree, extrapolate=False)


        # M-spline 归一化系数
        knot_span = knots[i + degree + 1] - knots[i]
        m_scale = (degree + 1) / knot_span if knot_span > 0 else 0.0

        # --- 生成 I-spline 取值 ---
        ispl = bs.antiderivative()
        ispl_vals = ispl(u) * m_scale


        # 边界修正逻辑
        #ispl_vals[u <= knots[i]] = 0.0
        #ispl_vals[u >= knots[i + degree + 1]] = 1.0
        #ispl_vals = np.nan_to_num(ispl_vals, nan=0.0)
        #ispl_vals[u >= knots[i + degree + 1]] = 1.0



        basis_matrix[:, i] = ispl_vals

        # --- 如果开启 Natural 模式，计算边界二阶导数 ---
        if natural:
            # I''(u) = M'(u)，即 B-spline 的一阶导乘以 m_scale
            d_mspl = bs.derivative()
            # 计算 0 和 1 处的导数
            d2_0 = np.nan_to_num(d_mspl(0.0)) * m_scale
            d2_1 = np.nan_to_num(d_mspl(1.0)) * m_scale
            d2_matrix_boundaries[0, i] = d2_0
            d2_matrix_boundaries[1, i] = d2_1

    if natural:
        return basis_matrix, d2_matrix_boundaries
    else:
        return basis_matrix



def compute_conditional_u(y_pred, s_groups):

    """计算 u = F_{f|s}(y)"""

    u_values = np.zeros_like(y_pred)

    unique_groups = np.unique(s_groups)

    for s in unique_groups:

        mask = (s_groups == s)

        group_preds = y_pred[mask]

        ranks = np.argsort(np.argsort(group_preds))

        n_group = len(group_preds)

        u_values[mask] = (ranks + 1) / (n_group + 1)

    return u_values


# ==========================================
# 4. 功能模块函数 (Pipeline)
# ==========================================
def compute_empirical_risk(y_pred, y_true, loss_type='LAD', **kwargs):
    """
    计算各种鲁棒损失函数的经验风险 (Empirical Risk)

    Args:
        y_pred: 预测值 (transformed values)
        y_true: 真实值
        loss_type: 'LAD', 'Quantile', 'Huber', 'Cauchy', 'Tukey'
        kwargs: 超参数 (tau, zeta, kappa, c/t)
    """
    r =  y_true -y_pred  # 残差 (Residual)

    # 1. Least Absolute Deviation (LAD)
    if loss_type == 'LAD':
        return np.mean(np.abs(r))

    # 2. Quantile Loss
    elif loss_type == 'Quantile':
        tau = kwargs.get('tau')
        # Check function: r * (tau - I(r<0))
        losses = np.where(r >= 0, tau * r, (tau - 1) * r)
        return np.mean(losses)


    # 3. Huber Loss
    elif loss_type == 'Huber':
        zeta = kwargs.get('zeta')  # 默认 1.345 对应 95% 效率
        is_small = np.abs(r) <= zeta
        losses = np.where(
            is_small,
            0.5 * r ** 2,
            zeta * np.abs(r) - 0.5 * zeta ** 2
        )
        return np.mean(losses)

    # 4. Cauchy Loss (Lorentzian)
    elif loss_type == 'Cauchy':
        kappa = kwargs.get('kappa')  # 尺度参数
        losses = np.log(1 + (kappa * r) ** 2)
        return np.mean(losses)

    # 5. Tukey's Biweight Loss
    elif loss_type == 'Tukey':
        c = kwargs.get('c')  # 默认 4.685
        # 这里的公式对应标准的 Loss function (积分形式)
        # L(r) = (c^2 / 6) * (1 - (1 - (r/c)^2)^3) if |r| <= c
        # L(r) = c^2 / 6 otherwise
        is_small = np.abs(r) <= c
        losses = np.where(
            is_small,
            (c ** 2 / 6.0) * (1 - (1 - (r / c) ** 2) ** 3),
            c ** 2 / 6.0
        )
        return np.mean(losses)
    elif loss_type == 'MSE':

        losses = r**2

        return np.mean(losses)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")






class QuantileLoss(nn.Module):
    def __init__(self, tau):
        super(QuantileLoss, self).__init__()
        self.tau = tau

    def forward(self, pred, target):
        err = target - pred
        return torch.mean(torch.max((self.tau - 1) * err, self.tau * err))


class CauchyLoss(nn.Module):
    def __init__(self, kappa):
        super(CauchyLoss, self).__init__()
        self.kappa = kappa

    def forward(self, pred, target):
        residual = pred - target
        return torch.mean(torch.log(1 + (self.kappa * residual) ** 2))


class TukeyLoss(nn.Module):
    def __init__(self, c):
        super(TukeyLoss, self).__init__()
        self.c = c

    def forward(self, pred, target):
        residual = pred - target
        # 避免绝对值大于 c 的部分梯度消失，通常需要配合良好的初始化
        # 这里实现标准的 Tukey Biweight 积分形式
        abs_r = torch.abs(residual)
        c = self.c

        loss_small = (c ** 2 / 6.0) * (1 - (1 - (residual / c) ** 2) ** 3)
        loss_large = c ** 2 / 6.0

        return torch.mean(torch.where(abs_r <= c, loss_small, loss_large))


# ==========================================
# 工厂函数：根据名称获取 PyTorch Criterion
# ==========================================
def get_torch_criterion(loss_type, **kwargs):
    """
    根据字符串返回对应的 PyTorch 损失函数对象
    """
    if loss_type == 'LAD':
        return nn.L1Loss()

    elif loss_type == 'Quantile':
        tau = kwargs.get('tau')
        return QuantileLoss(tau=tau)

    elif loss_type == 'Huber':
        # PyTorch 的 HuberLoss 参数名为 delta，对应你的 zeta
        zeta = kwargs.get('zeta')
        return nn.HuberLoss(delta=zeta)

    elif loss_type == 'Cauchy':
        kappa = kwargs.get('kappa')
        return CauchyLoss(kappa=kappa)

    elif loss_type == 'Tukey':
        c = kwargs.get('c')
        return TukeyLoss(c=c)

    elif loss_type == 'MSE':
        return  nn.MSELoss()

    else:
        raise ValueError(f"Unknown loss_type for PyTorch: {loss_type}")





def compute_kernel_conditional_u(y_pred, s_groups, bandwidth_type='normal_reference'):
    """
    使用 statsmodels 的核估计器计算 u = F_{f|s}(y)
    支持离散属性 s 的核平滑 (L_lambda) 和 连续预测值 y 的平滑 (W_h)
    """
    # 准备数据：y_pred 是连续变量 'c', s_groups 是离散变量 'u' (unordered discrete)
    # statsmodels 要求输入为 2D array
    endog = y_pred.reshape(-1, 1)
    exog = s_groups.reshape(-1, 1)

    # 初始化条件累计分布估计器
    # dep_type='c' 指预测值是连续的
    # indep_type='u' 指敏感属性是无序离散的 (对应你说的 L_lambda)
    dens = KDEMultivariateConditional(endog=endog, exog=exog,
                                      dep_type='c', indep_type='o',
                                      bw=[0.01, 0.01])

    # cdf() 函数返回的就是你公式里的 \tilde{F}_{f|s}(t)
    # 它内部自动处理了 W_h 和 L_lambda 的乘积与归一化
    u_values = dens.cdf(endog, exog)

    return u_values



def optimize_fair_transform(y_pred, s_groups, y_true, degree,
    n_knots, compute_conditional_F=compute_conditional_u, natural=False, lambda_nat=1e-4, loss_type='LAD', **loss_kwargs):
    """
    Step 2: 优化 I-Spline 变换参数 (支持自定义损失函数)
    """
    # 1. 计算 U 值和基矩阵



    """
        Step 2: 优化 I-Spline 变换参数 (已增强：支持 Natural I-spline 惩罚)
        """
    # 1. 计算 U 值和基矩阵 (适配新函数返回值)
    U = compute_conditional_F(y_pred, s_groups)

    if natural:
        Basis, d2_bounds = get_ispline_basis(U, degree, n_knots, natural=True)
    else:
        Basis = get_ispline_basis(U, degree, n_knots, natural=False)
        d2_bounds = None

    # 2. 增强优化目标：加入自然边界惩罚
    def objective(params, basis, y_target, d2_bounds_ptr):
        alpha_0 = params[0]
        alpha_rest = params[1:]
        y_trans = alpha_0 + basis @ alpha_rest

        # 计算基础经验风险
        main_loss = compute_empirical_risk(y_trans, y_target, loss_type, **loss_kwargs)

        # 如果开启了 Natural 模式，增加边界二阶导数惩罚
        if d2_bounds_ptr is not None:
            # d2_bounds_ptr 形状为 (2, n_basis)，对应 0 和 1 处的基函数二阶导数
            # 目标是让模型在两端的总曲率趋于 0
            curvature_left = alpha_rest @ d2_bounds_ptr[0]
            curvature_right = alpha_rest @ d2_bounds_ptr[1]
            natural_penalty = lambda_nat * (curvature_left ** 2 + curvature_right ** 2)
            return main_loss + natural_penalty

        return main_loss

    # 3. 初始化参数
    n_basis = Basis.shape[1]
    initial_params = np.ones(n_basis + 1) * 0.1
    initial_params[0] = np.median(y_true)

    # 4. 约束和优化 (L-BFGS-B 保持 beta >= 0 以维持单调性)
    bounds = [(None, None)] + [(0.0, None)] * n_basis

    result = minimize(
        objective, initial_params, args=(Basis, y_true, d2_bounds),
        method='L-BFGS-B', bounds=bounds,
        options={'maxiter': 5000, 'ftol': 1e-9, 'disp': False}
        )

    return result.x, degree, n_knots


def apply_transform(y_pred, s_groups, params, degree, n_knots, compute_conditional_F=compute_conditional_u):
    """应用变换 (适配新函数返回值)"""
    U = compute_conditional_F(y_pred, s_groups)

    # 获取基矩阵，注意处理可能返回的 tuple
    res = get_ispline_basis(U, degree, n_knots, natural=False)
    Basis = res[0] if isinstance(res, tuple) else res

    alpha_0 = params[0]
    alpha_rest = params[1:]
    return alpha_0 + Basis @ alpha_rest




def evaluate_metrics(y_pred, y_true, s_groups, loss_type, **loss_kwargs):
    """
    计算指定的 Loss 和 Unfairness Measure (Wasserstein Distance)

    Unfairness Measure 定义:
    sup_{s, s'} \int_{0}^{1} | Q_{s}^f(\tau) - Q_{s'}^f(\tau) | d\tau
    这在计算上等价于两个样本分布间的 1-Wasserstein 距离。
    """

    # 1. 计算指定的 Loss
    # 假设 compute_empirical_risk 已经在你的上下文中定义好了
    loss_val = compute_empirical_risk(y_pred, y_true, loss_type, **loss_kwargs)

    # 2. 计算 Unfairness (Wasserstein Distance)
    # 提取不同敏感群体的预测值
    unique_groups = np.unique(s_groups)

    # 情况 A: 只有两个组 (比如 0 和 1) - 最常见情况
    if len(unique_groups) == 2:
        y0 = y_pred[s_groups == unique_groups[0]]
        y1 = y_pred[s_groups == unique_groups[1]]

        # scipy 的 wasserstein_distance 自动处理样本大小不一致的情况
        # 它的计算原理正是利用分位数函数积分公式
        unfairness_measure = wasserstein_distance(y0, y1)

    # 情况 B: 超过两个组 (处理 sup_{s, s'})
    else:
        max_w1 = 0.0
        # 遍历所有两两组合，取最大值 (Supremum)
        for i in range(len(unique_groups)):
            for j in range(i + 1, len(unique_groups)):
                g_i = unique_groups[i]
                g_j = unique_groups[j]

                dist = wasserstein_distance(
                    y_pred[s_groups == g_i],
                    y_pred[s_groups == g_j]
                )
                if dist > max_w1:
                    max_w1 = dist
        unfairness_measure = max_w1

    return loss_val, unfairness_measure



def method_wasserstein_apply(y_pred_train, s_train, y_pred_test, s_test, sigma=1e-5):
    """
    Implementation of Algorithm 1 (Characterization of fair optimal prediction).
    g*(x, s) = (Sum p_s' Q_s') o F_s (f(x, s))
    """
    # 1. 计算每个组的经验频率 p_s
    unique_groups = np.unique(np.concatenate([s_train, s_test]))
    # 注意：论文中的 p_s 通常指总体概率。我们用训练集的比例估计。
    p_hat = {s: np.mean(s_train == s) for s in unique_groups}

    # 2. 准备 Calibration Data (Unlabeled Data U in Algo 1)
    # 按照 Algo 1，我们需要分割 U0 和 U1。为了简单且充分利用数据，
    # 我们将 y_pred_train 视为 U。为了严格遵循算法的split，我们进行内部划分。
    # 但实际上，使用全部数据构建 Quantile 更加稳定。
    # 这里我们简化：ar_0 (用于构建 Quantile) 使用 y_pred_train

    # 为每个组构建排序后的数组 (即经验分位数函数 Q)
    ar_0 = {}
    N_s = {}

    np.random.seed(42)  # 保证 Jitter 的一致性

    for s in unique_groups:
        # 获取该组的预测值
        preds = y_pred_train[s_train == s]

        # 添加 Jitter U([-sigma, sigma]) 以处理离散/ties
        jitter = np.random.uniform(-sigma, sigma, size=len(preds))
        preds_jittered = preds + jitter

        # 排序 (Step: sort(ar_0))
        ar_0[s] = np.sort(preds_jittered)
        N_s[s] = len(preds)

    # 3. 对测试点进行预测
    g_hat = np.zeros_like(y_pred_test)

    # 对测试集的每个点进行处理
    # 为了向量化加速，我们分租处理
    for s in unique_groups:
        mask_test = (s_test == s)
        if not np.any(mask_test): continue

        # 获取当前组的测试点预测值
        test_preds = y_pred_test[mask_test]
        # 对测试点也加 Jitter (Algo 1 要求)
        jitter_test = np.random.uniform(-sigma, sigma, size=len(test_preds))
        f_x_s = test_preds + jitter_test

        # === 核心步骤: 计算 rank (Evaluate F_s) ===
        # 我们使用 searchsorted 来找到 f_x_s 在 ar_0[s] 中的位置 k_s
        # searchsorted 返回索引 i，满足 ar[i-1] < v <= ar[i]
        # 这是经验 CDF 的非标准化形式
        # 使用 ar_0[s] 作为参考系 (对应 Algo 1 中的 ar_1，这里简化为同一数据集)
        # 如果严格遵循 Algo 1，需要把 train 分为两半，一半建 ar_0，一半建 ar_1。
        # 这里为了样本效率，让 ar_1 = ar_0 = train set。
        k_s = np.searchsorted(ar_0[s], f_x_s, side='right')

        # 处理边界 (防止索引越界)
        k_s = np.clip(k_s, 0, N_s[s] - 1)

        # === 核心步骤: Map to other groups and Average (Evaluate Eq 6) ===
        # g(x) = Sum p_s' * Q_s'( F_s(y) )
        weighted_sum = np.zeros_like(f_x_s)

        for s_prime in unique_groups:
            # 计算映射后的索引
            # index = ceil( N_s' * k_s / N_s )
            # 注意 Python 索引从 0 开始，公式需要微调
            # k_s / N_s 是分位数 (0~1)
            # index_prime 是在 s_prime 组对应的索引

            quantile_level = k_s / N_s[s]
            index_prime = np.ceil(quantile_level * N_s[s_prime]).astype(int) - 1
            index_prime = np.clip(index_prime, 0, N_s[s_prime] - 1)

            # 查表得到 Q_s'
            value_s_prime = ar_0[s_prime][index_prime]

            # 加权累加
            weighted_sum += p_hat[s_prime] * value_s_prime

        g_hat[mask_test] = weighted_sum

    return g_hat


# ==========================================
# 6. 画图函数 (Visualization)
# ==========================================
import matplotlib.pyplot as plt
import seaborn as sns


def plot_tradeoff_curves(df_plot, df_points):
    """
    绘制公平性-准确性权衡曲线 (分图绘制)
    Args:
        df_plot: 包含所有插值点的数据 (列: Lambda, Risk, Unfairness)
        df_points: 包含关键点的数据 (用于画基准点)
    """
    sns.set_style("whitegrid")

    # ==========================================
    # 图 1: Pareto Frontier (Unfairness vs Risk)
    # ==========================================
    # [关键步骤] 1. 先按 Lambda 聚合计算均值，用于画 Pareto 连线
    df_mean = df_plot.groupby('Lambda')[['Risk', 'Unfairness']].mean().reset_index()
    # 按照 Unfairness 排序，确保连线顺序是从左到右（或从右到左），防止线条乱窜
    df_mean = df_mean.sort_values('Unfairness')

    # ==========================================
    # 图 1: Pareto Frontier (Unfairness vs Risk)
    # ==========================================
    plt.figure(figsize=(8, 6))
    ax1 = plt.gca()

    # A. 画背景散点 (Raw Data)：展示每次 Run 的波动
    sns.scatterplot(
        data=df_plot,
        x='Unfairness',
        y='Risk',
        color='gray',
        alpha=0.3,  # 透明度设高，作为背景
        s=30,  # 点的大小
        label='Single Run Performance',
        ax=ax1,
        edgecolor=None  # 去掉点的边框，看起来更柔和
    )

    # B. 画插值均值曲线 (Interpolation Predictor) - 使用聚合后的 df_mean
    sns.lineplot(
        data=df_mean,
        x='Unfairness',
        y='Risk',
        marker='o',
        markersize=8,  # 均值点稍微大一点
        label='Interpolation Predictor (Average Performance)',
        ax=ax1,
        color='b',
        linewidth=2  # 线条加粗
    )

    # 2. 画基准点 (FRWB / Wasserstein Barycenter)
    wass_dp_mean = df_points['Wass_DP'].mean()
    wass_loss_mean = df_points['Wass_Loss'].mean()
    ax1.scatter(
        wass_dp_mean, wass_loss_mean,
        color='red', s=150, marker='*', label='FRWB', zorder=5
    )

    # 3. 标注 Lambda=0 和 Lambda=1
    mean_curve = df_plot.groupby('Lambda')[['Unfairness', 'Risk']].mean().reset_index()

    # Start (Lambda=0, Most Fair)
    start_pt = mean_curve.loc[mean_curve['Lambda'] == 0]
    if not start_pt.empty:
        ax1.text(start_pt['Unfairness'].values[0], start_pt['Risk'].values[0],
                 '  λ=0 (Fair)', verticalalignment='bottom', fontweight='bold')

    # End (Lambda=1, Base)
    end_pt = mean_curve.loc[mean_curve['Lambda'] == 1]
    if not end_pt.empty:
        ax1.text(end_pt['Unfairness'].values[0], end_pt['Risk'].values[0],
                 '  λ=1 (Base)', verticalalignment='top', fontweight='bold')

    # 设置图 1 的标签
    ax1.set_xlabel('Empirical Unfairness', fontsize=12)
    ax1.set_ylabel('Empirical Risk', fontsize=12)
    ax1.legend(fontsize=12)

    plt.tight_layout()
    plt.show()  # 显示第一张图

    # ==========================================
    # 图 2: Lambda Scaling (Metrics vs Lambda)
    # ==========================================
    fig2, ax2 = plt.subplots(figsize=(8, 6))  # 创建第二个独立的画布

    # 1. 画 Unfairness (左轴, 橙色)



    # 2. 画 Risk (右轴, 绿色)
 # 创建共享X轴的双Y轴
    sns.lineplot(
        data=df_plot,
        x='Lambda',
        y='Risk',
        ax=ax2,
        label='Empirical Risk',
        color='green',
        marker='^'
    )
    ax2.set_ylabel('Empirical Risk', color='green', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='green')





    ax3 = ax2.twinx()
    sns.lineplot(
        data=df_plot,
        x='Lambda',
        y='Unfairness',
        ax=ax3,
        label='Empirical Unfairness',
        color='orange',
        marker='s'
    )
    ax3.set_xlabel('Interpolation Parameter λ', fontsize=12)
    ax3.set_ylabel('Empirical Unfairness', color='orange', fontsize=12)
    ax3.tick_params(axis='y', labelcolor='orange')

    # 设置图 2 的标题

    # 合并图例 (左轴和右轴的图例合并显示)
    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax3.get_legend_handles_labels()
    ax3.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right', fontsize=12)

    # 移除 ax2 原有的图例防止重复
    if ax2.get_legend():
        ax2.get_legend().remove()

    plt.tight_layout()
    plt.show()  # 显示第二张图


import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns


def plot_enhanced_comparison(df_plot, df_points):
    sns.set_style("whitegrid")

    # --- 关键修改：直接克隆并重构数据标签 ---
    df_viz = df_plot.copy()
    # 将内部 ID 映射为最终想要在图例中显示的字符串
    method_map = {
        'kernel': 'Kernel-based CDF',
        'empirical': 'Empirical CDF'
    }
    df_viz['Estimation Method'] = df_viz['F_Estimator'].map(method_map)

    # 定义对应颜色
    palette = {'Kernel-based CDF': 'blue', 'Empirical CDF': 'orange'}
    # 定义顺序，确保 Kernel 在前
    hue_order = ['Kernel-based CDF', 'Empirical CDF']

    # ==========================================
    # 画布 1: Empirical Risk
    # ==========================================
    plt.figure(figsize=(9, 6))
    sns.lineplot(
        data=df_viz, x='Lambda', y='Risk',
        hue='Estimation Method',  # 使用重命名后的列
        hue_order=hue_order,
        style='Estimation Method',
        palette=palette,
        linewidth=2.5,
        marker='o', markersize=8,
        errorbar='sd'
    )
    plt.title('Empirical Risk vs $\lambda$', fontsize=14, fontweight='bold')
    plt.xlabel('Interpolation Parameter $\lambda$', fontsize=12)
    plt.ylabel('Empirical Risk', fontsize=12)
    # 此时只需调用 plt.legend()，标题会自动识别，不需要手动传 labels
    plt.legend(title='Estimation Method', frameon=True)
    plt.tight_layout()
    plt.show()

    # ==========================================
    # 画布 2: Empirical Unfairness (同理修改)
    # ==========================================
    plt.figure(figsize=(9, 6))
    sns.lineplot(
        data=df_viz, x='Lambda', y='Unfairness',
        hue='Estimation Method',
        hue_order=hue_order,
        style='Estimation Method',
        palette=palette,
        linewidth=2.5,
        marker='s', markersize=8,
        errorbar='sd'
    )
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    plt.title('Empirical Unfairness vs $\lambda$', fontsize=14, fontweight='bold')
    plt.xlabel('Interpolation Parameter $\lambda$', fontsize=12)
    plt.ylabel('Empirical Unfairness', fontsize=12)
    plt.legend(title='Estimation Method', frameon=True)
    plt.tight_layout()
    plt.show()