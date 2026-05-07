import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from Tool import *
from scipy.stats import skewnorm


# ==========================================
# 辅助函数: 数据生成
# ==========================================
def generate_data(n_samples=1000, seed=None):
    """
    生成模拟数据，用于 Robust Fairness 研究。
    包含 W 作为合法的控制变量 (Conditional Demographic Parity 的基础)。
    """
    if seed is not None:
        np.random.seed(seed)

    # 1. 生成特征 X (N, 5)
    X = np.random.normal(0, 1, size=(n_samples, 5))

    # W is correlated with S but is considered a valid predictor [cite: 603]
    W = np.random.choice([0, 1], size=n_samples)

    # 2. 生成敏感属性 S (S 依赖于 X 和 W)
    gamma = np.array([1.0, -1.0, 0.5, 0.0, 0.0])
    logits = X @ gamma + 0.5 * W
    r = 1 / (1 + np.exp(-logits))
    S = np.random.binomial(1, r, size=n_samples)

    # 3. 定义真实关系 (Ground Truth)
    true_coef = np.array([1.5, -0.5, 0.2, 0.0, 0.0])

    # 基础模型 (不含噪声)
    # 假设 W 也是一个合理的预测因子，权重为 0.8
    Y_clean = X @ true_coef + 1.0 * S + 0.8 * W

    # =======================================================
    # 4. 噪声注入与数据污染 
    # =======================================================
    Y = Y_clean.copy()
    noise = np.random.normal(0, 1, size=n_samples)
    Y = Y + noise * (2 * S + 0.5)

    return X, S, W, Y


# ==========================================
# 辅助函数: CDP Unfairness 评估
# ==========================================
def evaluate_cdp_metrics(y_pred, Y_true, S, W, loss_type, **loss_kwargs):
    """
    计算整体的 Risk 和 CDP 不公平性。
    CDP 不公平性定义为所有 W 层级中，最大的组间 Wasserstein 距离。
    """
    # 1. 整体风险 (Risk) 复用原有的全局评估
    global_loss, _ = evaluate_metrics(y_pred, Y_true, S, loss_type, **loss_kwargs)

    # 2. 分层计算 Unfairness
    cdp_unfairness = 0.0
    strata = np.unique(W)
    for w in strata:
        idx = (W == w)
        if np.sum(idx) > 0:  # 确保该层有数据
            _, dp_w = evaluate_metrics(y_pred[idx], Y_true[idx], S[idx], loss_type, **loss_kwargs)
            # 取各层级中最大的 Unfairness 作为整体 CDP Unfairness
            cdp_unfairness = max(cdp_unfairness, dp_w)

    return global_loss, cdp_unfairness


# ==========================================
# 5. 整合与运行 (Runner)
# ==========================================
def run_simulation_pipeline(seed, loss_type, **loss_kwargs):
    # -------------------------------------------------
    # 1. 数据生成
    # -------------------------------------------------
    X_train, S_train, W_train, Y_train = generate_data(n_samples=1000, seed=seed)
    X_test, S_test, W_test, Y_test = generate_data(n_samples=1000, seed=seed)

    # -------------------------------------------------
    # 2. 预处理
    # -------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 将 S 和 W 都加入输入特征，确保 f* 能学到所有相关性
    X_train_in = np.column_stack([X_train_scaled, S_train, W_train])
    X_test_in = np.column_stack([X_test_scaled, S_test, W_test])

    # -------------------------------------------------
    # 3. 训练 Base Predictor f* (Step 1)
    # -------------------------------------------------
    torch_criterion = get_torch_criterion(loss_type, **loss_kwargs)

    predict_fn = train_model_cv(
        LADRegressor,
        X_train_in,
        Y_train,
        criterion=torch_criterion,
        k_folds=5, lr=1e-3, epochs=5000, patience=20, batch_size=100
    )

    y_pred_train = predict_fn(X_train_in)
    y_pred_test = predict_fn(X_test_in)

    # -------------------------------------------------
    # 4A. 学习全局公平变换 Q* (传统 DP) 
    # -------------------------------------------------
    trans_params_dp, d_dp, k_dp = optimize_fair_transform(
        y_pred_train, S_train, Y_train,
        degree=3, n_knots=6, loss_type=loss_type, **loss_kwargs
    )
    y_final_test_dp = apply_transform(y_pred_test, S_test, trans_params_dp, d_dp, k_dp)

    # -------------------------------------------------
    # 4B. 学习分层公平变换 Q_w* (CDP) [cite: 611, 612]
    # -------------------------------------------------
    y_final_test_cdp = np.zeros_like(y_pred_test)
    strata = np.unique(W_train)

    for w in strata:
        idx_train = (W_train == w)
        idx_test = (W_test == w)

        if np.sum(idx_train) > 0 and np.sum(idx_test) > 0:
            # 针对 W=w 这个子群体学习专用的 Q_w* [cite: 676]
            params_w, d_w, k_w = optimize_fair_transform(
                y_pred_train[idx_train], S_train[idx_train], Y_train[idx_train],
                degree=3, n_knots=6, loss_type=loss_type, **loss_kwargs
            )
            # 应用变换到测试集的对应层级 [cite: 678]
            y_final_test_cdp[idx_test] = apply_transform(
                y_pred_test[idx_test], S_test[idx_test], params_w, d_w, k_w
            )

    # -------------------------------------------------
    # 6. Geodesic Interpolation (插值对比 DP vs CDP)
    # -------------------------------------------------
    lambda_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    interpolation_metrics = []

    for lam in lambda_grid:
        # A. 记录 DP 性能 (评估使用的是全局 DP Unfairness)
        y_test_dp_lam = lam * y_pred_test + (1 - lam) * y_final_test_dp
        loss_dp, dp_val = evaluate_metrics(y_test_dp_lam, Y_test, S_test, loss_type, **loss_kwargs)

        interpolation_metrics.append({
            'lambda': lam,
            'Loss': loss_dp,
            'Unfairness': dp_val,
            'Method': 'Standard DP'
        })

        # B. 记录 CDP 性能 (评估使用的是分层 CDP Unfairness)
        y_test_cdp_lam = lam * y_pred_test + (1 - lam) * y_final_test_cdp
        loss_cdp, cdp_val = evaluate_cdp_metrics(y_test_cdp_lam, Y_test, S_test, W_test, loss_type, **loss_kwargs)

        interpolation_metrics.append({
            'lambda': lam,
            'Loss': loss_cdp,
            'Unfairness': cdp_val,
            'Method': 'CDP (Ours)'
        })

    # -------------------------------------------------
    # 7. 对比方法: Wasserstein Barycenter (Algo 1)
    # -------------------------------------------------
    y_test_wass = method_wasserstein_apply(y_pred_train, S_train, y_pred_test, S_test)
    wass_loss, wass_dp = evaluate_metrics(y_test_wass, Y_test, S_test, loss_type, **loss_kwargs)

    # 汇总
    metrics = {
        'Interpolation': interpolation_metrics,
        'Method_Wasserstein_Direct': {
            'Loss': wass_loss,
            'DP': wass_dp
        }
    }
    return metrics


# ==========================================
# 5. 蒙特卡洛实验 (多次运行)
# ==========================================
def run_monte_carlo_experiment(n_repeats, loss_type, **loss_kwargs):
    print(f"\n🚀 Starting Monte Carlo Simulation ({n_repeats} runs)...")
    param_str = ", ".join([f"{k}={v}" for k, v in loss_kwargs.items()])
    print(f"⚙️  Settings: Loss={loss_type} | Params=[{param_str}]")

    raw_results = []

    for i in range(n_repeats):
        try:
            res = run_simulation_pipeline(seed=i, loss_type=loss_type, **loss_kwargs)
            raw_results.append(res)
        except Exception as e:
            print(f"⚠️ Run {i} failed: {e}")
            import traceback
            traceback.print_exc()

        print(f"   -> Run {i + 1}/{n_repeats} done.")

    if not raw_results:
        print("❌ No results collected.")
        return None, None

    # 提取画图数据
    plot_data = []
    point_metrics_list = []

    for run_id, res in enumerate(raw_results):
        for point in res['Interpolation']:
            plot_data.append({
                'Run_ID': run_id,
                'Lambda': point['lambda'],
                'Risk': point['Loss'],
                'Unfairness': point['Unfairness'],
                'Method': point['Method']
            })

        point_metrics_list.append({
            'Wass_Loss': res['Method_Wasserstein_Direct']['Loss'],
            'Wass_DP': res['Method_Wasserstein_Direct']['DP']
        })

    df_plot = pd.DataFrame(plot_data)
    df_points = pd.DataFrame(point_metrics_list)

    print("\n✅ Simulation complete. Generating plots to compare DP and CDP...")
    plot_tradeoff_curves(df_plot, df_points)

    return df_points, df_plot


# ==========================================
# 6. 画图函数 (Visualization)
# ==========================================
def plot_tradeoff_curves(df_plot, df_points):
    """
    在同一张图中绘制 DP 和 CDP 的公平性-准确性权衡曲线对比
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 7))
    ax1 = plt.gca()

    # 先求均值，保证连线平滑
    df_mean = df_plot.groupby(['Method', 'Lambda'])[['Risk', 'Unfairness']].mean().reset_index()

    # 1. 绘制 DP 曲线
    df_dp = df_mean[df_mean['Method'] == 'Standard DP'].sort_values('Unfairness')
    sns.lineplot(
        data=df_dp, x='Unfairness', y='Risk',
        marker='o', markersize=8, color='b', linewidth=2,
        label='Standard DP', ax=ax1
    )

    # 2. 绘制 CDP 曲线
    df_cdp = df_mean[df_mean['Method'] == 'CDP (Ours)'].sort_values('Unfairness')
    sns.lineplot(
        data=df_cdp, x='Unfairness', y='Risk',
        marker='s', markersize=8, color='g', linewidth=2, linestyle='--',
        label='CDP', ax=ax1
    )

    # 3. 绘制背景散点 (可根据需要注释掉以免太杂乱)
    sns.scatterplot(
        data=df_plot, x='Unfairness', y='Risk', hue='Method',
        palette={'Standard DP': 'blue', 'CDP (Ours)': 'green'},
        alpha=0.15, s=30, ax=ax1, legend=False
    )

    # 4. 绘制基准点 (FRWB)
    wass_dp_mean = df_points['Wass_DP'].mean()
    wass_loss_mean = df_points['Wass_Loss'].mean()
    ax1.scatter(
        wass_dp_mean, wass_loss_mean,
        color='red', s=150, marker='*', label='FRWB Benchmark', zorder=5
    )

    # 标签与展示
    ax1.set_xlabel('Empirical Conditional Unfairness', fontsize=12)
    ax1.set_ylabel('Empirical Risk', fontsize=12)
    ax1.set_title("Pareto Trade-off", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)

    plt.tight_layout()
    plt.show()


# ==========================================
# 7. 执行入口
# ==========================================
if __name__ == "__main__":
    run_monte_carlo_experiment(n_repeats=200, loss_type='Huber', zeta=1.345)