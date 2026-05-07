# cwh
# 2026/3/28
# cwh
# 2025/12/8
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Tool import  *
from scipy.stats import skewnorm
# ==========================================
# 辅助函数: 数据生成
# ==========================================
def generate_data(n_samples=1000, seed=None):
    """
    生成模拟数据，用于 Robust Fairness 研究。

    参数:
    ----------
    n_samples : int
        样本数量。
    setting : str
        数据生成模式:
        - 'clean': 标准正态噪声 (Gaussian)。
        - 'heavy_tail': t分布噪声 (df=1.5)，模拟厚尾，LAD 的主场。
        - 'outliers': 在 Y 中加入随机的大离群值 (Gross Errors)。
        - 'group_bias': 仅针对敏感群体 (S=1) 施加污染，模拟系统性数据偏差。
    contamination_rate : float
        在 'outliers' 或 'group_bias' 模式下的污染比例。
    seed : int
        随机种子。

    返回:
    ----------
    X : (n, 5) 特征矩阵
    S : (n,) 敏感属性 (0 or 1)
    Y : (n,) 响应变量
    """
    if seed is not None:
        np.random.seed(seed)

    # 1. 生成特征 X (N, 5)
    X = np.random.normal(0, 1, size=(n_samples, 5))

    # 2. 生成敏感属性 S (S 依赖于 X，导致 Fairness 隐患)
    gamma = np.array([1.0, -1.0, 0.5, 0.0, 0.0])
    logits = X @ gamma
    r = 1 / (1 + np.exp(-logits))
    S = np.random.binomial(1, r, size=n_samples)

    # 3. 定义真实关系 (Ground Truth)
    # 注意：这里 S 的系数是 1.0，代表真实的差别待遇 (Direct Discrimination)
    # 如果你想测试"去除偏差"，这个 1.0 是模型应该拟合的还是应该去除的，取决于你的公平性定义。
    # 通常在 DP 任务中，我们希望去除 S 带来的所有影响。
    true_coef = np.array([1.5, -0.5, 0.2, 0.0, 0.0])

    # 基础模型 (不含噪声)
    Y_clean = X @ true_coef + 1.0 * S

    # =======================================================
    # 4. 噪声注入与数据污染 (核心修改)
    # =======================================================

    # 初始化 Y
    Y = Y_clean.copy()
    noise  = np.random.standard_t(df=1.5, size=n_samples)

    Y = Y + noise * (2*S+0.5)


    return X, S, Y


def run_simulation_pipeline_single(seed, loss_type, lambda_nat=1e-3, **loss_kwargs):

    # -------------------------------------------------
    # 1-3. 数据生成、预处理与模型训练 (保持你的逻辑不变)
    # -------------------------------------------------
    X_train, S_train, Y_train = generate_data(n_samples=1000, seed=seed)
    X_test, S_test, Y_test = generate_data(n_samples=1000, seed=seed)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_in = np.hstack([X_train_scaled, S_train.reshape(-1, 1)])
    X_test_in = np.hstack([X_test_scaled, S_test.reshape(-1, 1)])

    torch_criterion = get_torch_criterion(loss_type, **loss_kwargs)
    predict_fn = train_model_cv(
        LADRegressor, X_train_in, Y_train,
        criterion=torch_criterion,
        k_folds=5, lr=1e-3, epochs=5000, patience=10, batch_size=100
    )

    y_pred_train = predict_fn(X_train_in)
    y_pred_test = predict_fn(X_test_in)

    # 准备绘图数据：生成一个超范围的 U 序列
    U_extrapolate = np.linspace(-0.5, 1.5, 500)
    plt.figure(figsize=(10, 6))

    iteration_data = []
    methods_config = [
        {'name': 'Standard I-Spline', 'natural': False, 'color': 'blue'},
        {'name': 'Natural I-Spline', 'natural': True, 'color': 'red'}
    ]

    for config in methods_config:
        # 4. 训练模型
        params, d, k = optimize_fair_transform(
            y_pred_train, S_train, Y_train,
            degree=3, n_knots=6,
            natural=config['natural'], lambda_nat=lambda_nat,
            loss_type=loss_type, **loss_kwargs
        )

        # --- 核心修改：计算超边界外推曲线 ---
        # 直接使用 U_extrapolate 生成基矩阵
        res_basis = get_ispline_basis(U_extrapolate, d, k, natural=False)
        Basis_ext = res_basis[0] if isinstance(res_basis, tuple) else res_basis

        alpha_0 = params[0]
        alpha_rest = params[1:]
        y_extrapolate = alpha_0 + Basis_ext @ alpha_rest

        # 绘制曲线
        plt.plot(U_extrapolate, y_extrapolate, label=config['name'], color=config['color'], linewidth=2)

        # --- (后续原有的评估逻辑保持不变) ---
        y_final_test = apply_transform(y_pred_test, S_test, params, d, k)
        lambda_grid = np.linspace(0, 1, 11)
        for lam in lambda_grid:
            y_test_lambda = lam * y_pred_test + (1 - lam) * y_final_test
            loss_val, dp_val = evaluate_metrics(y_test_lambda, Y_test, S_test, loss_type, **loss_kwargs)
            iteration_data.append({'lambda': lam, 'Risk': loss_val, 'Unfairness': dp_val, 'Method': config['name']})

    # --- 完善绘图细节 ---
    plt.axvspan(0, 1, color='gray', alpha=0.1, label='Training Region [0, 1]')
    plt.axvline(0, color='black', linestyle='--', alpha=0.5)
    plt.axvline(1, color='black', linestyle='--', alpha=0.5)

    plt.xlabel("$u$", fontsize=10)
    plt.ylabel("$\widehat{Q}(u)$", fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.05, 1.05)
    plt.show()

    # --- 5. Wass 对比 (保持不变) ---
    y_test_wass = method_wasserstein_apply(y_pred_train, S_train, y_pred_test, S_test)
    wass_loss, wass_dp = evaluate_metrics(y_test_wass, Y_test, S_test, loss_type, **loss_kwargs)
    iteration_data.append({'lambda': 0.0, 'Risk': wass_loss, 'Unfairness': wass_dp, 'Method': 'FRWB (Baseline)'})

    return iteration_data



# ==========================================
# 5. 整合与运行 (Runner)
# ==========================================
def run_simulation_pipeline(seed, loss_type, lambda_nat=1e-3, **loss_kwargs):
    """
    运行完整的 Pipeline，并在同一组 f* 上对比 Standard, Natural 和 Wass 方法。
    返回一个包含所有观测点的列表，每个点都带有 'Method' 标签。
    """
    # -------------------------------------------------
    # 1-3. 数据生成、预处理与模型训练 (保持你的逻辑不变)
    # -------------------------------------------------
    X_train, S_train, Y_train = generate_data(n_samples=1000, seed=seed)
    X_test, S_test, Y_test = generate_data(n_samples=1000, seed=seed)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_in = np.hstack([X_train_scaled, S_train.reshape(-1, 1)])
    X_test_in = np.hstack([X_test_scaled, S_test.reshape(-1, 1)])

    torch_criterion = get_torch_criterion(loss_type, **loss_kwargs)
    predict_fn = train_model_cv(
        LADRegressor, X_train_in, Y_train,
        criterion=torch_criterion,
        k_folds=5, lr=1e-3, epochs=5000, patience=10, batch_size=100
    )

    y_pred_train = predict_fn(X_train_in)
    y_pred_test = predict_fn(X_test_in)

    # 存储本轮实验所有结果的列表
    iteration_data = []

    # -------------------------------------------------
    # 4. 学习公平变换 (对比分支)
    # -------------------------------------------------
    methods_config = [
        {'name': 'Standard I-Spline', 'natural': False},
        {'name': 'Natural I-Spline', 'natural': True}
    ]

    for config in methods_config:
        params, d, k = optimize_fair_transform(
            y_pred_train, S_train, Y_train,
            degree=3, n_knots=6,
            natural=config['natural'], lambda_nat=lambda_nat,
            loss_type=loss_type, **loss_kwargs
        )

        y_final_test = apply_transform(y_pred_test, S_test, params, d, k)

        # 插值曲线计算
        lambda_grid = np.linspace(0, 1, 11)
        for lam in lambda_grid:
            y_test_lambda = lam * y_pred_test + (1 - lam) * y_final_test
            loss_val, dp_val = evaluate_metrics(y_test_lambda, Y_test, S_test, loss_type, **loss_kwargs)

            # 直接在这里封装成最终格式
            iteration_data.append({
                'lambda': lam,
                'Risk': loss_val,
                'Unfairness': dp_val,
                'Method': config['name']
            })

    # -------------------------------------------------
    # 5. 对比方法: Wasserstein Barycenter (Wass)
    # -------------------------------------------------
    y_test_wass = method_wasserstein_apply(y_pred_train, S_train, y_pred_test, S_test)
    wass_loss, wass_dp = evaluate_metrics(y_test_wass, Y_test, S_test, loss_type, **loss_kwargs)

    # Wass 通常只有单一结果点，可以标记为 lambda=0 或特殊的标识
    iteration_data.append({
        'lambda': 0.0,
        'Risk': wass_loss,
        'Unfairness': wass_dp,
        'Method': 'FRWB (Baseline)'
    })

    return iteration_data

# ==========================================
# 5. 蒙特卡洛实验 (多次运行)
# ==========================================
def run_monte_carlo_experiment(n_repeats, loss_type, lambda_nat=1e-3, **loss_kwargs):
    print(f"\n🚀 Starting Comparison Monte Carlo Simulation ({n_repeats} runs)...")
    all_plot_data = []

    for i in range(n_repeats):
        try:
            # 获取已经打好标签的列表
            iteration_results = run_simulation_pipeline(
                seed=i, loss_type=loss_type, lambda_nat=lambda_nat, **loss_kwargs
            )

            # 为本轮所有数据添加 Run_ID 标识
            for entry in iteration_results:
                entry['Run_ID'] = i

            all_plot_data.extend(iteration_results)
            print(f"   -> Run {i + 1}/{n_repeats} done.")

        except Exception as e:
            print(f"⚠️ Run {i} failed: {e}")
            import traceback
            traceback.print_exc()

    df_plot = pd.DataFrame(all_plot_data)

    # 打印统计摘要
    print("\n" + "=" * 65)
    print(f"📊 STATISTICAL SUMMARY (Lambda=0, N={n_repeats})")
    print("=" * 65)

    # 提取所有方法在 lambda=0 时的表现
    summary = df_plot[np.isclose(df_plot['lambda'], 0)].groupby('Method')[['Risk', 'Unfairness']].agg(['mean', 'std'])
    print(summary)

    # 绘图


    return df_plot

# 执行
if __name__ == "__main__":
    # 示例：运行 5 次 Quantile 回归对比
    df_results = run_monte_carlo_experiment(
        n_repeats=100,
        loss_type='Huber', zeta=1.345,
        lambda_nat=0.1
    )


    # results = run_simulation_pipeline_single(
    #     seed=42,
    #     loss_type='Huber', zeta=1.345,
    #     lambda_nat=0.1
    # )
    #
    # # 4. 打印返回的部分结果（例如完全公平点 lambda=0 的表现）
    # print("\n单次实验完全公平点 (lambda=0) 结果摘要:")
    # for entry in results:
    #     if entry['lambda'] == 0:
    #         print(f"方法: {entry['Method']:<20} | Risk: {entry['Risk']:.4f} | Unfairness: {entry['Unfairness']:.4f}")
    #
    #





#loss_type='Quantile', tau=0.75



# loss_type='LAD'
#loss_type='Huber', zeta=1.345
#loss_type='Cauchy', kappa=1
# loss_type='Tukey', c=4.685

