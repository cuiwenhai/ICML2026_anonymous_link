# cwh
# 2026/3/26
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
    S = np.random.randint(25, 45, size=n_samples)
    # 3. 定义真实关系 (Ground Truth)
    # 注意：这里 S 的系数是 1.0，代表真实的差别待遇 (Direct Discrimination)
    # 如果你想测试"去除偏差"，这个 1.0 是模型应该拟合的还是应该去除的，取决于你的公平性定义。
    # 通常在 DP 任务中，我们希望去除 S 带来的所有影响。
    true_coef = np.array([1.5, -0.5, 0.2, 0.0, 0.0])
    # Y_clean = X @ true_coef + S
    # 基础模型 (不含噪声)
    bias = 2 * ((S > 35) & (S <= 40)).astype(float) + 4 * ((S > 40) & (S <= 45)).astype(float)
    Y_clean = X @ true_coef +0.1*(S-35)

    # =======================================================
    # 4. 噪声注入与数据污染 (核心修改)
    # =======================================================

    # 初始化 Y
    Y = Y_clean.copy()
    noise = np.random.normal(0, 1, size=n_samples)

    Y = Y + noise


    return X, S, Y

# ==========================================
# 5. 整合与运行 (Runner)
# ==========================================
def run_simulation_pipeline(X_train, S_train, Y_train, X_test, S_test, Y_test, loss_type,f_method='kernel', **loss_kwargs):
    if f_method == 'kernel':
        f_func = compute_kernel_conditional_u
    else:
        f_func = compute_conditional_u



    # -------------------------------------------------
    # 2. 预处理
    # -------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 将 S 加入输入特征，确保 f* 能学到所有相关性 (Risk 最小化)
    X_train_in = np.hstack([X_train_scaled, S_train.reshape(-1, 1)])
    X_test_in = np.hstack([X_test_scaled, S_test.reshape(-1, 1)])

    # -------------------------------------------------
    # 3. 训练 Base Predictor f* (Step 1)
    # -------------------------------------------------
    torch_criterion = get_torch_criterion(loss_type, **loss_kwargs)

    predict_fn = train_model_cv(
        LADRegressor,
        X_train_in,
        Y_train,
        criterion=torch_criterion,
        k_folds=5, lr=1e-3, epochs=2000, patience=3, batch_size=250
    )

    # 获取 f* 的预测值
    # y_pred_test 对应理论中的 f*(x, s)
    y_pred_train = predict_fn(X_train_in)
    y_pred_test = predict_fn(X_test_in)

    # -------------------------------------------------
    # 4. 学习公平变换 Q* (Step 2)
    # -------------------------------------------------
    # 这里我们学习的是从 f* 到 g*_{Q^*} 的映射
    trans_params, d, k = optimize_fair_transform(
        y_pred_train, S_train, Y_train,
        degree=3, n_knots=6, compute_conditional_F=f_func,
        loss_type=loss_type, **loss_kwargs
    )

    # -------------------------------------------------
    # 5. 获取完全公平预测器 g*_{Q^*}
    # -------------------------------------------------
    # y_final_test 对应理论中的 g*_{Q^*}(x, s)
    y_final_test = apply_transform(y_pred_test, S_test, trans_params, d, k, compute_conditional_F=f_func)

    # -------------------------------------------------
    # 6. Geodesic Interpolation (核心新增部分)
    # 公式: g_lambda = lambda * f* + (1 - lambda) * g*_{Q^*}
    # -------------------------------------------------

    # 定义 lambda 列表 (从 0 到 1)
    # lambda=1 -> f* (不公平，准确)
    # lambda=0 -> g*_{Q^*} (完全公平)
    lambda_grid = [0.0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

    interpolation_metrics = []

    for lam in lambda_grid:
        # 执行线性插值
        # 注意：这里是向量加法，非常高效
        y_test_lambda = lam * y_pred_test + (1 - lam) * y_final_test

        # 评估当前 lambda 下的性能
        loss_val, dp_val = evaluate_metrics(
            y_test_lambda, Y_test, S_test, loss_type, **loss_kwargs
        )

        interpolation_metrics.append({
            'lambda': lam,
            'Loss': loss_val,
            'Unfairness': dp_val
        })

    # -------------------------------------------------
    # 7. (可选) 对比方法: Wasserstein Barycenter (Algo 1)
    # -------------------------------------------------
    y_test_wass = method_wasserstein_apply(y_pred_train, S_train, y_pred_test, S_test)
    wass_loss, wass_dp = evaluate_metrics(y_test_wass, Y_test, S_test, loss_type, **loss_kwargs)

    # -------------------------------------------------
    # 8. 汇总结果
    # -------------------------------------------------
    metrics = {
        'Interpolation': interpolation_metrics,  # 包含了一系列的 (Loss, Unfairness) 点
        'Method_Optimization': {  # Lambda=0 的情况
            'Loss': interpolation_metrics[0]['Loss'],
            'DP': interpolation_metrics[0]['Unfairness']
        },
        'Method_Base': {  # Lambda=1 的情况
            'Loss': interpolation_metrics[-1]['Loss'],
            'DP': interpolation_metrics[-1]['Unfairness']
        },
        'Method_Wasserstein_Direct': {  # 对比方法
            'Loss': wass_loss,
            'DP': wass_dp
        }
    }
   # print(f"Seed {seed} Completed.")
    # 可以选择打印 Lambda=0.5 的中间结果看看
    mid_idx = 0
    #print( metrics)
    #print(f"  Lambda=0.5 -> Loss: {interpolation_metrics[mid_idx]['Loss']:.4f}, DP: {interpolation_metrics[mid_idx]['Unfairness']:.4f}")

    return metrics



#
# ==========================================
# 6. 执行入口
# ==========================================
# ==========================================
# 5. 整合与运行 (整合了数据分发与统计报告)
# ==========================================

def run_simulation_pipeline(X_train, S_train, Y_train, X_test, S_test, Y_test,
                            loss_type, f_method='kernel', **loss_kwargs):
    """
    修改版 Pipeline：接收外部传入的数据，确保对比公平性
    """
    # 根据 f_method 选择估计函数
    f_func = compute_kernel_conditional_u if f_method == 'kernel' else compute_conditional_u

    # -------------------------------------------------
    # 2. 预处理
    # -------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_in = np.hstack([X_train_scaled, S_train.reshape(-1, 1)])
    X_test_in = np.hstack([X_test_scaled, S_test.reshape(-1, 1)])

    # -------------------------------------------------
    # 3. 训练 Base Predictor f*
    # -------------------------------------------------
    torch_criterion = get_torch_criterion(loss_type, **loss_kwargs)
    predict_fn = train_model_cv(
        LADRegressor, X_train_in, Y_train,
        criterion=torch_criterion, k_folds=5, epochs=2000  # 减少轮数加速演示
    )

    y_pred_train = predict_fn(X_train_in)
    y_pred_test = predict_fn(X_test_in)

    # -------------------------------------------------
    # 4. 学习公平变换与插值
    # -------------------------------------------------
    trans_params, d, k = optimize_fair_transform(
        y_pred_train, S_train, Y_train,
        degree=3, n_knots=6, compute_conditional_F=f_func,
        loss_type=loss_type, **loss_kwargs
    )

    y_final_test = apply_transform(y_pred_test, S_test, trans_params, d, k, compute_conditional_F=f_func)

    lambda_grid = np.linspace(0, 1, 11)
    interpolation_metrics = []

    for lam in lambda_grid:
        y_test_lambda = lam * y_pred_test + (1 - lam) * y_final_test
        loss_val, dp_val = evaluate_metrics(y_test_lambda, Y_test, S_test, loss_type, **loss_kwargs)
        interpolation_metrics.append({'lambda': lam, 'Loss': loss_val, 'Unfairness': dp_val})

    # 对比方法: Wasserstein Barycenter
    y_test_wass = method_wasserstein_apply(y_pred_train, S_train, y_pred_test, S_test)
    wass_loss, wass_dp = evaluate_metrics(y_test_wass, Y_test, S_test, loss_type, **loss_kwargs)

    return {
        'Interpolation': interpolation_metrics,
        'Method_Wasserstein_Direct': {'Loss': wass_loss, 'DP': wass_dp}
    }


def run_comparison_study(n_repeats=100, loss_type='Huber', **loss_kwargs):
    all_plots = []
    all_points = []

    # 蒙特卡洛循环 (逻辑保持你之前的：生成同一套数据给两种方法)
    for i in range(n_repeats):
        X_tr, S_tr, Y_tr = generate_data(n_samples=1000, seed=i)
        X_te, S_te, Y_te = generate_data(n_samples=1000, seed=i + 1000)

        print(f"repeat time{i}")

        for method in ['kernel', 'empirical']:
            res = run_simulation_pipeline(X_tr, S_tr, Y_tr, X_te, S_te, Y_te,
                                          loss_type=loss_type, f_method=method, **loss_kwargs)
            for point in res['Interpolation']:
                all_plots.append({'Run_ID': i, 'Lambda': point['lambda'], 'Risk': point['Loss'],
                                  'Unfairness': point['Unfairness'], 'F_Estimator': method})
            all_points.append({'Run_ID': i, 'Wass_Loss': res['Method_Wasserstein_Direct']['Loss'],
                               'Wass_DP': res['Method_Wasserstein_Direct']['DP'], 'F_Estimator': method})

    df_plot = pd.DataFrame(all_plots)
    df_points = pd.DataFrame(all_points)

    # --- [新增] 统计摘要报道 ---
    print("\n" + "="*60)
    print(f"📊 STATISTICS REPORT (Repeats: {n_repeats})")
    print("="*60)

    # 重点查看 Lambda = 0 (完全公平点) 和 Lambda = 1 (原始预测)
    for lam in [0.0, 1.0]:
        print(f"\n>>> Results at Lambda = {lam}:")
        subset = df_plot[df_plot['Lambda'] == lam]
        summary = subset.groupby('F_Estimator')[['Risk', 'Unfairness']].agg(['mean', 'std'])
        print(summary)

    # 查看基准方法表现
    print("\n>>> Baseline (FRWB) Results:")
    wass_summary = df_points.groupby('F_Estimator')[['Wass_Loss', 'Wass_DP']].agg(['mean', 'std'])
    print(wass_summary)

    # 执行绘图
    plot_enhanced_comparison(df_plot, df_points)

    return df_plot, df_points



# 在你的主程序中显式指定
if __name__ == "__main__":
    # 1. 对比实验
    df_results = run_comparison_study(n_repeats=10, loss_type='Huber', zeta=1.345)



