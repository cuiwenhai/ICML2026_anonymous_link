# cwh
# 2025/12/8
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from sklearn.linear_model import LinearRegression  # 新增: 用于极速生成 y_pred 隔离耗时
from scipy.stats import skewnorm
from Tool import * # 确保你的 Tool.py 在同级目录下
# ==========================================
# 辅助函数: 数据生成
# ==========================================
def generate_data(n_samples=1000, number_S=2,seed=None):
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
    S = np.random.randint(0, number_S, size=n_samples)

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
    noise = np.random.normal(0, 1, size=n_samples)

    Y = Y + noise * (2*S+0.5)


    return X, S, Y

# ==========================================
# 5. 整合与运行 (Runner)
# ==========================================
def run_simulation_pipeline(seed, loss_type, **loss_kwargs):
    # -------------------------------------------------
    # 1. 数据生成
    # -------------------------------------------------
    X_train, S_train, Y_train = generate_data(n_samples=1000, seed=seed)
    X_test, S_test, Y_test = generate_data(n_samples=1000, seed=seed)

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
        k_folds=5, lr=1e-3, epochs=5000, patience=20, batch_size=100
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
        degree=3, n_knots=6,
        loss_type=loss_type, **loss_kwargs
    )

    # -------------------------------------------------
    # 5. 获取完全公平预测器 g*_{Q^*}
    # -------------------------------------------------
    # y_final_test 对应理论中的 g*_{Q^*}(x, s)
    y_final_test = apply_transform(y_pred_test, S_test, trans_params, d, k)

    # -------------------------------------------------
    # 6. Geodesic Interpolation (核心新增部分)
    # 公式: g_lambda = lambda * f* + (1 - lambda) * g*_{Q^*}
    # -------------------------------------------------

    # 定义 lambda 列表 (从 0 到 1)
    # lambda=1 -> f* (不公平，准确)
    # lambda=0 -> g*_{Q^*} (完全公平)
    lambda_grid = [0.0]

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



# ==========================================
# 5. 蒙特卡洛实验 (多次运行)
# ==========================================
def run_monte_carlo_experiment(n_repeats, loss_type, **loss_kwargs):
    """
    运行 N 次模拟，打印关键点摘要和运行时间，并返回用于画图的详细数据
    """
    print(f"\n🚀 Starting Monte Carlo Simulation ({n_repeats} runs)...")
    param_str = ", ".join([f"{k}={v}" for k, v in loss_kwargs.items()])
    print(f"⚙️  Settings: Loss={loss_type} | Params=[{param_str}]")

    raw_results = []

    # 记录总实验开始时间
    total_start_time = time.time()

    for i in range(n_repeats):
        # 记录单次 Run 开始时间
        run_start_time = time.time()

        try:
            res = run_simulation_pipeline(
                seed=i,
                loss_type=loss_type,
                **loss_kwargs
            )
            raw_results.append(res)
        except Exception as e:
            print(f"⚠️ Run {i} failed: {e}")
            import traceback
            traceback.print_exc()

        # 记录单次 Run 结束时间并计算耗时
        run_end_time = time.time()
        run_duration = run_end_time - run_start_time

        # 简单进度条 (带耗时显示)
        print(f"   -> Run {i + 1}/{n_repeats} done in {run_duration:.2f} seconds.")

    # 记录总实验结束时间并计算总耗时
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    if not raw_results:
        print("❌ No results collected.")
        return None, None

    # ==========================================
    # 1. 整理关键点数据 (用于打印摘要)
    # ==========================================
    point_metrics_list = []
    for res in raw_results:
        row = {
            'Init_Loss': res['Method_Base']['Loss'],
            'Init_DP': res['Method_Base']['DP'],
            'Final_Loss': res['Method_Optimization']['Loss'],
            'Final_DP': res['Method_Optimization']['DP'],
            'Wass_Loss': res['Method_Wasserstein_Direct']['Loss'],
            'Wass_DP': res['Method_Wasserstein_Direct']['DP']
        }
        point_metrics_list.append(row)

    df_points = pd.DataFrame(point_metrics_list)
    stats_points = df_points.agg(['mean', 'std'])

    # ==========================================
    # 2. 整理插值曲线数据 (用于画图)
    # ==========================================
    plot_data = []
    for run_id, res in enumerate(raw_results):
        for point in res['Interpolation']:
            plot_data.append({
                'Run_ID': run_id,
                'Lambda': point['lambda'],
                'Risk': point['Loss'],
                'Unfairness': point['Unfairness']
            })

    df_plot = pd.DataFrame(plot_data)

    # ==========================================
    # 3. 打印关键点摘要 (Text Output)
    # ==========================================
    print("\n" + "=" * 60)
    print(f"📊 MONTE CARLO SUMMARY (N={n_repeats})")
    print(f"   Loss Function: {loss_type} {loss_kwargs}")

    # 打印总运行时间 (格式化为 分:秒)
    minutes, seconds = divmod(total_duration, 60)
    print(f"⏱️  Total Time: {int(minutes)}m {seconds:.2f}s (Avg: {total_duration / n_repeats:.2f}s/run)")
    print("=" * 60)

    print(f"{'Metric':<20} | {'Mean':<10} | {'Std Dev':<10}")
    print("-" * 46)

    keys_to_print = ['Init_Loss', 'Final_Loss', 'Wass_Loss', 'Init_DP', 'Final_DP', 'Wass_DP']
    for m in keys_to_print:
        mean_val = stats_points.loc['mean', m]
        std_val = stats_points.loc['std', m]
        print(f"{m:<20} | {mean_val:.4f} ({std_val:.4f})")

    print("-" * 46)
    print("✅ Interpolation details hidden. Generating plots...")

    plot_tradeoff_curves(df_plot, df_points)

    return df_points, df_plot


# ==========================================
# 7. 新增: 专门测试样条优化步骤 Scalability 的 Benchmark
# ==========================================
# ==========================================
# 7. 新增: 二维嵌套测试样条优化步骤 Scalability 的 Benchmark
# ==========================================
def run_optimization_scalability_benchmark(loss_type='Huber', n_repeats=5, **loss_kwargs):
    """
    n_repeats: 每个格子重复运行次数，用于计算 mean 和 std
    """
    print(f"\n🚀 Starting 2D Scalability Benchmark (Repeats per cell: {n_repeats})")

    results = []
    base_degree = 3
    test_N_list = [500, 1000, 2000, 5000, 10000, 20000]
    test_knots_list = [3, 6, 9, 12, 15, 20, 25]

    for N in test_N_list:
        print(f"\n⏳ [Testing N={N:<6}] Generating data...")
        X, S, Y = generate_data(n_samples=N, seed=42)
        X_in = np.hstack([X, S.reshape(-1, 1)])
        fast_lr = LinearRegression().fit(X_in, Y)
        y_pred = fast_lr.predict(X_in)


        knots= test_N_list

        # --- 新增：多次测量以计算 Mean 和 Std ---
        run_times = []
        for r in range(n_repeats):
            start_time = time.time()
            _ = optimize_fair_transform(
                y_pred, S, Y,
                degree=base_degree, n_knots=knots,
                loss_type=loss_type, **loss_kwargs
            )
            run_times.append(time.time() - start_time)

        avg_t = np.mean(run_times)
        std_t = np.std(run_times)

        results.append({
            'N': N,
            'J_n': J_n,
            'Mean': round(avg_t, 4),
            'Std': round(std_t, 4)
        })
        print(f"   -> J_n = {J_n:<4} | Mean = {avg_t:.4f}s | Std = {std_t:.4f}s")

    # ==========================================
    # 数据透视与可视化
    # ==========================================
    df = pd.DataFrame(results)

    # 构造两个透视表：一个存均值，一个存标准差
    pivot_mean = df.pivot(index='N', columns='J_n', values='Mean')
    pivot_std = df.pivot(index='N', columns='J_n', values='Std')

    # 构造一个展示用的“Mean (±Std)”字符串表格
    pivot_display = pivot_mean.copy().astype(str)
    for r in pivot_mean.index:
        for c in pivot_mean.columns:
            m = pivot_mean.loc[r, c]
            s = pivot_std.loc[r, c]
            pivot_display.loc[r, c] = f"{m:.3f} (±{s:.3f})"

    print("\n" + "=" * 80)
    print("✅ 2D BENCHMARK: MEAN RUNTIME (SECONDS) WITH STD DEV")
    print("=" * 80)
    print(pivot_display.to_string())
    print("=" * 80)

    return pivot_mean, pivot_std


def run_optimization_scalability_benchmark_S(loss_type='Huber', n_repeats=5, **loss_kwargs):
    """
    n_repeats: 每个格子重复运行次数，用于计算 mean 和 std
    """
    print(f"\n🚀 Starting 2D Scalability Benchmark: N vs number_S (Repeats: {n_repeats})")

    results = []
    base_degree = 3  # d = 3 (cubic splines)
    test_N_list = [500, 1000, 2000, 5000, 10000, 20000,40000,80000]
    number_S_list = [2, 4, 8, 16, 20, 25, 30]

    for N in test_N_list:
        # ==========================================
        # 核心修改：根据理论最优率动态计算 m_n 和 J_n
        # ==========================================
        m_n = int(np.round(N ** (1.0 / (2 * base_degree + 1))))
        J_n = m_n + base_degree
        print(f"\n⏳ [Testing N={N:<6}] Optimal knots: m_n={m_n}, J_n={J_n}")

        # 新增内层循环：遍历不同的 number_S
        for num_s in number_S_list:
            X, S, Y = generate_data(n_samples=N, number_S=num_s, seed=42)

            # 兼容 S 维度的拼接 (防止一维和二维拼接报错)
            S_feat = S.reshape(-1, 1) if S.ndim == 1 else S
            X_in = np.hstack([X, S_feat])

            fast_lr = LinearRegression().fit(X_in, Y)
            y_pred = fast_lr.predict(X_in)

            run_times = []
            for r in range(n_repeats):
                start_time = time.time()

                _ = optimize_fair_transform(
                    y_pred, S, Y,
                    degree=base_degree,
                    n_knots=m_n,  # 传入内部节点数 m_n
                    loss_type=loss_type, **loss_kwargs
                )
                run_times.append(time.time() - start_time)

            avg_t = np.mean(run_times)
            std_t = np.std(run_times)

            results.append({
                'N': N,
                'number_S': num_s,
                'Mean': round(avg_t, 4),
                'Std': round(std_t, 4)
            })
            print(f"   -> number_S = {num_s:<2} | Mean = {avg_t:.4f}s | Std = {std_t:.4f}s")

    # ==========================================
    # 数据透视与可视化 (重新生成 N vs number_S 的 2D 表格)
    # ==========================================
    df = pd.DataFrame(results)

    pivot_mean = df.pivot(index='N', columns='number_S', values='Mean')
    pivot_std = df.pivot(index='N', columns='number_S', values='Std')

    pivot_display = pivot_mean.copy().astype(str)
    for r in pivot_mean.index:
        for c in pivot_mean.columns:
            m = pivot_mean.loc[r, c]
            s = pivot_std.loc[r, c]
            pivot_display.loc[r, c] = f"{m:.3f} (±{s:.3f})"

    print("\n" + "=" * 80)
    print("✅ 2D BENCHMARK: MEAN RUNTIME (SECONDS) (N vs number_S)")
    print("=" * 80)
    print(pivot_display.to_string())
    print("=" * 80)

    # 恢复返回两个 DataFrame，适配底部的解包逻辑
    return pivot_mean, pivot_std


# ==========================================
# 8. 执行入口
# ==========================================

# ==========================================
# 8. 执行入口
# ==========================================
if __name__ == "__main__":
    # 建议先用较小的 n_repeats 测试
    mean_df, std_df = run_optimization_scalability_benchmark_S(
        loss_type='Huber', zeta=1.345, n_repeats=30
    )
#loss_type='Quantile', tau=0.75



# loss_type='LAD'
#loss_type='Huber', zeta=1.345
#loss_type='Cauchy', kappa=1
# loss_type='Tukey', c=4.685

# cwh
# 2026/3/26
