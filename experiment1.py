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
    lambda_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

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
    运行 N 次模拟，打印关键点摘要，并返回用于画图的详细数据
    """
    print(f"\n🚀 Starting Monte Carlo Simulation ({n_repeats} runs)...")
    param_str = ", ".join([f"{k}={v}" for k, v in loss_kwargs.items()])
    print(f"⚙️  Settings: Loss={loss_type} | Params=[{param_str}]")

    raw_results = []

    for i in range(n_repeats):
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

        # 简单进度条
        print(f"   -> Run {i + 1}/{n_repeats} done.")

    if not raw_results:
        print("❌ No results collected.")
        return None, None

    # ==========================================
    # 1. 整理关键点数据 (用于打印摘要)
    # ==========================================
    point_metrics_list = []
    for res in raw_results:
        row = {
            # Lambda=1 (最不公平，Risk最小)
            'Init_Loss': res['Method_Base']['Loss'],
            'Init_DP': res['Method_Base']['DP'],

            # Lambda=0 (最公平，Risk最大)
            'Final_Loss': res['Method_Optimization']['Loss'],
            'Final_DP': res['Method_Optimization']['DP'],

            # 对比方法 (Wasserstein)
            'Wass_Loss': res['Method_Wasserstein_Direct']['Loss'],
            'Wass_DP': res['Method_Wasserstein_Direct']['DP']
        }
        point_metrics_list.append(row)

    df_points = pd.DataFrame(point_metrics_list)
    stats_points = df_points.agg(['mean', 'std'])

    # ==========================================
    # 2. 整理插值曲线数据 (用于画图)
    # ==========================================
    # 我们将所有 Run 的所有插值点都展平成一个长列表，方便 Seaborn 画图
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
    print("=" * 60)

    print(f"{'Metric':<20} | {'Mean':<10} | {'Std Dev':<10}")
    print("-" * 46)

    # 只打印起止点和对比方法
    keys_to_print = ['Init_Loss', 'Final_Loss', 'Wass_Loss', 'Init_DP', 'Final_DP', 'Wass_DP']
    for m in keys_to_print:
        mean_val = stats_points.loc['mean', m]
        std_val = stats_points.loc['std', m]
        print(f"{m:<20} | {mean_val:.4f} ({std_val:.4f})")

    print("-" * 46)
    print("✅ Interpolation details hidden. Generating plots...")

    # 调用画图函数
    plot_tradeoff_curves(df_plot, df_points)

    return df_points, df_plot


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


# ==========================================
# 6. 执行入口
# ==========================================
if __name__ == "__main__":
    # 调用示例：分位数回归 (tau=0.9)
    run_monte_carlo_experiment(n_repeats=1, loss_type='Huber', zeta=1.345)

#loss_type='Quantile', tau=0.75



# loss_type='LAD'
#loss_type='Huber', zeta=1.345
#loss_type='Cauchy', kappa=1
# loss_type='Tukey', c=4.685

