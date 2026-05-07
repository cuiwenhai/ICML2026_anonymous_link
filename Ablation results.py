# 2025/12/8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from Tool import * # 确保你的 Tool 里面有 LADRegressor, train_model_cv, optimize_fair_transform 等
from scipy.stats import skewnorm

# 如果需要使用 sklearn 的模型作为 base estimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR


pd.set_option('display.max_columns', None)      # 显示所有列，不进行折叠
pd.set_option('display.width', 1000)             # 设置打印宽度足够大，防止换行
pd.set_option('display.colheader_justify', 'center') # 列标题居中对齐
pd.set_option('display.precision', 4)
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
# 新增: 统一的 Base Estimator 训练接口
# ==========================================
def train_base_estimator(model_name, X_train, Y_train, loss_type, **loss_kwargs):
    """
    根据传入的 model_name 返回训练好的预测函数 (predict_fn)。
    这使得我们可以轻松消融对比不同的 Base Estimators。
    """
    print(f"      [Training Base Estimator: {model_name}]")

    if model_name == 'DNN':
        # 你原有的 PyTorch CV 训练逻辑
        torch_criterion = get_torch_criterion(loss_type, **loss_kwargs)
        predict_fn = train_model_cv(
            LADRegressor,  # Tool.py 中的 PyTorch 模型
            X_train,
            Y_train,
            criterion=torch_criterion,
            k_folds=5, lr=1e-3, epochs=5000, patience=20, batch_size=100
        )
        return predict_fn

    elif model_name == 'Random Forest':
        model = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42)
        model.fit(X_train, Y_train)
        return model.predict

    elif model_name == 'Linear Regression':
        model = LinearRegression()
        model.fit(X_train, Y_train)
        return model.predict

    elif model_name == 'SVMR':
        model = SVR(kernel='rbf', C=1.0)
        model.fit(X_train, Y_train)
        return model.predict

    else:
        raise ValueError(f"Unknown model_name: {model_name}")
# ==========================================
# 5. 整合与运行 (Runner)
# ==========================================
# ==========================================
# 5. 整合与运行 (Runner)
# ==========================================
# ==========================================
# 5. 整合与运行 (Runner)
# ==========================================
def run_simulation_pipeline(seed, loss_type, model_names, **loss_kwargs):
    """
    在单次 run 中，生成一份固定数据，然后让所有的 Base Estimators (model_names)
    都在这份绝对相同的数据上进行训练和公平变换，以实现完美的配对对比测试。
    """
    # -------------------------------------------------
    # 1. 数据生成 (只生成一次，确保所有模型输入一致)
    # -------------------------------------------------
    X_train, S_train, Y_train = generate_data(n_samples=1000, seed=seed)
    X_test, S_test, Y_test = generate_data(n_samples=1000, seed=seed)

    # -------------------------------------------------
    # 2. 预处理
    # -------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_in = np.hstack([X_train_scaled, S_train.reshape(-1, 1)])
    X_test_in = np.hstack([X_test_scaled, S_test.reshape(-1, 1)])

    run_metrics = [] # 用于保存该 seed 下所有模型的结果

    # -------------------------------------------------
    # 遍历并在相同数据上评估不同模型
    # -------------------------------------------------
    for model_name in model_names:
        # 3. 训练 Base Predictor f* (换行修复在这里)
        predict_fn = train_base_estimator(model_name, X_train_in, Y_train, loss_type, **loss_kwargs)

        y_pred_train = predict_fn(X_train_in)
        y_pred_test = predict_fn(X_test_in)

        # 4. 学习公平变换 Q* (Step 2)
        trans_params, d, k = optimize_fair_transform(
            y_pred_train, S_train, Y_train,
            degree=3, n_knots=6,
            loss_type=loss_type, **loss_kwargs
        )

        # 5. 获取完全公平预测器 g*_{Q^*}
        y_final_test = apply_transform(y_pred_test, S_test, trans_params, d, k)

        # 6. Geodesic Interpolation (插值)
        lambda_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        interpolation_metrics = []

        for lam in lambda_grid:
            y_test_lambda = lam * y_pred_test + (1 - lam) * y_final_test
            loss_val, dp_val = evaluate_metrics(y_test_lambda, Y_test, S_test, loss_type, **loss_kwargs)

            interpolation_metrics.append({
                'Model': model_name,
                'Lambda': lam,
                'Risk': loss_val,
                'Unfairness': dp_val
            })

        # 7. 对比方法: Wasserstein Barycenter (Algo 1)
        y_test_wass = method_wasserstein_apply(y_pred_train, S_train, y_pred_test, S_test)
        wass_loss, wass_dp = evaluate_metrics(y_test_wass, Y_test, S_test, loss_type, **loss_kwargs)

        # 8. 汇总当前模型结果
        metrics = {
            'Model': model_name,
            'Interpolation': interpolation_metrics,
            'Method_Optimization': {  # Lambda=0
                'Loss': interpolation_metrics[0]['Risk'],
                'DP': interpolation_metrics[0]['Unfairness']
            },
            'Method_Base': {  # Lambda=1
                'Loss': interpolation_metrics[-1]['Risk'],
                'DP': interpolation_metrics[-1]['Unfairness']
            },
            'Method_Wasserstein_Direct': {
                'Loss': wass_loss,
                'DP': wass_dp
            }
        }
        run_metrics.append(metrics)

    return run_metrics

# ==========================================
# 5. 蒙特卡洛实验 (多次运行)
# ==========================================

def run_monte_carlo_experiment(n_repeats, loss_type, model_names, **loss_kwargs):
    print(f"\n🚀 Starting Monte Carlo Simulation ({n_repeats} runs)...")
    param_str = ", ".join([f"{k}={v}" for k, v in loss_kwargs.items()])
    print(f"⚙️  Settings: Loss={loss_type} | Params=[{param_str}]")
    print(f"🤖 Models to compare: {model_names}")

    raw_results = []

    # 外层循环现在只遍历 Seed
    for i in range(n_repeats):
        print(f"\n>>> Running Seed: {i}")
        try:
            # 返回的是该 seed 下所有模型的结果列表
            res_list = run_simulation_pipeline(
                seed=i,
                loss_type=loss_type,
                model_names=model_names,
                **loss_kwargs
            )
            raw_results.extend(res_list)  # 用 extend 展平列表
            print(f"   -> Run {i + 1}/{n_repeats} done.")
        except Exception as e:
            print(f"⚠️ Run {i} failed: {e}")

    if not raw_results:
        print("❌ No results collected.")
        return None, None

    # 1. 整理关键点数据
    point_metrics_list = []
    for res in raw_results:
        row = {
            'Model': res['Model'],
            'Init_Loss': res['Method_Base']['Loss'],
            'Init_DP': res['Method_Base']['DP'],
            'Final_Loss': res['Method_Optimization']['Loss'],
            'Final_DP': res['Method_Optimization']['DP'],
            'Wass_Loss': res['Method_Wasserstein_Direct']['Loss'],
            'Wass_DP': res['Method_Wasserstein_Direct']['DP']
        }
        point_metrics_list.append(row)

    df_points = pd.DataFrame(point_metrics_list)

    # 2. 整理插值曲线数据
    plot_data = []
    for res in raw_results:
        for point in res['Interpolation']:
            plot_data.append(point)

    df_plot = pd.DataFrame(plot_data)

    # 3. 打印摘要结果 (按模型分组统计)
    print("\n" + "=" * 60)
    print(f"📊 ABLATION STUDY SUMMARY (N={n_repeats})")
    print("=" * 60)

    # 按照 Model 分组计算 Mean 和 Std
    stats_df = df_points.groupby('Model').agg(['mean', 'std'])
    print(stats_df[['Init_Loss', 'Init_DP', 'Final_Loss', 'Final_DP']])

    # 调用之前的多模型画图函数
    plot_tradeoff_curves(df_plot, df_points)

    return df_points, df_plot

# ==========================================
# 6. 画图函数 (Visualization)
# ==========================================
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 6. 画图函数 (Visualization - 修复星号图例与 LaTeX)
# ==========================================
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 6. 画图函数 (Visualization - 固定 Y 轴范围 0-2)
# ==========================================
def plot_tradeoff_curves(df_plot, df_points):
    """
    绘制公平性-准确性权衡曲线 (支持多模型对比，优化了 4 个模型的布局)
    """
    sns.set_style("whitegrid")

    # 按 Lambda 和 Model 聚合计算均值
    df_mean = df_plot.groupby(['Model', 'Lambda'])[['Risk', 'Unfairness']].mean().reset_index()
    df_mean = df_mean.sort_values(['Model', 'Unfairness'])

    # 生成一个固定的颜色映射字典，确保折线和星星颜色完全一致
    unique_models = df_mean['Model'].unique()
    palette = dict(zip(unique_models, sns.color_palette("tab10", len(unique_models))))

    # ==========================================
    # 图 1: Pareto Frontier (所有模型画在同一张图对比)
    # ==========================================
    plt.figure(figsize=(10, 8))
    ax1 = plt.gca()

    # 画不同模型的插值均值曲线 (Trade-off curves)
    sns.lineplot(
        data=df_mean,
        x='Unfairness',
        y='Risk',
        hue='Model',
        style='Model',
        palette=palette,
        markers=True,
        markersize=9,
        linewidth=2.5,
        ax=ax1
    )

    # 画基准点 (FRWB / Wasserstein Barycenter)
    wass_mean = df_points.groupby('Model')[['Wass_DP', 'Wass_Loss']].mean().reset_index()

    # 遍历每个模型画星星，并强制写入图例
    for model in unique_models:
        model_data = wass_mean[wass_mean['Model'] == model]
        if not model_data.empty:
            ax1.scatter(
                model_data['Wass_DP'],
                model_data['Wass_Loss'],
                color=palette[model],
                marker='*',
                s=500,
                edgecolor='black',
                linewidth=0.5,
                label=f'FRWB - {model}',
                zorder=5
            )

    # 设置图 1 属性
    ax1.set_xlabel('Empirical Unfairness', fontsize=14)
    ax1.set_ylabel('Empirical Risk', fontsize=14)

    # 💡 新增：强制设置第一张图的 Y 轴范围为 0 到 2
    ax1.set_ylim(0.65, 2)

    # 优化图例放在图的外部右侧
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, fontsize=11, title="Methods", title_fontsize=12,
               bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

    # ==========================================
    # 图 2 到 图 5: 每个模型独立画一张 Lambda 趋势图
    # ==========================================
    # 遍历每个模型，为它们单独创建画布
    for model in unique_models:
        # 筛选当前模型的数据
        model_data = df_mean[df_mean['Model'] == model]

        # 创建一个独立的新画布
        fig, ax_left = plt.subplots(figsize=(7, 5))
        ax_right = ax_left.twinx()

        # 左轴：Risk (实线)
        sns.lineplot(data=model_data, x='Lambda', y='Risk', ax=ax_left, color='green', marker='^',
                     label='Empirical Risk')
        ax_left.set_xlabel(r'Interpolation parameter $\lambda$', fontsize=13)
        ax_left.set_ylabel('Empirical Risk', color='green', fontsize=13)
        ax_left.tick_params(axis='y', labelcolor='green')

        # 右轴：Unfairness (虚线)
        sns.lineplot(data=model_data, x='Lambda', y='Unfairness', ax=ax_right, color='orange', marker='s',
                     linestyle='--', label='Empirical Unfairness')
        ax_right.set_ylabel('Empirical Unfairness', color='orange', fontsize=13)
        ax_right.tick_params(axis='y', labelcolor='orange')

        # 💡 新增：强制设置后 4 张图中左右两个 Y 轴的范围都是 0 到 2
        ax_left.set_ylim(0, 2)
        ax_right.set_ylim(0, 2)

        # 合并左轴和右轴的图例，放在图表内部合适的位置
        lines_1, labels_1 = ax_left.get_legend_handles_labels()
        lines_2, labels_2 = ax_right.get_legend_handles_labels()
        # 清除默认的 seaborn 单边图例，防止重复
        if ax_left.get_legend(): ax_left.get_legend().remove()
        if ax_right.get_legend(): ax_right.get_legend().remove()
        # 统一设置图例
        ax_left.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=11)

        # 设置带 LaTeX 的标题
        plt.title(fr'Empirical risk (Huber) and unfairness with  the {model} Method', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.show()  # 依次显示第 2, 3, 4, 5 张图
# ==========================================
# 7. 执行入口
# ==========================================
if __name__ == "__main__":
    # 在这里填入你所有的 4 个方法！
    run_monte_carlo_experiment(
        n_repeats=200,  # 正式出图时可以改为 50 或 100
        loss_type='Huber',
        zeta=1.345,
        model_names=[
            'DNN',  # 方法 1
            'Random Forest',  # 方法 2
            'Linear Regression',  # 方法 3
            'SVMR'  # 方法 4
        ]
    )