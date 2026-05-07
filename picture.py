import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch


def draw_simplified_robust_fairness_pipeline():
    # 设置画布
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')  # 关闭坐标轴

    # 定义颜色 (更强调贡献的配色)
    c_stage1 = '#DAE8FC'  # Light Blue for Robust
    c_stage1_border = '#6C8EBF'
    c_stage2 = '#D5E8D4'  # Light Green for Fair Transformation (Core Contribution 1)
    c_stage2_border = '#82B366'
    c_stage3 = '#FFE6CC'  # Light Orange for Trade-off (Core Contribution 2)
    c_stage3_border = '#D79B00'
    c_text_main = '#333333'
    c_text_contrib = '#4A90E2'  # Blue for contributions

    # ====================
    # 1. Stage 1: General Robust Estimation
    # ====================
    box_s1 = patches.FancyBboxPatch((0.5, 2), 3, 2, boxstyle="round,pad=0.1",
                                    edgecolor=c_stage1_border, facecolor=c_stage1, linewidth=2)
    ax.add_patch(box_s1)
    ax.text(2, 3.5, "General Robust Estimation", ha='center', va='center', fontsize=12, fontweight='bold',
            color=c_text_main)
    ax.text(2, 2.7, r"$\min \sum \mathcal{L}_{rob}(\hat{f}, Y)$", ha='center', va='center', fontsize=12)
    ax.text(2, 2.3, "(Huber, Tukey, etc.)", ha='center', va='center', fontsize=9, color='#444444')

    # ====================
    # 2. Stage 2: Optimal Fair Transformation
    # ====================
    box_s2 = patches.FancyBboxPatch((4.5, 1.5), 3, 3, boxstyle="round,pad=0.1",
                                    edgecolor=c_stage2_border, facecolor=c_stage2, linewidth=2)
    ax.add_patch(box_s2)
    ax.text(6, 4.3, "Optimal Fair Transformation", ha='center', va='center', fontsize=12, fontweight='bold',
            color=c_text_main)
    ax.text(6, 3.5, r"$U = \widehat{F}(\widehat{f} \mid S)$", ha='center', va='center', fontsize=11, color='blue')
    ax.text(6, 2.8, r"$\widehat{g} = \widehat{Q}_{spline}(U)$", ha='center', va='center', fontsize=11, color='blue')
    ax.text(6, 2.1, "(Monotone I-Splines)", ha='center', va='center', fontsize=9, color='#444444')

    # 贡献点文字
    ax.text(6, 1.3, "Core Contribution 1:\nRank-Preserving, Risk-Minimal DP", ha='center', va='center',
            fontsize=9, color=c_text_contrib, fontweight='bold')

    # ====================
    # 3. Stage 3: Fairness-Accuracy Trade-off
    # ====================
    box_s3 = patches.FancyBboxPatch((8.5, 2), 3, 2, boxstyle="round,pad=0.1",
                                    edgecolor=c_stage3_border, facecolor=c_stage3, linewidth=2)
    ax.add_patch(box_s3)
    ax.text(10, 3.5, "Fairness-Accuracy Trade-off", ha='center', va='center', fontsize=12, fontweight='bold',
            color=c_text_main)
    ax.text(10, 2.7, r"$g_\lambda = \lambda \widehat{f} + (1-\lambda)\widehat{g}$", ha='center', va='center',
            fontsize=12, color='blue')

    # 贡献点文字
    ax.text(10, 2.1, "Core Contribution 2:\nLinear Scaling of Unfairness", ha='center', va='center',
            fontsize=9, color=c_text_contrib, fontweight='bold')

    # ====================
    # 4. Arrows
    # ====================

    # S1 -> S2
    arrow1 = FancyArrowPatch((3.5, 3), (4.4, 3), arrowstyle='simple', color='black', mutation_scale=20)
    ax.add_patch(arrow1)
    ax.text(4, 3.3, r"$\widehat{f}$ (Robust)", ha='center', va='center', fontsize=10)

    # S2 -> S3
    arrow2 = FancyArrowPatch((7.5, 3), (8.4, 3), arrowstyle='simple', color='black', mutation_scale=20)
    ax.add_patch(arrow2)
    ax.text(8, 3.3, r"$\widehat{g}$ (Fair)", ha='center', va='center', fontsize=10)

    # S1 -> S3 (Direct for accuracy path)
    arrow_skip = FancyArrowPatch((2, 1.9), (10, 1.9),
                                 arrowstyle='simple', color='#A0A0A0', mutation_scale=20,
                                 connectionstyle="arc3,rad=-0.3")  # 向下弯曲，避免重叠
    ax.add_patch(arrow_skip)
    ax.text(6, 1.5, r"$\widehat{f}$ (for $\lambda$ factor)", ha='center', va='center', fontsize=9, color='#A0A0A0')

    # ====================
    # 5. 保存与显示
    # ====================
    plt.tight_layout()
    plt.savefig('simplified_pipeline_diagram.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    draw_simplified_robust_fairness_pipeline()