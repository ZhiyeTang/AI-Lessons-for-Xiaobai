import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============ 图1：温度对SoftMax概率分布的影响 ============

def softmax_with_temperature(logits, temperature):
    """计算带温度的softmax"""
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))  # 数值稳定性
    return exp_logits / np.sum(exp_logits)

# 原始logits（模拟句子"今天天气很___"的候选词分数）
words = ['好', '糟糕', '蓝', '不错', '热', '冷', '桌子']
logits = np.array([2.5, 1.8, 0.5, 1.2, 0.8, 0.6, -3.0])  # "桌子"分数很低

# 温度值范围
temperatures = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

colors = plt.cm.viridis(np.linspace(0, 1, len(words)))

def update_softmax(frame):
    ax1.clear()
    ax2.clear()
    
    T = temperatures[frame]
    probs = softmax_with_temperature(logits, T)
    
    # 左图：柱状图
    bars = ax1.bar(words, probs, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Probability', fontsize=12)
    ax1.set_title(f'Temperature T = {T}', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # 在柱子上显示数值
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    # 右图：温度-概率曲线（展示每个词的概率随温度变化）
    T_range = np.linspace(0.1, 5, 100)
    for i, word in enumerate(words):
        probs_range = [softmax_with_temperature(logits, t)[i] for t in T_range]
        ax2.plot(T_range, probs_range, label=word, color=colors[i], linewidth=2)
    
    ax2.axvline(x=T, color='red', linestyle='--', linewidth=2, label=f'Current T={T}')
    ax2.set_xlabel('Temperature', fontsize=12)
    ax2.set_ylabel('Probability', fontsize=12)
    ax2.set_title('Probability vs Temperature', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    return bars

# 创建动画
ani1 = FuncAnimation(fig1, update_softmax, frames=len(temperatures), 
                     interval=1000, repeat=True, blit=False)

# 保存GIF
ani1.save('/Users/zhiyetang/.openclaw/workspace-taizi/AI扫盲课-Marp/images/temperature_effect.gif', 
          writer='pillow', fps=1, dpi=100)
print("✅ 温度对SoftMax影响图已保存")

plt.close()

# ============ 图2：交叉熵随预测概率变化 ============

def cross_entropy_loss(true_prob, pred_prob, epsilon=1e-10):
    """计算交叉熵损失"""
    pred_prob = np.clip(pred_prob, epsilon, 1 - epsilon)  # 防止log(0)
    return -true_prob * np.log(pred_prob) - (1 - true_prob) * np.log(1 - pred_prob)

# 真实标签："猫"是正确的（one-hot: [1, 0, 0, 0, 0]）
words2 = ['猫', '狗', '鸟', '鱼', '车']
true_dist = np.array([1.0, 0, 0, 0, 0])  # 真实分布（one-hot）

# 模拟模型从差到好的预测分布演变
prediction_stages = [
    # 初始：完全随机
    np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
    # 稍微偏向正确
    np.array([0.3, 0.25, 0.2, 0.15, 0.1]),
    # 更明显
    np.array([0.4, 0.25, 0.15, 0.12, 0.08]),
    # 较好
    np.array([0.5, 0.2, 0.15, 0.1, 0.05]),
    # 很好
    np.array([0.7, 0.15, 0.08, 0.05, 0.02]),
    # 非常好
    np.array([0.85, 0.08, 0.04, 0.02, 0.01]),
    # 几乎完美
    np.array([0.95, 0.03, 0.015, 0.003, 0.002]),
    # 完美
    np.array([0.99, 0.008, 0.0015, 0.0003, 0.0002]),
]

fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))

colors2 = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

def update_crossentropy(frame):
    ax3.clear()
    ax4.clear()
    
    pred_dist = prediction_stages[frame]
    ce_loss = cross_entropy_loss(1.0, pred_dist[0])  # 只计算正确类别的损失
    
    # 左图：真实分布 vs 预测分布对比
    x = np.arange(len(words2))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, true_dist, width, label='True Distribution', 
                    color='#2ecc71', alpha=0.8, edgecolor='black')
    bars2 = ax3.bar(x + width/2, pred_dist, width, label='Predicted Distribution', 
                    color='#e74c3c', alpha=0.8, edgecolor='black')
    
    ax3.set_ylabel('Probability', fontsize=12)
    ax3.set_title(f'Stage {frame+1}/8: True vs Predicted', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(words2)
    ax3.legend(fontsize=10)
    ax3.set_ylim(0, 1.1)
    ax3.grid(axis='y', alpha=0.3)
    
    # 显示交叉熵损失
    ax3.text(0.5, 0.95, f'Cross-Entropy Loss: {ce_loss:.4f}', 
             transform=ax3.transAxes, fontsize=14, fontweight='bold',
             ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # 右图：交叉熵损失曲线（展示所有阶段）
    all_losses = [cross_entropy_loss(1.0, s[0]) for s in prediction_stages]
    stages = list(range(1, len(prediction_stages) + 1))
    
    ax4.plot(stages, all_losses, 'o-', linewidth=2, markersize=10, color='#e74c3c')
    ax4.axvline(x=frame+1, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax4.fill_between(stages, all_losses, alpha=0.3, color='#e74c3c')
    
    ax4.set_xlabel('Training Stage', fontsize=12)
    ax4.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax4.set_title('Loss Decreasing During Training', fontsize=14, fontweight='bold')
    ax4.grid(alpha=0.3)
    ax4.set_xticks(stages)
    
    # 标注当前阶段
    current_loss = all_losses[frame]
    ax4.annotate(f'Current\n{current_loss:.4f}', 
                xy=(frame+1, current_loss),
                xytext=(frame+1, current_loss + 0.3),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                fontsize=10, ha='center', color='blue', fontweight='bold')
    
    plt.tight_layout()
    return bars1, bars2

# 创建动画
ani2 = FuncAnimation(fig2, update_crossentropy, frames=len(prediction_stages), 
                     interval=1200, repeat=True, blit=False)

# 保存GIF
ani2.save('/Users/zhiyetang/.openclaw/workspace-taizi/AI扫盲课-Marp/images/cross_entropy.gif', 
          writer='pillow', fps=1, dpi=100)
print("✅ 交叉熵损失变化图已保存")

plt.close()

print("\n🎉 所有可视化图表生成完成！")
print("📁 保存位置: /Users/zhiyetang/.openclaw/workspace-taizi/AI扫盲课-Marp/images/")
print("   - temperature_effect.gif")
print("   - cross_entropy.gif")
