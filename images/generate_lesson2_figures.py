import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 注意力权重热力图
fig1, ax1 = plt.subplots(figsize=(8, 6))

words = ['The', 'cat', 'sat', 'on', 'the', 'mat']
attention_weights = np.array([
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.2, 0.8, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.3, 0.5, 0.2, 0.0, 0.0],
    [0.0, 0.0, 0.1, 0.1, 0.4, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
    [0.0, 0.0, 0.0, 0.1, 0.2, 0.7]
])

im1 = ax1.imshow(attention_weights, cmap='YlOrRd', aspect='auto')
ax1.set_xticks(np.arange(len(words)))
ax1.set_yticks(np.arange(len(words)))
ax1.set_xticklabels(words)
ax1.set_yticklabels(words)
ax1.set_xlabel('Key (被关注的词)', fontsize=12)
ax1.set_ylabel('Query (当前词)', fontsize=12)
ax1.set_title('注意力权重热力图 (Attention Heatmap)', fontsize=14, fontweight='bold')

# 添加数值标注
for i in range(len(words)):
    for j in range(len(words)):
        text = ax1.text(j, i, f'{attention_weights[i, j]:.1f}',
                       ha="center", va="center", color="black" if attention_weights[i, j] < 0.5 else "white",
                       fontsize=10)

plt.colorbar(im1, ax=ax1, label='Attention Weight')
plt.tight_layout()
plt.savefig('/Users/zhiyetang/.openclaw/workspace-taizi/AI扫盲课-Marp/images/attention_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

print("✅ 注意力权重热力图已保存")

# 2. KV Cache示意图
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))

# 左图：没有KV Cache的情况
ax2a.set_title('没有KV Cache: 每步重新计算所有K,V', fontsize=12, fontweight='bold')
ax2a.set_xlim(0, 10)
ax2a.set_ylim(0, 10)

# 绘制输入token
for i, token in enumerate(['我', '叫', '小', '明']):
    y_pos = 8 - i * 2
    # Query
    ax2a.add_patch(plt.Rectangle((1, y_pos-0.4), 1.5, 0.8, facecolor='lightblue', edgecolor='black'))
    ax2a.text(1.75, y_pos, f'Q{i+1}', ha='center', va='center', fontsize=10)
    
    # 绘制所有K,V（重复计算）
    for j in range(i+1):
        x_pos = 4 + j * 1.5
        ax2a.add_patch(plt.Rectangle((x_pos, y_pos-0.3), 0.6, 0.6, facecolor='lightcoral', edgecolor='black', alpha=0.7))
        ax2a.add_patch(plt.Rectangle((x_pos+0.7, y_pos-0.3), 0.6, 0.6, facecolor='lightgreen', edgecolor='black', alpha=0.7))
        ax2a.text(x_pos+0.3, y_pos, f'K{j+1}', ha='center', va='center', fontsize=8)
        ax2a.text(x_pos+1, y_pos, f'V{j+1}', ha='center', va='center', fontsize=8)
    
    # 绘制计算箭头
    for j in range(i+1):
        ax2a.arrow(2.5, y_pos, 1.2 + j*1.5 - 2.5, 0, head_width=0.2, head_length=0.2, fc='gray', ec='gray', alpha=0.3)

ax2a.text(5, 1, '时间复杂度: O(n²)', ha='center', fontsize=11, color='red')
ax2a.axis('off')

# 右图：有KV Cache的情况
ax2b.set_title('有KV Cache: 只计算新的K,V，复用缓存', fontsize=12, fontweight='bold')
ax2b.set_xlim(0, 10)
ax2b.set_ylim(0, 10)

# 绘制缓存区域
ax2b.add_patch(plt.Rectangle((0.5, 0.5), 3, 9, facecolor='lightyellow', edgecolor='orange', linewidth=2, linestyle='--'))
ax2b.text(2, 9.3, 'KV Cache', ha='center', fontsize=11, fontweight='bold', color='orange')

for i, token in enumerate(['我', '叫', '小', '明']):
    y_pos = 8 - i * 2
    
    # Query（新计算）
    ax2b.add_patch(plt.Rectangle((1, y_pos-0.4), 1.5, 0.8, facecolor='lightblue', edgecolor='black'))
    ax2b.text(1.75, y_pos, f'Q{i+1}', ha='center', va='center', fontsize=10)
    
    # 新的K,V
    x_pos = 6
    ax2b.add_patch(plt.Rectangle((x_pos, y_pos-0.3), 0.8, 0.6, facecolor='lightcoral', edgecolor='black'))
    ax2b.add_patch(plt.Rectangle((x_pos+1, y_pos-0.3), 0.8, 0.6, facecolor='lightgreen', edgecolor='black'))
    ax2b.text(x_pos+0.4, y_pos, f'K{i+1}', ha='center', va='center', fontsize=9)
    ax2b.text(x_pos+1.4, y_pos, f'V{i+1}', ha='center', va='center', fontsize=9)
    
    # 存入Cache的箭头
    ax2b.arrow(x_pos+0.4, y_pos-0.4, -4, -0.5, head_width=0.2, head_length=0.2, fc='green', ec='green', alpha=0.6)
    ax2b.arrow(x_pos+1.4, y_pos-0.4, -3, -0.5, head_width=0.2, head_length=0.2, fc='green', ec='green', alpha=0.6)
    
    # 从Cache读取的箭头
    if i > 0:
        ax2b.arrow(3.5, y_pos+0.8, 2, 0, head_width=0.2, head_length=0.2, fc='blue', ec='blue', alpha=0.6, linestyle='dashed')

ax2b.text(5, 1, '时间复杂度: O(n)', ha='center', fontsize=11, color='green')
ax2b.axis('off')

plt.tight_layout()
plt.savefig('/Users/zhiyetang/.openclaw/workspace-taizi/AI扫盲课-Marp/images/kv_cache.png', dpi=150, bbox_inches='tight')
plt.close()

print("✅ KV Cache示意图已保存")

# 3. Transformer架构图 (简化版)
fig3, ax3 = plt.subplots(figsize=(10, 12))
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 14)

# 标题
ax3.text(5, 13.5, 'Transformer Architecture', ha='center', fontsize=16, fontweight='bold')

# 输入
ax3.add_patch(plt.Rectangle((2, 12), 6, 0.8, facecolor='lightblue', edgecolor='black', linewidth=2))
ax3.text(5, 12.4, 'Input Embedding + Positional Encoding', ha='center', va='center', fontsize=10)

# Encoder Stack
ax3.add_patch(plt.Rectangle((1, 9.5), 8, 2, facecolor='#E8F4F8', edgecolor='navy', linewidth=2))
ax3.text(5, 11.2, 'Encoder × N', ha='center', fontsize=12, fontweight='bold', color='navy')

# Multi-Head Attention
ax3.add_patch(plt.Rectangle((2, 10), 6, 0.7, facecolor='lightyellow', edgecolor='black'))
ax3.text(5, 10.35, 'Multi-Head Attention', ha='center', va='center', fontsize=10)

# Feed Forward
ax3.add_patch(plt.Rectangle((2, 9.6), 6, 0.7, facecolor='lightgreen', edgecolor='black'))
ax3.text(5, 9.95, 'Feed Forward', ha='center', va='center', fontsize=10)

# Decoder Stack
ax3.add_patch(plt.Rectangle((1, 5.5), 8, 3.5, facecolor='#FFF2E8', edgecolor='darkorange', linewidth=2))
ax3.text(5, 8.7, 'Decoder × N', ha='center', fontsize=12, fontweight='bold', color='darkorange')

# Masked Multi-Head Attention
ax3.add_patch(plt.Rectangle((2, 7.5), 6, 0.7, facecolor='lightyellow', edgecolor='black'))
ax3.text(5, 7.85, 'Masked Multi-Head Attention', ha='center', va='center', fontsize=9)

# Cross Attention
ax3.add_patch(plt.Rectangle((2, 6.5), 6, 0.7, facecolor='#FFE4B5', edgecolor='black'))
ax3.text(5, 6.85, 'Cross Attention', ha='center', va='center', fontsize=10)

# Feed Forward
ax3.add_patch(plt.Rectangle((2, 5.7), 6, 0.7, facecolor='lightgreen', edgecolor='black'))
ax3.text(5, 6.05, 'Feed Forward', ha='center', va='center', fontsize=10)

# 输出
ax3.add_patch(plt.Rectangle((2, 4), 6, 0.8, facecolor='lightcoral', edgecolor='black', linewidth=2))
ax3.text(5, 4.4, 'Output Probability', ha='center', va='center', fontsize=10)

# 箭头连接
arrow_style = dict(arrowstyle='->', color='black', lw=2)
ax3.annotate('', xy=(5, 12), xytext=(5, 11.5), arrowprops=arrow_style)
ax3.annotate('', xy=(5, 9.5), xytext=(5, 9), arrowprops=arrow_style)
ax3.annotate('', xy=(5, 5.5), xytext=(5, 4.8), arrowprops=arrow_style)

# Encoder到Decoder的箭头
ax3.annotate('', xy=(9, 7), xytext=(9, 10), arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax3.text(9.3, 8.5, 'Context', rotation=90, va='center', fontsize=9, color='blue')

ax3.axis('off')
plt.tight_layout()
plt.savefig('/Users/zhiyetang/.openclaw/workspace-taizi/AI扫盲课-Marp/images/transformer_arch.png', dpi=150, bbox_inches='tight')
plt.close()

print("✅ Transformer架构图已保存")

print("\n🎉 所有图表生成完成！")
