import re
import os

dir_path = "/Users/zhiyetang/.openclaw/workspace-taizi/AI扫盲课-Marp/深化版"

# 修复第01课
def fix_lesson01(content):
    # 修复自回归公式
    old = r'''$$P(\text{"我叫小明"}) = P(\text{"我"}) \times P(\text{"叫"}|\text{"我"}) \times P(\text{"小"}|\text{"我叫"}) \times P(\text{"明"}|\text{"我叫小"})$$'''
    new = r'''$$P(w_{1:4}) = P(w_1) \times P(w_2|w_1) \times P(w_3|w_{1:2}) \times P(w_4|w_{1:3})$$

其中: $w_1=$"我", $w_2=$"叫", $w_3=$"小", $w_4=$"明"'''
    content = content.replace(old, new)
    
    # 修复Softmax公式中的中文
    old = r'''$$P(w_i | \text{context}) = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}$$'''
    new = r'''$$P(w_i | c) = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}$$

其中 $c$ 表示上下文 (context)'''
    content = content.replace(old, new)
    
    # 修复交叉熵公式中的中文
    old = r'''$$\mathcal{L} = -\log P(\text{正确答案} | \text{context})$$'''
    new = r'''$$\mathcal{L} = -\log P(w_{true} | c)$$

其中 $w_{true}$ 是正确答案，$c$ 是上下文'''
    content = content.replace(old, new)
    
    return content

# 修复第03课
def fix_lesson03(content):
    # 修复One-hot编码公式
    old = r'''$$\text{"猫"} = [0, 0, 1, 0, ..., 0] \in \mathbb{R}^{|V|}$$'''
    new = r'''$$\mathbf{o}_{cat} = [0, 0, 1, 0, ..., 0] \in \mathbb{R}^{|V|}$$

($\mathbf{o}_{cat}$ 表示"猫"的one-hot向量)'''
    content = content.replace(old, new)
    
    # 修复词向量公式
    old = r'''$$\mathbf{e}_{\text{猫}} \in \mathbb{R}^{d}, \quad d \ll |V|$$'''
    new = r'''$$\mathbf{e}_{cat} \in \mathbb{R}^{d}, \quad d \ll |V|$$

($\mathbf{e}_{cat}$ 表示"猫"的词向量)'''
    content = content.replace(old, new)
    
    return content

# 处理每个文件
for filename in os.listdir(dir_path):
    if filename.endswith(".md"):
        filepath = os.path.join(dir_path, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        
        if "第01课" in filename:
            content = fix_lesson01(content)
        elif "第03课" in filename:
            content = fix_lesson03(content)
        
        if content != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ 已修复: {filename}")
        else:
            print(f"⏭️  无需修改: {filename}")

print("\n🎉 公式中文字符修复完成！")
