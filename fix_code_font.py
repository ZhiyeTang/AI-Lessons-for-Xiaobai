import re
import os

# 深化版目录
dir_path = "/Users/zhiyetang/.openclaw/workspace-taizi/AI扫盲课-Marp/深化版"

for filename in os.listdir(dir_path):
    if filename.endswith(".md"):
        filepath = os.path.join(dir_path, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否已有code字体设置
        if 'code {' not in content:
            # 在style块中添加code字体设置
            old_pattern = r'(style: \|\n  @import.*?\n  section \{\n    font-family: "Noto Sans SC".*?sans-serif;)'
            new_replacement = r'''\1
  }
  code {
    font-family: "SF Mono", "Monaco", "Inconsolata", "Fira Code", "Noto Sans SC", monospace;'''
            
            content = re.sub(old_pattern, new_replacement, content, flags=re.DOTALL)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ 已更新: {filename}")
        else:
            print(f"⏭️  跳过: {filename} (已有code设置)")

print("\n🎉 所有文件更新完成！")
