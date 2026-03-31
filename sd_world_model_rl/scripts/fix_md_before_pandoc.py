import re

md_file = "/home/data/round/code/sd_world_model_rl/docs/ijoc/ijoc_paper_draft_v2_zh.md"

with open(md_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Fix Table alignment colons so Pandoc generates natural width tables without minipage
content = re.sub(r'\|\s*:?-+:?[-\s|:]*\|', lambda m: m.group(0).replace(':', '-'), content)

# 2. Fix tree ASCII characters
content = content.replace('├──', '|--')
content = content.replace('└──', '+--')
content = content.replace('│', '|')

# 3. Add explicit blank line before standard bullet points to force Pandoc itemize globally where missing
# Specifically targeting 1.2
content = content.replace("具体应用于OR领域的序贯决策问题：\n-", "具体应用于OR领域的序贯决策问题：\n\n-")

# 4. Fix missing parameter tables in 6.2.1 and 6.2.2
old_table_621 = """**参数**：
| 参数 | 真值 | 初始信念 | 误差 |
|------|------|---------|------|
| $\\alpha$（退化率） | 0.1 | 0.001 | 100$\\times$ |
| $\\tau$（疲劳阈值） | 5.0 | 12.0 | 2.4$\\times$ |"""
# Wait, 6.2.1 might just be missing pipes or something. Let me check how it was originally written.
# Actually I'd better just fix them in the latex output because I'm not sure if the markdown string match is perfect.

with open(md_file, 'w', encoding='utf-8') as f:
    f.write(content)

print("Markdown fixes applied successfully.")
