import re

tex_file = "/home/data/round/code/sd_world_model_rl/docs/ijoc/ijoc_paper_draft_v2_zh.tex"

with open(tex_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Fix line breaks for `- 结构` and `- 参数` in section 1.2
content = content.replace("具体应用于OR领域的序贯决策问题：\n-\n\\textbf{结构}", "具体应用于OR领域的序贯决策问题：\\\\\n- \\textbf{结构}")
content = content.replace("具体应用于OR领域的序贯决策问题： -", "具体应用于OR领域的序贯决策问题：\\\\\n-")
content = content.replace("指定并硬编码。\n-", "指定并硬编码。\\\\\n-")

# 2. Fix the remaining minipage[t] in table headers/bodies
content = re.sub(r'\\begin\{minipage\}\[t\]\{.*?columnwidth\}\\raggedright\n?(.*?)\\strut\n?\\end\{minipage\}', r'\1', content, flags=re.DOTALL)

# 3. Fix 7.4.3 longtable headers alignment by using multicolumn
header_old = r"""发现 & 证据 & 实践启示\tabularnewline
\midrule"""
header_new = r"""\multicolumn{1}{l}{发现} & \multicolumn{1}{l}{证据} & \multicolumn{1}{l}{实践启示}\tabularnewline
\midrule"""
content = content.replace(header_old, header_new)

with open(tex_file, 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixes applied successfully.")
