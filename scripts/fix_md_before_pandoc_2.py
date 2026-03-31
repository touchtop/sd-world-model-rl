import re

md_file = "/home/data/round/code/sd_world_model_rl/docs/ijoc/ijoc_paper_draft_v2_zh.md"

with open(md_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix newline before parameter tables
content = content.replace("**参数**：\n| 参数", "**参数**：\n\n| 参数")

# Write changes
with open(md_file, 'w', encoding='utf-8') as f:
    f.write(content)

print("Markdown fixes applied successfully.")
