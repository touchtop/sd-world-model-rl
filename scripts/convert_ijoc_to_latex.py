import os
import subprocess

md_file = "/home/data/round/code/sd_world_model_rl/docs/ijoc/ijoc_paper_draft_v2_zh.md"
tex_ref = "/home/data/round/code/sd_world_model_rl/docs/neurips/paper/full_draft_zh.tex"
tex_out = "/home/data/round/code/sd_world_model_rl/docs/ijoc/ijoc_paper_draft_v2_zh.tex"

# 1. Read the preamble from the reference tex file
with open(tex_ref, 'r', encoding='utf-8') as f:
    lines = f.readlines()

preamble = []
for line in lines:
    preamble.append(line)
    if "\\begin{document}" in line:
        break

preamble = "".join(preamble)

# 2. Run Pandoc on the MD file to generate LaTeX body
cmd = [
    "pandoc",
    md_file,
    "-t", "latex",
    "--top-level-division=section"
]
result = subprocess.run(cmd, capture_output=True, text=True, check=True)
body = result.stdout

# 3. Apply manual fixes to the generated LaTeX body (tables and line breaks)
# Replace longtable with standard table if needed, but longtable is usually fine.
# Pandoc might miss some equation math blocks if they have wrong delimiters.
# Let's fix common markdown issues.

body = body.replace("\\textbackslash{}", "\\") # Fix escaped backslashes in math blocks sometimes

# 4. Construct final LaTeX document
final_tex = preamble + "\n" + body + "\n\\end{document}\n"

with open(tex_out, 'w', encoding='utf-8') as f:
    f.write(final_tex)

print("Conversion complete.")
