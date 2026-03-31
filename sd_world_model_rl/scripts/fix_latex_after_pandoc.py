import re

tex_file = "/home/data/round/code/sd_world_model_rl/docs/ijoc/ijoc_paper_draft_v2_zh.tex"

with open(tex_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Fix \linethickness
content = content.replace('\\linethickness', '0.4pt')

# 2. Fix Greek characters in Pseudo-code
code_old = r"""\begin{verbatim}
输入：环境 env，初始参数 θ₀, ψ₀
输出：训练好的世界模型 M_θ，价值网络 V_ψ

for episode = 1 to N do
    // 阶段1：数据收集
    trajectory ← CollectEpisode(env, MCTS_policy(M_θ, V_ψ))
    buffer.add(trajectory)

    // 阶段2：世界模型更新（可微分系统辨识）
    从buffer采样批次 {(sₜ, aₜ, sₜ₊₁)}
    L_model ← ||sₜ₊₁ - M_θ(sₜ, aₜ)||²
    θ ← θ - η_θ ∇_θ L_model

    // 阶段3：价值网络更新（TD学习）
    L_value ← ||V_ψ(sₜ) - (rₜ + γ·V_ψ(sₜ₊₁))||²
    ψ ← ψ - η_ψ ∇_ψ L_value

    // 阶段4：策略蒸馏（可选）
    π_target ← normalize(MCTS_visit_counts)
    L_policy ← CrossEntropy(π_φ(·|sₜ), π_target)
    φ ← φ - η_φ ∇_φ L_policy
end for
\end{verbatim}"""

code_new = r"""\begin{quote}
\ttfamily
输入：环境 env，初始参数 $\theta_0$, $\psi_0$ \\
输出：训练好的世界模型 $M_\theta$，价值网络 $V_\psi$ \\
~\\
for episode = 1 to N do \\
\hspace*{4em}// 阶段1：数据收集 \\
\hspace*{4em}trajectory $\leftarrow$ CollectEpisode(env, MCTS\_policy($M_\theta$, $V_\psi$)) \\
\hspace*{4em}buffer.add(trajectory) \\
~\\
\hspace*{4em}// 阶段2：世界模型更新（可微分系统辨识） \\
\hspace*{4em}从buffer采样批次 $\{(s_t, a_t, s_{t+1})\}$ \\
\hspace*{4em}L\_model $\leftarrow ||s_{t+1} - M_\theta(s_t, a_t)||^2$ \\
\hspace*{4em}$\theta \leftarrow \theta - \eta_\theta \nabla_\theta \text{L\_model}$ \\
~\\
\hspace*{4em}// 阶段3：价值网络更新（TD学习） \\
\hspace*{4em}L\_value $\leftarrow ||V_\psi(s_t) - (r_t + \gamma \cdot V_\psi(s_{t+1}))||^2$ \\
\hspace*{4em}$\psi \leftarrow \psi - \eta_\psi \nabla_\psi \text{L\_value}$ \\
~\\
\hspace*{4em}// 阶段4：策略蒸馏（可选） \\
\hspace*{4em}$\pi_{\text{target}} \leftarrow \text{normalize}(\text{MCTS\_visit\_counts})$ \\
\hspace*{4em}L\_policy $\leftarrow \text{CrossEntropy}(\pi_\phi(\cdot|s_t), \pi_{\text{target}})$ \\
\hspace*{4em}$\phi \leftarrow \phi - \eta_\phi \nabla_\phi \text{L\_policy}$ \\
end for
\end{quote}"""

content = content.replace(code_old, code_new)

with open(tex_file, 'w', encoding='utf-8') as f:
    f.write(content)

print("Latex fixes applied successfully.")
