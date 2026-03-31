import re

tex_file = "/home/data/round/code/sd_world_model_rl/docs/ijoc/ijoc_paper_draft_v2_zh.tex"

with open(tex_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Fix longtable headers (remove minipage from headers)
content = re.sub(r'\\begin\{minipage\}\[b\]\{.*?columnwidth\}\\raggedright\n?(.*?)\\strut\n?\\end\{minipage\}', r'\1', content, flags=re.DOTALL)

# 2. Fix inline dashed lists for linebreaks
content = content.replace(' - 到达：', '\\\\\n- 到达：')
content = content.replace(' - 服务率：', '\\\\\n- 服务率：')
content = content.replace(' - 效率函数：', '\\\\\n- 效率函数：')
content = content.replace(' - \\(f_t\\)：', '\\\\\n- \\(f_t\\)：')
content = content.replace(' - \\(a_t \\in', '\\\\\n- \\(a_t \\in')
content = content.replace(' - \\(\\kappa = 1.0\\)：', '\\\\\n- \\(\\kappa = 1.0\\)：')
content = content.replace(' - \\(\\beta = 0.1\\)：', '\\\\\n- \\(\\beta = 0.1\\)：')
content = content.replace(' - \\(I_{i,t}\\)：', '\\\\\n- \\(I_{i,t}\\)：')
content = content.replace(' - \\(B_{i,t}\\)：', '\\\\\n- \\(B_{i,t}\\)：')
content = content.replace(' - \\text{在途}_{i,t}\\)：', '\\\\\n- \\text{在途}_{i,t}\\)：')
content = content.replace(' - 信息延迟：', '\\\\\n- 信息延迟：')
content = content.replace(' - 运输延迟：', '\\\\\n- 运输延迟：')

# 3. Fix Greek characters in Pseudo-code
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


# 4. Fix missing parameter tables due to unparsed markdown
table_1_old = r"""\textbf{参数}： \textbar{} 参数 \textbar{} 真值 \textbar{} 初始信念
\textbar{} 误差 \textbar{}
\textbar{}------\textbar{}------\textbar{}---------\textbar{}------\textbar{}
\textbar{} \(\alpha\)（退化率） \textbar{} 0.1 \textbar{} 0.001
\textbar{} 100× \textbar{} \textbar{} \(\tau\)（疲劳阈值） \textbar{}
5.0 \textbar{} 12.0 \textbar{} 2.4× \textbar{}"""

table_1_new = r"""\textbf{参数}：

\begin{table}[h]
\centering
\begin{tabular}{llll}
\toprule
参数 & 真值 & 初始信念 & 误差 \\
\midrule
\(\alpha\)（退化率） & 0.1 & 0.001 & 100$\times$ \\
\(\tau\)（疲劳阈值） & 5.0 & 12.0 & 2.4$\times$ \\
\bottomrule
\end{tabular}
\end{table}"""

content = content.replace(table_1_old, table_1_new)

table_2_old = r"""\textbf{参数}： \textbar{} 参数 \textbar{} 真值 \textbar{} MPC信念
\textbar{} 误差 \textbar{}
\textbar{}------\textbar{}------\textbar{}--------\textbar{}------\textbar{}
\textbar{} \(F\)（崩溃阈值） \textbar{} 10 \textbar{} 50 \textbar{}
5×高估 \textbar{}"""

table_2_new = r"""\textbf{参数}：

\begin{table}[h]
\centering
\begin{tabular}{llll}
\toprule
参数 & 真值 & MPC信念 & 误差 \\
\midrule
\(F\)（崩溃阈值） & 10 & 50 & 5$\times$高估 \\
\bottomrule
\end{tabular}
\end{table}"""

content = content.replace(table_2_old, table_2_new)


# 5. Fix wide table in 7.4.3
summary_table_pattern = r'(\\hypertarget\{ux5173ux952eux7ed3ux679cux603bux7ed3\}.*?\\begin\{longtable\}\[\]\{@\{\})lll(@\{\}\})'
content = re.sub(summary_table_pattern, r'\1p{5.5cm} p{4cm} p{5.5cm}\2', content, flags=re.DOTALL)


# 6. Fix box drawing characters in verbatim block
content = content.replace('├──', '|--')
content = content.replace('└──', '+--')
content = content.replace('│', '|')


with open(tex_file, 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixes applied successfully.")
