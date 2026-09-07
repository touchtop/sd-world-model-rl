# Sokoban中的SD-MCTS：基于可解性资源动力学的系统动力学建模

## 核心观点

在 Sokoban 中，SD 不应建模“格子状态如何转移”，而应建模：

> **可解性资源如何被 push 消耗、释放、锁死和转化为目标完成度。**

Sokoban 的真正动力机制不是箱子从一格移动到另一格，而是：

> 在不可拉回、强约束、局部动作会永久改变未来动作集合的系统中，如何把箱子运输到目标，同时不耗尽通道容量、玩家可达性、箱子机动性和目标填充顺序等“可解性资源”。

---

# 1. 第一性原理：Sokoban的主要矛盾

Sokoban 的基本事实：

1. 箱子只能被推，不能被拉；
2. 玩家移动基本可逆，但 push 通常不可逆；
3. 目标完成要求所有箱子到达目标；
4. 每次 push 同时改变未来可解空间。

因此其本质是：

> **受容量、可达性、顺序和死锁约束的多箱子运输问题。**

建议将其抽象为：

**Goal-conditioned Feasible Transport System Dynamics（GFT-SD）**

即：

> 目标条件化可行运输系统动力学模型。

---

# 2. SD应建模什么

定义两层状态。

## 2.1 底账层（符号状态）

记为：

$$
y_t=(B_t,R_t)
$$

其中：

- $B_t$：所有箱子位置；
- $R_t$：当前箱子布局下玩家可达区域。

该层保证系统满足 Markov 性。

---

## 2.2 SD抽象层

定义：

$$
z_t=\Phi(y_t)
$$

其中：

$$
z_t=(P_t,M_t,A_t,K_t,D_t,Q_t,C_t,I_t)
$$

分别表示：

| 库存 | 含义 |
|--------|--------|
| $P_t$ | 目标运输进度 |
| $M_t$ | 箱子机动性 |
| $A_t$ | 玩家推位可达性 |
| $K_t$ | 区域容量裕度 |
| $D_t$ | Deadlock债务 |
| $Q_t$ | 顺序依赖债务 |
| $C_t$ | Goal commitment |
| $I_t$ | 当前动作相关性 |

SD 真正建模的是这些库存的变化。

---

# 3. 目标运输网络

构造地图图结构：

$$
G=(V,E)
$$

对于每个目标 $g$，建立 reverse-push graph。

定义：

$$
\mathcal R_g \subseteq V
$$

若：

$$
v\in \mathcal R_g
$$

表示箱子在位置 $v$ 时存在可行 push 序列到达目标 $g$。

进一步定义：

$$
G(v)=\{g \mid v\in \mathcal R_g\}
$$

若：

$$
G(v)=\emptyset
$$

则该位置是静态 dead square。

---

# 4. 运输进度库存 P

## 4.1 成本定义

对于箱子 $i$ 与目标 $g$：

$$
c_{i,g}=
d_g(b_i)
+\lambda_{turn}\cdot turns_{i,g}
+\lambda_{access}\cdot access\_risk_{i,g}
+\lambda_{zone}\cdot zone\_risk_{i,g}
$$

若不可达：

$$
c_{i,g}=+\infty
$$

求最优匹配：

$$
\mu_t=
\arg\min_\mu
\sum_i c_{i,\mu(i)}
$$

定义剩余运输功：

$$
P_t
=\sum_i c_{i,\mu_t(i)}
$$

---

## 4.2 流量

$$
f_t^P=P_t-P_{t+1}
$$

表示当前 push 对整体运输进度的贡献。

---

# 5. 箱子机动性库存 M

定义：

$$
M_i(t)
=
\sum_d
\mathbf 1[\text{legal}(i,d)]
\mathbf 1[G(b_i+d)\neq\emptyset]
\mathbf 1[\neg deadlock]
$$

总体机动性：

$$
M_t=\sum_i M_i(t)
$$

流量：

$$
f_t^M=M_{t+1}-M_t
$$

表示未来选择空间变化。

---

# 6. 玩家可达性库存 A

对于 push：

$$
e=(i,d)
$$

所需玩家站位：

$$
s_{i,d}=b_i-d
$$

定义：

$$
A_{i,d}(t)
=\mathbf 1[s_{i,d}\in R_t]
$$

总可达性：

$$
A_t=
\sum_{i,d}
A_{i,d}(t)
$$

流量：

$$
f_t^A=A_{t+1}-A_t
$$

描述通路开放与封闭。

---

# 7. 区域容量库存 K

地图被划分为区域：

$$
\mathcal Z=\{Z_1,\ldots,Z_m\}
$$

定义容量裕度：

$$
K_Z(t)
=
n_G(Z)
-
n_{committed}(Z,t)
-
blockedEntry(Z,t)
-
wrongBoxPressure(Z,t)
$$

若：

$$
K_Z(t)<0
$$

则意味着容量约束被破坏。

流量：

$$
K_Z(t+1)
=
K_Z(t)
-f_Z^{in}
+f_Z^{out}
-f_Z^{block}
+f_Z^{unblock}
$$

---

# 8. Deadlock债务 D

定义：

$$
D_t=(D_t^H,D_t^S)
$$

其中：

- $D_t^H$：Hard Deadlock
- $D_t^S$：Soft Deadlock Pressure

Soft deadlock：

$$
D_t^S
=
\alpha_1 mobilityLoss
+
\alpha_2 matchingFragility
+
\alpha_3 capacityStress
+
\alpha_4 accessLoss
+
\alpha_5 orderViolation
$$

---

# 9. 顺序依赖债务 Q

建立偏序图：

$$
\mathcal P_t=(N_t,E_t^\prec)
$$

若：

$$
i \prec j
$$

表示 i 必须先于 j。

定义：

$$
Q_t
=
\# violatedEdges
+
\gamma \cdot
\# cycles
$$

流量：

$$
Q_{t+1}
=
Q_t
+f_t^{Q,+}
-f_t^{Q,-}
$$

---

# 10. Goal Commitment C

定义匹配矩阵：

$$
C_t\in [0,1]^{n\times n}
$$

可由 Sinkhorn 获得：

$$
C_t=
Sinkhorn(-\beta c_{i,g})
$$

定义匹配熵：

$$
H_C(t)
=
-\sum_{i,g}
C_{i,g}(t)\log C_{i,g}(t)
$$

流量：

$$
f_t^C
=
H_C(t)-H_C(t+1)
$$

---

# 11. 动作相关性 I

定义动作影响域：

$$
\Omega(a)
$$

定义当前瓶颈集合：

$$
\mathcal B_t
$$

动作相关性：

$$
I(a,t)
=
overlap(\Omega(a),\mathcal B_t)
$$

也可写为：

$$
I(a,t)
=
\eta_1\Delta P_a^+
+
\eta_2\Delta D_a^-
+
\eta_3\Delta M_a^+
+
\eta_4\Delta K_a^+
+
\eta_5\mathbf 1[a\in bottleneck]
$$

---

# 12. 整体SD转移

给定：

$$
y_t=(B_t,R_t)
$$

动作：

$$
a_t=(i,d)
$$

执行后：

$$
B_{t+1}
=
B_t\setminus\{b_i\}
\cup
\{b_i+d\}
$$

玩家可达区域：

$$
R_{t+1}
=
ReachableRegion(b_i,B_{t+1})
$$

抽象状态：

$$
z_{t+1}
=
\Phi(B_{t+1},R_{t+1})
$$

于是：

$$
\Delta z_t
=
z_{t+1}-z_t
$$

即：

$$
\Delta z_t
=
(\Delta P_t,
\Delta M_t,
\Delta A_t,
\Delta K_t,
\Delta D_t,
\Delta Q_t,
\Delta C_t,
\Delta I_t)
$$

---

# 13. SD Reward

定义：

$$
r_{SD}
=
w_P(P_t-P_{t+1})
+
w_M(M_{t+1}-M_t)
+
w_A(A_{t+1}-A_t)
+
w_K(K_{t+1}-K_t)
-
w_D(D_{t+1}-D_t)
-
w_Q(Q_{t+1}-Q_t)
-c_{push}
$$

最终奖励：

$$
r_t
=
r_{SD}
+r_{goal}
+r_{dead}
$$

---

# 14. SD-MCTS

MCTS节点：

$$
n_t=(y_t,z_t)
$$

其中：

- $y_t$：精确状态转移；
- $z_t$：抽象动力学状态。

Action Prior：

$$
\pi_{SD}(a|z_t)
\propto
\exp(
\alpha_P\Delta P_a^+
+\alpha_M\Delta M_a
+\alpha_A\Delta A_a
+\alpha_K\Delta K_a
-\alpha_D\Delta D_a
-\alpha_Q\Delta Q_a
+\alpha_I I(a,t)
)
$$

PUCT：

$$
U(s,a)
=
Q(s,a)
+
c_{puct}
\pi_{SD}(a|z)
\frac{\sqrt{N(s)}}{1+N(s,a)}
$$

---

# 15. TVA

Leaf Value：

$$
V_{leaf}(z_h)
=
V_{SD}(z_h)
+
V_{TVA}(z_h)
$$

其中：

$$
V_{SD}(z)
=
-\beta_P P
+\beta_M M
+\beta_A A
+\beta_K K
-\beta_D D
-\beta_Q Q
$$

TVA 学习的是：

> 显式 SD 世界模型无法覆盖的长期可解性残差。

---

# 16. 最终定义

整个 Sokoban-SD 可写为：

$$
\boxed{
Sokoban\text{-}SD
=
\langle
G,
\mathcal R,
\mathcal Z,
y_t,
z_t,
\mathcal A(y_t),
T,
\Phi,
r_{SD}
\rangle
}
$$

其中：

- $G$：地图结构；
- $\mathcal R$：Goal-conditioned Transport Network；
- $\mathcal Z$：区域分解；
- $y_t$：符号状态；
- $z_t$：可解性资源库存；
- $\mathcal A$：Macro Push动作；
- $T$：精确状态转移；
- $\Phi$：显式SD计算图；
- $r_{SD}$：资源流量奖励。

---

# 核心结论

> 在 Sokoban 中，SD 应建模的不是箱子运动学，而是可解性资源动力学。

每个 push 都是在：

- 目标运输进度；
- 未来机动性；
- 可达性；
- 容量裕度；
- Deadlock 风险；
- 顺序约束；

之间进行不可逆资源交换。

因此：

> **MCTS 负责规划资源交换序列；SD 负责显式刻画资源动力机制；TVA 负责学习长期可解性残差。**
