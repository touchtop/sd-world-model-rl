import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskSDModel(nn.Module):
    """可微分的系统动力学世界模型（离散事件近似版）

    更准确地匹配真实环境的离散事件动力学：
    - 任务到达：Poisson过程的期望值（可学习）
    - 任务分配：离散事件（一次分配一个）
    - 任务完成：固定速率减少（可学习）
    - 疲劳机制：积压超过阈值时效率下降（可选）
    """

    def __init__(
        self,
        n_workers: int = 2,
        arrival_rate: float = 0.7,
        max_queue: float = 50.0,
        max_backlog: float = 50.0,
        fatigue_threshold: float = None,
        fatigue_penalty: float = 0.5,
        learn_arrival_rate: bool = True,
        learn_fatigue_dynamics: bool = False,
        # Probabilistic collapse learning parameters
        learn_probabilistic_collapse: bool = False,
        init_sigmoid_center: float = 10.0,  # Initial belief for μ (can be wrong)
        init_sigmoid_temp: float = 2.0,  # Initial belief for T
        probabilistic_collapse_penalty: float = 0.25,  # Efficiency during collapse
        # NEW: Efficiency degradation learning parameters
        learn_efficiency_degradation: bool = False,
        init_efficiency_alpha: float = 0.01,  # Initial belief for α
        init_efficiency_threshold: float = 5.0,  # Initial belief for θ
    ):
        super().__init__()
        self.n_workers = n_workers
        self.max_queue = max_queue
        self.max_backlog = max_backlog
        self.fatigue_threshold = fatigue_threshold
        self.fatigue_penalty = fatigue_penalty
        self.learn_arrival_rate = learn_arrival_rate
        self.learn_fatigue_dynamics = learn_fatigue_dynamics
        self.learn_probabilistic_collapse = learn_probabilistic_collapse
        self.probabilistic_collapse_penalty = probabilistic_collapse_penalty
        # 可学习参数：各Worker的处理速率
        self._completion_raw = nn.Parameter(torch.zeros(n_workers))
        # 可学习参数：任务到达率（用softplus确保正数，范围约[0, 3]）
        if learn_arrival_rate:
            # 初始化为接近arrival_rate的值（使用expm1和clamp避免数值问题）
            arrival_tensor = torch.tensor(float(arrival_rate))
            init_val = torch.log(torch.expm1(arrival_tensor).clamp_min(1e-6))
            self._arrival_raw = nn.Parameter(init_val)
        else:
            self.register_buffer('_arrival_rate_fixed', torch.tensor(arrival_rate))
        # 可学习参数：累积疲劳动力学
        if learn_fatigue_dynamics:
            self._fatigue_accum_threshold = nn.Parameter(torch.tensor(3.0))
            self._fatigue_accum_rate = nn.Parameter(torch.tensor(0.2))
            self._fatigue_recovery = nn.Parameter(torch.tensor(0.05))
            self._fatigue_collapse_thresh = nn.Parameter(torch.tensor(10.0))
            self._collapse_penalty_logit = nn.Parameter(torch.tensor(-1.0))
        # Learnable probabilistic collapse parameters
        if learn_probabilistic_collapse:
            # μ and T can start with wrong prior and be learned
            self._sigmoid_center = nn.Parameter(torch.tensor(float(init_sigmoid_center)))
            # T = softplus(raw) + 0.1 => raw = inverse_softplus(T - 0.1)
            # Using log(expm1(x)) as inverse of softplus
            target_t = max(float(init_sigmoid_temp) - 0.1, 0.01)
            init_raw = torch.log(torch.expm1(torch.tensor(target_t)).clamp_min(1e-6))
            self._sigmoid_temp_raw = nn.Parameter(init_raw)
            # Store T prior for regularization
            self._t_prior = init_sigmoid_temp
        # NEW: Learnable efficiency degradation parameters
        self.learn_efficiency_degradation = learn_efficiency_degradation
        if learn_efficiency_degradation:
            # α: quadratic coefficient (must be positive) - use softplus
            # efficiency = 1 - α * (fatigue - θ)² when fatigue > θ
            alpha_tensor = torch.tensor(float(init_efficiency_alpha))
            self._efficiency_alpha_raw = nn.Parameter(
                torch.log(torch.expm1(alpha_tensor).clamp_min(1e-6))
            )
            # θ: threshold (can be any positive value)
            self._efficiency_threshold = nn.Parameter(torch.tensor(float(init_efficiency_threshold)))
            # Store priors for regularization
            self._alpha_prior = init_efficiency_alpha
            self._theta_prior = init_efficiency_threshold

    def get_t_regularization_loss(self, lambda_t: float = 0.1) -> torch.Tensor:
        """Log-normal prior regularization on T to prevent divergence.

        Loss = λ * (log(T) - log(T_prior))²
        This penalizes T for moving away from the initial belief.
        """
        if not self.learn_probabilistic_collapse:
            return torch.tensor(0.0)
        current_t = F.softplus(self._sigmoid_temp_raw) + 0.1
        log_t = torch.log(current_t)
        log_t_prior = torch.log(torch.tensor(self._t_prior))
        return lambda_t * (log_t - log_t_prior) ** 2

    def get_efficiency_coef(self, fatigue: torch.Tensor) -> torch.Tensor:
        """Compute efficiency degradation coefficient.

        One-sided Lorentzian: efficiency = 1 / (1 + α * max(0, fatigue - θ)²)
        Only degrades when fatigue > threshold, full efficiency below threshold.
        """
        if not self.learn_efficiency_degradation:
            return torch.ones_like(fatigue)
        alpha = F.softplus(self._efficiency_alpha_raw)
        theta = self._efficiency_threshold
        diff = F.relu(fatigue - theta)  # One-sided: only penalize when fatigue > theta
        efficiency = 1.0 / (1.0 + alpha * diff * diff)
        return efficiency

    @property
    def efficiency_alpha(self) -> torch.Tensor:
        """Get current α value."""
        if self.learn_efficiency_degradation:
            return F.softplus(self._efficiency_alpha_raw)
        return torch.tensor(0.0)

    @property
    def efficiency_threshold(self) -> torch.Tensor:
        """Get current θ value."""
        if self.learn_efficiency_degradation:
            return self._efficiency_threshold
        return torch.tensor(0.0)

    @property
    def arrival_rate(self) -> torch.Tensor:
        if self.learn_arrival_rate:
            return F.softplus(self._arrival_raw)  # 确保正数
        return self._arrival_rate_fixed

    @property
    def completion_coef(self) -> torch.Tensor:
        # 使用sigmoid * 2确保在合理范围[0, 2]内
        return torch.sigmoid(self._completion_raw) * 2.0

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        更准确地模拟离散事件环境：
        1. 任务到达：期望arrival_rate个新任务
        2. 任务分配：根据action，将一个任务从队列移到worker
        3. 任务完成：worker backlog减少completion_coef
        """
        queue = state[..., 0]
        backlogs = state[..., 1:1 + self.n_workers]
        # Preserve extra dimensions (e.g., cumulative_fatigue) if present
        has_extra = state.shape[-1] > 1 + self.n_workers
        extra_dims = state[..., 1 + self.n_workers:] if has_extra else None

        # Step 1: 任务到达（使用期望值）
        new_queue = queue + self.arrival_rate

        # Step 2: 任务分配（离散：从队列移到worker）
        # 使用clamp替代二值阶跃，保持梯度流通
        has_task = torch.clamp(new_queue, 0.0, 1.0)
        action_onehot = F.one_hot(action.clamp(0, self.n_workers - 1), self.n_workers).float()
        valid_assign = (action < self.n_workers).float().unsqueeze(-1)
        assignment = action_onehot * valid_assign * has_task.unsqueeze(-1)

        # 队列减1，worker积压加1
        new_queue = new_queue - assignment.sum(dim=-1)
        new_backlogs = backlogs + assignment

        # Step 3: 任务完成（worker处理）
        # 计算有效处理速率（考虑疲劳）
        effective_rates = self.completion_coef.expand_as(new_backlogs)
        if self.fatigue_threshold is not None:
            # 使用sigmoid平滑过渡，避免不可微分的阶跃
            fatigue_factor = torch.sigmoid(
                (new_backlogs - self.fatigue_threshold) * 2.0
            )
            # 疲劳时效率降低到fatigue_penalty倍
            effective_rates = effective_rates * (
                1.0 - fatigue_factor * (1.0 - self.fatigue_penalty)
            )

        # 累积疲劳导致的坍塌效应（可学习）
        if self.learn_fatigue_dynamics and has_extra:
            curr_fatigue = extra_dims[..., -1]
            is_collapsed = torch.sigmoid((curr_fatigue - self._fatigue_collapse_thresh) * 5.0)
            penalty_val = torch.sigmoid(self._collapse_penalty_logit)
            collapse_coef = 1.0 - is_collapsed * (1.0 - penalty_val)
            effective_rates = effective_rates * collapse_coef.unsqueeze(-1)

        # Probabilistic collapse: expected penalty based on learned sigmoid parameters
        if self.learn_probabilistic_collapse:
            if not has_extra:
                raise ValueError(
                    "learn_probabilistic_collapse requires fatigue in state; "
                    "set observe_fatigue=True so state includes cumulative_fatigue."
                )
            curr_fatigue = extra_dims[..., -1]
            sigmoid_temp = F.softplus(self._sigmoid_temp_raw) + 0.1  # Ensure T > 0
            collapse_prob = torch.sigmoid((curr_fatigue - self._sigmoid_center) / sigmoid_temp)
            # Expected efficiency: (1 - P) * normal + P * penalty
            expected_coef = 1.0 - collapse_prob * (1.0 - self.probabilistic_collapse_penalty)
            effective_rates = effective_rates * expected_coef.unsqueeze(-1)

        # NEW: Efficiency degradation based on learned α and θ
        if self.learn_efficiency_degradation:
            if not has_extra:
                raise ValueError(
                    "learn_efficiency_degradation requires fatigue in state; "
                    "set observe_fatigue=True so state includes cumulative_fatigue."
                )
            curr_fatigue = extra_dims[..., -1]
            efficiency_coef = self.get_efficiency_coef(curr_fatigue)
            effective_rates = effective_rates * efficiency_coef.unsqueeze(-1)

        # 使用logsumexp实现的soft minimum: min(a,b) ≈ -logsumexp(-βa, -βb)/β
        # 保证非负且梯度流通
        beta = 5.0
        stacked = torch.stack([-beta * effective_rates, -beta * new_backlogs], dim=-1)
        completion = -torch.logsumexp(stacked, dim=-1) / beta
        new_backlogs = new_backlogs - completion

        # 上下界约束
        new_queue = torch.clamp(new_queue, min=0.0, max=self.max_queue)
        new_backlogs = torch.clamp(new_backlogs, min=0.0, max=self.max_backlog)

        # Reconstruct state
        if has_extra:
            if self.learn_fatigue_dynamics:
                curr_fatigue = extra_dims[..., -1]
                max_backlog = new_backlogs.max(dim=-1).values
                overload_score = torch.sigmoid(
                    (max_backlog - F.softplus(self._fatigue_accum_threshold)) * 5.0
                )
                accum_rate = F.softplus(self._fatigue_accum_rate)
                recovery_rate = F.softplus(self._fatigue_recovery)
                delta = overload_score * accum_rate - (1.0 - overload_score) * recovery_rate
                new_fatigue = F.relu(curr_fatigue + delta)
                if extra_dims.shape[-1] > 1:
                    new_extra = torch.cat([extra_dims[..., :-1], new_fatigue.unsqueeze(-1)], dim=-1)
                else:
                    new_extra = new_fatigue.unsqueeze(-1)
                return torch.cat([new_queue.unsqueeze(-1), new_backlogs, new_extra], dim=-1)
            else:
                return torch.cat([new_queue.unsqueeze(-1), new_backlogs, extra_dims], dim=-1)
        return torch.cat([new_queue.unsqueeze(-1), new_backlogs], dim=-1)

    def get_collapse_prob(self, state: torch.Tensor) -> torch.Tensor:
        """Compute collapse probability for explicit supervision (BCE loss)."""
        if not self.learn_probabilistic_collapse:
            raise RuntimeError("get_collapse_prob requires learn_probabilistic_collapse=True")
        has_extra = state.shape[-1] > 1 + self.n_workers
        if not has_extra:
            raise ValueError("State must include cumulative_fatigue for collapse probability")
        curr_fatigue = state[..., -1]
        sigmoid_temp = F.softplus(self._sigmoid_temp_raw) + 0.1
        z = (curr_fatigue - self._sigmoid_center) / sigmoid_temp
        z = torch.clamp(z, -100.0, 100.0)
        return torch.sigmoid(z)
