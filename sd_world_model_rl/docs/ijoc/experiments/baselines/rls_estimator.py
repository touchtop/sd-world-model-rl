"""
Recursive Least Squares (RLS) Parameter Estimator.

Provides online parameter estimation for Adaptive MPC baseline.
"""
import numpy as np
from typing import Tuple, Optional


class RLSEstimator:
    """Recursive Least Squares estimator for online parameter learning.

    Supports estimation for:
    1. Efficiency degradation parameters (α, θ)
    2. Delayed collapse parameters (α, β, F)

    Attributes:
        param_dim: Number of parameters to estimate.
        theta_hat: Current parameter estimates.
        P: Covariance matrix.
        lambda_rls: Forgetting factor (0 < λ ≤ 1).
    """

    def __init__(
        self,
        param_dim: int = 2,
        initial_params: Optional[np.ndarray] = None,
        initial_covariance: float = 100.0,
        forgetting_factor: float = 0.99,
    ):
        """Initialize RLS estimator.

        Args:
            param_dim: Number of parameters.
            initial_params: Initial parameter estimates.
            initial_covariance: Initial diagonal covariance value.
            forgetting_factor: λ ∈ (0, 1], lower = faster adaptation.
        """
        self.param_dim = param_dim
        self.lambda_rls = forgetting_factor

        if initial_params is not None:
            self.theta_hat = np.array(initial_params, dtype=np.float64)
        else:
            self.theta_hat = np.zeros(param_dim, dtype=np.float64)

        self.P = np.eye(param_dim, dtype=np.float64) * initial_covariance

        # Statistics
        self.n_updates = 0
        self.total_error = 0.0

    def update(self, phi: np.ndarray, y: float) -> float:
        """Perform one RLS update step.

        Standard RLS update equations:
            K = P * φ / (λ + φᵀ * P * φ)
            θ = θ + K * (y - φᵀ * θ)
            P = (P - K * φᵀ * P) / λ

        Args:
            phi: Regression vector (features).
            y: Observation (target).

        Returns:
            Prediction error before update.
        """
        phi = np.array(phi, dtype=np.float64).flatten()

        # Prediction
        y_pred = np.dot(phi, self.theta_hat)
        error = y - y_pred

        # Kalman gain
        P_phi = self.P @ phi
        denominator = self.lambda_rls + np.dot(phi, P_phi)
        K = P_phi / denominator

        # Parameter update
        self.theta_hat = self.theta_hat + K * error

        # Covariance update
        self.P = (self.P - np.outer(K, phi @ self.P)) / self.lambda_rls

        # Regularize P to prevent numerical issues
        self.P = 0.5 * (self.P + self.P.T)  # Ensure symmetry
        min_var = 1e-6
        self.P = np.maximum(self.P, np.eye(self.param_dim) * min_var)

        # Statistics
        self.n_updates += 1
        self.total_error += abs(error)

        return error

    def get_params(self) -> np.ndarray:
        """Get current parameter estimates."""
        return self.theta_hat.copy()

    def get_uncertainty(self) -> np.ndarray:
        """Get parameter uncertainty (diagonal of P)."""
        return np.diag(self.P).copy()


class EfficiencyDegradationEstimator:
    """Specialized estimator for efficiency degradation parameters (α, θ).

    Model: efficiency = 1 / (1 + α * max(0, fatigue - θ)²)

    Linearized form (when fatigue > θ):
        1/efficiency - 1 = α * (fatigue - θ)²
    """

    def __init__(
        self,
        init_alpha: float = 0.001,
        init_theta: float = 12.0,
        forgetting_factor: float = 0.99,
    ):
        self.alpha_hat = init_alpha
        self.theta_hat = init_theta
        self.forgetting_factor = forgetting_factor

        # Separate RLS for alpha (given theta estimate)
        self._alpha_rls = RLSEstimator(
            param_dim=1,
            initial_params=np.array([init_alpha]),
            initial_covariance=10.0,
            forgetting_factor=forgetting_factor,
        )

        # History for theta estimation
        self._observations = []
        self._theta_update_interval = 50
        self._min_observations_for_theta = 20

    def update(self, fatigue: float, observed_efficiency: float) -> Tuple[float, float]:
        """Update parameter estimates from observation.

        Args:
            fatigue: Observed fatigue level.
            observed_efficiency: Observed efficiency (0 < eff ≤ 1).

        Returns:
            (alpha_hat, theta_hat) current estimates.
        """
        # Numerical stability: validate efficiency
        if observed_efficiency <= 1e-6 or observed_efficiency > 1.0:
            return self.alpha_hat, self.theta_hat

        # Store observation
        self._observations.append((fatigue, observed_efficiency))

        # Only update alpha when fatigue > theta_hat (informative region)
        if fatigue > self.theta_hat + 0.1:  # Add small margin for stability
            # Linearized observation with numerical protection
            y = 1.0 / observed_efficiency - 1.0
            y = min(y, 1000.0)  # Prevent extreme values
            diff = fatigue - self.theta_hat
            phi = np.array([diff ** 2])

            self._alpha_rls.update(phi, y)
            # Bound alpha to reasonable range
            raw_alpha = self._alpha_rls.get_params()[0]
            self.alpha_hat = np.clip(raw_alpha, 1e-6, 10.0)

        # Periodically update theta estimate
        if len(self._observations) % self._theta_update_interval == 0:
            self._update_theta()

        return self.alpha_hat, self.theta_hat

    def _update_theta(self):
        """Update theta using grid search on recent observations."""
        if len(self._observations) < self._min_observations_for_theta:
            return

        recent = self._observations[-100:]  # Use last 100 observations
        best_theta = self.theta_hat
        best_loss = float('inf')

        # Grid search around current estimate
        for theta_candidate in np.linspace(
            max(0, self.theta_hat - 5), self.theta_hat + 5, 21
        ):
            loss = 0.0
            for fatigue, eff in recent:
                if fatigue > theta_candidate:
                    diff = fatigue - theta_candidate
                    pred_y = self.alpha_hat * diff ** 2
                    true_y = 1.0 / eff - 1.0
                    loss += (pred_y - true_y) ** 2
            if loss < best_loss:
                best_loss = loss
                best_theta = theta_candidate

        # Smooth update
        self.theta_hat = 0.9 * self.theta_hat + 0.1 * best_theta

    def get_params(self) -> Tuple[float, float]:
        """Get current (α, θ) estimates."""
        return self.alpha_hat, self.theta_hat


class DelayedCollapseEstimator:
    """Specialized estimator for delayed collapse parameters (α, β, F).

    Model: Δf = (α + β) * a - β

    This is linear in (α+β, β), can use standard RLS.
    F is estimated from collapse observations.
    """

    def __init__(
        self,
        init_alpha: float = 1.0,
        init_beta: float = 0.1,
        init_F: float = 50.0,
        forgetting_factor: float = 0.99,
    ):
        self.alpha_hat = init_alpha
        self.beta_hat = init_beta
        self.F_hat = init_F

        # RLS for (α+β, β) jointly
        # Δf = (α+β)*a - β = [a, -1] @ [α+β, β]ᵀ
        self._dynamics_rls = RLSEstimator(
            param_dim=2,
            initial_params=np.array([init_alpha + init_beta, init_beta]),
            initial_covariance=10.0,
            forgetting_factor=forgetting_factor,
        )

        # Collapse threshold estimation
        self._collapse_fatigue_values = []
        self._max_safe_fatigue = 0.0

    def update(
        self,
        action: int,
        fatigue_before: float,
        fatigue_after: float,
        collapsed: bool = False,
    ) -> Tuple[float, float, float]:
        """Update parameter estimates.

        Args:
            action: Action taken (0=rest, 1=work).
            fatigue_before: Fatigue before action.
            fatigue_after: Fatigue after action.
            collapsed: Whether system collapsed.

        Returns:
            (alpha_hat, beta_hat, F_hat) current estimates.
        """
        # Update dynamics parameters (α, β)
        delta_f = fatigue_after - fatigue_before
        phi = np.array([float(action), -1.0])
        y = delta_f

        self._dynamics_rls.update(phi, y)
        params = self._dynamics_rls.get_params()

        # Extract α and β from (α+β, β) with numerical bounds
        alpha_plus_beta = np.clip(params[0], 0.01, 100.0)
        self.beta_hat = np.clip(params[1], 0.001, 50.0)
        self.alpha_hat = np.clip(alpha_plus_beta - self.beta_hat, 0.01, 50.0)

        # Update F estimate
        if collapsed:
            # F is estimated from fatigue level AFTER action (when collapse occurred)
            self._collapse_fatigue_values.append(fatigue_after)
            # F is the minimum fatigue at which collapse occurred
            self.F_hat = min(self._collapse_fatigue_values)
        else:
            # Track maximum safe fatigue
            self._max_safe_fatigue = max(self._max_safe_fatigue, fatigue_after)
            # F must be above max safe fatigue
            if self.F_hat < self._max_safe_fatigue:
                self.F_hat = self._max_safe_fatigue + 1.0

        return self.alpha_hat, self.beta_hat, self.F_hat

    def get_params(self) -> Tuple[float, float, float]:
        """Get current (α, β, F) estimates."""
        return self.alpha_hat, self.beta_hat, self.F_hat
