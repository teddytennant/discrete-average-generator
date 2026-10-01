"""Kolmogorov identity, generator constraints, reparameterization, Potts."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from discrete_average_generator import (
    FROZEN_EPS,
    average_from_probability,
    average_generator,
    enumerate_configs,
    equation8_target,
    integrate_transition,
    integrate_transition_expm,
    mixture_rate,
    mixture_transition,
    potts_distribution,
    probability_from_average,
    scaled_generator,
    self_consistency_rhs,
)
from discrete_average_generator.schedules import (
    kappa_dot_linear,
    kappa_dot_quadratic,
    kappa_linear,
    kappa_quadratic,
)

SCHEDULES = {
    "linear": (kappa_linear, kappa_dot_linear),
    "quadratic": (kappa_quadratic, kappa_dot_quadratic),
}


def _total_variation_gap(lhs, rhs):
    return jnp.max(jnp.abs(lhs - rhs))


@pytest.mark.parametrize("schedule_name", ["linear", "quadratic"])
def test_self_consistency_matches_finite_difference(schedule_name):
    kappa, kappa_dot = SCHEDULES[schedule_name]
    t = jnp.asarray(0.25, dtype=jnp.float64)
    r = jnp.asarray(0.62, dtype=jnp.float64)
    h = jnp.asarray(1e-6, dtype=jnp.float64)
    n_steps = 240

    def rate_fn(s):
        return scaled_generator(s, kappa, kappa_dot)

    def u_at(tt):
        transition = integrate_transition(tt, r, rate_fn, n_steps=n_steps)
        return average_generator(transition, r, tt), transition

    u_bar, transition = u_at(t)
    u_plus, _ = u_at(t + h)
    u_minus, _ = u_at(t - h)
    du_fd = (u_plus - u_minus) / (2.0 * h)

    q_t = rate_fn(t)
    rhs = self_consistency_rhs(u_bar, q_t, r, t)
    gap = r - t

    # Cross-check the integrator: RK4 vs piecewise matrix exponential, and a finer RK4.
    coarse = integrate_transition_expm(t, r, rate_fn, n_steps=80)
    fine = integrate_transition(t, r, rate_fn, n_steps=n_steps * 2)
    assert float(jnp.max(jnp.abs(transition - fine))) < 1e-8
    assert float(jnp.max(jnp.abs(transition - coarse))) < 1e-5

    assert float(_total_variation_gap(du_fd, rhs)) < 1e-6

    # Right-multiplication is the wrong side of the backward equation.
    rhs_right = u_bar / gap - q_t / gap - u_bar @ q_t
    assert float(_total_variation_gap(du_fd, rhs_right)) > 1e-3

    # Sign error on the left-multiplication term.
    rhs_sign = u_bar / gap - q_t / gap + q_t @ u_bar
    assert float(_total_variation_gap(du_fd, rhs_sign)) > 1e-2


@pytest.mark.parametrize("schedule_name", ["linear", "quadratic"])
def test_rate_ubar_and_reconstructed_transition(schedule_name):
    kappa, kappa_dot = SCHEDULES[schedule_name]
    t = jnp.asarray(0.2, dtype=jnp.float64)
    r = jnp.asarray(0.55, dtype=jnp.float64)
    posterior = jax.nn.softmax(jnp.array([0.2, -0.4, 0.7, 0.1], dtype=jnp.float64))
    rate = kappa_dot(t) / (1.0 - kappa(t))
    generator = mixture_rate(posterior, rate)

    row_sum = jnp.sum(generator, axis=-1)
    off = generator * (1.0 - jnp.eye(generator.shape[-1]))
    assert jnp.allclose(row_sum, 0.0, atol=1e-12)
    assert jnp.all(off >= -1e-12)

    # Elementwise definition: off-diagonal is rate * p[j].
    expected_off = rate * posterior
    for i in range(posterior.shape[0]):
        for j in range(posterior.shape[0]):
            if i == j:
                continue
            assert jnp.allclose(generator[i, j], expected_off[j], atol=1e-12)

    def rate_fn(s):
        return mixture_rate(posterior, kappa_dot(s) / (1.0 - kappa(s)))

    numerical = integrate_transition(t, r, rate_fn, n_steps=200)
    closed = mixture_transition(t, r, posterior, kappa, kappa_dot)
    assert jnp.allclose(numerical, closed, atol=1e-8)

    u_bar = average_generator(closed, r, t)
    assert jnp.allclose(jnp.sum(u_bar, axis=-1), 0.0, atol=1e-12)

    reconstructed = jnp.eye(closed.shape[0]) + (r - t) * u_bar
    assert jnp.allclose(reconstructed, closed, atol=1e-12)
    assert jnp.allclose(jnp.sum(reconstructed, axis=-1), 1.0, atol=1e-12)
    assert jnp.all(reconstructed >= -1e-12)


def test_review_pair_equation8_includes_transport():
    """Review pair: quadratic scaled generator at t=0.11, r=0.83.

    Equation 8 holds. The training target is equation8_target, which keeps the
    transport term (r - t) Q U. The formula that drops it does not.
    """
    t = jnp.asarray(0.11, dtype=jnp.float64)
    r = jnp.asarray(0.83, dtype=jnp.float64)

    def rate_fn(s):
        return scaled_generator(s, kappa_quadratic, kappa_dot_quadratic)

    transition = integrate_transition(t, r, rate_fn, n_steps=256)
    u_bar = average_generator(transition, r, t)
    q_t = rate_fn(t)
    du = self_consistency_rhs(u_bar, q_t, r, t)

    eq8 = equation8_target(u_bar, du, q_t, r, t)
    eq8_gap = float(jnp.max(jnp.abs(u_bar - eq8)))
    dropped = q_t - (t - r) * du
    dropped_gap = float(jnp.max(jnp.abs(eq8 - dropped)))
    assert eq8_gap < 1e-12
    assert dropped_gap > 0.2


def test_reparameterization_round_trip():
    q_hat = jax.nn.softmax(jnp.array([0.3, -0.8, 0.2, 1.1], dtype=jnp.float64))
    x = jnp.asarray(2)
    kappa_r = jnp.asarray(0.8)
    kappa_t = jnp.asarray(0.15)
    u_row = average_from_probability(q_hat, x, kappa_r, kappa_t)
    recovered = probability_from_average(u_row, x, kappa_r, kappa_t)
    assert jnp.allclose(recovered, q_hat, atol=1e-12)
    assert jnp.allclose(jnp.sum(u_row), 0.0, atol=1e-12)
    # Implied transition row is the probability vector, hence sums to 1.
    assert jnp.allclose(jnp.sum(recovered), 1.0, atol=1e-12)


def test_potts_distribution_matches_enumeration():
    q = potts_distribution()
    assert q.shape == (9,)
    assert jnp.all(q >= 0.0)
    assert jnp.allclose(jnp.sum(q), 1.0, atol=1e-6)

    configs = enumerate_configs()
    assert configs.shape == (9, 2)
    agree = (configs[:, 0] == configs[:, 1]).astype(FROZEN_EPS.dtype)
    log_weights = 1.5 * agree + FROZEN_EPS
    expected = jax.nn.softmax(log_weights)
    assert jnp.allclose(q, expected, atol=1e-6)

    redraw = 0.3 * jax.random.normal(jax.random.PRNGKey(0), (9,), dtype=jnp.float32)
    assert jnp.allclose(FROZEN_EPS, redraw)
