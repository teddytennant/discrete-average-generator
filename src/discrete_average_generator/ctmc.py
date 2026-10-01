"""Transition matrices and the average-generator identity.

Row-vector convention: P[i, j] = Prob(X_r = j | X_t = i).
Forward Kolmogorov: dP/ds = P @ Q(s).
Backward Kolmogorov: dP/dt = -Q(t) @ P.
"""

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm


def integrate_transition(t, r, rate_fn, n_steps=256):
    """Integrate dP/ds = P @ Q(s) from t to r with classical RK4.

    rate_fn(s) returns a generator at clock time s. The result is the
    transition matrix P_{t->r}.
    """
    t = jnp.asarray(t)
    r = jnp.asarray(r)
    q0 = rate_fn(t)
    vocab = q0.shape[-1]
    eye = jnp.eye(vocab, dtype=q0.dtype)
    ds = (r - t) / n_steps

    def body(carry, i):
        s0 = t + i * ds

        def field(matrix, s):
            return matrix @ rate_fn(s)

        k1 = field(carry, s0)
        k2 = field(carry + 0.5 * ds * k1, s0 + 0.5 * ds)
        k3 = field(carry + 0.5 * ds * k2, s0 + 0.5 * ds)
        k4 = field(carry + ds * k3, s0 + ds)
        nxt = carry + (ds / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return nxt, None

    transition, _ = jax.lax.scan(body, eye, jnp.arange(n_steps))
    return transition


def integrate_transition_expm(t, r, rate_fn, n_steps=64):
    """Piecewise-frozen matrix exponential for the same forward equation.

    On each step, Q is frozen at the midpoint and the step map is
    expm(ds * Q), applied on the right.
    """
    t = jnp.asarray(t)
    r = jnp.asarray(r)
    q0 = rate_fn(t)
    vocab = q0.shape[-1]
    eye = jnp.eye(vocab, dtype=q0.dtype)
    ds = (r - t) / n_steps

    def body(carry, i):
        mid = t + (i + 0.5) * ds
        step = expm(ds * rate_fn(mid))
        return carry @ step, None

    transition, _ = jax.lax.scan(body, eye, jnp.arange(n_steps))
    return transition


def average_generator(transition, r, t):
    """U(t, r) = (P_{t->r} - I) / (r - t), for r > t.

    The denominator is clock time, not kappa(r) - kappa(t). Proposition 1
    (equation 8) is derived for this object: U = Q_t - (t - r) dU/dt + Q_t @ U.
    Dividing by the kappa gap instead makes that identity fail on the quadratic
    schedule, where kappa(r) - kappa(t) is not r - t.
    """
    vocab = transition.shape[-1]
    eye = jnp.eye(vocab, dtype=transition.dtype)
    return (transition - eye) / (r - t)


def self_consistency_rhs(u_bar, q_t, r, t):
    """dU/dt at fixed r, from the backward equation.

    U = (P - I) / (r - t) and dP/dt = -Q_t @ P, so

        dU/dt = U / (r - t) - Q_t / (r - t) - Q_t @ U.

    Equivalently, equation 8: U = Q_t - (t - r) dU/dt + Q_t @ U.
    The product is left-multiplication. Right-multiplication agrees only when
    the two matrices commute.
    """
    gap = r - t
    return u_bar / gap - q_t / gap - q_t @ u_bar


def mixture_transition(t, r, posterior, kappa, kappa_dot):
    """Exact P_{t->r} for Q(s) = lam(s) (1 p - I) with p constant in time.

    lam(s) = kappa_dot(s) / (1 - kappa(s)) integrates to
    Lambda = (1 - kappa(r)) / (1 - kappa(t)), and
    P = Lambda * I + (1 - Lambda) * 1 p.
    kappa_dot is part of the defining rate even though the closed form depends
    on kappa only.
    """
    del kappa_dot  # closed form uses the integral of kappa_dot/(1-kappa)
    posterior = jnp.asarray(posterior)
    stay = (1.0 - kappa(r)) / (1.0 - kappa(t))
    vocab = posterior.shape[-1]
    eye = jnp.eye(vocab, dtype=posterior.dtype)
    rows = jnp.broadcast_to(posterior, (vocab, vocab))
    return stay * eye + (1.0 - stay) * rows
