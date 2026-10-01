"""Clock-time generators. Rows sum to 0 and act on the right."""

import jax.numpy as jnp

from discrete_average_generator.schedules import lam


def mixture_rate(p, rate):
    """Mixture generator from a posterior probability vector p.

    Q[i, j] = rate * p[j] for j != i, and
    Q[i, i] = -sum_{j != i} Q[i, j].
    Rows sum to 0. Off-diagonal entries are nonnegative when rate >= 0 and p >= 0.
    Probabilities are rows, so the forward equation is dp/dt = p @ Q.
    """
    p = jnp.asarray(p)
    rate = jnp.asarray(rate)
    vocab = p.shape[-1]
    rate = jnp.broadcast_to(rate, p.shape[:-1])
    eye = jnp.eye(vocab, dtype=p.dtype)
    off = rate[..., None, None] * p[..., None, :] * (1.0 - eye)
    exit_rate = jnp.sum(off, axis=-1)
    return off - eye * exit_rate[..., None, :]


def scaled_generator(t, kappa, kappa_dot):
    """Vocabulary-3 generator whose scale is lam(t) = kappa_dot / (1 - kappa).

    The rate pattern is not rank-one, so Q(t) does not commute with a generic
    average generator. Off-diagonal entries stay nonnegative for kappa(t) in [0, 1).
    """
    t = jnp.asarray(t)
    dtype = t.dtype
    scale = lam(t, kappa, kappa_dot)
    base = jnp.array(
        [
            [0.0, 0.4, 1.1],
            [0.7, 0.0, 0.2],
            [0.3, 0.9, 0.0],
        ],
        dtype=dtype,
    )
    wobble = jnp.array(
        [
            [0.0, 0.15, -0.05],
            [-0.1, 0.0, 0.2],
            [0.1, -0.15, 0.0],
        ],
        dtype=dtype,
    )
    eye = jnp.eye(base.shape[0], dtype=dtype)
    off = scale * (base + kappa(t) * wobble)
    off = jnp.clip(off, 0.0) * (1.0 - eye)
    exit_rate = jnp.sum(off, axis=-1)
    return off - eye * exit_rate[None, :]
