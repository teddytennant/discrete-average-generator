"""Probability reparameterization of one average-generator row."""

import jax
import jax.numpy as jnp


def average_from_probability(q_hat, x, r, t):
    """U[z] = (q_hat[z] - 1_{z=x}) / (r - t).

    If q_hat is a probability vector, this row sums to 0. The denominator is
    clock time, matching U = (P - I) / (r - t). On the linear schedule this is
    the same as dividing by kappa(r) - kappa(t).
    """
    q_hat = jnp.asarray(q_hat)
    gap = r - t
    one_hot = jax.nn.one_hot(x, q_hat.shape[-1], dtype=q_hat.dtype)
    return (q_hat - one_hot) / gap


def probability_from_average(u_row, x, r, t):
    """Inverse map: q_hat = one_hot(x) + (r - t) * U."""
    u_row = jnp.asarray(u_row)
    gap = r - t
    one_hot = jax.nn.one_hot(x, u_row.shape[-1], dtype=u_row.dtype)
    return one_hot + gap * u_row
