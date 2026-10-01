"""Noise schedules kappa and the mixture intensity lam."""

import jax.numpy as jnp


def kappa_linear(t):
    """kappa(t) = t."""
    return jnp.asarray(t)


def kappa_dot_linear(t):
    """Derivative of kappa(t) = t."""
    return jnp.ones_like(jnp.asarray(t))


def kappa_quadratic(t):
    """kappa(t) = t**2."""
    t = jnp.asarray(t)
    return t ** 2


def kappa_dot_quadratic(t):
    """Derivative of kappa(t) = t**2."""
    t = jnp.asarray(t)
    return 2.0 * t


def lam(t, kappa, kappa_dot):
    """lam(t) = kappa_dot(t) / (1 - kappa(t)), for kappa(t) < 1."""
    return kappa_dot(t) / (1.0 - kappa(t))
