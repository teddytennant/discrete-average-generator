"""Regression loss for the average generator.

Equation 8, stopped on the right-hand side:

    u_tgt = Q - (t - r) (dU/dt + Q @ U)

The transport term (r - t) Q U is required. L is the mean square of
(u_theta - stop_gradient(u_tgt)).
"""

import jax
import jax.numpy as jnp

from discrete_average_generator.ctmc import equation8_target
from discrete_average_generator.rates import mixture_rate
from discrete_average_generator.schedules import lam


def generator_matching_loss(model, x, t, r, posterior, kappa, kappa_dot):
    """Mean square error of (u_theta - stop_gradient(u_tgt)).

    The network outputs a probability q_hat over the vocabulary. The average
    generator row is u = (q_hat - one_hot(x)) / (r - t), clock time.

    u_tgt is the row of equation 8 at the current state. dU/dt is the full
    time derivative of the network's U, and Q @ U is the transport term.
    The whole right-hand side is stopped, as in the paper's U-prediction loss,
    so the quotient piece of dU/dt does not cancel the live residual.

    When r == t the reparameterization is singular. Fall back to the mean
    square of (q_hat - true_posterior).
    """
    posterior = jnp.asarray(posterior)
    alpha = r - t
    vocab = posterior.shape[-1]

    def main_loss():
        states = jnp.arange(vocab)

        def u_matrix(tt):
            def row(y):
                q_hat = model(y, tt, r)
                eye = jax.nn.one_hot(y, vocab, dtype=q_hat.dtype)
                return (q_hat - eye) / (r - tt)

            return jax.vmap(row)(states)

        u_mat, du_mat = jax.jvp(u_matrix, (t,), (jnp.ones_like(t),))
        q_mat = mixture_rate(posterior, lam(t, kappa, kappa_dot))
        target = equation8_target(u_mat, du_mat, q_mat, r, t)
        u_theta = u_mat[x]
        u_tgt = jax.lax.stop_gradient(target[x])
        return jnp.mean((u_theta - u_tgt) ** 2)

    def endpoint_loss():
        q_hat = model(x, t, r)
        return jnp.mean((q_hat - posterior) ** 2)

    return jax.lax.cond(jnp.abs(alpha) < 1e-8, endpoint_loss, main_loss)
