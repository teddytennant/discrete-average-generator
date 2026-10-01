"""Regression loss for the average generator.

u_tgt = Q_denoise + ((kappa(r) - kappa(t)) / kappa_dot(t)) * jvp_t(u_theta)
L = mean square of (u_theta - stop_gradient(u_tgt)).
"""

import jax
import jax.numpy as jnp

from discrete_average_generator.rates import mixture_rate
from discrete_average_generator.schedules import lam


def generator_matching_loss(model, x, t, r, posterior, kappa, kappa_dot):
    """Mean square error of (u_theta - stop_gradient(u_tgt)).

    The network outputs a probability q_hat over the vocabulary. The average
    generator row is u = (q_hat - one_hot(x)) / (r - t), clock time.

    jvp_t is the network time derivative mapped into generator coordinates,
    with the schedule denominator held fixed. Differentiating the quotient
    1/(kappa(r)-kappa(t)) would add a copy of u into the tangent. That copy
    cancels u_theta against the (alpha/kappa_dot)*jvp term, and the stopped
    target would no longer supervise q_hat.

    When r == t, kappa(r) - kappa(t) is zero and that reparameterization is
    singular. Fall back to the mean square of (q_hat - true_posterior).
    """
    posterior = jnp.asarray(posterior)
    alpha = r - t

    def main_loss():
        def q_of(tt):
            return model(x, tt, r)

        q_hat, dq_hat = jax.jvp(q_of, (t,), (jnp.ones_like(t),))
        u_theta = (q_hat - jax.nn.one_hot(x, q_hat.shape[-1], dtype=q_hat.dtype)) / alpha
        # Clock-time denominator, held fixed in the tangent: du/dt = dq/dt / (r - t).
        jvp_t = dq_hat / alpha
        rate = lam(t, kappa, kappa_dot)
        q_denoise = mixture_rate(posterior, rate)[x]
        # Equation 8 rearranged, with the network's own time derivative stopped
        # so the target does not depend on the parameters being matched.
        u_tgt = q_denoise + (r - t) * jvp_t
        return jnp.mean((u_theta - jax.lax.stop_gradient(u_tgt)) ** 2)

    def endpoint_loss():
        # r == t reduction: supervise the network probability directly.
        q_hat = model(x, t, r)
        return jnp.mean((q_hat - posterior) ** 2)

    return jax.lax.cond(jnp.abs(alpha) < 1e-8, endpoint_loss, main_loss)
