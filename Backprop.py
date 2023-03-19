import numpy as np
from autograd.misc import flatten

# Backpropagation step.
##
# ADAM Optimizer
##
def exponential_decay(step_size):
  if step_size > 0.001:
      step_size *= 0.999
  return step_size

def adam_minmin(grad_both, init_params_nn, init_params_nn2, callback=None, num_iters=100, step_size=0.001, b1=0.9, b2=0.999, eps=10**-8):
  x_nn, unflatten_nn = flatten(init_params_nn)
  x_nn2, unflatten_nn2 = flatten(init_params_nn2)

  m_nn, v_nn = np.zeros(len(x_nn)), np.zeros(len(x_nn))
  m_nn2, v_nn2 = np.zeros(len(x_nn2)), np.zeros(len(x_nn2))
  for i in range(num_iters):
    g_nn_uf, g_nn2_uf = grad_both(unflatten_nn(x_nn), unflatten_nn2(x_nn2), i)
    g_nn, _ = flatten(g_nn_uf)
    g_nn2, _ = flatten(g_nn2_uf)

    if callback: 
      callback(unflatten_nn(x_nn), unflatten_nn2(x_nn2), i)
    
    step_size = exponential_decay(step_size)

    # Update parameters
    m_nn = (1 - b1) * g_nn      + b1 * m_nn  # First  moment estimate.
    v_nn = (1 - b2) * (g_nn**2) + b2 * v_nn  # Second moment estimate.
    mhat_nn = m_nn / (1 - b1**(i + 1))    # Bias correction.
    vhat_nn = v_nn / (1 - b2**(i + 1))
    x_nn = x_nn - step_size * mhat_nn / (np.sqrt(vhat_nn) + eps)

    # Update parameters
    m_nn2 = (1 - b1) * g_nn2      + b1 * m_nn2  # First  moment estimate.
    v_nn2 = (1 - b2) * (g_nn2**2) + b2 * v_nn2  # Second moment estimate.
    mhat_nn2 = m_nn2 / (1 - b1**(i + 1))    # Bias correction.
    vhat_nn2 = v_nn2 / (1 - b2**(i + 1))
    x_nn2 = x_nn2 - step_size * mhat_nn2 / (np.sqrt(vhat_nn2) + eps)
  return unflatten_nn(x_nn), unflatten_nn2(x_nn2)