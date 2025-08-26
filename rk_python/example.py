import numpy as np
from scipy.integrate import solve_ivp


# Super trivial example on how to use scipy runge kutta solver

def f(t, y):
    return np.array([y[1], -y[0]])

t_span = (0, 10)
y0 = [1.0, 0.0]

sol = solve_ivp(f, t_span, y0, method='DOP853', t_eval=np.linspace(0, 10, 1000), rtol=1.0e-12, atol=1.0e-12)

import matplotlib.pyplot as plt
cos_curve = np.cos(sol.t) - sol.y[0]

plt.plot(sol.t, cos_curve, 'k-', label='cos(t)')
# plt.plot(sol.t, sol.y[0], 'r--', label='y(t) from ODE')

plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.savefig('solution_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("Relative Energy loss: ", 1. - (sol.y[0][-1]**2 + sol.y[1][-1]**2) )