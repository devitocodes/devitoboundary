import numpy as np
from devito import Eq
from devitoboundary import Stencil_Gen

s_o = 4

ext = Stencil_Gen(s_o)

bc_0 = Eq(ext.u(ext.x_b), 0)
bc_2 = Eq(ext.u(ext.x_b, 2), 0)
bc_4 = Eq(ext.u(ext.x_b, 4), 0)
bcs = [bc_0, bc_2, bc_4]

ext.add_bcs(bcs)

ext.all_variants(2)

dx = 1


def quadratic(x, eta_l, eta_r):
    return (x - eta_r*dx)*(x - eta_l*dx)


l_vals = np.linspace(-0.75, -1.75, 10)
r_vals = np.linspace(0.75, 1.75, 10)
avg_tol = 0

for i in range(l_vals.shape[0]):
    for j in range(r_vals.shape[0]):
        stencil = ext.subs(eta_l=l_vals[i], eta_r=r_vals[j])
        stencil /= dx**2
        derivative = 0
        for k in range(stencil.shape[0]):
            derivative += stencil[k]*quadratic(k*dx - dx*s_o/2, l_vals[i], r_vals[j])
        assert 100*abs(derivative-2)/2 < 16., "Accuracy of calculated derivative insufficient"
        avg_tol += abs(derivative-2)
assert 100*(avg_tol/100)/2 < 8, "Average accuracy of calculated derivatives insufficient"
