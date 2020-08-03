import sympy as sp
from devitoboundary import Ext_Poly

x_r = sp.symbols('x_r')
s_o = 4
n_pts = int(s_o/2)

ext = Ext_Poly(s_o, n_pts)
bc_0 = sp.Eq(ext.u(x_r), 0)
bc_2 = sp.Eq(ext.u(x_r, 2), 0)
bc_4 = sp.Eq(ext.u(x_r, 4), 0)
bcs = [bc_0, bc_2, bc_4]

ext.add_bcs(bcs)
print(ext.coeff_gen())
