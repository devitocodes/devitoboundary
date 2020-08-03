import sympy as sp
from devitoboundary import Stencil_Gen

s_o = 4
n_pts = int(s_o/2)

ext = Stencil_Gen(s_o)

bc_0 = sp.Eq(ext.u(ext.x_b), 0)
bc_2 = sp.Eq(ext.u(ext.x_b, 2), 0)
bc_4 = sp.Eq(ext.u(ext.x_b, 4), 0)
bcs = [bc_0, bc_2, bc_4]

ext.add_bcs(bcs)

# print(ext._coeff_gen(n_pts))
indy_poly, uni_poly = ext._stencils()
print(indy_poly, "\n", uni_poly)
