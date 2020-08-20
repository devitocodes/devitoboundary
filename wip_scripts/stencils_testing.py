from devitoboundary import Stencil_Gen
from devito import Eq

s_o = 4
n_pts = int(s_o/2)

ext = Stencil_Gen(s_o, stencil_file='test_cache.dat')

bc_0 = Eq(ext.u(ext.x_b), 0)
bc_2 = Eq(ext.u(ext.x_b, 2), 0)
bc_4 = Eq(ext.u(ext.x_b, 4), 0)
bcs = [bc_0, bc_2]  # , bc_4]

ext.add_bcs(bcs)

ext.all_variants(2, stencil_out='test_cache.dat')

print(ext.subs(eta_l=-0.2))
print(ext.subs(eta_r=1.75))
print(ext.subs(eta_l=-0.2, eta_r=0.2))
print(ext.subs(eta_l=-0.7, eta_r=0.2))
print(ext.subs(eta_l=-0.2, eta_r=0.7))
print(ext.subs(eta_l=-0.72, eta_r=1.21))
print(ext.subs(eta_l=-1.84, eta_r=1.63))
print(ext.subs(eta_l=-0.84, eta_r=0.63))
print(ext.subs(eta_l=-1.73, eta_r=0.6))
