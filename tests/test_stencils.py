import numpy as np
import sympy as sp
from devitoboundary import Stencil_Gen
from devito import Eq

s_o = 4  # Order of the discretization


class TestStencils:
    """
    A class for testing the stencils generated by Stencil_Gen
    """

    def test_convergence(self):
        """
        Convergence test to check that calculated derivatives trend towards
        the actual with decreasing grid increment.
        """
        ext = Stencil_Gen(s_o)

        bc_0 = Eq(ext.u(ext.x_b), 0)
        bc_2 = Eq(ext.u(ext.x_b, 2), 0)
        bc_4 = Eq(ext.u(ext.x_b, 4), 0)
        bcs = [bc_0, bc_2, bc_4]

        ext.add_bcs(bcs)

        ext.all_variants(2)

        for i in range(1, len(ext.stencil_list)):
            prev = None
            print("\n Variant", i)
            test_eta_r = 0.5*s_o - 0.5*i + 0.25
            test_stencil = ext.stencil_list[0][i]
            for j in range(1, 11):
                dx = 1/j
                evaluated = test_stencil
                for k in range(len(ext.stencil_list)):
                    evaluated = evaluated.subs(ext._f[k-int(s_o/2)],
                                               np.sin(np.pi-test_eta_r*dx+(k-int(s_o/2))*dx))
                evaluated = evaluated.subs(ext._eta_r, test_eta_r)
                evaluated /= dx**2
                diff = evaluated + np.sin(np.pi-test_eta_r*dx)
                if prev is not None:
                    if abs(diff) > abs(prev) and not np.isclose(np.finfo(np.float32).eps, float(diff), rtol=10):
                        raise RuntimeError("Convergence failed. Diff: %f Prev: %f" % (abs(diff), abs(prev)))
                prev = diff

    def test_extrapolations(self):
        """
        Test to check that extrapolations recover polynomials of equivalent order.
        """
        order_max = 4  # The maximum stencil order to test up to

        for i in range(1, int(order_max/2)+1):
            ext = Stencil_Gen(s_o)

            def zero_even_gen(order_M):
                """Returns list of zeroed even bcs for a given number of derivatives"""
                even_bcs = []
                for j in range(int(order_M/2)+1):
                    even_bcs.append(Eq(ext.u(ext.x_b, 2*j), 0))
                return even_bcs

            zero_bcs = zero_even_gen(2*i)
            ext.add_bcs(zero_bcs)
            e_poly_coeffs = ext._coeff_gen(2*i)
            # e_poly_coeffs = ext_poly(zero_bcs, 2*i, 2*i)

            e_poly = 0
            for j in range(len(e_poly_coeffs)):
                e_poly += e_poly_coeffs[ext._a[j]]*ext._x_c**j

            def odd_poly(val, order):
                """Generates an odd term polynomial of given order"""
                poly = 0
                for j in range(1, order+1):
                    if j % 2 != 0:
                        poly += val**j
                return poly

            t_poly = odd_poly((ext._x_c-ext.x_b), 2*i)

            for j in range(int(order_max/2)):
                e_poly = e_poly.subs([(ext._x[j], ext.x_b - (1+j)),
                                      (ext._u_x[j], t_poly.subs(ext._x_c, ext.x_b - (1+j)))])

            assert sp.simplify(e_poly-t_poly) == 0, "Polynomial was not recovered"