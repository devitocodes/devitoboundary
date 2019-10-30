import numpy as np

from devito import Grid, Function

import pandas as pd

__all__ = ['Boundary']


class Boundary(object):

    """
    An object that contains the data relevant for implementing the
    immersed boundary method on a given domain.

    :param param0: Description.
    :param param1: (Optional) Description.

    Note: To add.
    """

    def __init__(self, Grid, BoundaryFunction, InverseBoundaryFunction = None,
                 method_order = 4):
        self._method_order = method_order
        
        """ Some things we'll need from, and derived from, Grid: """
        shape = np.asarray(Grid.shape)
        extent = np.asarray(Grid.extent)
        spacing = extent/(shape-1)
    
        """
        Step1: Check what kind of boundary function we have.
        To start, this will only work for boundaries of the form:
        x_b = const : 1d.
        y_b = f(x), x_b = f^-1(y_b) : 2d.
        """
        # FIX ME: Add support for boundaries specified in text files.
        if not callable(BoundaryFunction):
            raise NotImplementedError
        
        self._primary_nodes(BoundaryFunction, shape, extent, spacing)
        
        if 0 < np.asarray(Grid.shape).size < 3:
            """ Now work out the full list. """
            self._node_list(shape)
        else:
            raise NotImplementedError
            
        """ Generate eta's. FIX ME: Shouldn't be 'private'? """
        self._eta_list(BoundaryFunction, InverseBoundaryFunction,
                       shape, extent, spacing)
        
        """ Finally, generate our stencil. """
        self._fd_stencil()
        
    @property
    def method_order(self):
        """
        Order of the FD discretisation.
        Currently this is only implemented for 4th order stencils.
        """
        return self._method_order
        

    def _primary_nodes(self, BoundaryFunction, shape, extent, spacing):
        
        """
        Compute the primary boundary nodes.
        Other nodes in the boundary sub-domain can be derived from
        these.
        """

        if shape.size > 2:
            raise NotImplementedError
        
        if shape.size == 1:
            x_coords = np.linspace(0,extent[0],shape[0])
            # In this case the boundary is a single node
            boundary = BoundaryFunction()
            pn = np.floor(boundary/spacing[0]).astype(int)
            # These two cases shouldn't occur in 1D:
            if pn < 0 or pn >= shape[0]:
                raise ValueError("Given boundary location is not \
                                  in the computational domain.")
        elif shape.size == 2:
            # FIX ME: Add support for non (0,0) origins
            x_coords = np.linspace(0,extent[0],shape[0])
            y_coords = np.linspace(0,extent[1],shape[1])
            boundary = BoundaryFunction(x_coords)
            pn = np.floor(boundary/spacing[1]).astype(int)
            # Represent nodes outside the computational domain with -1.
            pn[pn < 0] = -1 
            pn[pn >= shape[1]] = -1
        else:
            # FIX ME: Add 3D case etc.
            raise NotImplementedError
        
        self._primary_nodes = pn
        
        return self._primary_nodes

    def _node_list(self, shape):
        
        """
        Generate list of possible nodes (with redundancy)
        that require their stencil modified.
        """
        # FIX ME: Needs testing for complicated topography.
        
        """
        Make list from creating a box around primary nodes
        then remove duplicate entries.
        """
        pn = self._primary_nodes
        
        """ Default box size """
        ds = max(np.array([self.method_order/2-1, 1], dtype=int))
        
        """ Check if we're dealing with a 1D case. """
        if pn.size == 1:
            node_dict = ()
            for j in range(0,ds+1):
                node_dict += (pn-ds+j,)
            self._node_list = node_dict
            return self._node_list
        
        dpnf = np.zeros((pn.size,), dtype=int)
        dpnb = np.zeros((pn.size,), dtype=int)
        
        for j in range(0,pn.size-1):
            if (pn[j] < 0) or (pn[j+1] < 0):
                dpnf[j] = pn.size # Our 'int NaN'
            else:
                dpnf[j] = pn[j+1]-pn[j]
            
        if dpnf[-2] == pn.size:
            dpnf[-1] = pn.size
        
        for j in range(1,pn.size):
            if (pn[j] < 0) or (pn[j-1] < 0):
                dpnb[j] = pn.size # Our 'int NaN'
            else:
                dpnb[j] = pn[j]-pn[j-1]
        if dpnb[1] == pn.size:
            dpnb[0] = pn.size
            
        """ Node boxes """
        box = np.zeros((pn.size,), dtype=int)
        
        for j in range(0,pn.size):
            d = np.array([abs(dpnb[j]), abs(dpnf[j])], dtype=int)
            d[d >= shape[1]] = -1
            dm = max(d)
            if dm < 0:
                box[j] = 0
            elif dm <= ds+1:
                box[j] = ds
            else:
                box[j] = dm-1
        
        """ Create boundary domain node list - initial size unknown """
        # FIX ME: Should be a better algorithm than this available.
        node_dict = ()
        for i in range(0,pn.size):
            for j in range(-box[i],box[i]):
                for k in range(-box[i],box[i]):
                    node_dict = node_dict + ((i+j,pn[i]+k),)

        """ Remove 'out of bounds' entries. """
        node_dict = tuple((t for t in node_dict if not min(t) < 0))
        node_dict = tuple((t for t in node_dict if not max(t) >= pn.size))
        """ Remove repeated entries. """
        node_dict = tuple(set(node_dict))
                
        self._node_list = node_dict
        
        return self._node_list
    
    def _eta_list(self, BoundaryFunction, InverseBoundaryFunction,
                  shape, extent, spacing):
    
        pn = self._primary_nodes
        node_list = self._node_list
        
        x_coords = np.linspace(0,extent[0],shape[0])
        y_coords = np.linspace(0,extent[1],shape[1])
        
        x_list = ()
        y_list = ()
        
        for j in range(0,len(node_list)):
            
            etax = 0
            etay = 0
            
            """ Compute etay (the easy bit). """
            element_node = node_list[j]
            etay = (BoundaryFunction(x_coords[element_node[0]])- \
                                     y_coords[element_node[1]])/spacing[1]
            
            """ Now compute etax. """
            if InverseBoundaryFunction == None:
                # FIX ME: Implement an attempt to use fsolve (possibly with a warning).
                raise NotImplementedError
            else:
                """ Note: Possibly a multivalued result or NaN. """
                etax = (InverseBoundaryFunction(y_coords[element_node[1]])- \
                                                x_coords[element_node[0]])/spacing[0]
            
            x_list = x_list + (etax,)
            y_list = y_list + (etay,)
            
         
        nodes = pd.Series(node_list)
        ex = pd.Series(x_list)
        ey = pd.Series(y_list)
        
        """ Data structure """
        eta_list = pd.DataFrame({'Node': nodes,
                              'etax': ex, 'etay': ey})
        is_below =  eta_list['etay'] > -2*np.pi*np.finfo(float).eps
        eta_list = eta_list[is_below]
        
        self._eta_list = eta_list
        
        return self._eta_list
    
    def _fd_stencil(self):
    
        eta_list = self._eta_list
        
        nnodes = len(eta_list.index)
        
        if self.method_order != 4:
            raise NotImplementedError
        
        # Stencils:
        def w1(eta):
            w = np.zeros(self.method_order+1)
            w[0] = 1/12
            w[1] = -4/3
            w[2] = (29+eta*(93+58*eta))/(12*(1+eta)*(1+2*eta))
            w[3] = -(5+7*eta)/(3*(1+2*eta))
            w[4] = 0
            return w
        def w2(eta):
            w = np.zeros(self.method_order+1)
            w[0] = 1/12
            w[1] = -(8+3*eta*(3+eta))/((2+eta)*(3+2*eta))
            w[2] = (29+eta*(49+22*eta))/(4*(1+eta)*(3+2*eta))
            w[3] = -4/3
            w[4] = 0
            return w
        def w3(eta):
            w = np.zeros(self.method_order+1)
            w[0] = 1/12
            w[1] = -(2+eta*(21+eta))/(3*(1+eta)*(1+2*eta))
            w[2] = (6+eta*(79+2*eta))/(12*eta*(1+2*eta))
            w[3] = 0
            w[4] = 0
            return w
        def w4(eta):
            w = np.zeros(self.method_order+1)
            w[0] = (eta*(10*eta-7))/(2*(2+eta)*(3+2*eta))
            w[1] = -(2*eta*(2+5*eta))/((1+eta)*(3+2*eta))
            w[2] = 5/2
            w[3] = 0
            w[4] = 0
            return w
        def wn(eta):
            w = np.zeros(self.method_order+1)
            w[0] = 1/12
            w[1] = -4/3
            w[2] = 5/2
            w[3] = -4/3
            w[4] = 1/12
            return w
        
        D_xx_list = ()
        D_yy_list = ()
        
        for j in range(0,nnodes):
            ex = eta_list.iat[j,1]
            ey = eta_list.iat[j,2]
            
            if ex.size > 1:
                if abs(ex[0]-ex[1]) < 2*np.pi*np.finfo(float).eps:
                    ex = ex[0]
                else:
                    ex = ex[abs(ex) < 2+2*np.pi*np.finfo(float).eps]
            if ex.size > 1:
                raise NotImplementedError
            if abs(ex) > 2+2*np.pi*np.finfo(float).eps:
                w = wn(ex)
                D_xx_list = D_xx_list + (w,)
            elif 1.5 <= abs(ex) <= 2+2*np.pi*np.finfo(float).eps:
                w = w1(abs(np.mod(ex,1)))
                if ex < 0:
                    w = w[::-1]
                D_xx_list = D_xx_list + (w,)
            elif 1.0 < abs(np.mod(ex,1)) < 1.5:
                w = w2(abs(np.mod(ex,1)))
                if ex < 0:
                    w = w[::-1]
                D_xx_list = D_xx_list + (w,)
            elif 0.5 <= abs(ex) <= 1:
                w = w3(abs(ex))
                if ex < 0:
                    w = w[::-1]
                D_xx_list = D_xx_list + (w,)
            elif 0.0+2*np.pi*np.finfo(float).eps < abs(ex) < 0.5:
                w = w4(abs(ex))
                if ex < 0:
                    w = w[::-1]
                D_xx_list = D_xx_list + (w,)
            else:
                w = wn(ex)
                D_xx_list = D_xx_list + (w,)
                
            if abs(ey) > 2+2*np.pi*np.finfo(float).eps:
                w = wn(ey)
                D_yy_list = D_yy_list + (w,)
            elif 1.5 <= abs(ey) <= 2+2*np.pi*np.finfo(float).eps:
                w = w1(abs(np.mod(ey,1)))
                if ey < 0:
                    w = w[::-1]
                D_yy_list = D_yy_list + (w,)
            elif 1.0 < abs(np.mod(ey,1)) < 1.5:
                w = w2(abs(np.mod(ey,1)))
                if ey < 0:
                    w = w[::-1]
                D_yy_list = D_yy_list + (w,)
            elif 0.5 <= abs(ey) <= 1:
                w = w3(abs(ey))
                if ey < 0:
                    w = w[::-1]
                D_yy_list = D_yy_list + (w,)
            elif 0.0+2*np.pi*np.finfo(float).eps < abs(ey) < 0.5:
                w = w4(abs(ey))
                if ey < 0:
                    w = w[::-1]
                D_yy_list = D_yy_list + (w,)
            else:
                w = wn(ey)
                D_yy_list = D_yy_list + (w,)
                
        D_xx = pd.Series(D_xx_list)
        D_yy = pd.Series(D_yy_list)
                
        fd_stencil = pd.DataFrame({'Node': eta_list["Node"].values,
                              'D_xx_stencil': D_xx, 'D_yy_stencil': D_yy})
        
        self._fd_stencil = fd_stencil
        
        return self._fd_stencil

    def stencil(self):
        return self._fd_stencil
