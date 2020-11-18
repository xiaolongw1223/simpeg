import numpy as np
import scipy.sparse as sp

from .. import Utils
from .base import BaseRegularization


class BaseCoupling(BaseRegularization):

    '''
    BaseCoupling
    
        ..note::
            
            You should inherit from this class to create your own
            coupling term.
    '''
    def __init__(self, mesh, indActive, mapping, **kwargs):
        
        self.as_super = super(BaseCoupling, self)
        self.as_super.__init__(mesh, indActive=indActive, mapping=mapping)
        
    def deriv(self):
        ''' 
        First derivative of the coupling term with respect to individual models.
        Returns an array of dimensions [k*M,1],
        k: number of models we are inverting for.
        M: number of cells in each model.
        '''
        raise NotImplementedError(
            "The method deriv has not been implemented for {}".format(
                self.__class__.__name__
            )
        )

    def deriv2(self):
        ''' 
        Second derivative of the coupling term with respect to individual models.
        Returns either an array of dimensions [k*M,1], or
        sparse matrix of dimensions [k*M, k*M]
        k: number of models we are inverting for.
        M: number of cells in each model.
        '''
        raise NotImplementedError(
            "The method _deriv2 has not been implemented for {}".format(
                self.__class__.__name__
            )
        )

    def __call__(self):
        ''' Returns the computed value of the coupling term. '''
        raise NotImplementedError(
            "The method __call__ has not been implemented for {}".format(
                self.__class__.__name__
            )
        )
            
            
###############################################################################
#                                                                             #
#                            Joint Total Variation                            #
#                                                                             #
###############################################################################


class JTV(BaseCoupling):
    '''
    The joint total variation constraint for joint inversions.
    
    ..math::
        \phi_c({\mathbf m}_{\mathbf1},{\mathbf m}_{\mathbf2})=\lambda\sum_{i=1}^M
        \sqrt{\left|\nabla{{\mathbf m}_{\mathbf1}}_i\right|^2+
        \left|\nabla{{\mathbf m}_2}_i\right|^2}\triangle V_i
    
    All methods assume that we are working with two models only.
    '''
    def __init__(self, mesh, indActive, mapping, **kwargs):
        self.as_super = super(JTV, self)
        self.as_super.__init__(mesh, indActive, mapping, **kwargs)
        self.map1, self.map2 = mapping.maps ### Assume a map has been passed for each model.
        
        assert mesh.dim in (2,3), 'JTV is only defined for 2D or 3D'       
    
    @property
    def epsilon(self):
        if getattr(self, '_epsilon', None) is None:
            
            self._epsilon = 1e-10

        return self._epsilon

    @epsilon.setter
    def epsilon(self, val):
        self._epsilon = val

    def models(self, ind_models):
        '''
        Method to pass models to Joint Total Variation object and ensures models are compatible
        for further use. Checks that models are of same size.
        
        :param container of numpy.ndarray ind_models: [model1, model2,...]
        
        rtype: list of numpy.ndarray models: [model1, model2,...]
        return: models
        '''
        models = []
        n = len(ind_models) # number of individual models
        for i in range(n):
            ### check that the models are either a list, tuple, or np.ndarray
            assert isinstance(ind_models[i], (list,tuple,np.ndarray))
            if isinstance(ind_models[i], (list,tuple)):
                ind_models[i] = np.array(ind_models[i]) ### convert to np.ndarray
        
        ### check if models are of same size
        it = iter(ind_models)
        length = len(next(it))
        if not all(len(l)==length for l in it):
            raise ValueError('not all models are of the same size!')
        
        for i in range(n):
            models.append(ind_models[i])

        return models


    def A(self, m):
        '''
        construct a A matrix to sum the gradient in different components at each cell center

        :param numpy.ndarray m: model inverted from one geophysical data set
        :rtype: sp.sparse.csr_matrix

        :return: |1 0 0 1 0 0 1 0 0|: dimensions: M*2M(2D) or M*3M(3D): M: number of model cells.
                 |0 1 0 0 1 0 0 1 0|
                 |0 0 1 0 0 1 0 0 1|
  
        '''
        
        tmp = sp.eye(len(m))

        if self.regmesh.mesh.dim == 2:

            return sp.hstack([tmp, tmp])
        
        elif self.regmesh.mesh.dim == 3:

            return sp.hstack([tmp, tmp, tmp])

    
    @property
    def D(self):
        '''
        stack finit difference matrixs with different components.
        '''
        if self.regmesh.mesh.dim == 2:
            Dx = self.regmesh.aveFx2CC.dot(self.regmesh.cellDiffx) 
            Dy = self.regmesh.aveFy2CC.dot(self.regmesh.cellDiffy)
            
            D = sp.vstack([Dx, Dy])

        elif self.regmesh.mesh.dim == 3:
            Dx = self.regmesh.aveFx2CC.dot(self.regmesh.cellDiffx) 
            Dy = self.regmesh.aveFy2CC.dot(self.regmesh.cellDiffy)
            Dz = self.regmesh.aveFz2CC.dot(self.regmesh.cellDiffz)
            
            D = sp.vstack([Dx, Dy, Dz])

        return D 


    @property
    def vol(self):
        '''
        reduced volumn vector

        :rtype: numpy.ndarray
        :return: reduced cell volumn
        '''
        return self.regmesh.vol


    def JTV_core(self, m1, m2):
        
        assert m1.size == m2.size, 'models must be of same size'
        if isinstance(m1, (list,tuple)):
            m1 = np.array(m1)
        if isinstance(m2, (list, tuple)):
            m2 = np.array(m2)
        
        D = self.D
        A = self.A(m1)
        D_m1, D_m2 = D.dot(m1), D.dot(m2)
        
        return A.dot(D_m1**2) + A.dot(D_m2**2) + self.epsilon



    def __call__(self, model):
        '''
        Computes the sum of all joint total variation values at all cell centers.
        
        :param numpy.ndarray model: stacked array of individual models
                                    np.c_[model1, model2,...]
        
        :rtype: float
        :returns: the computed value of the joint total variation term.
        '''
        m1 = self.map1*model
        m2 = self.map2*model
        m1, m2 = self.models([m1,m2])
        V = self.vol  
        
        core = self.JTV_core(m1, m2)
        temp2 = core**0.5

        result = V.T.dot(temp2)
        return result
    
    
    def deriv(self, model):
        '''
        Computes the Jacobian of the joint total variation.
        
        :param list of numpy.ndarray ind_models: [model1, model2,...]
        
        :rtype: numpy.ndarray
        :return: result: gradient of the joint total variation with respect to model1, model2
        '''
        m1 = self.map1*model
        m2 = self.map2*model
        m1, m2 = self.models([m1,m2])
        V = self.vol
        A = self.A(m1)
        D = self.D
        core = self.JTV_core(m1, m2)

        temp1 = A.T.dot(Utils.sdiag(core**(-0.5))).dot(V) # dimension: 3M*1
        dc_dm1 = D.T.dot(Utils.sdiag(temp1)).dot(D.dot(m1))
        dc_dm2 = D.T.dot(Utils.sdiag(temp1)).dot(D.dot(m2))

        result = np.concatenate((dc_dm1,dc_dm2))

        return result
    

    def deriv2(self, model, v=None):
        '''
        Computes the Hessian of the joint total variation.
        
        :param list of numpy.ndarray ind_models: [model1, model2, ...]
        :param numpy.ndarray v: vector to be multiplied by Hessian
        :rtype: scipy.sparse.csr_matrix if v is None
                numpy.ndarray if v is not None
        :return Hessian matrix: | h11, h21 | :dimension 2M*2M if v is None
                                |          |
                                | h12, h22 | 
                Hessian multiplied by vector if v is not None
        '''
        

        
        m1 = self.map1*model
        m2 = self.map2*model
        m1, m2 = self.models([m1, m2])
        V = self.vol
        A = self.A(m1)
        D = self.D
        core = self.JTV_core(m1, m2)
        
        if v is not None:
            assert v.size == 2*m1.size, 'vector v must be of size 2*M'
            v1 = self.map1*v
            v2 = self.map2*v
        
        # h12
        temp1 = D.T.dot(Utils.sdiag(D.dot(m2))) # M*3M
        temp2 = A.T.dot(Utils.sdiag(core**(-1.5))) # 3M*M
        temp3 = D.T.dot(Utils.sdiag(D.dot(m1))).dot(A.T.dot(Utils.sdiag(V))) #M*M
        d_dm2_dc_dm1 = -temp1.dot(temp2).dot(temp3.T)

        # h21
        temp1 = D.T.dot(Utils.sdiag(D.dot(m1))) # M*3M
        temp3 = D.T.dot(Utils.sdiag(D.dot(m2))).dot(A.T.dot(Utils.sdiag(V))) #M*M
        d_dm1_dc_dm2 = -temp1.dot(temp2).dot(temp3.T)

        # h11
        temp1 = A.T.dot(Utils.sdiag(core**(-0.5))).dot(V)
        term1 = D.T.dot(Utils.sdiag(temp1)).dot(D)
        
        temp1 = D.T.dot(Utils.sdiag(D.dot(m1)))
        temp2 = A.T.dot(Utils.sdiag(core**(-1.5))).dot(Utils.sdiag(V))
        temp3 = A.dot(Utils.sdiag(D.dot(m1))).dot(D)
        term2 = temp1.dot(temp2).dot(temp3)
        d2c_dm1 = term1 - term2

        # h22
        temp1 = D.T.dot(Utils.sdiag(D.dot(m2)))
        temp2 = A.T.dot(Utils.sdiag(core**(-1.5))).dot(Utils.sdiag(V))
        temp3 = A.dot(Utils.sdiag(D.dot(m2))).dot(D)
        term2 = temp1.dot(temp2).dot(temp3)
        d2c_dm2 = term1 - term2

        temp1 = sp.vstack((d2c_dm1,d_dm2_dc_dm1))
        temp2 = sp.vstack((d_dm1_dc_dm2, d2c_dm2))
        result = sp.hstack((temp1,temp2))
        result = sp.csr_matrix(result)
        
        if v is not None:
            d2c_dm1 = d2c_dm1.dot(v1)
            d2c_dm2 = d2c_dm2.dot(v2)
            d_dm2_dc_dm1 = d_dm2_dc_dm1.dot(v1)
            d_dm1_dc_dm2 = d_dm1_dc_dm2.dot(v2)
            result = np.concatenate((d2c_dm1 + d_dm1_dc_dm2, d_dm2_dc_dm1 + d2c_dm2))
        else:
            temp1 = sp.vstack((d2c_dm1,d_dm2_dc_dm1))
            temp2 = sp.vstack((d_dm1_dc_dm2, d2c_dm2))
            result = sp.hstack((temp1,temp2))
            result = sp.csr_matrix(result)
        
        return result
