import numpy as np
import scipy.sparse as sp

from .. import Utils
from .base import BaseRegularization


# test github

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
#                                Cross-Gradient                               #
#                                                                             #
###############################################################################


class CrossGradient(BaseCoupling):
    '''
    The cross-gradient constraint for joint inversions.

    ..math::
        \phi_c(\mathbf{m_1},\mathbf{m_2}) = \lambda \sum_{i=1}^{M} \|
        \nabla \mathbf{m_1}_i \times \nabla \mathbf{m_2}_i \|^2

    All methods assume that we are working with two models only.
    '''
    def __init__(self, mesh, indActive, mapping, **kwargs):
        self.as_super = super(CrossGradient, self)
        self.as_super.__init__(mesh, indActive, mapping, **kwargs)
        self.map1, self.map2 = mapping.maps ### Assume a map has been passed for each model.

        assert mesh.dim in (2,3), 'Cross-Gradient is only defined for 2D or 3D'

    def models(self, ind_models):
        '''
        Method to pass models to CrossGradient object and ensures models are compatible
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

    def calculate_gradient(self, model):
        '''
        Calculate the spatial gradients of the model using central difference.

        :param numpy.ndarray model: model
        :param bool vectorize: return gradients in 1D if True
        :param bool noNaN: return gradients without any numpy.nan caused by actvMap if True.

        :rtype: tuple of numpy.ndarray
        :return: (x-gradient, y-gradient) if 2D
                 (x-gradient, y-gradient, z-gradient) if 3D
        '''

        if self.regmesh.mesh.dim == 2:
            x_grad = self.regmesh.aveFx2CC.dot(self.regmesh.cellDiffx.dot(model))
            y_grad = self.regmesh.aveFy2CC.dot(self.regmesh.cellDiffy.dot(model))

            return x_grad, y_grad

        elif self.regmesh.mesh.dim == 3:
            x_grad = self.regmesh.aveFx2CC.dot(self.regmesh.cellDiffx.dot(model))
            y_grad = self.regmesh.aveFy2CC.dot(self.regmesh.cellDiffy.dot(model))
            z_grad = self.regmesh.aveFz2CC.dot(self.regmesh.cellDiffz.dot(model))

            return x_grad, y_grad, z_grad

    def gradient_list(self, x_grad, y_grad, z_grad=np.array([])):
        '''
        Concatenates gradient components into a single array.
        [[x_grad1, y_grad1, z_grad1],
         [x_grad2, y_grad2, z_grad2],
         [x_grad3, y_grad3, z_grad3],...]

        :param numpy.ndarray x_grad: x-gradients
        :param numpy.ndarray y_grad: y-gradients
        :param numpy.ndarray z_grad: z-gradients

        :rtype: numpy.ndarray
        :return: gradient_vector: array where each row represents a model cell,
                 and each column represents a component of the gradient.
        '''
        if z_grad.size == 0:
            gradient_vector = np.c_[x_grad, y_grad]
        else:
            gradient_vector = np.c_[x_grad, y_grad, z_grad]

        return gradient_vector

    def construct_norm_vectors(self, model, fltr=True, fltr_per=0.05):
        '''
        Computes the norms of the gradients for two models.

        :param numpy.ndarray model: stacked array of individual models
                                    np.c_[model1, model2,...]
        :param bool fltr: if True, filter out predefined percentage of lowest norms.
        :param float fltr_per: Percentage of lowest norms to be filtered out.
                               Default is 0.05

        :rtype tuple of numpy.ndarray
        :return (norms_1, norms_2, tot_norms)
                norms_1: np.array containing reciprocals of the gradient norms for model1
                norms_2: np.array containing reciprocals of the gradient norms for model2
                tot_norms: np.array containing element-wise product of norms_1 and norms_2.
        '''
        m1 = self.map1*model
        m2 = self.map2*model

        per = int(fltr_per*self.regmesh.nC)

        if self.regmesh.mesh.dim == 2:
            Dx_m1, Dy_m1 = self.calculate_gradient(m1)
            Dx_m2, Dy_m2 = self.calculate_gradient(m2)

            norms_1 = np.sqrt(Dx_m1**2 + Dy_m1**2)
            norms_1 = np.divide(1, norms_1, out=np.zeros_like(norms_1), where=norms_1>1e-10)
            norms_2 = np.sqrt(Dx_m2**2 + Dy_m2**2)
            norms_2 = np.divide(1, norms_2, out=np.zeros_like(norms_2), where=norms_2>1e-10)

        elif self.regmesh.mesh.dim == 3:
            Dx_m1, Dy_m1, Dz_m1 = self.calculate_gradient(m1)
            Dx_m2, Dy_m2, Dz_m2 = self.calculate_gradient(m2)

            norms_1 = np.sqrt(Dx_m1**2 + Dy_m1**2 + Dz_m1**2)
            norms_1 = np.divide(1, norms_1, out=np.zeros_like(norms_1), where=norms_1>1e-10)
            norms_2 = np.sqrt(Dx_m2**2 + Dy_m2**2 + Dz_m2**2)
            norms_2 = np.divide(1, norms_2, out=np.zeros_like(norms_2), where=norms_2>1e-10)

        # set lowest 5% of norms (largest 5% of 1/norms) to 0.0
        if fltr:
            inds1 = norms_1.argsort()[-per:]
            norms_1[inds1] = 0.0
            inds2 = norms_2.argsort()[-per:]
            norms_2[inds2] = 0.0

        tot_norms = norms_1*norms_2

        return (norms_1, norms_2, tot_norms)

    def normalized_gradients(self,grad_list):
        '''
        Normalizes the spatial gradients of a model.

        :param numpy.ndarray grad_list: array where each row represents a model cell,
                                        and each column represents a component of the gradient.

        :rtype: numpy.ndarray
        :return: norm_gradient: array where the gradients have been normalized by their norms.
                 Each row represents a model cell, and each column represents the normalized
                 component of the gradient.
        '''
        elems = grad_list.shape[0]
        norm_gradients = np.zeros_like(grad_list)
        for i in range(elems):
            gradient = grad_list[i]
            norm = np.linalg.norm(gradient)
            if norm<1e-10:
                continue
            else:
                norm_gradients[i] = gradient / norm

        return norm_gradients

    def calculate_cross_gradient(self, model, normalized=False):
        '''
        Calculates the cross-gradients of two models at each cell center.

        :param numpy.ndarray model: stacked array of individual models
                                    np.c_[model1, model2,...]
        :param bool normalized: normalizes gradients if True

        :rtype: numpy.ndarray
        :returns: array where at each location, we've computed the cross-product
                  of the gradients of two models.
        '''
        m1 = self.map1*model
        m2 = self.map2*model
        m1, m2 = self.models([m1,m2])

        ### Compute the gradients and concatenate components.
        if self.regmesh.mesh.dim == 2:
            Dx_m1, Dy_m1 = self.calculate_gradient(m1)
            Dx_m2, Dy_m2 = self.calculate_gradient(m2)

            grad_list_1 = self.gradient_list(Dx_m1, Dy_m1)
            grad_list_2 = self.gradient_list(Dx_m2, Dy_m2)

            if normalized:
                grad_list_1 = self.normalized_gradients(grad_list_1)
                grad_list_2 = self.normalized_gradients(grad_list_2)

        elif self.regmesh.mesh.dim == 3:
            Dx_m1, Dy_m1, Dz_m1 = self.calculate_gradient(m1)
            Dx_m2, Dy_m2, Dz_m2 = self.calculate_gradient(m2)

            grad_list_1 = self.gradient_list(Dx_m1, Dy_m1, Dz_m1)
            grad_list_2 = self.gradient_list(Dx_m2, Dy_m2, Dz_m2)

            if normalized:
                grad_list_1 = self.normalized_gradients(grad_list_1)
                grad_list_2 = self.normalized_gradients(grad_list_2)

        cross_prod_list = []
        num_cells = len(m1)
        ### for each model cell, compute the cross product of the gradient vectors.
        for x in range(num_cells):
            if self.regmesh.mesh.dim == 3:
                cross_prod_vector = np.cross(grad_list_1[x],grad_list_2[x])
                cross_prod = np.linalg.norm(cross_prod_vector)
            else:
                cross_prod = np.cross(grad_list_1[x],grad_list_2[x])
            cross_prod_list.append(cross_prod)
        cross_prod = np.array(cross_prod_list)

        return cross_prod

    def __call__(self, model, normalized=False):
        '''
        Computes the sum of all cross-gradient values at all cell centers.

        :param numpy.ndarray model: stacked array of individual models
                                    np.c_[model1, model2,...]
        :param bool normalized: returns value of normalized cross-gradient if True

        :rtype: float
        :returns: the computed value of the cross-gradient term.
        '''
        m1 = self.map1*model
        m2 = self.map2*model
        m1, m2 = self.models([m1,m2])

        if self.regmesh.mesh.dim == 2:
            Dx_m1, Dy_m1 = self.calculate_gradient(m1)
            Dx_m2, Dy_m2 = self.calculate_gradient(m2)

            temp1 = Dx_m1**2 + Dy_m1**2
            temp2 = Dx_m2**2 + Dy_m2**2
            term1 = temp1.dot(temp2)

            temp1 = Dx_m1*Dx_m2 + Dy_m1*Dy_m2
            term2 = (np.linalg.norm(temp1))**2
            result = term1 - term2

            if normalized:
                norms1, norms2, norms = self.construct_norm_vectors(model, fltr=False)
                temp1 = (Dx_m1*norms1)**2 + (Dy_m1*norms1)**2
                temp2 = (Dx_m2*norms2)**2 + (Dy_m2*norms2)**2
                term1 = temp1.dot(temp2)

                temp1 = Dx_m1*Dx_m2*norms + Dy_m1*Dy_m2*norms
                term2 = (np.linalg.norm(temp1))**2
                result = term1 - term2

        elif self.regmesh.mesh.dim == 3:
            Dx_m1, Dy_m1, Dz_m1 = self.calculate_gradient(m1)
            Dx_m2, Dy_m2, Dz_m2 = self.calculate_gradient(m2)

            temp1 = Dx_m1**2 + Dy_m1**2 + Dz_m1**2
            temp2 = Dx_m2**2 + Dy_m2**2 + Dz_m2**2
            term1 = temp1.dot(temp2)

            temp1 = Dx_m1*Dx_m2 + Dy_m1*Dy_m2 + Dz_m1*Dz_m2
            term2 = (np.linalg.norm(temp1))**2
            result = term1 - term2

            if normalized:
                norms1, norms2, norms = self.construct_norm_vectors(model, fltr=False)
                temp1 = (Dx_m1*norms1)**2 + (Dy_m1*norms1)**2 + (Dz_m1*norms1)**2
                temp2 = (Dx_m2*norms2)**2 + (Dy_m2*norms2)**2 + (Dz_m2*norms2)**2
                term1 = temp1.dot(temp2)

                temp1 = Dx_m1*Dx_m2*norms + Dy_m1*Dy_m2*norms + Dz_m1*Dz_m2*norms
                term2 = (np.linalg.norm(temp1))**2
                result = term1 - term2

        return 0.5*result

    def deriv(self, model):
        '''
        Computes the Jacobian of the cross-gradient.

        :param list of numpy.ndarray ind_models: [model1, model2,...]

        :rtype: numpy.ndarray
        :return: result: gradient of the cross-gradient with respect to model1, model2
        '''
        m1 = self.map1*model
        m2 = self.map2*model
        m1, m2 = self.models([m1,m2])

        if self.regmesh.mesh.dim == 2:
            Dx = self.regmesh.aveFx2CC.dot(self.regmesh.cellDiffx)
            Dy = self.regmesh.aveFy2CC.dot(self.regmesh.cellDiffy)

            ### common terms for dt_dm1
            Dx_m1, Dy_m1 = Dx.dot(m1), Dy.dot(m1)
            Dx_m2, Dy_m2 = Dx.dot(m2), Dy.dot(m2)
            v1 = Dx_m2**2 + Dy_m2**2
            v2 = Dx_m1**2 + Dy_m1**2
            w = Dx_m1*Dx_m2 + Dy_m1*Dy_m2

            dt_dm1 = (Dx.T.dot(Dx_m1*v1) + Dy.T.dot(Dy_m1*v1) -
                      Dx.T.dot(Dx_m2*w) - Dy.T.dot(Dy_m2*w))
            dt_dm2 = (Dx.T.dot(Dx_m2*v2) + Dy.T.dot(Dy_m2*v2) -
                     Dx.T.dot(Dx_m1*w) - Dy.T.dot(Dy_m1*w))
            result = np.concatenate((dt_dm1,dt_dm2))
        elif self.regmesh.mesh.dim == 3:
            Dx = self.regmesh.aveFx2CC.dot(self.regmesh.cellDiffx)
            Dy = self.regmesh.aveFy2CC.dot(self.regmesh.cellDiffy)
            Dz = self.regmesh.aveFz2CC.dot(self.regmesh.cellDiffz)

            ### common terms for dt_dm1
            Dx_m1, Dy_m1, Dz_m1 = Dx.dot(m1), Dy.dot(m1), Dz.dot(m1)
            Dx_m2, Dy_m2, Dz_m2 = Dx.dot(m2), Dy.dot(m2), Dz.dot(m2)
            v1 = Dx_m2**2 + Dy_m2**2 + Dz_m2**2
            v2 = Dx_m1**2 + Dy_m1**2 + Dz_m1**2
            w = Dx_m1*Dx_m2 + Dy_m1*Dy_m2 + Dz_m1*Dz_m2

            dt_dm1 = (Dx.T.dot(Dx_m1*v1) + Dy.T.dot(Dy_m1*v1) + Dz.T.dot(Dz_m1*v1) -
                      Dx.T.dot(Dx_m2*w) - Dy.T.dot(Dy_m2*w) - Dz.T.dot(Dz_m2*w))
            dt_dm2 = (Dx.T.dot(Dx_m2*v2) + Dy.T.dot(Dy_m2*v2) + Dz.T.dot(Dz_m2*v2) -
                      Dx.T.dot(Dx_m1*w) - Dy.T.dot(Dy_m1*w) - Dz.T.dot(Dz_m1*w))
            result = np.concatenate((dt_dm1,dt_dm2))

        return result

    def _func_hessian(self, D, grad1, grad2):
        '''
        Method for internal use only.
        Computes the off-diagonals blocks of the Hessian.

        :param tuple of scipy.sparse.csr_matrix D: (Dx, Dy, Dz) in 3D
                                                   (Dx, Dy) in 2D
        :param tuple of numpy.ndarray grad1: gradients for model 1
                                             (x_grad1, y_grad1, z_grad1)
        :param tuple of numpy.ndarray grad2: gradients for model2
                                             (x_grad2, y_grad2, z_grad2)

        :rtype: scipy.sparse.csr_matrix
        :return: off-diagonal term of Hessian matrix
        '''
        n = len(D)
        D_result = np.zeros_like(D[0])
        for i in range(n):
            for j in range(n):
                if j==i:
                    continue
                else:
                    D_result += 2*(D[i].T.dot(Utils.sdiag(grad1[j]*grad2[i])).dot(D[j]))
                    D_result -= D[i].T.dot(Utils.sdiag(grad1[j]*grad2[j])).dot(D[i])
                    D_result -= D[j].T.dot(Utils.sdiag(grad1[j]*grad2[i])).dot(D[i])

        D_result = sp.csr_matrix(D_result)
        return D_result

    def deriv2(self, model, v=None):
        '''
        Computes the Hessian of the cross-gradient.

        :param list of numpy.ndarray ind_models: [model1, model2, ...]
        :param numpy.ndarray v: vector to be multiplied by Hessian

        :rtype: scipy.sparse.csr_matrix if v is None
                numpy.ndarray if v is not None
        :return Hessian matrix if v is None
                Hessian multiplied by vector if v is not None
        '''
        m1 = self.map1*model
        m2 = self.map2*model
        m1, m2 = self.models([m1, m2])

        if v is not None:
            assert v.size == 2*m1.size, 'vector v must be of size 2*M'
            v1 = self.map1*v
            v2 = self.map2*v

        func1 = self._func_hessian

        if self.regmesh.mesh.dim == 2:
            Dx = self.regmesh.aveFx2CC.dot(self.regmesh.cellDiffx)
            Dy = self.regmesh.aveFy2CC.dot(self.regmesh.cellDiffy)
            ### define common terms
            Dx_m1, Dy_m1 = Dx.dot(m1), Dy.dot(m1)
            Dx_m2, Dy_m2 = Dx.dot(m2), Dy.dot(m2)
            a = Dx_m2**2 + Dy_m2**2
            b = Dx_m1**2 + Dy_m1**2
            A = Dx.T.dot(Utils.sdiag(Dx_m2)) + Dy.T.dot(Utils.sdiag(Dy_m2))
            B = Dx.T.dot(Utils.sdiag(Dx_m1)) + Dy.T.dot(Utils.sdiag(Dy_m1))

            d2t_dm1 = (Dx.T.dot(Utils.sdiag(a)).dot(Dx) +
                       Dy.T.dot(Utils.sdiag(a)).dot(Dy) - A.dot(A.T))
            d2t_dm2 = (Dx.T.dot(Utils.sdiag(b)).dot(Dx) +
                       Dy.T.dot(Utils.sdiag(b)).dot(Dy) - B.dot(B.T))

            d_dm2_dt_dm1 = func1((Dx, Dy), (Dx_m1, Dy_m1), (Dx_m2, Dy_m2))
            d_dm1_dt_dm2 = d_dm2_dt_dm1.T

        elif self.regmesh.mesh.dim == 3:
            Dx = self.regmesh.aveFx2CC.dot(self.regmesh.cellDiffx)
            Dy = self.regmesh.aveFy2CC.dot(self.regmesh.cellDiffy)
            Dz = self.regmesh.aveFz2CC.dot(self.regmesh.cellDiffz)
            ### define common terms
            Dx_m1, Dy_m1, Dz_m1 = Dx.dot(m1), Dy.dot(m1), Dz.dot(m1)
            Dx_m2, Dy_m2, Dz_m2 = Dx.dot(m2), Dy.dot(m2), Dz.dot(m2)
            a = Dx_m2**2 + Dy_m2**2 + Dz_m2**2
            b = Dx_m1**2 + Dy_m1**2 + Dz_m1**2
            A = (Dx.T.dot(Utils.sdiag(Dx_m2)) + Dy.T.dot(Utils.sdiag(Dy_m2)) +
                 Dz.T.dot(Utils.sdiag(Dz_m2)))
            B = (Dx.T.dot(Utils.sdiag(Dx_m1)) + Dy.T.dot(Utils.sdiag(Dy_m1)) +
                 Dz.T.dot(Utils.sdiag(Dz_m1)))

            d2t_dm1 = (Dx.T.dot(Utils.sdiag(a)).dot(Dx) +
                       Dy.T.dot(Utils.sdiag(a)).dot(Dy) +
                       Dz.T.dot(Utils.sdiag(a)).dot(Dz) - A.dot(A.T))
            d2t_dm2 = (Dx.T.dot(Utils.sdiag(b)).dot(Dx) +
                       Dy.T.dot(Utils.sdiag(b)).dot(Dy) +
                       Dz.T.dot(Utils.sdiag(b)).dot(Dz) - B.dot(B.T))

            d_dm2_dt_dm1 = func1((Dx, Dy, Dz), (Dx_m1, Dy_m1, Dz_m1), (Dx_m2, Dy_m2, Dz_m2))
            d_dm1_dt_dm2 = d_dm2_dt_dm1.T

        if v is not None:
            d2t_dm1 = d2t_dm1.dot(v1)
            d2t_dm2 = d2t_dm2.dot(v2)
            d_dm2_dt_dm1 = d_dm2_dt_dm1.dot(v1)
            d_dm1_dt_dm2 = d_dm1_dt_dm2.dot(v2)
            result = np.concatenate((d2t_dm1 + d_dm1_dt_dm2, d_dm2_dt_dm1 + d2t_dm2))
        else:
            temp1 = sp.vstack((d2t_dm1,d_dm2_dt_dm1))
            temp2 = sp.vstack((d_dm1_dt_dm2, d2t_dm2))
            result = sp.hstack((temp1,temp2))
            result = sp.csr_matrix(result)

        return result
