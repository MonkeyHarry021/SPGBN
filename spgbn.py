import os
import copy
import time
import numpy as np

from basic_model import Basic_Model
from sampler import Basic_Sampler
from scipy.stats import poisson
from utils import *


class SPGBN(Basic_Model):
    def __init__(self, K: list, L: list, device='gpu'):
        """
        The basic model for PGBN
        Inputs:
            K      : [list] number of topics at diffrent layers in PGBN;
            device : [str] 'cpu' or 'gpu';

        Attributes:
            @public:
                global_params : [Params] the global parameters of the probabilistic model
                local_params  : [Params] the local parameters of the probabilistic model

            @private:
                _model_setting   : [Params] the model settings of the probabilistic model
                _hyper_params    : [Params] the hyper parameters of the probabilistic model
                _model_setting.T : [int] the network depth

        """
        super(SPGBN, self).__init__()
        setattr(self, '_model_name', 'PGBN')
        self._model_setting.K = K
        self._model_setting.T = len(K)
        self._model_setting.device = device

        assert self._model_setting.device in ['cpu', 'gpu'], 'Device Type Error: the device should be ''cpu'' or ''gpu'''

        self._sampler = Basic_Sampler(self._model_setting.device)
        self._eps = 1e-38
        self.L = L

    def initial(self, data: np.ndarray):
        '''
        Initial the parameters of PGBN with the input documents
        Inputs:
            dataset : [np.ndarray] V*N matrix, N bag-of-words vectors with a vocabulary length of V

        Attributes:
            @public:
                global_params.Phi  : [list] T (K_t-1)*(K_t) factor loading matrices at different layers

            @private:
                _model_setting.V        : [int] scalar, the length of the vocabulary
                _hyper_params.Phi_eta   : [int] scalar, the parameter in the prior of Phi
                _hyper_params.Theta_r_k : [int] scalar, the parameter in the prior of Theta
                _hyper_params.p_j_a0    : [int] scalar, the parameter in the prior of p_j
                _hyper_params.p_j_b0    : [int] scalar, the parameter in the prior of p_j
                _hyper_params.c_j_a0    : [int] scalar, the parameter in the prior of c_j
                _hyper_params.c_j_b0    : [int] scalar, the parameter in the prior of c_j

        '''
        assert type(data) is np.ndarray, 'Data type error: the input dataset should be a 2-D np.ndarray'
        self._model_setting.V = data.shape[0]

        self.global_params.Phi = []
        self._hyper_params.Phi_eta = []
        self._hyper_params.Theta_r_k = np.ones([self._model_setting.K[self._model_setting.T - 1], 1]) / self._model_setting.K[self._model_setting.T - 1]
        self._hyper_params.p_j_a0 = 0.01
        self._hyper_params.p_j_b0 = 0.01
        self._hyper_params.c_j_e0 = 1
        self._hyper_params.c_j_f0 = 1

        for t in range(self._model_setting.T):
            self._hyper_params.Phi_eta.append(0.01)
            if t == 0:
                self.global_params.Phi.append(0.2 + 0.8 * np.random.rand(self._model_setting.V, self._model_setting.K[t]))
            else:
                self.global_params.Phi.append(0.2 + 0.8 * np.random.rand(self._model_setting.K[t-1], self._model_setting.K[t]))
            self.global_params.Phi[t] = self.global_params.Phi[t] / np.maximum(realmin, self.global_params.Phi[t].sum(0))


    def train(self, data:np.ndarray, num_epochs: int=1, is_train: bool = True, is_initial_local: bool=True):
        '''
        Inputs:
            num_epochs   : [int] scalar, the iterations of gibbs sampling
            dataset       : [np.ndarray] V*N matrix, N bag-of-words vectors with a vocabulary of length V
            is_train   : [bool] True or False, whether to update the global params in the probabilistic model

        Attributes:
            @public:
                local_params.Theta : [list] T (K_t)*(N) topic proportions at different layers
                local_params.c_j   : [list] T+1 1*N vector, the variables in the scale parameter in the Theta
                local_params.p_j   : [list] T+1 1*N vector, the variables in the scale parameter in the Theta

            @private:
                _model_setting.N         : [int] scalar, the number of the documents in the corpus
                _model_setting.Iteration : [int] scalar, the iterations of gibbs sampling

        Outputs:
            local_params  : [Params] the local parameters of the probabilistic model

        '''
        assert type(data) is np.ndarray, 'Data type error: the input dataset should be a 2-D np.ndarray'
        self._model_setting.N = data.shape[1]
        self._model_setting.Iteration = num_epochs

        # initial local params
        if is_initial_local or not hasattr(self.local_params, 'Theta') or not hasattr(self.local_params, 'c_j') or not hasattr(self.local_params, 'p_j'):
            self.local_params.Theta = []
            self.local_params.c_j = []
            for t in range(self._model_setting.T):  # from layer 1 to T
                self.local_params.Theta.append(np.ones([self._model_setting.K[t], self._model_setting.N]) / self._model_setting.K[t])
                self.local_params.c_j.append(np.ones([1, self._model_setting.N]))
            self.local_params.c_j.append(np.ones([1, self._model_setting.N]))
            self.local_params.p_j = self._calculate_pj(self.local_params.c_j, self._model_setting.T)

        Xt_to_t1 = []
        WSZS = []
        for t in range(self._model_setting.T):
            Xt_to_t1.append(np.zeros(self.local_params.Theta[t].shape))
            WSZS.append(np.zeros(self.global_params.Phi[t].shape))

        # the params of the matrix Phi^t
        A = [] # A^t = Z^t hardmard D^t
        Z = [] # Z^t binary matrix
        D = []
        H = [] # H^t auxiliary variables matrix to infer the posterior distribution of Z
        P = [] # P^t vector sampled from Beta distribution
        for t in range(self._model_setting.T):
            A.append(np.ones(self.global_params.Phi[t].shape) * 0.01)
            Z.append(np.ones(self.global_params.Phi[t].shape))
            D.append(np.ones(self.global_params.Phi[t].shape) * 0.01)
            H.append(np.random.randint(0,5,size = self.global_params.Phi[t].shape))
            P.append(np.random.rand(self._model_setting.K[t]))

        #the latent count matrix M^t and corresponding structure params u,v,lambda
        M = []
        M_augment = []
        M_intensity_augment = [] 
        U = []
        V = []
        Lam = []
        u_ai = 0.01
        u_bi = 0.01
        v_aj = 0.01
        v_bj = 0.01
        lam_r0 = 1
        lam_beta = 1
        for t in range(self._model_setting.T):
            M.append(np.ones(self.global_params.Phi[t].shape))
            M_augment.append(np.zeros(self.global_params.Phi[t].shape + (self.L[t],)))
            M_intensity_augment.append(np.zeros(self.global_params.Phi[t].shape + (self.L[t],), dtype = np.float32))
            U.append(np.random.rand(self.global_params.Phi[t].shape[0], self.L[t]))
            V.append(np.random.rand(self.global_params.Phi[t].shape[1], self.L[t]))
            Lam.append(np.random.rand(self.L[t]))
             
        # gibbs sampling
        for iter in range(self._model_setting.Iteration):
            start_time = time.time()

            # update global params
            for t in range(self._model_setting.T):  # from layer 1 to T
                if t == 0:
                    Xt_to_t1[t], WSZS[t] = self._sampler.multi_aug(data, self.global_params.Phi[t], self.local_params.Theta[t])
                else:
                    Xt_to_t1[t], WSZS[t] = self._sampler.crt_multi_aug(Xt_to_t1[t-1], self.global_params.Phi[t], self.local_params.Theta[t])
                if is_train:
                    self.global_params.Phi[t] = self._update_Phi(WSZS[t], A[t])

            # update local params
            if self._model_setting.T > 1:
                self.local_params.p_j[1][0, :] = self._sampler.beta(Xt_to_t1[0].sum(0) + self._hyper_params.p_j_a0, self.local_params.Theta[1].sum(0) + self._hyper_params.p_j_b0)
            else:
                self.local_params.p_j[1][0, :] = self._sampler.beta(Xt_to_t1[0].sum(0) + self._hyper_params.p_j_a0, self._hyper_params.Theta_r_k.sum(0) + self._hyper_params.p_j_b0)
            self.local_params.p_j[1] = np.minimum(np.maximum(self.local_params.p_j[1], realmin), 1 - realmin)  # make sure p_j is not too large or small
            self.local_params.c_j[1] = (1 - self.local_params.p_j[1]) / self.local_params.p_j[1]

            for t in [i for i in range(self._model_setting.T + 1) if i > 1]:  # from layer 3 to layer T+1
                if t == self._model_setting.T:
                    self.local_params.c_j[t][0, :] = self._sampler.gamma(self._hyper_params.Theta_r_k.sum(0) + self._hyper_params.c_j_e0, 1) \
                                                      / (self.local_params.Theta[t-1].sum(0) + self._hyper_params.c_j_f0 + realmin)
                else:
                    self.local_params.c_j[t][0, :] = self._sampler.gamma(self.local_params.Theta[t].sum(0) + self._hyper_params.c_j_e0, 1) \
                                                      / (self.local_params.Theta[t-1].sum(0) + self._hyper_params.c_j_f0 + realmin)

            p_j_tmp = self._calculate_pj(self.local_params.c_j, self._model_setting.T)
            self.local_params.p_j[2:] = p_j_tmp[2:]

            for t in range(self._model_setting.T - 1, -1, -1):  # from layer T to 1
                if t == self._model_setting.T - 1:
                    shape = np.repeat(self._hyper_params.Theta_r_k, self._model_setting.N, axis=1)
                else:
                    shape = np.dot(self.global_params.Phi[t + 1], self.local_params.Theta[t + 1])
                self.local_params.Theta[t] = self._update_Theta(Xt_to_t1[t], shape, self.local_params.c_j[t + 1], self.local_params.p_j[t])

            # M^t
            for t in range(self._model_setting.T):
                M_intensity_augment[t] = (np.einsum('ik,jk,k -> ijk', U[t], V[t], Lam[t]))
                M[t] = Z[t] * self._sampler.Po_plus(M_intensity_augment[t].sum(axis = -1))
                for i in range(self.global_params.Phi[t].shape[0]):
                    for j in range(self.global_params.Phi[t].shape[1]):
                        n = M[t][i,j]
                        p = M_intensity_augment[t][i,j] + self._eps
                        p = p/p.sum()
                        M_augment[t][i,j] = self._sampler.multinomial(n ,p)

            # U^t
            for t in range(self._model_setting.T):
                shp = u_ai + M_augment[t].sum(axis = 1)  # size: I^t * K
                rte = u_bi + np.einsum('jk, k -> k', V[t], Lam[t])
                rte = np.tile(rte , (shp.shape[0], 1))
                U[t] = self._sampler.gamma(shp, 1/rte)

            # V^t
            for t in range(self._model_setting.T):
                shp = v_aj + M_augment[t].sum(axis = 0)  # size: J^t * K
                rte = v_bj + np.einsum('ik, k -> k', U[t], Lam[t])
                rte = np.tile(rte, (shp.shape[0], 1))
                V[t] = self._sampler.gamma(shp, 1/rte)

            # Lambda^t
            for t in range(self._model_setting.T):
                shp = lam_r0/self.L[t] + M_augment[t].sum(axis = (0,1))  # size: K
                rte = lam_beta + np.einsum('ik, jk -> k', U[t], V[t])
                Lam[t] = self._sampler.gamma(shp, 1/rte)
            
            # Z^t
            for t in range(self._model_setting.T):
                H[t] = self._sampler.crt(WSZS[t], A[t])
                P[t] = self._sampler.beta(WSZS[t].sum(axis = 0) + self._eps, A[t].sum(axis = 0) + self._eps)
                pij1, pij0 = self._calculate_pij1_pij0(U[t], V[t], Lam[t], H[t], P[t], D[t]) # (K_t-1)*(K_t)
                Z[t] = self._sampler.binomial(np.ones(Z[t].shape, dtype = np.int32), pij1/(pij1+pij0+self._eps))

            # D^t
            for t in range(self._model_setting.T):
                shp = 0.01 + H[t]
                rte = 0.01 - Z[t] * np.log(1 - np.tile(P[t], (D[t].shape[0], 1)) + self._eps)
                D[t] = self._sampler.gamma(shp, 1/rte)
            
            # A^t
            for t in range(self._model_setting.T):
                A[t] = Z[t] * D[t]

            end_time = time.time()
            stages = 'Training' if is_train else 'Testing'
            print(f'{stages} Stage: ',
                  f'epoch {iter:3d} takes {end_time - start_time:.2f} seconds')

        return copy.deepcopy(self.local_params)


    def test(self, data: np.ndarray, num_epochs: int=1, is_initial_local=True):
        '''
        Inputs:
            num_epochs   : [int] scalar, the iterations of gibbs sampling
            dataset       : [np.ndarray] V*N matrix, N bag-of-words vectors with a vocabulary of length V

        Outputs:
            local_params  : [Params] the local parameters of the probabilistic model

        '''
        local_params = self.train(data, num_epochs=num_epochs, is_train=False, is_initial_local=is_initial_local)

        return local_params


    def load(self, model_path: str):
        '''
        Load the model parameters from the specified directory
        Inputs:
            model_path : [str] the directory path to load the model.

        '''
        assert os.path.exists(model_path), 'Path Error: can not find the path to load the model'
        model = np.load(model_path, allow_pickle=True).item()
        model['_model_setting'].device = self._model_setting.device
        for params in ['global_params', 'local_params', '_model_setting', '_hyper_params']:
            if params in model:
                setattr(self, params, model[params])


    def save(self, model_path: str = '../save_models'):
        '''
        Save the model to the specified directory.
        Inputs:
            model_path : [str] the directory path to save the model, default '../save_models/PGBN.npy'
        '''
        # create the directory path
        if not os.path.isdir(model_path):
            os.mkdir(model_path)

        # save the model
        model = {}
        for params in ['global_params', 'local_params', '_model_setting', '_hyper_params']:
            if params in dir(self):
                model[params] = getattr(self, params)

        np.save(model_path + '/' + self._model_name + '.npy', model)
        print('model have been saved by ' + model_path + '/' + self._model_name + '.npy')


    def _calculate_pj(self, c_j: list, T: int):
        '''
        calculate p_j from layer 1 to T+1 according to c_j
        Inputs:
            c_j  : [list] T+1 1*N vector, the variables in the scale parameter in the Theta
            T    : [int] network depth
        Outputs:
            p_j  : [list] T+1 1*N vector, the variables in the scale parameter in the Theta

        '''
        p_j = []
        N = c_j[1].size
        p_j.append((1 - np.exp(-1)) * np.ones([1, N]))  # p_j_1
        p_j.append(1 / (1 + c_j[1]))                    # p_j_2

        for t in [i for i in range(T + 1) if i > 1]:    # p_j_3_T+1; only T>=2 works
            tmp = -np.log(np.maximum(1 - p_j[t - 1], realmin))
            p_j.append(tmp / (tmp + c_j[t]))

        return p_j
    
    def _calculate_pij1_pij0(self, u, v, lam, h, p, d):
        '''
        Inputs:
            h : [np.ndarray] (K_t-1)*(K_t) auxiliary variable
            p : [np.ndarray] (K_t) vector, the probabilities sampled from beta distribution
            d : [np.ndarray] (K_t-1)*(K_t) the t-th matrix of parameter D
        Outputs:
            pij1 : [np.ndarray] (K_t-1)*(K_t) the posterior probability of zij = 1 given the likelihood l (auxiliary var)
            pij0 : [np.ndarray] (K_t-1)*(K_t) the posterior probability of zij = 0 given the likelihood l (auxiliary var)
        '''
        sum_over_uvlam = np.einsum('ik, jk, k -> ij', u, v, lam)
        pij1_prior = 1 - np.exp(-sum_over_uvlam)
        pij0_prior = 1 - pij1_prior

        pij1_intensity = -d*np.log(1 - np.tile(p, (h.shape[0], 1)) + self._eps) # zij = 1
        pij0_intensity = np.zeros(h.shape) # zij = 0
        pij1_likelihood = poisson.pmf(h, pij1_intensity)
        pij0_likelihood = poisson.pmf(h, pij0_intensity)

        pij1 = pij1_prior * pij1_likelihood
        pij0 = pij0_prior * pij0_likelihood

        return pij1, pij0

    def _update_Phi(self, WSZS_t, A_t):
        '''
        update Phi_t at layer t
        Inputs:
            WSZS_t  : [np.ndarray]  (K_t-1)*(K_t) count matrix appearing in the likelihood of Phi_t
            Eta_t   : [np.ndarray]  scalar, the variables in the prior of Phi_t
        Outputs:
            Phi_t   : [np.ndarray]  (K_t-1)*(K_t), topic matrix at layer t

        '''
        Phi_t_shape = WSZS_t + A_t
        Phi_t = self._sampler.gamma(Phi_t_shape, 1) + self._eps
        Phi_t = Phi_t / Phi_t.sum(0)

        return Phi_t


    def _update_Theta(self, Xt_to_t1_t, shape, c_j_t1, p_j_t):
        '''
        update Theta_t at layer t
        Inputs:
            Xt_to_t1_t : [np.ndarray]  (K_t-1)*(K_t) count matrix appearing in the likelihood of Phi_t
            shape      : [np.ndarray]  scalar, the variables in the prior of Phi_t
            c_j_t1     : [np.ndarray]  N * 1 vector, the variables in the scale parameter in the Theta_t+1
            p_j_t      : [np.ndarray]  N * 1 vector, the variables in the scale parameter in the Theta_t
        Outputs:
            Theta_t   : [np.ndarray]  (K_t-1)*(K_t), topic proportion matrix at layer t

        '''
        Theta_t_shape = Xt_to_t1_t + shape
        Theta_t = self._sampler.gamma(Theta_t_shape, 1) / (c_j_t1[0, :] - np.log(np.maximum(realmin, 1 - p_j_t[0, :])))

        return Theta_t


if __name__ == '__main__':
    print('Start PGBN')
    pgbn = SPGBN(K=[10,10,10],  L = [5,5,5], device = 'cpu')
    data = np.array([[1,2,3,4], [2,3,4,5], [3,4,5,6]])
    pgbn.initial(data)
    pgbn.train(data, 10)
    print('end')