"""
===========================================
Model Sampler implemented on CPU
===========================================

"""

# Author: Chaojie Wang <xd_silly@163.com>; Jiawen Wu
# License: BSD-3-Clause

import numpy as np
import numpy.ctypeslib as npct
import ctypes  
from ctypes import *
import os

class model_sampler_cpu(object):

    def __init__(self, system_type='Windows', seed=0):
        """
        The basic class for sampling distribution on cpu
        """
        super(model_sampler_cpu, self).__init__()

        self.system_type = system_type
        self.seed = seed

        array_2d_double = npct.ndpointer(dtype=np.double, ndim=2, flags='C')
        array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='C')
        array_int = npct.ndpointer(dtype=np.int32, ndim=0, flags='C')
        ll = ctypes.cdll.LoadLibrary

        if system_type == "Windows":
            self.Crt_lib = ll(os.path.dirname(__file__) + "\_compact\crt_cpu.dll")
            self.Multi_lib = ll(os.path.dirname(__file__) + "\_compact\multi_aug_cpu.dll")
            self.Crt_Multi_lib = ll(os.path.dirname(__file__) + "\_compact\crt_multi_aug_cpu.dll")
        else:
            self.Crt_lib = ll(os.path.dirname(__file__) + "/_compact/crt_cpu.so")
            self.Multi_lib = ll(os.path.dirname(__file__) + "/_compact/multi_aug_cpu.so")
            self.Crt_Multi_lib = ll(os.path.dirname(__file__) + "/_compact/crt_multi_aug_cpu.so")


        self.Multi_lib.Multi_Sample.restype = None
        self.Multi_lib.Multi_Sample.argtypes = [array_2d_double, array_2d_double, array_2d_double, array_2d_double,
                                           array_2d_double, c_int, c_int, c_int]

        self.Crt_Multi_lib.Crt_Multi_Sample.restype = None
        self.Crt_Multi_lib.Crt_Multi_Sample.argtypes = [array_2d_double, array_2d_double, array_2d_double, array_2d_double,
                                                   array_2d_double, c_int, c_int, c_int]

        self.Crt_lib.Crt_Sample.restype = None
        self.Crt_lib.Crt_Sample.argtypes = [array_2d_double, array_2d_double, array_2d_double, c_int, c_int]

    def multi_aug(self, X_t, Phi_t, Theta_t):

        X_t = np.array(X_t, order='C').astype('double')
        Phi_t = np.array(Phi_t, order='C').astype('double')
        Theta_t = np.array(Theta_t, order='C').astype('double')

        V = X_t.shape[0]
        J = X_t.shape[1]
        K = Theta_t.shape[0]
        Xt_to_t1_t = np.zeros([K, J], order='C').astype('double')
        WSZS_t = np.zeros([V, K], order='C').astype('double')
        self.Multi_lib.Multi_Sample(X_t, Phi_t, Theta_t, WSZS_t, Xt_to_t1_t, V, K, J)

        return Xt_to_t1_t, WSZS_t

    def crt(self, Xt_to_t1_t, p):

        Xt_to_t1_t = np.array(Xt_to_t1_t, order='C')
        p = np.array(p, order='C')

        K_t = Xt_to_t1_t.shape[0]
        J = Xt_to_t1_t.shape[1]
        X_t1 = np.zeros([K_t, J], order='C').astype('double')

        self.Crt_lib.Crt_Sample(Xt_to_t1_t, p, X_t1, K_t, J)

        return X_t1

    def crt_multi_aug(self, Xt_to_t1_t, Phi_t1, Theta_t1):

        Xt_to_t1_t = np.array(Xt_to_t1_t, order='C').astype('double')
        Phi_t1 = np.array(Phi_t1, order='C').astype('double')
        Theta_t1 = np.array(Theta_t1, order='C').astype('double')

        K_t = Xt_to_t1_t.shape[0]
        J = Xt_to_t1_t.shape[1]
        K_t1 = Theta_t1.shape[0]
        Xt_to_t1_t1 = np.zeros([K_t1, J], order='C').astype('double')
        WSZS_t1 = np.zeros([K_t, K_t1], order='C').astype('double')

        self.Crt_Multi_lib.Crt_Multi_Sample(Xt_to_t1_t, Phi_t1, Theta_t1, WSZS_t1, Xt_to_t1_t1, K_t, K_t1, J)

        return Xt_to_t1_t1, WSZS_t1

    def Po_plus(self, rate):
        r1 = rate[rate >= 1]
        r2 = rate[rate < 1]
        m = np.zeros_like(rate)
        m1 = np.zeros_like(r1)
        m2 = np.zeros_like(r2)

        while True:
            dex = np.nonzero(m1 == 0)  # 使用np.nonzero()获取m1中值为0的索引
            if dex[0].size == 0:  # 当m1中的值全不为0时循环终止
                break
            else:
                r_dex = r1[dex]
                temp = np.random.poisson(r_dex)
                idex = temp > 0
                m1[dex] = np.where(idex, temp, m1[dex])
        m[rate >= 1] = m1

        while True:
            dex = np.nonzero(m2 == 0)
            if dex[0].size == 0:
                break
            else:
                r_dex = r2[dex]
                temp = 1 + np.random.poisson(r_dex)
                idex = np.random.rand(*temp.shape) < (1 / temp)
                m2[dex] = np.where(idex, temp, m2[dex])
        m[rate < 1] = m2

        return m



