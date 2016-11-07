import numpy as np

float_type = np.float64

class GMM:
    def __init__(self,ndim,nmix):
        self.nmix = nmix
        self.ndim = ndim
        self.wts = (1/nmix)*np.ones(nmix,dtype=float_type,order='F')
        self.mean = np.zeros((ndim,nmix),dtype=float_type,order='F')
        self.covs = np.ones((ndim,nmix),dtype=float_type,order='F')

