import numpy as np


class ParamProvider:
    def __getitem__(self, idx):
        pass

    def __call__(self, idx, **context):
        pass

    def __repr__(self):
        return [x for x in self].__repr__()


class sequence(ParamProvider):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __call__(self, idx, **context):
        return self.data[idx]


def admm_log_descent(sigma=2.55/255, iter_num=15, modelSigma1=49.0, modelSigma2=2.55, w=1.0, lam=0.23):
    '''
    One can change the sigma to implicitly change the trade-off parameter
    between fidelity term and prior term
    '''
    modelSigmaS = np.logspace(np.log10(modelSigma1), np.log10(modelSigma2), iter_num).astype(np.float32)
    modelSigmaS_lin = np.linspace(modelSigma1, modelSigma2, iter_num).astype(np.float32)
    sigmas = (modelSigmaS*w+modelSigmaS_lin*(1-w))/255.
    rhos = list(map(lambda x: lam*(sigma**2)/(x**2), sigmas))
    return sequence(rhos), sequence(sigmas)
