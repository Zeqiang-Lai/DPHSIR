from functools import partial
from .sisr import interpolate, CloseFormed_ADMM_Prox


class proxs:
    def CloseFormedADMM(kernel):
        return CloseFormed_ADMM_Prox(kernel, sf=1)


class inits:
    interpolate = partial(interpolate, sf=1)
