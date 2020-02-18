#!/bin/ipython

import numpy as _np


def distribution_product(covariances, means, activations, inv_func=_np.linalg.pinv ):

    invCovariancesWeightedSum = 0.0
    meansWeigthedSum = 0.0
    for cov, mean, alpha in zip(covariances, means, activations):
        invCoveWeighted = alpha * inv_func(cov)
        invCovariancesWeightedSum += invCoveWeighted
        meansWeigthedSum  += _np.dot(invCoveWeighted, mean)
    covariancesCombined = inv_func(invCovariancesWeightedSum)
    meanCombined = _np.dot(covariancesCombined, meansWeigthedSum)
    return covariancesCombined, meanCombined

def distribution_product_regularized(covariances, means, activations, cov_ref, mean_ref, inv_func=_np.linalg.pinv ):
    activations_sum = _np.sum(activations)
    activations_sum2 = _np.sum(activations**2)

    invCovariancesWeightedSum = 0.0
    meansWeigthedSum = 0.0
    for cov, mean, alpha in zip(covariances, means, activations):
        cov_decorrelated = activations_sum2 * cov + activations_sum * (1.0-alpha) * cov_ref
        invCoveWeighted = alpha * inv_func(cov_decorrelated)
        invCovariancesWeightedSum += invCoveWeighted
        meansWeigthedSum  += _np.dot(invCoveWeighted, mean)
    covariancesCombined = inv_func(invCovariancesWeightedSum)
    meanCombined = _np.dot(covariancesCombined, meansWeigthedSum)
    return covariancesCombined, meanCombined

def deterministic_interpolation(covariances, means, activations, inv_func=_np.linalg.pinv, covariance_fake = 0.05):

    gainsWeightedSum = 0.0
    baseVarianceSum = 0.0
    meansWeigthedSum = 0.0
    for cov, mean, alpha in zip(covariances, means, activations):
        gain = -cov[1,0] / cov[0,0]
        gainsWeightedSum += gain * alpha
        meansWeigthedSum += mean * alpha
        baseVarianceSum  += cov[0,0] * alpha
    baseVariance = baseVarianceSum / _np.sum(activations)
    gainsWeighted = gainsWeightedSum / _np.sum(activations)
    meanCombined = meansWeigthedSum / _np.sum(activations)
    covariancesCombined = covariance_fake * _np.array([[1.0,-gainsWeighted],[-gainsWeighted,gainsWeighted**2+0.0001]])
    return covariancesCombined, meanCombined
