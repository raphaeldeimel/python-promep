#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel
@copyright 2017
@licence: 2-clause BSD licence

This file contains the main code for representing ProMPs, visualizing them, and
also to sample trajectories from a ProMP

The algorithms are described in:
[1] A. Paraschos, C. Daniel, J. Peters, and G. Neumann,
“Using probabilistic movement primitives in robotics,” Autonomous Robots, pp. 1–23, 2017.

"""

import numpy as _np
import scipy as _scipy
import itertools as _it
import collections as _collections
import hdf5storage as _h5
import time as _time
import os as _os
import matplotlib.pyplot as _plt
import matplotlib as _mpl

from ._tensorfunctions import dot, T, pinv, addToRidge, flattenCovarianceTensor, makeCovarianceTensorUncorrelated, I, eigenValues, getDiagView
from . import interpolationkernels as _ik
from . import controller as _controller
from . import MechanicalStateDistributionDescription as _MechanicalStateDistributionDescription
from  scipy.special import betainc as _betainc
from sklearn import covariance as _sklearncovariance


def estimateMean(wSamples, initialVariance, tolerance=1e-7):
    nSamples = wSamples.shape[0]
    wsamples_flat = _np.reshape(wSamples, (nSamples,-1) )  
    means_flat = _np.mean(wsamples_flat, axis=0)
    means = means_flat.reshape(wSamples.shape[1], wSamples.shape[2])
    return means

def estimateMultivariateGaussianCovariances(wSamples, initialVariance=None, tolerance=1e-7):
    """
    estimate the values of a multivariate gaussian distribution
    from the given set of samples

    wSamples: array of size (n, i1, i2) with n=number of samples

    initialVariance: initial variance to use as regularization when using few samples

    returns: (means, covarianceTensor)
        means: matrix of shape (i1, i2)
        covarianceTensor: array of shape (i1, i2, i2, i1)
    """
    nSamples = wSamples.shape[0]
    wsamples_flat = _np.reshape(wSamples, (nSamples,-1) )  
    means_flat = _np.mean(wsamples_flat, axis=0)
    deviations_flat = _np.sqrt(_np.var(wsamples_flat, axis=0))
    deviations_regularized = _np.clip(deviations_flat, tolerance, _np.inf)
    wsamples_flat_normalized = (wsamples_flat - means_flat) / deviations_regularized
    tensorshape =  (wSamples.shape[1], wSamples.shape[2], wSamples.shape[1], wSamples.shape[2])
    
    #use sklearn for more sophisticated covariance estimators:
    model  = _sklearncovariance.OAS()  #Alternatives: OAS, GraphLasso , EmpiricalCovariance
    model.fit(wsamples_flat_normalized)
    correlationMatrix = model.covariance_.copy()
    assertCovTensorIsWellFormed(correlationMatrix.reshape(tensorshape), tolerance)    

    scale = _np.diag(deviations_regularized)
    covarianceTensor = _np.dot(_np.dot(scale,correlationMatrix),scale.T) 
    covarianceTensor = 0.5*(covarianceTensor + covarianceTensor.T) #ensure symmetry in face of numeric deviations
    if initialVariance is not None:
        covarianceTensor += (_np.eye(wSamples.shape[1]*wSamples.shape[2])*initialVariance - covarianceTensor) * (1./(nSamples+1))
    covarianceTensor.shape = tensorshape
    assertCovTensorIsWellFormed(covarianceTensor, tolerance)
    return covarianceTensor


def assertCovTensorIsWellFormed(covarianceTensor, tolerance):
        covarianceMatrix = flattenCovarianceTensor(covarianceTensor)
        if _np.any( _np.abs(covarianceMatrix.T - covarianceMatrix) > tolerance):
            raise RuntimeWarning("Notice: Covariance Matrix is not symmetric. This usually is an error")
        #test the cov matrix of each dof individually to better pinpoint errors:
        for i in range(covarianceTensor.shape[1]):
            try:
                _np.linalg.cholesky(covarianceTensor[:,i,:,i])  #will fail if matrix is not positive-semidefinite
            except _np.linalg.linalg.LinAlgError:
                eigenvals = _np.linalg.eigvals(covarianceTensor[:,i,:,i])
                raise RuntimeWarning("Notice: Covariance Matrix of dof {0} is not positive semi-definite. This usually is an error.\n Eigenvalues: {1}".format(i, eigenvals))
        #test complete tensor, i.e. also across dofs:
        try:
            _np.linalg.cholesky(covarianceMatrix)  #will fail if matrix is not positive-semidefinite
        except _np.linalg.linalg.LinAlgError:
            eigenvals = _np.linalg.eigvals(covarianceMatrix)
            raise RuntimeWarning("Notice: Covariance tensor is not positive semi-definite. This usually is an error.\n Eigenvalues of flattened tensor: {1}".format(i, eigenvals))


#set a sensible default color map for correlations
_cmapCorrelations = _mpl.colors.LinearSegmentedColormap('Correlations', {
         'red':   ((0.0, 0.2, 0.2),
                   (0.33, 1.0, 1.0),
                   (1.0, 1.0, 1.0)),

         'green': ((0.0, 0.0, 0.0),
                   (0.33, 1.0, 1.0),
                   (0.66, 1.0, 1.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 1.0),
                   (0.66, 1.0, 1.0),
                   (1.0, 0.2, 0.2))
        }
)

def plotYSpaceCovarianceTensor(cov, normalized=True, interpolationKernel=None):
        """
        plot a correlation tensor of shape (mstates x dofs, mstates x dofs)
        """
        imgsize = cov.shape[0]*cov.shape[1]
        image = _np.zeros((imgsize, imgsize))
        mstates = cov.shape[0]
        n_dofs = cov.shape[1]
        if not normalized:
            #compute the mean variance for effort and motion domains for normalizing them w.r.t. each other:
            sigma_per_mstate = _np.sqrt(_np.mean(_t.getDiagView(cov), axis=1))
            scaler = 1.0 / _np.dot(sigma_per_mstate[:,_np.newaxis], sigma_per_mstate[_np.newaxis,:])
            cov = cov * scaler[:,_np.newaxis,:,_np.newaxis]
        
        #cov = _np.transpose(cov_reshaped, (1,0,2,4,3,5))
        image =_np.reshape(cov, (imgsize,imgsize) )
        gridvectorX = _np.arange(0, imgsize, 1)
        gridvectorY = _np.arange(imgsize,0, -1)

        fig = _plt.figure(figsize=(3.4,3.4))
        _plt.pcolor(gridvectorX, gridvectorY,image, cmap=_cmapCorrelations, vmin=-1, vmax=1)
        
        _plt.axis([0, imgsize, 0,imgsize])
        _plt.gca().set_aspect('equal', 'box')

        for j in range(0,mstates):
                if j== 0:
                    linewidth=1.0
                else:
                    linewidth=0.1
                _plt.axhline(j * n_dofs , color='k', linewidth=linewidth)
                _plt.axvline(j * n_dofs, color='k', linewidth=linewidth)

        if interpolationKernel is None:
            elementnames = list(range(mstates,0,-1))
        else:
            elementnames = interpolationKernel.mStateNames
        ticks = range(imgsize - n_dofs//2,0, -n_dofs)
        _plt.xticks(ticks, elementnames)
        ticks = range( n_dofs//2, imgsize, n_dofs)
        _plt.yticks(ticks, elementnames)
        _plt.colorbar(shrink=0.6, aspect=40, ticks=[-1,0,1], fraction=0.08)
        _plt.title("Covariances (scaled)")
        ax = _plt.gca()        
        textlevel2_offset = 0.08
#        _plt.text(0.25,-1.5*textlevel2_offset, "effort", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
#        _plt.text(0.75,-1.5*textlevel2_offset, "motion", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
#        _plt.text(-textlevel2_offset, 0.75, "effort", horizontalalignment='center', verticalalignment='center', rotation='vertical', transform=ax.transAxes)
#        _plt.text(-textlevel2_offset, 0.25, "motion", horizontalalignment='center', verticalalignment='center', rotation='vertical', transform=ax.transAxes)
        #_plt.tight_layout()




class ProMP(object):
    """
    This class implements the Probabilistic Motion Primitives as described in the paper:

    [1] A. Paraschos, C. Daniel, J. Peters, and G. Neumann,
    “Using probabilistic movement primitives in robotics,” Autonomous Robots, pp. 1–23, 2017.


    Modifications to the notation have been done to improve usability/generalization.

    A ProMP defines a distribution of trajectories over a number of degrees of freedom by
    specifying a discrete number of probabilistic "supports" along the trajectory of each DOF.
    In the original paper, the distribution is defined by the Theta parameters (mean and
    covariances of a multivariate gaussian distribution), which (when sampled) yields a
    weight matrix w which can be multiplied with an interpolation tensor (Psi) to yield
    the generalized states for all DOFs at any given phase.

    The (phase-dependent) interpolation tensor Psi is computed for a given phase by a helper class.
    In the original paper, the interpolation is governed over phase/time by gaussian functions,
    which is implemented by the class InterpolationKernelGaussian.
    Furhtermore, the interpolation class provides interpolation not only for position,
    but for the generalized state, i.e. the tensor Psi maps from weight matrix w to a matrix of
    dofs and their mechanical state (force, position, velocity,..).


    This class deviates from the referenced paper in one important aspect: instead of "flattening" all
    combinations of degrees of freedom and supports (w vectors) into a single long vector, we keep
    these properties segregated on different axes during the computations. Therefore, the class
    asssumes four distinct domains on which it operates:
        In parameter space:
            dofs_w: number of concurrent trajectories in parameter space (usually equal to dofs_y)
            supports: the set of time-dependent functions used as a basis to represent trajectories
        In observation space:
            dofs_y: degrees of freeedom in observation space (i.e. the controlled joints)
            mechanical state: variables used to represent a generalized state
                (usually 0th and 1st derivative of position plus effort: torque, position, velocity)

    Consequently, the normal distribution in parameter space is represented by:
        * a means matrix of size (domains x supportCount x dofs_w)
        * a covariance tensor of size (domains, supports, dofs_w, domain, supports, dofs_w).

    The tensor psi maps from parameter space into observations space, and therfore is
    a tensor of size < mstates, dofs_y | domains, supports, dofs_y >

    To convert covariances between internal and paper representations, you can use the
    functions flattenCovarianceTensor() and foldCovarianceMatrix()


    Note on the seemingly superfluous/computationally inefficient distinction between dofs_y and dofs_w:
      The association between a specific dof in w-space to a specific dof in y space is arbitrary.
      Practically though,there is no reason to map dofs_w to dofs_y other than 1:1.
      In this module, the mapping is explicitly defined by the tensor psi. The explicit
      separation makes all equations for the single-dof case generalize directly to the multi-dof case
      via tensor notation (Ricci calculus)


    Missing Features:
        * Untested: conditioning distribution on observations



    """

    def __init__(self, meansMatrix, covarianceTensor, interpolationKernel=None, name="unnamed", derivativesCountEffort=0, derivativesCountMotion=2, fakeSigmaTaus=None, expectedDuration=1.0, phaseVelocityRelativeFloor=0.0):
        """
            meansMatrix: matrix of means of each DOF at each support point ( (supportsEffort+supportsMotion) x dofs)
            
            covarianceTensor: 4D matrix of covariances, for all pairs of (support, dof) tuples,
            i.e. the covariances between all dofs at all supports
            shape: < supports x dofs | supports x dofs >

            interpolationKernel: reference to object used to compute interpolation vectors.
                                  If None, a default InterpolationKernelGaussian() is instantiated
            
            name:  Name of the movement primitive (for introspection, identification)
            
            tauDerivativesCount: The number of variables from the distributino used to describe torque
                        By convention, the distribution is defined over the state vector (dtau, tau, q, dq, ddq...)
                        tauDerivativesCount is the number of elements dedicated to describing tau
                        When tauDerivativesCount = 0, the class behaves like described in the original ProMP papers [1]
                        
                        When tauDerivativesCount = 1, then the object describes a joint distribution over position, velocity AND force
            
            expectedDuration: time the promp is expected to execute in. Use it for scaling plots and general debugging of learning
                        
        """
        self.name = name
        self.phaseAssociable = True #indicate that this motion generator is parameterized by phase
        self.timeAssociable = False #indicate that this motion generator is not parameterizable by time
        self.tolerance=1e-7
        self.expectedDuration  =  expectedDuration
        self.phaseVelocityRelativeFloor = phaseVelocityRelativeFloor
        if interpolationKernel is None: #create a default interpolation kernel
            supportsCount, dofs = meansMatrix.shape
            if derivativesCountEffort > 0:
                supportsCount = [supportsCount//2, supportsCount//2]
            self._md = _MechanicalStateDistributionDescription(dofs=dofs, derivativesCountMotion=derivativesCountMotion, derivativesCountEffort=derivativesCountEffort)
            self.interpolationKernel = _ik.InterpolationKernelGaussian(self._md, supportsCount)
        else:
            self.interpolationKernel = interpolationKernel
            self._md = self.interpolationKernel._md

        #fake Effort if requested:
        self.fakeEffort(fakeSigmaTaus)
        self.setInitialPriorDistribution( meansMatrix, covarianceTensor)
        self.resetDistribution()


    def fakeEffort(self, fakeSigmaTaus):
        """
        set the distributions torque-related covariances to a fixed value
        in case it isn't provided by the ProMP
        
        """
        if fakeSigmaTaus is None: 
           self.fakeEffortStates = False
           self.fakeSigmaTaus=None
        elif 'torque' in self._md.mStateNames2Index:
           raise RuntimeWarning("ProMP already provides torques! Ignoring fake torques")
           self.fakeEffortStates = False
           self.fakeSigmaTaus=None
        else:
            self.fakeEffortStates = True
            m = self._md.derivativesCountMotion + 1
            fakeSigmaTaus = _np.asarray(fakeSigmaTaus)
            if  fakeSigmaTaus.shape == (m,self._md.dofs):
                self.fakeSigmaTaus = fakeSigmaTaus
            elif fakeSigmaTaus.size == m:
                self.fakeSigmaTaus = fakeSigmaTaus.reshape((m,1)).repeat(self._md.dofs, axis=1)
            else:
                raise ValueError(fakeSigmaTaus.shape)

        if self.fakeEffortStates:
            self.derivativesCountEffort = 1
            self.derivativesCountMotion = self._md.derivativesCountMotion
            self.mstateNames = ['torque'] + self._md.mStateNames
            self.mStateNames2Index = {'torque': 0}
            for name in self._md.mStateNames2Index:
                self.mStateNames2Index[name] = self._md.mStateNames2Index[name] + 1
        else:
            self.derivativesCountEffort = self._md.derivativesCountEffort
            self.derivativesCountMotion = self._md.derivativesCountMotion
            self.mstateNames = self._md.mStateNames
            self.mStateNames2Index = self._md.mStateNames2Index
        self.mechanicalStatesCount = self.derivativesCountEffort + self.derivativesCountMotion

    def serialize(self):
        """

        returns a python dictionary that contains all internal data
        necessary to recreate this ProMP

        """
        serializedDict = self.interpolationKernel.serialize()
        serializedDict[u'promp class'] = type(self).__name__
        serializedDict[u'multivariate normal distribution means'] = self.meansMatrix
        serializedDict[u'multivariate normal distribution covariances'] = self.covarianceTensor
        if self.fakeEffortStates:
            serializedDict[u'fake effort position variance'] = self.kp
            serializedDict[u'fake effort velocity variance'] = self.kd
        serializedDict[u'serialization format version'] = "2"  #version 1 differs in the index order of the covariance tensor
        serializedDict[u'name'] = self.name
        serializedDict[u'expected duration'] = self.expectedDuration
        serializedDict[u'phase velocity relative floor'] = self.phaseVelocityRelativeFloor
        return serializedDict

    def saveToFile(self, forceName=None, path='./', withTimeStamp=False):
        """
        save the (current) ProMP to the given file

        The data can be used by ProMPFactory to recreate the ProMP

        Note: the current, not the initial distribution is saved

        Format: Matlab 7.3 MAT file
        """
        d  = self.serialize()
        if forceName is not None:
            d[u'name']=forceName
        
        if withTimeStamp:
            filename = '{0}_{1}.promp.mat'.format(_time.strftime('%Y%m%d%H%M%S'), d[u'name'])
        else:
            filename = '{0}.promp.mat'.format(d[u'name']) 
        filepath= _os.path.join(path, filename)
        _h5.write(d, filename=filepath, store_python_metadata=True, matlab_compatible=True)
        return filepath

    def setInitialPriorDistribution(self, meansMatrix, covarianceTensor, priorSampleCount = 1):
        """
        update the prior distribution of the ProMP

        (prior to execution, not prior to learning)
        """
        if self._md.derivativesCountEffort == 0 and  meansMatrix.shape[0] == 2*self.interpolationKernel.supportsCount:
            #assume we got handed distributions for torques too, but we don't model them. drop them silently
            meansMatrix = meansMatrix[self.interpolationKernel.supportsCountEffort:,:]
            covarianceTensor = covarianceTensor[self.interpolationKernel.supportsCountEffort:,:, self.interpolationKernel.supportsCountEffort:,:]
# commented out, we'd need to look at the correlation matrix instead of covariance matrix for a sensible common tolerance:
#        maxerror = _np.max(_np.abs(T(covarianceTensor) - covarianceTensor))
#        if _np.any( maxerror > self.tolerance):
#            print("Notice: covariance tensor is not symmetric ({0}). This usually is an error".format(maxerror))
        minEigenVal = _np.min(eigenValues(covarianceTensor))
        if minEigenVal < -self.tolerance:
            print("Notice: covariance tensor is not positive semi-definite (smallest eigenvalue={0}). This usually is an error.".format(minEigenVal))
        #set the distribution unconditioned on observed data:
        self.priorMeansMatrix = meansMatrix.copy()
        self.priorPrecisionTensorSum = priorSampleCount * pinv(covarianceTensor)
        self.priorPrecisionTensorSumCount = priorSampleCount

    def setInitialPriorDistributionFromCurrent(self):
        """
        set the prior distribution to the current distribution

        This method can be used make conditioned ProMPs "permanent" w.r.t. calling reset()
        """
        self.priorMeansMatrix = self.meansMatrix
        self.priorPrecisionTensorSum = self.precisionTensorSum
        self.priorPrecisionTensorSumCount = self.precisionTensorSumCount
        
    def resetDistribution(self):
        """
        reset the ProMP (posterior) distribution to the initial prior, i.e. forget all observations
        """
        self.meansMatrix = self.priorMeansMatrix.copy()
        self.priorPrecisionTensorSum = self.priorPrecisionTensorSum.copy()
        self.meansMatrixWeightedSum = dot(self.priorPrecisionTensorSum, self.meansMatrix, shapes=((2,2),(2,0)))
        self.precisionTensorSumCount = self.priorPrecisionTensorSumCount
        self.covarianceTensor = self.precisionTensorSumCount * pinv(self.priorPrecisionTensorSum)
        self.observations = []
        

    def conditionToObservation(self, phaseObserved, meansObserved, sigmasObserved, dofsAdded=1):
        """
        Add an observation to the ProMP and update the ProMP's trajectory distribution accordingly

        meansObserved: matrix of shape (mstates x dofs)
        sigmasObserved: matrix of shape (mstates x dofs) with observation variances

        """
        if sigmasObserved.ndim == 2:
            covarianceTensorObservation = _np.einsum('ij,kl->ijkl', sigmasObserved, sigmasObserved)
        elif sigmasObserved.ndim == 4:
            covarianceTensorObservation = sigmasObserved
        else:
            raise ValueError("can only take 2D-array of sigmas or a full covariance tensor")
        
        if self._md.derivativesCountEffort == 0 and  meansObserved.shape[0] == 2*self.interpolationKernel.supportsCount:
            #assume we got handed distributions for torques too, but we don't model them. drop them silently
            meansObserved = meansObserved[self.interpolationKernel.supportsCountEffort:,:]
            covarianceTensorObservation = covarianceTensorObservation[self.interpolationKernel.supportsCountEffort:,:, self.interpolationKernel.supportsCountEffort:,:]

        #project covariances into weight space:
        Psi = self.interpolationKernel.getPsi(phaseObserved)
        
        #compute the updates to the gaussian distribution:
        #
        # First, assume that the current precision tensor (inverse of the covariance tensor) is the result of n observations.
        # Then the expectation of n times the precision tensor follows a Wishart distribution of W(precisionTensorSum_n, dofs+n)
        # where:
        #           precisionTensorSumCount_n = n
        #                precisionTensorSum_n = sum( n precision tensors )
        #
        # To retrieve the expectation of the covariance tensor at n, we can compute:
        #          CovTensor_n  = n * inv(precisionTensorSum_n)
        #
        # In order to add a new observation, we can update the Wishart distribution's parameters: 
        # 
        #           precisionTensorSum_{n+1} = precisionTensorSum_n + inv(covTensorObserved)
        #      precisionTensorSumCount_{n+1} = precisionTensorSumCount_n + 1
        #
        precisionObservation_y = pinv(covarianceTensorObservation)
        L = dot(T(Psi),precisionObservation_y, shapes=((2,2),(2,2))) #intermediate result for reuse
        precisionObservationWeightSpace =  dot(L, Psi, shapes=((2,2),(2,2))) 
        meansObservationWeightedWeightSpace = dot(L, meansObserved, shapes=((2,2),(2,0)))
        precisionObservationSum = precisionObservationWeightSpace * dofsAdded
        
        #update statistics:
        self.priorPrecisionTensorSum +=  precisionObservationSum
        self.priorPrecisionTensorSumCount += dofsAdded
        self.meansMatrixWeightedSum += meansObservationWeightedWeightSpace
        
        #update the covariance tensor and means matrix from the updated statistics:
        covarianceTensorDivididedByCount = pinv(self.priorPrecisionTensorSum)
        self.meansMatrix  = dot(covarianceTensorDivididedByCount, self.meansMatrixWeightedSum)
        self.covarianceTensor = self.precisionTensorSumCount * covarianceTensorDivididedByCount
        self.observations.append((meansObserved, sigmasObserved, dofsAdded))





    def sample(self):
        """
        return a parameter sample from the ProMP's distribution

        returns a (supports x dofs_w) matrix of actual values for each support and each dof
        """
        means=self.meansMatrix.T.flatten()
        covariances = flattenCovarianceTensor(self.covarianceTensor)
        sample = _np.random.multivariate_normal(means, covariances)
        sample.shape = (self.interpolationKernel.dofs_w, self.interpolationKernel.supportsCount)
        return sample.T

    def getInstantStateVectorDistribution(self, phase=0.5, phaseVelocity=1.0, currentDistribution=None, phaseVelocitySigma=None):
        """
        return the distribution of the (possibly multidimensional) state at the given phase
           for now, it returns the parameters (means, derivatives) of a univariate gaussian

            phase: at which phase the distribution should be computed
            phaseVelocity: how fast does the phase progress in time? (d phi / dt) 
                        It is needed in case you want your distribution's derivatives to be scaled correctly 
                        
            currentDistribution: not used by ProMP
            currentMassMatrixInverse: not used by ProMP
            
            phaseVelocitySigma: expected variation of the phase velocity. 
                    If set to None or 0, phase velocity is assumed to be perfectly known (original ProMP behavior)
                    Else, its influence is approximated by computing the covariances from two phase velocities 2*phaseVelocitySigma apart
                    The latter avoids (computationally problematic and practically unnecessary) zero variances in velocity (and torque) when phase velocities approach zero

           means: (derivatives x dofs) array
           covariances: (derivatives x dofs x dofs x derivatives) array

        """
        #map from parameter space to joint space:
        psi_time = self.interpolationKernel.getPsiTime(phase, phaseVelocity)  #psi has shape (phase_derivatives x dofs_y  x supports x dofs_w)
        means = dot(psi_time, self.meansMatrix)  # result: (derivatives x dofs_y)

        psi_variance = psi_time
        #negative time direction also flips some covariances which implies negative gains (e.g. kp). 
        #While theoretically correct, we usually do not want to reverse time for feedback gains as we cannot
        #reverse time direction for the controlled plant itself.
        #Therefore, we use absolute phase velocities for mapping the covariances:
        if _np.sign(phaseVelocity) < 0.0:
            #self.interpolationKernel.reverseTimeDirection(psi_variance)
            phaseVelocity = -phaseVelocity
            psi_variance = None

        if phaseVelocity * self.expectedDuration < self.phaseVelocityRelativeFloor:
            phaseVelocity = self.phaseVelocityRelativeFloor / self.expectedDuration
            psi_variance = None
        #Instead of computing the theoretic Cov_y = (Psi . Cov_w . Psi^T), 
        #we compute it at two phase velocities in order to approximate the influence of phase velocity variations at very small phase velocities
        #This also avoids zero velocity variances when phase velocity is zero
        if phaseVelocitySigma is not None:
            phaseVelocity  = max(phaseVelocity, phaseVelocitySigma)
            psi_variance =  None

        if psi_variance is None:
            psi_variance = self.interpolationKernel.getPsiTime(phase, phaseVelocity)  #compute the delta in psi given a change in velocity
        covariances = dot(psi_variance, self.covarianceTensor, T(psi_variance)) #textbook implementation

        
        
        if self.fakeEffortStates: #torques are not modelled by the distribution, but we can fake them by assuming that the observed motion was produced by a PD-controller:
            iTau = self.mStateNames2Index['torque']
            iVel = self.mStateNames2Index['velocity']
            iPos = self.mStateNames2Index['position']
            meansMotionOnly = means
            means = _np.empty((self._md.mechanicalStatesCount+1,self._md.dofs))
            means[iPos:,:] = meansMotionOnly
            means[iTau,:] = 0

            covariancesMotion = covariances
            covariances = _np.zeros((self.mechanicalStatesCount, self._md.dofs, self.mechanicalStatesCount, self._md.dofs))
            covariances[1:,:, 1:,:] = covariancesMotion
            for d in range(self._md.dofs):
                covariances[iTau,d, iPos,d] = self.fakeSigmaTaus[1,d]  * _np.sqrt(covariances[iPos,d, iPos,d])
                covariances[iTau,d, iVel,d] = self.fakeSigmaTaus[2,d]  * _np.sqrt(covariances[iVel,d, iVel,d])
                covariances[iTau,d, iTau,d] = self.fakeSigmaTaus[0,d]**2 + self.fakeSigmaTaus[1,d]**2 + self.fakeSigmaTaus[2,d]**2
                covariances[iPos,d, iTau,d] = covariances[iTau,d, iPos,d]
                covariances[iVel,d, iTau,d] = covariances[iTau,d, iVel,d]
        return (means, covariances)


    def getCovarianceMatrix(self):
        """
        returns the covariance matrix of the trajectory parameters distribution
            has size ( supports*dofs x supports*dofs )
        """
        return flattenCovarianceTensor(self.covarianceTensor)





    def plot(self, dofs='all',
                   num=100,
                   linewidth=0.5,
                   withSampledTrajectories=100,
                   withConfidenceInterval=True,
                   withSupportsMarked=False,
                   withGainsPlots=True,
                   posLimits=None,
                   velLimits=None,
                   torqueLimits=None,
                   sampledTrajectoryStyleCycler=_plt.cycler('color', ['#6666FF']),
                   scaleToExpectedDuration=True,
                   ):
        """
        visualize the trajectory distribution as parameterized by the means of each via point,
        and the covariance matrix, which determins the variation at each via (sigma) and the
        correlations between variations at each sigma

        E.g. usually, neighbouring via points show a high, positive correlation due to dynamic acceleration limits


        interpolationKernel:  provide a function that defines the linear combination vector for interpolating the via points
                                If set to None (default), promp.makeInterpolationKernel() is used to construct a Kernel as used in the ProMP definition
        """
        supportsColor = '#008888'
        confidenceColor = "#DDDDDD"
        meansColor = '#BBBBBB'
        observedColor = '#880000'
        kpColor = '#DD0000'
        kvColor = '#008800'
        kpCrossColor = '#DD8888'
        kvCrossColor = '#88FF88'
        c = _np.linspace(0.0, 1.0, num)
        if scaleToExpectedDuration:
            dcdts = _np.full_like(c, 1.0/self.expectedDuration)
        else:
            dcdts = _np.full_like(c, 1.0)

        limits_tightest = {
            'torque': [-1,1],
            'position': [0.0,1.0],
            'velocity': [-1.0,1.0],
            'gains': [-10,100.0],
        }
        if scaleToExpectedDuration:
            units={
                'torque': '[Nm]',
                'position': '[rad]',
                'velocity': '[rad/s]',
                'gains': '[Nm/rad], [Nm/rad/s]',
            }
        else:
            units={
                'torque': '[Nm]',
                'impulse rate': '[Nms/1]',
                'position': '[rad]',
                'velocity': '[rad/1]',
                'gains': '[Nm/rad/1]',
            }

        limits={}
        for limitname in self.mstateNames:
            limits[limitname] = limits_tightest[limitname]
        
        a,b = 0.5*_np.min(self.meansMatrix),0.5*_np.max(self.meansMatrix)
        avg = a+b
        delta = max(b-a, 0.1*abs(avg))
        ylimits = [ avg-delta, avg + delta ]
        limits['position'] = list(ylimits)

        if dofs=='all':
            dofs=list(range(self._md.dofs))

        mstates = self._md.mechanicalStatesCount
        mstateNames = self._md.mStateNames
        iPos =self._md.mStateNames2Index['position']
        iVel =self._md.mStateNames2Index['velocity']
        
        if self.fakeEffortStates:
            mstates += 1
            mstateNames = ['torque'] + mstateNames
            iPos += 1
        
        subplotfigsize=2.0
        plotrows=mstates
        plotrownames= mstateNames
        if withGainsPlots:
            plotrows=mstates + 1
            plotrownames= mstateNames + ['gains']
            limits['gains'] = limits_tightest['gains']
            
        fig, axesArray = _plt.subplots(plotrows,len(dofs), squeeze=False, figsize=(max(len(dofs), plotrows)*subplotfigsize, plotrows*subplotfigsize), sharex='all', sharey='row')
            
        #gather the data:
        data_mean  = _np.empty((num,mstates, self._md.dofs))
        data_sigma = _np.empty((num,mstates, self._md.dofs))
        data_gains = _np.empty((num,self._md.dofs,2, self._md.dofs))
            
        for i,phase in enumerate(c):
            phaseVelocity = dcdts[i]
            means, covariances =  self.getInstantStateVectorDistribution(phase, phaseVelocity=phaseVelocity)
            data_mean[i,:,:] = means
            data_sigma[i,:,:] = _np.sqrt( getDiagView(covariances) )
            if mstates ==3:
                data_gains[i,:,:,:]  = _controller.extractPDGains(covariances)

        #draw the confidence intervals and means/variance indicators for the supports
        for i, dof in enumerate(dofs):
            #plot the zero-variance trajectory + 95% confidence interval
            if withConfidenceInterval:
                for m in range(data_mean.shape[1]):
                    meanvalues = data_mean[:,m,dof]
                    sigmavalues = data_sigma[:,m,dof]
                    axesArray[m,i].fill_between(c,meanvalues-1.96*sigmavalues, meanvalues+1.96*sigmavalues, label="95%",  color=confidenceColor)
                    axesArray[m,i].plot(c,meanvalues, label="mean",  color=meansColor)

            if withSupportsMarked:
                sigmas = _np.sqrt(_np.diag(self.covarianceTensor[:,dof,:,dof]))
                for x,y,d in zip(self.interpolationKernel.phasesOfSupports, self.meansMatrix[:,dof], sigmas):
                    self._drawGaussianDist(axesArray[ iPos, i], x,y,d,supportsColor)

            if withConfidenceInterval or withSupportsMarked:
                for m in range(data_mean.shape[1]):
                    for limitname in mstateNames:
                        mstateIndex = self.mStateNames2Index[limitname]                        
                        ymin = _np.min(data_mean[:,mstateIndex]-1.96*data_sigma[:,mstateIndex])
                        ymax = _np.max(data_mean[:,mstateIndex]+1.96*data_sigma[:,mstateIndex]) 
                        if not _np.isnan(ymin):
                            limits[limitname][0] = min(ymin, limits[limitname][0])
                        if not _np.isnan(ymax):
                            limits[limitname][1] = max(ymax, limits[limitname][1])

            if withGainsPlots:
                limits['gains'][0] = min(limits['gains'][0], _np.min(data_gains))
                limits['gains'][1] = max(limits['gains'][1], _np.max(data_gains))
                axesArray[mstates+0,i].axhline(0.0, label=None,  color=(0.4,0.4,0.4), linestyle=':')
                for j in enumerate(dofs):
                    if j!=i:
                        axesArray[mstates+0,i].plot(c,data_gains[:,i,0,j], label=None,  color=kpCrossColor, linestyle=':')
                        axesArray[mstates+0,i].plot(c,data_gains[:,i,1,j], label=None,  color=kvCrossColor, linestyle=':')
                #plot the joint-local gains prominently and on top:
                axesArray[mstates+0,i].plot(c,data_gains[:,i,0,i], label="gain kp",  color=kpColor)
                axesArray[mstates+0,i].plot(c,data_gains[:,i,1,i], label="gain kv",  color=kvColor)


        #sample the distribution to plot actual trajectories, times the number given by "withSampledTrajectories":
        alpha = _np.sqrt(2.0 / (1+withSampledTrajectories))
        for ax in axesArray.flatten():
            ax.set_prop_cycle(sampledTrajectoryStyleCycler)

        for j in range(withSampledTrajectories):
            yvalues = self.sampleTrajectory(c, dcdt = dcdts)
            #update the desired plotting limits:
            for m in range(len(mstateNames)):
                ymin = _np.min(yvalues[:,m,:])
                ymax = _np.max(yvalues[:,m,:])
                limits[mstateNames[m]][0] = min(limits[mstateNames[m]][0],ymin)
                limits[mstateNames[m]][1] = max(limits[mstateNames[m]][1],ymax)
                for i, dof in enumerate(dofs):
                    axesArray[ m, i].plot(c, yvalues[:,m,dof], alpha=alpha, linewidth=linewidth )

        if 'observedTrajectories' in self.__dict__:
            for traj in self.observedTrajectories:
                phase, duration, posvalues, velvalues, tauvalues = traj
                
                if scaleToExpectedDuration:
                    times = phase * self.expectedDuration
                    dcdt = 1.0 / self.expectedDuration
                else:
                    times = phase 
                    dcdt = 1.0
                if tauvalues is not None:
                    m = 0
                    #limits['torque'][0] = min(limits['torque'][0], _np.min(tauvalues))
                    #limits['torque'][1] = max(limits['torque'][1], _np.max(tauvalues))
                    for i, dof in enumerate(dofs):
                        axesArray[ m, i].plot(phase, tauvalues[dof,:]*dcdt, alpha=alpha, linewidth=linewidth, color=observedColor )

                m = iPos
                limits['position'][0] = min(limits['position'][0], _np.min(posvalues))
                limits['position'][1] = max(limits['position'][1], _np.max(posvalues))
                for i, dof in enumerate(dofs):
                    axesArray[ m, i].plot(phase, posvalues[dof,:], alpha=alpha, linewidth=linewidth, color=observedColor )
                m = iPos+1
                if velvalues is None: #compute velocities from the positions if they are not specified
                    d_posvalues = (posvalues[:,1:] - posvalues[:,:-1]) / (phase[1:] - phase[:-1]) * dcdt
                else:
                    d_posvalues = (velvalues[:,1:]+velvalues[:,1:])* 0.5* dcdt
                d_phasevalues = 0.5* (phase[1:]+phase[:-1])
                limits['velocity'][0] = min(limits['velocity'][0], _np.min(d_posvalues))
                limits['velocity'][1] = max(limits['velocity'][1], _np.max(d_posvalues))
                for i, dof in enumerate(dofs):
                    axesArray[ m, i].plot(d_phasevalues, d_posvalues[dof,:], alpha=alpha, linewidth=linewidth, color=observedColor )

        #override scaling:
        if posLimits is not None:
            limits['position'] = posLimits
        if velLimits is not None:
            limits['velocity'] = velLimits
        if torqueLimits is not None:
            limits['torque']=torqueLimits

        padding=0.05
        for i, dof in enumerate(dofs):
            for m in range(plotrows):
                axes = axesArray[m,i]  
                axes.set_title(r"{0} {1}".format(plotrownames[m], dof))
                axes.set_xlim((0.0, 1.0))
                if m == plotrows - 1:
                    axes.set_xticks([0.0,0.5, 1.0])
                    if scaleToExpectedDuration:
                        axes.set_xticklabels(['0', 'time [s]', '{0:0.1f}'.format(self.expectedDuration)])
                    else:
                        axes.set_xticklabels(['0', 'phase', '1'])
                else:
                    axes.get_xaxis().set_visible(False)
                if i == 0:
                    axes.set_ylabel(units[plotrownames[m]])
                else:
                    axes.get_yaxis().set_visible(False)
                lim = limits[plotrownames[m]]
                avg = _np.mean(lim)
                delta = max(0.5*(lim[1]-lim[0]), 0.1*abs(avg))
                ymax = avg+delta*(1+padding)
                ymin = avg-delta*(1+padding)
                axes.set_ylim(ymin,ymax )
        _plt.tight_layout()


    def plotCovarianceTensor(self, normalized=True):
        """
        plot a correlation tensor of shape (supports x dofs_w, supports x dofs_w,)
        """
        cov = self.covarianceTensor
        
        imgsize = cov.shape[0]*cov.shape[1]
        image = _np.zeros((imgsize, imgsize))
        n_supports = cov.shape[0]
        n_domains = 2
        n_dofs = cov.shape[1] // n_domains
        cov_reshaped = _np.reshape(cov.copy(), (n_supports, n_domains, n_dofs, n_supports, n_domains, n_dofs))

        if not normalized:
            #compute the mean variance for effort and motion domains for normalizing them w.r.t. each other:
            varsum = 0
            for i in range(n_dofs):
                for j in range(n_supports):
                    varsum += cov_reshaped[j,0,i,j,0,i]
            variances_effort = varsum / (n_dofs*n_supports)
            varsum = 0
            for i in range(n_dofs):
                for j in range(n_supports):
                    varsum += cov_reshaped[j,1,i,j,1,i]
            variances_motion = varsum / (n_dofs*n_supports)
            cov_reshaped[:,0,:,:,0,:] *= 1.0/variances_effort
            cov_reshaped[:,1,:,:,1,:] *= 1.0/variances_motion
            cov_reshaped[:,0,:,:,1,:] *= 1.0/_np.sqrt(variances_motion*variances_effort)
            cov_reshaped[:,1,:,:,0,:] *= 1.0/_np.sqrt(variances_motion*variances_effort)
        
        
        cov_reordered = _np.transpose(cov_reshaped, (1,0,2,4,3,5))
        image =_np.reshape(cov_reordered, (imgsize,imgsize) )
        if normalized:
            normalizer = _np.diag(1.0 / _np.sqrt(_np.diag(image)))
            image = _np.dot(_np.dot(normalizer, image),normalizer)
        gridvectorX = _np.arange(0, imgsize, 1)
        gridvectorY = _np.arange(imgsize,0, -1)

        fig = _plt.figure(figsize=(3.4,3.4))
        _plt.pcolor(gridvectorX, gridvectorY,image, cmap=_cmapCorrelations, vmin=-1, vmax=1)
        
        _plt.axis([0, imgsize, 0,imgsize])
        _plt.gca().set_aspect('equal', 'box')

        for i in range(n_domains):
            for j in range(0,n_supports):
                if j == 0 and i==0:
                    continue
                if j== 0:
                    linewidth=1.0
                else:
                    linewidth=0.1
                _plt.axhline(i *n_supports*n_dofs + j * n_dofs , color='k', linewidth=linewidth)
                _plt.axvline(i *n_supports*n_dofs + j * n_dofs, color='k', linewidth=linewidth)

        elementnames = list(range(n_supports,0,-1)) + list(range(n_supports,0,-1))
        ticks = range(imgsize - n_dofs//2,0, -n_dofs)
        _plt.xticks(ticks, elementnames)
        ticks = range( n_dofs//2, imgsize, n_dofs)
        _plt.yticks(ticks, elementnames)
        _plt.colorbar(shrink=0.6, aspect=40, ticks=[-1,0,1], fraction=0.08)
        if normalized:
            _plt.title("Correlations")
        else:
            _plt.title("Covariances (scaled)")
        ax = _plt.gca()        
        textlevel2_offset = 0.08
        _plt.text(0.25,-1.5*textlevel2_offset, "effort", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        _plt.text(0.75,-1.5*textlevel2_offset, "motion", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        _plt.text(-textlevel2_offset, 0.75, "effort", horizontalalignment='center', verticalalignment='center', rotation='vertical', transform=ax.transAxes)
        _plt.text(-textlevel2_offset, 0.25, "motion", horizontalalignment='center', verticalalignment='center', rotation='vertical', transform=ax.transAxes)
        #_plt.tight_layout()

    def _drawGaussianDist(self, axes, x,y,d, supportsColor):
        """
        draw a poor-man's gaussian
        """
        w=0.005
        axes.fill((    x, x+0.5*w, x+2*w, x+0.5*w,     x, x-0.5*w, x-2*w, x-0.5*w,   x),
            (y-2*d,     y-d,     y,     y+d, y+2*d,     y+d,     y, y-d, y-2*d,),
            linewidth=0, alpha=1, color=supportsColor)


    def sampleTrajectory(self, c, dcdt=1.0, noise=0, supportvalues=None):
        """
        return a function that represents a single trajectory, sampled from the promp's distribution

        The returned function maps generalized phase (phase, phase velocity) to generalized state (position, velocity) x dofs

        supportvalues: provide a vector (of size self.interpolationKernel.supportsCount) that contains the values of the supports
                   If None, then the trajectory is sampled from the ProMP's distribution
        """
        if supportvalues is None:
            supportvalues = self.sample()
        else:
            supportvalues = supportvalues.copy()

        mstates = self._md.mechanicalStatesCount
        offset=0
        iPos=0
        iVel=1
        c = _np.asarray(c)
        dcdt = _np.asarray(dcdt)
        if self.fakeEffortStates:
            offset = 1
            mstates += offset
            iPos += offset
            iVel += offset

        if _np.isscalar(c):
            psi = self.interpolationKernel.getPsiTime(c, dcdt)
            points = _np.zeros((mstates,self._md.dofs))
            points += rand
            points[offset:,:] =  dot(psi, supportvalues)  #derivatices x dofs
        else:
            points = _np.zeros( (c.size, mstates,self._md.dofs))
            for i in range(c.size):
                points[i,offset:,:] =  dot(self.interpolationKernel.getPsiTime(c[i], dcdt[i]), supportvalues)

        if noise > self.tolerance:  #optimize for noise=0
            points += _np.random.normal(0, noise, points.shape)

        return points





class ProMPFactory(object):
    """
    collection of class methods to create ProMPs in different ways
    """

    @classmethod
    def copy(cls, promp, name='unnamed copy'):
        """
        Duplicate a promp with a different name
        """
        return ProMP(promp.meansMatrix, promp.covarianceTensor, promp.interpolationKernel, name=name)


    @classmethod
    def makeFromGaussianDistribution(cls, name, meansMatrix, covarianceTensor):
        """
        Create a ProMP from a gaussian multivariate distribution over interpolation parameters

        meansMatrix: expectation value of the parameters, of shape (supports x dofs)
        covarianceTensor: covariances between the parameters, of shape (supports x dofs x supports x dofs)
        """
        supportsCount, dofs = meansMatrix.shape
        ik = _ik.InterpolationKernelGaussian(supportsCount, dofs)
        return ProMP(meansMatrix, covarianceTensor, ik, name=name)


    @classmethod
    def makeFromDict(cls, serializedDict):
        """
        Create a ProMP from a description yielded by ProMP.serialize()
        """
        if int(serializedDict["serialization format version"]) > 2:
            raise RuntimeError("Unknown (future?) serialization format version: {0}".format(serializedDict["serialization format version"]))

        ikclass = _ik.__dict__[serializedDict[u'interpolation kernel class']]
        ik  = ikclass(0,0,serializedDict=serializedDict)
        mu = serializedDict[u'multivariate normal distribution means']
        cov =  serializedDict[u'multivariate normal distribution covariances']
        name =  serializedDict[u'name']
        if u'expected duration' in serializedDict:
            expectedDuration = serializedDict[u'expected duration']
        else:
            expectedDuration = 1.0
        if u'phase velocity relative floor' in serializedDict:
             phaseVelocityRelativeFloor = serializedDict[u'phase velocity relative floor'] 
        else:
            phaseVelocityRelativeFloor = 0.0
            
        try:
            fakeSigmaTaus = (serializedDict[u'fake effort position variance'],serializedDict[u'fake effort velocity variance'])
            fakeEffort = True
        except KeyError:
            fakeEffort = False
            fakeSigmaTaus = None
    
        #backwards compatibility:
        if serializedDict["serialization format version"] == "1":
            print("Data are in old serialization format! Consider migrating it!")
            cov = cov.transpose(0,1,3,2)

        return ProMP(mu, cov, ik, fakeSigmaTaus=fakeSigmaTaus, name=name, expectedDuration=expectedDuration, phaseVelocityRelativeFloor=phaseVelocityRelativeFloor)

    @classmethod
    def makeReverseInPhase(cls, name, prompObj):
        """
        Create a ProMP by reversing the phase of an existing ProMP 

        """
        meansMatrix = prompObj.priorMeansMatrix[::-1,:]
        covarianceTensor = prompObj.priorCovarianceTensor[::-1,:,::-1,:]
        return ProMP(meansMatrix, covarianceTensor, prompObj.interpolationKernel, name=name)

    @classmethod
    def makeFromFile(cls, filename, h5path='/'):
        """
        Create a ProMP from a description saved in a hdf5 file using ProMP.saveToFile()
        """
        d = _h5.read(path=h5path, filename=filename)
        if isinstance(d, _np.ndarray): #un-wrap from hdf5storage
            d = d[0]

        #TODO It's probably better to make sure ascii and uncode strings are not mixed when these files are
        #       created
        myDict = {}
        for k,v in d.items():
            try:
                k = k.decode("utf-8")
            except AttributeError:
                pass
            try:
                v = v.decode("utf-8")
            except AttributeError:
                pass
            myDict[k] = v

        d = myDict
        return ProMPFactory.makeFromDict(d)


    @classmethod
    def makeFromTrajectories(cls, 
        name, 
        supportsCount, 
        positionsList, 
        velocitiesList,
        torquesList, 
        phasesList=None, 
        phaseVelocitiesList=None,
        initialVariance=1e-3,
        computeMeanFromSubset=None,
        expectedDuration = 3.0,
        useSyntheticTorques=True,
        syntheticKp=20,
        syntheticKv=10,
        phaseVelocityRelativeFloor=0.0,
        ):
        """
        Create a ProMeP from a list of position and torque trajectories

        name: name string of the new ProMeP (ProMP over motion and effort)
        
        supportsCount:  how many supports should be used to parameterize the distribution?

        positionsList: list of arrays of shape (dofs x sampleCount)
                The rows are of form [dof0, dof1, dof2,....]

        velocitiesList: None or list of arrays of shape (dofs x sampleCount)
                The rows are of form [dof0, dof1, dof2,....]
                
                Note: Velocities are not used for learning, but are used for plotting. 
                      If None is specified, then plotting infers velocities from the positions and phases

        torquesList:list of vectors giving the torques of shape (dofs x sampleCount)
                The rows are of form [dof0, dof1, dof2,....]

        phasesList: list of vectors giving the phase for each sample
                        if None, position samples are assumed to be equally spaced
                        if None, we assume all trajectories to have the same number of samples
                    (we assume samples to be sorted by phase)
        phaseVelocitiesList: list of vectors giving the phase velocity for each sample
                        if None, velocities are set by discrete differention of the phase

        initialVariance: initial variance to use to regularize the parameter learning. 
                            This is especially useful for very small sample sizes to specify a desired minimum variance
                            Note: The influence diminishes when increasing the number of samples
        
        computeMeanFromSubset: If set to a list of indices, use only those trajectory observations for computing the trajectory meanTorque
                            This can be used to exclude outlier trajectories, or synthetically generated trajectories that would only add noise
                            If None, all provided trajectories will be used to compute the distribution mean

        expectedDuration: The nominal / expected duration of the ProMP. Used to correctly scale gains during learning, 
                          and to plot velocities and gains w.r.t. time instead of phase
                            
        useSyntheticTorques: If true, create synthetic training data for the covariance matrix by emulating a pd controller on the distribution over motion
        syntheticKp, syntheticKv: position and velocity gains of the pd controller, either:
                                    - a scalar (use as SISO gain, same value for all dofs)
                                    - a vector (use as SISO gain)
                                    - arrays of shape (dofs x dofs) (MIMO gains)
                                    - a dictionary of keyframes of the above, keyed by phase. I.e. {0.0: [10,20,30], 0.5: [10,0,30], 1.0: [0,0,30]}. 
                                      (gains get linearly interpolated, beyond first and last keyframe their respective value is used)
        
        
        syntheticKv:         velocity gain of the pd controller, array of shape (dofs x sampleCount) (you can use numpy casting rules) 
                            
                            
        phaseVelocityRelativeFloor: If set to > 0, clip phase velocity to a minimal value phaseVelocityRelativeFloor * 1.0/(expectedDuration)
        """
        if len(positionsList) == 0:
            raise ValueError("no positions supplied")
        dofs = positionsList[0].shape[0]
        #todo: check that all trajectories have same dof count
        md = _MechanicalStateDistributionDescription(dofs=dofs)
        ik = _ik.InterpolationKernelGaussian(md, supportsCount)

        #compute the pseudoinverse of the array mapping parameter to observed positions / torques:
        if phasesList is None:
            print("Warning: no phase provided")
            if _np.any([ positionsList[0].shape != p.shape for p in positionsList]):
                raise ValueError("If no phases are specificed, all trajectories must have the same number of samples!")
            phases = _np.linspace(0.0, 1.0, positionsList[0].shape[-1])
            if  phaseVelocitiesList is None:
                phaseVelocitiesList = [_np.full(phases.shape, 1.0 / len(phases))] #assume dphidt=1.0
            AInv = ik.getParameterEstimator(phases)
            AInvIterator = _it.repeat(AInv)  #we can reuse the expensive computation for all other trajectories
        elif isinstance(phasesList , _np.ndarray):
            if _np.any([ positionsList[0].shape != p.shape for p in positionsList]):
                raise ValueError("If only one phase vector is specificed, all trajectories must have the same number of samples!")
            if  phaseVelocitiesList is None:
                raise ValueError("Please provide a phase velocity too")
            AInv = ik.getParameterEstimator(phasesList,phaseVelocitiesList)
            AInvIterator = _it.repeat(AInv)  #we can reuse the expensive computation for all other trajectories
            phasesList = _it.repeat(phasesList)
            phaseVelocitiesList = _it.repeat(phaseVelocitiesList)
        else:
            AInvIterator = [ik.getParameterEstimator(p, dphidt) for p,dphidt in zip(phasesList,phaseVelocitiesList) ]
        
        #select the subset of trajectories to use for means estimation (i.e. to skip synthetic trajectories)
        if computeMeanFromSubset is not None:
            useForMeanList = [False] * computeMeanFromSubset[0] + [True] * (computeMeanFromSubset[1]-computeMeanFromSubset[0]) + [False] * (len(positionsList)-computeMeanFromSubset[1])
        else:
                useForMeanList = _it.repeat(True)
        
        #compute least-squares estimates for the parameters that generated each trajectory:
        wSamples_mean = []
        observedTrajectories=[]
        if velocitiesList is None:
            velocitiesList = _it.repeat(None)
            
        for pos, vel, tau, phi, dphidt, AInv, useForMean in zip(positionsList, velocitiesList, torquesList, phasesList, phaseVelocitiesList, AInvIterator, useForMeanList):
            #estimate mean weights, save observation
            if useForMean:
                mstate = _np.stack((tau,pos), axis=1)
                w = dot(AInv, T(mstate), shapes=((2,3),(3,0)))
                wSamples_mean.append(w)
            observedTrajectories.append((phi, expectedDuration, pos, vel/dphidt, tau/dphidt ))
        wSamples_mean = _np.stack(wSamples_mean)
        meansEstimated = estimateMean(wSamples_mean, initialVariance)

        covariancesEstimated = estimateMultivariateGaussianCovariances(wSamples_mean, 1e-3)

        if useSyntheticTorques:

            promp_nosynth = ProMP(0*meansEstimated, covariancesEstimated, ik, name=name)

            sample_multiplier = 5
            sample_multiplier_supports = 5
            
            #estimate covariance matrix:
            wSamples = []
            #first, generate a reference mean trajectory:
            n = ik.supportsCount * sample_multiplier_supports
            dt=1.0/n
            meanTrajectory = _np.empty((dofs, n))
            meanTrajectoryVel = _np.empty((dofs, n))
            meanTorque = _np.empty((dofs, n))
            synth_phase = _np.linspace(0.0, 1.0, n)
            synth_phasevel = _np.full((n) , 1.0)
            iPos = md.mStateNames2Index['position']
            iVel = md.mStateNames2Index['velocity']
            iTau = md.mStateNames2Index['torque']
            
            #create mean trajectory for plotting:
            for i in range(n):
                psi = ik.getPsiTime( synth_phase[i], synth_phasevel[i] )
                y = dot(psi, meansEstimated, shapes=((2,2),(2,0)))
                meanTrajectory[:,i] = y[iPos, :]
                meanTrajectoryVel[:,i] = y[iVel, :]
                meanTorque[:,i] = y[iTau, :]
            synthAinv, synthA = ik.getParameterEstimator(synth_phase, synth_phasevel, alsoNonInv=True)

            #create the gains matrix of the PD controller used to create synthetic torques:
            K = _np.zeros(( n, md.mechanicalStatesCount, dofs, n, md.mechanicalStatesCount, dofs))

            #create synthetic gains based on keyframes: #TODO
            def cast_gains(k):
                """
                sane gains casting rule:
                    scalar -> 
                """
                if _np.isscalar(k):
                    k = _np.full(dofs, k)
                else:
                    k = _np.asarray(k)
                if k.ndim == 1:
                    k = _np.diag(k)
                return k

            for syntheticKx, factor, i1, i2 in ((syntheticKp, expectedDuration, iTau, iPos),(syntheticKv, 1.0,  iTau, iVel)):
                try: 
                    syntheticKxOrderedDict = _collections.OrderedDict(sorted(syntheticKx.items()))
                except (AttributeError, TypeError):
                    syntheticKxOrderedDict=None
                    
                if syntheticKxOrderedDict is None:
                    syntheticKxPerSample = _np.empty((dofs, n))
                    if _np.isscalar(syntheticKx):
                        syntheticKx = _np.full((dofs), syntheticKx)
                    else:
                        syntheticKx = _np.asarray(syntheticKx)
                    if syntheticKx.ndim==1:
                        syntheticKx = _np.diag(syntheticKx)
                    getDiagView(K[:, i1, :, :, i2, :])[:,:]= -syntheticKx * factor
                else: #else assume that keyframes have been specified:
                    syntheticKxKeyframeValue = _np.zeros((len(syntheticKxOrderedDict), dofs, dofs))
                    syntheticKxKeyframePhases= _np.zeros((len(syntheticKxOrderedDict)))
                    for i,k in enumerate(syntheticKxOrderedDict):
                        syntheticKxKeyframePhases[i] = k
                        syntheticKxKeyframeValue[i,:,:] = cast_gains(syntheticKxOrderedDict[k])
                    kfp_after  = 1
                    for i,p in enumerate(synth_phase):
                        while p >= syntheticKxKeyframePhases[kfp_after] and kfp_after < len(syntheticKxKeyframePhases)-1:
                            kfp_after +=1
                        a = (p - syntheticKxKeyframePhases[kfp_after-1]) / (syntheticKxKeyframePhases[kfp_after]-syntheticKxKeyframePhases[kfp_after-1])
                        a = _np.clip(a, 0.0, 1.0)
                        interpolated_K = a * syntheticKxKeyframeValue[kfp_after-1,:,:] + (1.0-a) * syntheticKxKeyframeValue[kfp_after,:,:]
                        K[i, i1, :, i, i2, :] = -interpolated_K * factor

            #complete the K tensor:
            getDiagView(K)[:,(iPos,iVel),:] = 1
                            
            #then synthesize errors and control effort:
            n_synthetic= ik.dofs_w * ik.supportsCount * sample_multiplier
            for i in range(n_synthetic):
                ydelta =  promp_nosynth.sampleTrajectory(synth_phase, synth_phasevel[:,_np.newaxis])
                ydelta_desired = dot(K, ydelta, shapes=((3,3),(3,0)) )
#                ydelta_desired[:,iTau,:] += _np.random.normal(scale=1e-2, size=(n,dofs))
                w = dot(synthAinv, ydelta_desired[:,(iTau, iPos),:], shapes=((2,3),(3,0)) )
                wSamples.append(w)
                #if i<20:
                #    observedTrajectories.append( (synth_phase, (meanTrajectory + ydelta[:,iPos,:].T)/synth_phasevel , (meanTrajectoryVel + ydelta[:,iVel,:].T)/synth_phasevel, (meanTorque + ydelta[:,iTau,:].T)/synth_phasevel ) )
            
            wSamples = _np.stack(wSamples)
            covariancesEstimated = estimateMultivariateGaussianCovariances(wSamples)

        #construct and return the promp with the estimated parameters:
        promp = ProMP(meansEstimated, covariancesEstimated, ik, name=name, expectedDuration=expectedDuration, phaseVelocityRelativeFloor=phaseVelocityRelativeFloor)
        #attach trajectory samples to promp:
        promp.wSamples=wSamples_mean
        promp.observedTrajectories = observedTrajectories
        return promp




    @classmethod 
    def makePositionMaintainer(cls, velocitySigmas=[0.1], positionSigmas=[10.0]): 
        """ 
        return a ProMP that maintains a position 
        
        Note: this is not tested, and only approximative
        """ 
        n=5 #nr of supports / basis functions 
        dofs=len(velocitySigmas) 
        derivs=2 
 
        goals = _np.zeros((derivs, dofs)) 
        goalSigmas = _np.array([positionSigmas,velocitySigmas]) 
 
        startSigmas  = 10.0 
        #create means matrix 
        meansMatrix =_np.zeros((n, dofs)) 
        dt=1.0 
        meansMatrix[n-2,:] = goals[0] - goals[1]*dt 
        meansMatrix[n-1,:] = goals[0] + goals[1]*dt 
        #create covariance matrix 
        covarianceMatrix = makeCovarianceTensorUncorrelated(n, dofs, startSigmas) 
        for dof in range(dofs): 
            covarianceMatrix[n-2,dof, n-1,dof] = 0.98*goalSigmas[0]**2 
            covarianceMatrix[n-1,dof, n-2,dof] = covarianceMatrix[n-2,dof,n-1,dof] 
            covarianceMatrix[n-3,dof, n-1,dof] = 0.98*goalSigmas[0]**2 
            covarianceMatrix[n-1,dof, n-3,dof] = covarianceMatrix[n-3,dof,n-1,dof] 
            covarianceMatrix[n-3,dof, n-2,dof] = 0.98*goalSigmas[0]**2 
            covarianceMatrix[n-2,dof, n-3,dof] = covarianceMatrix[n-3,dof,n-2,dof] 
 
            #perfectly correlate start and end supports: 
            covarianceMatrix[1,  dof, n-3,dof] = 0.98*startSigmas**2 
            covarianceMatrix[n-3,dof, 1,  dof] = covarianceMatrix[1,dof,n-3,dof] 
            covarianceMatrix[1,  dof, n-2,dof] = 0.98*startSigmas**2 
            covarianceMatrix[n-2,dof, 1,  dof] = covarianceMatrix[1,dof,n-2,dof] 
            covarianceMatrix[1,  dof, n-1,dof] = 0.98*startSigmas**2 
            covarianceMatrix[n-1,dof, 1,  dof] = covarianceMatrix[1,dof,n-1,dof] 

        #instantiate PRoMP 
        md = _MechanicalStateDistributionDescription()
        interpolationKernel = _ik.InterpolationKernelGaussian(md, n, sigma=0.5) 
        promp = ProMP(meansMatrix, covarianceMatrix, interpolationKernel=interpolationKernel) 
 
        #condition it on the controller goal: 
        promp.conditionToObservation(1.0, goals, goalSigmas) 
        promp.setInitialPriorDistributionFromCurrent() 
        return promp 



