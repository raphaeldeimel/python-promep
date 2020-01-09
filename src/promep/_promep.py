#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel
@copyright 2019
@licence: 2-clause BSD licence

This file contains the main code for representing ProMePs, visualizing them, and
also to sample trajectories from a ProMeP

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

from . import _namedtensors, _interpolationkernels

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


class MechanicalStateDistribution(object):

    def __init__(self, means, covariances):
        self.means = _np.array(means)
        self.covariances = _np.array(covariances)
        self.shape = means.shape
        self.indexNames = ['r','d','g']
        self.indexNames_transposed = ['r_','d_','g_']
        
        #create a view on the variances within the covariance tensor:
        self.variancesView = _np.einsum('ijkijk->ijk', self.covariances)
    
    def __repr__(self):
        text  = "Realms: {}\n".format(self.shape[0])
        text += "Dofs: {}\n".format(self.shape[1])
        text += "Derivatives: {}\n".format(self.shape[2])
        for g_idx in range(self.shape[2]):            
            text += "\nDerivative {}:\n       Means:\n{}\n       Variances:\n{}\n".format(g_idx, self.means[:,:,g_idx], self.variancesView[:,:,g_idx])
        return text

class JointSpaceToJointSpaceTransform(object):
    """
    
    """
    
    def __init__(self, ntm):
        self._ntm = _namedtensors.NamedTensorsManager(ntm)
        #Path to compute T:
        self._ntm.registerTensor('Xref', (('rtilde','g', 'dtilde',),()) )        
        self._ntm.registerTensor('Yref', (('r','g', 'd',),()) )        
        self._ntm.registerTensor('Jt', (('d',),('dtilde',)))
        self._ntm.registerTensor('Jinv', (('d',),('dtilde',)))
        self._ntm.registerBasisTensor('e_effort_effort', (('r',),('rtilde',)), (('effort',), ('effort',)) )
        self._ntm.registerBasisTensor('e_motion_motion', (('r',),('rtilde',)), (('motion',), ('motion',)) )
        self._ntm.registerContraction('e_effort_effort', 'Jt')
        self._ntm.registerContraction('e_motion_motion', 'Jinv')
        self._ntm.registerAddition('e_effort_effort:Jt', 'e_motion_motion:Jinv', result_name='T') #has indices (('r', 'd')('rtilde', 'dtilde'))
        self._Jt  = _np.eye(self._ntm.indexSizes['d'])  #trivial Jacobian between joint space and joint space 

        #conceptually; to also consider dotJ:
        #self._ntm.registerTensor('dotJinv', (('d',),('dtilde',)))
        #self._ntm.registerBasisTenso('e_motion_1_motion_0', (('r',),('rtilde',)), (('motion',), ('motion',)) )
        #self._ntm.registerContraction('e_motion_1_motion_0', 'dotJinv')        
        #self._ntm.registerAddition('e_effort_effort:Jt+e_motion_motion:Jinv', 'e_motion_1_motion_0:dotJinv', result_name='T') #has indices (('r', 'd')('rtilde', 'dtilde'))
        self.update()
        
    def update(self):
        self._ntm.update()

    def get_T_view(self):
        return self._ntm.tensorData['T']
    
    def get_Xref_view(self):
        return self._ntm.tensorData['Xref']

    def get_Yref_view(self):
        return self._ntm.tensorData['Yref']



class ProMeP(object):
    """
    This class implements the Probabilistic Mechanical Primitives as described in the paper:

    [TODO]

    """



    def __init__(self, index_sizes=None, Wmean=None, Wcov=None, name="unnamed", expected_duration=1.0):
        """
            index_sizes: dictionary containing the sizes of all internal indices:
                indices:
                    'r': number of realms (usually 2) 
                    'd': number of DoFs being controlled (usually joint space dof)
                    'g': number of time derivatives to consider (usually 2, zeroth and first derivative)
                    'gphi': number of phase derivatives to consider when mapping to time-based trajectories (usually equal to  g)
                    'gtilde': number of phase derivatives to consider in parameter space 
                    'stilde': number of parameters used for interpolation
                    'dtilde': DOFs in parameter space
                    'r': number of realms in parameter space (usually the same as 'rtilde')                     
                    
            Default:
            index_sizes = {
                'r':  2,  
                'd':  8,
                'g':  2,
                'gphi':  2,
                'gtilde':  2,
                'stilde':  7,
                'dtilde':  8,
                'rtilde':  2,  
            }
        
            meansMatrix: matrix of means of each DOF at each support point ( (supportsEffort+supportsMotion) x dofs)
            
            covarianceTensor: 4D matrix of covariances, for all pairs of (support, dof) tuples,
            i.e. the covariances between all dofs at all supports
            shape: < supports x dofs | supports x dofs >

            name:  Name of the movement primitive (for introspection, identification)
            
            tauDerivativesCount: The number of variables from the distributino used to describe torque
                        By convention, the distribution is defined over the state vector (dtau, tau, q, dq, ddq...)
                        tauDerivativesCount is the number of elements dedicated to describing tau
                        When tauDerivativesCount = 0, the class behaves like described in the original ProMP papers [1]
                        
                        When tauDerivativesCount = 1, then the object describes a joint distribution over position, velocity AND force
            
            expectedDuration: time the promp is expected to execute in. Use it for scaling plots and general debugging of learning
                        
        """
        self.name = name
        self._ntm = _namedtensors.NamedTensorsManager() #band-aid to have proper multilinear algebra semantics for numpy etc.

        #register all index names being used and all index sizes:
        if index_sizes == None:
            index_sizes = {
                'r':  2,
                'd':  8,
                'g':  2,
                'gphi':  2,
                'gtilde':  2,
                'stilde':  7,
                'dtilde':  8,
                'rtilde':  2,
            }
        self.index_sizes = index_sizes
        for name in index_sizes:
            if name == 'r' or 'rtilde':
                  self._ntm.registerIndex(name, index_sizes[name], values=['motion', 'effort'])
            else:
                  self._ntm.registerIndex(name, index_sizes[name])
                  
        #hooks to compute tensors PHI and T:
        self.phi_computer =  _interpolationkernels.InterpolationGaussian(self._ntm.indexSizes['gphi'], self._ntm.indexSizes['stilde'])  #should provide a getPhi method
        self.T_computer = JointSpaceToJointSpaceTransform(self._ntm) #should implement a get_T_view(), get_Xref_view() and get_Yref_view() method
        
        
        #register all tensors not being computed by operators:
        #i.e. input tensors:
        self._ntm.registerTensor('Wmean', (('rtilde','gtilde','stilde','dtilde'),()) )
        self._ntm.registerTensor('Wcov', (( 'rtilde','gtilde','stilde','dtilde'),( 'rtilde_','gtilde_','stilde_','dtilde_')) )
        self._ntm.registerTensor('PHI', (('gphi',),('stilde',)) )
        self._ntm.registerTensor('P', (('g',),('gphi','gtilde')) )
        self._ntm.registerTensor('Xref', (('rtilde','g', 'dtilde',),()), external_array = self.T_computer.get_Xref_view() )        
        self._ntm.registerTensor('Yref', (('r','g', 'd',),()) , external_array = self.T_computer.get_Yref_view())
        self._ntm.registerTensor('T', (('r','d'),('rtilde', 'dtilde')), external_array = self.T_computer.get_T_view())
        #register all operations being used on tensors:
        
        #path to compute PSI:
        self._ntm.registerContraction('P', 'PHI')
        self._ntm.registerContraction('T', 'P:PHI', result_name='PSI')
        self._ntm.registerTranspose('PSI')
        
        #compute any offsets required by the linearization of the task space map into T
        self._ntm.registerContraction('T', 'Xref')
        self._ntm.registerSubtraction('Yref', 'T:Xref', result_name='O')
        
        #project the mean of weights to mean of trajectory:
        self._ntm.registerContraction('PSI', 'Wmean')
        for key in self._ntm.tensorIndices:
            print(f"{key}      {self._ntm.tensorIndices[key]}")
        self._ntm.registerAddition('PSI:Wmean', 'O', result_name='Ymean')
        
        #project the covariances of weights to covariances of trajectory:
        self._ntm.registerContraction('PSI', 'Wcov')
        self._ntm.registerContraction('PSI:Wcov', '(PSI)^t', result_name='Ycov')


        
        
        self.phaseAssociable = True #indicate that this motion generator is parameterized by phase
        self.timeAssociable = False #indicate that this motion generator is not parameterizable by time
        self.tolerance=1e-7
        
        self.expectedDuration = expected_duration
        if Wmean is not None:
            self._ntm.setTensor('Wmean', Wmean)
        if Wcov is not None:        
            self._ntm.setTensor('Wcov', Wcov)
        
        #temporary/scratch tensors:
        self._ntm.registerTensor('Wsample', (('r','gtilde','stilde','dtilde'),()) )
        self._ntm.registerTensor('phase', (('g'),()) )  #generalized phase vector
        


    def serialize(self):
        """

        returns a python dictionary that contains all internal data
        necessary to recreate this ProMeP

        """
        serializedDict = collections.OrderedDict()
        serializedDict[u'class'] = type(self).__name__
        serializedDict[u'name'] = self.name
        serializedDict[u'promep_version'] = "3"
        serializedDict[u'index_sizes'] = self.index_sizes
        serializedDict[u'Wmean'] = self.tensorData['Wmean']
        serializedDict[u'Wcov'] = self.tensorData['Wcov']
        serializedDict[u'expected_duration'] = self.expectedDuration
        return serializedDict


    @classmethod
    def makeFromDict(cls, serializedDict):
        """
        Create a ProMeP from a description yielded by ProMP.serialize()
        """
        if int(serializedDict["promep_version"]) > 3:
            raise RuntimeError("Unknown (future?) serialization format version: {0}".format(serializedDict["serialization_format_version"]))

        if int(serializedDict["promep_version"]) < 3:
            raise RuntimeError("Old incompatible serialization format version: {0}".format(serializedDict["serialization_format_version"]))

        kwargs = {
            'index_sizes': serializedDict[u'index_sizes'],
            'Wmean': serializedDict[u'Wmean'],
            'Wcov': serializedDict[u'Wcov'],
            'name': serializedDict[u'name'],
            'expected_duration':  serializedDict[u'expected_duration'],         
        }
        return cls(**kwargs)
        

    def saveToFile(self, forceName=None, path='./', withTimeStamp=False):
        """
        save the (current) ProMeP data to the given file

        """
        d  = self.serialize()
        if forceName is not None:
            d[u'name']=forceName
        
        if withTimeStamp:
            filename = '{0}_{1}.promep.h5'.format(_time.strftime('%Y%m%d%H%M%S'), d[u'name'])
        else:
            filename = '{0}.promep.h5'.format(d[u'name']) 
        filepath= _os.path.join(path, filename)
        _h5.write(d, filename=filepath, store_python_metadata=True)
        return filepath


    @classmethod
    def makeFromFile(cls, filename, h5path='/'):
        """
        Create a ProMeP from a description saved in a hdf5 file using saveToFile()
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
        return cls.makeFromDict(d)
    
    



    def sample(self):
        """
        return a parameter sample from the ProMP's distribution
meansMatrix
        returns a (supports x dofs_w) matrix of actual values for each support and each dof
        """
        self._ntm.tensorDataAsFlattened['Wsample'][:,0] = _np.random.multivariate_normal(self._ntm.tensorDataAsFlattened['Wmean'][:,0], self._ntm.tensorDataAsFlattened['Wcov'])
        return self._ntm.tensorData['Wsample']


    def getDistribution(self, generalized_phase=None, currentDistribution=None):
        """
        return the distribution of the (possibly multidimensional) state at the given phase
           for now, it returns the parameters (means, derivatives) of a univariate gaussian

            generalized_phase: at which phase and its derivatives the distribution should be computed
                        
            currentDistribution: not used by ProMP

            returns a MechanicalStateDistribution object          

        """
        self._ntm.setTensor('phase', generalized_phase)
        
        self._updateP()
        
        #call the code that computes the interpolating map PHI:
        self._ntm.setTensor('PHI', self.phi_computer.getPhi(self._ntm.tensorData['phase'][0]))
        
        #call the code that computes the linear(-ized) task-space map T:
        self.T_computer.update()
                
        #Now to the computation of the actual ProMeP equation:
        self._ntm.update('P:PHI', 'PSI', 'T:Xref', 'O', 'PSI:Wmean','Ymean', 'PSI:Wcov', 'Ycov')

        return MechanicalStateDistribution(self._ntm.tensorData['Ymean'], self._ntm.tensorData['Ycov'])

    def _updateP(self):
        """
        update the tensor mapping from phase-derivatives to time-derivatives
        Basically, it implements Faa di Bruno's algorithm
        """
        P = self._ntm.tensorData['P']
        phase = self._ntm.tensorData['phase']
        
        if self._ntm.indexSizes['gtilde'] > 4 or self._ntm.indexSizes['g'] > 4:
            raise NotImplementedError()
        
        #compute the scaling factors according to Faa di Brunos formula:
        faadibruno = _np.zeros(((4,4)))
        faadibruno[0,0] = 1.0
        if self._ntm.indexSizes['g'] > 1:
            faadibruno[1,1] = phase[1]
            faadibruno[2,2] = phase[1]**2
            faadibruno[3,3] = phase[1]**3
        if self._ntm.indexSizes['g'] > 2:
            faadibruno[2,1] = phase[2]
            faadibruno[3,2] = 3*phase[1]*phase[2]
        if self._ntm.indexSizes['g'] > 3:
            faadibruno[3,1] = phase[3]
        
        #copy them into P, but shifted for/by gtilde
        P = self._ntm.tensorData['P']
        for i in range(self._ntm.indexSizes['gtilde']):
            j = self._ntm.indexSizes['g'] - i
            if j > 0:
                P[i:,:,i] = faadibruno[:j,:self._ntm.indexSizes['gphi']]

        
        
        
        
        

    def sampleTrajectory(self, generalized_phases, W=None):
        """
        return a function that represents a single trajectory, sampled from the promp's distribution

        The returned function maps generalized phase (phase, phase velocity,..) to generalized state (position, velocity) x dofs
        
        generalized_phases: array of shape ( n, |gphi| )

        W: provide a sample to create the trajectory from
                   If None, then a sample is drawn from the internal distribution
        """
        generalized_phases  = _np.asarray(generalized_phases)
        num = generalized_phases.shape[0]
        if generalized_phases.shape[1] != self.indexSizes['gphi']:
            raise ValueError()

        if W is None:
            W = self.sample()
            
        #compute the trajectory:    
        Wmean_saved = _np.array(self._ntm.tensorData['Wmean'], copy=True)
        self._ntm.setTensor('Wmean', W)
        points = _np.zeros( (num, self._ntm.indexSizes['r'],self._ntm.indexSizes['g'],self._ntm.indexSizes['d']))        
        for i in range(num):
            self._ntm.setTensor('PHI', self.phi_computer.getPhi(generalized_phases[i,0]))
            self._ntm.setTensor('P'  , self.getP(self._ntm.indexSizes['gphi'], generalized_phases[i,:]))
            self._ntm.update()
            points[i,:,:,:] = self._ntm.tensorData['Ymean']
        self._ntm.setTensor('Wmean', Wmean_saved) #restore abused Wmean

        return points



    def plot(self, dofs='all',
                   num=100,
                   linewidth=0.5,
                   withSampledTrajectories=100,
                   withConfidenceInterval=True,
                   withGainsPlots=True,
                   plotRanges = { 'torque': [-1,1], 'position': [0.0,1.0], 'velocity': [-1.0,1.0], 'gains': [-10,100.0],},
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

#        iMotion = self._ntm.indexValues['r'].index('motion')
#        iEffort = self._ntm.indexValues['r'].index('effort')

        if dofs=='all':
            dofs=self._ntm.indexValues['d']

        subplotfigsize=2.0
        plotrownames2indexvalues= OrderedDict({
            'torque': [1,2],
            'position': [0,0],
            'velocity': [0,1],
            'gains': None,
        })

        if withGainsPlots:
            plotrownames2indexvalues['gains'] = None
        plotrows = len(plotrownames2indexvalues)

        #make an array with plot limits:
        limits=_np.ones((plotrows, 2))
        for i, plotrowname in enumerate(plotrownames2indexvalues):
            limits[i,0] = plotRanges[plotrowname][0]
            limits[i,1] = plotRanges[plotrowname][1]
        
            
        fig, axesArray = _plt.subplots(plotrows,len(dofs), squeeze=False, figsize=(max(len(dofs), plotrows)*subplotfigsize, plotrows*subplotfigsize), sharex='all', sharey='row')
            
        #gather the data:
        data_mean  = _np.empty((num,plotrows, self._ntm.indexSizes['d']))
        data_sigma = _np.empty((num,plotrows, self._ntm.indexSizes['d']))
        data_gains = _np.empty((num,self._ntm.indexSizes['d'],2, self._ntm.indexSizes['d']))
            
        for i,phase in enumerate(c):
            phaseVelocity = dcdts[i]
            dist =  self.getDistribution(phase, phaseVelocity=phaseVelocity)
            for rowname in plotrownames2indexvalues:
                indices=plotrownames2indexvalues[rowname]
                if rowname == 'gains':
                    if withGainsPlots:
                        data_gains[i,:,:,:]  = _controller.extractPDGains(covariances)
                else:                
                    data_mean[i,:,:] = dist.means[i,j,:]
                    data_sigma[i,:,:] = _np.sqrt( dist.variancesView[i,j,:] )

        #draw the confidence intervals and means/variance indicators for the supports
        for i, dof in enumerate(dofs):
            #plot the zero-variance trajectory + 95% confidence interval
            if withConfidenceInterval:
                for m in range(data_mean.shape[1]):
                    meanvalues = data_mean[:,m,dof]
                    sigmavalues = data_sigma[:,m,dof]
                    axesArray[m,i].fill_between(c,meanvalues-1.96*sigmavalues, meanvalues+1.96*sigmavalues, label="95%",  color=confidenceColor)
                    axesArray[m,i].plot(c,meanvalues, label="mean",  color=meansColor)


            if withConfidenceInterval:
                for m in range(data_mean.shape[1]):
                    for row_idx, rowname in enumerate(plotrownames2indexvalues):
                        if name is 'gains':
                            continue
                        ymin = _np.min(data_mean[:,row_idx]-1.96*data_sigma[:,row_idx])
                        ymax = _np.max(data_mean[:,row_idx]+1.96*data_sigma[:,row_idx]) 
                        if not _np.isnan(ymin):
                            limits[row_idx,0] = min(ymin, limits[row_idx,0])
                        if not _np.isnan(ymax):
                            limits[row_idx,1] = max(ymax, limits[row_idx,0])

            if withGainsPlots:
                i_gain = plotrownames2indexvalues['gains']
                limits[i_gain,0] = min(limits[i_gain,0], _np.min(data_gains))
                limits[i_gain,1] = max(limits[i_gain,1], _np.max(data_gains))
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

        generalized_phases = _np.zeros((num, self._ntm.indexSizes['gphi']))
        generalized_phases[:,0] = c
        generalized_phases[:,1] = dcdts
        for j in range(withSampledTrajectories):
            yvalues = self.sampleTrajectory(generalized_phases)
            #update the desired plotting limits:
            for m, rowname in enumerate(plotrownames2indexvalues):
                r_idx, g_idx = plotrownames2indexvalues[rowname]
                ymin = _np.min(yvalues[:,r_idx,g_idx,:])
                ymax = _np.max(yvalues[:,r_idx,g_idx,:])
                limits[m][0] = min(limits[m][0],ymin)
                limits[m][1] = max(limits[m][1],ymax)
                for i, dof in enumerate(dofs):
                    axesArray[ m, i].plot(c, yvalues[:,r_idx,g_idx,dof], alpha=alpha, linewidth=linewidth )

        if 'observedTrajectories' in self.__dict__:
            for traj in self.observedTrajectories:
                phase, duration, posvalues, velvalues, tauvalues = traj

                #compute values for velocity plots:
                if velvalues is None: #compute velocities from the positions if they are not specified
                    d_posvalues = (posvalues[:,1:] - posvalues[:,:-1]) / (phase[1:] - phase[:-1]) * dcdt
                else:
                    d_posvalues = (velvalues[:,1:]+velvalues[:,1:])* 0.5* dcdt
                d_phasevalues = 0.5* (phase[1:]+phase[:-1])

                data = {
                    'position': [phases, posvalues],
                    'velocity': [d_phases, d_phasevalues],
                    'torque':   [phases, tauvalues]
                }
                

                if scaleToExpectedDuration:
                    times = phase * self.expectedDuration
                    dcdt = 1.0 / self.expectedDuration
                else:
                    times = phase 
                    dcdt = 1.0

                for m, rowname in enumerate(plotrownames2indexvalues):
                    if rowname == 'torque' and tauvalues == None :
                        continue
                    r_idx, g_idx = plotrownames2indexvalues[rowname]
                    x,y = data[rowname]
                    ymin = _np.min(y[:,r_idx,g_idx,:])
                    ymax = _np.max(y[:,r_idx,g_idx,:])
                    limits[m][0] = min(limits[m][0],ymin)
                    limits[m][1] = max(limits[m][1],ymax)

                    for i, dof in enumerate(dofs):
                        axesArray[ m, i].plot(x, y[dof,:], alpha=alpha, linewidth=linewidth, color=observedColor )

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
        cov_flat = self._ntm.tensorDataAsFlattened['Wcov']
        
        cov = self._ntm.tensorData['Wcov'].transpose(axes=[0,1,2,3,4,5,6,7])
        variance_view = _np.einsum('ijklijkl->ijkl', self._ntm.tensorData['Wcov'])
        
        sigmas = _np.sqrt(variance_view)

        if normalized:  #do complete normalization - yields correlations   
            cov = cov * sigmas[:,:,:,:, None,None,None,None] * sigmas[None,None,None,None, :,:,:,:] 
        else:           #only scale for each domain 
            sigmamax = _np.zeros((2))
            sigmamax[0] = _np.max(sigmas[0,:,:,:])
            sigmamax[1] = _np.max(sigmas[1,:,:,:])
            cov = cov * sigmamax[:,None,None,None, None,None,None,None] * sigmamax[None,None,None,None, :,None,None,None]

        cov_reordered = _np.transpose(cov, (1,0,2,3, 5,4,6,7))
        image =_np.reshape(cov_reordered, self.tensorShapeFlattened['Wcov'])
        gridvectorX = _np.arange(0, image.shape[0], 1)
        gridvectorY = _np.arange(image.shape[1], 0, -1)

        fig = _plt.figure(figsize=(3.4,3.4))
        _plt.pcolor(gridvectorX, gridvectorY, cmap=_cmapCorrelations, vmin=-1, vmax=1)
        
        _plt.axis([0, image.shape[0], 0, image.shape[1]])
        _plt.gca().set_aspect('equal', 'box')

        for i in range(self._ntm.indexSizes['rtilde']):
            for j in range(self._ntm.indexSizes['stilde']):
                for k in range(self._ntm.indexSizes['dtilde']):
                    if j == 0 and i==0:
                        continue
                    if j== 0:
                        linewidth=1.0
                    else:
                        linewidth=0.1
                    _plt.axhline(i *n_supports*n_dofs + j * n_dofs , color='k', linewidth=linewidth)
                    _plt.axvline(i *n_supports*n_dofs + j * n_dofs, color='k', linewidth=linewidth)

        elementnames = list(range(self._ntm.indexSizes['stilde'],0,-1)) + list(range(self._ntm.indexSizes['stilde'],0,-1))
        xticks = range(image.shape[0] - self._ntm.indexSizes['dtilde']//2,0, -self._ntm.indexSizes['dtilde'])
        _plt.xticks(xticks, elementnames)
        yticks = range( self._ntm.indexSizes['dtilde']//2, image.shape[0], self._ntm.indexSizes['dtilde'])
        _plt.yticks(yticks, elementnames)
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





