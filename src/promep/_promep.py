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
import pprint as  _pprint
import multiprocessing as _multiprocessing
import random as _random

from promep import *

from . import _namedtensors, _interpolationkernels, _mechanicalstate, _taskspaces, _kumaraswamy

from  scipy.special import betainc as _betainc
from sklearn import covariance as _sklearncovariance




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


class ProMeP(object):
    """
    This class implements the Probabilistic Mechanical Primitives as described in the paper:

    [TODO]

    """



    def __init__(self, 
                 index_sizes={}, 
                 Wmean=None, 
                 Wcov=None, 
                 name="unnamed", 
                 expected_duration=1.0, 
                 PHI_computer_cls=_interpolationkernels.InterpolationGaussian,   #class to compute the PHI tensor with 
                 T_computer_cls=_taskspaces.JointSpaceToJointSpaceTransform,                 #class to compute the T tensor with
        ):
        """
            index_sizes: dictionary containing the sizes of all internal indices:
                indices:
                    'r': number of realms (usually 2) 
                    'rtilde': number of realms in parameter space (the same as 'r')                     
                    'g': number of time derivatives to provide (usually 2, zeroth and first derivative)
                    'gtilde': number of phase derivatives to represent in parameter space (usually equal or less than gphi)
                    'gphi': number of derivatives of the phase variable to consider when mapping to time-based trajectories (usually equal to g)
                    'stilde': number of discrete parameters used for interpolating each trajectory
                    'd': number of DoFs being controlled (usually joint space dof)
                    'dtilde': DOFs in parameter space (usually the same as d)
            
            For convenience, you can also specify the following meta-parameters instead of individual index sizes:
                'dofs'        (sets 'd' and 'dtilde' )
                'realms'      (sets 'r' and 'rtilde' )
                'derivatives' (sets 'g', 'gphi' and 'gtilde' )
                'interpolation_parameters' (sets 'stilde' ) 
            

            Alternatively, you can also specify every index size directly, if needed:

            The default values are:
            index_sizes = {
                'r':  2,  
                'rtilde':  2,  
                'g':  3,
                'gtilde': 3,
                'gphi':  3,
                'stilde':  5,
                'd':  4,
                'dtilde':  4,
            }
            
           
            The following size relationships are not wrong, but useless as they don't change anything and only slow down computation:
                gphi > g 
                gtilde > g                
        
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
            
            expected_duration: time the promp is expected to execute in. Use it for scaling plots and general debugging of learning
                        
        """
        self.name = name
        self.tns = _namedtensors.TensorNameSpace() #band-aid to have proper multilinear algebra semantics for numpy etc.

        #register all index names being used and all index sizes:
        self.index_sizes = {
            'r':  2,
            'rtilde':  2,
            'g':  3,
            'gtilde':  3,
            'gphi':  3,
            'd':  4,
            'dtilde':  4,
            'stilde':  5,
        }
        
        #for convenience: accept standard meta-parameters with sensible names:
        index_sizes = dict(index_sizes) 
        if 'dofs' in index_sizes:
            self.index_sizes['d']      = index_sizes['dofs']
            self.index_sizes['dtilde'] = index_sizes['dofs']
            del index_sizes['dofs']
        if 'interpolation_parameters' in index_sizes:
            self.index_sizes['stilde'] = index_sizes['interpolation_parameters']
            del index_sizes['interpolation_parameters']
        if 'realms' in index_sizes:
            self.index_sizes['r']      = index_sizes['realms']
            self.index_sizes['rtilde'] = index_sizes['realms']
            del index_sizes['realms']       
        if 'derivatives' in index_sizes:
            self.index_sizes['g']      = index_sizes['derivatives']
            self.index_sizes['gtilde'] = index_sizes['derivatives']
            self.index_sizes['gphi']   = index_sizes['derivatives']
            del index_sizes['derivatives']       
        self.index_sizes.update(index_sizes) #add/overwrite any remaining indices that have been specified explicitly
        
        for name in self.index_sizes:
            if name == 'r' or name == 'rtilde':
                  self.tns.registerIndex(name, self.index_sizes[name], values=['motion', 'effort'][:self.index_sizes[name]])
            else:
                  self.tns.registerIndex(name, self.index_sizes[name], values = list(range(self.index_sizes[name])))

        #human-readable names for (realm, derivative) index combinations:
        motion_names = ('position', 'velocity', 'acceleration', 'jerk')
        effort_names = ('int_int_torque', 'impulse', 'torque', 'torque_rate')
        if self.index_sizes['g'] < 3:   #if less than 3 derivatives are computed, make sure that torque stays represented:
            self.gtilde_effort_shiftvalue = 3-self.index_sizes['g'] #remember by how much gtilde for effort is shifted
            effort_names = effort_names[self.gtilde_effort_shiftvalue:]
        else:
            self.gtilde_effort_shiftvalue = 0

        #set up translation dict from human-readable names to indices used within the promep data structures:
        d={}
        self._gain_names = set()
        for g_idx in range(self.index_sizes['g']):
            d[motion_names[g_idx]] = (0,g_idx)
            d[effort_names[g_idx]] = (1,g_idx)
        if 'torque' in d and 'position' in d:
            d['kp'] = (d['torque'][1], d['position'][1])  #gains get the derivative indices of effort&motion
            self._gain_names.add('kp')
        if 'torque' in d and 'velocity' in d:
            d['kv'] = (d['torque'][1], d['velocity'][1])
            self._gain_names.add('kv')
        self.readable_names_to_realm_derivative_indices = d

        _pprint.pprint(self.readable_names_to_realm_derivative_indices)


        self.PHI_computer = PHI_computer_cls(self.tns)   
                  
        #register all tensors not being computed by operators:
        #i.e. input tensors:
        self.tns.registerTensor('phase', (('g',),()) )  #generalized phase vector        
        self.tns.registerTensor('Wmean', (('rtilde','gtilde','stilde','dtilde'),()) )
        self.tns.registerTensor('Wcov', (( 'rtilde','gtilde','stilde','dtilde'),( 'rtilde_','gtilde_','stilde_','dtilde_')) )
        self.tns.registerTensor('PHI', (('gphi',),('stilde',)), external_array = self.PHI_computer.get_PHI_view() )
        self.tns.registerTensor('P', (('g',),('gphi','gtilde')) )
        self.tns.registerTensor('Yref', (('r','d','g',),()) )    
        self.tns.registerTensor('Xref', (('rtilde','dtilde','g',),()) )   
        self.tns.registerTensor('T', (('r','d'),('rtilde', 'dtilde')) )       #link to T_computer's Xref tensor
        #register all operations being used on tensors:

        #set up hooks to compute tensors PHI and T:
        self.T_computer = T_computer_cls(self.tns) #should implement a get_T_view(), get_Xref_view() and get_Yref_view() method
        
        #path to compute PSI:
        self.tns.registerContraction('P', 'PHI')
        self.tns.registerContraction('T', 'P:PHI', result_name='PSI')
        self.tns.registerTranspose('PSI')
        
        #compute any offsets required by the linearization of the task space map into T
        self.tns.registerContraction('T', 'Xref')
        self.tns.registerSubtraction('Yref', 'T:Xref', result_name='O')
        
        #project the mean of weights to mean of trajectory:
        self.tns.registerContraction('PSI', 'Wmean')
        
        self.tns.registerAddition('PSI:Wmean', 'O', result_name='Ymean')
        
        #project the covariances of weights to covariances of trajectory:
        self.tns.registerContraction('PSI', 'Wcov')
        self.tns.registerContraction('PSI:Wcov', '(PSI)^T', result_name='Ycov')

       
        self.phaseAssociable = True #indicate that this motion generator is parameterized by phase
        self.timeAssociable = False #indicate that this motion generator is not parameterizable by time
        self.tolerance=1e-7
        
        self.expected_duration = expected_duration

        self.tns.setTensor('Wmean', Wmean)
        self.tns.setTensor('Wcov', Wcov)
        
        #temporary/scratch tensors:
        self.tns.registerTensor('Wsample', (('r','gtilde','stilde','dtilde'),()) )

        #some initial plot range values so we can plot something at all
        self.plot_range_guesses = { 'torque': [-20,20], 'position': [-1.5,1.5], 'velocity': [-2.0,2.0], 'gains': [-10,100.0],}
        

    def __repr__(self):
        strings = ["Indices:"]
        for key in self.tns.indexSizes:
            strings.append("{}: {}".format(key,self.tns.indexSizes[key]))
        strings.append("")
        strings.append("Tensors:")
        for key in self.tns.tensorIndices:
            strings.append("{}     {}".format(key,self.tns.tensorIndices[key]))
        strings.append("")
        return '\n'.join(strings)

    def serialize(self):
        """

        returns a python dictionary that contains all internal data
        necessary to recreate this ProMeP

        """
        serializedDict = {}
        serializedDict[u'class'] = type(self).__name__
        serializedDict[u'name'] = self.name
        serializedDict[u'serialization_version'] = "3"
        serializedDict[u'index_sizes'] = {key: self.tns.indexSizes[key] for key in self.tns.indexSizes if not key.endswith('_')}
        serializedDict[u'Wmean'] = self.tns.tensorData['Wmean']
        serializedDict[u'Wcov'] = self.tns.tensorData['Wcov']
        serializedDict[u'expected_duration'] = self.expected_duration
        return serializedDict


    @classmethod
    def makeFromDict(cls, serializedDict):
        """
        Create a ProMeP from a description yielded by ProMP.serialize()
        """
        if int(serializedDict["serialization_version"]) > 3:
            raise RuntimeError("Unknown (future?) serialization format version: {0}".format(serializedDict["serialization_version"]))

        if int(serializedDict["serialization_version"]) < 3:
            raise RuntimeError("Old incompatible serialization format version: {0}".format(serializedDict["serialization_version"]))

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
    

    def setParameterDistribution(self, Wmean, Wcov):
        self.tns.setTensor('Wmean', Wmean)
        self.tns.setTensor('Wcov', Wcov)
    

    def sample(self):
        """
        return a parameter sample from the ProMP's distribution
meansMatrix
        returns a (supports x dofs_w) matrix of actual values for each support and each dof
        """
        self.tns.tensorDataAsFlattened['Wsample'][:,0] = _np.random.multivariate_normal(self.tns.tensorDataAsFlattened['Wmean'][:,0], self.tns.tensorDataAsFlattened['Wcov'])
        return self.tns.tensorData['Wsample']



    def getDistribution(self, generalized_phase=None, currentDistribution=None):
        """
        return the distribution of the (possibly multidimensional) state at the given phase
           for now, it returns the parameters (means, derivatives) of a univariate gaussian

            generalized_phase: at which phase and its derivatives the distribution should be computed
                        
            currentDistribution: not used by ProMP

            returns a MechanicalStateDistribution object          

        """
        self.tns.setTensor('phase', generalized_phase)
        if currentDistribution is None:
            self.tns.setTensor('Yref', 0.0)  #for joint-space as task space, we do not need to care
        else:
            self.tns.setTensor('Yref', currentDistribution.means) #but usually, set a linearization reference point for the T_computer
        self._updateP() #call the code that computes map between phase and time derivatives. implemented directly in this class
        self.PHI_computer.update() #call the hook that computes the interpolating map PHI:
        self.T_computer.update() #call the hook that computes the linear(-ized) task-space map T:
        #Now compute the actual ProMeP equation:
        self.tns.update('P:PHI', 'PSI', 'T:Xref', 'O', 'PSI:Wmean', 'Ymean', 'PSI:Wcov', 'Ycov')

        return _mechanicalstate.MechanicalStateDistribution(self.tns.tensorData['Ymean'], self.tns.tensorData['Ycov'])



    def _updateP(self):
        """
        Hook you can overload to change how P is computed 
        
        update the tensor mapping from phase-derivatives to time-derivatives
        Basically, it implements Faa di Bruno's algorithm
        """
        if self.tns.indexSizes['gtilde'] > 4 or self.tns.indexSizes['g'] > 4 or self.tns.indexSizes['gphi'] > 4:
            raise NotImplementedError()
            
        P = self.tns.tensorData['P']
        phase_fdb=_np.zeros((4))
        phase_fdb[:self.tns.indexSizes['gphi']] = self.tns.tensorData['phase']
        
        
        #compute the scaling factors according to Faa di Brunos formula
        #0th, 1st and 2nd derivative:
        faadibruno = _np.zeros(((4,4)))
        faadibruno[0,0] = 1.0        
        faadibruno[1,1] = phase_fdb[1]
        faadibruno[2,2] = phase_fdb[1]**2
        faadibruno[2,1] = phase_fdb[2]

        if self.tns.indexSizes['g'] > 3:  #is probably not used
            faadibruno[3,3] = phase_fdb[1]**3
            faadibruno[3,2] = 3*phase_fdb[1]*phase_fdb[2]
            faadibruno[3,1] = phase_fdb[3]
        
        #copy them into P, but shifted for/by gtilde
        P = self.tns.tensorData['P']  
        for gtilde in range(self.tns.indexSizes['gtilde']):
            g_start = gtilde
            fdb_g_end = self.tns.indexSizes['g'] - g_start
            if fdb_g_end > 0:
                #index order: g, gphi, gtilde
                P[g_start:,:,gtilde] = faadibruno[:fdb_g_end,:self.tns.indexSizes['gphi']]


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
        if generalized_phases.shape[1] != self.tns.indexSizes['gphi']:
            raise ValueError()

        if W is None:
            W = self.sample()
            
        #compute the trajectory:    
        Wmean_saved = _np.array(self.tns.tensorData['Wmean'], copy=True)
        self.tns.setTensor('Wmean', W)
        points_list = []   
        for i in range(num):
            points_list.append(self.getDistribution(generalized_phases[i]).means)
        self.tns.setTensor('Wmean', Wmean_saved) #restore abused Wmean
        return _np.array(points_list)



    def plot(self, dofs='all',
                   whatToPlot=['position', 'velocity', 'kp', 'kv', 'impulse', 'torque'],
                   num=101,
                   linewidth=0.5,
                   addExampleTrajectories=10,
                   withConfidenceInterval=True,
                   plotRanges = None,
                   exampleTrajectoryStyleCycler=_plt.cycler('color', ['#6666FF']),
                   useTime=True,
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

        phases = _np.zeros((num, self.tns.indexSizes['gphi']))
        
        if useTime: 
            times = _np.linspace(0, self.expected_duration, num)     
            phases[:,0] = _kumaraswamy.cdf(2, 2, _np.linspace(0,1.0,num))
            plot_x = times
        else:
            phases[:,0] = _np.linspace(0.0, 1.0, num)
            plot_x = phases[:,0]
            
        if self.tns.indexSizes['gphi'] > 1:
            if useTime:
                phases[:,1] = 1.0/self.expected_duration
            else:
                phases[:,1] = 1.0

        if useTime:
            units={
                'impulse': '[Nms]',
                'torque': '[Nm]',
                'position': '[rad]',
                'velocity': '[rad/s]',
                'acceleration': '[rad/s^2]',
                'kp': '[Nm/rad]',
                'kv': '[Nm/rad/s]',
            }
        else:
            units={
                'impulse': '[Nms/1]',
                'torque': '[Nm]',
                'position': '[rad]',
                'velocity': '[rad/1]',
                'acceleration': '[rad/1^2]',
                'kp': '[Nm/rad]',
                'kv': '[Nm/rad/1]',
            }
        
        whatToPlot = [name for name in whatToPlot if name in self.readable_names_to_realm_derivative_indices ]

        if plotRanges is None:
            plotRanges = self.plot_range_guesses   
        #e.g. { 'torque': [-20,20], 'position': [-1.5,1.5], 'velocity': [-2.0,2.0], 'gains': [-10,100.0],}

        if dofs=='all' or dofs == None:
            if self.tns.indexValues['d'] != None:
                dofs_to_plot=self.tns.indexValues['d']
            else:
                dofs_to_plot=list(range(self.tns.indexSizes['d']))
        else:
            dofs_to_plot = dofs
        subplotfigsize=2.0

        plotrows = len(whatToPlot)
        #make an array with plot limits:
        limits=_np.zeros((plotrows, 2))
        for row_idx, plotrowname in enumerate(whatToPlot):
            if plotRanges is not None and plotrowname in plotRanges:
                limits[row_idx,0] = plotRanges[plotrowname][0]
                limits[row_idx,1] = plotRanges[plotrowname][1]
            else:
                limits[row_idx,0] = -0.1
                limits[row_idx,1] = 0.1
            
        #gather the data:
        data_mean  = _np.zeros((num,plotrows, self.tns.indexSizes['d']))
        data_sigma = _np.zeros((num,plotrows, self.tns.indexSizes['d']))
        data_gains = {
            'kp': _np.zeros((num,self.tns.indexSizes['d'], self.tns.indexSizes['d'])),
            'kv': _np.zeros((num,self.tns.indexSizes['d'], self.tns.indexSizes['d'])),
        }
        generalized_phase =_np.zeros((self.tns.indexSizes['g']))            
        for i,phase in enumerate(phases):
            dist =  self.getDistribution(phase)
            for row_idx, rowname in enumerate(whatToPlot):
                if rowname in data_gains:
                    gains  = dist.extractPDGains()
                    g_idx, g2_idx = self.readable_names_to_realm_derivative_indices[rowname]
                    data_gains[rowname][i,:,:] = gains[:,g_idx,:,g2_idx]
                else:                
                    r_idx, g_idx = self.readable_names_to_realm_derivative_indices[rowname]                
                    data_mean[i,row_idx,:] = dist.means[r_idx,:,g_idx] 
                    data_sigma[i,row_idx,:] = _np.sqrt( dist.variancesView[r_idx,:,g_idx] )

        fig, axesArray = _plt.subplots(plotrows,len(dofs_to_plot), squeeze=False, figsize=(max(len(dofs_to_plot), plotrows)*subplotfigsize, plotrows*subplotfigsize), sharex='all', sharey='row')
            
        #draw confidence intervals and means/variance indicators for the supports
        #plot the zero-variance trajectory + 95% confidence interval        
        for row_idx, row_name in enumerate(whatToPlot):
            for col_idx, dof in enumerate(dofs_to_plot):
                if rowname in data_gains:
                    limits[row_idx,0] = min(limits[row_idx,0], _np.min(data_gains[rowname]))
                    limits[row_idx,1] = max(limits[row_idx,1], _np.max(data_gains[rowname]))
                    axesArray[row_idx,col_idx].axhline(0.0, label=None,  color=(0.4,0.4,0.4), linestyle=':')
                    for col_idx2, dof2 in enumerate(dofs_to_plot):
                        if dof != dof2:
                            axesArray[row_idx,col_idx].plot(plot_x,data_gains[rowname][:,dof,dof2], label=None,  color=kpCrossColor, linestyle=':')
                    #plot the joint-local gains prominently, and on top (i.e. last)
                    axesArray[row_idx,col_idx].plot(plot_x,data_gains[rowname][:,dof,dof], label=rowname,  color=kpColor)
                    
                else:
                    meanvalues = data_mean[:,row_idx,dof]
                    sigmavalues = data_sigma[:,row_idx,dof]
                    upper_boundary = meanvalues+1.96*sigmavalues
                    lower_boundary = meanvalues-1.96*sigmavalues
                    if withConfidenceInterval:
                        axesArray[row_idx,col_idx].fill_between(plot_x,lower_boundary, upper_boundary, label="95%",  color=confidenceColor)
                    axesArray[row_idx,col_idx].plot(plot_x,meanvalues, label="mean",  color=meansColor)

                    #update limits:
                    ymin = _np.min(lower_boundary)
                    ymax = _np.max(upper_boundary) 
                    if not _np.isnan(ymin):
                        limits[row_idx,0] = min(ymin, limits[row_idx,0])
                    if not _np.isnan(ymax):
                        limits[row_idx,1] = max(ymax, limits[row_idx,0])

        #sample the distribution to plot actual trajectories, times the number given by "addExampleTrajectories":
        if addExampleTrajectories is None:
            addExampleTrajectories = 0
        alpha = _np.sqrt(2.0 / (1+addExampleTrajectories))
        for ax in axesArray.flatten():
            ax.set_prop_cycle(exampleTrajectoryStyleCycler)

        for j in range(addExampleTrajectories):
            yvalues = self.sampleTrajectory(phases)
            #update the desired plotting limits:
            for row_idx, rowname in enumerate(whatToPlot):
                if rowname in self._gain_names:
                    continue
                r_idx, g_idx = self.readable_names_to_realm_derivative_indices[rowname]
                ymin = _np.min(yvalues[:,r_idx,:,g_idx])
                ymax = _np.max(yvalues[:,r_idx,:,g_idx])
                limits[row_idx,0] = min(limits[row_idx,0],ymin)
                limits[row_idx,1] = max(limits[row_idx,1],ymax)
                for i, dof in enumerate(dofs_to_plot):
                    axesArray[row_idx,col_idx].plot(plot_x, yvalues[:,r_idx,dof,g_idx], alpha=alpha, linewidth=linewidth )
        largest_observed_time = plot_x[-1]
        if 'observedTrajectories' in self.__dict__:
            for observation_idx, (times, phases_observation, values, Xrefs, Yrefs, Ts) in enumerate(self.observedTrajectories):        
                for row_idx, rowname in enumerate(whatToPlot):
                    if rowname in self._gain_names:
                        continue
                    r_idx, g_idx = self.readable_names_to_realm_derivative_indices[rowname]
                    if useTime:
                        y = values[:,r_idx,:,g_idx]
                        for col_idx, dof in enumerate(dofs_to_plot):
                            axesArray[ row_idx, col_idx].plot(times, y[:,dof], alpha=alpha, linewidth=linewidth, color=observedColor )
                    else:
                        if g_idx == 1:
                            y = values[:,r_idx,:,g_idx] #/ phases_observation[:,1]
                        else:
                            y = values[:,r_idx,:,g_idx] 
                        for col_idx, dof in enumerate(dofs_to_plot):
                            axesArray[ row_idx, col_idx].plot(phases_observation[:,0], y[:,dof], alpha=alpha, linewidth=linewidth, color=observedColor )

            largest_observed_time = max(largest_observed_time,  _np.max([ tup[0] for tup in self.observedTrajectories]))

        limit_padding=0.05
        for col_idx, dof in enumerate(dofs_to_plot):
            for row_idx, rowname in enumerate(whatToPlot):
                axes = axesArray[row_idx,col_idx]  
                axes.set_title(r"{0} {1}".format(rowname, dof))
                if row_idx == plotrows-1: #last row?
                    if useTime:
                        axes.set_xlim(plot_x[0],largest_observed_time)
                        axes.set_xticks( [ 0.0, 0.5*plot_x[-1], plot_x[-1] ] )
                        axes.set_xticklabels(['0', 'time [s]', '{0:0.1f}'.format(plot_x[-1])])
                    else:
                        axes.set_xlim(0,1.0)
                        axes.set_xticks([0.0,0.5, 1.0])
                        axes.set_xticklabels(['0', 'phase', '1'])
                else:
                    axes.get_xaxis().set_visible(False)
                if col_idx == 0: #first column?
                    axes.set_ylabel(units[rowname])
                else:
                    axes.get_yaxis().set_visible(False)
                delta = (limits[row_idx,1] - limits[row_idx,0])* limit_padding
                #axes.set_ylim(limits[row_idx,0]-delta,limits[row_idx,1]+delta)
        _plt.tight_layout()
        return fig


    def plotCovarianceTensor(self, normalized=True):
        """
        plot a correlation tensor of shape (supports x dofs_w, supports x dofs_w,)
        """
        cov_flat = self.tns.tensorDataAsFlattened['Wcov']
        
        cov = self.tns.tensorData['Wcov'].transpose(axes=[0,1,2,3,4,5,6,7])
        variance_view = _np.einsum('ijklijkl->ijkl', self.tns.tensorData['Wcov'])
        
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

        for i in range(self.tns.indexSizes['rtilde']):
            for j in range(self.tns.indexSizes['stilde']):
                for k in range(self.tns.indexSizes['dtilde']):
                    if j == 0 and i==0:
                        continue
                    if j== 0:
                        linewidth=1.0
                    else:
                        linewidth=0.1
                    _plt.axhline(i *n_supports*n_dofs + j * n_dofs , color='k', linewidth=linewidth)
                    _plt.axvline(i *n_supports*n_dofs + j * n_dofs, color='k', linewidth=linewidth)

        elementnames = list(range(self.tns.indexSizes['stilde'],0,-1)) + list(range(self.tns.indexSizes['stilde'],0,-1))
        xticks = range(image.shape[0] - self.tns.indexSizes['dtilde']//2,0, -self.tns.indexSizes['dtilde'])
        _plt.xticks(xticks, elementnames)
        yticks = range( self.tns.indexSizes['dtilde']//2, image.shape[0], self.tns.indexSizes['dtilde'])
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



    def learnFromObservations(self, 
            observations, 
            useOnly=((0,0),(1,2)),  
            max_iterations=150, 
            minimal_relative_improvement=1e-3,
            ):
        """
        
        compute parameters from a list of observations
        
        Observations are tuples of (times, phases, values, Xrefs, Yrefs) tensors (data arrays)
        
        phases: Generalized phases of shape (n, 'd')
        means: observed values at the given phases, of shape (n, 'd', 'g')
        Xref, Yref: Task space linearization references, of shape (n, 'rtilde', 'dtilde', 'gtilde', 'stilde' ) and (n, 'r', 'd', 'g') respectively
        

        useonly: only use the combination of (realm, derivative) to learn, assume that in all other combinations data are not trustworthy
       
        stabilize: value in range 0-1, 0 does does full EM updates, nonzero values slow down convergence but improve stability
       
        Implements the expectation-maximization procedure of the paper "Using probabilistic movement primitives in robotics" by Paraschos et al.
        """
        pool = _multiprocessing.Pool(8)        
        
        # set up computation of PSIhatInv = (PSIhat)^-1 = (Yref - (Yrefhat - T:Xrefhat) )^-1

        tns_learn = _namedtensors.TensorNameSpace(self.tns)
        self.tns_learn =tns_learn
        
        #tensors to do EM on:
        tns_learn.registerTensor('Wmean', (['rtilde', 'gtilde', 'stilde', 'dtilde'], ()) )
        tns_learn.registerTensor('Wcov', (['rtilde', 'gtilde', 'stilde','dtilde'], ['rtilde_', 'gtilde_', 'stilde_','dtilde_'],) )        
                
        #compute the precision matrix in Y space for a current sample i:
        tns_learn.registerTensor('PSIi', self.tns.tensorIndices['PSI'])
        tns_learn.registerTranspose('PSIi')
        tns_learn.registerContraction('PSIi', 'Wcov')
        tns_learn.registerContraction('PSIi:Wcov', '(PSIi)^T')
        tns_learn.registerInverse('PSIi:Wcov:(PSIi)^T', result_name='Yiprecision', flip_underlines=False)
        
        #compute the expectations:
        tns_learn.registerTensor('Yi', (('r','d','g',),()) )
        tns_learn.registerContraction('PSIi', 'Wmean')
        tns_learn.registerSubtraction('Yi', 'PSIi:Wmean')
        tns_learn.registerContraction('Yiprecision', '(Yi-PSIi:Wmean)')
        tns_learn.registerContraction('(PSIi)^T', 'Yiprecision:(Yi-PSIi:Wmean)', result_name= 'Wmeanidelta', flip_underlines=True)
        tns_learn.registerAddition('Wmean', 'Wmeanidelta', result_name= 'Wmeani')
        
        #compute negative-log-likelihood increment to judge progress:
        tns_learn.registerTranspose('(Yi-PSIi:Wmean)')
        tns_learn.registerContraction('(Yi-PSIi:Wmean)', 'Yiprecision' )
        tns_learn.registerContraction('(Yi-PSIi:Wmean):Yiprecision','((Yi-PSIi:Wmean))^T', result_name='negLLidelta'  )
        
        #compute part of the maximization term for the covariance tensor:
        tns_learn.registerTranspose('PSIi:Wcov')
        tns_learn.registerContraction('(PSIi:Wcov)^T', 'Yiprecision')
        tns_learn.registerContraction('(PSIi:Wcov)^T:Yiprecision', 'PSIi:Wcov', result_name='Wcovidelta')
        tns_learn.registerSubtraction('Wcov', 'Wcovidelta', result_name= 'Wcovi')

        first_part_steps = len(tns_learn.update_order)  #when computing, stop here for averaging over all computed Wmeani

        #this is done separately, after Wmean has been updated:  
        tns_learn.registerSubtraction('Wmeani', 'Wmean')      
        tns_learn.registerTranspose('(Wmeani-Wmean)')  
        tns_learn.registerContraction('(Wmeani-Wmean)', '((Wmeani-Wmean))^T', result_name='Wcoviempirical')
    
        #_pprint.pprint(tns_learn.tensorIndices)
       
    
        iteration_count = 0 #maximum number of E-M iterations
        negLL = []
        tns_learn.setTensorToIdentity('Wcov', scale=0.1) #set the prior
        tns_learn.setTensor('Wmean', 0.0) #set the prior  
      
        #set update() to not know the second part:
        tns_learn.update_order = tns_learn.update_order[:first_part_steps]

        #preprocess observations into pairs of Yi and PSIi
        estimation_parameters = []
        for observation_idx, (times, phases, values, Xrefs, Yrefs, Ts) in enumerate(observations):
            num = phases.shape[0]
            for n in range(num):
                #compute PSIi:
                self.tns.setTensor('phase', phases[n,:], (('g'),()) )
                self.tns.setTensor('Xref', Xrefs[n,:,:,:], (('rtilde', 'dtilde', 'g'),()) ) 
                self.tns.setTensor('Yref', Yrefs[n,:,:,:], (('r', 'd', 'g'),()) )                
                self.tns.setTensor('T', Ts[n,:,:,:,:], (('r', 'd'),('rtilde', 'dtilde')) )
                self._updateP() #call the code that computes map between phase and time derivatives. implemented directly in this class
                self.PHI_computer.update() #call the hook that computes the interpolating map PHI from the phase:
                #compute PSI only:
                self.tns.update('P:PHI', 'PSI', 'O')
                ydelta = values[n,:,:,:] #- self.tns.tensorData['O']
                estimation_parameters.append( (tns_learn, tns_learn.tensorData['Wmean'], tns_learn.tensorData['Wcov'], self.tns.tensorData['PSI'].copy(), ydelta) ) #saving a view is enough

        relative_improvement = 1.0
        global _tnscopy_perprocess
        _tnscopy_perprocess.clear()
        relative_changes = 1.0
        while iteration_count < max_iterations:
            iteration_count = iteration_count+1        
            
            #if we have many data points, and changes are still big, select a subset to speed up iteration:
            if  len(estimation_parameters) > 0.2*tns_learn.tensorData['Wcov'].size and relative_changes > 0.1:
                subset = [ _random.choice(estimation_parameters) for i in range(tns_learn.tensorData['Wcov'].size//5)]
                stabilize = max(0.5, ((1.0*iteration_count) / max_iterations))
            else: 
                subset = estimation_parameters
                stabilize = max(0.05, ((1.0*iteration_count) / max_iterations)-0.5)
                
            estimates = pool.starmap(_estimation_func, subset)
            #estimates = list(_it.starmap(_estimation_func, subset)) #non-parallel version
            
            #do maximization step for means:
            Wmean_prime = (1.0-stabilize) * _np.mean([e[0] for e in estimates], axis=0) + stabilize * tns_learn.tensorData['Wmean']
            tns_learn.setTensor('Wmean', Wmean_prime )
            #do maximization step for covariances, using Wmean_prime
            negLLDeltaSum = 0.0
            sumWcov = 0.0          
            for Wmeani, Wcovi, negLLdelta in estimates:
                tns_learn.setTensor('Wmeani', Wmeani)
                tns_learn.update('(Wmeani-Wmean)','((Wmeani-Wmean))^T','Wcoviempirical')
                sumWcov = sumWcov + Wcovi + tns_learn.tensorData['Wcoviempirical']
                negLLDeltaSum += negLLdelta
            Wcov_prime = (1.0-stabilize) * sumWcov * (1.0/len(subset)) + stabilize * tns_learn.tensorData['Wcov']
            tns_learn.setTensor('Wcov', Wcov_prime, tns_learn.tensorIndices['Wcoviempirical'] )
            #re-symmetrize to make iteration numerically stable:
            tns_learn.tensorDataAsFlattened['Wcov'][:,:] = 0.5*tns_learn.tensorDataAsFlattened['Wcov'] + 0.5* tns_learn.tensorDataAsFlattened['Wcov'].T
            negLL.append(negLLDeltaSum / len(subset))
            
            #terminate early if neg-log-likelihood of observations stops increasing:
            n= 4
            if len(negLL) > n:
                relative_changes = [ abs((negLL[-i]-negLL[-i-1])/negLL[-i-1]) for i in range(1,n) ]
                eigenvalues = _np.linalg.eigvals(tns_learn.tensorDataAsFlattened['Wcov'])
                if _np.any(_np.iscomplex(eigenvalues)) or _np.any(eigenvalues < 0.0):                    
                    print("Warning: At iteration {} eigenvalues became {} ".format(iteration_count, eigenvalues))
                #print(_np.diag(tns_learn.tensorDataAsFlattened['Wcov']))
                #print(tns_learn.tensorData['Wmean'])
                print("{:.4}".format(relative_changes[0]))
                relative_changes = max(relative_changes)
                if relative_changes < minimal_relative_improvement:
                    print("stopping early at iteration {}".format(iteration_count))
                    break
                    
        self.negLL = _np.array(negLL)
        #update yourself to the learnt estimate
        self.tns.setTensor('Wmean', tns_learn.tensorData['Wmean'], tns_learn.tensorIndices['Wmean'])
        self.tns.setTensor('Wcov', tns_learn.tensorData['Wcov'], tns_learn.tensorIndices['Wcov'])
        self.expected_duration = _np.mean([times[-1] for(times, phases, values, Xrefs, Yrefs, Ts) in observations])

        self.observedTrajectories = observations #remember the data we learned from, for plotting etc.


    def learnFromObservationsUsingLinReg(self, observations, useOnly=((0,0),(1,2)) ):
        """
        
        compute parameters from a list of observations
        
        Observations are tuples of (phases, values, Xrefs, Yrefs) tensors (data arrays)
        
        phases: Generalized phases of shape (n, 'd')
        means: observed values at the given phases, of shape (n, 'd', 'g')
        Xref, Yref: Task space linearization references, of shape (n, 'rtilde', 'dtilde', 'gtilde', 'stilde' ) and (n, 'r', 'd', 'g') respectively
        

        useonly: only use the combination of (realm, derivative) to learn, assume that in all other combinations data are not trustworthy
        """
        
        # set up computation of PSIhatInv = (PSIhat)^-1 = (Yref - (Yrefhat - T:Xrefhat) )^-1
        tns_learn = _namedtensors.TensorNameSpace(self.tns)
        tns_learn.registerIndex('observations', len(observations))
        tns_learn.registerIndex('samples', tns_learn.indexSizes['stilde']*10)
        
        #input tensors for each observation
        tns_learn.registerTensor('PSIhat', (['r', 'd', 'g'], ['rtilde', 'dtilde', 'gtilde', 'stilde', 'samples']) )
        tns_learn.registerTensor('Ohat', (('r', 'd', 'g'), ('samples',)) ) 
        tns_learn.registerTensor('Yhat', (('r', 'd', 'g'), ('samples',)) ) 
        #computations to perform: invert PSI, subtract the task frame offset O from the data Yhat, and contract both
        tns_learn.registerInverse('PSIhat')
        tns_learn.registerSubtraction('Yhat', 'Ohat')
        tns_learn.registerContraction('(PSIhat)^#', '(Yhat-Ohat)') #yields '(PSIhat)^#:(Yhat-Ohat)'

        #for now, computation of these tensors is hard-coded:
        tns_learn.registerTensor('Wobservations', ((['observations', 'rtilde', 'dtilde', 'gtilde', 'stilde']), ()) )
        tns_learn.registerTensor('Wmean', (['rtilde', 'dtilde', 'gtilde', 'stilde'], ()) )
        tns_learn.registerTensor('Wcov', (['rtilde', 'dtilde', 'gtilde', 'stilde'], ['rtilde_', 'dtilde_', 'gtilde_', 'stilde_'],) )
        
        print(self.tns)
        print(tns_learn)
        #do linear regression on each observation individually to obtain samples in W:
        for observation_idx, (phases, values, Xrefs, Yrefs) in enumerate(observations):
            num = phases.shape[0]
            
            #gather PSI projections for each sample into PSIhat:
            for n in range(num):
                self.tns.setTensor('phase', phases[n,:])
                self.tns.setTensor('Xref', Xrefs[n,:,:,:]) #TODO: would be computed/overridden by T_computer!!
                self.tns.setTensor('Yref', Yrefs[n,:,:,:])
                
                self._updateP() #call the code that computes map between phase and time derivatives. implemented directly in this class
                self.PHI_computer.update() #call the hook that computes the interpolating map PHI:
                self.T_computer.update() #call the hook that computes the linear(-ized) task-space map T, Xref and Yref:
                #Now compute the actual ProMeP equation:
                self.tns.update('P:PHI', 'PSI')

                #save into PSIhat and Ohat of the tns_learn namespace:
                tns_learn.tensorData['PSIhat'][n,:,:,:,:,:,:,:] = self.tns.tensorData['PSI']
                tns_learn.tensorData['Ohat'][n,:,:,:] = self.tns.tensorData['O']
            
            #compute the sample:
            tns_learn.setTensor('Yhat', values) #set Yhat to current observed trajectory samples
            tns_learn.update() #estimate a W for the given observation
            #aggregate every Wsample into Wsamples:
            tns_learn.tensorData['Wobservations'][observation_idx,:,:,:,:] = tns_learn.tensorData['(PSIhat)^#:(Yhat-Ohat)']
        
        #samples in W have been gathered. Now estimate the distribution parameters from them:

        #estimate distribution from the estimated samples:        
        tns_learn.setTensor('Wmean', _np.mean(tns_learn.tensorData['Wobservations'], axis=-1))  #need to make sure indices are ordered correctly!

        #estimate the covariance tensor. indices are hard-coded for now:
        model  = _sklearn.covariance.OAS()  #Alternatives: OAS, GraphLasso, EmpiricalCovariance algorithms
        #for OAS and GraphLasso, normalize the input first to avoid lopsided importance between dofs / realms / derivatives due to differing units
        stddevs = _np.sqrt(_np.var(tns_learn.tensorData['Wobservations'], axis=0)) + tolerance  #adding tolerance ensure that stddevs never are exactly 0
        Wsamples_normalized = (tns_learn.tensorData['Wobservations'] - tns_learn.tensorData['Wmean'][:,:,:,:,None]) / stddevs
        model.fit( Wsamples_normalized.reshape((Wsamples_normalized.shape[0], -1)) ) #flatten tensor to estimate correlations with sklearn
        covariance_tensor_flat = _np.dot(_np.dot(scale,model.covariance_),scale.T)   #re-scale back to covariances
        tns_learn.setTensorFromFlattened('Wcov', covariance_tensor_flat) 
        
        #for introspection:
        self.observations = observations

        #update yourself to the learnt estimate
        self.tns.setTensor('Wmean', tns_learn.tensorData['Wmean'], tns_learn.tensorIndices['Wmean'])
        self.tns.setTensor('Wcov', tns_learn.tensorData['Wcov'], tns_learn.tensorIndices['Wmean'])


# parallelizable function for EM learning:
_tnscopy_perprocess = {}
def _estimation_func(tns_base, Wmean, Wcov, PSIi, Ydeltai):
        global _tnscopy_perprocess
        pid = _os.getpid()
        try:
            tns_local = _tnscopy_perprocess[pid]
        except KeyError:
            tns_local = tns_base.copy()
            _tnscopy_perprocess[pid] = tns_local
            #print("deep-copying tns for pid {}".format(pid))
        tns_local.setTensor('Wmean', Wmean)
        tns_local.setTensor('Wcov', Wcov)
        tns_local.setTensor('PSIi', PSIi)
        #set the observed sample:
        tns_local.setTensor('Yi', Ydeltai, arrayIndices= (('r', 'd', 'g'),()))
        #compute the estimation step:
        tns_local.update()
        return tns_local.tensorData['Wmeani'].copy(), tns_local.tensorData['Wcovi'].copy(), tns_local.tensorData['negLLidelta'].copy()



