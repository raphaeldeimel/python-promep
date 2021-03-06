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
import mechanicalstate  as _mechanicalstate
import namedtensors as _namedtensors

from . import _trajectorycompositors, _kumaraswamy

import pandas as _pandas

from  scipy.special import betainc as _betainc
from sklearn import covariance as _sklearncovariance


def gradient(series):
    g = _np.zeros_like(series)
    g[1:-1] = 0.5* (series[2:] - series[0:-2])
    g[0] = series[1] - series[0]
    g[-1] = series[-1] - series[-2]
    return g




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
                 expected_phase_profile_params=(1.0, 1.0),  #parameters of a Kumaraswamy distribution
                 task_space_name = 'jointspace',
                 trajectory_composition_method='gaussians'
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
                  self.tns.registerIndex(name, self.index_sizes[name])

        self.trajectory_composition_method = trajectory_composition_method

        self.task_space_name = task_space_name

        ##define the computational graph:

        #register all tensors not being computed by operators:
        #i.e. input tensors:
        self.tns.registerTensor('phase', (('g',),()) )  #generalized phase vector        
        self.tns.registerTensor('Wmean', (('rtilde','gtilde','stilde','dtilde'),()) )
        self.tns.registerTensor('Wcov', (( 'rtilde','gtilde','stilde','dtilde'),( 'rtilde_','gtilde_','stilde_','dtilde_')) )

        #set which trajectory composition method generates PHI?
        if self.trajectory_composition_method=='gaussians':
            self.trajectorycompositor=_trajectorycompositors.TrajectoryCompositorGaussian(tensornamespace=self.tns) 
            self.tns.registerExternalFunction(self.trajectorycompositor.update, ['phase'], ['PHI'],  [(('gphi',),('stilde',))])
        else: #Currently, we have only one method
            raise NotImplementedError()        

        #set function to compute chain derivatives (P tensor):
        self.tns.registerExternalFunction(_updateP, ['phase'], ['P'],  [(('g',),('gphi','gtilde'))] )
        
        #set up tensors involved in taskspace mapping:
        self.tns.registerTensor('Yref', (('r','g','d'),()) )     
        self.tns.registerTensor('Xref',  (('rtilde', 'g', 'dtilde'),()) )    
        self.tns.registerTensor('T',   (('r','d'),('rtilde', 'dtilde')) ) 
        if self.task_space_name=='jointspace':
            self.tns.setTensorToIdentity('T')

        
        self.tns.registerContraction('T', 'Xref')  
        self.tns.registerSubtraction('Yref', 'T:Xref', result_name = 'O')  # observations may not have used the task space computer - set manually

        
        #path to compute PSI:
        self.tns.registerContraction('P', 'PHI')
        self.tns.registerContraction('T', 'P:PHI', result_name='PSI')
        self.tns.registerTranspose('PSI')
        
        
        #project the mean of weights to mean of trajectory:
        self.tns.registerContraction('PSI', 'Wmean')
        
        self.tns.registerAddition('PSI:Wmean', 'O', result_name='Ymean', align_result_to=(('r','g','d'),()))
        
        #project the covariances of weights to covariances of trajectory:
        self.tns.registerContraction('PSI', 'Wcov')
        self.tns.registerContraction('PSI:Wcov', '(PSI)^T', result_name='Ycov', align_result_to=(('r','g','d'),('r_','g_','d_',)) )

        ######
       
        self.phaseAssociable = True #indicate that this motion generator is parameterized by phase
        self.timeAssociable = False #indicate that this motion generator is not parameterizable by time
        self.tolerance=1e-7
        
        self.expected_duration = expected_duration
        self.expected_phase_profile_params = expected_phase_profile_params

        if not Wmean is None:
            self.tns.setTensor('Wmean', Wmean)
        if not Wcov is None:
            self.tns.setTensor('Wcov', Wcov)
        
        #temporary/scratch tensors:
        self.tns.registerTensor('Wsample', (('r','gtilde','stilde','dtilde'),()) )
        
        #setup a MechanicalStateDistribution() object to return results:
        self.msd_expected = _mechanicalstate.MechanicalStateDistribution(self.tns, 'Ymean', 'Ycov')
        
        #get the mapping from human-readable names to indices of m-state distributions
        self.commonnames2rg = self.msd_expected.commonnames2rg 
        self.rg_commonnames = self.msd_expected.rg_commonnames
        

        #some initial plot range values so we can plot something at all
        self.plot_range_guesses = { 'torque': [-20,20], 'position': [-1.5,1.5], 'velocity': [-2.0,2.0], 'gains': [-10,100.0],}
        self.parameterMask = None  #the mask used for learning parameters

    def __repr__(self):
        strings = ["Indices:"]
        for key in self.tns.index_names:
            strings.append("{}: {}".format(key,self.tns[key].size))
        strings.append("")
        strings.append("Tensors:")
        for key in self.tns.tensor_names:
            strings.append("{}     {} / {}".format(key,self.tns[key].indices_upper,self.tns[key].indices_lower ))
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
        serializedDict[u'index_sizes'] = {key: self.tns[key].size for key in self.tns.index_names if not key.endswith('_')}
        serializedDict[u'Wmean'] = self.tns['Wmean'].data
        serializedDict[u'Wcov'] = self.tns['Wcov'].data
        serializedDict[u'expected_duration'] = self.expected_duration
        serializedDict[u'expected_phase_profile_params'] = self.expected_phase_profile_params
        serializedDict[u'trajectory_composition_method'] = self.trajectory_composition_method
        serializedDict[u'task_space_name'] = self.task_space_name
        
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
            'expected_phase_profile_params': serializedDict[u'expected_phase_profile_params'],
            'trajectory_composition_method': serializedDict[u'trajectory_composition_method'],
            'task_space_name': serializedDict[u'task_space_name'],
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
        self.tns['Wsample'].data_flat[:,0] = _np.random.multivariate_normal(self.tns['Wmean'].data_flat[:,0], self.tns['Wcov'].data_flat)
        return self.tns['Wsample'].data



    def getDistribution(self, generalized_phase=None, current_msd=None, task_spaces={}):
        """
        return the distribution of the (possibly multidimensional) state at the given phase
           for now, it returns the parameters (means, derivatives) of a univariate gaussian

            generalized_phase: generalized phase to compute the distribution for
                        
            currentDistribution: not used
            
            taskspaces: dictionary of task space mappings. May be used 

            returns a MechanicalStateDistribution object with the expected distribution  

        """
        self.tns.setTensor('phase', generalized_phase)
        if self.task_space_name != 'jointspace':  #only update task space mappings if we actually need to
            taskspacemapping_tensors = task_spaces[self.task_space_name]
            self.tns.setTensor('Yref', taskspacemapping_tensors['Yref']) 
            self.tns.setTensor('Xref', taskspacemapping_tensors['Xref']) 
            self.tns.setTensor('T', taskspacemapping_tensors['T']) 
        #Now compute the actual ProMeP equation:
        self.tns.update()
        return self.msd_expected  #sends back reference to Ymean and Ycov






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
        if generalized_phases.shape[1] != self.tns['gphi'].size:
            raise ValueError()

        if W is None:
            W = self.sample()
            
        #compute the trajectory:    
        Wmean_saved = _np.array(self.tns['Wmean'].data, copy=True)
        self.tns.setTensor('Wmean', W)
        points = _np.zeros( (num, self.tns['r'].size, self.tns['g'].size, self.tns['d'].size))
        for i in range(num):
            _np.copyto(points[i,...], self.getDistribution(generalized_phase=generalized_phases[i]).getMeansData())
        self.tns.setTensor('Wmean', Wmean_saved) #restore abused Wmean
        return points



    def plot(self, dofs='all',
                   whatToPlot=['position', 'velocity', 'kp', 'kv', 'int_kv','int_ka', 'impulse', 'torque'],
                   num=101,
                   linewidth=1.0,
                   addExampleTrajectories=10,
                   withConfidenceInterval=True,
                   plotRanges = None,
                   exampleTrajectoryStyleCycler=_plt.cycler('color', [(0.0,0.0,0.8)]),
                   useTime=True,
                   margin=0.05
                   ):
        """
        visualize the trajectory distribution as parameterized by the means of each via point,
        and the covariance matrix, which determins the variation at each via (sigma) and the
        correlations between variations at each sigma

        E.g. usually, neighbouring via points show a high, positive correlation due to dynamic acceleration limits

        """
        supportsColor = '#008888'
        confidenceColor = "#DDDDDD"
        meansColor = '#BBBBBB'
        observedColor = (0.8,0.0,0.0)
        kpColor = '#000000'
        kvColor = '#000000'
        kpCrossColor = '#666666'
        kvCrossColor = '#666666'

        phases = _np.zeros((num, self.tns['gphi'].size))
        
        if useTime: 
            phases[:,0] = _kumaraswamy.cdf(self.expected_phase_profile_params[0], self.expected_phase_profile_params[1], _np.linspace(0,1.0,num))
            plot_x = _np.linspace(0, self.expected_duration, num) #linear time
            for g_idx in range(1,self.tns['gphi'].size):
                phases[:,g_idx] = gradient(phases[:,g_idx-1]) * num / self.expected_duration 
        else:
            phases[:,0] = _np.linspace(0.0, 1.0, num)
            plot_x = phases[:,0]
            if self.tns['gphi'].size > 1:
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
                'int_kv': '[Nm/rad/s]',
                'int_ka': '[Nm/rad/s^2]',
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
                'int_kv': '[Nm/rad/1]',
                'int_ka': '[Nm/rad/1^2]',
            }
        
        whatToPlot = [name for name in whatToPlot if name in self.commonnames2rg ]

        if dofs=='all' or dofs == None:
            dofs_to_plot=self.tns['d'].values
        else:
            dofs_to_plot = dofs
        subplotfigsize=2.0

        plotrows = len(whatToPlot)
            
        #gather the data:
        data_mean  = _np.zeros((num,plotrows, self.tns['d'].size))
        data_sigma = _np.zeros((num,plotrows, self.tns['d'].size))

        data_gains={}
        for names in whatToPlot:
            if len(self.commonnames2rg[names])==4:
                data_gains[names] = _np.zeros((num,self.tns['d'].size, self.tns['d'].size))
                
        generalized_phase =_np.zeros((self.tns['g'].size))            
        for i,phase in enumerate(phases):
            dist =  self.getDistribution(generalized_phase=phase)
            gains  = dist.extractTorqueControlGains()
            means =  dist.getMeansData()
            variances = dist.getVariancesData()
            for row_idx, row_name in enumerate(whatToPlot):
                if len(self.commonnames2rg[row_name]) == 4: #name of a gain
                    r_idx, g_idx, r2_idx, g2_idx = self.commonnames2rg[row_name]
                    data_gains[row_name][i,:,:] = gains[g_idx,:,g2_idx,:]
                else:                
                    r_idx, g_idx = self.commonnames2rg[row_name]                
                    data_mean[i,row_idx,:] = means[r_idx,g_idx,:] 
                    data_sigma[i,row_idx,:] = _np.sqrt(variances[r_idx,g_idx,:] )        

        fig, axesArray = _plt.subplots(plotrows,len(dofs_to_plot), squeeze=False, figsize=(max(len(dofs_to_plot), plotrows)*subplotfigsize, plotrows*subplotfigsize), sharex='all', sharey='row')
        for ax in axesArray.flat:
            ax.margins(x=0.0, y=margin)
        _plt.suptitle('trajectories')
  
        #draw confidence intervals and means/variance indicators for the supports
        #plot the zero-variance trajectory + 95% confidence interval        
        for row_idx, row_name in enumerate(whatToPlot):
            for col_idx, dof in enumerate(dofs_to_plot):
                if row_name in data_gains:
                    axesArray[row_idx,col_idx].axhline(0.0, label=None,  color=(0.4,0.4,0.4), linestyle=':')
                    for col_idx2, dof2 in enumerate(dofs_to_plot):
                        if dof != dof2:
                            axesArray[row_idx,col_idx].plot(plot_x,data_gains[row_name][:,dof,dof2], label=None,  color=kpCrossColor, linestyle=':')
                    #plot the joint-local gains prominently, and on top (i.e. last)
                    axesArray[row_idx,col_idx].plot(plot_x,data_gains[row_name][:,dof,dof], label=row_name,  color=kpColor)
                    
                else:
                    meanvalues = data_mean[:,row_idx,dof]
                    sigmavalues = data_sigma[:,row_idx,dof]
                    upper_boundary = meanvalues+1.96*sigmavalues
                    lower_boundary = meanvalues-1.96*sigmavalues
                    if withConfidenceInterval:
                        axesArray[row_idx,col_idx].fill_between(plot_x,lower_boundary, upper_boundary, label="95%",  color=confidenceColor)
                    axesArray[row_idx,col_idx].plot(plot_x,meanvalues, label="mean",  color=meansColor)


        if 'observedTrajectories' in self.__dict__: #plot observations after scaling y axes
            alpha = _np.clip(1.0 / _np.sqrt(1.0*len(self.observedTrajectories)), 0.1, 1.0)
            linewidthfactor =  1.0 / (alpha *_np.sqrt(1.0*len(self.observedTrajectories)))
            for observation_idx, (times, phases_observation, values, Xrefs, Yrefs, Ts) in enumerate(self.observedTrajectories): 
                for row_idx, row_name in enumerate(whatToPlot):
                    if len(self.commonnames2rg[row_name])==4: #is a name for a gain
                        continue
                    r_idx, g_idx = self.commonnames2rg[row_name]
                    if useTime:
                        y = values[:,r_idx,g_idx,:]
                        for col_idx, dof in enumerate(dofs_to_plot):
                            axesArray[ row_idx, col_idx].plot(times, y[:,dof], alpha=alpha, linewidth=linewidth*linewidthfactor, color=observedColor )
                    else:
                        if phases_observation.shape[1] > g_idx:   #only plot in phase if we have phase derivatives to scale observations with 
                            if g_idx > 0:
                                scaler = 1.0 / phases_observation[:,1]
                                y = values[:,r_idx,g_idx,:] * (scaler[:,None]**g_idx)
                            else:
                                y = values[:,r_idx,g_idx,:]
                            for col_idx, dof in enumerate(dofs_to_plot):
                                axesArray[ row_idx, col_idx].plot(phases_observation[:,0], y[:,dof], alpha=alpha, linewidth=linewidth*linewidthfactor, color=observedColor)



        #sample the distribution to plot actual trajectories, times the number given by "addExampleTrajectories":
        if addExampleTrajectories is None:
            addExampleTrajectories = 0
        alpha = _np.clip(1.0 / _np.sqrt(1.0*(1+addExampleTrajectories)), 0.1, 1.0)
        for ax in axesArray.flatten():
            ax.set_prop_cycle(exampleTrajectoryStyleCycler)

        for j in range(addExampleTrajectories):
            yvalues = self.sampleTrajectory(phases)
            for row_idx, row_name in enumerate(whatToPlot):
                if len(self.commonnames2rg[row_name])==4: #is a name for a gain
                    continue
                r_idx, g_idx = self.commonnames2rg[row_name]
                for i, dof in enumerate(dofs_to_plot):
                    axesArray[row_idx,col_idx].plot(plot_x, yvalues[:,r_idx,g_idx,dof], alpha=alpha, linewidth=linewidth )
        


        #set ylimits to be equal in each row and xlimits to be equal everywhere
        for row_idx, row_name in enumerate(whatToPlot):
                axes_row = axesArray[row_idx,:]  
                if plotRanges is None or not row_name in plotRanges:
                    ylimits  = _np.array([ ax.get_ylim() for ax in axes_row ])
                    ylimit_common = _np.max(ylimits, axis=0)
                    ylimit_common_symm = _np.max(_np.abs(ylimits))
                    ylimit_common = (-ylimit_common_symm, ylimit_common_symm) 
                else:
                    if _np.isscalar(plotRanges[row_name]):
                        ylimit_common = (-plotRanges[row_name], plotRanges[row_name]) #else treat it as a single value
                    else:
                        ylimit_common = plotRanges[row_name][0:2]  #try using it as a2-tuple
                        
                for dof, ax in zip(dofs_to_plot, axes_row):
                    ax.set_title('\detokenize{{{} {}}}'.format(row_name, dof))
                    ax.set_ylim(ylimit_common)                
                    ax.get_yaxis().set_visible(False)
                    yticks = [ ylimit_common[0], 0.0, ylimit_common[1] ]                    
                    yticklabels = ['{0:0.1f}'.format(ylimit_common[0]), '\detokenize{{{}}}'.format(units[row_name]), '{0:0.1f}'.format(ylimit_common[1])]
                    yticks, yticklabels = tuple(zip(*sorted(zip(yticks, yticklabels))))
                    ax.set_yticks(yticks)
                    ax.set_yticklabels(yticklabels)





        for ax in axesArray[:,0]:
#             ax.set_ylabel(units[row_name])
             ax.get_yaxis().set_visible(True)

        if useTime:  
           xlimits_common  = _np.max(_np.array([ ax.get_xlim() for ax in axesArray.flat ]), axis=0)     
           for ax in axesArray.flat:
                ax.set_xlim(xlimits_common)                
                ax.get_xaxis().set_visible(False)                    
           for ax in axesArray[-1,:]:
                ax.set_xticks( [ 0.0, 0.5*plot_x[-1], plot_x[-1] ] )
                ax.set_xticklabels(['0', 'time [s]', '{0:0.1f}'.format(plot_x[-1])])
                ax.get_xaxis().set_visible(True)                    
        else:
           for ax in axesArray.flat:
                ax.set_xlim(0,1.0)
           for ax in axesArray[-1,:]:
                ax.set_xticks([0.0,0.5, 1.0])
                ax.set_xticklabels(['0', 'phase', '{0:0.1f}'.format(plot_x[-1])])


        _plt.tight_layout()
        return fig


    def plotCovarianceTensor(self, normalize_indices='rg', color_maskedvalues=(0.9, 0.9, 0.9), omit_masked_parameters=False):
        """
        plot the covariances/correlations in parameter space
        
        normalize_indices: string of index letters used to select which dimension to normalize
            'rgsd': all indices (correlation matrix)
            '': verbatim covariance matrix
            ''rg': variances between realms and between derivatives are normalized (default)
        """
        cov = self.tns['Wcov'].data
        variance_view = _np.einsum('ijklijkl->ijkl', self.tns['Wcov'].data)

        sigmas = _np.sqrt(variance_view)

        firstletters = [string[0] for string in self.tns['Wcov'].indices_upper]
        axes_to_keep = tuple([ firstletters.index(letter) for letter in normalize_indices ])
        axes_to_marginalize = tuple(set(range(4))- set(axes_to_keep))
        if len(axes_to_marginalize) == 4:
            sigmamax = _np.max(sigmas, axis=axes_to_marginalize, keepdims=True)
            title = "Covariances"
        elif len(axes_to_marginalize) == 0:
            sigmamax = _np.max(sigmas, axis=axes_to_marginalize, keepdims=True)
            title = "Correlations"
        else:
            sigmamax = _np.max(sigmas, axis=axes_to_marginalize, keepdims=True)
            title = "Covariances with {} normalized".format(",".join(normalize_indices))
        
        sigmamax_inv = 1.0 / _np.clip(sigmamax, 1e-6, _np.inf)        
        cov_scaled = sigmamax_inv[:,:,:,:, None,None,None,None] * cov * sigmamax_inv[None,None,None,None, :,:,:,:] 
        vmax=_np.max(cov_scaled)

        #mask covariances that are not useful to interpretation as the underlying parameters were masked during learning
        if self.parameterMask is None or not omit_masked_parameters:
            cov_masked = cov_scaled
        else:
            m = _np.logical_or(self.parameterMask[:,:,:,:,None,None,None,None],self.parameterMask[None,None,None,None,:,:,:,:])
            cov_masked = _np.ma.masked_array(cov_scaled, mask=m)

        cov_reordered =_np.transpose(cov_masked, axes=(2,0,1,3, 2+4,0+4,1+4,3+4)) #to srgd
        image =_np.reshape(cov_reordered, self.tns['Wcov'].shape_flat)

        gridvectorX = _np.arange(0, image.shape[0]+1, 1)
        gridvectorY = _np.arange(image.shape[1], -1, -1)

        fig = _plt.figure(figsize=(3.4,3.4))
        _plt.suptitle('covariances_' + normalize_indices)        
        _plt.pcolor(gridvectorX, gridvectorY, image, cmap=_cmapCorrelations, vmin=-vmax, vmax=vmax)
        #_plt.pcolor(gridvectorX, gridvectorY, mask , facecolor=('g'), vmin=0.0, vmax=1.0)
        fig.axes[0].set_facecolor(color_maskedvalues)
        
        _plt.axis([0, image.shape[0], 0, image.shape[1]])
        _plt.gca().set_aspect('equal', 'box')

        len_all = self.tns['Wcov'].shape_flat[0]
        len_rtilde = self.tns['rtilde'].size
        len_stilde = self.tns['stilde'].size
        len_dtilde = self.tns['dtilde'].size
        len_gtilde = self.tns['gtilde'].size
        line_positions = _np.reshape(_np.arange(self.tns['Wcov'].shape_flat[0]), cov_reordered.shape[:4])
        linewidth_base=1.0
        for r_idx in range(len_rtilde):
          for g_idx in range(len_gtilde):
            for s_idx in range(len_stilde):
                for d_idx in range(len_dtilde):
                    linewidth=linewidth_base
                    linestyle='-'
                    color=(0,0,0)
                    if r_idx!=0:
                        linewidth=0.1*linewidth_base
                        linestyle='-'
                        color=(0.5,0.5,0.5)
                    if g_idx!=0:
                        linewidth=0.1*linewidth_base
                        linestyle=':'
                        color=(0.5,0.5,0.5)
                    if d_idx!=0:
                        linewidth=0.0
                        color=(0.8,0.8,0.8)
                    if linewidth>0.0:
                        _plt.axhline(line_positions[s_idx,r_idx, g_idx,d_idx], color=color, linewidth=linewidth, linestyle=linestyle)
                        _plt.axvline(line_positions[s_idx,r_idx, g_idx,d_idx], color=color, linewidth=linewidth, linestyle=linestyle)

        baselength = len_gtilde*len_dtilde
        ticklabels = []
        ticks=[]
        offsets=[]
        for s in range(len_stilde):
            for r2 in range(2*len_rtilde):
                ticks.append( ((s)*len_rtilde + r2/2)*baselength )
                if r2 == 3:
                    ticklabels.append('m')
                    offsets.append(0.0)
                elif r2 == 2:
                    ticklabels.append("{}".format(len_stilde-s-1))
                    offsets.append(-0.2*baselength)
                elif r2 == 1:
                    ticklabels.append("e")
                    offsets.append(0.0)
                else:
                    ticklabels.append("")
                    offsets.append(0.)
        for tick, label, offset in zip(ticks, ticklabels, offsets):
            t = _plt.text(offset, tick, label, {'verticalalignment':'center', 'horizontalalignment':'right', 'size':'xx-small'})
        _plt.yticks([])
        _plt.text(0.0,ticks[0]+0.3*baselength, "$\widetilde{r}$", fontdict={'verticalalignment':'bottom', 'horizontalalignment':'right', 'size':'small'})
        _plt.text(-10.0,ticks[0]+0.3*baselength, "$\widetilde{s}$", fontdict={'verticalalignment':'bottom', 'horizontalalignment':'right', 'size':'small'})

        #ticks in x:
        ticks = range( (len_dtilde)//2, len_all, (len_dtilde))
        ticklabels = []
        ticks=[]
        offsets=[]
        for s in range(len_stilde):
            for r in range(len_rtilde):
                for g in range(len_gtilde):
                    ticks.append( (((s)*len_rtilde + r)*len_gtilde + g)*len_dtilde + len_dtilde/2 )
                    ticklabels.append(  '\detokenize{{{}}}'.format(self.rg_commonnames[r][g]))
                    offsets.append(-0.2)
        for tick, label, offset in zip(ticks, ticklabels, offsets):
            t = _plt.text(tick, offset, label, fontdict={'verticalalignment':'top', 'horizontalalignment':'center', 'size':'xx-small'}, rotation=90)
        _plt.text(ticks[-1]+10, 0.0, "$\widetilde{g}$", fontdict={'verticalalignment':'top', 'horizontalalignment':'left', 'size':'small'})
            
        _plt.xticks([])


        _plt.colorbar(shrink=0.6, aspect=40, ticks=[-vmax,0,vmax], fraction=0.08)
        _plt.title(title)
        ax = _plt.gca()        
        #_plt.tight_layout()


    def plotExpectedPhase(self):
        """
        Visualize the phase curve expected by this ProMeP
        """
        fig= _plt.figure()
        num=200
        x = _np.linspace(0.0, 1.0, num)
        y = _kumaraswamy.cdf(self.expected_phase_profile_params[0],self.expected_phase_profile_params[1],x)
        ydot  = gradient(y) * num / self.expected_duration
        if 'observed_phaseprofiles' in self.__dict__:    
            _plt.plot(self.expected_duration*self.observed_phaseprofiles[:,0], self.observed_phaseprofiles[:,1], '.', color=(0.8,0.5,0.5))        

        _plt.plot(self.expected_duration*x,y, color=(0.2,0.2,1.0))
        _plt.xlabel('Time')
        _plt.ylabel('Phase')
        _plt.suptitle('expectedphase')
        ax_right = _plt.gca().twinx()
        ax_right.plot(self.expected_duration*x,ydot, color=(0.6,0.6,0.6), linewidth=0.3)
        ax_right.set_ylabel('Phase velocity')
        _plt.tight_layout()
        return fig

        
    def learnFromObservations(self, 
            observations,
            max_iterations=10, 
            minimal_relative_improvement=1e-9,
            target_space_observations_indices = (('r','g','d'),()),
            task_space_observations_indices = (('rtilde', 'g', 'dtilde'),()),
            task_map_observations_indices= (('r', 'd'),('rtilde', 'dtilde')),
            mask = None,
            ):
        """
        
        compute parameters from a list of observations
        
        Observations are tuples of  (times, phases, values, Xrefs, Yrefs, Ts) tensors (data arrays)
        
        phases: Generalized phases of shape (n, 'd')
        means: observed values at the given phases, of shape (n,) + observations_indices
        Xref, Yref: Task space linearization references, of shape (n, 'rtilde', 'gtilde', 'stilde','dtilde' ) and (n, 'r', 'd', 'g') respectively
        #
       
        Implements the expectation-maximization procedure of the paper "Using probabilistic movement primitives in robotics" by Paraschos et al.
        """
        pool = _multiprocessing.Pool()        
        
        #truncate equations that we do not need to compute:
        update_psionly = self.tns.update_order[:self.tns.update_order.index('PSI')+1]
               
        #set up equations to be computed for each sample, .i.e. PSImasked and Yhatslice:
        tns_perSample = _namedtensors.TensorNameSpace(self.tns)
        
        #tensors to use as inputs from self.tns:
        tns_perSample.registerTensor('O', self.tns['O'] )
        tns_perSample.registerTensor('PSI', self.tns['PSI'])
        tns_perSample.renameIndices('PSI', {'rtilde': 'rtilde_', 'gtilde': 'gtilde_', 'stilde': 'stilde_', 'dtilde': 'dtilde_', })
                
        #subtract the offset caused by the task map linearization when computing the per-sample data, so during learning we ignore Xref and Yref
        tns_perSample.registerTensor('Yobserved', (('r', 'g', 'd'),()) )
        tns_perSample.registerSubtraction('Yobserved', 'O', result_name='Yhatslice')


        tns_perSample.registerTensor('mask', (('rtilde_', 'gtilde_', 'stilde_', 'dtilde_'),('rtilde', 'gtilde', 'stilde', 'dtilde')) , initial_values='identity')
        tns_perSample.registerTensor('parameterMask', (('rtilde', 'gtilde', 'stilde', 'dtilde'),()))
        if mask != None:       
            for indices in mask:
                tns_perSample.setTensorSlice('mask', indices, 0.0)
                tns_perSample.setTensorSlice('parameterMask', indices, 1.0)
            self.parameterMask = tns_perSample['parameterMask'].data.copy()
        tns_perSample.registerContraction('renamed(PSI)', 'mask', result_name='PSImasked')
        
        self.tns_perSample = tns_perSample
        #we only need to compute until we have PSI:
        tns_Observations =[]
        self.tns_Observations = tns_Observations

        #"precompute" the mapping from joint space to joint space:
        T_jointspace =_np.eye( (self.tns['r'].size*self.tns['d'].size) )
        T_jointspace.shape = (self.tns['r'].size,self.tns['d'].size,self.tns['rtilde'].size,self.tns['dtilde'].size) #r,d,rtilde, dtilde
        
        durations = []

        observations_converted = []
        for observation_idx, observation in enumerate(observations):            
            if isinstance(observation, tuple):   #old interface: tuple of arrays
                observations_converted.append(observation)
            elif isinstance(observation, _pandas.DataFrame): #preferred one: pandas dataframe

                samples_total = len(observation.index)
                data_shape = (samples_total, self.tns['r'].size,self.tns['g'].size,self.tns['d'].size)
                values = _np.zeros(data_shape)
                for r_idx in range(self.tns['r'].size):
                    for g_idx in range(self.tns['g'].size):
                        name = self.rg_commonnames[r_idx][g_idx]
                        for d_idx in range(8):                        
                            try:
                                v_observed = observation[ ('observed', name, str(d_idx) ) ]
                            except KeyError:
                                print("Warning:observation {}'s table does not contain column for {},{},{}".format(observation_idx, 'observed', name, d_idx))
                                v_observed = 0.0
                            values[:, r_idx, g_idx, d_idx] = v_observed
                times = observation['observed','time','t']
                phases = _np.zeros((samples_total,self.tns['g'].size ))
                for i in range(self.tns['g'].size):
                    phases[:,i] = observation['observed', 'phi', i]
                #setup for joint-space learning:
                Yrefs = _np.zeros( (samples_total, self.tns['r'].size,self.tns['g'].size,self.tns['d'].size) ) 
                Xrefs = _np.zeros( (samples_total, self.tns['rtilde'].size,self.tns['g'].size,self.tns['dtilde'].size) ) 
                Ts = _np.tile(T_jointspace, (samples_total,1,1,1,1))
                observations_converted.append( (times, phases, values, Xrefs, Yrefs, Ts) )

        for observation_idx, observation in enumerate(observations_converted):            
            if isinstance(observation, tuple):   #old interface: tuple of arrays
                times, phases, values, Xrefs, Yrefs, Ts = observation
                samples_total = phases.shape[0]
                if phases.shape[0] != values.shape[0]:
                    raise ValueError()
                if phases.shape[0] != times.shape[0]:
                    raise ValueError()
            else:
                raise ValueError("Unsupported data structure for observations argument")

            #make sure we don't rely on the order that samples are specified in. also makes sure that contiguous partitions are likely to cover the whole trajectory:
            sampleindices = _np.arange(samples_total)
            _np.random.shuffle(sampleindices)

            #compute how we partition the observation samples into subsets to reduce computational effort and increase the number of w samples available for covariance estimation:
            target_samplesetsize  = self.tns['stilde'].size * 3  #no need to add much more samples than we have interpolation parameters over time            
            
            partitions = samples_total // target_samplesetsize
            samplesetsize = samples_total // partitions
            orphans = samples_total % partitions
            partitions_startindices = _np.array([samplesetsize *i for i in range(partitions+1) ])
            for o in range(orphans+1):
                i = partitions - orphans + o
                partitions_startindices[i]  = partitions_startindices[i] + o
            partitions_startindices[-1] = samples_total #make sure to not lose any samples / go beyond last sample
            
            for partition_idx in range(partitions):
                
                
                sampleindices_partition = sampleindices[partitions_startindices[partition_idx]:partitions_startindices[partition_idx+1]]
                samplesetsize = sampleindices_partition.size
                
                tns_perObservation = _namedtensors.TensorNameSpace(self.tns)
                self.tns_Observations.append(tns_perObservation)            
                
                tns_perObservation.registerIndex('samples', samplesetsize)        

                tns_perObservation.registerTensor('Yhat', (('samples','r', 'g', 'd'),()))
                tns_perObservation.registerTensor('PSIhat', (('samples','r', 'g','d'), ('rtilde', 'dtilde','gtilde', 'stilde')))
                tns_perObservation.registerTensor('Wmean', self.tns['Wmean']) #not a copy                    
                tns_perObservation.registerTensor('Wcov', self.tns['Wcov'])   #not a copy
                
                #precomputatable:
                tns_perObservation.registerInverse('PSIhat', flip_underlines=True)
                tns_perObservation.registerContraction( '(PSIhat)^#', 'PSIhat', result_name='PP')
                tns_perObservation.registerTensor('I_PP', tns_perObservation['PP'].index_tuples, initial_values='identity') 
                tns_perObservation.registerSubtraction('I_PP', 'PP')
                tns_perObservation.registerTranspose('(I_PP-PP)', result_name='PPnullspace', flip_underlines=False)
                
                #recompute every iteration:
                tns_perObservation.registerContraction('PSIhat', 'Wmean', result_name='Ymean')
                tns_perObservation.registerSubtraction('Yhat', 'Ymean', result_name='Yerror')
                tns_perObservation.registerContraction('(PSIhat)^#', 'Yerror', result_name='Werror', flip_underlines=True)
                           
                tns_perObservation.registerTranspose('Werror')                
                tns_perObservation.registerContraction('Werror', '(Werror)^T', result_name='Wcovprojected')
                tns_perObservation.registerElementwiseMultiplication('Wcov', 'PPnullspace', result_name='Wcovnullspace')
                tns_perObservation.registerAddition('Wcovprojected', 'Wcovnullspace', result_name='Wcovestimate')
                
                tns_perObservation.lazyupdate = tns_perObservation.update_order[tns_perObservation.update_order.index('Ymean'):]                


                #preprocess samples into pairs of Yhat and PSIhat for each observation:
                for i, sample in enumerate(sampleindices_partition):
                    #compute PSIi:
                    self.tns.setTensor('Xref',       Xrefs[sample,...], task_space_observations_indices ) 
                    self.tns.setTensor('Yref',       Yrefs[sample,...], target_space_observations_indices )                
                    self.tns.setTensor('T',             Ts[sample,...], task_map_observations_indices )
                    self.tns.setTensor('phase',     phases[sample,:], (('g'),()) )
                    self.tns.update(*update_psionly) #computes PSI and O
                    tns_perSample.setTensor('Yobserved', values[sample,...], target_space_observations_indices )          
                    tns_perSample.update()
                    #aggregate per-sample tensors into the per-trajectory observation tensors:
                    tns_perObservation.setTensorSlice('PSIhat', {'samples': i}, 'PSImasked',  slice_namespace=tns_perSample)
                    tns_perObservation.setTensorSlice('Yhat',   {'samples': i}, 'Yhatslice',  slice_namespace=tns_perSample)
                    

        #now do expectation-maximization:
        relative_ll_changes = 1.0
        
        self.tns.setTensor('Wmean', 10.0)
        self.tns.setTensorToIdentity('Wcov', scale=1000.0)  #start with a very large variance - observations will only propagate this value in their nullspace
        negLLHistory = []
        [tns_local.update() for tns_local in tns_Observations]#do precomputation
        
        #check if the projection has full row rank; if not, warn the user:
        PP_ranks = [_np.linalg.matrix_rank(tns_local['PP'].data_flat) for tns_local in tns_Observations]
        if any( [r < self.tns['Wmean'].data.size for r in PP_ranks]):
            print("Danger, Will Robinson!")
            self.PP_ranks = PP_ranks
        
        for iteration_count in range(max_iterations):
            iteration_count = iteration_count+1        
            
            #compute errors in W-space from Wmean and Wcov:
            #pool.map(_estimation_step, tns_Observations)
            #map(_estimation_step, tns_Observations)
            [tns_local.update(*tns_local.lazyupdate) for tns_local in tns_Observations]

            #re-estimate Wmean and Wcov
            deltaW = _np.mean([tns_perObservation['Werror'].data for tns_perObservation in tns_Observations], axis=0)
            deltaW_indices = tns_Observations[0]['Werror'].index_tuples

            Wcov = _np.mean([tns_perObservation['Wcovestimate'].data for tns_perObservation in tns_Observations], axis=0)
            Wcov_indices = tns_Observations[0]['Wcovestimate'].index_tuples
            
            self.tns.addToTensor('Wmean', deltaW, deltaW_indices)
            self.tns.setTensor('Wcov', Wcov, Wcov_indices)
            self.tns.setTensor('Wcov', Wcov, Wcov_indices)

            #mask out parameters that we dont learn:
            if mask != None:                   
                mask_ = []
                for indices in mask:
                    indices_ = {}
                    for index in indices:
                        indices_[index+'_'] = indices[index]
                    mask_.append(indices_)
                    
                for indices in mask_:
                    self.tns.setTensorSlice('Wcov', indices, 0.0)
                for indices in mask:
                    self.tns.setTensorSlice('Wcov', indices, 0.0)
                    self.tns.setTensorSlice('Wmean', indices, 0.0)


            rms = _np.sqrt(_np.mean(deltaW * deltaW))
            negLLHistory.append(rms)

            if rms < minimal_relative_improvement:
                print("Converged Wmean after {} iterations".format(iteration_count))
                break
        else:
                print("Residual mean error (RMS) after {} iterations: {}".format(iteration_count, rms))
                
        durations = _np.array([times[-1]-times[0] for(times, phases, values, Xrefs, Yrefs, Ts) in observations_converted])
        self.expected_duration = _np.mean(durations)
        self.expected_duration_sigma = _np.std( durations - self.expected_duration)
        
        
        #estimate a phase profile too:
        n_total = _np.sum([phases.shape[0] for times, phases, values, Xrefs, Yrefs, Ts in observations_converted])
        xy = _np.empty((n_total, 2))
        current_i = 0
        for times, phases, values, Xrefs, Yrefs, Ts in observations_converted:
            next_i  = current_i + phases.shape[0]
            duration = (times[-1]-times[0]) #this is a heuristic - we could also optimize this to improve phase alignments
            xy[current_i:next_i,0] = times / duration
            xy[current_i:next_i,1] = phases[:,0]
            current_i  = next_i
            
        a,b,error,iterations = _kumaraswamy.approximate(xy, accuracy=1e-4, max_iterations=10000)
        if error > 1e-2:
            raise RuntimeWarning("Large error when estimating phase profile: {}".format(error))        
        self.expected_phase_profile_params = (a,b)
        
        self.observed_phaseprofiles = xy
        
        self.observedTrajectories = observations #remember the data we learned from, for plotting etc.
        self.negLLHistory = negLLHistory
        



# parallelizable function for EM learning. Unfortunately, we need to place it outside of the class:
def _estimation_step(tns_local):
    tns_local.update(*tns_local.equationsForEstimation)
    return tns_local



def _updateP(tns, in_tensor_names, out_tensor_names):
    """
    function to compute a P tensor
    
    update the tensor mapping from phase-derivatives to time-derivatives
    Basically, it implements Faa di Bruno's algorithm
    """
    if tns['gtilde'].size > 4 or tns['g'].size > 4 or tns['gphi'].size > 4:
        raise NotImplementedError()
        
    P = tns[out_tensor_names[0]].data
    phase_fdb=_np.zeros((4))
    phase_fdb[:tns['gphi'].size] = tns[in_tensor_names[0]].data[:tns['gphi'].size] #only use as many derivatives as specified
    
    
    #compute the scaling factors according to Faa di Brunos formula
    #0th, 1st and 2nd derivative:
    faadibruno = _np.zeros(((4,4)))
    faadibruno[0,0] = 1.0        
    faadibruno[1,1] = phase_fdb[1]
    faadibruno[2,2] = phase_fdb[1]**2
    faadibruno[2,1] = phase_fdb[2]

    if tns['g'].size > 3:  #is probably not used
        faadibruno[3,3] = phase_fdb[1]**3
        faadibruno[3,2] = 3*phase_fdb[1]*phase_fdb[2]
        faadibruno[3,1] = phase_fdb[3]
    
    #copy them into P, but shifted for/by gtilde
    P = tns['P'].data
    for gtilde in range(tns['gtilde'].size):
        g_start = gtilde
        fdb_g_end = tns['g'].size - g_start
        if fdb_g_end > 0:
            #index order: g, gphi, gtilde
            P[g_start:,:,gtilde] = faadibruno[:fdb_g_end,:tns['gphi'].size]





#default color map used for correlations/covariances plots:
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

