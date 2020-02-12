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
                  self.tns.registerIndex(name, self.index_sizes[name], values = list(range(self.index_sizes[name])))

        #get a mapping from human-readable names to indices of m-state distributions
        msd_reference = MechanicalStateDistribution(index_sizes)
        self.readable_names_to_realm_derivative_indices = msd_reference.readable_names_to_realm_derivative_indices

        self._gain_names = set()
        for gain_name in ('kp', 'kv'):
            if gain_name in self.readable_names_to_realm_derivative_indices:
                self._gain_names.add('kp')
        
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
            PHI_computer=_trajectorycompositors.TrajectoryCompositorGaussian(),   #Currently, we have only one method
            self.tns.registerExternalFunction(PHI_computer.update, ['phase'], ['PHI'],  [(('gphi',),('stilde',))])
        else:
            raise NotImplementedError()        

        #set function to compute chain derivatives (P tensor):
        self.tns.registerExternalFunction(_updateP, ['phase'], ['P'],  [(('g',),('gphi','gtilde'))] )
        
        #set up tensors involved in taskspace mapping:
        self.tns.registerTensor('Yref', (('r','d','g',),()) )     
        self.tns.registerTensor('Xref',  (('rtilde', 'dtilde', 'g'),()) )    
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
        
        self.tns.registerAddition('PSI:Wmean', 'O', result_name='Ymean')
        
        #project the covariances of weights to covariances of trajectory:
        self.tns.registerContraction('PSI', 'Wcov')
        self.tns.registerContraction('PSI:Wcov', '(PSI)^T', result_name='Ycov')

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
        self.tns.tensorDataAsFlattened['Wsample'][:,0] = _np.random.multivariate_normal(self.tns.tensorDataAsFlattened['Wmean'][:,0], self.tns.tensorDataAsFlattened['Wcov'])
        return self.tns.tensorData['Wsample']



    def getDistribution(self, * , generalized_phase=None, current_msd=None, task_spaces={}):
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

        return _mechanicalstate.MechanicalStateDistribution(self.tns.tensorData['Ymean'], self.tns.tensorData['Ycov'])






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
            points_list.append(self.getDistribution(generalized_phase=generalized_phases[i]).means)
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
        observedColor = '#880000'
        kpColor = '#000000'
        kvColor = '#000000'
        kpCrossColor = '#666666'
        kvCrossColor = '#666666'

        phases = _np.zeros((num, self.tns.indexSizes['gphi']))
        
        if useTime: 
            times = _np.linspace(0, self.expected_duration, num)     
            phases[:,0] = _kumaraswamy.cdf(self.expected_phase_profile_params[0], self.expected_phase_profile_params[1], _np.linspace(0,1.0,num))
            plot_x = times
        else:
            phases[:,0] = _np.linspace(0.0, 1.0, num)
            plot_x = phases[:,0]
            
        if self.tns.indexSizes['gphi'] > 1:
            if useTime:
                phases[:,1] = _kumaraswamy.pdf(self.expected_phase_profile_params[0], self.expected_phase_profile_params[1], _np.linspace(0,1.0,num))
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

        if dofs=='all' or dofs == None:
            if self.tns.indexValues['d'] != None:
                dofs_to_plot=self.tns.indexValues['d']
            else:
                dofs_to_plot=list(range(self.tns.indexSizes['d']))
        else:
            dofs_to_plot = dofs
        subplotfigsize=2.0

        plotrows = len(whatToPlot)
            
        #gather the data:
        data_mean  = _np.zeros((num,plotrows, self.tns.indexSizes['d']))
        data_sigma = _np.zeros((num,plotrows, self.tns.indexSizes['d']))
        data_gains = {
            'kp': _np.zeros((num,self.tns.indexSizes['d'], self.tns.indexSizes['d'])),
            'kv': _np.zeros((num,self.tns.indexSizes['d'], self.tns.indexSizes['d'])),
        }
        generalized_phase =_np.zeros((self.tns.indexSizes['g']))            
        for i,phase in enumerate(phases):
            dist =  self.getDistribution(generalized_phase=phase)
            for row_idx, row_name in enumerate(whatToPlot):
                if row_name in data_gains:
                    gains  = dist.extractPDGains()
                    g_idx, g2_idx = self.readable_names_to_realm_derivative_indices[row_name]
                    data_gains[row_name][i,:,:] = gains[:,g_idx,:,g2_idx]
                else:                
                    r_idx, g_idx = self.readable_names_to_realm_derivative_indices[row_name]                
                    data_mean[i,row_idx,:] = dist.means[r_idx,:,g_idx] 
                    data_sigma[i,row_idx,:] = _np.sqrt( dist.variancesView[r_idx,:,g_idx] )
        

        fig, axesArray = _plt.subplots(plotrows,len(dofs_to_plot), squeeze=False, figsize=(max(len(dofs_to_plot), plotrows)*subplotfigsize, plotrows*subplotfigsize), sharex='all', sharey='row')
        for ax in axesArray.flat:
            ax.margins(x=0.0, y=margin)
            
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

        #sample the distribution to plot actual trajectories, times the number given by "addExampleTrajectories":
        if addExampleTrajectories is None:
            addExampleTrajectories = 0
        alpha = _np.sqrt(2.0 / (1+addExampleTrajectories))
        for ax in axesArray.flatten():
            ax.set_prop_cycle(exampleTrajectoryStyleCycler)

        for j in range(addExampleTrajectories):
            yvalues = self.sampleTrajectory(phases)
            for row_idx, row_name in enumerate(whatToPlot):
                if row_name in self._gain_names:
                    continue
                r_idx, g_idx = self.readable_names_to_realm_derivative_indices[row_name]
                for i, dof in enumerate(dofs_to_plot):
                    axesArray[row_idx,col_idx].plot(plot_x, yvalues[:,r_idx,dof,g_idx], alpha=alpha, linewidth=linewidth )
        


        #set ylimits to be equal in each row and xlimits to be equal everywhere
        for row_idx, row_name in enumerate(whatToPlot):
                axes_row = axesArray[row_idx,:]  
                ylimits  = _np.array([ ax.get_ylim() for ax in axes_row ])
                ylimit_common = _np.max(ylimits, axis=0)
                ylimit_common_symm = _np.max(_np.abs(ylimits))
                ylimit_common = (-ylimit_common_symm, ylimit_common_symm) 
                for dof, ax in zip(dofs_to_plot, axes_row):
                    ax.set_title(r"{0} {1}".format(row_name, dof))
                    ax.set_ylim(ylimit_common)                
                    ax.get_yaxis().set_visible(False)
                    yticks = [ ylimit_common[0], 0.0, ylimit_common[1] ]                    
                    yticklabels = ['{0:0.1f}'.format(ylimit_common[0]), '{}'.format(units[row_name]), '{0:0.1f}'.format(ylimit_common[1])]
                    yticks, yticklabels = tuple(zip(*sorted(zip(yticks, yticklabels))))
                    ax.set_yticks(yticks)
                    ax.set_yticklabels(yticklabels)

        if 'observedTrajectories' in self.__dict__: #plot observations after scaling y axes
            for observation_idx, (times, phases_observation, values, Xrefs, Yrefs, Ts) in enumerate(self.observedTrajectories):        
                for row_idx, row_name in enumerate(whatToPlot):
                    if row_name in self._gain_names:
                        continue
                    r_idx, g_idx = self.readable_names_to_realm_derivative_indices[row_name]
                    if useTime:
                        y = values[:,r_idx,:,g_idx]
                        for col_idx, dof in enumerate(dofs_to_plot):
                            axesArray[ row_idx, col_idx].plot(times, y[:,dof], alpha=alpha, linewidth=linewidth, color=observedColor )
                    else:
                        if phases_observation.shape[1] > g_idx:   #only plot in phase if we have phase derivatives to scale observations with 
                            if g_idx > 0:
                                scaler = 1.0 / phases_observation[:,1]
                                y = values[:,r_idx,:,g_idx] * (scaler[:,None]**g_idx)
                            else:
                                y = values[:,r_idx,:,g_idx]
                            for col_idx, dof in enumerate(dofs_to_plot):
                                axesArray[ row_idx, col_idx].plot(phases_observation[:,0], y[:,dof], alpha=alpha, linewidth=linewidth, color=observedColor )




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


    def plotCovarianceTensor(self, normalize_indices='rg'):
        """
        plot the covariances/correlations in parameter space
        
        normalize_indices: string of index letters used to select which dimension to normalize
            'rgsd': all indices (correlation matrix)
            '': verbatim covariance matrix
            ''rg': variances between realms and between derivatives are normalized (default)
        """
        cov = self.tns.tensorData['Wcov']
        variance_view = _np.einsum('ijklijkl->ijkl', self.tns.tensorData['Wcov'])

        sigmas = _np.sqrt(variance_view)

        firstletters = [string[0] for string in self.tns.tensorIndices['Wcov'][0]]
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

        cov_reordered =_np.transpose(cov_scaled, axes=(2,0,1,3, 2+4,0+4,1+4,3+4)) #to srgd
        image =_np.reshape(cov_reordered, self.tns.tensorShapeFlattened['Wcov'])
        gridvectorX = _np.arange(0, image.shape[0], 1)
        gridvectorY = _np.arange(image.shape[1], 0, -1)

        fig = _plt.figure(figsize=(3.4,3.4))
        _plt.pcolor(gridvectorX, gridvectorY, image, cmap=_cmapCorrelations, vmin=-vmax, vmax=vmax)
        
        _plt.axis([0, image.shape[0], 0, image.shape[1]])
        _plt.gca().set_aspect('equal', 'box')

        len_all = self.tns.tensorShapeFlattened['Wcov'][0]
        len_rtilde = self.tns.indexSizes['rtilde']
        len_stilde = self.tns.indexSizes['stilde']
        len_dtilde = self.tns.indexSizes['dtilde']
        len_gtilde = self.tns.indexSizes['gtilde']
        line_positions = _np.reshape(_np.arange(self.tns.tensorShapeFlattened['Wcov'][0]), cov_reordered.shape[:4])
        for r_idx in range(len_rtilde):
          for g_idx in range(len_gtilde):
            for s_idx in range(len_stilde):
                for d_idx in range(len_dtilde):
                    linewidth=0.5
                    linestyle='-'
                    if r_idx!=0:
                        linewidth=0.2
                        linestyle='-'
                    if g_idx!=0:
                        linewidth=0.2
                        linestyle=':'
                    if d_idx!=0:
                        linewidth=0.0
                    if linewidth>0.0:
                        _plt.axhline(line_positions[s_idx,r_idx, g_idx,d_idx], color='k', linewidth=linewidth, linestyle=linestyle)
                        _plt.axvline(line_positions[s_idx,r_idx, g_idx,d_idx], color='k', linewidth=linewidth, linestyle=linestyle)

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
                    ticklabels.append(g)
                    offsets.append(-1.0)
        for tick, label, offset in zip(ticks, ticklabels, offsets):
            t = _plt.text(tick, offset, label, fontdict={'verticalalignment':'top', 'horizontalalignment':'center', 'size':'xx-small'}, rotation=0)
        _plt.text(ticks[-1]+10, 0.0, "$\widetilde{g}$", fontdict={'verticalalignment':'top', 'horizontalalignment':'left', 'size':'small'})
            
        _plt.xticks([])


        _plt.colorbar(shrink=0.6, aspect=40, ticks=[-vmax,0,vmax], fraction=0.08)
        _plt.title(title)
        ax = _plt.gca()        
        #_plt.tight_layout()


    def plotExpectedPhase(self):
        fig = _plt.figure()
        x = _np.linspace(0.0, 1.0, 200)
        y = _kumaraswamy.cdf(self.expected_phase_profile_params[0],self.expected_phase_profile_params[1],x)
        _plt.plot(self.expected_duration*x,y)
        _plt.xlabel('Time')
        _plt.ylabel('Phase')
        if 'observed_phaseprofiles' in self.__dict__:    
            _plt.plot(self.expected_duration*self.observed_phaseprofiles[:,0], self.observed_phaseprofiles[:,1], '.', color='black')        
        _plt.tight_layout()
        return fig

#    def plotLearningProgress(self):
#        """
#        plot a graph of the learning progress, as determined by the negative log-likelihood over the training data
#        """
#        if not 'negLLHistory' in self.__dict__:
#            raise RuntimeWarning("No learning has happened - cannot plot progress.")
#        _plt.figure()
#        _plt.plot(self.negLLHistory)
#        _plt.xlabel("Iteration")
#        _plt.ylabel("-log( p(data|W) )")
#        _plt.ylim(0, 10 * _np.mean(self.negLLHistory[-10:]))

        
    def learnFromObservations(self, 
            observations,
            max_iterations=10, 
            minimal_relative_improvement=1e-9,
            ):
        """
        
        compute parameters from a list of observations
        
        Observations are tuples of  (times, phases, values, Xrefs, Yrefs, Ts) tensors (data arrays)
        
        phases: Generalized phases of shape (n, 'd')
        means: observed values at the given phases, of shape (n, 'd', 'g')
        Xref, Yref: Task space linearization references, of shape (n, 'rtilde', 'dtilde', 'gtilde', 'stilde' ) and (n, 'r', 'd', 'g') respectively
        #
       
        Implements the expectation-maximization procedure of the paper "Using probabilistic movement primitives in robotics" by Paraschos et al.
        """
        pool = _multiprocessing.Pool()        
        
        tns_perSample = self.tns.copy()
        #truncate equations that we do not need to compute:
        tns_perSample.update_order = tns_perSample.update_order[:tns_perSample.update_order.index('PSI')+1]
        #subtract the offset caused by the task map linearization when computing the per-sample data, so during learning we ignore Xref and Yref
        tns_perSample.registerTensor('Yobserved', (('r','d','g'),()) )
        tns_perSample.registerSubtraction('Yobserved', 'O', 'Yhatslice')
        
        self.tns_perSample = tns_perSample
        #we only need to compute until we have PSI:
        tns_Observations =[]
        self.tns_Observations = tns_Observations

        #"precompute" the mapping from joint space to joint space:
        T_jointspace =_np.eye( (self.tns.indexSizes['r']*self.tns.indexSizes['d']) )
        T_jointspace.shape = (self.tns.indexSizes['r'],self.tns.indexSizes['d'],self.tns.indexSizes['rtilde'],self.tns.indexSizes['dtilde']) #r,d,rtilde, dtilde
        
        for observation_idx, observation in enumerate(observations):            
            import pandas as _pandas
            if observation.isinstance(x, _pandas.DataFrame):
                samples_total = len(observation.index)
                data_shape = (samples, self.tns.indexSizes['r'],self.tns.indexSizes['d'],self.tns.indexSizes['g'])
                values = _np.zeros(data_shape)
                for name in self.readable_names_to_realm_derivative_indices: 
                    r_idx, g_idx = self.readable_names_to_realm_derivative_indices[name]
                    for d_idx in range(8):
                        values[:, r_idx, d_idx, g_idx] = observation[ ('observed', name, d_idx) ]
                times = observation['t']
                phases = observation[['phi','dphidt','ddphidt2']]
                #setup for joint-space learning:
                Yrefs = _np.zeros(data_shape)   # or observation[ ('observed', 'position') ]
                Xrefs = _np.zeros(data_shape)   # or observation[ ('observed', 'position') ]
                Ts = _np.tile(T_jointspace, (samples,1,1,1,1))
            else: #old interface: try to interprete it as tuple of arrays:
                times, phases, values, Xrefs, Yrefs, Ts = observation
                samples_total = phases.shape[0]

            #compute how we partition the observation samples into subsets to reduce computational effort and increase the number of w samples available for covariance estimation:
            samplesetsize  = self.tns.indexSizes['stilde'] * 3  #no need to add much more samples than we have interpolation parameters over time
            partitions = samples_total // samplesetsize
            #construct an array of shuffled indices:
            sampleindices = _np.arange(partitions * samplesetsize)
            _np.random.shuffle(sampleindices)
            sampleindices.shape = (partitions, samplesetsize)


            for partition in range(partitions):
                
                tns_perObservation = _namedtensors.TensorNameSpace(self.tns)
                self.tns_Observations.append(tns_perObservation)            
                
                tns_perObservation.registerIndex('samples', samplesetsize)        

                tns_perObservation.registerTensor('Yhat', (('samples','r', 'd', 'g'),()))
                tns_perObservation.registerTensor('PSIhat', (('samples','r', 'd', 'g'), ('rtilde', 'dtilde','gtilde', 'stilde')))
                tns_perObservation.registerTensor('Wmean', self.tns.tensorIndices['Wmean'])                     
                tns_perObservation.registerTensor('Wcov', self.tns.tensorIndices['Wcov'])
                
                #precomputatable:
                tns_perObservation.registerInverse('PSIhat', flip_underlines=True)
                tns_perObservation.registerContraction( '(PSIhat)^#', 'PSIhat', result_name='PP')
                tns_perObservation.registerTensor('I_PP', tns_perObservation.tensorIndices['PP'], initial_values='identity') 
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
                for i, sample in enumerate(sampleindices[partition,:]):
                    #compute PSIi:
                    tns_perSample.setTensor('Xref',       Xrefs[sample,...], (('rtilde', 'dtilde', 'g'),()) ) 
                    tns_perSample.setTensor('Yref',       Yrefs[sample,...], (('r', 'd', 'g'),()) )                
                    tns_perSample.setTensor('T',             Ts[sample,...], (('r', 'd'),('rtilde', 'dtilde')) )
                    tns_perSample.setTensor('phase',     phases[sample,...], (('g'),()) )
                    tns_perSample.setTensor('Yobserved', values[sample,...], (('r', 'd', 'g'),()) )          
                    tns_perSample.update()
                    slice_indices = (tns_perObservation.tensorIndices['PSIhat'][0][1:],tns_perObservation.tensorIndices['PSIhat'][1])
                    psi_aligned = tns_perObservation._alignDimensions(slice_indices, tns_perSample.tensorIndices['PSI'], tns_perSample.tensorData['PSI'])
                    _np.copyto(tns_perObservation.tensorData['PSIhat'][i,...], psi_aligned) #aggregate data into PSIhat
                    _np.copyto(tns_perObservation.tensorData['Yhat'][i,...], tns_perSample.tensorData['Yhatslice']) #aggregate values into Yhat


        #now do expectation-maximization:
        relative_ll_changes = 1.0
        
        self.tns.setTensor('Wmean', 10.0)
        self.tns.setTensorToIdentity('Wcov', scale=100)
        negLLHistory = []
        [tns_local.update() for tns_local in tns_Observations]#do precomputation
        
        #check if the projection has full row rank; if not, warn the user:
        PP_ranks = [_np.linalg.matrix_rank(tns_local.tensorDataAsFlattened['PP']) for tns_local in tns_Observations]
        if any( [r < self.tns.tensorData['Wmean'].size for r in PP_ranks]):
            print("Danger, Will Robinson!")
            self.PP_ranks = PP_ranks
        
        for iteration_count in range(max_iterations):
            iteration_count = iteration_count+1        
            
            #set priors for each observation estimator:
            for tns_perObservation in  tns_Observations:
                tns_perObservation.setTensor('Wmean', self.tns.tensorData['Wmean'], self.tns.tensorIndices['Wmean'])
                tns_perObservation.setTensor('Wcov',  self.tns.tensorData['Wcov'], self.tns.tensorIndices['Wcov'])
                
            #compute most likely values: (E-step)
            #pool.map(_estimation_step, tns_Observations)
            #map(_estimation_step, tns_Observations)
            [tns_local.update(*tns_local.lazyupdate) for tns_local in tns_Observations]

            #maximize mean based on likely values  (M-step)
            deltaW = _np.mean([tns_perObservation.tensorData['Werror'] for tns_perObservation in tns_Observations], axis=0)
            deltaW_indices = tns_Observations[0].tensorIndices['Werror']

            Wcov = _np.mean([tns_perObservation.tensorData['Wcovestimate'] for tns_perObservation in tns_Observations], axis=0)
            Wcov_indices = tns_Observations[0].tensorIndices['Wcovestimate']
            
            self.tns.addToTensor('Wmean', deltaW, deltaW_indices)
            self.tns.setTensor('Wcov', Wcov, Wcov_indices)
            
            rms = _np.sqrt(_np.mean(deltaW * deltaW))
            negLLHistory.append(rms)

            if rms < minimal_relative_improvement:
                print("Converged Wmean after {} iterations".format(iteration_count))
                break
        else:
                print("Residual mean error (RMS) after {} iterations: {}".format(iteration_count, rms))
                
        self.expected_duration = _np.mean([times[-1] for(times, phases, values, Xrefs, Yrefs, Ts) in observations])
        
        
        #estimate a phase profile too:
        n_total = _np.sum([phases.shape[0] for times, phases, values, Xrefs, Yrefs, Ts in observations])
        xy = _np.empty((n_total, 2))
        current_i = 0
        for times, phases, values, Xrefs, Yrefs, Ts in observations:
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
        

#    def learnFromObservationsEM(self, 
#            observations,
#            max_iterations=100, 
#            minimal_relative_improvement=1e-4,
#            minimal_absolute_negll=1e-9,
#            ):
#        """
#        
#        compute parameters from a list of observations
#        
#        Observations are tuples of (times, phases, values, Xrefs, Yrefs) tensors (data arrays)
#        
#        phases: Generalized phases of shape (n, 'd')
#        means: observed values at the given phases, of shape (n, 'd', 'g')
#        Xref, Yref: Task space linearization references, of shape (n, 'rtilde', 'dtilde', 'gtilde', 'stilde' ) and (n, 'r', 'd', 'g') respectively
#        #

#        useonly: only use the combination of (realm, derivative) to learn, assume that in all other combinations data are not trustworthy
#       
#        stabilize: value in range 0-1, 0 does does full EM updates, nonzero values slow down convergence but improve stability
#       
#        Implements the expectation-maximization procedure of the paper "Using probabilistic movement primitives in robotics" by Paraschos et al.
#        """
#        pool = _multiprocessing.Pool()        
#        
#        sample_subsetsize = 8
#        
#        tns_perSample = self.tns.copy()
#        # observations may not have used the task space computer - set T and Xref from data
#        tns_perSample.update_order.remove('T,Xref')
#        #truncate equations that we do not need to compute:
#        tns_perSample.update_order = tns_perSample.update_order[:tns_perSample.update_order.index('PSI')+1]
#        
#        self.tns_perSample = tns_perSample
#        #we only need to compute until we have PSI:
#        tns_Observations =[]
#        self.tns_Observations = tns_Observations

#        for observation_idx, (times, phases, values, Xrefs, Yrefs, Ts) in enumerate(observations):
#            #samplesubset = _np.arange(phases.shape[0])
#            for j in range(10):
##                samplesubset = _np.arange(0, phases.shape[0], phases.shape[0]//5)
#                samplesubset = _np.random.choice(phases.shape[0], sample_subsetsize, replace=False)
#                
#                tns_perObservation = _namedtensors.TensorNameSpace(self.tns)
#                tns_Observations.append(tns_perObservation)            
#                #tns_perObservation.registerIndex('i', phases.shape[0])        
#                tns_perObservation.registerIndex('samples', sample_subsetsize)        

#                tns_perObservation.registerTensor('Yhat', (('samples','r', 'd', 'g'),()))
#                tns_perObservation.registerTensor('PSIhat', (('samples','r', 'd', 'g'), ('rtilde', 'dtilde','gtilde', 'stilde')))
#                tns_perObservation.registerTensor('Wmean', self.tns.tensorIndices['Wmean'])
#                tns_perObservation.registerTensor('Wcov', self.tns.tensorIndices['Wcov'], initial_values='identity')

#                #likelihood estimators:
#                tns_perObservation.registerTranspose('PSIhat')
#                tns_perObservation.registerContraction('PSIhat', 'Wcov')
#                tns_perObservation.registerContraction('PSIhat:Wcov', '(PSIhat)^T', result_name='Ycov')
#                tns_perObservation.registerInverse('Ycov', result_name='Ycovinv', flip_underlines=False)
#                tns_perObservation.registerContraction('(PSIhat)^T', 'Ycovinv')
#                tns_perObservation.registerContraction('(PSIhat)^T:Ycovinv','PSIhat', result_name='Wcovinv')

#                #estimate likely mean:
#                tns_perObservation.registerContraction('PSIhat', 'Wmean')
#                tns_perObservation.registerSubtraction('Yhat', 'PSIhat:Wmean', result_name='Yerror')
#                tns_perObservation.registerContraction('Ycovinv', 'Yerror')
#                tns_perObservation.registerContraction('(PSIhat)^T', 'Ycovinv:Yerror', flip_underlines=True, result_name='deltaWmeanestimate')
#                tns_perObservation.registerAddition('Wmean', 'deltaWmeanestimate', result_name='Wmeanestimate')
#                #estimate likely covariance:
#                tns_perObservation.registerContraction('Wcovinv', 'Wcov')            
#                tns_perObservation.registerTranspose('Wcov')
#                tns_perObservation.registerContraction('(Wcov)^T', 'Wcovinv:Wcov')           
#                tns_perObservation.registerSubtraction('Wcov', '(Wcov)^T:Wcovinv:Wcov', result_name='Wcovestimate') 
#                
#                #compute negative-log-likelihood increment to judge progress:
#                tns_perObservation.registerTranspose('Yerror')
#                tns_perObservation.registerContraction('(Yerror)^T', 'Ycovinv:Yerror' , result_name ='negLLDelta')


#                tns_perObservation.equationsForEstimation = tns_perObservation.update_order.copy()
#                tns_perObservation.update_order = []

#                tns_perObservation.registerTensor('deltaWmeanprime', tns_perObservation.tensorIndices['Wmeanestimate'])

#                tns_perObservation.registerSubtraction('deltaWmeanestimate', 'deltaWmeanprime', result_name='deltaWprime')
#                tns_perObservation.registerTranspose('deltaWprime')
#                
#                tns_perObservation.registerContraction('deltaWprime', '(deltaWprime)^T', result_name='Wcovempirical')
#                #tns_perObservation.registerAddition('Wcovempirical', 'Wcovestimate', 'Wcovprime' ) 
#                tns_perObservation.registerSubtraction('Wcovempirical', '(Wcov)^T:Wcovinv:Wcov', result_name='deltaWcovprime' ) 
#                tns_perObservation.equationsForWcovprime = tns_perObservation.update_order.copy()

#                #preprocess samples into pairs of Yhat and PSIhat for each observation:
#                for i, sample in enumerate(samplesubset):
#                    #compute PSIi:
#                    tns_perSample.setTensor('T', Ts[sample,:,:,:,:], (('r', 'd'),('rtilde', 'dtilde')) )
#                    tns_perSample.setTensor('Xref', Xrefs[sample,:,:,:], (('rtilde', 'dtilde', 'g'),()) ) 
#                    tns_perSample.setTensor('phase', phases[sample,:], (('g'),()) )
#                    tns_perSample.setTensor('Yref', Yrefs[sample,:,:,:], (('r', 'd', 'g'),()) )                
#                    tns_perSample.update()
#                    slice_indices = (tns_perObservation.tensorIndices['PSIhat'][0][1:],tns_perObservation.tensorIndices['PSIhat'][1])
#                    psi_aligned = tns_perObservation._alignDimensions(slice_indices, tns_perSample.tensorIndices['PSI'], tns_perSample.tensorData['PSI'])
#                    _np.copyto(tns_perObservation.tensorData['PSIhat'][i,...], psi_aligned) #aggregate data into PSIhat
#                    _np.copyto(tns_perObservation.tensorData['Yhat'][i,...], values[i,...]) #aggregate values into Yhat

#        

#        #now do expectation-maximization:
#        relative_ll_changes = 1.0
#        
#        self.tns.setTensor('Wmean', 0.0)
#        self.tns.setTensorToIdentity('Wcov', scale=100)
#        negLLHistory = []
#        for iteration_count in range(max_iterations):
#            iteration_count = iteration_count+1        
#            
#            #set priors for each observation estimator:
#            for tns_perObservation in  tns_Observations:
#                tns_perObservation.setTensor('Wmean', self.tns.tensorData['Wmean'], self.tns.tensorIndices['Wmean'])
#                tns_perObservation.setTensor('Wcov',  self.tns.tensorData['Wcov'], self.tns.tensorIndices['Wcov'])
#                
#            #compute most likely values: (E-step)
#            #pool.map(_estimation_step, tns_Observations)
#            #map(_estimation_step, tns_Observations)
#            [tns_local.update(*tns_local.equationsForEstimation) for tns_local in tns_Observations]

#            #maximize mean based on likely values  (M-step)
#            deltaWmeanprime = _np.mean([tns_perObservation.tensorData['deltaWmeanestimate'] for tns_perObservation in tns_Observations], axis=0)
#            deltaWmeanprime_indices = tns_Observations[0].tensorIndices['deltaWmeanestimate']

#            negLLDeltaSum = _np.mean([tns_perObservation.tensorData['negLLDelta'] for tns_perObservation in tns_Observations], axis=0)
#            print(negLLDeltaSum)

#            #maximize covariance based on likely values   (M-step 2nd part)
#            for tns_perObservation in tns_Observations:
#                tns_perObservation.setTensor('deltaWmeanprime', deltaWmeanprime, deltaWmeanprime_indices)
#                tns_perObservation.update(*tns_perObservation.equationsForWcovprime)
#                print(tns_perObservation.tensorDataAsFlattened['deltaWprime'])
#            
#            deltaWcovprime = _np.mean([tns_perObservation.tensorData['Wcovempirical'] for tns_perObservation in tns_Observations], axis=0)
#            deltaWcovprime_indices = tns_Observations[0].tensorIndices['Wcovempirical']

#            #Wcovprime = 0.5 * ( Wcovprime + _np.transpose(Wcovprime, axes=(4,5,6,7,0,1,2,3)))
#            #save new maximized parameters:
#            self.tns.addToTensor('Wmean', deltaWmeanprime, deltaWmeanprime_indices)
#            self.tns.addToTensor('Wcov', deltaWcovprime, deltaWcovprime_indices)
#            print(_np.max(deltaWcovprime), _np.min(deltaWcovprime))
#            
#            negLLHistory.append(negLLDeltaSum)
#            
#            #terminate early if neg-log-likelihood of observations stops increasing:
#            #if negLLDeltaSum < minimal_absolute_negll:
#            #    print("stopping early at iteration {}".format(iteration_count))            
#            #    break
#            n= 4
#            if len(negLLHistory) > n:
#                relative_ll_changes = [abs((negLLHistory[-i]-negLLHistory[-i-1])/negLLHistory[-i-1]) for i in range(1,n) ]
#                eigenvalues = _np.linalg.eigvals(self.tns.tensorDataAsFlattened['Wcov'])
#                if _np.any(_np.iscomplex(eigenvalues)) or _np.any(eigenvalues < 0.0):                    
#                    print("Warning: At iteration {} eigenvalues became {} ".format(iteration_count, eigenvalues))
#                print("change: {:.4}".format(relative_ll_changes[0]))
#                if max(relative_ll_changes) < minimal_relative_improvement:
#                    print("stopping early at iteration {}".format(iteration_count))
#                    break
#                    
#        self.negLLHistory = _np.array(negLLHistory)
#        self.expected_duration = _np.mean([times[-1] for(times, phases, values, Xrefs, Yrefs, Ts) in observations])

#        self.observedTrajectories = observations #remember the data we learned from, for plotting etc.



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
    if tns.indexSizes['gtilde'] > 4 or tns.indexSizes['g'] > 4 or tns.indexSizes['gphi'] > 4:
        raise NotImplementedError()
        
    P = tns.tensorData[out_tensor_names[0]]
    phase_fdb=_np.zeros((4))
    phase_fdb[:tns.indexSizes['gphi']] = tns.tensorData[in_tensor_names[0]]
    
    
    #compute the scaling factors according to Faa di Brunos formula
    #0th, 1st and 2nd derivative:
    faadibruno = _np.zeros(((4,4)))
    faadibruno[0,0] = 1.0        
    faadibruno[1,1] = phase_fdb[1]
    faadibruno[2,2] = phase_fdb[1]**2
    faadibruno[2,1] = phase_fdb[2]

    if tns.indexSizes['g'] > 3:  #is probably not used
        faadibruno[3,3] = phase_fdb[1]**3
        faadibruno[3,2] = 3*phase_fdb[1]*phase_fdb[2]
        faadibruno[3,1] = phase_fdb[3]
    
    #copy them into P, but shifted for/by gtilde
    P = tns.tensorData['P']  
    for gtilde in range(tns.indexSizes['gtilde']):
        g_start = gtilde
        fdb_g_end = tns.indexSizes['g'] - g_start
        if fdb_g_end > 0:
            #index order: g, gphi, gtilde
            P[g_start:,:,gtilde] = faadibruno[:fdb_g_end,:tns.indexSizes['gphi']]


