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

_psihatinv_func_tnscopy_perprocess = {}


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

        #register all tensors not being computed by operators:
        #i.e. input tensors:
        self.tns.registerTensor('phase', (('g',),()) )  #generalized phase vector        
        self.tns.registerTensor('Wmean', (('rtilde','gtilde','stilde','dtilde'),()) )
        self.tns.registerTensor('Wcov', (( 'rtilde','gtilde','stilde','dtilde'),( 'rtilde_','gtilde_','stilde_','dtilde_')) )
        PHI_computer = PHI_computer_cls() 
        self.tns.registerExternalFunction(PHI_computer.update, ['phase'], ['PHI'],  [(('gphi',),('stilde',))])
        self.tns.registerExternalFunction(_updateP, ['phase'], ['P'],  [(('g',),('gphi','gtilde'))] )

        #tensors involved in taskspace mapping:
        self.tns.registerTensor('Yref', (('r','d','g',),()) )     #T_computer input
        self.tns.registerTensor('Xref', (('rtilde','dtilde','g',),()) ) #T_computer input
        T_computer = T_computer_cls()
        self.tns.registerExternalFunction(T_computer.update, ['Xref', 'Yref'], ['T'],  [(('r','d'),('rtilde', 'dtilde'))] )
        
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
        #Now compute the actual ProMeP equation:
        self.tns.update('P', 'PHI', 'T', 'P:PHI', 'PSI', 'T:Xref', 'O', 'PSI:Wmean', 'Ymean', 'PSI:Wcov', 'Ycov')

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
                   margin=0.05
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
            dist =  self.getDistribution(phase)
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
        
        if 'observedTrajectories' in self.__dict__:
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
                        if g_idx == 1:
                            y = values[:,r_idx,:,g_idx] #/ phases_observation[:,1]
                        else:
                            y = values[:,r_idx,:,g_idx] 
                        for col_idx, dof in enumerate(dofs_to_plot):
                            axesArray[ row_idx, col_idx].plot(phases_observation[:,0], y[:,dof], alpha=alpha, linewidth=linewidth, color=observedColor )



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


    def plotLearningProgress(self):
        """
        plot a graph of the learning progress, as determined by the negative log-likelihood over the training data
        """
        if not 'negLL' in self.__dict__:
            raise RuntimeWarning("No learning has happened - cannot plot progress.")
        _plt.figure()
        _plt.plot(self.negLL)
        _plt.xlabel("Iteration")
        _plt.ylabel("-log( p(data|W) )")
        _plt.ylim(0, 10 * _np.mean(self.negLL[-10:]))

    def learnFromObservations(self, 
            observations,
            max_iterations=150, 
            minimal_relative_improvement=1e-4,
            ):
        """
        
        compute parameters from a list of observations
        
        Observations are tuples of (times, phases, values, Xrefs, Yrefs) tensors (data arrays)
        
        phases: Generalized phases of shape (n, 'd')
        means: observed values at the given phases, of shape (n, 'd', 'g')
        Xref, Yref: Task space linearization references, of shape (n, 'rtilde', 'dtilde', 'gtilde', 'stilde' ) and (n, 'r', 'd', 'g') respectively
        

        useonly: only use the combination of (realm, derivative) to learn, assume that in all other combinations data are not trustworthy
       
        This method estimates likely parameters for each observed trajectory, then estimates a distribution from those parameters
        """
        #pool = _multiprocessing.Pool(8)        
        
        # set up computation of PSIhatInv = (PSIhat)^-1 = (Yref - (Yrefhat - T:Xrefhat) )^-1

        tns_perObservation = self.tns.copy()

        #additional tensors needed:
        tns_perObservation.registerTensor('Oneso', (('o'),()), initial_values='ones')
        tns_perObservation.registerTensor('Iparameter', (('rtilde', 'gtilde', 'stilde', 'dtilde'),('rtilde_', 'gtilde_', 'stilde_', 'dtilde_')), initial_values='identity')

        tns_perObservation.registerInverse('PSI')        
        tns_perObservation.registerContraction('PSI', '(PSI)^-1')
        tns_perObservation.registerSubtraction('Yi', 'O')
        tns_perObservation.registerContraction('(PSI)^-1', '(Yi-O)', result_name='Westimate')
        
        original_tns = self.tns
        self.tns = tns_perObservation
        
        #preprocess observations into pairs of Yi and PSIi
        estimation_parameters = []
        for observation_idx, (times, phases, values, Xrefs, Yrefs, Ts) in enumerate(observations):
            num = phases.shape[0]
            for n in range(num):
                #compute PSIi:
                tns_perObservation.setTensor('phase', phases[n,:], (('g'),()) )
                tns_perObservation.setTensor('Xref', Xrefs[n,:,:,:], (('rtilde', 'dtilde', 'g'),()) ) 
                tns_perObservation.setTensor('Yref', Yrefs[n,:,:,:], (('r', 'd', 'g'),()) )                
                tns_perObservation.setTensor('T', Ts[n,:,:,:,:], (('r', 'd'),('rtilde', 'dtilde')) )
                _updateP(tns_perObservation) #call the code that computes map between phase and time derivatives. implemented directly in this class
                self.PHI_computer.update(tns_perObservation) #call the hook that computes the interpolating map PHI from the phase:
                #compute PSI only:
                self.tns.update('P:PHI', 'PSI', 'O', '(PSI)^-1', 'M')
                ydelta = values[n,:,:,:] #- self.tns.tensorData['O']
                estimation_parameters.append( (tns_learn, tns_learn.tensorData['Wmean'], tns_learn.tensorData['Wcov'], self.tns.tensorData['PSI'].copy(), ydelta) ) #saving a view is enough
        
        
        
#        first_part_steps = len(tns_learn.update_order)  #when computing, stop here for averaging over all computed Wmeani

#        #this is done separately, after Wmean has been updated:  
#        tns_learn.registerSubtraction('Wmeani', 'Wmean')      
#        tns_learn.registerTranspose('(Wmeani-Wmean)')  
#        tns_learn.registerContraction('(Wmeani-Wmean)', '((Wmeani-Wmean))^T', result_name='Wcoviempirical')
#    
        #_pprint.pprint(tns_learn.tensorIndices)
       
    
        iteration_count = 0 #maximum number of E-M iterations
        negLL = []
        tns_learn.setTensorToIdentity('Wcov', scale=100**2) #set the prior
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
                _updateP(self.tns) #call the code that computes map between phase and time derivatives. implemented directly in this class
                self.PHI_computer.update() #call the hook that computes the interpolating map PHI from the phase:
                #compute PSI only:
                self.tns.update('P:PHI', 'PSI', 'O', '(PSI)^-1', 'M')
                ydelta = values[n,:,:,:] #- self.tns.tensorData['O']
                estimation_parameters.append( (tns_learn, tns_learn.tensorData['Wmean'], tns_learn.tensorData['Wcov'], self.tns.tensorData['PSI'].copy(), ydelta) ) #saving a view is enough

        relative_improvement = 1.0
        global _tnscopy_perprocess
        _tnscopy_perprocess.clear()
        relative_changes = 1.0
        while iteration_count < max_iterations:
            iteration_count = iteration_count+1        
            
            subset = estimation_parameters
            stabilize = max(0.0, ((1.0*iteration_count) / max_iterations)-0.5)
            #if we have many data points, and changes are still big, select a subset to speed up iteration:
#            if  len(estimation_parameters) > 0.2*tns_learn.tensorData['Wcov'].size and relative_changes > 0.1:
#                subset = [ _random.choice(estimation_parameters) for i in range(tns_learn.tensorData['Wcov'].size//5)]
#                stabilize = max(0.5, ((1.0*iteration_count) / max_iterations))
                
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
#            Wcov_prime = (1.0-stabilize) * sumWcov * (1.0/len(estimates)) + stabilize * tns_learn.tensorData['Wcov']
            Wcov_prime = sumWcov * (1.0/len(estimates))
            tns_learn.setTensor('Wcov', Wcov_prime, tns_learn.tensorIndices['Wcoviempirical'] )
            #re-symmetrize to make iteration numerically stable:
#            tns_learn.tensorDataAsFlattened['Wcov'][:,:] = 0.5*tns_learn.tensorDataAsFlattened['Wcov'] + 0.5* tns_learn.tensorDataAsFlattened['Wcov'].T
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


    
## parallelized function for inverting psihats. Unfortunately, we need to place it outside of the class:
#_psihatinv_func_tnscopy_perprocess = {}
#def _psihatinv_func(ProMePobject, observation):
#        """
#        reads in distribution parameters, Yi observation points and corresponding PSIi tensors
#        
#        returns the expectation for each observation points given the distribution, as well as the data point's negative log-likelihood
#        
#        (PSIi,Yi, Wmena, Wcov) -> Wmeani, Wcovi
#        """
#        times, phases, values, Xrefs, Yrefs, Ts = observation

#        global _psihatinv_func_tnscopy_perprocess
#        pid = _os.getpid()
#        try:
#            tns_local = _psihatinv_func_tnscopy_perprocess[pid]
#        except KeyError:
#            tns_local = tns_base.copy()
#            _psihatinv_func_tnscopy_perprocess[pid] = tns_local
#            #print("deep-copying tns for pid {}".format(pid))
#            
#        num = phases.shape[0]
#        for n in range(num):
#            #compute PSIi:
#            tns_local.setTensor('phase', phases[n,:], (('g'),()) )
#            tns_local.setTensor('Xref', Xrefs[n,:,:,:], (('rtilde', 'dtilde', 'g'),()) ) 
#            tns_local.setTensor('Yref', Yrefs[n,:,:,:], (('r', 'd', 'g'),()) )                
#            tns_local.setTensor('T', Ts[n,:,:,:,:], (('r', 'd'),('rtilde', 'dtilde')) )
#            self._updateP() #call the code that computes map between phase and time derivatives. implemented directly in this class
#            self.PHI_computer.update() #call the hook that computes the interpolating map PHI from the phase:
#            #compute PSI only:
#            self.tns.update('P:PHI', 'PSI', 'O')
#            ydelta = values[n,:,:,:] #- self.tns.tensorData['O']
#            estimation_parameters.append( (tns_learn, tns_learn.tensorData['Wmean'], tns_learn.tensorData['Wcov'], self.tns.tensorData['PSI'].copy(), ydelta) ) #saving a view is enough

##            
#            
#        tns_local.setTensor('PSIhato', PSIhat)
#        #compute the estimation step:
#        tns_local.update('PSIhatInvo', 'PSIhato:PSIhatInvo')
#        return tns_local.tensorData['PSIhatInvo'].copy(), tns_local.tensorData['PSIhato:PSIhatInvo'].copy()




#    def learnFromObservationsWrongWay(self, 
#            observations,
#            max_iterations=150, 
#            minimal_relative_improvement=1e-4,
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
#        

#        useonly: only use the combination of (realm, derivative) to learn, assume that in all other combinations data are not trustworthy
#       
#        stabilize: value in range 0-1, 0 does does full EM updates, nonzero values slow down convergence but improve stability
#       
#        Implements the expectation-maximization procedure of the paper "Using probabilistic movement primitives in robotics" by Paraschos et al.
#        """
#        pool = _multiprocessing.Pool(8)        
#        
#        # set up computation of PSIhatInv = (PSIhat)^-1 = (Yref - (Yrefhat - T:Xrefhat) )^-1

#        tns_learn = _namedtensors.TensorNameSpace(self.tns)
#        self.tns_learn =tns_learn
#        
#        #tensors to do EM on:
#        tns_learn.registerTensor('Wmean', (['rtilde', 'gtilde', 'stilde', 'dtilde'], ()) )
#        tns_learn.registerTensor('Wcov', (['rtilde', 'gtilde', 'stilde','dtilde'], ['rtilde_', 'gtilde_', 'stilde_','dtilde_'],) )        
#                
#        #compute the precision matrix in Y space for a current sample i:
#        tns_learn.registerTensor('PSIi', self.tns.tensorIndices['PSI'])
#        tns_learn.registerTranspose('PSIi')
#        tns_learn.registerContraction('PSIi', 'Wcov')
#        tns_learn.registerContraction('PSIi:Wcov', '(PSIi)^T')
#        tns_learn.registerInverse('PSIi:Wcov:(PSIi)^T', result_name='Yprecisioni', flip_underlines=False)
#        
#        #compute the expectations:
#        tns_learn.registerTensor('Yi', (('r','d','g',),()) )
#        tns_learn.registerContraction('PSIi', 'Wmean')
#        tns_learn.registerSubtraction('Yi', 'PSIi:Wmean')
#        tns_learn.registerContraction('Yprecisioni', '(Yi-PSIi:Wmean)')
#        tns_learn.registerContraction('(PSIi)^T', 'Yprecisioni:(Yi-PSIi:Wmean)', result_name= 'Wmeanidelta', flip_underlines=True)
#        tns_learn.registerAddition('Wmean', 'Wmeanidelta', result_name= 'Wmeani')
#        
#        #compute negative-log-likelihood increment to judge progress:
#        tns_learn.registerTranspose('(Yi-PSIi:Wmean)')
#        tns_learn.registerContraction('(Yi-PSIi:Wmean)', 'Yprecisioni' )
#        tns_learn.registerContraction('(Yi-PSIi:Wmean):Yprecisioni','((Yi-PSIi:Wmean))^T', result_name='negLLidelta'  )
#        
#        #compute part of the maximization term for the covariance tensor:
#        tns_learn.registerTranspose('PSIi:Wcov')
#        tns_learn.registerContraction('(PSIi:Wcov)^T', 'Yprecisioni')
#        tns_learn.registerContraction('(PSIi:Wcov)^T:Yprecisioni', 'PSIi:Wcov', result_name='Wcovidelta')
#        tns_learn.registerSubtraction('Wcov', 'Wcovidelta', result_name= 'Wcovi')

#        first_part_steps = len(tns_learn.update_order)  #when computing, stop here for averaging over all computed Wmeani

#        #this is done separately, after Wmean has been updated:  
#        tns_learn.registerSubtraction('Wmeani', 'Wmean')      
#        tns_learn.registerTranspose('(Wmeani-Wmean)')  
#        tns_learn.registerContraction('(Wmeani-Wmean)', '((Wmeani-Wmean))^T', result_name='Wcoviempirical')
#    
#        #_pprint.pprint(tns_learn.tensorIndices)
#       
#    
#        iteration_count = 0 #maximum number of E-M iterations
#        negLL = []
#        tns_learn.setTensorToIdentity('Wcov', scale=100**2) #set the prior
#        tns_learn.setTensor('Wmean', 0.0) #set the prior  
#      
#        #set update() to not know the second part:
#        tns_learn.update_order = tns_learn.update_order[:first_part_steps]

#        #preprocess observations into pairs of Yi and PSIi
#        estimation_parameters = []
#        for observation_idx, (times, phases, values, Xrefs, Yrefs, Ts) in enumerate(observations):
#            num = phases.shape[0]
#            for n in range(num):
#                #compute PSIi:
#                self.tns.setTensor('phase', phases[n,:], (('g'),()) )
#                self.tns.setTensor('Xref', Xrefs[n,:,:,:], (('rtilde', 'dtilde', 'g'),()) ) 
#                self.tns.setTensor('Yref', Yrefs[n,:,:,:], (('r', 'd', 'g'),()) )                
#                self.tns.setTensor('T', Ts[n,:,:,:,:], (('r', 'd'),('rtilde', 'dtilde')) )
#                self._updateP() #call the code that computes map between phase and time derivatives. implemented directly in this class
#                self.PHI_computer.update() #call the hook that computes the interpolating map PHI from the phase:
#                #compute PSI only:
#                self.tns.update('P:PHI', 'PSI', 'O')
#                ydelta = values[n,:,:,:] #- self.tns.tensorData['O']
#                estimation_parameters.append( (tns_learn, tns_learn.tensorData['Wmean'], tns_learn.tensorData['Wcov'], self.tns.tensorData['PSI'].copy(), ydelta) ) #saving a view is enough

#        relative_improvement = 1.0
#        global _tnscopy_perprocess
#        _tnscopy_perprocess.clear()
#        relative_changes = 1.0
#        while iteration_count < max_iterations:
#            iteration_count = iteration_count+1        
#            
#            subset = estimation_parameters
#            stabilize = max(0.0, ((1.0*iteration_count) / max_iterations)-0.5)
#            #if we have many data points, and changes are still big, select a subset to speed up iteration:
##            if  len(estimation_parameters) > 0.2*tns_learn.tensorData['Wcov'].size and relative_changes > 0.1:
##                subset = [ _random.choice(estimation_parameters) for i in range(tns_learn.tensorData['Wcov'].size//5)]
##                stabilize = max(0.5, ((1.0*iteration_count) / max_iterations))
#                
#            estimates = pool.starmap(_estimation_func, subset)
#            #estimates = list(_it.starmap(_estimation_func, subset)) #non-parallel version
#            
#            #do maximization step for means:
#            Wmean_prime = (1.0-stabilize) * _np.mean([e[0] for e in estimates], axis=0) + stabilize * tns_learn.tensorData['Wmean']
#            tns_learn.setTensor('Wmean', Wmean_prime )
#            #do maximization step for covariances, using Wmean_prime
#            negLLDeltaSum = 0.0
#            sumWcov = 0.0          
#            for Wmeani, Wcovi, negLLdelta in estimates:
#                tns_learn.setTensor('Wmeani', Wmeani)
#                tns_learn.update('(Wmeani-Wmean)','((Wmeani-Wmean))^T','Wcoviempirical')
#                sumWcov = sumWcov + Wcovi + tns_learn.tensorData['Wcoviempirical']
#                negLLDeltaSum += negLLdelta
##            Wcov_prime = (1.0-stabilize) * sumWcov * (1.0/len(estimates)) + stabilize * tns_learn.tensorData['Wcov']
#            Wcov_prime = sumWcov * (1.0/len(estimates))
#            tns_learn.setTensor('Wcov', Wcov_prime, tns_learn.tensorIndices['Wcoviempirical'] )
#            #re-symmetrize to make iteration numerically stable:
##            tns_learn.tensorDataAsFlattened['Wcov'][:,:] = 0.5*tns_learn.tensorDataAsFlattened['Wcov'] + 0.5* tns_learn.tensorDataAsFlattened['Wcov'].T
#            negLL.append(negLLDeltaSum / len(subset))
#            
#            #terminate early if neg-log-likelihood of observations stops increasing:
#            n= 4
#            if len(negLL) > n:
#                relative_changes = [ abs((negLL[-i]-negLL[-i-1])/negLL[-i-1]) for i in range(1,n) ]
#                eigenvalues = _np.linalg.eigvals(tns_learn.tensorDataAsFlattened['Wcov'])
#                if _np.any(_np.iscomplex(eigenvalues)) or _np.any(eigenvalues < 0.0):                    
#                    print("Warning: At iteration {} eigenvalues became {} ".format(iteration_count, eigenvalues))
#                #print(_np.diag(tns_learn.tensorDataAsFlattened['Wcov']))
#                #print(tns_learn.tensorData['Wmean'])
#                print("{:.4}".format(relative_changes[0]))
#                relative_changes = max(relative_changes)
#                if relative_changes < minimal_relative_improvement:
#                    print("stopping early at iteration {}".format(iteration_count))
#                    break
#                    
#        self.negLL = _np.array(negLL)
#        #update yourself to the learnt estimate
#        self.tns.setTensor('Wmean', tns_learn.tensorData['Wmean'], tns_learn.tensorIndices['Wmean'])
#        self.tns.setTensor('Wcov', tns_learn.tensorData['Wcov'], tns_learn.tensorIndices['Wcov'])
#        self.expected_duration = _np.mean([times[-1] for(times, phases, values, Xrefs, Yrefs, Ts) in observations])

#        self.observedTrajectories = observations #remember the data we learned from, for plotting etc.



## parallelized function for EM learning. Unfortunately, we need to place it outside of the class:
#_tnscopy_perprocess = {}
#def _estimation_func(tns_base, Wmean, Wcov, PSIi, Ydeltai):
#        """
#        reads in distribution parameters, Yi observation points and corresponding PSIi tensors
#        
#        returns the expectation for each observation points given the distribution, as well as the data point's negative log-likelihood
#        
#        (PSIi,Yi, Wmena, Wcov) -> Wmeani, Wcovi
#        """
#        global _tnscopy_perprocess
#        pid = _os.getpid()
#        try:
#            tns_local = _tnscopy_perprocess[pid]
#        except KeyError:
#            tns_local = tns_base.copy()
#            _tnscopy_perprocess[pid] = tns_local
#            #print("deep-copying tns for pid {}".format(pid))
#        tns_local.setTensor('Wmean', Wmean)
#        tns_local.setTensor('Wcov', Wcov)
#        tns_local.setTensor('PSIi', PSIi)
#        #set the observed sample:
#        tns_local.setTensor('Yi', Ydeltai, arrayIndices= (('r', 'd', 'g'),()))
#        #compute the estimation step:
#        tns_local.update()
#        return tns_local.tensorData['Wmeani'].copy(), tns_local.tensorData['Wcovi'].copy(), tns_local.tensorData['negLLidelta'].copy()



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


