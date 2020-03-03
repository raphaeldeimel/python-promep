#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel
@copyright 2018
@licence: 2-clause BSD licence

This file contains classes that compute the evolution of state distributions of classic controllers

Classic PD-Controller goal


"""

import os as _os
import numpy as _np
import itertools as _it
import hdf5storage as _h5
import time as _time
import matplotlib.pyplot as _plt

import ruamel.yaml as _yaml

import mechanicalstate as _mechanicalstate

import namedtensors as _nt

class LTIGoal(object):
    """
    This class implements the behavior of a PD controller goal (linear time invariant goal)
    
    The man purpose of this class is to provide a mechanical state distribution adhering to a PD control law
    
    """

    def __init__(self, tensornamespace, * , current_msd_from=None, task_space='jointspace', name='unnamed', expected_torque_noise=0.123, **kwargs):

        self.name = name
        self.phaseAssociable = False #indicate that this motion generator is not parameterized by phase
        self.timeAssociable = True #indicate that this motion generator is parameterizable by time
        self.taskspace_name = task_space
        
        if not tensornamespace is None:
            self.tns = _nt.TensorNameSpace(tensornamespace) #inherit index sizes
            if self.tns['r'].size < 2:
                raise NotImplementedError() #sorry, this "trivial" functionality is not implemented. Try using a simple, fixed msd instead
        else:
            self.tns = _mechanicalstate.makeTensorNameSpaceForMechanicalStateDistributions(r=2, g=2, d=1)

        self.tns.cloneIndex('r', 'r2')
        self.tns.cloneIndex('g', 'g2')
        self.tns.cloneIndex('d', 'd2')

            
        #desired values:
        self.tns.registerTensor('DesiredMean', (('r','g','d'),()) )
        self.tns.registerTensor('DesiredCov', (('r','g','d'),('r_', 'g_', 'd_')))
        self.msd_desired = _mechanicalstate.MechanicalStateDistribution(self.tns, "DesiredMean", "DesiredCov")
        
        #initialize the desired torque variance to some non-zero value:        
        self.tns.setTensorToIdentity('DesiredCov', scale=1e-6)
        self.msd_desired.addVariance('torque', expected_torque_noise**2)
#        self.msd_desired.addVariance('impulse', expected_torque_noise**2)

        self.commonnames2rg = self.msd_desired.commonnames2rg #we use them a lot here, so remember a shorthand
        self.commonnames2rglabels = self.msd_desired.commonnames2rglabels

        if current_msd_from is None: #local copy
            self.tns.registerTensor('CurrentMean', (('r','g','d'),()) )
            self.tns.registerTensor('CurrentCov', (('r','g','d'),('r_', 'g_', 'd_')) )
            self.msd_current = _mechanicalstate.MechanicalStateDistribution(self.tns, "CurrentMean", "CurrentCov")
        else:   #use data from somewhere else:
            self.tns.registerTensor('CurrentMean', current_msd_from.tns[current_msd_from.meansName].index_tuples , external_array=current_msd_from.tns[current_msd_from.meansName].data, initial_values='keep' )
            self.tns.registerTensor('CurrentCov', current_msd_from.tns[current_msd_from.covariancesName].index_tuples , external_array=current_msd_from.tns[current_msd_from.covariancesName].data, initial_values='keep')
            self.msd_current = current_msd_from




        rlabel_torque, glabel_torque = self.commonnames2rglabels['torque']
        rlabel_pos, glabel_pos = self.commonnames2rglabels['position']
        self.tns.registerBasisTensor('e_ktau',  (('r2', 'g2'),('r', 'g')), ((rlabel_torque, glabel_torque),(rlabel_torque, glabel_torque)) )
        self.tns.registerTensor('delta_d2d',  (('d2',),('d',)), initial_values='identity')
        self.tns.registerBasisTensor('e_kp',  (('r2', 'g2'),('r', 'g')), ((rlabel_torque, glabel_torque),(rlabel_pos, glabel_pos)))
        self.tns.registerTensor('Kp', (('d2',),('d',)) )
        
        
        slice_tau = self.tns.registerContraction('e_ktau', 'delta_d2d')
        slice_kp = self.tns.registerContraction('e_kp', 'Kp')
        if 'velocity' in self.commonnames2rg:
            rlabel_vel, glabel_vel = self.commonnames2rglabels['velocity']
            self.tns.registerBasisTensor('e_kv',  (('r2', 'g2'),('r', 'g')), ((rlabel_torque, glabel_torque),(rlabel_vel, glabel_vel)))
            self.tns.registerTensor('Kv', (('d2',),('d',)) )
            slice_kv = self.tns.registerContraction('e_kv', 'Kv' )

            #add together:
            self.tns.registerSum(slice_tau, slice_kp, slice_kv, result_name='U')
        else:
            self.tns.registerAddition(slice_tau, slice_kp, result_name='U') 

        self.tns.registerTensor('I', self.tns['U'].index_tuples, initial_values='identity')

        if 'velocity' in self.commonnames2rg:  #can we add damping terms? (kd)
            #Compute the K tensor: 
            self.tns.registerTensor('Kd', (('d2',),('d',)) )  #kd is equal to kv but with goal velocity=0  -> only appears here and not in the computation of 'U'
            slice_kd = self.tns.registerContraction('e_kv', 'Kd') 
            U_minus_damping = self.tns.registerAddition('U',slice_kd)
            
            self.tns.registerSubtraction('I', U_minus_damping, result_name='K')
        else:
            self.tns.registerSubtraction('I', 'U', result_name = 'K')

        self.tns.registerTranspose('U')
        self.tns.registerTranspose('K')


        
        #influence of desired mean on expected mean:
        term_meanU = self.tns.registerContraction('U', 'DesiredMean')
        
        #influence of desired cov on expected cov:
        previous = 'DesiredCov'
#        previous = self.tns.registerAddition('DesiredCov','CurrentCov')
        previous = self.tns.registerContraction('U', previous)
        term_covU = self.tns.registerContraction(previous, '(U)^T')

        #up to here we can precompute the equations if goals don't change
        self._update_cheap_start = len(self.tns.update_order)


        
        #influence of current cov to expected cov:
        previous = self.tns.registerContraction('K', 'CurrentCov')
        term_covK = self.tns.registerContraction(previous, '(K)^T')
        self.tns.registerAddition(term_covK, term_covU, result_name='ExpectedCov')

        #influence of current mean on expected mean:        
        previous = self.tns.registerContraction('K', 'CurrentMean')
        self.tns.registerAddition(previous, term_meanU, result_name='ExpectedMean')

        droptwos={'r2':'r', 'g2':'g', 'd2':'d', 'r2_':'r_', 'g2_':'g_', 'd2_':'d_'}
        self.tns.renameIndices('ExpectedCov', droptwos, inPlace=True)
        self.tns.renameIndices('ExpectedMean', droptwos, inPlace=True)

        #package the result:
        self.msd_expected = _mechanicalstate.MechanicalStateDistribution(self.tns, "ExpectedMean", "ExpectedCov")
                
        #set values, if provided:
        self.setDesired(**kwargs)
        self.tns.update(*self.tns.update_order[:self._update_cheap_start])


    def setDesired(self, desiredMean=None, desiredCov=None, **kwargs):
        """
        set new desired means and covariances and recompute any dependent internal variables
        
        desiredMean: array of shape ('r', 'g', 'd')

        desiredCov: array of shape ('r', 'g', 'd', r_, g_, d_)

       
        Kp: array of shape ('d', 'd_')    (order: torque, position)

        Kv: array of shape ('d', 'd_')    (order: torque, velocity)
        """
        known_gaintensors = ['Kp', 'Kv', 'Kd']
        known_goaltypes = ['position','impulse','velocity','torque']
        if self.msd_current.tns is self.tns:  #local data?
            if not desiredMean is None:            
                self.tns.setTensor('DesiredMean', desiredMean)
            if not desiredCov is None:
                self.tns.setTensor('DesiredCov', desiredCov)
            for name in kwargs:
                if name in known_gaintensors:
                    if kwargs[name] is None:
                        continue
                    if not name in self.tns.tensor_names:
                        continue
                    value = _np.asarray(kwargs[name])
                    if value.ndim==0:
                        gains = _np.eye(self.tns['d'].size) * value
                    elif value.ndim == 1:
                        gains = _np.diag(value)
                    elif value.ndim==2:
                        gains=value
                    else:
                        raise ValueError()
                    self.tns.setTensor(name, gains)
                elif name in known_goaltypes:
                    if name in self.commonnames2rg:
                        rg = self.commonnames2rg[name]
                        self.tns['DesiredMean'].data[rg][:] = kwargs[name]
                elif name in ('r', 'g', 'd'): #for simplicity of serialization/deserialization, ignore these kwargs
                    pass 
                else:
                    raise ValueError("'{}' is not a valid argument (I know: {})".format(name, ", ".join(known_goaltypes + known_gaintensors )))
        else:
            if not (desiredMean is None and desiredCov is None):
                raise ValueError("I don't dare setting a data array I don't own.")
            else:
                pass   
        self.tns.update(*self.tns.update_order[:self._update_cheap_start])
            


    def getDistribution(self, *, msd_current=None, task_spaces=None, **kwargs):
        """
            return an expected mechanicalstate distribution 
            constructed from the current msd and the lti goals
        
        """
        if not msd_current is None:
            self.tns.setTensor('CurrentMean', msd_current.getMeansData())
            self.tns.setTensor('CurrentCov', msd_current.getCovariancesData())

        self.tns.update(*self.tns.update_order[self._update_cheap_start:])
        
        return self.msd_expected
    
    
    
    def serialize(self):
        """

        returns a python dictionary that contains all internal data
        necessary to recreate this Controller

        """
        serializedDict = {}        
        serializedDict[u'name'] = self.name
        serializedDict[u'r'] = self.tns['r'].size
        serializedDict[u'g'] = self.tns['g'].size
        serializedDict[u'd'] = self.tns['d'].size
        serializedDict[u'task_space'] = self.taskspace_name

        data = self.msd_desired.getMeansData()
        for name in ('position', 'velocity', 'torque', 'impulse'):
            if name in self.commonnames2rg:
                serializedDict[name] = data[self.commonnames2rg[name]].tolist()

        for name in ('Kp', 'Kv', 'Kd'):
            if name in self.tns.tensor_names:
                serializedDict[name] = self.tns[name].data.tolist()
        
        return serializedDict

    @classmethod
    def makeFromDict(cls, params, tns=None):
        """
        Create a controller from the given dictionary
        
        The controller classes possess a serialize() function to create those dictionaries
        
        Either the dictionary contains the index sizes (r,g,d), or you can provide a TensorNameSpace instead
        """
        if tns==None:
            tns = _mechanicalstate.makeTensorNameSpaceForMechanicalStateDistributions(r=params['r'],g=params['g'],d=params['d'])
        c = LTIGoal(tns, **params)
        return c
        

    def saveToFile(self, forceName=None, path='./', withTimeStamp=False):
        """
        save the LTIGoal parameters to a file

        """
        d  = self.serialize()
        if forceName is not None:
            d[u'name']=forceName
        
        if withTimeStamp:
            filename = '{0}_{1}.ltigoal.yaml'.format(_time.strftime('%Y%m%d%H%M%S'), d[u'name'])
        else:
            filename = '{0}.ltigoal.yaml'.format(d[u'name']) 
        filepath= _os.path.join(path, filename)   
        with open(filepath, 'w') as f:
            _yaml.dump(self.serialize(), stream=f)
        return filepath

    @classmethod
    def makeFromFile(cls, filepath, tns=None):
        """
        Create a controller from the given yaml file
        
        The controller classes possess a saveToFile() function to create that file
        
        Either the dictionary contains the index sizes (r,g,d), or you can provide a TensorNameSpace instead
        """
        with open(filepath, 'r') as f:
            d = _yaml.load(f, Loader=_yaml.Loader)
        ltigoal = cls.makeFromDict(d, tns=tns)
        return ltigoal


        
    def __repr__(self):
        text  = "Name: {}\n".format(self.name)
        text += "r,g,d: {},{},{}\n".format(self.tns['r'].size,self.tns['g'].size, self.tns['d'].size )
        for name in ('Kp', 'Kv', 'Kd', 'DesiredMean'):            
            if name in self.tns.tensor_names:
                text += "{}:\n".format(name)
                text += "{}\n".format(self.tns[name].data)
        return text
    
    

    def plot(self, dofs='all',
                   distInitial = None,
                   num=100,
                   duration = 1.0,
                   linewidth=2.0,
                   withSampledTrajectories=100,
                   withConfidenceInterval=True,
                   posLimits=None,
                   velLimits=None,
                   torqueLimits=None,
                   sampledTrajectoryStyleCycler=_plt.cycler('color', ['#0000FF']),
                   massMatrixInverseFunc = None, 
                   ):
        """
        visualize the trajectory distribution as parameterized by the means of each via point,
        and the covariance matrix, which determins the variation at each via (sigma) and the
        correlations between variations at each sigma

        E.g. usually, neighbouring via points show a high, positive correlation due to dynamic acceleration limits


        interpolationKernel:  provide a function that defines the linear combination vector for interpolating the via points
                                If set to None (default), promp.makeInterpolationKernel() is used to construct a Kernel as used in the ProMP definition
        """
        tns_plot = _nt.TensorNameSpace(self.tns)
        confidenceColor = '#EEEEEE'
        dt = duration / num 
        t = _np.linspace(0.0, dt*num, num)
        tns_plot.registerTensor('data_mean', (('r','g','d'),()))
        tns_plot.registerTensor('data_sigma', (('r','g','d'),()))

        tns_plot.registerTensor('mean_initial', (('r','g','d'),()))
        tns_plot.registerTensor('cov_initial', (('r','g','d'),('r_','g_','d_')))

        tns_plot.registerTensor('mean_sampled', (('r','g','d'),()))
        tns_plot.registerTensor('cov_sampled', (('r','g','d'),('r_','g_','d_')))
        if distInitial is None:
            #create a useful initial dist for plotting:
            tns_plot['mean_initial'].data[self.commonnames2rg['position']][:] = 1.4
            tns_plot['mean_initial'].data[self.commonnames2rg['position']][:] = 0.0
        else:
            tns_plot.setTensor('mean_initial', distInitial.means)
            tns_plot.setTensor('cov_initial', distInitial.covariances)
        
        a,b = 0.5*_np.min(tns_plot['mean_initial'].data), 0.5*_np.max(tns_plot['mean_initial'].data)
        limits_tightest = {
            'torque': [-1,1],
            'impulse': [-1,1],
            'position': [(a+b)-1*(b-a), (a+b) + 1*(b-a) ],
            'velocity': [(a+b)-1*(b-a), (a+b) + 1*(b-a) ],
        }
        limits={}
        for limitname in self.commonnames2rg:
            limits[limitname] = limits_tightest[limitname]

        if dofs_to_plot=='all':
            dofs_to_plot=list(range(self.tns['d'].size))

        mStateNames = list(self.tns.commonnames2rg)
        mstates = len(mStateNames)

        #if no mass matrix inverse is provided, assume a decoupled unit mass system for plotting
        if massMatrixInverseFunc is None:
            massMatrixInverseFunc = lambda q: _np.eye(self.tns['d'].size)
        #instantiate a time integrator for simulation
        integrator = _mechanicalstate.TimeIntegrator(self.tns)

        subplotfigsize=2.0
        fig, axesArray = _plt.subplots(mstates,len(dofs_to_plot), squeeze=False, figsize=(max(len(dofs_to_plot),mstates)*subplotfigsize, mstates*subplotfigsize), sharex='all', sharey='row')

        #compute the distribution over time
        dist = distInitial 
        for i in range(num):
            dist = self.getInstantStateVectorDistribution(currentDistribution=dist)
            dist = integrator.integrate(dist, dt)
            data_mean[i,:] = dist[0]
            for dof in range(self.tns['d'].size):
                data_sigma[i,:,dof] = _np.sqrt(_np.diag(dist[1][:,dof,:,dof]))


        axesArray.shape = (mstates, len(dofs_to_plot))
        #draw the confidence intervals and means/variance indicators for the supports
        for i, dof in enumerate(dofs_to_plot):

            #plot the zero-variance trajectory + 95% confidence interval
            if withConfidenceInterval:
                axesArray[ self._iTau, i].fill_between(t, data_mean[:,self._iTau,dof]-1.96*data_sigma[:,self._iTau,dof], data_mean[:,self._iTau,dof]+1.96*data_sigma[:,self._iTau,dof], alpha=1.0, label="95%",  color=confidenceColor)

                axesArray[ self._iPos, i].fill_between(t, data_mean[:,self._iPos,dof]-1.96*data_sigma[:,self._iPos,dof], data_mean[:,self._iPos,dof]+1.96*data_sigma[:,self._iPos,dof], alpha=1.0, label="95%",  color=confidenceColor)
                axesArray[ self._iVel, i].fill_between(t, data_mean[:,self._iVel,dof]-1.96*data_sigma[:,self._iVel,dof], data_mean[:,self._iVel,dof]+1.96*data_sigma[:,self._iVel,dof], alpha=1.0, label="95%",  color=confidenceColor)
                #adjust plot limits
                for limitname in limits:
                    for m in range(data_mean.shape[1]):
                        mstateIndex = mStateNames.index(limitname)
                        limits[limitname][0] = min(_np.min(data_mean[:,mstateIndex]-1.96*data_sigma[:,mstateIndex]), limits[limitname][0])
                        limits[limitname][1] = max(_np.max(data_mean[:,mstateIndex]+1.96*data_sigma[:,mstateIndex]), limits[limitname][1])


        #sample the distribution to plot actual trajectories, times the number given by "withSampledTrajectories":
        alpha = _np.sqrt(2.0 / (1+withSampledTrajectories))
        for ax in axesArray.flatten():
            ax.set_prop_cycle(sampledTrajectoryStyleCycler)

        #draw examples of trajectories with sampled initial states:
        for j in range(withSampledTrajectories):
            tns_plot.setTensorFromFlattened('mean_sampled', _np.random.multivariate_normal(tns_plot['mean_initial'].data_flat) )
            tns_plot.setTensorFromFlattened('cov_sampled',  _np.random.multivariate_normal(tns_plot['cov_initial' ].data_flat) )

            msd = MechanicalStateDistribution(self['mean_sampled'].data, self['cov_sampled'].data,)
            for i in range(num):
                msd_expected = self.getDistribution(current_msd=msd)
                msd = integrator.integrate(msd_expected, dt)
                self['data_mean'].data[i][...] = msd.means
                
                for dof in range(self.tns['d']):
                    tns_plot['data_sigma'].data[i,:,:,dof] = _np.sqrt(_np.diag(msd.covariances[:,:,dof,:,:,dof]))
                
            #update the desired plotting limits:
            for limitname in limits:
                for m in range(data_mean.shape[1]):
                    mstateIndex = self._md.mStateNames2Index[limitname]
                    sigma =tns_plot['data_sigma'].data[self.commonnames2rg[limitname]]
                    mean =tns_plot['data_mean'].data[self.commonnames2rg[limitname]]
                    limits[limitname][0] = min(_np.min(mean-1.96*sigma), limits[limitname][0])
                    limits[limitname][1] = max(_np.max(mean+1.96*sigma), limits[limitname][1])
            #plot all dofs
            for d, dof in enumerate(dofs_to_plot):
                for j, name in enumerate(mStateNames):
                    slicedef  = (None) + self.commonnames2rg[name] + (dof)
                    axesArray[j,d].plot(t, tns_plot['data_mean'].data[slicedef], alpha=alpha, linewidth=linewidth )

        #override scaling:
        if posLimits is not None:
            limits['position'] = posLimits
        if velLimits is not None:
            limits['velocity'] = velLimits
        if torqueLimits is not None:
            limits['torque']=torqueLimits

        padding=0.05
        for i, dof in enumerate(dofs_to_plot):
            for m in range(len(mStateNames)):
                axes = axesArray[m,i]  
                axes.set_title(r"{0} {1}".format(mStateNames[m], dof))
                axes.set_xlim((0.0, 1.0))
                lim = limits[mStateNames[m]]
                avg = _np.mean(lim)
                delta = max(0.5*(lim[1]-lim[0]), 0.1*abs(avg))
                axes.set_ylim(avg-delta*(1+padding), avg+delta*(1+padding))
        _plt.tight_layout()


