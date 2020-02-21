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


import mechanicalstate as _mechanicalstate

import namedtensors as _nt

class LTIGoal(object):
    """
    This class implements the behavior of a PD controller goal (linear time invariant goal)
    
    The man purpose of this class is to provide a mechanical state distribution adhering to a PD control law
    
    """

    def __init__(self, tensornamespace, * , current_msd_from=None, task_space='jointspace', kp=10.0, kv=5.0, kd=0.0, desiredTensorData=None, expectedTorqueNoise=0.01 , name='unnamed'):

        self.name = name
        self.phaseAssociable = False #indicate that this motion generator is not parameterized by phase
        self.timeAssociable = True #indicate that this motion generator is parameterizable by time
        self.taskspace_name = task_space
        
        if not tensornamespace is None:
            self.tns = _nt.TensorNameSpace(tensornamespace) #inherit index sizes
            if self.tns.indexSizes['r'] < 2:
                raise NotImplementedError() #sorry, this "trivial" functionality is not implemented. Try using a simple, fixed msd instead
        else:
            self.tns = _nt.TensorNameSpace()
            self.tns.registerIndex('r',2)
            self.tns.registerIndex('g',2)
            self.tns.registerIndex('d',1)

        self.tns.registerIndex('r2',self.tns.indexSizes['r'], self.tns.indexValues['r'])
        self.tns.registerIndex('g2',self.tns.indexSizes['g'])
        self.tns.registerIndex('d2',self.tns.indexSizes['d'],)

            
        self.msd_desired = _mechanicalstate.MechanicalStateDistribution(self.tns, "DesiredMean", "DesiredCov")
        self.msd_expected = _mechanicalstate.MechanicalStateDistribution(self.tns, "ExpectedMean", "ExpectedCov")
        self.commonnames2rg = self.msd_desired.commonnames2rg

        if current_msd_from is None: #local copy
            self.tns.registerTensor('CurrentMean', (('r','g','d'),()) )
            self.tns.registerTensor('CurrentCov', (('r','g','d'),('r_', 'g_', 'd_')) )
            self.msd_current = _mechanicalstate.MechanicalStateDistribution(self.tns, "CurrentMean", "CurrentCov")
        else:   #use data from somewhere else:
            self.tns.registerTensor('CurrentMean', current_msd_from.tns.tensorIndices[current_msd_from.meansName] , external_array=current_msd_from.tns.tensorData[current_msd_from.meansName], initial_values='keep' )
            self.tns.registerTensor('CurrentCov', current_msd_from.tns.tensorIndices[current_msd_from.covariancesName] , external_array=current_msd_from.tns.tensorData[current_msd_from.covariancesName], initial_values='keep')
            self.msd_current = current_msd_from


        #desired values:
        self.tns.registerTensor('DesiredMean', (('r','g','d'),()) )
        self.tns.registerTensor('DesiredCov', (('r','g','d'),('r_', 'g_', 'd_')) )


        r_torque, g_torque = self.commonnames2rg['torque']
        r_pos, g_pos = self.commonnames2rg['position']  
        self.tns.registerBasisTensor('e_ktau',  (('r2', 'g2'),('r', 'g')), (('effort', g_torque),('effort', g_torque)) )
        self.tns.registerTensor('Uktau',  (('d2',),('d',)), initial_values='identity')
        self.tns.registerBasisTensor('e_kp',  (('r2', 'g2'),('r', 'g')), (('effort', g_torque),('motion', g_pos)))
        self.tns.registerTensor('Kp', (('d2',),('d',)) )
        
        
        slice_tau = self.tns.registerContraction('e_ktau', 'Uktau')
        slice_kp = self.tns.registerContraction('e_kp', 'Kp')
        if 'velocity' in self.commonnames2rg:
            r_vel, g_vel = self.commonnames2rg['velocity']  
            self.tns.registerBasisTensor('e_kv',  (('r2', 'g2'),('r', 'g')), (('effort', g_torque),('motion', g_vel)))
            self.tns.registerTensor('Kv', (('d2',),('d',)) )
            slice_kv = self.tns.registerContraction('e_kv', 'Kv' )

            #add together:
            self.tns.registerSum(slice_tau, slice_kp, slice_kv, result_name='U')
        else:
            self.tns.registerAddition(slice_tau, slice_kp, result_name='U') 

        self.tns.registerTensor('I', self.tns.tensorIndices['U'], initial_values='identity')

        if 'velocity' in self.commonnames2rg:  #can we add damping terms? (kd)
            #Compute the K tensor: 
            self.tns.registerTensor('Kd', (('d2',),('d',)) )  #kd is equal to kv but with goal velocity=0  -> only appears here and not in the computation of 'U'
            slice_kd = self.tns.registerContraction('e_kv', 'Kd') 
            U_plus_damping = self.tns.registerAddition(slice_kd, 'U')
            
            self.tns.registerSubtraction('I', U_plus_damping, result_name='K')
        else:
            self.tns.registerSubtraction('I', 'U', result_name = 'K')

        self.tns.registerTranspose('U')
        self.tns.registerTranspose('K')

        
        self.tns.registerTensor('noise', (('r','d','d'),('r_','g_','d_')) )

        
        #influence of desired mean on expected mean:
        term_meanU = self.tns.registerContraction('U', 'DesiredMean')
        
        #influence of desired cov on expected cov:
        previous = self.tns.registerContraction('U', 'DesiredCov')
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


        #set values, if provided:
        self.setDesired(desiredMean=desiredTensorData, Kp=kp, Kv=kv)
        self.tns.update(*self.tns.update_order[:self._update_cheap_start])


    def setDesired(self, desiredMean=None, desiredCov=None, Kp=None, Kv=None):
        """
        set new desired means and covariances and recompute any dependent internal variables
        
        desiredMean: array of shape ('r', 'g', 'd')

        desiredCov: array of shape ('r', 'g', 'd', r_, g_, d_)

       
        Kp: array of shape ('d', 'd_')    (order: torque, position)

        Kv: array of shape ('d', 'd_')    (order: torque, velocity)
        """
        if self.msd_current.tns is self.tns:  #local data?
            if not desiredMean is None:            
                self.tns.setTensor('DesiredMean', desiredMean)
            if not desiredCov is None:
                self.tns.setTensor('DesiredCov', desiredCov)
        else:
            if not (desiredMean is None and desiredCov is None):
                raise ValueError("I don't dare setting a data array I don't own.")
            
        for value, tensorname in ( (Kp, 'Kp'), (Kv, 'Kv')):
            if value is None:
                continue
            if _np.isreal(value):
                pass
            elif value.ndim == 1:
                Kp = _np.diag(Kp)
            else:
                pass
            self.tns.setTensor(tensorname, value)
        self.tns.update(*self.tns.update_order[:self._update_cheap_start])
            


    def getDistribution(self, *, current_msd=None, task_spaces=None, **kwargs):
        """
            return an expected mechanicalstate distribution 
            constructed from the current msd and the lti goals
        
        """
        if not current_msd is None:
            self.tns.setTensor('CurrentMean', current_msd.getMeansData())
            self.tns.setTensor('CurrentCov', current_msd.getCovariancesData())

        self.tns.update(*self.tns.update_order[self._update_cheap_start:])
        
        return self.msd_expected
    
    
    
    def serialize(self):
        """

        returns a python dictionary that contains all internal data
        necessary to recreate this Controller

        """
        serializedDict = {}
        serializedDict[u'name'] = self.name
        serializedDict[u'r'] = self.tns.indexSizes['r']
        serializedDict[u'g'] = self.tns.indexSizes['g']
        serializedDict[u'd'] = self.tns.indexSizes['d']
        serializedDict[u'task_space'] = self.taskspace_name
        serializedDict[u'kp'] = self.kp
        serializedDict[u'kv'] = self.kv
        serializedDict[u'kd'] = self.kd
        
        return serializedDict

    @classmethod
    def makeFromDict(cls, params):
        """
        Create a controller from the given dictionary
        
        The controller classes possess a serialize() function to create those dictionaries
        """
        c = PDController(**params)
        return c
        

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
            filename = '{0}_{1}.ltigoal.h5'.format(_time.strftime('%Y%m%d%H%M%S'), d[u'name'])
        else:
            filename = '{0}.ltigoal.h5'.format(d[u'name']) 
        filepath= _os.path.join(path, filename)
        _h5.write(d, filename=filepath, store_python_metadata=True)
        return filepath

        
    def __repr__(self):
        text  = "Name: {}\n".format(self.name)
        text += "r,g,d: {},{},{}\n".format(self.tns.indexSizes['r'],self.tns.indexSizes['g'], self.tns.indexSizes['d'] )
        for name in ('Kp', 'Kv', 'Kd'):            
            if name in self.tns.tensorData:
                text += "{}:\n".format(name)
                text += "{}\n".format(self.tns.tensorData[name])
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
            self.tensorData['mean_initial'][self.commonnames2rg['position']][:] = 1.4
            self.tensorData['mean_initial'][self.commonnames2rg['position']][:] = 0.0
        else:
            self.setTensor('mean_initial', distInitial.means)
            self.setTensor('cov_initial', distInitial.covariances)
        
        a,b = 0.5*_np.min(self.tensorData['mean_initial']), 0.5*_np.max(self.tensorData['mean_initial'])
        limits_tightest = {
            'torque': [-1,1],
            'position': [(a+b)-1*(b-a), (a+b) + 1*(b-a) ],
            'velocity': [(a+b)-1*(b-a), (a+b) + 1*(b-a) ],
        }
        limits={}
        for limitname in self.commonnames2rg:
            limits[limitname] = limits_tightest[limitname]

        if dofs_to_plot=='all':
            dofs_to_plot=list(range(self.tns.indexSizes['d']))

        mStateNames = list(self.tns.commonnames2rg)
        mstates = len(mStateNames)

        #if no mass matrix inverse is provided, assume a decoupled unit mass system for plotting
        if massMatrixInverseFunc is None:
            massMatrixInverseFunc = lambda q: _np.eye(self.tns.indexSizes['d'])
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
            for dof in range(self.tns.indexSizes['d']):
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
            tns_plot.setTensorFromFlattened('mean_sampled', _np.random.multivariate_normal(tns_plot.tensorDataAsFlattened['mean_initial']) )
            tns_plot.setTensorFromFlattened('cov_sampled',  _np.random.multivariate_normal(tns_plot.tensorDataAsFlattened['cov_initial' ]) )

            msd = MechanicalStateDistribution(self.tensorData['mean_sampled'], self.tensorData['cov_sampled'],)
            for i in range(num):
                msd_expected = self.getDistribution(current_msd=msd)
                msd = integrator.integrate(msd_expected, dt)
                self.tensorData['data_mean'][i][...] = msd.means
                
                for dof in range(self.tns.indexSizes['d']):
                    tns_plot.tensorData['data_sigma'][i,:,:,dof] = _np.sqrt(_np.diag(msd.covariances[:,:,dof,:,:,dof]))
                
            #update the desired plotting limits:
            for limitname in limits:
                for m in range(data_mean.shape[1]):
                    mstateIndex = self._md.mStateNames2Index[limitname]
                    sigma =tns_plot.tensorData['data_sigma'][self.commonnames2rg[limitname]]
                    mean =tns_plot.tensorData['data_mean'][self.commonnames2rg[limitname]]
                    limits[limitname][0] = min(_np.min(mean-1.96*sigma), limits[limitname][0])
                    limits[limitname][1] = max(_np.max(mean+1.96*sigma), limits[limitname][1])
            #plot all dofs
            for d, dof in enumerate(dofs_to_plot):
                for j, name in enumerate(mStateNames):
                    slicedef  = (None) + self.commonnames2rg[name] + (dof)
                    axesArray[j,d].plot(t, tns_plot.tensorData['data_mean'][slicedef], alpha=alpha, linewidth=linewidth )

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


