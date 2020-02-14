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


class LTIGoal(object):
    """
    This class implements the behavior of a PD controller goal (linear time invariant goal)
    
    The man purpose of this class is to provide a mechanical state distribution adhering to a PD control law
    
    """

    def __init__(self, task_space='jointspace', kp=10.0, kv=5.0, kd=0.0, name='unnamed', mechanicalstate=None, dofs=None, desiredPosition=None, desiredTorque=None, desiredVelocity=None, expectedTorqueNoise=0.01 , **kwargs):
        self.name = name
        self.phaseAssociable = False #indicate that this motion generator is not parameterized by phase
        self.timeAssociable = True #indicate that this motion generator is parameterizable by time
        self.taskspace_name = task_space
        
        self.tns = _nt.TensorNameSpace()
        if mechanicalstate is not None:            
            [ self.registerIndex(iname,mechanicalstate.tns.indexSizes[iname]) for iname in mechanicalstate.tns.indexSizes ]
        else: #assume default sizes:
            self.tns.registerIndex('r',2)
            self.tns.registerIndex('g',2)
            self.tns.registerIndex('d',2)
        self.name2rg = mechanicalstate.name2rg
        
        ones = _np.ones(self.tns.indexSizes['d'])
        kp = ones * _np.asarray(kp)
        kd = ones * _np.asarray(kd)
        kv = ones * _np.asarray(kv)

                
        self.tns.registerTensor('desired', (('r','g','d'),()) )
        if desiredPosition is not None:
           self.tns.tensorData['desired'][self.name2rg['position']][:] = desiredPosition
        if desiredVelocity is not None:
           self.tns.tensorData['desired'][self.name2rg['velocity']][:] = desiredVelocity
        if desiredTorque is not None:
           self.tns.tensorData['desired'][self.name2rg['torque']][:] = desiredTorque
        
        self.tns.registerTensor('U', (('r2','g2','d2'),('r','g','d')) )
        self.tns.registerTensor('K', (('r2','g2','d2'),('r','g','d')) )

        self.tns.registerTensor('cov', (('r','g','d'),('r_','g_','d_')) )
        self.tns.registerTensor('mean', (('r','g','d'),()) )
        self.tns.registerTensor('noise', (('r2','g2','d2'),('r2_','g2_','d2_')) )
        tautau  = self.name2rg['torque'] +(None) + self.name2rg['torque'] + (None)
        self.tns.tensorData['noise'][tautau] = _np.asarray(expectedTorqueNoise)

        self.tns.registerTranspose('U')
        self.tns.registerTranspose('K')
        self.tns.registerContraction('U', 'desired')
        
        
        desiredU = _t.dot( self.U, self.desired ) 

        self.registerContraction('K', 'cov')
        self.registerContraction('K:cov', '(K)^T')
        self.registerAddition('K:cov:(K)^T', 'noise', result_name='covExpected')
        self.registerContraction('K', 'mean')
        self.registerAddition('K:mean', 'U:desired', result_name='meanExpected')
        
        self.operations_every_time = self.tns.update_order[self.tns.update_order.index('K:cov'),:]

        self._updateKU(newKp=kp, newKv=kv, newKd=kd)
        


    def _updateKU(self, newKp=None, newKv=None, newKd=None):
        """
        update the state evolution matrices according to the current mass and (optionally) timestep duration
        """
        #first, save values and decide on what to recompute
        updateAll=False
        if newKp is not None:
            self.kp = _np.asarray(newKp)
            updateAll = True
        if newKv is not None:
            self.kv = _np.asarray(newKv)
            updateAll = True
        if newKd is not None:
            self.kd = _np.asarray(newKd)
            updateAll = True
        
        
        #recompute the mass-free update matrices:
        if updateAll:
            for dof in range(self.tns.indexSizes['d']):
                self.tensorData['U'][:,:,dof,:,:,dof] = [
                    [1, self.kp[dof], self.kv[dof]],
                    [0,0,0],
                    [0,0,0],
                ]
                self.tensorData['K'][:,:,dof,:,:,dof] = [
                    [0, -self.kp[dof], (-self.kv[dof]-self.kd[dof])],
                    [0,1,0],
                    [0,0,1],
                ]
            self.tns.update(('U:desired'))

            
#    def setDesiredPosition(self, desiredPosition):
#        """
#        set a new desired position
#        """
#        
#        self.tns.setTensor('desired'], 0.0)
#        self.tns.tensorData['desired'][self.name2rg['position']][:] = desiredPosition
#        self.tns.update(('U:desired'))

#    def setDesiredTorque(self, desiredTorque):
#        """
#        set a new desired position
#        """
#        self.tns.tensorData['desired'][self.name2rg['torque']][:] = desiredTorque
#        self.tns.update(('U:desired'))


    def setDesired(self, desiredMean, expectedTorqueNoise):
        """updateDesired
        set new desired means and covariances and recompute any dependent internal variables
        """
        self.setTensor('desired', desiredMean)
        self.tns.update(('U:desired'))
        
        tautau  = self.name2rg['torque'] +(None) + self.name2rg['torque'] + (None)
        self.tns.tensorData['noise'][tautau] = _np.asarray(expectedTorqueNoise)
                


    def getDistribution(self, generalized_phase=None, current_msd=None, task_spaces=None):
        """
        modify current_msd so it obeys the pd-control law
            (i.e. changes the torque-related means and covariances)
        
           phase: not used (ignored)

           phaseVelocity: not used (ignored)

           currentDistribution: (means, covariances) tuple of multivariate normal distribution
           
           phaseVelocitySigma: not used (ignored)

           returns: (means, covariances) tuple
               means: (derivatives x dofs) array
               covariances: (derivatives x dofs x derivatives x dofs) array
        """
        self.tns.setTensor('mean', current_msd.means)
        self.tns.setTensor('cov', current_msd.covariances)

        self.tns.update(self.operations_every_time)
        
        return MechanicalStateDistribution(self.tns, 'mean', 'cov')
    
    
    
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
            self.tensorData['mean_initial'][self.name2rg['position']][:] = 1.4
            self.tensorData['mean_initial'][self.name2rg['position']][:] = 0.0
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
        for limitname in self.name2rg:
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
                    slicedef  = (None) + self.name2rg[name] + (dof)
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


