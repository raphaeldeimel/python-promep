#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel
@copyright 2018
@licence: 2-clause BSD licence

This file contains classes that compute the evolution of state distributions of classic controllers

Classic PD-Controller


"""

import os as _os
import numpy as _np
import itertools as _it
import hdf5storage as _h5
import time as _time
import matplotlib.pyplot as _plt
import math
import tf as _tf 


from . import _tensorfunctions as _t
from . import _timeintegrator as _tint
from . import MechanicalStateDistributionDescription as _MechanicalStateDistributionDescription

def makeControllerFromDict(params):
    """
    Create a controller from the given dictionary
    
    The controller classes possess a serialize() function to create those dictionaries
    """
    controllerType = params[u'type']
    if controllerType == u'PDController':
        c = PDController(**params)
    elif controllerType == u'TaskSpaceController':
        c = TaskSpaceController(**params)
    else:
        raise ValueError("Unknown state controller type!")
    return c


def extractPDGains(CovarianceTensor, derivativesCountEffort=1, derivativesCountMotion=2):
    """
    (class function)
    compute and return PD controller gains from the given covariance matrix
    
    CovarianceTensor: tensor of shape ((mstate, dofs),(mstate, dofs))
        with mstate index = [torque, position, velocity]
    
    returns a tensor of shape (dofs, motion derivatives, dofs)
    
    """
    dofs = CovarianceTensor.shape[1]
    gains = _np.zeros((dofs,derivativesCountMotion,dofs))

    sigma_qt = CovarianceTensor[0,    :, derivativesCountEffort:, :]
    sigma_qq = CovarianceTensor[derivativesCountEffort:, :, derivativesCountEffort:, :]
    sigma_qq_inv = _t.pinv(sigma_qq,regularization=0.0)
    gains = -1 * _t.dot(sigma_qt,sigma_qq_inv, shapes=((1,2),(2,2))) 

    return gains

class PDController(object):
    """
    This class implements the behavior of a PD controller
    
    The man purpose of this class is to evolve a mechanical state distribution according to 
    a PD control law
    
    """

    def __init__(self, kp=10.0, kv=5.0, kd=0.0, name='unnamed', mstateDescription=None, dofs=None, desiredPosition=None, desiredTorque=None, desiredVelocity=None, expectedTorqueNoise=0.01 , derivativesCountEffort=1, **kwargs):
        self.name = name
        self.phaseAssociable = False #indicate that this motion generator is not parameterized by phase
        self.timeAssociable = True #indicate that this motion generator is parameterizable by time
        
        if mstateDescription is not None:
            self._md = mstateDescription
        elif desiredPosition is not None:
            self._md = _MechanicalStateDistributionDescription(dofs=_np.asarray(desiredPosition).size, derivativesCountEffort=1)
        else:
            self._md = _MechanicalStateDistributionDescription(dofs=_np.asarray(kp).size, derivativesCountEffort=1)
        
        self._iPos = self._md.mStateNames2Index['position']
        self._iVel = self._md.mStateNames2Index['velocity']
        self._iTau = self._md.mStateNames2Index['torque']
        self._iMotion = (self._iPos, self._iVel)

        ones = _np.ones(self._md.dofs)
        kp = ones * _np.asarray(kp)
        kd = ones * _np.asarray(kd)
        kv = ones * _np.asarray(kv)

        self.expectedTorqueNoise = _np.asarray(expectedTorqueNoise)
                
        self.desired = _np.zeros((self._md.mechanicalStatesCount,self._md.dofs))
        if desiredPosition is not None:
            self.desired[self._iPos,:] = desiredPosition
        if desiredVelocity is not None:
            self.desired[self._iVel,:] = desiredVelocity
        if desiredTorque is not None:
            self.desired[self._iTau,:] = desiredTorque
        
        if self._md.mechanicalStatesCount != 3:
            raise ImplementationError("Controller not implemented for mstates != 3")

        self.U = _np.zeros((self._md.mechanicalStatesCount, self._md.dofs, self._md.mechanicalStatesCount, self._md.dofs))
        self.K = _np.zeros((self._md.mechanicalStatesCount, self._md.dofs, self._md.mechanicalStatesCount, self._md.dofs))
        self._updateKU(newKp=kp, newKv=kv, newKd=kd)
        self.setDesired(self.desired, self.expectedTorqueNoise)
        
        self.mean = _np.zeros((self._md.mechanicalStatesCount,self._md.dofs))

        

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
            for dof in range(self._md.dofs):
                self.U[:,dof,:,dof] = [
                    [1, self.kp[dof], self.kv[dof]],
                    [0,0,0],
                    [0,0,0],
                ]
                self.K[:,dof,:,dof] = [
                    [0, -self.kp[dof], (-self.kv[dof]-self.kd[dof])],
                    [0,1,0],
                    [0,0,1],
                ]
            self.K_T = _t.T(self.K)
            self.U_T = _t.T(self.U)
            self.setDesired(self.desired, self.expectedTorqueNoise) #trigger update
            
    def setDesiredPosition(self, desiredPosition):
        """
        set a new desired position
        """
        self.desired[self._iPos,:] = _np.asarray(desiredPosition) 
        self.desired[self._iVel,:] = 0
        self.desired[self._iTau,:] = 0
        self.setDesired(self.desired, self.expectedTorqueNoise) #trigger update

    def setDesiredTorque(self, desiredTorque, expectedTorqueNoise):
        """
        set a new desired position
        """
        self.desired[self._iTau,:] = _np.asarray(desiredTorque) 
        self.setDesired(self.desired, self.expectedTorqueNoise) #trigger update

    def setDesired(self, desiredMean, expectedTorqueNoise):
        """updateDesired
        set new desired means and covariances and recompute any dependent internal variables
        """
        self.desired[:,:] = desiredMean
        self.desiredU = _t.dot( self.U, self.desired ) 
        self.expectedTorqueNoise = _np.asarray(expectedTorqueNoise)


    def getInstantStateVectorDistribution(self, phase=None, phaseVelocity=None, currentDistribution=None, phaseVelocitySigma=None):
        """
        modify currentDistribution so it obeys the pd-control law
            (i.e. changes the torque-related means and covariances)
        
           phase: not used (ignored)

           phaseVelocity: not used (ignored)

           currentDistribution: (means, covariances) tuple of multivariate normal distribution
           
           phaseVelocitySigma: not used (ignored)

           returns: (means, covariances) tuple
               means: (derivatives x dofs) array
               covariances: (derivatives x dofs x derivatives x dofs) array
        """
        mean, cov = currentDistribution
        covNext =  _t.dot( self.K, cov , self.K_T ) 
        _t.getDiagView(covNext)[self._iTau,:] += self.expectedTorqueNoise
        mean = _t.dot( self.K, mean ) + self.desiredU
        return mean, covNext
    
    def serialize(self):
        """

        returns a python dictionary that contains all internal data
        necessary to recreate this Controller

        """
        serializedDict = {}
        serializedDict[u'name'] = self.name
        serializedDict[u'dofs'] = self._md.dofs
        serializedDict[u'controller class'] = type(self).__name__
        serializedDict[u'kp'] = self.kp
        serializedDict[u'kv'] = self.kv
        serializedDict[u'kd'] = self.kd
        
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
            filename = '{0}_{1}.controller.mat'.format(_time.strftime('%Y%m%d%H%M%S'), d[u'name'])
        else:
            filename = '{0}.controller.mat'.format(d[u'name']) 
        filepath= _os.path.join(path, filename)
        _h5.write(d, filename=filepath, store_python_metadata=True, matlab_compatible=True)
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
        confidenceColor = '#EEEEEE'
        dt = duration / num 
        t = _np.linspace(0.0, dt*num, num)
        data_mean = _np.empty((num, self._md.mechanicalStatesCount, self._md.dofs))
        data_sigma = _np.empty((num, self._md.mechanicalStatesCount, self._md.dofs))
        if distInitial is None:
            #create a useful initial dist for plotting:
            distInitial = (_np.zeros((self._md.mechanicalStatesCount, self._md.dofs)), _t.makeCovarianceTensorUncorrelated(self._md.mechanicalStatesCount, self._md.dofs, 1.0) )
            distInitial[0][self._iPos,:] = 1.4 # means start position
            distInitial[0][self._iVel,:] = 0.0 #mean start velocity

        a,b = 0.5*_np.min(distInitial[0]), 0.5*_np.max(distInitial[0])
        limits_tightest = {
            'torque': [-1,1],
            'position': [(a+b)-1*(b-a), (a+b) + 1*(b-a) ],
            'velocity': [(a+b)-1*(b-a), (a+b) + 1*(b-a) ],
        }
        limits={}
        for limitname in self._md.mStateNames:
            limits[limitname] = limits_tightest[limitname]


        if dofs=='all':
            dofs=list(range(self._md.dofs))

        mstates = self._md.mechanicalStatesCount
        mStateNames = self._md.mStateNames

        #if no mass matrix inverse is provided, assume a decoupled unit mass system for plotting
        if massMatrixInverseFunc is None:
            massMatrixInverseFunc = lambda q: _np.eye(self._md.dofs)
        #instantiate a time integrator for simulation
        integrator = _tint.TimeIntegrator(self._md.dofs, mstateDescription=self._md)

        subplotfigsize=2.0
        fig, axesArray = _plt.subplots(mstates,len(dofs), squeeze=False, figsize=(max(len(dofs),mstates)*subplotfigsize, mstates*subplotfigsize), sharex='all', sharey='row')

        #compute the distribution over time
        dist = distInitial 
        for i in range(num):
            dist = self.getInstantStateVectorDistribution(currentDistribution=dist)
            dist = integrator.integrate(dist, dt)
            data_mean[i,:] = dist[0]
            for dof in range(self._md.dofs):
                data_sigma[i,:,dof] = _np.sqrt(_np.diag(dist[1][:,dof,:,dof]))


        axesArray.shape = (self._md.mechanicalStatesCount,len(dofs))
        #draw the confidence intervals and means/variance indicators for the supports
        for i, dof in enumerate(dofs):

            #plot the zero-variance trajectory + 95% confidence interval
            if withConfidenceInterval:
                axesArray[ self._iTau, i].fill_between(t, data_mean[:,self._iTau,dof]-1.96*data_sigma[:,self._iTau,dof], data_mean[:,self._iTau,dof]+1.96*data_sigma[:,self._iTau,dof], alpha=1.0, label="95%",  color=confidenceColor)

                axesArray[ self._iPos, i].fill_between(t, data_mean[:,self._iPos,dof]-1.96*data_sigma[:,self._iPos,dof], data_mean[:,self._iPos,dof]+1.96*data_sigma[:,self._iPos,dof], alpha=1.0, label="95%",  color=confidenceColor)
                axesArray[ self._iVel, i].fill_between(t, data_mean[:,self._iVel,dof]-1.96*data_sigma[:,self._iVel,dof], data_mean[:,self._iVel,dof]+1.96*data_sigma[:,self._iVel,dof], alpha=1.0, label="95%",  color=confidenceColor)
                #adjust plot limits
                for limitname in limits:
                    for m in range(data_mean.shape[1]):
                        mstateIndex = self._md.mStateNames2Index[limitname]
                        limits[limitname][0] = min(_np.min(data_mean[:,mstateIndex]-1.96*data_sigma[:,mstateIndex]), limits[limitname][0])
                        limits[limitname][1] = max(_np.max(data_mean[:,mstateIndex]+1.96*data_sigma[:,mstateIndex]), limits[limitname][1])


        #sample the distribution to plot actual trajectories, times the number given by "withSampledTrajectories":
        alpha = _np.sqrt(2.0 / (1+withSampledTrajectories))
        for ax in axesArray.flatten():
            ax.set_prop_cycle(sampledTrajectoryStyleCycler)

        #draw examples of trajectories with sampled initial states:
        meansFlat=distInitial[0].flatten()
        covariancesFlat = _t.flattenCovarianceTensor(distInitial[1])
        for j in range(withSampledTrajectories):
            stateSampled = _np.random.multivariate_normal(meansFlat, covariancesFlat).reshape((self._md.mechanicalStatesCount, self._md.dofs))            
            dist = (stateSampled, _np.zeros((self._md.mechanicalStatesCount, self._md.dofs, self._md.mechanicalStatesCount, self._md.dofs)) )
            for i in range(num):
                dist = self.getInstantStateVectorDistribution(currentDistribution=dist)
                dist = integrator.integrate(dist, dt)
                data_mean[i,:,:] = dist[0]
                for dof in range(self._md.dofs):
                    data_sigma[i,:,dof] = _np.sqrt(_np.diag(dist[1][:,dof,:,dof]))
                
            #update the desired plotting limits:
            for limitname in limits:
                for m in range(data_mean.shape[1]):
                    mstateIndex = self._md.mStateNames2Index[limitname]
                    limits[limitname][0] = min(_np.min(data_mean[:,mstateIndex]-1.96*data_sigma[:,mstateIndex]), limits[limitname][0])
                    limits[limitname][1] = max(_np.max(data_mean[:,mstateIndex]+1.96*data_sigma[:,mstateIndex]), limits[limitname][1])
            #plot all dofs
            for i, dof in enumerate(dofs):
                axesArray[ self._iTau, i].plot(t, data_mean[:,self._iTau,dof], alpha=alpha, linewidth=linewidth )
                axesArray[ self._iPos, i].plot(t, data_mean[:,self._iPos,dof], alpha=alpha, linewidth=linewidth )
                axesArray[ self._iVel, i].plot(t, data_mean[:,self._iVel,dof], alpha=alpha, linewidth=linewidth )

        #override scaling:
        if posLimits is not None:
            limits['position'] = posLimits
        if velLimits is not None:
            limits['velocity'] = velLimits
        if torqueLimits is not None:
            limits['torque']=torqueLimits

        padding=0.05
        for i, dof in enumerate(dofs):
            for m in range(len(mStateNames)):
                axes = axesArray[m,i]  
                axes.set_title(r"{0} {1}".format(mStateNames[m], dof))
                axes.set_xlim((0.0, 1.0))
                lim = limits[mStateNames[m]]
                avg = _np.mean(lim)
                delta = max(0.5*(lim[1]-lim[0]), 0.1*abs(avg))
                axes.set_ylim(avg-delta*(1+padding), avg+delta*(1+padding))
        _plt.tight_layout()


class TaskSpaceController(PDController):
    """
    This class implements the behavior of a task space controller
    
    The man purpose of this class is to evolve a mechanical state distribution according to 
    a PD control law
    
    """
    from scipy.spatial.transform import Rotation as _R #lazy-load as ubuntu 18.04 does not contain it yet
    def __init__(self, kp=10.0, kv=5.0, kd=0.0, name='unnamed', mstateDescription=None, dofs=None, desiredPosition=None, desiredTorque=None, desiredVelocity=None, desiredEEWrench = None, desiredBaseWrench = None, expectedTorqueNoise=0.01 , derivativesCountEffort=1, **kwargs):
        self.name = name
        self.phaseAssociable = False #indicate that this motion generator is not parameterized by phase
        self.timeAssociable = True #indicate that this motion generator is parameterizable by time

        # Is going to be used in getInstantStateVectorDistribution to calc new torques
        self.desiredEEWrench = desiredEEWrench
        self.desiredBaseWrench = desiredBaseWrench

        self.dofsTaskSpace = 6

        #URDF Model 
        import subprocess
        import pandadynamicsmodel
        import os
        home_dir = os.environ['HOME']
        robotDescriptionString = subprocess.check_output(['xacro',home_dir + '/ws_phastapromp/src/franka_ros/franka_description/robots/panda_arm_hand.urdf.xacro'])
        self.urdfModel = pandadynamicsmodel.PandaURDFModel(robotDescriptionString=robotDescriptionString)
        dynamicsModel = pandadynamicsmodel.PandaDynamicsModel()

        
        if mstateDescription is not None:
            self._md = mstateDescription
        else:
            self._md = _MechanicalStateDistributionDescription(dofs=dynamicsModel.dofs, derivativesCountEffort=1)
        
        self._iPos = self._md.mStateNames2Index['position']
        self._iVel = self._md.mStateNames2Index['velocity']
        self._iTau = self._md.mStateNames2Index['torque']
        self._iMotion = (self._iPos, self._iVel)

        # For initialization 
        self.firstTimeDefault = True

        #TODO: What is this for?
        self.lastJaco = _np.zeros((self.dofsTaskSpace,self._md.dofs))

        # TS Transformation objects
        self.meanTrans = self.Trans(self._md.mechanicalStatesCount,self._md.dofs,self.dofsTaskSpace,self._iTau,self._iPos,self._iVel)
        self.currentTrans = self.Trans(self._md.mechanicalStatesCount,self._md.dofs,self.dofsTaskSpace,self._iTau,self._iPos,self._iVel)

        # Task Space Control Error
        self.errTS = _np.zeros((self._md.mechanicalStatesCount, self.dofsTaskSpace))


        # Transform desired position to taskspace
        self.desiredTransform = None
        if self.urdfModel is not None and desiredPosition is not None:
            self.urdfModel.setJointPosition(desiredPosition)
            self.desiredTransform = self.urdfModel.getEELocation()

        ones = _np.ones(self.dofsTaskSpace)
        initkp = ones * _np.asarray(kp)
        initkd = ones * _np.asarray(kd)
        initkv = ones * _np.asarray(kv)
        
        dofweighingvector = _np.ones(self._md.dofs)[_np.newaxis,:]

        self.expectedTorqueNoise = _np.asarray(expectedTorqueNoise)
                
        self.desired = _np.zeros((self._md.mechanicalStatesCount,self._md.dofs))
        if desiredPosition is not None:
            self.desired[self._iPos,:] = desiredPosition
        if desiredVelocity is not None:
            self.desired[self._iVel,:] = desiredVelocity
        if desiredTorque is not None:
            self.desired[self._iTau,:] = desiredTorque
        
        if self._md.mechanicalStatesCount != 3:
            raise ImplementationError("Controller not implemented for mstates != 3")

        self.U = _np.zeros((self._md.mechanicalStatesCount, self.dofsTaskSpace, self._md.mechanicalStatesCount, self.dofsTaskSpace))
        self.K = _np.zeros((self._md.mechanicalStatesCount, self.dofsTaskSpace, self._md.mechanicalStatesCount, self.dofsTaskSpace))
        self._updateKU(newKp=initkp, newKv=initkv, newKd=initkd)
        self.setDesired(self.desired, self.expectedTorqueNoise)
        
        self.mean = _np.zeros((self._md.mechanicalStatesCount,self._md.dofs))
        self.meanLast = _np.copy(self.mean)

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
            for dof in range(self.dofsTaskSpace):
                self.U[:,dof,:,dof] = [
                    [1, self.kp[dof], self.kv[dof]],
                    [0,0,0],
                    [0,0,0],
                ]
                self.K[:,dof,:,dof] = [
                    [0, -self.kp[dof], (-self.kv[dof]-self.kd[dof])],
                    [0,1,0],
                    [0,0,1],
                ]
            self.K_T = _t.T(self.K)
            self.U_T = _t.T(self.U)
            self.setDesired(self.desired, self.expectedTorqueNoise) #trigger update

    class Trans:
        def __init__(self,mechanicalStatesCount,dofs,dofsTaskSpace,iTau,iPos,iVel):
            self._iTau = iTau
            self._iVel = iVel
            self._iPos = iPos
            self.dofs = dofs
            #Jacobians
            self.jacobianEE = _np.zeros((dofsTaskSpace,dofs))
            self.jacobianBase = _np.zeros((dofsTaskSpace,dofs))
            # Task space Koint space Transformation Matrix
            self.T = _np.zeros((mechanicalStatesCount, dofs, mechanicalStatesCount, dofsTaskSpace))
            # Inverse Task space Koint space Transformation Matrix 
            self.T_I = _np.zeros((mechanicalStatesCount, dofsTaskSpace, mechanicalStatesCount, dofs))
            # Transposed Task space Koint space Transformation Matrix 
            self.T_T  = _np.zeros((mechanicalStatesCount, dofsTaskSpace, mechanicalStatesCount, dofs))
            # Transposed Inverse Task space Koint space Transformation Matrix
            self.T_I_T = _np.zeros((mechanicalStatesCount,dofs, mechanicalStatesCount, dofsTaskSpace))

        def _updateT(self, jacobianEE):
            """
            update the transformation matrix
            """
            #first, save values and decide on what to recompute
            updateAll=False
            if jacobianEE is not None:
                self.jacobianEE[:,:jacobianEE.shape[1]] = jacobianEE # Extend Jacobian
                _jacobian = self.jacobianEE
                updateAll = True
            else:
                raise TypeError("Endeffector Jacobian is None")
            
            #recompute transformation matrices :
            if updateAll:
                _jacobianT = _jacobian.T
                _jacobianPinv = _np.linalg.pinv(_jacobian, rcond=1e-10)
                _jacobianPinvT = _jacobianPinv.T
                self.T[self._iTau,:,self._iTau,:] = _jacobianT
                self.T[self._iPos,:,self._iPos,:] = _jacobianPinv
                self.T[self._iVel,:,self._iVel,:] = _jacobianPinv
                self.T_T = _t.T(self.T)
                self.T_I[self._iTau,:,self._iTau,:] = _jacobianPinvT
                self.T_I[self._iPos,:,self._iPos,:] = _jacobian
                self.T_I[self._iVel,:,self._iVel,:] = _jacobian
                self.T_I_T = _t.T(self.T_I)
                self.nullSpaceProjector = _np.eye(self.dofs) - _np.matmul(_jacobianT,_jacobianPinvT)

    def setDesired(self, desiredMean, expectedTorqueNoise):
        """updateDesired
        set new desired means and covariances and recompute any dependent internal variables
        """
        self.desired[:,:] = desiredMean
        self.expectedTorqueNoise = _np.asarray(expectedTorqueNoise)

    def getInstantStateVectorDistribution(self, phase=None, phaseVelocity=None, currentDistribution=None, phaseVelocitySigma=None, hTransformGoal=None, jacobianCallback = None, hTransformCallback = None, qCallback = None):
        """
        modify currentDistribution so it obeys the pd-control law
            (i.e. changes the torque-related means and covariances)
        
           phase: not used (ignored)

           phaseVelocity: not used (ignored)

           currentDistribution: (means, covariances) tuple of multivariate normal distribution
           
           phaseVelocitySigma: not used (ignored)

           jacobianEE: np.ndarray of shape (6,7) containing current jacobian in ee frame

           jacobianBase: np.ndarray of shape (6,7) containing current jacobian in base frame

           hTransform: np.ndarray of shape (4,4) containing current base ee transform 

           returns: (means, covariances) tuple
               means: (derivatives x dofs) array
               covariances: (derivatives x dofs x derivatives x dofs) array
        """
        mean, cov = currentDistribution
        if self.desiredTransform is None or self.urdfModel is None:
            self.meanLast = mean
            return mean, cov

        if hTransformGoal is not None:
            self.desiredTransform = hTransformGoal

        # Get EE Transform in Base frame  
        self.urdfModel.setJointPosition(mean[self._iPos,:])
        _meanHTransform = self.urdfModel.getEELocation()

        currentHTransform = None
        if hTransformCallback is not None: 
            _hTransformCallback = hTransformCallback()
            if _hTransformCallback is not None: 
                currentHTransform = _hTransformCallback

        currentQ = None
        if qCallback is not None: 
            _qCallback = qCallback()
            if _qCallback is not None: 
                currentQ = _qCallback

        # Calculate EE Jacobian from Base Jacobian (5.22) in Modern Robotics 
        
        # Get transform from EE to Base
        _meanHTransformI = _np.linalg.inv(_meanHTransform)

        # Get Base Jacobian at mean pose 
        jacoKDLBase = self.urdfModel.getJacobian()

        # From (3.30) in Modern Robotics
        hTranslateSkewSym = _np.array([ [0,-_meanHTransformI[2,3],_meanHTransformI[1,3]],
                                        [_meanHTransformI[2,3], 0, -_meanHTransformI[0,3]],
                                        [-_meanHTransformI[1,3], _meanHTransformI[0,3], 0]])
        
        # From Definition 3.20 in modern robotics
        _meanHTransformIAdjoint =  _np.vstack((_np.hstack((_meanHTransformI[0:3,0:3], _np.zeros((3,3)))), _np.hstack((_np.matmul(hTranslateSkewSym,_meanHTransformI[0:3,0:3]),_meanHTransformI[0:3,0:3]))))
        _meanJacobianEE = _np.matmul(_meanHTransformIAdjoint,jacoKDLBase)

        if jacobianCallback is not None:
            _currentJacobian = jacobianCallback()
            if _currentJacobian is not None:
                if self.firstTimeDefault == False:
                    print("Getting live values now.")
                    self.firstTimeDefault = True
                #self._updateT(jacobianEE=_currentJacobian)
                _currentJacobianEE = _np.zeros((6,self._md.dofs))
                _currentJacobianEE[:,:_currentJacobian.shape[1]] += _currentJacobian
            else:
                if self.firstTimeDefault:
                    print("Running Task Space Controller with default values. Waiting for topics.")
                    self.firstTimeDefault = False
                meanNext = mean.copy()
                meanNext[self._iTau,:] = 0.0
                meanNext[self._iVel,:] = 0.0
                covNext = cov.copy()
                covNext[self._iTau,:,:,:] = 0.0
                covNext[:,:,self._iTau,:] = 0.0
                _t.getDiagView(covNext)[self._iTau,:] += self.expectedTorqueNoise
                return meanNext, covNext

        self.meanTrans._updateT(jacobianEE=_meanJacobianEE)
        Kj = _t.dot( self.meanTrans.T, self.K, self.meanTrans.T_I ) # map gains to joint space
        covNext =  _t.dot(Kj,cov,_t.T(Kj)) # transform to joint space
        _t.getDiagView(covNext)[self._iTau,:] += self.expectedTorqueNoise
        
        # Add noise to joint space distribution
        _t.getDiagView(covNext)[:,:] += 0.0001 

        # Add variance to joint space
        #covNext[self._iTau,:,self._iTau,:] += _t.T(self.meanTrans.nullSpaceProjector)*10
        #covNext[self._iPos,:,self._iPos,:] += self.meanTrans.nullSpaceProjector*0.1
        #covNext[self._iVel,:,self._iVel,:] += self.meanTrans.nullSpaceProjector*0.1

        desiredTransformEE = _np.matmul(_meanHTransformI, self.desiredTransform) # Get transformation from EE to Desired 

        #for task frame == reference frame:
        refJointSpace = mean
        refJointSpace[self._iTau,:] = 0.0
        refTaskSpace = 0
        currentJointSpace = mean
        desiredPosition  = desiredTransformEE[0:3,3] # get translation
        desiredRotation = self._R.as_rotvec(self._R.from_dcm(desiredTransformEE[0:3,0:3]))


        currentTaskSpace =  _t.dot(self.meanTrans.T_I, currentJointSpace - refJointSpace) - refTaskSpace
        desiredTaskSpace = _np.zeros_like(currentTaskSpace)
        desiredTaskSpace[self._iPos,0:3] = desiredPosition
        desiredTaskSpace[self._iPos,3:6] = desiredRotation
        desiredTaskSpace[self._iVel,:] = -1 *  _t.dot(self.meanTrans.T_I, currentJointSpace)[self._iVel,:]

        expectedTaskSpace =  _t.dot( self.K, currentTaskSpace) + _t.dot( self.U, desiredTaskSpace) 
        expectedDeltaTaskSpace = expectedTaskSpace - currentTaskSpace
        expectedDeltaJointSpace = _t.dot(self.meanTrans.T, expectedDeltaTaskSpace )
        meanNext = mean + expectedDeltaJointSpace

        # Current Pose
        # if currentHTransform is not None and currentQ is not None and False:
        #     self.currentTrans._updateT(jacobianEE=_currentJacobianEE)
        #     currentErrorTransform = _np.matmul(_meanHTransformI, currentHTransform)
        #     currentError = _np.zeros_like(errorTIEE)
        #     currentError[self._iPos,0:3] = currentErrorTransform[0:3,3]
        #     currentError[self._iPos,3:6] = _R.as_rotvec(_R.from_dcm(currentErrorTransform[0:3,0:3]))
        #     currentForceNext = -1 * _t.dot( self.K, currentError)
        #     #meanNext[self._iTau,:] += _np.matmul(_currentJacobianEE.T,_t.dot( self.K, currentError)[self._iTau,:]) * 0.1
        #     currentMeanNext = _t.dot(self.currentTrans.T, currentForceNext )
        #     meanNext[self._iTau,:] += 0.01 * currentMeanNext[self._iTau,:]
        #     # currentNullSpaceVec = _np.matmul(self.currentTrans.nullSpaceProjector,currentQ)
        #     # meanNullSpaceVec = _np.matmul(self.meanTrans.nullSpaceProjector,meanNext[self._iPos,:])
        #     # u, s, vh = _np.linalg.svd(_np.row_stack([currentNullSpaceVec[:7], meanNullSpaceVec[:7]]), full_matrices=False)
        #     # linearDependency = s           
        #     # print(currentNullSpaceVec)
        #     # print(meanNullSpaceVec)
        #     # print(s)
        #     #meanNext[self._iPos,:] = meanNext[self._iPos,:] + 0.1 * (currentQ - meanNext[self._iPos,:])
        return meanNext, covNext

    def plot(self, dofs='all',
                   distInitial = None,
                   num=1000,
                   duration = 1.0,
                   linewidth=2.0,
                   withSampledTrajectories=100,
                   withConfidenceInterval=True,
                   posLimits=None,
                   velLimits=None,
                   torqueLimits=None,
                   sampledTrajectoryStyleCycler=_plt.cycler('color', ['#0000FF']),
                   massMatrixInverseFunc = None,
                   dynamicsModel=None,
                   urdfModel=None
                   ):
        """
        visualize the trajectory distribution as parameterized by the means of each via point,
        and the covariance matrix, which determins the variation at each via (sigma) and the
        correlations between variations at each sigma

        E.g. usually, neighbouring via points show a high, positive correlation due to dynamic acceleration limits


        interpolationKernel:  provide a function that defines the linear combination vector for interpolating the via points
                                If set to None (default), promp.makeInterpolationKernel() is used to construct a Kernel as used in the ProMP definition
        """
        confidenceColor = '#EEEEEE'
        dt = duration / num 
        t = _np.linspace(0.0, dt*num, num)
        data_mean = _np.empty((num, self._md.mechanicalStatesCount, self._md.dofs))
        data_gains = _np.empty((num, 2, self._md.dofs))
        data_sigma = _np.empty((num, self._md.mechanicalStatesCount, self._md.dofs))
        if distInitial is None:
            #create a useful initial dist for plotting:
            distInitial = (_np.zeros((self._md.mechanicalStatesCount, self._md.dofs)), _t.makeCovarianceTensorUncorrelated(self._md.mechanicalStatesCount, self._md.dofs, 1.0) )
            distInitial[0][self._iPos,:] = 1.4 # means start position
            distInitial[0][self._iVel,:] = 0.0 #mean start velocity

        a,b = 0.5*_np.min(distInitial[0]), 0.5*_np.max(distInitial[0])
        limits_tightest = {
            'torque': [-1,1],
            'position': [(a+b)-1*(b-a), (a+b) + 1*(b-a) ],
            'velocity': [(a+b)-1*(b-a), (a+b) + 1*(b-a) ],
        }
        limits={}
        for limitname in self._md.mStateNames:
            limits[limitname] = limits_tightest[limitname]


        if dofs=='all':
            dofs=list(range(self._md.dofs))

        mstates = self._md.mechanicalStatesCount
        mStateNames = self._md.mStateNames

        #if no mass matrix inverse is provided, assume a decoupled unit mass system for plotting
        if massMatrixInverseFunc is None:
            massMatrixInverseFunc = lambda q: _np.eye(self._md.dofs)
        #instantiate a time integrator for simulation
        integrator = _tint.TimeIntegrator(self._md.dofs, mstateDescription=self._md , dynamicsModel=dynamicsModel)

        subplotfigsize=2.0
        fig, axesArray = _plt.subplots(mstates+1,len(dofs), squeeze=False, figsize=(max(len(dofs),mstates)*subplotfigsize, mstates*subplotfigsize), sharex='all', sharey='row')

        #compute the distribution over time
        dist = distInitial 
        for i in range(num):
            if urdfModel is not None:
                urdfModel.setJointPosition(dist[0][self._iPos])
                dist = self.getInstantStateVectorDistribution(currentDistribution=dist)
            else:
                dist = self.getInstantStateVectorDistribution(currentDistribution=dist)
            dist = integrator.integrate(dist, dt)
            data_mean[i,:] = dist[0]
            for dof in range(self._md.dofs):
                #with _np.errstate(invalid='raise'):
                data_sigma[i,:,dof] = _np.sqrt(_np.diag(dist[1][:,dof,:,dof]))
                data_gains[i,:,dof] = extractPDGains(dist[1])[dof,:,dof]


        axesArray.shape = (self._md.mechanicalStatesCount+1,len(dofs))
        #draw the confidence intervals and means/variance indicators for the supports
        for i, dof in enumerate(dofs):

            #plot the zero-variance trajectory + 95% confidence interval
            if withConfidenceInterval:
                axesArray[ self._iTau, i].fill_between(t, data_mean[:,self._iTau,dof]-1.96*data_sigma[:,self._iTau,dof], data_mean[:,self._iTau,dof]+1.96*data_sigma[:,self._iTau,dof], alpha=1.0, label="95%",  color=confidenceColor)

                axesArray[ self._iPos, i].fill_between(t, data_mean[:,self._iPos,dof]-1.96*data_sigma[:,self._iPos,dof], data_mean[:,self._iPos,dof]+1.96*data_sigma[:,self._iPos,dof], alpha=1.0, label="95%",  color=confidenceColor)
                axesArray[ self._iVel, i].fill_between(t, data_mean[:,self._iVel,dof]-1.96*data_sigma[:,self._iVel,dof], data_mean[:,self._iVel,dof]+1.96*data_sigma[:,self._iVel,dof], alpha=1.0, label="95%",  color=confidenceColor)
                #adjust plot limits
                for limitname in limits:
                    for m in range(data_mean.shape[1]):
                        mstateIndex = self._md.mStateNames2Index[limitname]
                        limits[limitname][0] = min(_np.min(data_mean[:,mstateIndex]-1.96*data_sigma[:,mstateIndex]), limits[limitname][0])
                        limits[limitname][1] = max(_np.max(data_mean[:,mstateIndex]+1.96*data_sigma[:,mstateIndex]), limits[limitname][1])


        #sample the distribution to plot actual trajectories, times the number given by "withSampledTrajectories":
        alpha = _np.sqrt(2.0 / (1+withSampledTrajectories))
        for ax in axesArray.flatten():
            ax.set_prop_cycle(sampledTrajectoryStyleCycler)

        #draw examples of trajectories with sampled initial states:
        meansFlat=distInitial[0].flatten()
        covariancesFlat = _t.flattenCovarianceTensor(distInitial[1])
        for j in range(withSampledTrajectories):
            stateSampled = _np.random.multivariate_normal(meansFlat, covariancesFlat).reshape((self._md.mechanicalStatesCount, self._md.dofs))            
            dist = (stateSampled, _np.zeros((self._md.mechanicalStatesCount, self._md.dofs, self._md.mechanicalStatesCount, self._md.dofs)) )
            for i in range(num):
                if urdfModel is not None:
                    urdfModel.setJointPosition(dist[0][self._iPos])
                    dist = self.getInstantStateVectorDistribution(currentDistribution=dist)
                else:
                    dist = self.getInstantStateVectorDistribution(currentDistribution=dist)
                if _np.any(dist[0] > 0.2):
                    pass 
                dist = integrator.integrate(dist, dt)
                data_mean[i,:,:] = dist[0]
                for dof in range(self._md.dofs):
                    data_sigma[i,:,dof] = _np.sqrt(_np.diag(dist[1][:,dof,:,dof]))
                    data_gains[i,:,dof] = extractPDGains(dist[1])[dof,:,dof]
                
            #update the desired plotting limits:
            for limitname in limits:
                for m in range(data_mean.shape[1]):
                    mstateIndex = self._md.mStateNames2Index[limitname]
                    limits[limitname][0] = min(_np.min(data_mean[:,mstateIndex]-1.96*data_sigma[:,mstateIndex]), limits[limitname][0])
                    limits[limitname][1] = max(_np.max(data_mean[:,mstateIndex]+1.96*data_sigma[:,mstateIndex]), limits[limitname][1])
            #plot all dofs
            for i, dof in enumerate(dofs):
                axesArray[ self._iTau, i].plot(t, data_mean[:,self._iTau,dof], alpha=alpha, linewidth=linewidth )
                axesArray[ self._iPos, i].plot(t, data_mean[:,self._iPos,dof], alpha=alpha, linewidth=linewidth )
                axesArray[ self._iVel, i].plot(t, data_mean[:,self._iVel,dof], alpha=alpha, linewidth=linewidth )
                axesArray[ len(mStateNames), i].plot(t, data_gains[:,0,dof], alpha=alpha, linewidth=linewidth )

        #override scaling:
        if posLimits is not None:
            limits['position'] = posLimits
        if velLimits is not None:
            limits['velocity'] = velLimits
        if torqueLimits is not None:
            limits['torque']=torqueLimits

        for i, dof in enumerate(dofs):
            for m in range(len(mStateNames)):
                axes = axesArray[m,i]  
                axes.set_title(r"{0} {1}".format(mStateNames[m], dof))
                axes.set_xlim((0.0, duration))
                lim = limits[mStateNames[m]]
                avg = _np.mean(lim)
                delta = max(0.5*(lim[1]-lim[0]), 0.1*abs(avg))
                #axes.set_ylim(avg-delta*(1+padding), avg+delta*(1+padding))
            # gains
            axes = axesArray[len(mStateNames),i]
            axes.set_title(r"kp gain {0}".format(dof))
            axes.set_xlim((0.0, duration))
        _plt.tight_layout()
        return True 


 

