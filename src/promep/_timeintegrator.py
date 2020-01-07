#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel
@copyright 2018
@licence: 2-clause BSD licence

This file contains a class for evolving mechanical state distributions over time using Euler integration

"""
import warnings
import numpy as _np
import itertools as _it

from . import _tensorfunctions as _t
from . import MechanicalStateDistributionDescription as _MechanicalStateDistributionDescription



class TimeIntegrator(object):

    def __init__(self, dofs, mstateDescription=None, noiseFloorSigmaTorque=0.01, noiseFloorSigmaPosition=0, noiseFloorSigmaVelocity=0, dynamicsModel = None):
        """
        
        dofs: number of dofs of the system
        dynamicsModel: object that provides a getInertiaMatrix(position) method to query the inertia matrix of the system to integrate
        """
        if mstateDescription is not None:
            self._md = mstateDescription
        else:
            self._md = _MechanicalStateDistributionDescription(dofs=dofs, derivativesCountEffort=1) #assume default assignment

        self._iPos = self._md.mStateNames2Index['position']
        self._iVel = self._md.mStateNames2Index['velocity']
        self._iTau = self._md.mStateNames2Index['torque']

        self.A_template = _t.I(self._md.mechanicalStatesCount, self._md.dofs) #matrix to estimate the current mechanical state from the last time step
        self.substepMaxLength = 1.0#0.0051 #0.0501 # maximum integration time step to avoid bad Euler integration. 
        self.dt = _np.inf
        self.dt2 = _np.inf
        
        self.noiseCov = _np.zeros((self._md.mechanicalStatesCount, self._md.dofs,self._md.mechanicalStatesCount, self._md.dofs))

        self.noiseFloorSigma = _np.zeros((self._md.mechanicalStatesCount, self._md.dofs))
        self.noiseFloorSigma[self._iTau,:] = noiseFloorSigmaTorque**2
        self.noiseFloorSigma[self._iPos,:] = noiseFloorSigmaPosition**2
        self.noiseFloorSigma[self._iVel,:] = noiseFloorSigmaVelocity**2
        
        if dynamicsModel is None:
            self.dynamicsModel = FakeDynamicsModel(dofs)
            print("TimeIntegrator: Using fake mass matrix for time integration.")
        else:
            self.dynamicsModel = dynamicsModel
    
    def integrate(self, mStateDistribution, duration):
            """
            
            Integrate the time using Euler integration, expressed as a matrix multiplication:

                       |   1      0     0 |
                  A =  | dt^2/m   1    dt |
                       |  dt/m    0     1 |

                x(t+dt) = A x(t)

             and C(t+dt) = A C(t) A^T


            In order to include viscuous friction (eta = F_friction / Velocity):
                       |   1      0     0-eta          |
                  A =  | dt^2/m   1    dt-(eta/m)*dt^2 |
                       |  dt/m    0     1-(eta/m)*dt   |

            
            """
            substeps = 1+int(duration/self.substepMaxLength) #make sure we don't do too large steps when integrating
            dt_substeps = duration/substeps
            if abs(self.dt - dt_substeps) > 1e-6: #update the elements dependent on timestep:
                    self.dt = dt_substeps
                    self.dt2 = 0.5*self.dt*self.dt

            meansCurrent, covCurrent = mStateDistribution
            
            #get dynamics parameters:
            L = self.dynamicsModel.getInertiaMatrix(meansCurrent[self._iPos,:])
            eta = self.dynamicsModel.getViscuousFrictionCoefficients(meansCurrent)
            Linv = _np.linalg.inv(L)
            Linv_eta = Linv * eta[:,_np.newaxis]  # = Linv dot diag(eta)

            self.A = self.A_template.copy()
            #update mass matrix dependent elements:
            self.A[self._iPos,:,self._iTau,:]  = Linv * self.dt2
            self.A[self._iVel,:,self._iTau,:]  = Linv * self.dt
            #add inertial motion:
            for dof in range(self._md.dofs):
                self.A[self._iPos,dof,self._iVel,dof] = self.dt
            #add viscuous friction
            self.A[self._iTau,:,self._iVel,:] -= Linv_eta 
            self.A[self._iPos,:,self._iVel,:] -= Linv_eta * self.dt2
            self.A[self._iVel,:,self._iVel,:] -= Linv_eta * self.dt
            A_T = _t.T(self.A)
            #integrate:
            for i in range(substeps):
                meansCurrent = _t.dot(self.A, meansCurrent, shapes=((2,2),(2,0)))
                covCurrent = _t.dot(self.A, covCurrent, A_T, shapes=((2,2),(2,2),(2,2))) 
                #this is a very simple "plant model" to account for limits in execution.
                #it also avoid unrealistic convergence to zero variances
                _t.getDiagView(covCurrent)[:,:] += self.noiseFloorSigma

            if _np.any(covCurrent > 1e10):
                raise RuntimeWarning("TimeIntegrator: covariance matrix has elements > 1e10!")

            #to ensure continued symmetry of the covariance matrix:
            covCurrent = 0.5*(covCurrent + _t.T(covCurrent))
            return (meansCurrent, covCurrent) 


class FakeDynamicsModel():
    """
    placeholder class if user does not provide any dynamics of the system to integrate with
    
    This class is primarily intended for checking code and test plotting.
    """
    def __init__(self, dofs):
        self.L = _np.eye(dofs)
        self.viscuousFriction = _np.ones((dofs))
        
    def getInertiaMatrix(self, position):
        return self.L

    def getViscuousFrictionCoefficients(self, jointState=None):
        return self.viscuousFriction
    

