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

import namedtensors as _nt


class TimeIntegrator(object):

    def __init__(self, tensornamespace=None, noiseFloorSigmaTorque=0.01, noiseFloorSigmaPosition=0, noiseFloorSigmaVelocity=0, dynamicsModel = None):
        """
        
        dynamicsModel: object that provides a getInertiaMatrix(position) method to query the inertia matrix of the system to integrate
        """
        
        if tensornamespace is not None:
            self.tns = _nt.TensorNameSpace(tensornamespace)
        else:
            self.tns = _nt.TensorNameSpace(tensornamespace)
            self.tns.registerIndex('r',2, ('motion', 'effort'))
            self.tns.registerIndex('g',2)
            self.tns.registerIndex('d',1)
           
        self.tns.registerTensor('A_template', ('r_','g_''d_'),('r', 'g','d'), initial_values='identity' )
        self.tns.registerTensor('A', ('r','g''d'),('r_', 'g_','d_') )
        self.tns.registerTensor('noiseFloorCov', ('r_','g_''d_'),('r', 'g','d'))
        self.tns.tensorData['noiseFloorCov'][0,0,:,0,0,:] = noiseFloorSigmaPosition**2
        self.tns.tensorData['noiseFloorCov'][0,1,:,0,1,:] = noiseFloorSigmaVelocity**2
        self.tns.tensorData['noiseFloorCov'][1,0,:,1,0,:] = noiseFloorSigmaTorque**2
        #self.tns.tensorData['noiseFloorCov'][1,1,:,1,1,:] = noiseFloorSigmaTorqueRate**2

        self.tns.setTensor('meansCurrent', ('r','g''d'),())
        self.tns.setTensor('covCurrent', ('r','g''d'),('r_', 'g_','d_'))

        self.tns.registerInverse('A')
        self.tns.registerContraction('A', 'meansCurrent', result_name='meansNext')
        self.tns.registerContraction('(A)^T', 'covCurrent')
        self.tns.registerContraction('(A)^T:covCurrent', 'A')
        #this is a very simple "plant model" to account for limits in execution.
        #it also avoid unrealistic convergence to zero variances        
        self.tns.registerAddition('(A)^T:covCurrent:A', 'noiseFloorCov', result_name='covNext')  


        self.tns.registerInverse('covNext', flip_underlines=True)
        self.tns.registerAddition('covNext','(covNext)^T')
        self.tns.registerScalarMultiplication('(covNext+(covNext)^T)', 0.5, result_name = 'covNext_resymmetrized')
            
        self.substepMaxLength = 1.0#0.0051 #0.0501 # maximum integration time step to avoid bad Euler integration. 
        self.dt = _np.inf
        self.dt2 = _np.inf
        
        
        if dynamicsModel is None:
            self.dynamicsModel = FakeDynamicsModel(self.tns.indexSizes['d'])
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

            self.tns.setTensor('meansNext',  mStateDistribution.means)
            self.tns.setTensor('covNext',  mStateDistribution.covariances)
            
            #get dynamics parameters:
            pos =  mStateDistribution.means[mStateDistribution.name2rg['position']]
            L = self.dynamicsModel.getInertiaMatrix()
            eta = self.dynamicsModel.getViscuousFrictionCoefficients(mStateDistribution.means)
            Linv = _np.linalg.inv(L)
            Linv_eta = Linv * eta[:,_np.newaxis]  # = Linv dot diag(eta)

            self.tns.tensorData['A']
            #update mass matrix dependent elements:
            postau = self.tns.makeSliceDef('A', {'r':0, 'g':0, 'r_': 1, 'g_': 0})
            veltau = self.tns.makeSliceDef('A', {'r':0, 'g':1, 'r_': 1, 'g_': 0})
            posvel = self.tns.makeSliceDef('A', {'r':0, 'g':0, 'r_': 0, 'g_': 1})
            velvel = self.tns.makeSliceDef('A', {'r':0, 'g':1, 'r_': 0, 'g_': 1})
            tauvel = self.tns.makeSliceDef('A', {'r':1, 'g':0, 'r_': 0, 'g_': 1})
            self.tns.tensorData['A'][postau]  = Linv * self.dt2  #could be more elegantly formulated with a separate tensor for dt-mapping/scaling of Linv
            self.tns.tensorData['A'][veltau]  = Linv * self.dt
            #add inertial motion:
            for dof in range(self.tns.indexSizes['d']):
                self.tns.tensorData['A'][posvel] = self.dt
            #add viscuous friction
            self.tns.tensorData['A'][tauvel] -= Linv_eta 
            self.tns.tensorData['A'][posvel] -= Linv_eta * self.dt2
            self.tns.tensorData['A'][velvel] -= Linv_eta * self.dt
            #integrate:
            for i in range(substeps):   
                self.setTensor('meansCurrent', self.tns.tensorData['meansNext'],self.tns.tensorIndices['meansNext'])
                self.setTensor('covCurrent', self.tns.tensorData['covNext'],self.tns.tensorIndices['covNext'])
                self.tns.update()

            if _np.any(self.tns.tensorData['covNext'] > 1e10):
                raise RuntimeWarning("TimeIntegrator: covariance matrix has elements > 1e10!")

            #to ensure continued symmetry of the covariance matrix:
            self.tns.tensorData['covNext'] = 0.5*(self.tns.tensorData['covNext'] + self.tns.tensorData['(covNext)^T'])
            covNext_resymmetrized
            
            return MechanicalStateDistribution(self.tns.tensorData['meansNext'], self.tns.tensorData['covNext_resymmetrized']) 


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
    

