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

from . import _mechanicalstate


class TimeIntegrator(object):

    def __init__(self, tensornamespace, noiseFloorSigmaTorque=0.01, noiseFloorSigmaPosition=0, noiseFloorSigmaVelocity=0, dynamicsModel = None):
        """
        
        dynamicsModel: object that provides a getInertiaMatrix(position) method to query the inertia matrix of the system to integrate
        
        
        
        
        
        Integrate the time using Euler integration, expressed as a matrix multiplication:
    
                     pos       vel            impulse      torque 
                  |   1    dt - eta*Linv*dt²     0      dt²/2 * Linv  |   pos
             A =  |   0    1 - eta*Linv*m*dt     0      dt    * Linv  |   vel
                  |   0        0                 1            0       |   impulse
                  |   0        0                 0            1       |   torque


            or, with basis tensors:
            
             Linv = inertia inverse
              A_newton =    delta^rgd_rgd                         #identity
                       + e(pos,       vel) : (delta^d_d * dt)       #effect of constant velocity on position (integrate)
                       + e(impulse,torque) : (delta^d_d * dt)       #effect of constant torque on impulse (integrate)
                       + e(pos,    torque) : (     Linv * dt²/2)    #effect of constant torque on position
                       + e(vel,    torque) : (     Linv * dt)       #effect of constant torque on position

                                                                  #damping/viscuous friciton
                       - e(pos, vel) : eta * (     Linv * dt²/2)    #change of position
                       - e(vel, vel) : eta * (     Linv * dt)       #change of velocity
                       



                To integrate over dt:
                Mean:           M(t+dt) = A:M(t) 
                Covariances:    C(t+dt) = A:C(t):A^T

        
        """
        
        self.tns = _nt.TensorNameSpace(tensornamespace)
        
        if self.tns.indexSizes['d'] > 2:
            NotImplementedError()
        
        self.tns.registerTensor('LastMean', (('r','g','d'),()))
        self.tns.registerTensor('LastCov', (('r','g','d'),('r_', 'g_','d_')))
        self.msd_last  = _mechanicalstate.MechanicalStateDistribution(self.tns, 'LastMean', 'LastCov')
        
        
        #create all the basis tensors we need:
        rgrg= (('r_', 'g_'),('r', 'g'))
        name2rg  = self.msd_last.commonnames2rg
        self.tns.registerBasisTensor( 'e_pos_vel', rgrg, (name2rg['position'],name2rg['velocity']), ignoreLabels=True )
        self.tns.registerBasisTensor( 'e_imp_tau', rgrg, (name2rg['impulse'], name2rg['torque']), ignoreLabels=True )
        self.tns.registerBasisTensor( 'e_pos_tau', rgrg, (name2rg['position'],name2rg['torque']), ignoreLabels=True )
        self.tns.registerBasisTensor( 'e_vel_tau', rgrg, (name2rg['velocity'],name2rg['torque']), ignoreLabels=True )
        self.tns.registerBasisTensor( 'e_vel_vel', rgrg, (name2rg['velocity'],name2rg['velocity']), ignoreLabels=True )
        
        dd= (('d_',),('d',))
        self.tns.registerTensor('delta_dd',dd, initial_values='identity') # delta^d_d 

        #inputs:
        self.tns.registerTensor('Linv',dd) # Linv
        self.tns.registerTensor('dt',((),()) ) # dt scalar
        self.tns.registerTensor('eta_neg',((),()) ) # damping factor (negative)

        self.tns.registerTensor('noiseFloorCov', (('r_','g_','d_'),('r', 'g','d')))
        self.tns.tensorData['noiseFloorCov'][0,0,:,0,0,:] = noiseFloorSigmaPosition**2
        self.tns.tensorData['noiseFloorCov'][0,1,:,0,1,:] = noiseFloorSigmaVelocity**2
        self.tns.tensorData['noiseFloorCov'][1,0,:,1,0,:] = noiseFloorSigmaTorque**2
        #self.tns.tensorData['noiseFloorCov'][1,1,:,1,1,:] = noiseFloorSigmaTorqueRate**2



        #computation:
        sum_terms_noLinv = []  #all terms that will be summed up to yield 'A'

        dt_squared = self.tns.registerElementwiseMultiplication('dt','dt') 
        dt2 = self.tns.registerScalarMultiplication(dt_squared , 0.5, result_name='dt2')  #dt²/2 scalar
        Idt     = self.tns.registerScalarMultiplication('delta_dd','dt') # Linv*dt
        Idt2    = self.tns.registerScalarMultiplication('delta_dd','dt2') # Linv*dt²/2

        
        sum_terms_noLinv.append(self.tns.registerContraction('e_pos_vel', Idt))
        sum_terms_noLinv.append(self.tns.registerContraction('e_imp_tau', Idt))

        self.tns.registerSum(*sum_terms_noLinv, result_name='A_noLinv')
        
        #from here on, Linv influences computation:
        index_first_equation_no_dt_change = len(self.tns.update_order)

        sum_terms_Linv = []  #all terms that will be summed up to yield 'A'

        Linvdt2 = self.tns.registerScalarMultiplication('Linv','dt2') # Linv*dt²/2
        sum_terms_Linv.append(self.tns.registerContraction('e_pos_tau', Linvdt2))

        Linvdt  = self.tns.registerScalarMultiplication('Linv','dt') # Linv*dt
        sum_terms_Linv.append(self.tns.registerContraction('e_vel_tau', Linvdt))

        #Damping:        
        etaLinvdt2 = self.tns.registerScalarMultiplication(Linvdt2,'eta_neg') # Linv*dt²/2
        sum_terms_Linv.append(self.tns.registerContraction('e_pos_vel', etaLinvdt2))

        etaLinvdt  = self.tns.registerScalarMultiplication(Linvdt,'eta_neg') # Linv*dt
        sum_terms_Linv.append(self.tns.registerContraction('e_vel_vel', etaLinvdt))


        self.tns.registerSum(*sum_terms_Linv, result_name='A_Linvonly')
        
        #finally, compute A:        
        self.tns.registerAddition('A_noLinv', 'A_Linvonly', result_name='A')
        

        self.tns.registerContraction('A', 'LastMean', result_name='CurrentMean')
        
        index_first_equation_A_unchanged = len(self.tns.update_order)
        
        self.tns.registerTranspose('A')
        previous = self.tns.registerContraction('LastCov', '(A)^T')
        previous = self.tns.registerContraction('A', previous)
        #this is a very simple "plant model" to account for limits in execution.
        #it also avoid unrealistic convergence to zero variances        
        currentcov_unsymmetrized = self.tns.registerAddition(previous, 'noiseFloorCov', result_name='CurrentCov_unsym')  

        #to avoid accumulation of numeric errors, re-symmetrize covariances after each step:
        previous = self.tns.registerTranspose(currentcov_unsymmetrized, flip_underlines=True)
        previous = self.tns.registerAddition(currentcov_unsymmetrized, previous)
        self.tns.registerScalarMultiplication(previous, 0.5, result_name = 'CurrentCov')

        #associate result tensors to an msd:
        self.msd_current  =_mechanicalstate.MechanicalStateDistribution(self.tns, 'CurrentMean', 'CurrentCov')
            
        #determine which equations to re-compute when only Linv changes:
        self._equations_dt_unchanged = self.tns.update_order[index_first_equation_no_dt_change:]
        self._equations_A_unchanged = self.tns.update_order[index_first_equation_A_unchanged:]


        
        if dynamicsModel is None:
            self.dynamicsModel = FakeDynamicsModel(self.tns.indexSizes['d'])
            print("TimeIntegrator: Using fake mass matrix for time integration.")
        else:
            self.dynamicsModel = dynamicsModel
    
        

    
    def integrate(self, mStateDistribution, dt, times=1):
            """
            
            mStateDistribution: distribution to integrate
            
            dt: timestep to use for integration. Faster if kept constant
            
            times: How many times the time integration should pe performed


            Old description:                           

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
            #copy the current data into the local tensors:
            self.tns.setTensor('LastMean',  mStateDistribution.getMeansData())
            self.tns.setTensor('LastCov',  mStateDistribution.getCovariancesData())

            if times < 1:
                return self.msd_current
 
            #get dynamics parameters:
            pos =  mStateDistribution.means[mStateDistribution.name2rg['position']]
            L = self.dynamicsModel.getInertiaMatrix()
            Linv = _np.linalg.inv(L)
            eta = self.dynamicsModel.getViscuousFrictionCoefficients(mStateDistribution.means)

            self.tns.setTensor('Linv', Linv)
            self.tns.setTensor('eta_neg', -eta)

            #first iteration: also check whether we need to recompute A:
            if abs(dt - self.tns.tensorData['dt'].value) < 1e-6:
                    self.tns.update(*self._equations_dt_unchanged)
            else:                
                    self.tns.setTensor('dt', dt_substeps)
                    self.tns.update() #needs a full recomputation

            #integrate for the requested number of times:
            for i in range(times-1):   
                self.setTensor('LastMean', self.tns.tensorData['CurrentMean'],self.tns.tensorIndices['CurrentMean'])
                self.setTensor('LastCov', self.tns.tensorData['CurrentCov'],self.tns.tensorIndices['CurrentCov'])
                self.tns.update(_equations_A_unchanged)
            
            if _np.any(self.tns.tensorData['covNext'] > 1e10):
                raise RuntimeWarning("TimeIntegrator: covariance matrix has elements > 1e10!")
            
            return self.msd_current


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
    

