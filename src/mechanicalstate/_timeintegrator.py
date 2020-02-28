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

    def __init__(self, tensornamespace, noiseFloorSigmaTorque=0.1, noiseFloorSigmaPosition=0.1, noiseFloorSigmaVelocity=0.1, dynamicsModel = None, accumulateQuantizationErrors=False):
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
        
        if self.tns['d'].size > 2:
            NotImplementedError()
        
        #make indices signifying data from the last timestep:
        self.tns.cloneIndex('r', 'rl')
        self.tns.cloneIndex('g', 'gl')
        self.tns.cloneIndex('d', 'dl')
        
        self.tns.registerTensor('LastMean', (('r','g','d'),()))
        self.tns.registerTensor('LastCov', (('r','g','d'),('r_', 'g_','d_')))
        self.msd_last  = _mechanicalstate.MechanicalStateDistribution(self.tns, 'LastMean', 'LastCov')
        
        
        #this is a very simple "plant model" to account for noise in actuation and sensing.
        #it also avoid unrealistic convergence to zero variances 
        self.tns.registerTensor('noiseFloorCov', (('r','g','d'),('r_', 'g_','d_')))
        self.tns['noiseFloorCov'].data[0,0,:,0,0,:] = noiseFloorSigmaPosition**2
        self.tns['noiseFloorCov'].data[0,1,:,0,1,:] = noiseFloorSigmaVelocity**2
        self.tns['noiseFloorCov'].data[1,1,:,1,1,:] = noiseFloorSigmaTorque**2
#        self.tns['noiseFloorCov'].data[1,0,:,1,0,:] = noiseFloorSigmaImpulse**2
        
#        self.tns.registerAddition('LastCov', 'noiseFloorCov', result_name='LastCov_plantnoise') 
        
        #
        #in order to distinguish last and current vector spaces, we need to rename the inputs:
        self.tns.renameIndices('LastMean', {'r':'rl', 'g':'gl', 'd':'dl', 'r_':'rl_', 'g_':'gl_', 'd_':'dl_', })
        self.tns.renameIndices('LastCov', {'r':'rl', 'g':'gl', 'd':'dl', 'r_':'rl_', 'g_':'gl_', 'd_':'dl_', })
        
        #create all the basis tensors we need:
        rgrg= (('r', 'g'),('rl', 'gl'))
        name2rg  = self.msd_last.commonnames2rg
        self.tns.registerBasisTensor( 'e_pos_vel', rgrg, (name2rg['position'],name2rg['velocity']), ignoreLabels=True )
        self.tns.registerBasisTensor( 'e_imp_tau', rgrg, (name2rg['impulse'], name2rg['torque']), ignoreLabels=True )
        self.tns.registerBasisTensor( 'e_pos_tau', rgrg, (name2rg['position'],name2rg['torque']), ignoreLabels=True )
        self.tns.registerBasisTensor( 'e_vel_tau', rgrg, (name2rg['velocity'],name2rg['torque']), ignoreLabels=True )
        self.tns.registerBasisTensor( 'e_vel_vel', rgrg, (name2rg['velocity'],name2rg['velocity']), ignoreLabels=True )
        
        dd= (('d',),('dl',))
        self.tns.registerTensor('delta_dd',dd, initial_values='identity') # delta^d_d 

        #inputs:
        self.tns.registerTensor('Linv',dd, initial_values='identity') # Linv
        self.tns.registerTensor('dt',((),()) ) # dt scalar
        self.tns.setTensor('dt', _np.inf ) 
        self.tns.cloneIndex('d', 'de')
        self.tns.registerTensor('eta_neg',(('dl',),('de',))) # damping factor (negative)


        self.tns.registerTensor('I', (('r','g','d'),('rl', 'gl','dl')), initial_values='identity') # delta^d_d 

        #computation:
        sum_terms_noLinv = []  #all terms that will be summed up to yield 'A'
        if not accumulateQuantizationErrors:
            sum_terms_noLinv.append('I')

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
        etaLinvdt2 = self.tns.registerContraction(Linvdt2,'eta_neg') # Linv*dt²/2
        self.tns.renameIndices(etaLinvdt2, {'de': 'dl'}, inPlace=True)
        sum_terms_Linv.append(self.tns.registerContraction('e_pos_vel', etaLinvdt2))

        etaLinvdt  = self.tns.registerContraction(Linvdt,'eta_neg') # Linv*dt
        self.tns.renameIndices(etaLinvdt, {'de': 'dl'}, inPlace=True)
        sum_terms_Linv.append(self.tns.registerContraction('e_vel_vel', etaLinvdt))

        self.tns.registerSum(*sum_terms_Linv, result_name='A_Linvonly')
        
        #finally, compute A:        
        self.tns.registerAddition('A_noLinv', 'A_Linvonly', result_name='A')
        self.tns.registerTranspose('A')

       
        index_first_equation_A_unchanged = len(self.tns.update_order)

        if not accumulateQuantizationErrors:
            self.tns.registerContraction('A', 'renamed(LastMean)', result_name='CurrentMean')
            previous = self.tns.registerContraction('A', 'renamed(LastCov)')
            previous = self.tns.registerContraction(previous, '(A)^T', result_name = 'CurrentCov')  #these are the changes to the covariance matrix
        
        else:

            #compute the changes to mean and covariances for this timestep:
            self.tns.registerContraction('A', 'renamed(LastMean)', result_name='deltaMean')


            #compute the changes to the covariance matrix, i.e. delta = (A+I)C(A+I)^T - C = ACA^T + AC + CA^T
            previous = self.tns.registerContraction('A', 'renamed(LastCov)', result_name='AC')
            previous = self.tns.registerContraction(previous, '(A)^T', result_name = 'ACA^T')  #these are the changes to the covariance matrix
            self.tns.renameIndices('AC', {'rl_': 'r_', 'gl_': 'g_', 'dl_': 'd_', }, inPlace=True)
            self.tns.registerTranspose('AC')
            self.tns.registerSum('ACA^T', 'AC', '(AC)^T', result_name='deltaCov')


            #add changes to the last mean, but also accumulate quantization errors over many timesteps:
            self.tns.registerTensor('deltaMeanSum', (('r','g','d'),()), initial_values='zeros')
            self.tns.registerAdditionToSlice('deltaMeanSum', 'deltaMean')
            self.tns.registerAddition('deltaMeanSum', 'LastMean', 'CurrentMean')
            previous = self.tns.registerSubtraction('LastMean', 'CurrentMean') #compute the negative quantized change
            self.tns.registerAdditionToSlice('deltaMeanSum', previous) #deduct from accumulator

            
            #add changes to the last covariances, but also accumulate quantization errors over many timesteps:
            self.tns.registerTensor('deltaCovSum', (('r','g','d'),('r_', 'g_','d_')), initial_values='zeros')
            self.tns.registerAdditionToSlice('deltaCovSum', 'deltaCov')
            self.tns.registerAddition('deltaCovSum', 'LastCov', 'CurrentCov_nonoise')
            previous = self.tns.registerSubtraction('LastCov', 'CurrentCov') #compute the actual numeric change
            self.tns.registerAdditionToSlice('deltaCovSum', previous) #deduct change from the small-values tensor
        
        
        


        #associate result tensors to an msd:
        self.msd_current  =_mechanicalstate.MechanicalStateDistribution(self.tns, 'CurrentMean', 'CurrentCov')

        #determine which equations to re-compute when only Linv changes:
        self._equations_dt_unchanged = self.tns.update_order[index_first_equation_no_dt_change:]
        self._equations_A_unchanged = self.tns.update_order[index_first_equation_A_unchanged:]

       
        if dynamicsModel is None:
            self.dynamicsModel = FakeDynamicsModel(self.tns['d'].size)
            print("TimeIntegrator: Using fake mass matrix for time integration.")
        else:
            self.dynamicsModel = dynamicsModel
    
        self.views_motion = []
        for g_idx in range(self.tns['gl'].size):
            view, indices  = self.tns.makeTensorSliceView('LastMean', {'rl': 'motion', 'gl':g_idx})
            self.views_motion.append(view)
        self.views_motion  = tuple(self.views_motion)
    
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
            self.tns.setTensor('LastMean',  mStateDistribution.means)
            self.tns.setTensor('LastCov',  mStateDistribution.covariances)

            if times < 1:
                return self.msd_current
 
            #get dynamics parameters: 
            self.dynamicsModel.update(*self.views_motion) #give the model the newest data            
            #then, query the dynamics parameters:
            Linv = self.dynamicsModel.getInertiaMatrixInverse()
            eta = self.dynamicsModel.getViscuousFrictionCoefficients()

            self.tns.setTensor('Linv', Linv)
            self.tns['eta_neg'].data_diagonal[:] = -eta

            #before the first iteration: also check whether we need to recompute A:
            if abs(dt - self.tns['dt'].data) > 1e-6: #needs a full recomputation
                self.tns.setTensor('dt', dt)
                self.tns.update() 
            else: #save some computation time:
                self.tns.update(*self._equations_dt_unchanged)


            #integrate for the requested number of times:
            for i in range(times-1):
                self.tns.setTensor('LastMean', self.tns['CurrentMean'].data)
                self.tns.setTensor('LastCov', self.tns['CurrentCov'].data)
                self.tns.update(self._equations_A_unchanged)
          
            #make sure that we don't drop covariances to unreasonable precision:
            self.tns['CurrentCov'].data_diagonal[...] = _np.maximum(self.tns['CurrentCov'].data_diagonal[...], self.tns['noiseFloorCov'].data_diagonal[...])
          
            if _np.any(self.tns['CurrentCov'].data > 1e10):
                raise RuntimeWarning("TimeIntegrator: covariance matrix exploded: It has elements > 1e10!")
            
            return self.msd_current


class FakeDynamicsModel():
    """
    placeholder class if user does not provide any dynamics of the system to integrate with
    
    This class is primarily intended for checking code and test plotting.
    """
    def __init__(self, dofs):
        self.L = _np.eye(dofs)
        self.viscuousFriction = _np.ones((dofs))
        
    def update(self, position, velocity=None, acceleration=None):
        self.position = position
    
    def getInertiaMatrix(self):
        return self.L

    def getInertiaMatrixInverse(self):
        return self.L

    def getViscuousFrictionCoefficients(self):
        return self.viscuousFriction
    

