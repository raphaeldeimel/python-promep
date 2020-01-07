#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel
@copyright 2017
@licence: 2-clause BSD licence

This file contains classes for mixing / coactivating several ProMPs


The algorithm for mixing at a given time instant is described in:

[1] A. Paraschos, C. Daniel, J. Peters, and G. Neumann,
“Using probabilistic movement primitives in robotics,” Autonomous Robots, pp. 1–23, 2017.


"""
import warnings
import numpy as _np
import itertools as _it
import matplotlib.pyplot as _plt

from . import _tensorfunctions as _t
from . import _timeintegrator


class ProMPMixer(object):
    """
    Implements a mixer that can blend the trajectory distributions of a list of ProMP objects


        phaseIntervals: List of 2-tuples that specifies the phase intervalof each motion primitive that 
                        is stretched to phase interval [0...1] at the mixer interface
                            (p0, p1): The ProMP interval [p0...p1] is stretched to [0...1]
                            (0,1):  no stretching, 1:1 mapping
                        This property enables to "cut out" parts of a ProMP

        weighingMethod:  select which weighing method is being used
                           'Paraschos': weighted combination of distributions in parameter space
                                   method from the original ProMP paper [1]
                           'Deimel': averaging of distributions that are ridge-regularized according to activation
                           
        doPreconditioning: If true, then the previously computed combined inverse covariance matrix is used 
                           as a preconditioner for inverting the controller distributions.
                           This improves blending behavior when small alpha values are present.
                           Note: for "Deimel", precondintioning is a requirement, and this argument is ignored

        inverseRegularization: Ridge regularization to be used when inverting controller distributions. 
                            

    """

    def __init__(self, mstateDescription=None, ProMPList=[], stateControllerList=[], phaseIntervals=None, weighingMethod='Paraschos', doPreconditioning=True, inverseRegularization=1e-9):
        
        if isinstance(mstateDescription, (list,)):
            raise DeprecationWarning("List provided for first argument. Update the function signature asap! Trying guessing the right arguments...")
            ProMPList = mstateDescription
            mstateDescription = None
            
        if mstateDescription is None:
            self._md = ProMPList[0]._md #try to guess mstateDescription from provided promps
        else:
            self._md = mstateDescription
        
        for i,p in enumerate(ProMPList):
            if not p.phaseAssociable:
                raise ValueError("ProMPList requires motion generator classes that are phase-based ! (offending promp:#{0} {1})".format(i, p.name))
            if self._md != p._md:
                raise ValueError("All mixed objects need to have the same mechanical state description! (offending promp:#{0}, \n\n{1}\n\ninstead of\n\n{2})".format(i, p._md.serialize(), self._md.serialize()))

        self.hasControllers = False
        for i,p in enumerate(stateControllerList):
            self.hasControllers = True
            if not p.timeAssociable:
                raise ValueError("ProMPList requires controller classes that are time-based ! (offending controller:#{0})".format(i))
            if self._md != p._md:
                raise ValueError("All mixed objects need to have the same mechanical state description! (offending controller:#{0}, \n\n{1}\n\ninstead of\n\n{2})".format(i, p._md.serialize(), self._md.serialize()))
                
        self.promps = ProMPList
        self.stateControllers = stateControllerList
        self.numStates = len(self.stateControllers)
        self.n = len(ProMPList) + len(stateControllerList)
        self.CovarianceTensorNone = _t.makeCovarianceTensorUncorrelated(self._md.mechanicalStatesCount,self._md.dofs, sigmas=1e7)
        self.invCovarianceTensorNone = _t.pinv(self.CovarianceTensorNone)
        self.meansNone=_np.zeros((self._md.mechanicalStatesCount,self._md.dofs))
        self.lastDistribution = None
        if phaseIntervals is None:
            self.prompPhaseIntervals = [(0,1)]*(len(ProMPList))
        else:
            self.prompPhaseIntervals = phaseIntervals
        if not weighingMethod in ['Paraschos', 'Deimel', 'WinnerTakeAll']:
            raise ValueError("weighingMethod {0} not known".format(weighingMethod))
        self.weighingMethod=weighingMethod
        if self.weighingMethod == 'Deimel':
            self.doPreconditioning = True
        else:
            self.doPreconditioning = doPreconditioning
        self.preconditioner = None
        self.inverseRegularization = inverseRegularization


    def updateCurrentState(self, phaseVector, meansObserved, CovarianceTensorObserved ):
        """
        This method conditions the initial distribution on the given state

        Any preceding conditionings (such as old current states) are forgotten
        """
        #update conditioning all ProMPs given their phase and activation
        for i, mp in enumerate(self.promps):
            mp.resetDistribution() #forget old conditioning, it is subsumed by conditioning on the newer mixed distribution
            mp.conditionToObservation(phaseVector[i], meansObserved, CovarianceTensorObserved )


    def getMixedDistribution(self,
            activationVectorProMPs,
            phaseVectorProMPs,
            phaseVelocityVectorProMPs,
            activationVectorStateControllers=None,
            currentDistribution = None,
            phaseVelocitySigma=None,
            hTransformGoal = None,
            jacobianCallback = None,
            hTransformCallback = None,
            qCallback = None
        ):
        """
        compute the mixture of promps by the given activation vector at the given phase

        activationVector: list or numpy vector that contains the activation value for each ProMP registered in the mixer

             phaseVector: scalar or vector that contains the phases for each PromP
                            If a scalar is provided, then all ProMPs use this as their phase


        returns (meansCombined, CovarianceTensorCombined)

                   meansCombined: matrix of size (derivs, dofs) with the means of the
                                    resolved distribution at the given phase

        CovarianceTensorCombined: array of size (derivs, dofs, derivs, dofs) with the means
                                    of the resolved distribution at the given phase
                                    
        """
        #sanitize activationVector
        activationVectorProMPs = _np.asarray(activationVectorProMPs)
        phaseVectorProMPs = _np.asarray(phaseVectorProMPs)
        phaseVelocityVectorProMPs = _np.asarray(phaseVelocityVectorProMPs)
        if activationVectorStateControllers is not None:
            activationVectorStateControllers = _np.asarray(activationVectorStateControllers)
        #
        if currentDistribution is not None:
            covShape = (self._md.mechanicalStatesCount, self._md.dofs, self._md.mechanicalStatesCount, self._md.dofs)
            if currentDistribution[1].shape != covShape:
                raise ValueError("covariance tensor should be {0}, is {1}".format(covShape, currentDistribution[1].shape))
        elif activationVectorStateControllers is not None:
            raise RuntimeError("Cannot guess a previous distribution for state controllers to use, and None was supplied")

            
        if len(self.promps) != activationVectorProMPs.size:
            raise ValueError("activationVector is of size {0} (should be {1})".format(activationVectorProMPs.size, len(self.promps)))
        activationVectorProMPs = _np.clip(activationVectorProMPs, 0.0, 1.0)
        if activationVectorStateControllers is None:
            activationVector  = activationVectorProMPs
        else:
            activationVector  = _np.concatenate((activationVectorStateControllers,activationVectorProMPs)) 


        #sanitize phaseVector
        phaseVectorProMPs  = _np.clip(phaseVectorProMPs, 0.0, 1.0)
        if phaseVectorProMPs.size == 1:
            phaseVectorProMPs = _np.repeat(phaseVectorProMPs, len(self.promps))

        #map phase to [0...1] the desired phase interval given by self.prompPhaseIntervals:
        phasesStretched = [ (p*plimits[1] + (1.0-p)*plimits[0]) for p, plimits in zip(phaseVectorProMPs, self.prompPhaseIntervals) ]

        #compute some statistics used by mixing methods:
        sumActivations =  _np.sum(activationVector) 
        sumActivations2 =  _np.sum(activationVector**2) 

        # compute the residual to activate the default behavior accordingly        
        activationResidual = 1 - sumActivations
        if activationResidual < 0.0:
            activationResidual = 0.0            
            warnings.warn("ProMPMixer: Sum of activations is larger than 1.0!", RuntimeWarning)

        ##uncomment this to make the mixer act like a hybrid automaton
        #activationVector = 1 * ( activationVector >= _np.max(activationVector))

        #determine combined weighting vectors:
        if self.weighingMethod == 'WinnerTakeAll':
                imax = _np.argmax(activationVector)
                alphaVector = _np.zeros(activationVector.shape) 
                alphaVector[imax] = 1.0
        else: 
                alphaVector = activationVector

        padList = [None] * len(self.stateControllers)
        #get the state distributions of all movement primitives and state controllers:
        

        if self.doPreconditioning:
            if self.preconditioner is None and currentDistribution is not None: #we need to initialize the preconditioner at some point
                self.preconditioner = _t.pinv(currentDistribution[1])
        else:
                self.preconditioner = None
        

        activeSet = [] #gather tuples of active distributions, activation values, and types
        #query all controllers for their desired/expected distribution:
        for controller, phi, dphidt, alpha in zip( _it.chain(self.stateControllers,self.promps), _it.chain(padList, phasesStretched), _it.chain(padList,phaseVelocityVectorProMPs), alphaVector):
            if alpha <= 0.01:
                continue
            isProMP = (phi is None) # This actually means the opposite, right? 

            # For now the hTransformGoal is only handled by PDControllers
            if isProMP:
                if type(controller).__name__ == "PDController":
                    m,S = controller.getInstantStateVectorDistribution(phase=phi, phaseVelocity=dphidt, currentDistribution=currentDistribution, phaseVelocitySigma=phaseVelocitySigma)
                elif type(controller).__name__ == "TaskSpaceController":
                    m,S = controller.getInstantStateVectorDistribution(phase=phi, phaseVelocity=dphidt, currentDistribution=currentDistribution, phaseVelocitySigma=phaseVelocitySigma, hTransformGoal=hTransformGoal, jacobianCallback=jacobianCallback, hTransformCallback = hTransformCallback)
                else:
                    raise TypeError("Unexpected controller type.")
            else:
                m,S = controller.getInstantStateVectorDistribution(phase=phi, phaseVelocity=dphidt, currentDistribution=currentDistribution, phaseVelocitySigma=phaseVelocitySigma)
            
            activeSet.append( (m,S,alpha,isProMP,controller.name) )

        #make sure variances are strictly positive and nonzero:
        for m,S, alpha,phi,name in activeSet:
            varViewS = _t.getDiagView(S)
            varmin = _np.min(varViewS)
            if varmin < -0.001: #only complain if there is a substantial amount of negative variance
                for dof in range(self._md.dofs):
                    if _np.any(varViewS[:,dof] < -0.001):
                        print("ProMPMixer: encountered negative variances: controller {0}, dof {1}: {2}".format(name, dof, _np.array_repr(varViewS[:,dof], precision=6, suppress_small=False)))
            if varmin < 0.0:  #try to repair the covariance matrix
                varViewS[:,:] -= 2*varmin


        #Prepare some variables for the actual mixing loop:
        if self.weighingMethod == 'Deimel':
            pass
        elif self.weighingMethod == 'Deimel_old': #deprecated, here for reference
            SweightedMean = _np.sum( [S*alpha for m,S,alpha,isProMP,name in activeSet] , axis=0) / _np.sum(alphaVector)
            mWeightedMean = _np.sum( [m*alpha for m,S,alpha,isProMP,name in activeSet] , axis=0) / _np.sum(alphaVector)
        else:
            #if activations do not add up to 1.0, add the current distribution to fail gracefully
            if activationResidual > 0.2:  
                activeSet.append( (
                    currentDistribution[0],
                    currentDistribution[1],
                    activationResidual,
                    False
                ) )
                
        ####Actual Mixing Code ####
        
        #combine distributions by adding their weighted precision/information matrices (inverse of covariances) 
        means = []
        covarianceTensorsWeighted = []
        invCovarianceTensorsWeighted = []
        for m,S, alpha,phi,name in activeSet:
            mdecorrelated = m
            # Contrary to controllers, ProMPs do not take the current distribution into account - which makes them 
            # overly confident when activation is small (especially at small phase velocities)
            # To "regularize" them, we add the mean distribution when partially activated:
            if self.weighingMethod == 'Deimel':
                    Sdecorrelated = sumActivations2 * _t.dot(S,self.preconditioner) 
                    _t.addToRidge(Sdecorrelated,  sumActivations * (1.0-alpha)) 
                
            elif self.weighingMethod == 'Deimel_old':
                if isProMP is True:
                    Sdecorrelated =  alpha * S + (1-alpha) * SweightedMean
                else:
                    Sdecorrelated = S                
            else:
                Sdecorrelated = S
                    
            #Invert the covariance tensor. By using the last distribution as preconditioner and regularize on the preconditioned matrix, we can numerically stabilize the double inversion
            if self.preconditioner is not None and self.weighingMethod != 'Deimel':  
                Sdecorrelated =_t.dot(Sdecorrelated, self.preconditioner)
 
            #invert the covariance matrix:
            SInv = _t.pinv(Sdecorrelated,regularization=self.inverseRegularization)

            SInvWeighted = SInv * alpha
            if self.weighingMethod == 'Deimel_old':
                SInvWeighted = SInvWeighted * alpha

            means.append(m)
            covarianceTensorsWeighted.append(S)
            if self.preconditioner is not None:    
                SInvWeighted = _t.dot(self.preconditioner, SInvWeighted)
            invCovarianceTensorsWeighted.append(SInvWeighted)

        invCovarianceTensorCombined = _np.sum(invCovarianceTensorsWeighted, 0)
        CovarianceTensorCombined = _t.pinv(invCovarianceTensorCombined) 

        #compute Eq.33 of [1]
        means1 = [ _t.dot(mu, t, shapes=((0,2),(2,2))) for mu, t in zip(means, invCovarianceTensorsWeighted)]
        means2 = _np.sum( means1, 0)
        meansCombined = _t.dot(CovarianceTensorCombined, means2, shapes=((2,2),(2,0)))

        goodComputation=True
        for dof in range(self._md.dofs):
            variances = _np.diag(CovarianceTensorCombined[:,dof,:,dof])
            if _np.any(variances < -0.01):
                print("ProMPMixer: incorrect combined covariance tensor")
                print(dof,CovarianceTensorCombined[:,dof,:,dof])
                goodComputation=False

#        if not goodComputation:
#            meansCombined= currentDistribution[0]
#            CovarianceTensorCombined=currentDistribution[1]
#            invCovarianceTensorCombined = _t.pinv(currentDistribution[1])

        if self.doPreconditioning:
            self.preconditioner = invCovarianceTensorCombined

        #make sure variances are strictly positive and nonzero:
#        for dof in range(self._md.dofs):
#            for m in range(self._md.mechanicalStatesCount):
#                if CovarianceTensorCombined[m,dof, m, dof] < 1e-7:
#                    CovarianceTensorCombined[m,dof, m, dof] = 1e-7
            
        return meansCombined, CovarianceTensorCombined, invCovarianceTensorCombined


    def plot(self, activations, 
                  phaseVectors=None, 
                  activationsStates=None,
                  dofs='all', 
                  derivatives='all', 
                  duration=1.0, 
                  linewidth=2.0, 
                  withSampledTrajectories=None, 
                  alphaIndividualProMPs=0.2,
                  withGainsPlots=True,
                  ylimits = {'position': (-3.14,3.14), 'velocity': (-1,1), 'gains': (-10,100)},
                  sectionIndicatorPositions=None,
                  sigma_control=_np.inf, 
                  timeintegrator=None):
        """
        plot the mixture of promps
        
        activations: list of activation lists (or numpy array)
        phaseVectors: list of phase lists (or numpy array)
        
        dofs:   list of dofs to plot. Default to all dofs
        
        duration: time period for the list of activations
        
        ylimits:  set the limits for each derivative explicitly:
                    E.g.: ylimits= [(-3.14,3.14), (-5,5), (-10,10) ]
        
        sectionIndicatorPositions: List of time positions where a vertical dotted line should be painted
        
        timeintegrator: object to integrate the mechanical state distribution with
            if None, a fake model (with unit diagonal inertia matrix) is used
        """
        confidenceColor = "#DDDDDD"
        meansColor = '#BBBBBB'
        individualProMPColor='#CCCCFF'
        observedColor = '#880000'
        kpColor = '#DD0000'
        kvColor = '#0000DD'
        rowOfActiveControllerAnnotation=0
        if dofs=='all':
            dofsList=list(range(self._md.dofs))
        else:
            dofsList = dofs
        
        if sectionIndicatorPositions is None:
            sectionIndicatorPositions = []
            
        if derivatives=='all':
            derivativesList=list(range(self._md.mechanicalStatesCount))
        else:
            derivativesList = derivatives
            
        activations = _np.asarray(activations)            
        if len(self.promps) == activations.shape[0]:
            pass
        elif len(self.promps) == activations.shape[-1]:
            activations = activations.T
        else:
            raise ValueError("activations matrix is of wrong shape! {0}".format(activations.shape))

        if activationsStates is not None:
            activationsStates = _np.asarray(activationsStates)            
            if len(self.stateControllers) == activationsStates.shape[0]:
                pass
            elif len(self.stateControllers) == activationsStates.shape[-1]:
                activationsStates = activationsStates.T
            else:
                raise ValueError("activationStates matrix is of wrong shape! {0}".format(activationsStates.shape))

        num = activations.shape[1] #number of points to plot
        t = _np.linspace(0.0,duration,num)
        td = duration / num
        invtd = num/duration

        m = len(self.promps)
        if phaseVectors is None:
            phaseVectors  = _np.linspace(0,1,num).reshape((1,num)).repeat(m, axis=0)
        else:
            phaseVectors = _np.asarray(phaseVectors)
        if m == phaseVectors.shape[0]:
            pass
        elif m == phaseVectors.shape[-1]:
            phaseVectors = phaseVectors.T
        else:
            raise ValueError("phaslimitseVectors matrix is of wrong shape!")

            

        means = _np.empty((num, self._md.mechanicalStatesCount, self._md.dofs))
        sigmas = _np.empty((num, self._md.mechanicalStatesCount, self._md.dofs))
        data_gains = _np.empty((num,2, self._md.dofs))
        
        #set up a noise covariance matrix tophaseVelocitiesVector fake plant/action/sensor noise
        cov_control = _np.zeros((self._md.mechanicalStatesCount, self._md.dofs, self._md.mechanicalStatesCount, self._md.dofs))
        for i,j in _it.product(range(self._md.dofs),range(self._md.mechanicalStatesCount)):
            cov_control[j,i,j,i] = sigma_control**2 / num

        means_MPs = _np.empty((self.n,num, self._md.mechanicalStatesCount, self._md.dofs))
        sigmas_MPs = _np.empty((self.n,num,self._md.mechanicalStatesCount, self._md.dofs))


        #compute phase velocities, assume phase=0 before the first phase
        phaseVelocitiesVectors =  phaseVectors.copy()
        phaseVelocitiesVectors[:,1:] -= phaseVectors[:,:-1]
        phaseVelocitiesVectors *= invtd

        #set the initial distribution to start with:
        lastDistribution = ( _np.zeros((self._md.mechanicalStatesCount, self._md.dofs)), _t.I(self._md.mechanicalStatesCount, self._md.dofs) )
        if activationsStates is None:
            statesVector = None
        else:
            statesVector = activationsStates[:,0]
        
#        m, cov, invCov = ProMPMixer.getMixedDistribution(self, activations[:,0], phaseVectors[:,0], phaseVelocitiesVectors[:,0], activationVectorStateControllers=statesVector, currentDistribution=lastDistribution)
#        lastDistribution = (m, cov)
        
        if timeintegrator is None:
            timeintegrator = _timeintegrator.TimeIntegrator(self._md.dofs) 
        
        for i in range(num):

            phaseVector = phaseVectors[:,i]
            phaseVelocitiesVector = phaseVelocitiesVectors[:,i]
            interpolVector = activations[:,i]
            if activationsStates is None:
                statesVector = None
            else:
                statesVector = activationsStates[:,i]

            #get current distribution:
#            currentDistribution=timeintegrator.integrate(lastDistribution, td)
            currentDistribution=lastDistribution
            
            #get dists of each mp individually
            for j,mp in enumerate(self.promps):
                m, cov = mp.getInstantStateVectorDistribution(phaseVector[j])
                means_MPs[j,i,:,:] = m
                sigmas_MPs[j,i,:,:] = _t.getStdDeviationsOfStateCovarianceTensor(cov)

            #compute combined dist
            m, cov, invCov = ProMPMixer.getMixedDistribution(self, interpolVector, phaseVector, phaseVelocitiesVector, activationVectorStateControllers=statesVector, currentDistribution=currentDistribution)
            lastDistribution = (m, cov)
            means[i,:,:] = m
            sigmas[i,:,:] = _t.getStdDeviationsOfStateCovarianceTensor(cov)
            #print(_np.max(cov))

            invcovmotion = _t.pinv(cov[1:,:,1:,:])
            data_gains_all = -1* _t.dot(cov[0,:,1:,:], invcovmotion, shapes=((1,2),(2,2)) )
            if withGainsPlots:
                for dof in range(self._md.dofs):
                    data_gains[i,:,dof] = data_gains_all[dof,:,dof]

            #if requested, limit the change in variance introducable by control:
            cov_observed = cov + cov_control
            if _np.isfinite(sigma_control):  #skip if it is infinite (speedup)
                self.updateCurrentState(phaseVector, m, cov_observed)

        subplotfigsize=2.0
        plotrows = self._md.mechanicalStatesCount+1
        plotcols = len(dofsList)
        plotrownames=  ['activation'] + self._md.mStateNames
        if withGainsPlots:
            plotrows=plotrows + 1
            plotrownames= plotrownames + ['gains']

        fig, axesArray = _plt.subplots(plotrows,plotcols, squeeze=False,  figsize=(max(plotcols,plotrows)*subplotfigsize,plotrows*subplotfigsize), sharex='all', sharey='row')
        axesArray.shape = (plotrows,plotcols)



        #plot distributions
        for dof in dofsList:                
           for m in derivativesList:
                ax = axesArray[m+1,dof]
                if alphaIndividualProMPs > 0.01:
                    for j in range(self.n):
                        lower = means_MPs[j,:,m,dof]-1.96*sigmas_MPs[j,:,m,dof]
                        upper = means_MPs[j,:,m,dof]+1.96*sigmas_MPs[j,:,m,dof]
                        ax.fill_between(t[m:], lower[m:], upper[m:], alpha=alphaIndividualProMPs, label="95%",  color=individualProMPColor)
                        ax.plot(t[m:], lower[m:], alpha=(0.25+0.75*alphaIndividualProMPs), color=individualProMPColor,linewidth=linewidth)
                        ax.plot(t[m:], upper[m:], alpha=(0.25+0.75*alphaIndividualProMPs), color=individualProMPColor, linewidth=linewidth)

                #plot the mixed distribution:                        
                lower = means[:,m,dof]-1.96*sigmas[:,m,dof]
                upper = means[:,m,dof]+1.96*sigmas[:,m,dof]
                ax.fill_between(t[m:], lower[m:], upper[m:], alpha=0.7, label="95%",  color=confidenceColor)
                ax.plot(t[m:], means[m:,m,dof], color=meansColor, linewidth=linewidth)
                for indicator_x in sectionIndicatorPositions:
                    ax.axvline(indicator_x, linestyle=':')

           if withGainsPlots:
                ax=axesArray[-1,dof]
                ax.plot(t,data_gains[:,0,dof], label="gain kp",  color=kpColor)
                ax.plot(t,data_gains[:,1,dof], label="gain kv",  color=kvColor)
                for indicator_x in sectionIndicatorPositions:
                    ax.axvline(indicator_x, linestyle=':')
                ax.axhline(0.0, linestyle=':')


        #unify scaling
        if ylimits == None: #if ylimits are not specified, use the largest limits for each derivative:
            ylimits=[]
            for m in derivativesList:
                ylim = [0,0]
                for dof in dofsList:                
                     ylimDOF = axesArray[m+1, dof].get_ylim()
                     ylim[0] = min(ylimDOF[0], ylim[0])
                     ylim[1] = max(ylimDOF[1], ylim[1])
                ylimits.append(ylim)
        for i, rowname in enumerate(plotrownames):
            for dof in dofsList:                
                axesArray[i, dof].set_xlim((0.0, duration))
            if rowname in ylimits:
                bounds = ylimits[rowname]
                for dof in dofsList:                
                    axesArray[i, dof].set_ylim(bounds)
            

        #add x axis labels per column
        for dof in dofsList:
             axesArray[-1,dof].set_xlabel("DoF {0}".format(dof))
        #add y axis label per row
        axesArray[0,0].set_ylabel('activation')
        for row,m in enumerate(derivativesList):
             axesArray[row+1,0].set_ylabel(self._md.mStateNames[m])
        if withGainsPlots:
            axesArray[-1,0].set_ylabel('gains')



        #gather names and activations history for each statecontroller and promp:
        namesAll = [self.stateControllers[j].name for j in range(len(self.stateControllers)) ] +  [self.promps[j].name for j in range(len(self.promps)) ]
        activationsAll =  [ activationsStates[j,:] for j in range(len(self.stateControllers)) ] + [activations[j,:] for j in range(len(self.promps)) ]
        yposAll = [i-len(namesAll)-10 for i in range(len(namesAll))]
        for dof in dofsList:
            ax = axesArray[rowOfActiveControllerAnnotation,dof]
            if dof != 0:
                ax.set_visible(False)
                continue
                
            ax.set_yticks([], [])
            #ax.set_yticks(yposAll, namesAll)
            ax.set_ylim([min(yposAll)-1, max(yposAll)+1])
            ax.set_ylabel("activity")
            for name, a, y0 in zip(namesAll, activationsAll, yposAll):
                isStart = (a[1:] > 0.1) * (a[:-1] <= 0.1)
                isEnd = (a[1:] < 0.05) * (a[:-1] >= 0.05)
                isEnd[-1] = True
                if a[0] > 0.1:
                    idxStart = 0
                else:
                    idxStart = None                        
                for i in range(num-1):
                    if isStart[i]:
                        idxStart = i
                    if isEnd[i] and idxStart is not None:
                        y_lower = y0 - 0.33*a[idxStart:i]
                        y_upper = y0 + 0.33*a[idxStart:i]
                        ax.plot((t[0], t[-1]),(y0, y0), color=individualProMPColor, linewidth=0.2)
                        ax.fill_between(t[idxStart:i], y_lower, y_upper, color=individualProMPColor)
                        if dof==0: #annotate only in the first dof to avoid clutter
                            middlex = (idxStart+i)//2
                            ax.text(t[-1], y0, " " + name, 
                                    verticalalignment='center', 
                                    horizontalalignment='left',
#                                            fontdict={'size' : 'x-small', 'stretch': 'condensed'}
                            )
                        idxStart = None
                
                          
        return means, sigmas



class ProMPMatrixMixer(ProMPMixer):
    """
    MP mixer that takes a matrix of promps and a matrix of activations

    This class is intended to simplify the connection to a PhaSta kernel
    """

    def __init__(self, mstateDescription, ProMPList, associatedTransitions, stateControllerList,  phaseIntervals=None,  weighingMethod='Paraschos'):
        """
        set up the association between MPs and PhaSta states / transitions

        ProMPList: List of MPs for transitions
        associatedTransitions: respective list of transitions to associate the MPs with

        stateControllers: List of MPs used as state controllers, for each state respectively
        """
        ProMPMixer.__init__(self, mstateDescription, ProMPList, stateControllerList=stateControllerList, phaseIntervals=phaseIntervals, weighingMethod=weighingMethod)
        self.associatedTransitions = associatedTransitions

    def _matrixToList(self, activationMatrix, phaseMatrix, phaseVelocitiesMatrix):
        """
        compute the mixture of promps by the given activation vector at the given phase

        activationMatrix: array that contains the activation value for each ProMP registered in the mixer

        phaseMatrix: array that contains the phases for each PromP
                        If a scalar is provided, then all ProMPs use this as their phase
        """
        #controllers (MPs associated with the diagonal) do not have a phase, we simply assume a
        # fixed factor between phase progression and time step, and
        # always restart from the beginning each timestep
        activationsState = []
        for i in range(len(self.stateControllers)):
            activationsState.append(activationMatrix[i,i])
        
        activationsTransition = []
        phases = []
        phaseVelocities = []
        for mp, transition in zip(self.promps, self.associatedTransitions):
            nextState, prevState = transition
            activationsTransition.append(activationMatrix[nextState, prevState])
            phases.append(phaseMatrix[nextState, prevState])
            phaseVelocities.append(phaseVelocitiesMatrix[nextState, prevState])
            
        return activationsTransition, phases, phaseVelocities, activationsState

    def getActiveControllerNames(self, activationMatrix, threshold=0.5, phasesMatrix=None):
        activenames=[]
        activations=[]
        phases=[]
        for mp, transition in zip(self.promps, self.associatedTransitions):
            nextState, prevState = transition
            a = activationMatrix[nextState, prevState]
            if a  > threshold:
                activenames.append(mp.name)
                activations.append(a)
                if phasesMatrix is not None:
                    phases.append(phasesMatrix[nextState, prevState])
        for state, controller in enumerate(self.stateControllers):
            a  = activationMatrix[state, state]
            if  a > threshold:
                activenames.append(controller.name)
                activations.append(a)
                if phasesMatrix is not None:
                    phases.append(None)
        if phases is None:
            return activenames, activations
        else:
            return  activenames, activations, phases
            
            

    def getMixedDistribution(self, activationMatrix, phaseMatrix, phaseVelocitiesMatrix, currentDistribution, phaseVelocitySigma=None, hTransformGoal = None, jacobianCallback = None, hTransformCallback = None, qCallback = None):
        activationsTransition, phases, phaseVelocities, activationsState = self._matrixToList(activationMatrix, phaseMatrix, phaseVelocitiesMatrix)
        return ProMPMixer.getMixedDistribution(self, activationsTransition, phases,  phaseVelocities, activationVectorStateControllers=activationsState, currentDistribution=currentDistribution, phaseVelocitySigma=phaseVelocitySigma, hTransformGoal = hTransformGoal, jacobianCallback = jacobianCallback, hTransformCallback = hTransformCallback, qCallback = qCallback)
        
        
    def plot(self, activationMatrices, phaseMatrices, **kwargs):
        m = len(self.stateControllers)
        for i, A in enumerate(activationMatrices):
            if A.shape != (m,m):
                print("activation matrix {0} is of wrong shape({2})! (should be {1} x {1})".format(i,m, A.shape))
                print(A)
                raise ValueError()
                
        #estimate phase velocities, this is good enough for printing:
        invdt  = len(phaseMatrices)/1.0
        phaseVelocityMatrices = [(a-b)*invdt for a,b in zip(phaseMatrices[:-1], phaseMatrices[1:])] + [phaseMatrices[-1]]
        
        activationsTransition, phases, phaseVelocities, activationsState = zip(*[ self._matrixToList(a,b,c) for a,b,c in zip(activationMatrices, phaseMatrices, phaseVelocityMatrices) ])
        ProMPMixer.plot(self, activationsTransition, phases, activationsState, **kwargs)
        



