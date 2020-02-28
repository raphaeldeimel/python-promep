#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Raphael Deimel
@copyright 2020
@licence: 2-clause BSD licence

This file contains a class that represents a mechanical state distribution

"""

import numpy as _np
import collections as _collections
import matplotlib.pylab as _plt
import matplotlib as _mpl

import namedtensors as _nt


#Precompute this to avoid large overhead every time a MechanicalStateDistribution object is instantiated (which very probably all have the same r and g size)
def _makeMetadataLookuptable(r_max=2, g_max=4):
    namelookup_table = [[None]*(g_max+1)]*(r_max+1)
    for r in range(1,r_max+1):
        for g in range(1,g_max+1):
            metadata = {}
            
            metadata['indexNames'] = ('r','d','g')
            metadata['indexNames_transposed'] = ('r_','d_','g_')

            motion_names = ('position', 'velocity', 'acceleration', 'jerk', 'snap', 'crackle', 'pop', 'lock', 'drop')
            effort_names = ('int_int_torque', 'impulse', 'torque', 'torque_rate', 'deriv_deriv_torque')
            
            # For motion, position is always included, irrespective of g
            # But for effort, we want to make sure torque is always included.
            # So if g <=2, we drop lower derivatives instead of higher ones:
            if g <=2:   
                shift_effort_by = 3-g #remember by how much gtilde for effort is shifted
            else:
                shift_effort_by = 0
            names_all = [ motion_names[:g], effort_names[shift_effort_by:shift_effort_by+g]][:r]
            
            metadata['rg_commonnames'] = names_all
            metadata['realm_names'] = ['motion', "effort"]
                       
            #set up translation dict from human-readable names to indices used within the promep data structures:
            names2rg={}
            names2rglabels={}
            for g_idx in range(g):
                for r_idx in  range(r):
                    plain_name = metadata['rg_commonnames'][r_idx][g_idx]
                    r_label = metadata['realm_names'][r_idx]
                    g_label = g_idx  #is an integer index
                    names2rg[plain_name] = (r_idx,g_idx)
                    names2rglabels[plain_name] = (r_label, g_label)

            #a somewhat elaborate list of gains names for all combinations up to fourth derivative:
            gain_names =  [

                ('torque_rate',    'position', 'delta_kabsement'), 

                ('torque',         'position',                 'kp'), 
                ('torque_rate',    'velocity',           'delta_kp'),

                ('impulse',        'position',            'int_kv'), 
                ('torque',         'velocity',                'kv'),
                ('torque_rate',    'acceleration',      'delta_kv'),

                ('int_int_torque', 'position',        'int_int_ka'), 
                ('impulse',        'velocity',            'int_ka'),
                ('torque',         'acceleration',            'ka'),
                ('torque_rate',    'jerk',              'delta_ka'),

                ('int_int_torque', 'velocity',     'int_int_kjerk'), 
                ('int_torque',     'acceleration',     'int_kjerk'), 
                ('torque',         'jerk',                 'kjerk'), 
                
                ('int_int_torque', 'acceleration', 'int_int_ksnap'), 
                ('impulse',        'jerk',             'int_ksnap'), 

                ('int_int_torque', 'jerk',   'int_int_int_kcrackle'), 
                
            
            ]
            #add gains to dictionary, if they exist in the distribution:                
            for effort_name, motion_name, gains_name in gain_names:        
                if effort_name in names2rg and motion_name in names2rg:
                    names2rg[gains_name] = names2rg[effort_name] + names2rg[motion_name]


            metadata['commonnames2rg'] = names2rg
            metadata['commonnames2rglabels'] = names2rglabels
            namelookup_table[r][g] = metadata
    return namelookup_table

_static_namelookup_table = _makeMetadataLookuptable()



def makeTensorNameSpaceForMechanicalStateDistributions(r=2, g=2, d=1):
    """
    Create a TensorNameSpace object with suitable index definitions 
    to support MechanicalStateDistribution objectss
    """
    tns = _nt.TensorNameSpace()
    metadata = _static_namelookup_table[r][g]
    tns.registerIndex('r', r, metadata['realm_names'])
    tns.registerIndex('g', g)
    tns.registerIndex('d', d)
    return tns
    
    


class MechanicalStateDistribution(object):

    def __init__(self, tensornamespace, meansName, covariancesName, precisionsName=None):
        """

        tensornamespace: namespace the data are defined/managed in 
        
        meansName, covariancesName: nanamelookup_tablemes of the tensor in the tensor namespace
        
        precisionsName: optional tensor that holds inverses

        means: the epectation of the distribution
                        shape: (r,g,d)  
        covariances: the covariance tensor of the distribution 
                        shape: (r,g,d, r,g,d)  
        precisions: inverse of the covariance tensor. Set this if available so multiple inversions canbe avoided
                        
        Usually: r=2, g=2, d=8
        """    
        self.tns = tensornamespace
        self.meansName = meansName
        self.covariancesName = covariancesName
        self.precisionsName = precisionsName
        metadata = _static_namelookup_table[self.tns['r'].size][self.tns['g'].size]
        self.commonnames2rg = metadata['commonnames2rg']
        self.commonnames2rglabels = metadata['commonnames2rglabels']
        self.rg_commonnames = metadata['rg_commonnames']        
        self.realm_names = metadata['realm_names']
        self.indexNames = metadata['indexNames']
        self.indexNames_transposed = metadata['indexNames_transposed']      
        
        self.means = self.tns[self.meansName]
        self.covariances = self.tns[self.covariancesName]
        
        self._advancedmethodsUsable=False

    def _makeAdvancedMethodsUsable(self):
        if not self._advancedmethodsUsable:
            self._tns_local = _nt.TensorNameSpace(self.tns)
            self._tns_local.cloneIndex('g', 'g2')
            self._tns_local.cloneIndex('d', 'd2')
            self._tns_local.registerTensor('covs', self.tns[self.covariancesName].index_tuples,  external_array=self.tns[self.covariancesName].data, initial_values='keep')
            self._tns_local.renameIndices('covs', {'g_':'g2', 'd_':'d2'})
            self._tns_local.registerSlice('renamed(covs)', {'r':'motion', 'r_':'motion'},  result_name='motionmotion')
            self._tns_local.registerSlice('renamed(covs)', {'r':'effort', 'r_':'motion'},  result_name='effortmotion')
#            self._tns_local.renameIndices('motionmotion', {'g_':'g2', 'd_':'d2'})
#            self._tns_local.renameIndices('effortmotion', {'g_':'g2', 'd_':'d2'})

            self._tns_local.registerInverse('motionmotion', side='right', regularization=1e-5)
            self._tns_local.registerContraction('effortmotion', '(motionmotion)^#', result_name='gains_neg', align_result_to=(('g', 'd'),('g_','d_')) )        
            self._tns_local.registerScalarMultiplication('gains_neg', -1.0, result_name = 'gains')
            self._gainsequations = self._tns_local.update_order[:]

            self._tns_local.registerInverse('covs', result_name='precision', flip_underlines=False)
            self._precisionequations = ['precision']

            #set up slices to access each mechanical property individually:
            self._tns_local.registerTensor('means', self.tns[self.meansName])
            for r_idx, r_label in enumerate(self._tns_local['r'].values):
                for g_idx, g_label in enumerate(self._tns_local['g'].values):
                    commonname = self.rg_commonnames[r_idx][g_idx]
                    self._tns_local.registerSlice('covs', {'r':r_label, 'g':g_label, 'r_': r_label, 'g_': g_label},  result_name="covs_"+commonname)
                    slicename = self._tns_local.registerSlice('means', {'r':r_label, 'g':g_label},  result_name="means_"+commonname)

            self._advancedmethodsUsable = True
        
    def __repr__(self):
        text  = "Realms: {}, ".format(self.tns['r'].size)
        text += "Derivatives: {}, ".format(self.tns['g'].size)
        text += "Dofs: {}\n".format(self.tns['d'].size)
        for r_idx in range(self.tns['r'].size):            
            for g_idx in range(self.tns['g'].size):            
                name = self.rg_commonnames[r_idx][g_idx]
                mean = self.tns[self.meansName].data[r_idx,g_idx,:]
                var = _np.sqrt(self.tns[self.covariancesName].data_diagonal[r_idx,g_idx,:])
                text += "{}:\n     Mu: {}\n  Sigma: {}\n".format(name, mean, var)
        return text
    

        
    def extractTorqueControlGains(self):
        """
        compute and return PD controller gains implied by the covariance matrix
        
        Indices are in g,d,g,d order
        """
        #not very efficient to construct this every time, but it is convenient:
        self._makeAdvancedMethodsUsable()
        self._tns_local.update(self._gainsequations)
        return self._tns_local['gains'].data


    def getPrecision(self):
        """
        Interface to get a precision tensor even if it was not yet computed
        """                
        if self.precisionsName != None:
            return self.tns[self.precisionsName]
        else:
            self._makeAdvancedMethodsUsable()        
            self._tns_local.update(self._precisionequations)
            return self._tns_local['precision']
        

    def getMeansData(self, commonname=None):
        """
        return the means in a canonical form ( indicces in r,g,d order)
        
        If commonname is specified, return a view on the related subset of means
        """
        if commonname is None:
            return self.tns._alignDimensions( (('r','g','d'), ()),self.tns[self.meansName].index_tuples,   self.tns[self.meansName].data) 
        else:
            self._makeAdvancedMethodsUsable()
            return self._tns_local['means_'+commonname].data

    def getCovariancesData(self,commonname=None):
        """
        return the covariances in a canonical form ( indicces in r,g,d order)

        If commonname is specified, return a view on the related subset of covariances
        """
        if commonname is None:
            return self.tns._alignDimensions( (('r','g','d'), ('r_','g_','d_')),self.tns[self.covariancesName].index_tuples,   self.tns[self.covariancesName].data) 
        else:
            self._makeAdvancedMethodsUsable()
            return self._tns_local['covs_'+commonname].data

    def getVariancesData(self):
        """
        return the covariances in a canonical form ( indicces in r,g,d order)
        """
        return self.tns._alignDimensions( (('r','g','d'), ()), (self.tns[self.covariancesName].indices_upper,()),   self.tns[self.covariancesName].data_diagonal) 


    def addVariance(self, commonname, value):
        """
        Add the variances of the mentioned name to the given values
            
        Mostly useful for initializing distributions, or defining/adding uncorrelated noise    
        """
        self._makeAdvancedMethodsUsable()
        slicedtensor = self._tns_local["covs_"+commonname]
        slicedtensor.data_diagonal[...] += value
        

    def plotCorrelations(self):
        """
        plot the covariances/correlations
        
        normalize_indices: string of index letters used to select which dimension to normalize
            'rgsd': all indices (correlation matrix)
            '': verbatim covariance matrix
            ''rg': variances between realms and between derivatives are normalized (default)
        """
        cov = self.tns[self.covariancesName].data
        sigmas = _np.sqrt(self.tns[self.covariancesName].data_diagonal)
        title="Correlations"

        sigmamax_inv = 1.0 / _np.clip(sigmas, 1e-6, _np.inf)        
        cov_scaled = sigmamax_inv[:,:,:,None,None,None] * cov * sigmamax_inv[None,None,None,:,:,:] 
        vmax=_np.max(cov_scaled)

        len_r = self.tns['r'].size
        len_d = self.tns['d'].size
        len_g = self.tns['g'].size
        len_all = len_r * len_g * len_d

        cov_reordered =_np.transpose(cov_scaled, axes=(0,1,2, 0+3,1+3,2+3)) #to srgd
        image =_np.reshape(cov_reordered, (len_all,len_all))
        gridvectorX = _np.arange(0, len_all, 1)
        gridvectorY = _np.arange(len_all, 0, -1)

        fig = _plt.figure(figsize=(3.4,3.4))
        _plt.pcolor(gridvectorX, gridvectorY, image, cmap=_cmapCorrelations, vmin=-vmax, vmax=vmax)
        
        _plt.axis([0, image.shape[0], 0, image.shape[1]])
        _plt.gca().set_aspect('equal', 'box')

        line_positions = _np.reshape(_np.arange(len_all), cov_reordered.shape[:3])
        for r_idx in range(len_r):
           for g_idx in range(len_g):
                for d_idx in range(len_d):
                    linewidth=0.5
                    linestyle='-'
                    if g_idx!=0:
                        linewidth=0.2
                        linestyle=':'
                    if d_idx!=0:
                        linewidth=0.0
                    if linewidth>0.0:
                        _plt.axhline(line_positions[r_idx, g_idx,d_idx], color='k', linewidth=linewidth, linestyle=linestyle)
                        _plt.axvline(line_positions[r_idx, g_idx,d_idx], color='k', linewidth=linewidth, linestyle=linestyle)

        baselength = len_d
        ticklabels = []
        ticks=[]
        offsets=[]
        for r in range(len_r):
            for g2 in range(2*len_g):
                ticks.append( ((r)*len_r + g2/2)*baselength )
                g2mod2 = g2 % 2
                if g2mod2 == 1:
                    ticklabels.append(g2//2)
                    offsets.append(0.0)
                else:
                    ticklabels.append(r)
                    offsets.append(baselength)
        for tick, label, offset in zip(ticks, ticklabels, offsets):
            t = _plt.text(offset, tick, label, {'verticalalignment':'center', 'horizontalalignment':'right', 'size':'xx-small'})
        _plt.yticks([])
        _plt.text(0.0,ticks[0]+0.3*baselength, "$g$", fontdict={'verticalalignment':'bottom', 'horizontalalignment':'right', 'size':'small'})
        _plt.text(-10.0,ticks[0]+0.3*baselength, "$r$", fontdict={'verticalalignment':'bottom', 'horizontalalignment':'right', 'size':'small'})

#        #ticks in x:
#        ticks = range( (len_dtilde)//2, len_all, (len_dtilde))
#        ticklabels = []
#        ticks=[]
#        offsets=[]
#        for s in range(len_stilde):
#            for r in range(len_rtilde):
#                for g in range(len_gtilde):
#                    ticks.append( (((s)*len_rtilde + r)*len_gtilde + g)*len_dtilde + len_dtilde/2 )
#                    ticklabels.append(g)
#                    offsets.append(-1.0)
#        for tick, label, offset in zip(ticks, ticklabels, offsets):
#            t = _plt.text(tick, offset, label, fontdict={'verticalalignment':'top', 'horizontalalignment':'center', 'size':'xx-small'}, rotation=0)
#        _plt.text(ticks[-1]+10, 0.0, "$\widetilde{g}$", fontdict={'verticalalignment':'top', 'horizontalalignment':'left', 'size':'small'})
            
        _plt.xticks([])


        _plt.colorbar(shrink=0.6, aspect=40, ticks=[-vmax,0,vmax], fraction=0.08)
        _plt.title(title)
        ax = _plt.gca()        
        #_plt.tight_layout()




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



