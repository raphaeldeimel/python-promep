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

class MechanicalStateDistribution(object):

    def __init__(self, means, covariances):
        self.means = _np.array(means)
        self.covariances = _np.array(covariances)
        self.shape = means.shape
        self.indexNames = ['r','d','g']
        self.indexSizes = _collections.OrderedDict({  #OrderedDict for python2 backward compatibility
                'r': self.covariances.shape[0],
                'd': self.covariances.shape[1],
                'g': self.covariances.shape[2],
                'r_': self.covariances.shape[3],
                'd_': self.covariances.shape[4],
                'g_': self.covariances.shape[5],
        })
        self.indexNames_transposed = ['r_','d_','g_']
        
        #create a view on the variances within the covariance tensor:
        self.variancesView = _np.einsum('ijkijk->ijk', self.covariances)
    
    def __repr__(self):
        text  = "Realms: {}\n".format(self.shape[0])
        text += "Dofs: {}\n".format(self.shape[1])
        text += "Derivatives: {}\n".format(self.shape[2])
        for g_idx in range(self.shape[2]):            
            text += "\nDerivative {}:\n       Means:\n{}\n       Variances:\n{}\n".format(g_idx, self.means[:,:,g_idx], self.variancesView[:,:,g_idx])
        return text
        
    def extractPDGains(self, realm_motion=0, realm_effort=1):
        """
        compute and return PD controller gains implied by the covariance matrix
        """
        dofs = self.indexSizes['d']
        subcov_shape = (dofs * self.indexSizes['g'], dofs * self.indexSizes['g'])
        gains = _np.zeros((dofs,dofs,self.indexSizes['g']))

        sigma_qt = self.covariances[realm_motion, :, :,realm_effort, :,:].reshape(subcov_shape)
        sigma_qq = self.covariances[realm_motion, :, :,realm_motion, :,:].reshape(subcov_shape)

        sigma_qq_inv = _np.linalg.pinv(sigma_qq)
        gains = -1 * _np.dot(sigma_qt,sigma_qq_inv) 
        gains.shape = (dofs, self.indexSizes['g'], dofs, self.indexSizes['g'])

        return gains


    def plotCorrelations(self):
        """
        plot the covariances/correlations
        
        normalize_indices: string of index letters used to select which dimension to normalize
            'rgsd': all indices (correlation matrix)
            '': verbatim covariance matrix
            ''rg': variances between realms and between derivatives are normalized (default)
        """
        cov = self.covariances
        variance_view = _np.einsum('ijkijk->ijk', self.covariances)

        sigmas = _np.sqrt(variance_view)
        title="Correlations"

        sigmamax_inv = 1.0 / _np.clip(sigmas, 1e-6, _np.inf)        
        cov_scaled = sigmamax_inv[:,:,:, None,None,None] * cov * sigmamax_inv[None,None,None, :,:,:] 
        vmax=_np.max(cov_scaled)

        len_r = self.indexSizes['r']
        len_d = self.indexSizes['d']
        len_g = self.indexSizes['g']
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



