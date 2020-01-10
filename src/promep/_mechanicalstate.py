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

class MechanicalStateDistribution(object):

    def __init__(self, means, covariances):
        self.means = _np.array(means)
        self.covariances = _np.array(covariances)
        self.shape = means.shape
        self.indexNames = ['r','d','g']
        self.indexSizes = _collections.OrderedDict({  #OrderedDict for python2 backward compatibility
                'r': self.shape[0],
                'd': self.shape[1],
                'g': self.shape[2],
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

