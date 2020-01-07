#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

test mixing of two promps

@author: raphael
"""

import sys
sys.path.insert(0,"../")
import promp

import numpy as _np
import itertools as _it
import os
import scipy
import matplotlib.pylab as pylab

import hdf5storage as _h5


#try to change this:
nSupports=9 #nr of supports / basis functions
dofs = 1
nSteps= 100
nTrajectories = 10


trajectorypositions = []
phases = _np.linspace(0.0,1.0,nSteps)
for i in range(nTrajectories):
    a = _np.zeros((dofs, nSteps))
    a[0,:] = phases * 100.0  + _np.random.normal(0.0, 10.0) * (1-phases)
    trajectorypositions.append(a)


mp1 = promp.ProMPFactory.makeFromPositionTrajectories("mp1", nSupports, trajectorypositions)

tmppath = "/tmp/python-promp-tests/"
os.makedirs(tmppath, exist_ok=True)

#serialize via file:
filename = mp1.saveToFile(path=tmppath)
dRestored = _h5.read(filename=filename)
mpRestored = promp.ProMPFactory.makeFromFile(filename)
if not (_np.allclose(mp1.meansMatrix, mpRestored.meansMatrix) and  _np.allclose(mp1.covarianceTensor, mpRestored.covarianceTensor)):
    print("means and covariances not saved and loaded correctly #1")

#serialize via dict:
serialized = mpRestored.serialize()
mpRestoredRestored = promp.ProMPFactory.makeFromDict(serialized)
if not (_np.allclose(mpRestoredRestored.meansMatrix, mpRestored.meansMatrix) and  _np.allclose(mpRestoredRestored.covarianceTensor, mpRestored.covarianceTensor)):
    print("means and covariances not saved and loaded correctly #2")



if __name__=='__main__':
    import os
    for n in pylab.get_fignums():    
        myname = os.path.splitext(os.path.basename(__file__))[0]
        if "REFERENCE" in os.environ:
            filename="./plots/{0}_fig{1}_ref.pdf".format(myname,n)
        else:
            filename="./plots/{0}_fig{1}.pdf".format(myname,n)
        pylab.figure(n).savefig(filename)
