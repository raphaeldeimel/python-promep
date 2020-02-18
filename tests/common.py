#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Test the trajectory compositors
@author: Raphael Deimel
"""

import os
import matplotlib.pylab as pylab
import inspect

def savePlots():
    try:
        os.mkdir('plots')
    except FileExistsError:
        pass

    #get callee name:
    frame,filename,line_number,function_name,lines,index = inspect.stack()[1] 
    myname = os.path.splitext(os.path.basename(filename))[0]

    for n in pylab.get_fignums():    
        fig = pylab.figure(n)
        ax  = fig.axes[0]
        ax.title.set_visible(False)        
        if not fig._suptitle is None:
            title=fig._suptitle.get_text()
            fig._suptitle.set_visible(False)
        else:
            title = ax.get_title()
            
        descriptor = "_".join((myname, "fig"+str(n), title ))
        print("saving {}".format(descriptor))
        if "REFERENCE" in os.environ:
            filename="./plots/{0}_ref.pdf".format(descriptor)
        else:
            filename="./plots/{0}.pdf".format(descriptor)
        fig.savefig(filename)
        


        
