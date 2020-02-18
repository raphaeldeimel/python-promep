#!/bin/ipython

import numpy
import numpy as _np

import common

#visualize the decision landscape for a simple case: three 1D distributions:

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d


import numpy as _np
import matplotlib.pyplot as plt
import matplotlib.patches
from matplotlib.animation import FuncAnimation

steps_animation=500
sequences_n = 4


def plot_cov_ellipse(ellipses, cov, pos=[0.0, 0.0], nstds=[0.0,1.0,2.0], **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstds : The radii of the ellipses in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = _np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]


    vals, vecs = eigsorted(cov)
    theta = _np.degrees(_np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    sigma_max = 0.5
    alpha = min(0.8,  _np.prod(sigma_max /_np.sqrt(vals)))
    for i,e in enumerate(ellipses):
        sigma = nstds[i]
        width, height = 2 * sigma * _np.sqrt(vals)
        #ellipses[i].center = pos
        e.set_alpha(alpha)
        if sigma > 0.1: #if this is below, then treat ellipse as a center circle and do not modify size at all
            e.width = width
            e.height= height
            e.angle = theta
        e.center = pos
        e.set(**kwargs)

#        e.fill=True
#        e.set_linewidth(0.0)


    return ellipses


S1 = _np.array([[0.710,0.70],[0.70, 0.710]])
S2 = _np.array([[1.0,0.0],[0.0,1.0]])
S3 = _np.array([[0.715,-0.705],[-0.705, 0.715]])
S4 = _np.array([[0.860,-0.600],[-0.600, 0.450]])
S5 = _np.array([[2.5,0.0],[0.0,2.5]])



#covariances = [S3, S3, S2, S2, S2]
#means = [_np.array([-0.0 , 0.]),_np.array([0.0,0.0]),_np.array([-0.0 , 0.]),_np.array([-0.0 , 0.]),_np.array([-0.0 , 0.])]

#covariances = [S3, S3, S2, S2, S2]
#means = [_np.array([-0.7 , 0.]),_np.array([0.4,0.4]),_np.array([-0.0 , 0.]),_np.array([-0.0 , 0.]),_np.array([-0.0 , 0.])]

covariances = [S1, S3, S4, S5, S2]
means = [_np.array([-0.5 , 0.]),_np.array([0.5,0.5]),_np.array([-0.0 , -1.2]),_np.array([0.0 , -0.0]),_np.array([-0.0 , 0.])]
colors_dists = [(1,0,0), (0,1,0), (0,0.7,0.3), (0,0.3,0.7), (0.5,0.5,0.5)]




fig, ax_array = plt.subplots(2,1, gridspec_kw={'width_ratios':[1.0],'height_ratios':[1.0, 0.2]}, figsize=(10.0, 12.0))
ax = ax_array[0]
ax.set_axis_off()
ax_supplemental = ax_array[1]

ax.set_aspect('equal', adjustable='box')
ax_supplemental.set_aspect(0.2, adjustable='box')

#set up ellipses that indicate the distributions: being mixed
ellipses_decoration= []
for i in range(sequences_n):
    e =  matplotlib.patches.Ellipse(xy=[0,0], width=0.0, height=0.0, angle=0., joinstyle='round', fill=True, lw=0.0, alpha=0.2)
    ax.add_patch(e)
    ellipses_decoration.append(e)
    plot_cov_ellipse([e],covariances[i], means[i], color=(0.5,0.5,0.5), alpha=0.3, nstds=[1.0])


#set up two sets of ellipses for visualizing the two different mixing methods:

cov_combined = covariances[0].copy()
mean_combined =means[0].copy()
mean_combined_paraschos = means[0].copy()
mean_combined_deterministic=means[0].copy()

centercirclesize=0.05
ellipses_reference = [
        matplotlib.patches.Ellipse(xy=[0,0], width=centercirclesize, height=centercirclesize, angle=0., joinstyle='round', fill=True, lw=0.0),
        matplotlib.patches.Ellipse(xy=[0,0], width=2.0, height=0.4, angle=0, joinstyle='round', fill=False, lw=4.0),
]
ellipses_proposedmethod = [
        matplotlib.patches.Ellipse(xy=[0,0], width=centercirclesize, height=centercirclesize, angle=0., joinstyle='round', fill=True, lw=0.0),
        matplotlib.patches.Ellipse(xy=[0,0], width=2.0, height=0.4, angle=0, joinstyle='round', fill=False, lw=4.0),
]
ellipses_deterministic= [
        matplotlib.patches.Ellipse(xy=[0,0], width=centercirclesize, height=centercirclesize, angle=0., joinstyle='round', fill=True, lw=0.0),
        matplotlib.patches.Ellipse(xy=[0,0], width=2.0, height=0.4, angle=0, joinstyle='round', fill=False, lw=4.0),
]
ellipses_animated = ellipses_proposedmethod + ellipses_reference + ellipses_deterministic
for e in ellipses_animated :
    e.set_animated(True)
    ax.add_patch(e)

culling_steps = 1 # 10
history_center_x = _np.full((steps_animation//culling_steps), _np.nan)
history_center_y = _np.full((steps_animation//culling_steps), _np.nan)
if culling_steps == 1:
    tracelines_center = ax.plot(history_center_x, history_center_y, color='k', lw=0.2)
else:
    tracelines_center = ax.plot(history_center_x, history_center_y, color='k', marker='x', lw=0.0);
center_index_counter = 0

def update_trace(x,y, which=0):
    global tracelines_center, history_center_x, history_center_y, center_index_counter
    center_index_counter = (center_index_counter+ 1)
    if center_index_counter % culling_steps == 0:
        if center_index_counter >= history_center_x.size * culling_steps:
             center_index_counter = 0
        i = center_index_counter // culling_steps
        history_center_x[i] = x
        history_center_y[i] = y
        tracelines_center[0].set_data(history_center_x,history_center_y)
    return tracelines_center

#activations plot:
activationlines_n = 4
times_index=0
times = _np.linspace(0.,sequences_n,steps_animation, endpoint=False)
histories_activation = _np.full((steps_animation,activationlines_n), _np.nan)
activations_offsets = 0.0 * _np.arange(0,activationlines_n)
activations_colors = [ (i, 0, 1-i) for i in _np.linspace(0.,1.,activationlines_n)]
activation_lines = []
for i,k in enumerate(_np.linspace(0.,1.0,activationlines_n)):
    l, = ax_supplemental.plot(times , histories_activation[:,i], color=(k, 1.0*(k**2), 1.0-k), lw=3.0, alpha=0.7)
    l.set_animated(True)
    activation_lines.append(l)    


def update_activations(factors):
    global times_index, histories_activation, activation_lines, activations_offsets
    times_index = (times_index+ 1) % steps_animation
    histories_activation[times_index,:] = factors[:activationlines_n] + activations_offsets
    for i in range(histories_activation.shape[1]):
        activation_lines[i].set_data(times, histories_activation[:,i])
    return activation_lines


def init():
    global ellipses_decoration, video_variant, activation_lines, histories_activation, center_index_counter,  tracelines_center
    ax.set_xlim(-2.5, 3.5)
    ax.set_ylim(-3.0, 1.5)
    ax_supplemental.set_xlim(0, sequences_n)
    ax_supplemental.set_ylim(-0.05, 1.05)
    histories_activation[:,:] = _np.nan
    history_center_x[:] = _np.nan
    history_center_y[:] = _np.nan
    center_index_counter  = 0
    tracelines_center[0].set_data(history_center_x,history_center_y)
#    return update(0.0)
    artists_to_draw = ellipses_decoration + activation_lines
    if show_trace:
        artists_to_draw += tracelines_center
    return  artists_to_draw




def update(frame):
    global cov_combined, mean_combined, covariances, means, ellipses_proposedmethod, ellipses_reference
    global show_reference, show_proposed, show_deterministic, show_trace
    global means_velocity, mean_combined_paraschos, mean_combined_deterministic

    artists_to_draw = ellipses_decoration.copy()

    stepsize = 4.0 / steps_animation
    idx = int(frame / stepsize)
    
    
    sequence = int(frame // 1)
    c = frame - sequence
    factors_reference = [0.0] * sequences_n
    factors_reference[ sequence] = _np.clip(1.*(1. - c), 0.0, 1.0)
    factors_reference[(sequence+1) % sequences_n] = _np.clip(1.*c, 0.0, 1.0)

    factors = _np.clip(activations_scale*_np.array(factors_reference), 0.0, 1.0)

    if show_reference:
        mean_last = mean_combined_paraschos
        cov_combined_paraschos, mean_combined_paraschos = common.distribution_product(covariances, means, factors_reference)
        plot_cov_ellipse(ellipses_reference,cov_combined_paraschos, mean_combined_paraschos, color=(0.2, 0.2, 0.2))
        artists_to_draw += ellipses_reference
        means_velocity[idx, 0] = _np.linalg.norm( (mean_combined_paraschos - mean_last)/stepsize)
        mean_traced = mean_combined_paraschos
    else:
        plot_cov_ellipse(ellipses_reference,cov_combined, mean_combined, alpha=0.0)
    #for using weighted mean of covariances instead of last value:
    #cov_combined = _np.sum([C*f for C,f in zip(covariances, factors)], axis=0)
    if show_deterministic:
        mean_last = mean_combined_deterministic
        cov_combined_deterministic, mean_combined_deterministic = common.deterministic_interpolation(covariances, means, factors_reference)
        plot_cov_ellipse(ellipses_deterministic,cov_combined_deterministic, mean_combined_deterministic, color=(0.2, 0.2, 0.2))
        artists_to_draw += ellipses_deterministic
        means_velocity[idx, 2] = _np.linalg.norm( (mean_combined_deterministic - mean_last)/stepsize)
        mean_traced = mean_combined_deterministic
    else:
        plot_cov_ellipse(ellipses_deterministic,cov_combined, mean_combined,  alpha=0.0)
    if show_proposed:
        mean_last = mean_combined
        for j in range(1):
            cov_combined, mean_combined = common.distribution_product_regularized(covariances, means, factors, cov_combined, mean_combined )
        plot_cov_ellipse(ellipses_proposedmethod, cov_combined, mean_combined, color=(0.0, 1.0, 0.0))
        artists_to_draw += ellipses_proposedmethod
        means_velocity[idx, 1] = _np.linalg.norm((mean_combined-mean_last)/stepsize)        
        mean_traced = mean_combined
    else:
        plot_cov_ellipse(ellipses_proposedmethod, cov_combined, mean_combined, alpha=0.0)
    artists_activation = update_activations(factors)

    if show_trace:
        traceartists = update_trace(mean_traced[0],mean_traced[1])
        artists_to_draw += traceartists

    return artists_to_draw + artists_activation

frames = _np.linspace(0.0, 4.0, steps_animation, endpoint=False)

means_velocity = _np.zeros((steps_animation,3))


activations_scale=1.0


show_deterministic=True
show_reference=True
show_proposed=True
show_trace = False
ani = FuncAnimation(fig, update, frames=frames,init_func=init, blit=True, interval=20)
ani.save("video_all.mp4")
del ani
print('finished video_all.mp4')


show_deterministic=False
show_reference=True
show_proposed=False
show_trace = True
ani = FuncAnimation(fig, update, frames=frames,init_func=init, blit=True, interval=20)
ani.save("video_reference.mp4")
del ani
print('finished video_reference.mp4')

show_deterministic=True
show_reference=False
show_proposed=False
show_trace = True
ani = FuncAnimation(fig, update, frames=frames,init_func=init, blit=True, interval=20)
ani.save("video_deterministic.mp4")
del ani
print('finished video_deterministic.mp4')


show_deterministic=False
show_reference=False
show_proposed=True
show_trace = True
ani = FuncAnimation(fig, update, frames=frames,init_func=init, blit=True, interval=20)
ani.save("video_proposed.mp4")
del ani
print('finished video_proposed.mp4')

#plt.figure()
#plt.plot(frames, means_velocity[:,0], label='reference')
#plt.plot(frames, means_velocity[:,1], label='proposed')
#plt.plot(frames, means_velocity[:,2], label='deterministic')
#plt.savefig("velocity_mean.pdf", bbox_inches='tight')

#plt.figure()
#plt.plot(frames[:-1], _np.abs(steps_animation/4.0*(means_velocity[1:,0]- means_velocity[:-1,0])), label='reference')
#plt.plot(frames[:-1], _np.abs(steps_animation/4.0*(means_velocity[1:,1]- means_velocity[:-1,1])), label='proposed')
#plt.plot(frames[:-1], _np.abs(steps_animation/4.0*(means_velocity[1:,2]- means_velocity[:-1,2])), label='deterministic')
#plt.savefig("acceleration_mean.pdf", bbox_inches='tight')


activations_scale=0.1

show_deterministic=True
show_reference=False
show_proposed=True
show_trace = True
ani = FuncAnimation(fig, update, frames=frames,init_func=init, blit=True, interval=20)
ani.save("video_activation_01.mp4")
del ani
print('finished video_activation_01.mp4')

activations_scale=3.0

show_deterministic=False
show_reference=True
show_proposed=True
show_trace = True
ani = FuncAnimation(fig, update, frames=frames,init_func=init, blit=True, interval=20)
ani.save("video_activation_3.mp4")
del ani
print('finished video_activation_3.mp4')

#plt.ion();plt.show()

