################################################################################
#                                                                              #
#                                   NOVELTY                                    #
#                             OFFLINE AND ONLINE                               # 
#                                                                              #
#                                17 JULY 2023                                  #
#                                                                              #
#                          Juan Ignacio Mendoza Garay                          #
#                               doctoral student                               #
#                 Department of Music, Art and Culture Studies                 #
#                            University of Jyväskylä                           #
#                                                                              #
################################################################################

# INFORMATION:

# Tested with Python 3.8.10

# Description:
#     Demonstrates computation of novelty. The online method gets more efficient
#     as the input increases in size.

# Instructions:
#     Edit the values indicated with an arrow like this: <---
#     Run the program, close your eyes and hope for the best.

# ==============================================================================

import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plot
import tkinter
from scipy import signal
from scipy.spatial.distance import pdist, squareform
import time

# ------------------------------------------------------------------------------
# PARAMETERS:

# input signal:
c = 2        # <--- cycles
g = 0.08     # <--- grid's granularity
l = 3.14 * 2 # <--- signal's length
s_type = 'square' # <--- signal type ('sine','square')

# novelty:
kernel_size = 10  # <--- kernel's size
s = kernel_size/4 # <--- kernel's gaussian taper standard deviation

# visualisation:
vis_opt = 2 # <--- 0 = don't visualise, 1 = show plots, 2 = show animation
            #      0 and 1 also display computation time

# ------------------------------------------------------------------------------

if kernel_size%2 != 0 :
    print('\n Error: kernel_size should be an even numer.')
    exit(1)

x = np.arange(0, l, g) # grid

if vis_opt == 1:

    if x.size > 5000:
        ans_1 = input ('\n WARNING:the Process might get very slow.\n Proceed anyway? (y/n)').strip()
        if ans_1 != ('y'or'Y'):
            exit(1)

elif vis_opt == 2:

    if x.size > 150:
        ans_1 = input ('\n WARNING:the animation might get very slow.\n Proceed anyway? (y/n)').strip()
        if ans_1 != ('y'or'Y'):
            exit(1)
    
    plot.rcParams.update({'font.size': 14})
    fig, axs = plot.subplots(4)
    fig.tight_layout()
    plot.setp( [axs[a] for a in [0,1,3]] , xlim=(0,x[x.size-1]) , xlabel='time' )
    plot.show(block=False)
    plot.draw()
    
    root = tkinter.Tk()
    root.withdraw()
    WIDTH, HEIGHT = root.winfo_screenwidth(), root.winfo_screenheight()
    fig.set_size_inches(WIDTH/fig.dpi,HEIGHT/fig.dpi)
    fig.canvas.manager.window.wm_geometry('+0+0')

    axs[0].set_title('INPUT SIGNAL')
    axs[0].set_ylabel('amplitude')

    axs[1].set_title('OFFLINE NOVELTY')
    axs[1].set_ylabel('novelty')

    axs[2].set_yticks([])
    axs[2].set_xticks([])

    axs[3].set_title('ONLINE NOVELTY')
    axs[3].set_ylabel('novelty')

    pos_2 = axs[2].get_position()
    x_step_2 = pos_2.width / x.size
    x_pad_2 = ( pos_2.width * kernel_size )/ ( 2 * x.size )
    x_offset_2 = pos_2.x0 + x_pad_2 - (pos_2.height / 2) + 0 # last value is fine-tuning

# ------------------------------------------------------------------------------
# INPUT SIGNAL:

if s_type == 'sine':
    y = np.sin(x*c)
elif s_type == 'square':
    y = signal.square( x * c )

if vis_opt == 1:

    plot_y = plot.figure()
    plot.plot(x,y)
    plot.title('INPUT SIGNAL')
    plot.xlabel('time')
    plot.ylabel('amplitude')
    plot_y.show()

elif vis_opt == 2:
    
    axs[0].plot(x,y,'k')
    
# ..............................................................................
# OFFLINE NOVELTY:
# The kernel is correlated upon the distance matrix of the whole signal (Foote, 2000).

kernel_size_half = int(kernel_size/2)

xx = np.linspace(-kernel_size_half,kernel_size_half,kernel_size)
yy = np.linspace(-kernel_size_half,kernel_size_half,kernel_size)
xx, yy = np.meshgrid(xx, yy)

gauss_2D = 1. / (2. * np.pi * s**2) * np.exp(-(xx**2. / (2. * s**2.) + yy**2. / (2. * s**2.)))

kron_cb = np.kron( [[-1,1],[1,-1]] , np.ones( (kernel_size_half,kernel_size_half) ) )

gausscb_kernel = gauss_2D * kron_cb

novelty_off = np.empty( y.size ) # initialise novelty vector
novelty_off.fill(np.nan)

tic = time.time()

dist_m = squareform( pdist( np.column_stack( (y, y) ) , 'euclidean') )

for i_start in np.arange(0,y.size-kernel_size):
    i_end = i_start + kernel_size
    novelty_off[ i_start + kernel_size_half ] = np.sum( gausscb_kernel * dist_m[ i_start:i_end , i_start:i_end ] )

toc =  time.time() - tic

if vis_opt == 1:

    plot_novelty_off = plot.figure()
    plot.plot(x,novelty_off)
    plot.title('OFFLINE NOVELTY')
    plot.xlabel('time')
    plot.ylabel('novelty')
    plot_novelty_off.show()

    
elif vis_opt == 2:

    axs[1].plot(x,novelty_off,'b')

if vis_opt in [0,1]:
    
    print( '\n OFFLINE NOVELTY computation time: %0.5f s.' % toc )

# ..............................................................................
# ONLINE NOVELTY:
# The kernel is correlated upon a local distance matrix (Schätti, 2007).
# To make it efficient, only half of the symmetric kernel is used. Also, the
# distances are computed only once and then shifted.


novelty_on = np.empty( y.size ) # initialise novelty vector
novelty_on.fill(np.nan)

gausscb_kernel_v = gausscb_kernel[np.triu_indices(kernel_size, k = 1)] # upper triangle of kernel without main diagonal as vector (concatenated rows)
gausscb_kernel_v = gausscb_kernel_v * 2.8284 # I dunno why

local_dist_v = np.zeros( gausscb_kernel_v.size ) # initialise local distance matrix as vector
local_dist_v_end = local_dist_v.size - 1

# make indices of matrix rows in the local distance vector:

local_buffer_end = kernel_size - 1
iv = np.zeros((local_buffer_end,3),dtype=int) # initialise indices where cols: start of range to shift (start of row), end of range to shift (end of row-1), end of row (new distance)

this_end = 0

for i_row in range(0,local_buffer_end): 

    iv[i_row,0] = this_end # start of row
    iv[i_row,2] = iv[i_row,0] + local_buffer_end - i_row # end of row
    this_end = iv[i_row,2]

iv[:,1] = iv[:,2] - 1 # end of row - 1
n_shifts = kernel_size-2;

tic = time.time()
    
for i_start in range(0,y.size-kernel_size):
    
    i_end = i_start + kernel_size
    local_buffer = y[ i_start:i_end ]

    for i_kr in range(0,n_shifts):

        local_dist_v[ iv[i_kr,0] : iv[i_kr,1] ] = local_dist_v[ iv[i_kr+1,0] : iv[i_kr+1,2] ] # shift rows up using indices of vector
        local_dist_v[ iv[i_kr,1] ] = np.absolute( local_buffer[ local_buffer_end ] - local_buffer[ i_kr ] ) # last in row is new distance

    local_dist_v[ local_dist_v_end ] = np.absolute( local_buffer[ local_buffer_end ] - local_buffer[ local_buffer_end - 1 ] ) # last of triu
    novelty_on[ i_start + kernel_size_half ] = np.dot( gausscb_kernel_v , local_dist_v )

    if vis_opt == 2:

        axs[2].set_position([ i_start * x_step_2 + x_offset_2 , pos_2.y0 , pos_2.height , pos_2.height ])
        axs[2].imshow( squareform(local_dist_v) , 'summer')
        axs[2].set_xlabel('time: %s' %round(x[i_start],3) )

        axs[3].plot(x,novelty_on,'r')
        
        plot.pause(0.01)
        
toc =  time.time() - tic

if vis_opt == 1:
    
    plot_novelty_on = plot.figure()
    plot.plot(x,novelty_on)
    plot.title('ONLINE NOVELTY')
    plot.xlabel('time')
    plot.ylabel('novelty')
    plot_novelty_on.show()

if vis_opt in [0,1]:
    
    print( '\n ONLINE NOVELTY computation time: %0.5f s.' % toc )
        
print('\n done')
