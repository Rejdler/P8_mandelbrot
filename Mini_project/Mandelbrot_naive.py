# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 18:02:06 2021

@author: dksan
"""

#%% Importing important libraries
import time
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

#%% Naive implementatin of the mandelbrot set

def mandelbrot_naive(im_res, re_res, ite):  # Inputs are resolution of the real and imaginary axis and the amount of iterations to check
    time_a = time.perf_counter()            # Start time
    
    mandelset = np.zeros((re_res, im_res))  # Creates a matrix of zeroes of appropriate size
    
    mandelset = matrix_calc(mandelset, ite) # Sends the matrix to the function responsible indexing and calculating values
    
    time_b = time.perf_counter()            # End time
    time_dif = time_b-time_a                # Time differences
    
    fig1 = plt.figure(figsize = (15,15))    # Crewates a figure of certain size
    ax1 = fig1.add_subplot(1,1,1)           # Adds a supplot where an image can be added to
    ax1.imshow(mandelset,'hot',interpolation = 'none' ,extent=[-2.0,1.0,-1.5,1.5]) # Loads the matrix, set the color gradient, turns off interpolation and sets axis values
    ax1.set_title('Naive implementation, Time: ' + str(round(time_dif,3)) + " Res: " + str(im_res)) # Adds title, and time to calculate
    
    return 1


def matrix_calc(matrix, max_ite):   #Takes the matrix and delegates the work
    x_val = 3/(len(matrix[0])-1)    # Finds the columns of the matrix
    y_val = 3/(len(matrix)-1)       # Finds the rows of the matrix
    x_pos = -2.0    # Current x position
    y_pos = -1.5    # Current y position
    x_ite = 0       # Current column number
    y_ite = 0       # Current row number
    
    while(y_ite <= len(matrix) - 1):        # Iterates over the rows
        while(x_ite <= len(matrix[0]) - 1): # Iterates over the columns
            matrix[y_ite,x_ite] = divergence_naive(x_pos, y_pos, max_ite)   # Calcualtes the iterations value for the current point
            x_pos = x_pos + x_val           # Updates x position
            x_ite = x_ite + 1               # Updates column number
            
        y_pos = y_pos + y_val   # Updates y position
        x_pos = -2.0            # Resets column number for inner loop
        x_ite = 0               # Resets column number
        y_ite = y_ite + 1       # Updates row number
    
    return matrix               # Returns result


def divergence_naive(x_cord, y_cord, ite): # Function to check wether a point is diverging or not

    xy_tmp = complex(x_cord,y_cord) # Complex value for current coordinate
    z_tmp = complex(0,0)            # Value that get's iterated to see if it diverges
    z_len = abs(xy_tmp)             # Length of current iteration z
    cur_ite = 0                     # Iteration counter
    
    while(z_len <= 2 and cur_ite <= ite - 1):   # Loop that test the mandelbrot set conditions of divergens
        
        z_tmp = z_tmp**2 + xy_tmp               # Update equation
        
        z_len = abs(z_tmp)       # Calculates length of new z value
        
        cur_ite = cur_ite + 1   # Update iteration counter
        
    return cur_ite # Returns the amount of iterations before convergense or max if reached



un_val = 100                  # Matrix resolution
mandelbrot_naive(un_val,un_val,50)  # Calling the function
