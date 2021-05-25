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
from numba import jit

#%% Naive implementatin of the mandelbrot set
@jit
def mandelbrot_naive(im_res, re_res, ite): # Inputs are resolution of the real and imaginary axis and the amount of iterations to check
    time_a = time.perf_counter() # Start time
    
    mandelset = np.zeros((re_res, im_res)) # Creates a matrix of zeroes of appropriate size
    
    mandelset = matrix_calc(mandelset, ite) # Sends the matrix to the function responsible indexing and calculating values
    
    time_b = time.perf_counter() # End time
    time_dif = time_b-time_a # Time differences
    
    fig1 = plt.figure(figsize = (15,15)) # Crewates a figure of certain size
    ax1 = fig1.add_subplot(1,1,1) # Adds a supplot where an image can be added to
    ax1.imshow(mandelset,'hot',interpolation = 'none' ,extent=[-2.0,1.0,-1.5,1.5]) # Loads the matrix, set the color gradient, turns off interpolation and sets axis values
    ax1.set_title('Vector numba implementation, Time: ' + str(round(time_dif,3)) + " Res: " + str(im_res)) # Adds title, and time to calculate
    
    return 1

@jit
def matrix_calc(matrix, max_ite): #Takes the matrix and delegates the work
    x_val = 3/(len(matrix[0])-1) # Finds the columns of the matrix
    y_val = 3/(len(matrix)-1) # Finds the rows of the matrix
    y_pos = -1.5 # Current y position
    y_ite = 0 # Current row number
    
    
    # 32 kb lvl 1 chahce, complex number is 80 bytes, iteration counter is integer, 4 bytes. length is float 8 bytes. Total 172 bytes per number pair. 32000/172 = 186.
    # or around 180 points at once in lvl 1 chache.
    
    while(y_ite <= len(matrix)-1):

        matrix_width = len(matrix[0]-1)
            
        matrix[y_ite,:] = divergence_vector(matrix_width ,y_pos, x_val, max_ite)
            
        y_pos = y_pos + y_val
        y_ite = y_ite + 1
            

        
        
        
    return matrix # Returns result

@jit
def divergence_vector(x_size, y_cord, x_ini, ite): # Function to check wether a point is diverging or not

    cord_vector = np.zeros((1,x_size),dtype=complex)
    vec_len = len(cord_vector[0])
    
    ite_var = 0
    x_tmp = -2.0
    while(ite_var <= vec_len-1):
        cord_vector[0,ite_var] = complex(x_tmp,y_cord)
        ite_var = ite_var + 1
        x_tmp = x_tmp + x_ini

    div_list = list(range(0,vec_len))
    z_tmp_vec = np.zeros((1,vec_len),dtype=complex)
    z_len_vec = abs(cord_vector)
    cur_ite_vec = np.zeros((1,vec_len)) # Iteration counter vector

    
    ite_var = 0
    while(np.amin(z_len_vec) <= 2 and ite_var <= ite):
        
        
        div_list = np.where((z_len_vec <= 2) == 1)
        
        z_tmp_vec[0,div_list[1]] = ((z_tmp_vec[0,div_list[1]])**2 + (cord_vector[0,div_list[1]]))
        
        z_len_vec[0,div_list[1]] = np.abs(z_tmp_vec[0,div_list[1]])

        cur_ite_vec[0,div_list[1]] = cur_ite_vec[0,div_list[1]] + 1

        
        ite_var = ite_var + 1

    
    return cur_ite_vec # Returns the amount of iterations before convergense or max if reached




un_val = 1000 # Matrix resolution
mandelbrot_naive(un_val,un_val,50) # Calling the function
