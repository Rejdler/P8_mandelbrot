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
def mandelbrot_naive(im_res, re_res, ite): # Inputs are resolution of the real and imaginary axis and the amount of iterations to check
    time_a = time.perf_counter() # Start time
    
    mandelset = np.zeros((re_res, im_res)) # Creates a matrix of zeroes of appropriate size
    
    mandelset = matrix_calc(mandelset, ite) # Sends the matrix to the function responsible indexing and calculating values
    
    time_b = time.perf_counter() # End time
    time_dif = time_b-time_a # Time differences
    
    fig1 = plt.figure(figsize = (15,15)) # Crewates a figure of certain size
    ax1 = fig1.add_subplot(1,1,1) # Adds a supplot where an image can be added to
    ax1.imshow(mandelset,'hot',interpolation = 'none' ,extent=[-2.0,1.0,-1.5,1.5]) # Loads the matrix, set the color gradient, turns off interpolation and sets axis values
    ax1.set_title('Vector implementation, Time: ' + str(round(time_dif,3)) + " Res: " + str(im_res)) # Adds title, and time to calculate
    
    return 1


def matrix_calc(matrix, max_ite): #Takes the matrix and delegates the work
    x_val = 3/(len(matrix[0])-1) # Finds the columns of the matrix
    y_val = 3/(len(matrix)-1) # Finds the rows of the matrix
    x_pos = -2.0 # Current x position
    y_pos = -1.5 # Current y position
    x_ite = 0 # Current column number
    y_ite = 0 # Current row number
    
    size_cache = 590
    # 32 kb lvl 1 chahce, complex number is 80 bytes, iteration counter is integer, 4 bytes. length is float 8 bytes. Total 172 bytes per number pair. 32000/172 = 186.
    # or around 180 points at once in lvl 1 chache.
    
    # while(y_ite <= len(matrix) - 1): # Iterates over the rows
    #    while(x_ite <= len(matrix[0]) - 1): # Iterates over the columns
    #         matrix[y_ite,x_ite] = divergence_naive(x_pos, y_pos, max_ite) # Calcualtes the iterations value for the current point
    #        x_pos = x_pos + x_val # Updates x position
    #        x_ite = x_ite + 1 # Updates column number
    #        
    #    y_pos = y_pos + y_val # Updates y position
    #    x_pos = -2.0 # Resets column number for inner loop
    #    x_ite = 0 # Resets column number
    #    y_ite = y_ite + 1 # Updates row number
       
    max_vecs = len(matrix[0])//size_cache
    remain = len(matrix[0])-size_cache*max_vecs
    
    while(y_ite <= len(matrix)-1):
        while(x_ite <= max_vecs):
            if(x_ite == max_vecs):
                #max_val = len(matrix[0]) # For matrix copy
                max_val = remain # For zeros matrix creation
                mat_ind = len(matrix[0])
            else:
                #max_val = (x_ite+1)*size_cache # For matrix copy
                max_val = size_cache # For zeros matrix creation
                mat_ind = (x_ite+1)*size_cache
            #pos_vector = matrix[[y_ite],(x_ite*180):max_val] # For matrix copy
            pos_vector = np.zeros((1,max_val),dtype=complex) # For zeros matrix creation
            
            matrix[y_ite,(x_ite*size_cache):mat_ind] = divergence_vector(pos_vector, x_pos, y_pos, x_val, max_ite)
            x_pos = x_pos + (x_val*max_val)
            x_ite = x_ite + 1
            
        y_pos = y_pos + y_val
        x_pos = -2.0
        x_ite = 0
        y_ite = y_ite + 1
            
            #divergence_vector(pos_vector)
            # KOMMET HERTIL !!!
        
        
        
    return matrix # Returns result


def divergence_vector(cord_vector, x_cord, y_cord, x_ini, ite): # Function to check wether a point is diverging or not

    vec_len = len(cord_vector[0])
    ite_var = 0
    x_tmp = x_cord
    while(ite_var <= vec_len-1):
        cord_vector[0,ite_var] = complex(x_tmp,y_cord)
        ite_var = ite_var + 1
        x_tmp = x_tmp + x_ini

    #print("1", cord_vector)
    div_list = list(range(0,vec_len))

    z_tmp_vec = np.zeros((1,vec_len),dtype=complex)
    z_len_vec = abs(cord_vector)
    cur_ite_vec = np.zeros((1,vec_len)) # Iteration counter vector
    
    # Virker 1
    # m = np.full((1,vec_len), True, dtype = bool)
    
    ite_var = 0
    #print("3", np.amin(z_len_vec))
    while(np.amin(z_len_vec) <= 2 and ite_var <= ite - 1):
        
        
        div_list = np.where((z_len_vec <= 2) == 1)
        
        cur_ite_vec[0,div_list] = cur_ite_vec[0,div_list] + 1
        
        
        z_len_vec[0,div_list] = np.abs(z_tmp_vec[0,div_list])
        z_tmp_vec[0,div_list] = ((z_tmp_vec[0,div_list])**2 + (cord_vector[0,div_list]))
            #z_tmp_vec = (z_tmp_vec**2 + cord_vector)
            #z_len_vec = abs(z_tmp_vec)
            #cur_ite_vec = cur_ite_vec + (z_len_vec <= 2)
        cur_ite_vec[0,div_list] = cur_ite_vec[0,div_list] + 1
        
        # Virker 1
        # z_tmp_vec[m] = ((z_tmp_vec[m])**2 + (cord_vector[m]))
        # div_list = np.less(z_len_vec, 2, out = np.full((1,vec_len), True), where = m)
        # cur_ite_vec[m] = cur_ite_vec[m] + 1
        # m[np.abs(z_tmp_vec) > 2] = False
        
        ite_var = ite_var + 1
    #print("1", cur_ite_vec)
    
    #while(z_len <= 2 and cur_ite <= ite - 1): # Loop that test the mandelbrot set conditions of divergens
    #    z_tmp = z_tmp**2 + xy_tmp # Update equation
    #    
    #    cur_ite = cur_ite + 1 # Update iteration counter
    #    z_len = abs(z_tmp) # Calculates length of new z value
    
    return cur_ite_vec # Returns the amount of iterations before convergense or max if reached




un_val = 600 # Matrix resolution
mandelbrot_naive(un_val,un_val,50) # Calling the function
