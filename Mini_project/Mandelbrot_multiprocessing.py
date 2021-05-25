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

def mandelbrot_naive(im_res, re_res, ite, threads):  # Inputs are resolution of the real and imaginary axis and the amount of iterations to check
    time_a = time.perf_counter()            # Start time
    
    mandelset = np.zeros((re_res, im_res))  # Creates a matrix of zeroes of appropriate size
    
    mandelset = matrix_calc(mandelset, ite, threads) # Sends the matrix to the function responsible indexing and calculating values
    
    time_b = time.perf_counter()            # End time
    time_dif = time_b-time_a                # Time differences
    
    fig1 = plt.figure(figsize = (15,15))    # Crewates a figure of certain size
    ax1 = fig1.add_subplot(1,1,1)           # Adds a supplot where an image can be added to
    ax1.imshow(mandelset,'hot',interpolation = 'none' ,extent=[-2.0,1.0,-1.5,1.5]) # Loads the matrix, set the color gradient, turns off interpolation and sets axis values
    ax1.set_title('Multiproc naive implementation, Time: ' + str(round(time_dif,3)) + " Res: " + str(im_res)) # Adds title, and time to calculate
    
    return 1


def matrix_calc(matrix, max_ite, threads):   #Takes the matrix and delegates the work

    partitions = len(matrix[0])//threads # Calculates how many whole parts the matrix can be split into
    x_res = len(matrix[0])
    y_res = len(matrix)
    

    work_list = []

    y_end = 0
    part_count = 0
    while(part_count <= threads-1):     # Calculates how to split the matrix between the threads/processes
        y_start = part_count*partitions
        if part_count < threads-1:
            y_end = y_end + partitions
        else:
            y_end = len(matrix[0])
        tmp = [y_start,y_end,x_res,y_res,max_ite] # The start and end rows that a single thread/processor should work on is put into the work list
        work_list.append(tmp)
        part_count = part_count + 1
    
    with mp.Pool(processes = threads) as pool:
        k = pool.starmap(divergence_multi,work_list) # Creates multiple threads/processors that each takes care of some part of the set
        # The matrix gets returned in multiple parts and have to be stitched back together
    result = np.zeros((y_res,x_res))
    count_var = 0
    while(count_var <= threads-1):

        result[(work_list[count_var][0]):(work_list[count_var][1]),:] = k[count_var] #Don't mind the spaghetti indexing happening here, actually puts the matrix back together

        count_var = count_var + 1

    return result



def divergence_multi(start_y, end_y, x_total, y_total, ite): # Function to check wether a point is diverging or not
    matrix = np.zeros(((end_y-start_y),x_total))
    x_val = 3/(x_total-1)    # Finds the columns of the matrix
    y_val = 3/(y_total-1)    # Finds the rows of the matrix
    x_pos = -2.0    # Current x position
    y_pos = -1.5 + (start_y*y_val)
    x_ite = 0       # Current column number
    y_ite = start_y       # Current row number
    
    while(y_ite <= end_y - 1):        # Iterates over the rows
        while(x_ite <= x_total - 1): # Iterates over the columns
            matrix[(y_ite-start_y),x_ite] = divergence_naive(x_pos, y_pos, ite)   # Calcualtes the iterations value for the current point
            x_pos = x_pos + x_val           # Updates x position
            x_ite = x_ite + 1               # Updates column number
            
        y_pos = y_pos + y_val   # Updates y position
        x_pos = -2.0            # Resets column number for inner loop
        x_ite = 0               # Resets column number
        y_ite = y_ite + 1       # Updates row number
    return matrix

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


if __name__ == '__main__':
    un_val = 100              # Matrix resolution
    mandelbrot_naive(un_val,un_val,50, 7)  # Calling the function
