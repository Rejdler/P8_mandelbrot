# -*- coding: utf-8 -*-
"""
Created on Sun May 23 15:15:49 2021

@author: dksan
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 18:02:06 2021

@author: dksan
"""

#%% Importing important libraries
import time
import numpy as np
from dask.distributed import Client, wait
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
    ax1.set_title('Dask naive implementation, Time: ' + str(round(time_dif,3)) + " Res: " + str(im_res)) # Adds title, and time to calculate
    
    return 1


def matrix_calc(matrix, max_ite, workers):   #Takes the matrix and delegates the work

    partitions = len(matrix[0])//workers # Calculates how many whole parts the matrix can be split into
    x_res = len(matrix[0])
    y_res = len(matrix)
    

    work_list = []

    y_end = 0
    part_count = 0
    while(part_count <= workers-1):     # Calculates how to split the matrix between the workers
        y_start = part_count*partitions
        if part_count < workers-1:
            y_end = y_end + partitions
        else:
            y_end = len(matrix[0])
        tmp = [y_start,y_end,x_res,y_res,max_ite] # The start and end rows that a single workers should work on is put into the work list
        work_list.append(tmp)
        part_count = part_count + 1
      
    daskers = Client(n_workers = workers) # Creates the clients/workers
    mat_out = daskers.map(divergence_dask,work_list) # Distributes the inputs to them
    work = daskers.submit(matrix_stitcher,mat_out) # Sends the work to them, and stitches the output back together in a single matrix
    wait(work) # Wait for it to be finished
    
    result = work.result() # Collects the results

    daskers.close() # Closes workers
    return result

def matrix_stitcher(matrix_list): # Takes the output of the dask workers and put them together in a single matrix
    daskers = len(matrix_list)
    dim_list = [0]
    counter = 0
    while(counter <= daskers-1):
        dim_list.append(dim_list[counter]+len(matrix_list[counter]))
        counter += 1

    result = np.zeros((dim_list[-1],len(matrix_list[0][0])))
    counter = 0
    while(counter <= daskers - 1):
        result[(dim_list[counter]):(dim_list[counter+1]),:] = matrix_list[counter]
        counter += 1
    return result
    

def divergence_dask(list_of_work): # Function to check wether a point is diverging or not__        start_y, end_y, x_total, y_total, ite
    matrix = np.zeros(((list_of_work[1]-list_of_work[0]),list_of_work[2]))
    x_val = 3/(list_of_work[2]-1)    # Finds the columns of the matrix
    y_val = 3/(list_of_work[3]-1)    # Finds the rows of the matrix
    x_pos = -2.0    # Current x position
    #y_pos = -1.5    # Current y position
    y_pos = -1.5 + (list_of_work[0]*y_val)
    x_ite = 0       # Current column number
    y_ite = list_of_work[0]       # Current row number
    
    while(y_ite <= list_of_work[1] - 1):        # Iterates over the rows
        while(x_ite <= list_of_work[2] - 1): # Iterates over the columns
            matrix[(y_ite-list_of_work[0]),x_ite] = divergence_naive(x_pos, y_pos, list_of_work[4])   # Calcualtes the iterations value for the current point
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
    mandelbrot_naive(un_val,un_val,50, 2)  # Calling the function