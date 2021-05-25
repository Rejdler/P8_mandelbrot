# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 18:02:06 2021

@author: dksan
"""

#%% Importing important libraries
import time
import numpy as np
import matplotlib.pyplot as plt
import pyopencl as cl

#%% Naive implementatin of the mandelbrot set

def mandelbrot_naive(im_res, re_res, ite):  # Inputs are resolution of the real and imaginary axis and the amount of iterations to check
    time_a = time.perf_counter()            # Start time
    
    mandelset = np.zeros((re_res, im_res), np.uint)  # Creates a matrix of zeroes of appropriate size
    
    mandelset = matrix_calc(mandelset, ite) # Sends the matrix to the function responsible indexing and calculating values
    
    #print(mandelset)
    
    time_b = time.perf_counter()            # End time
    time_dif = time_b-time_a                # Time differences
    
    fig1 = plt.figure(figsize = (15,15))    # Crewates a figure of certain size
    ax1 = fig1.add_subplot(1,1,1)           # Adds a supplot where an image can be added to
    ax1.imshow(mandelset,'hot',interpolation = 'none' ,extent=[-2.0,1.0,-1.5,1.5]) # Loads the matrix, set the color gradient, turns off interpolation and sets axis values
    ax1.set_title('GPU implementation, Time: ' + str(round(time_dif,3)) + " Res: " + str(im_res)) # Adds title, and time to calculate
    
    return 1


def matrix_calc(matrix, max_ite):   #Takes the matrix and delegates the work for the gpu
    
    mat_columns = np.uint(len(matrix[0]))

    platforms = cl.get_platforms() # Gets info that's necessary for opencl
    
    devices = platforms[0].get_devices(cl.device_type.GPU)[0] #Simply picks the first GPU it finds
    
    gpu_context = cl.Context([devices]) #Create context
    
    gpu_program = cl.Program(gpu_context, """
        #include <pyopencl-complex.h>
        __kernel void mandel_gpu(__global int *result, __global uint *max_ite) //The kernel itself dosent take an input other than max iterations
        {
          __private int gid1 = get_global_id(0); // get it's own position
          __private int gid2 = get_global_id(1);
          __private int gs1 = get_global_size(0); // Finds the total size of the matrix
          __private int gs2 = get_global_size(1);
          __private int counter = 0;
          __private float z_len = 0;
            
          __private cfloat_t start_complex = cfloat_new(((float)3/(gs1-1) * gid1)-2,((float)3/(gs2-1) * gid2)-1.5); // Calculates the points complex value from it's position in the global grid
          __private cfloat_t cur_cord = cfloat_new(0,0);
            
          while(cfloat_abs(cur_cord) <= 2 && counter <= max_ite[0]) // Calculates the iterations to diverge
          {
              cur_cord = cfloat_add(cfloat_mul(cur_cord,cur_cord),start_complex);
              
              z_len = cfloat_abs(cur_cord);
              
              counter = counter + 1;
              }
            
          result[gid1 + gs1 * gid2] = counter; // Returns the result to the global grid
        }
        """).build()

    gpu_queue = cl.CommandQueue(gpu_context) # Creates a queue for the context
    
    mf = cl.mem_flags 
    gpu_ite = cl.Buffer(gpu_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.array(max_ite, dtype=np.uint)) # Creates space in gpu memory for max iterations
    gpu_out_matrix = cl.Buffer(gpu_context, mf.WRITE_ONLY, matrix.nbytes) # Creates space in gpu memory for the matrix containing results
    
    gpu_program.mandel_gpu(gpu_queue, matrix.shape, None, gpu_out_matrix, gpu_ite) # Execute the kernel
    
    cl.enqueue_copy(gpu_queue,matrix,gpu_out_matrix)

    return matrix        # Returns result



un_val = 100                  # Matrix resolution
mandelbrot_naive(un_val,un_val,50)  # Calling the function, inpits are, imaginary resoltuion, real reasolution, max iterations, and openCL device by number