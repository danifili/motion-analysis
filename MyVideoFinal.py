#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 12:31:12 2017

@author: danifili
"""
from MyVideoHelper2 import MyVideoHelper2
from MyVideoOpticalFlow import HornShunck
from TimonerFreeman import TimonerFreeman
from Plot import Plot

import numpy as np
from scipy.optimize import leastsq

class MyVideo(MyVideoHelper2):
    
    def __init__(self, images):
        MyVideoHelper2.__init__(self, images)
        
        #precompute matrices of gradients
        min_corner = (0, 0)
        max_corner = (self.width-1, self.height-1)
        
        image_shape = (self.width, self.height, self.duration-1)
        self.__Ex = np.zeros(image_shape)
        self.__Ey = np.zeros(image_shape)
        self.__Et = np.zeros(image_shape)

        print ("Computing gradients...")
        
        for t in range(self.duration-1):
            self.__Ex[:, :, t] = self._Ex(min_corner, max_corner, t)
            self.__Ey[:, :, t] = self._Ey(min_corner, max_corner, t)
            self.__Et[:, :, t] = self._Et(min_corner, max_corner, t)

        self.__video_displacement_ROI = TimonerFreeman(images, self.__Ex, self.__Ey, self.__Et)
        self.__video_optical_flow = HornShunck(images, self.__Ex, self.__Ey, self.__Et)
    
    def get_displacement_ROI(self, min_corner, max_corner, t, max_margin = 0):
        """
        Given the upper-left corner and lower-right corner of 
        a region of interest and a time t, it calculates the displacement 
        between time t and t+1 in such ROI
        
        params:
            -min_corner: A tuple of size 2 representing the (x, y) coordinates
             of the upper-left corner of our region of interest.
             It must satisfy:
                * 0 <= min_corner[0] < self.width
                * 0 <= min_corner[1] < self.height
                
            -max_corner: A tuple of size 2 representing the (x, y) coordinates 
            of the lower-right corner of our region of interest.
            It must satisfy:
                * min_corner[0] < max_corner[0] < self.width
                * min_corner[1] < max_corner[1] < self.height
            
            -t: it must satisfy 0 <= t < self.duration-1
            
            -max_margin: a positive integer; the maximum margin used to compute the displacement
            in the region of interest.
            
            -min_eigen_ratio: a positive float; the min acceptable ratio between the eigenvalues and the area
            of the region of interest of the matrix used to compute its displacement.

        
        returns:
            A 2-length numpy arrat representing the average displacement of region of interest from time 
            t to time (t+1). The first element represents the x direction and the y element represents
            the y direction
        """
        x_min, y_min = min_corner
        x_max, y_max = max_corner
        
        roi_width = x_max - x_min + 1
        roi_height = y_max - y_min + 1
        
        #find the best margin
        x_margin = min(x_min, self.width-1-x_max)
        y_margin = min(y_min, self.height-1-y_max)
        margin = min(max_margin, x_margin, y_margin)
        
        new_min_corner = (x_min - margin, y_min - margin)
        new_max_corner = (x_max + margin, y_max + margin)
        
        #get the minimum eigenvalue of the region of interest to determine if there is enough contrast
        l1, l2 = self.__video_displacement_ROI.get_eigenvalues(new_min_corner,new_max_corner, t)
        new_area = (2*margin + roi_width) * (2*margin + roi_height)
        
        return self.__video_displacement_ROI.get_displacement_ROI(new_min_corner, new_max_corner, t), l1 / new_area, l2 /new_area 
        
          
    def get_optical_flow_ROI(self, min_corner, max_corner, t, win_min = 5, win_max = 25, quality_level = 0.07, max_iterations = 100,  smoothness = 100):
        """
        Given the upper-left corner and lower-right corner of 
        a region of interest and a time t, it calculates the
        optical flow
        
        params:
            -min_corner: A tuple of size 2 representing the (x, y) coordinates
             of the upper-left corner of our region of interest.
             It must satisfy:
                * 0 < min_corner[0] < self.width
                * 0 < min_corner[1] < self.height
                
            -max_corner: A tuple of size 2 representing the (x, y) coordinates 
            of the lower-right corner of our region of interest.
            It must satisfy:
                * min_corner[0] < max_corner[0] < self.width
                * min_corner[1] < max_corner[1] < self.height
            
            -t: it must satisfy 0 <= t < self.duration-1 
            
            -win_min: The minimum size of the regions used to calculate the initial displacements of this algorithm
            
            -win_max: The maximum size of the regions used to calculate the initial displacements of this algorithm
            
            -max_iterations: maximum number of iterations that the algorithm can perform
            
            -smoothness: positive flow used to control the smoothness of the optical flow. The closer to 0 the smoother
            
            -min_eigen: a positive float; the min acceptable ratio between the eigenvalues and the area
            of the region of interest of the matrix used to compute its displacement.
        
        returns:
            a numpy array of floats of size W x H x 2, where W and H are the width and height of the
            region of interest, in which the values at (x, y, 0) and (x, y, 1) represent the x-component
            and y-component of the optical flow at pixel (x, y) relative to the ROI
        """
        print ("Computing displacements from frame " + str(t) + " to frame " + str(t+1))
        x_min, y_min = min_corner
        x_max, y_max = max_corner
        
        roi_width = x_max - x_min + 1
        roi_height = y_max - y_min + 1
        
        #initialize initial displacements
        displacements_mask = np.zeros((roi_width, roi_height, 2))
        
        #get maximum margin
        max_margin = (win_max - win_min) // 2
        
        points = []
        displacements = []
        plot_points = []
        mask = np.zeros((roi_width, roi_height, 2))

        #start = timer()
        for x in range(roi_width // win_min):
            for y in range(roi_height // win_min):
                new_min_corner = np.array([win_min * x + x_min, win_min * y + y_min])
                new_max_corner = np.array([win_min * (x+1) + x_min-1, win_min*(y+1)+y_min-1])
                
                x_min_rel, y_min_rel = new_min_corner - np.array([x_min, y_min])
                x_max_rel, y_max_rel = new_max_corner - np.array([x_min, y_min])
                
                
                #get average displacements 
                avg_displacement, l1, l2 = self.get_displacement_ROI(new_min_corner, new_max_corner, t,\
                                                            max_margin)

                #plot_points.append([(x_min_rel + x_max_rel)//2, (y_min_rel + y_max_rel)//2])
                #points.append([(x_min_rel + x_max_rel)//2, (y_min_rel + y_max_rel)//2])

                mask[(x_min_rel + x_max_rel)//2, (y_min_rel + y_max_rel)//2, :] = np.array([l1, l2])
                displacements_mask[(x_min_rel + x_max_rel)//2, (y_min_rel + y_max_rel)//2, :] = np.array(avg_displacement)

        mask[:, :, 0] = mask[:, :, 0]  / np.max(mask[:, :, 0])
        mask[:, :, 1] = mask[:, :, 1]  / np.max(mask[:, :, 1])

        for x in range((win_min-1)//2, roi_width, win_min):
            for y in range((win_min-1)//2, roi_height, win_min):

                if min(mask[x,y]) > quality_level:
                    displacements.append(displacements_mask[x,y])
                    points.append([x,y])
        
        #end = timer()
        #print (end-start)
  
#                displacements[x_min_rel : x_max_rel+1, y_min_rel : y_max_rel+1] = np.array(avg_displacement)
#        cut = roi_width*roi_height//(10 * win_min**2)
#        ones_matrix = np.ones((roi_width, roi_height))
#        top_x = np.partition(mask[:, :, 0].flatten(), cut)
#        top_y = np.partition(mask[:, :, 1].flatten(), cut)
#        ix = np.searchsorted(top_x, 0)-1
#        iy = np.searchsorted(top_y, 0)-1
#        thr0 = top_x[ix]
#        thr1 = top_y[iy]
#        
#        mask[:, :, 0] = np.minimum(mask[:, :, 0] / thr0, ones_matrix)
#        mask[:, :, 1] = np.minimum(mask[:, :, 1] / thr1, ones_matrix)
        
#        Plot.vector_heat_map(self, min_corner, max_corner, t, mask, np.pi/2, alpha=0.3)
#        Plot.vector_heat_map(self, min_corner, max_corner, t, mask, 0, alpha=0.3)
        # plot_points = np.array(points)
        # Plot.scatter_plot(self.images[t], min_corner, max_corner, plot_points[:, 0], plot_points[:, 1], color = "red")
        #print ("hi")
        #start = timer()
        displacements = self._interpolate(roi_width, roi_height, np.array(points), np.array(displacements))
        #end = timer()
        #print (end-start)
        #use algorithm for calculating optical flow given the average displacements as the initial ones
        return self.__video_optical_flow.get_optical_flow_ROI(min_corner, max_corner, t, \
                initial_displacements = displacements, max_iterations = max_iterations, smoothness = smoothness, input_mask=mask)
    
    def get_cumulative_displacements(self, min_corner, max_corner, win_min = 5, win_max = 25, quality_level = 0.07, max_iterations = 100,  smoothness = 100):
        x_min, y_min = min_corner
        x_max, y_max = max_corner
        
        roi_width = x_max - x_min + 1
        roi_height = y_max - y_min + 1

        directions = 2

        cumulative_displacements = np.zeros((self.duration, roi_width, roi_height, directions))

        for t in range(self.duration-1):
            new_displacements = self.get_optical_flow_ROI(min_corner, max_corner, t, win_min=win_min, win_max=win_max, quality_level=quality_level,\
                                                    max_iterations=max_iterations, smoothness=smoothness)

            cumulative_displacements[t+1] = cumulative_displacements[t] + new_displacements

        return cumulative_displacements

    def sinusoidal_fit(self, cumulative_displacements):
        """
        Given a region of interest that have sinusoidal movement with an expected frequency, it calculates the
        amplitude and phase for every pixel
        
        params:
            -min_corner: A tuple of size 2 representing the (x, y) coordinates
             of the upper-left corner of our region of interest.
             It must satisfy:
                * 0 <= min_corner[0] < self.width
                * 0 <= min_corner[1] < self.height
                
            -max_corner: A tuple of size 2 representing the (x, y) coordinates 
            of the lower-right corner of our region of interest.
            It must satisfy:
                * min_corner[0] < max_corner[0] < self.width
                * min_corner[1] < max_corner[1] < self.height
            
            -win_min: The minimum size of the regions used to calculate the initial displacements of this algorithm
            
            -win_max: The maximum size of the regions used to calculate the initial displacements of this algorithm
            
            -max_iterations: maximum number of iterations that the algorithm can perform
            
            -smoothness: positive flow used to control the smoothness of the optical flow. The closer to 0 the smoother
            
            -min_eigen: a positive float; the min acceptable ratio between the eigenvalues and the area
            of the region of interest of the matrix used to compute its displacement.
            
        returns: 
            a numpy array of size W x H x 4, where W and H are the width and height of the region of interest, in which
            the (x, y) entry is a numpy array of size 4 containing amplitude_x, amplitude_y, phase_x, phase_y
        """
        frames, width, height, directions = cumulative_displacements.shape
        
        data = np.zeros((width, height, 4))

        print ("Fitting data to a sinousoidal movement...")
        
        for x in range(width):
            for y in range(height):
                x_displacements = cumulative_displacements[:, x, y, 0]
                y_displacements = cumulative_displacements[:, x, y, 1]
                amp_x, phase_x = self._sinusoidal_fit_helper(x_displacements)
                amp_y, phase_y = self._sinusoidal_fit_helper(y_displacements)
                
                data[x, y, :] = np.array([amp_x, amp_y, phase_x, phase_y])
                
        return data
        
    def _sinusoidal_fit_helper(self, cumulative_displacements):
        """
        Given the displacements in a certain direction for a pixel that have sinusoidal movement with a given frequency,
        it calculates the amplitude and phase of the sine wave
        
        params:
            -displacements: a numpy array containing the displacements in a certain direction of two consecutive images
            
        returns:
            a tuple in which the first entry is the amplitude of the sine wave and the second entry is the phase of
            the sine wave
        """
        
        period = len(cumulative_displacements)
        
        guess_amp = 0
        guess_phase = 0
        
        frequency = 2 * np.pi / period
        
        t = np.array([frequency * t for t in range(period)])
        
        optimize_function = lambda x: x[0] * np.sin(t) + x[1] * (np.cos(t) - 1) - cumulative_displacements
        
        A, B = leastsq(optimize_function, [guess_amp, guess_phase])[0]
        
        amp = np.sqrt(A**2 +  B**2)
        phase = np.arctan2(B, A)
        
        if phase < 0:
            phase += 2*np.pi
        
        return amp, phase % (2 * np.pi)
        
                                                          
        
        
        