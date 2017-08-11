#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 12:04:33 2017

@author: danifili
"""

import sys
from MyVideoFinal import MyVideo
from MyImage import MyImage
from Plot import Plot
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


METAVAR_IMAGE = tuple("frame" + str(t) for t in range(8))


def generate_data(args):
    args["image"].sort()
    images = [MyImage(image) for image in args["image"]]
    video = MyVideo(images)
    

    min_corner = tuple(args["c"][:2])
    x_max, y_max = tuple(args["c"][2:])
    if x_max < 0:
        x_max += video.width-1
    if y_max < 0:
        y_max += video.height-1
    max_corner = (x_max, y_max)

    quality_level = args["q"]

    cumulative_displacements = video.get_cumulative_displacements(min_corner, max_corner, quality_level=quality_level)
    data = video.sinusoidal_fit(cumulative_displacements)
    
    args["data"] = data
    args["cumulative_displacements"] = cumulative_displacements
    args["video"] = video
    args["min_corner"] = min_corner
    args["max_corner"] = max_corner

def save_data(args):
    """
    Saves the amplitudes and phases in the x and y directions of a certain ROI.        
    """
    data = args["data"]
    root = args["root"]
    save_csv = args["x"]

    if not save_csv:
        return

    wildcards = ["amplitudes_x", "amplitudes_y", "phases_x", "phases_y"]

    for i in range(4):
        np.savetxt(root + wildcards[i] + ".csv", data[:, :, i], delimiter=",")

def plot_data(args):
    root = args["root"]
    video = args["video"]
    data = args["data"]
    min_corner = args["min_corner"]
    max_corner = args["max_corner"]
    thr_x, thr_y = args["t"]

    save_plots = args["s"]
    show_plots = args["p"]

    amplitudes_x = np.array(data[:, :, 0])
    amplitudes_y = np.array(data[:, :, 1])
    phases_x = np.array(data[:, :, 2])
    phases_y = np.array(data[:, :, 3])

    phases_x[amplitudes_x < thr_x] = float('nan')
    amplitudes_x[amplitudes_x < thr_x] = float('nan')

    phases_y[amplitudes_y < thr_y] = float('nan')
    amplitudes_y[amplitudes_y < thr_y] = float('nan')


    phases_x_fig = plt.figure(1)
    Plot.phase_heat_map(video, min_corner, max_corner, 0, phases_x, alpha=0.3)
    
    phases_y_fig = plt.figure(2)
    Plot.scalar_heat_map(video, min_corner, max_corner, 0, amplitudes_x, alpha=0.3)
    
    amplitudes_x_fig = plt.figure(3)
    Plot.phase_heat_map(video, min_corner, max_corner, 0, phases_y, alpha=0.3)
    
    amplitudes_y_fig = plt.figure(4)
    Plot.scalar_heat_map(video, min_corner, max_corner, 0, amplitudes_y, alpha=0.3)


    if save_plots:
        phases_x_fig.savefig(root + "phases_x" + ".png")
        phases_y_fig.savefig(root + "phases_y" + ".png")
        amplitudes_x_fig.savefig(root + "amplitudes_x" + ".png")
        amplitudes_y_fig.savefig(root + "amplitudes_y" + ".png")

    if show_plots:
        plt.show()

def motion_mag(args):
    factor = args["motionmag"]
    if factor is None:
        return

    cumulative_displacements = args["cumulative_displacements"]
    video = args["video"]
    root = args["root"]
    x_min, y_min = args["min_corner"]
    x_max, y_max = args["max_corner"]

    u = lambda x, y, t: factor * cumulative_displacements[t, x, y, 0]
    v = lambda x, y, t: factor * cumulative_displacements[t, x, y, 1]


    original_image = MyImage.image_from_matrix(video[x_min:x_max+1, y_min:y_max+1, 0], root + "motion_mag_factor" + str(factor) + "_" + str(0) + ".bmp")

    for t in range(1, video.duration):
        u_t = lambda x, y: u(x, y, t)
        v_t = lambda x, y: v(x, y, t)
        original_image.shift_image(u_t, v_t, root + "motion_mag_factor" + str(factor) + "_" + str(t) + ".bmp")

def motion_stop(args):
    if not args["motionstop"]:
        return

    cumulative_displacements = args["cumulative_displacements"]
    video = args["video"]
    root = args["root"]
    x_min, y_min = args["min_corner"]
    x_max, y_max = args["max_corner"]

    u = lambda x, y, t: -cumulative_displacements[t, x, y, 0]
    v = lambda x, y, t: -cumulative_displacements[t, x, y, 1]
    
    for t in range(video.duration):
        u_t = lambda x, y: u(x, y, t)
        v_t = lambda x, y: v(x, y, t)
        image = MyImage.image_from_matrix(video[x_min:x_max+1, y_min:y_max+1, t], root + "motion_stop" + "_" + str(t) + ".bmp")
        image.shift_image(u_t, v_t, root + "motion_stop" + "_" + str(t) + ".bmp")



HELP = {"image": "8 file paths. After sorting them, the i-th image will be consider as the i-th frame in the motion analysis",
        "root": "the root used to store all the files to be saved",
        "-x": "save the amplitudes and phases in x and y into 4 different csv files",
        "-s": "disable option of saving plots of amplitudes and phases",
        "-p": "disable option of showing plots of amplitudes and phases",
        "-t": "only display amplitudes and phases in x and y of pixels with amplitudes in x and y greater than thr_x and thr_y",
        "-c": "specify a region of interest, where x_min and y_min represent the top-left corner of the ROI and x_max and y_max " + \
              "represent the bottom right corner of the ROI. For x_max and y_max, negative integers are allowed and the value " + \
              "resulting from substracting from the width-1 and the height-1 of the image will be used",
        "-q": "quality level. It is a float between 0 and 1",
        "--motionmag": "store 8 images resulting from the motion magnification of the original 8 frames. The input mag_factor determines" + \
                       "the factor by which the displacements will be multiplied",
        "--motionstop": "shift all the images by negating the displacements computed. If the algorithm is accurate, " + \
                        "all the images should be similar to the first frame"
        }

if __name__ == "__main__":
    flags = [{"image" : dict(action="store", nargs=8, help=HELP["image"], metavar="frame")},
             {"root": dict(action="store", help=HELP["root"])},
             {"-x": dict(action="store_true", help=HELP["-x"])},
             {"-s": dict(action="store_false", help=HELP["-s"])},
             {"-p": dict(action="store_false", help=HELP["-p"])},
             {"-t": dict(action="store", help=HELP["-t"], nargs=2, metavar=("thr_x", "thr_y"), default=(0.0, 0.0), type=float)},
             {"-c": dict(action="store", help=HELP["-c"], nargs=4, metavar=("x_min", "y_min", "x_max", "y_max"), type=int, default=(0,0,0,0))},
             {"-q": dict(action="store", help=HELP["-q"], metavar="quality-level", type=float, default=0.07)},
             {"--motionmag": dict(action="store", help=HELP["--motionmag"], type=float, metavar="factor")},
             {"--motionstop": dict(action="store_true", help=HELP["--motionstop"])}]

    functions = [generate_data, save_data, motion_mag, motion_stop, plot_data]

    parser = argparse.ArgumentParser(description="TODO")

    for flag in flags:
        key = list(flag.keys())[0]
        parser.add_argument(key, **flag[key])

    args = vars(parser.parse_args(sys.argv[1:]))

    for f in functions:
        f(args)




