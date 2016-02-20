import cv2
import numpy as np
import dicom
import json
import os
import random
import re
import shutil
import sys
from matplotlib import image
from scipy.ndimage import label
from scipy.ndimage.morphology import binary_erosion
from scipy.fftpack import fftn, ifftn
from scipy.signal import argrelmin, correlate
from scipy.spatial.distance import euclidean
from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

from parameters import *


#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def log(index, msg, lvl):
    string = ""
    for i in range(lvl):
        string += " "
    string += msg
    print str(index) + string


#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def save_image_contours(name, img, contours):
    
    colorimg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    colorimg = colorimg.astype(np.uint8)

    color = (0, 255, 0)
    

    
    image.imsave(name, colorimg)
    
    #
    cv2.drawContours(colorimg, contours, -1, color, -1)        

    image.imsave("cont_" + name, colorimg)
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def get_image_gray(img):
    
    uint16 = False
    
    if uint16 is False:
        
        return img.astype(np.uint8)
        
    img8 = cv2.convertScaleAbs(img.astype(np.uint16))
    
    return img8

    
#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def save_image_gray(name, img):

    colorimg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    image.imsave(name, colorimg)

    return

        
#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def save_image(name, img, mask, roi, line, point, meta):
    

    #------------------------------------------------------------------------------------------------------------------------------------
    # color image
    #------------------------------------------------------------------------------------------------------------------------------------
    
    colorimg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    colorimg = colorimg.astype(np.uint8)
    
    
    #------------------------------------------------------------------------------------------------------------------------------------
    # outline of mask
    #------------------------------------------------------------------------------------------------------------------------------------
        
    eroded = binary_erosion(mask)
    hollow_mask = np.where(eroded, 0, mask)

    #print "mask shape:        " + str(mask.shape)                
    #print "hollow mask shape: " + str(hollow_mask.shape)
    
    colorimg[hollow_mask != 0] = [255, 0, 255]

    
    #------------------------------------------------------------------------------------------------------------------------------------
    # outline of ROI
    #------------------------------------------------------------------------------------------------------------------------------------
        
    #print all_rois.shape
    #roi = all_rois[s]
    eroded = binary_erosion(roi)
    hollow_mask = np.where(eroded, 0, roi)
    colorimg[hollow_mask != 0] = [0, 255, 255]
    

    #------------------------------------------------------------------------------------------------------------------------------------
    # Patient Orientation
    #------------------------------------------------------------------------------------------------------------------------------------    
           

    scale = 100
    
    xVec = np.multiply(meta.patCoords[0], scale)
    yVec = np.multiply(meta.patCoords[1], scale)
    zVec = np.multiply(meta.patCoords[2], scale)

#    print xVec
#    print yVec
#    print zVec
#    print ""
    
#    hollow_mask = roi*0        
#    cv2.line(hollow_mask, (0, 10), (100, 10), (255, 255, 255) )
#    colorimg[hollow_mask != 0] = [255, 255, 255]            
#
#    hollow_mask = roi*0    
#    cv2.line(hollow_mask, (10, 0), (10, 100), (255, 255, 255) )
#    colorimg[hollow_mask != 0] = [0, 0, 255]  
    

    x0 = hollow_mask.shape[1]/2
    y0 = hollow_mask.shape[0]/2
    
#    print x0
#    print y0
#    print ""
    
    #
    hollow_mask = roi*0    
    cv2.line(hollow_mask, (x0, y0), ( int(xVec[0]), int(xVec[1])), (255, 255, 255) )
    colorimg[hollow_mask != 0] = [255, 255, 255]    
     
    #
    hollow_mask = roi*0    
    cv2.line(hollow_mask, (x0, y0), ( int(yVec[0]), int(yVec[1])), (255, 255, 255) )
    colorimg[hollow_mask != 0] = [0, 0, 255]    
     
    # 
    hollow_mask = roi*0    
    cv2.line(hollow_mask, (x0, y0), ( int(zVec[0]), int(zVec[1])), (255, 255, 255) )
    colorimg[hollow_mask != 0] = [0, 255, 0]    
    
    
#    line
#    #print all_lines.shape
#    #ine_t = all_lines[t]
#    
#    #print type(line_t)
#    #print len(line_t)
#    
#    #line_t_s = line[s][
#    
#    #print "t=" + str(t) + ", s=" + str(s)
#    for i in range(len(line)):
#        xy = line[i]
#        x = xy[0]
#        y = xy[1]
#        
#        #print str(x) + ", " + str(y)
#        hollow_mask[y][x] = 1
#            
#    #
#    colorimg[hollow_mask != 0] = [255, 0, 0]            
                    
                    
    #------------------------------------------------------------------------------------------------------------------------------------
    # outline of line 
    #------------------------------------------------------------------------------------------------------------------------------------       
        
    hollow_mask = roi*0                
    
    #print all_lines.shape
    #ine_t = all_lines[t]
    
    #print type(line_t)
    #print len(line_t)
    
    #line_t_s = line[s][2] 
    
    #print type(line_t_s)
    #print len(line_t_s)
    
    #print line_t_s              
    #print line_t_s[0]                
    #print line_t_s[1]
    
    #print "t=" + str(t) + ", s=" + str(s)
    for i in range(len(line)):
        xy = line[i]
        x = xy[0]
        y = xy[1]
        
        #print str(x) + ", " + str(y)
        hollow_mask[y][x] = 1
            
    #
    colorimg[hollow_mask != 0] = [255, 0, 0]                


    #------------------------------------------------------------------------------------------------------------------------------------
    # threshold point
    #------------------------------------------------------------------------------------------------------------------------------------
    
    hollow_mask = roi*0             

    #print "HERE"
    #print all_coords.shape
    
    #coords_t = all_coords[t]
    
    #print type(coords_t)
    #print len(coords_t)

    
    #coord_t_s = point[s]

    #print type(coord_t_s)
    #print len(coord_t_s)
            
    #print coord_t_s
    #print coord_t_s[0]
    #print coord_t_s[1]
    
    #exit(1)
    
    for i in range(len(point)):
        xy = point[i]
        x = xy[0]
        y = xy[1]                

        hollow_mask[y][x] = 1
        
    #print coords.shape
    
    #eroded = binary_erosion(coords)
    #hollow_mask = np.where(eroded, 0, coords)
    colorimg[hollow_mask != 0] = [0, 255, 0]
    
    #print "---"
    
    
    #------------------------------------------------------------------------------------------------------------------------------------
    # save image
    #------------------------------------------------------------------------------------------------------------------------------------
        
    #loc = np.abs(dataset.metas[s][t].SliceLocation)
    
    image.imsave(name, colorimg)
                
                
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def save_masks_to_dir(dataset, all_masks, all_rois, all_circles, all_lines, all_coords, closest, metas):
    

    try:
        
        os.mkdir("output/%s" % dataset.name)


        #-----------------------------------------------------------------------------------------------------------------------------------------------------
        #-----------------------------------------------------------------------------------------------------------------------------------------------------
            
        for t in range(len(dataset.time)):
            
            os.mkdir("output/%s/time%02d" % (dataset.name, t))

            #-----------------------------------------------------------------------------------------------------------------------------------------------------
            #-----------------------------------------------------------------------------------------------------------------------------------------------------
                        
            for s in range(len(dataset.slices)):

                meta = metas[s][t]
    
                #------------------------------------------------------------------------------------------------------------------------------------
                #------------------------------------------------------------------------------------------------------------------------------------

                loc = np.abs(dataset.metas[s][t].SliceLocation)
                
                name = "output/%s/time%02d/slice%02d_%04f_color.png" % (dataset.name, t, s, loc)
                
                save_image(name, dataset.images[s][t], all_masks[t][s], all_rois[s], all_lines[t][s][2], all_coords[t][s], meta)


                #------------------------------------------------------------------------------------------------------------------------------------
                #------------------------------------------------------------------------------------------------------------------------------------
                
                if (s == closest):
                
                    name = "middle_slices/case%04d_time%02d_slice%02d_%04f_color.png" % (int(dataset.name), t, s, loc)
                
                    save_image(name, dataset.images[s][t], all_masks[t][s], all_rois[s], all_lines[t][s][2], all_coords[t][s], meta)


    except Exception, e:
        
        print "Exception: " + str(e)
    
    return
                         
                        
#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def calc_rois(images, index):
    (num_slices, _, _, _) = images.shape
    log(index, "Calculating mean...", 2)
    dc = np.mean(images, 1)

    def get_H1(i):
        log(index, "Fourier transforming on slice %d..." % i, 3)
        ff = fftn(images[i])
        first_harmonic = ff[1, :, :]
        log(index, "Inverse Fourier transforming on slice %d..." % i, 3)
        result = np.absolute(ifftn(first_harmonic))
        log(index, "Performing Gaussian blur on slice %d..." % i, 3)
        result = cv2.GaussianBlur(result, (5, 5), 0)
        return result

    log(index, "Performing Fourier transforms...", 2)
    h1s = np.array([get_H1(i) for i in range(num_slices)])
    m = np.max(h1s) * 0.05
    h1s[h1s < m] = 0

    log(index, "Applying regression filter...", 2)
    regress_h1s = regression_filter(h1s, index)
    
    log(index, "Post-processing filtered images...", 2)
    proc_regress_h1s, coords = post_process_regression(regress_h1s, index)
    
    log(index, "Determining ROIs...", 2)
    rois, circles = get_ROIs(dc, proc_regress_h1s, coords, index)
    
    return rois, circles


#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def calc_all_areas(images, rois, circles, index):
    
    closest_slice = get_closest_slice(rois)
    (_, times, _, _) = images.shape


    #------------------------------------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------

    def calc_areas(time):

        log(index, "Calculating areas at time %d..." % time, 2)

        mask, mean, line, coord = locate_lv_blood_pool(images, rois, circles, closest_slice, time)
        masks, areas = propagate_segments(images, rois, mask, mean, closest_slice, time)

        lines = dict()
        coords = dict()
        
        for key in masks:
            lines[key] = line #masks[key] #0 #line
            coords[key] = coord #masks[key] #0 #line

        return (masks, areas, lines, coords)


    #------------------------------------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------

    result = map(calc_areas, range(times))

    result = np.transpose(result)

    all_masks = result[0]
    all_areas = result[1]
    all_lines = result[2]
    all_coords = result[3]


    return all_masks, all_areas, all_lines, all_coords, closest_slice


#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def calc_total_volume(areas, area_multiplier, dist):
    slices = np.array(sorted(areas.keys()))
    modified = [areas[i] * area_multiplier for i in slices]
    vol = 0
    for i in slices[:-1]:
        a, b = modified[i], modified[i+1]
        subvol = (dist/3.0) * (a + np.sqrt(a*b) + b)
        vol += subvol / 1000.0  # conversion to mL
        
    return vol
    
    
    
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def get_centroid(img):
    nz = np.nonzero(img)
    pxls = np.transpose(nz)
    weights = img[nz]
    avg = np.average(pxls, axis=0, weights=weights)
    return avg


#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def regress_centroids(cs):
    num_slices = len(cs)
    y_centroids = cs[:, 0]
    x_centroids = cs[:, 1]
    z_values = np.array(range(num_slices))

    (xslope, xintercept, _, _, _) = linregress(z_values, x_centroids)
    (yslope, yintercept, _, _, _) = linregress(z_values, y_centroids)

    return (xslope, xintercept, yslope, yintercept)


#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def get_weighted_distances(imgs, coords, xs, xi, ys, yi):
    a = np.array([0, yi, xi])
    n = np.array([1, ys, xs])

    zeros = np.zeros(3)

    def dist(p):
        to_line = (a - p) - (np.dot((a - p), n) * n)
        d = euclidean(zeros, to_line)
        return d

    def weight(p):
        (z, y, x) = p
        return imgs[z, y, x]

    dists = np.array([dist(c) for c in coords])
    weights = np.array([weight(c) for c in coords])
    
    return (coords, dists, weights)


#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def gaussian_fit(dists, weights):
    
    # based on http://stackoverflow.com/questions/11507028/fit-a-gaussian-function
    (x, y) = histogram_transform(dists, weights)
    fivep = int(len(x) * 0.05)
    xtmp = x
    ytmp = y
    fromFront = False
    while True:
        if len(xtmp) == 0 and len(ytmp) == 0:
            if fromFront:
                # well we failed
                idx = np.argmax(y)
                xmax = x[idx]
                p0 = [max(y), xmax, xmax]
                (A, mu, sigma) = p0
                return mu, sigma, lambda x: gauss(x, A, mu, sigma)
            else:
                fromFront = True
                xtmp = x
                ytmp = y

        idx = np.argmax(ytmp)
        xmax = xtmp[idx]

        def gauss(x, *p):
            A, mu, sigma = p
            return A*np.exp(-(x-mu)**2/(2.*sigma**2))

        p0 = [max(ytmp), xmax, xmax]
        try:
            coeff, var_matrix = curve_fit(gauss, xtmp, ytmp, p0=p0)
            (A, mu, sigma) = coeff
            return (mu, sigma, lambda x: gauss(x, A, mu, sigma))
        except RuntimeError:
            if fromFront:
                xtmp = xtmp[fivep:]
                ytmp = ytmp[fivep:]
            else:
                xtmp = xtmp[:-fivep]
                ytmp = ytmp[:-fivep]

    return


#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def histogram_transform(values, weights):
    
    hist, bins = np.histogram(values, bins=NUM_BINS, weights=weights)
    bin_width = bins[1] - bins[0]
    bin_centers = bins[:-1] + (bin_width / 2)

    return (bin_centers, hist)


#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def get_outliers(coords, dists, weights):
    
    fivep = int(len(weights) * 0.05)
    ctr = 1
    while True:
        (mean, std, fn) = gaussian_fit(dists, weights)
        low_values = dists < (mean - STD_MULTIPLIER*np.abs(std))
        high_values = dists > (mean + STD_MULTIPLIER*np.abs(std))
        outliers = np.logical_or(low_values, high_values)
        if len(coords[outliers]) == len(coords):
            weights[-fivep*ctr:] = 0
            ctr += 1
        else:
            return coords[outliers]
    

#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def regress_and_filter_distant(imgs):
    
    centroids = np.array([get_centroid(img) for img in imgs])
    raw_coords = np.transpose(np.nonzero(imgs))
    (xslope, xintercept, yslope, yintercept) = regress_centroids(centroids)
    (coords, dists, weights) = get_weighted_distances(imgs, raw_coords, xslope,
                                                      xintercept, yslope,
                                                      yintercept)
    outliers = get_outliers(coords, dists, weights)
    imgs_cpy = np.copy(imgs)
    for c in outliers:
        (z, x, y) = c
        imgs_cpy[z, x, y] = 0
        
    return imgs_cpy


#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def regression_filter(imgs, index):
    
    condition = True
    iternum = 0
    
    while(condition):
        log(index, "Beginning iteration %d of regression..." % iternum, 3)
        iternum += 1
        imgs_filtered = regress_and_filter_distant(imgs)
        c1 = get_centroid(imgs)
        c2 = get_centroid(imgs_filtered)
        dc = np.linalg.norm(c1 - c2)
        imgs = imgs_filtered
        condition = (dc > 1.0)  # because python has no do-while loops
        
    return imgs


#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def post_process_regression(imgs, index):
    
    (numimgs, _, _) = imgs.shape
    centroids = np.array([get_centroid(img) for img in imgs])
    log(index, "Performing final centroid regression...", 3)
    (xslope, xintercept, yslope, yintercept) = regress_centroids(centroids)
    imgs_cpy = np.copy(imgs)

    def filter_one_img(zlvl):
        points_on_zlvl = np.transpose(imgs[zlvl].nonzero())
        points_on_zlvl = np.insert(points_on_zlvl, 0, zlvl, axis=1)
        (coords, dists, weights) = get_weighted_distances(imgs, points_on_zlvl,
                                                          xslope, xintercept,
                                                          yslope, yintercept)
        outliers = get_outliers(coords, dists, weights)
        for c in outliers:
            (z, x, y) = c
            imgs_cpy[z, x, y] = 0

    log(index, "Final image filtering...", 3)
    for z in range(numimgs):
        log(index, "Filtering image %d of %d..." % (z+1, numimgs), 4)
        filter_one_img(z)

    return (imgs_cpy, (xslope, xintercept, yslope, yintercept))


#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def floats_draw_circle(img, center, r, color, thickness):
    
    (x, y) = center
    x, y = int(np.round(x)), int(np.round(y))
    r = int(np.round(r))
    cv2.circle(img, center=(x, y), radius=r, color=color, thickness=thickness)
    
    return


#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def filled_ratio_of_circle(img, center, r):
    
    mask = np.zeros_like(img)
    floats_draw_circle(mask, center, r, 1, -1)
    masked = mask * img
    (x, _) = np.nonzero(mask)
    (x2, _) = np.nonzero(masked)
    if x.size == 0:
        return 0
        
    return float(x2.size) / x.size


#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def circle_smart_radius(img, center):
    domain = np.arange(1, 100)
    (xintercept, yintercept) = center

    def ratio(r):
        return filled_ratio_of_circle(img, (xintercept, yintercept), r)*r

    y = np.array([ratio(d) for d in domain])
    most = np.argmax(y)
    return domain[most]


#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def get_ROIs(originals, h1s, regression_params, index):
    
    (xslope, xintercept, yslope, yintercept) = regression_params
    (num_slices, _, _) = h1s.shape
    results = []
    circles = []
    for i in range(num_slices):
        log(index, "Getting ROI in slice %d..." % i, 3)
        o = originals[i]
        h = h1s[i]
        ctr = (xintercept + xslope * i, yintercept + yslope * i)
        r = circle_smart_radius(h, ctr)
        tmp = np.zeros_like(o)
        floats_draw_circle(tmp, ctr, r, 1, -1)
        results.append(tmp * o)
        circles.append((ctr, r))

    return (np.array(results), np.array(circles))


#-----------------------------------------------------------------------------------------------------------------------------------------------------
# using some pseudocode from
# https://en.wikipedia.org/wiki/Xiaolin_Wu%27s_line_algorithm
# and also https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def bresenham(x0, x1, y0, y1, fn):
    

    
    steep = abs(y1-y0) > abs(x1-x0)
    if steep:
        x0, x1, y0, y1 = y0, y1, x0, x1
    if x0 > x1:
        x0, x1, y0, y1 = x1, x0, y1, y0

    def plot(x, y):
        if steep:
            fn(y, x)
        else:
            fn(x, y)

    dx = x1 - x0
    dy = y1 - y0

    D = 2*np.abs(dy) - dx
    plot(x0, y0)
    y = y0

    for x in range(x0+1, x1+1):  # x0+1 to x1
        D = D + 2*np.abs(dy)
        if D > 0:
            y += np.sign(dy)
            D -= 2*dx
        plot(x, y)

#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def line_thru(bounds, center, theta):
    
    (xmin, xmax, ymin, ymax) = bounds
    (cx, cy) = center

    if np.cos(theta) == 0:
        return (cx, ymin, cx, ymax)
    slope = np.tan(theta)

    x0 = xmin
    y0 = cy - (cx - xmin) * slope
    if y0 < ymin:
        y0 = ymin
        x0 = max(xmin, cx - ((cy - ymin) / slope))
    elif y0 > ymax:
        y0 = ymax
        x0 = max(xmin, cx - ((cy - ymax) / slope))

    x1 = xmax
    y1 = cy + (xmax - cx) * slope
    if y1 < ymin:
        y1 = ymin
        x1 = min(xmax, cx + ((ymin - cy) / slope))
    elif y1 > ymax:
        y1 = ymax
        x1 = min(xmax, cx + ((ymax - cy) / slope))

    return (x0, x1, y0, y1)


#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def get_line_coords(w, h, cx, cy, theta):
    
    coords = np.floor(np.array(line_thru((0, w-1, 0, h-1), (cx, cy), theta)))
    
    return coords.astype(np.int)


#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def trim_zeros_indices(has_zeros):
    first = 0
    for i in has_zeros:
        if i == 0:
            first += 1
        else:
            break

    last = len(has_zeros)
    for i in has_zeros[::-1]:
        if i == 0:
            last -= 1
        else:
            break

    return first, last


#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def get_line(roi, cx, cy, theta):
    
    (h, w) = roi.shape
    (x0, x1, y0, y1) = get_line_coords(w, h, cx, cy, theta)

    intensities = []
    coords = []

    def collect(x, y):
        if y < 0 or y >= h or x < 0 or x >= w:
            return
        intensities.append(roi[y, x])
        coords.append((x, y))

    bresenham(x0, x1, y0, y1, collect)

    def geti(idx):
        return intensities[idx]

    getiv = np.vectorize(geti)
    x = np.arange(0, len(intensities))
    y = getiv(x)
    first, last = trim_zeros_indices(y)
    trimy = y[first:last]
    trimcoords = coords[first:last]

    trimx = np.arange(0, trimy.size)
    
    return (trimx, trimy, trimcoords)


#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def find_best_angle(roi, circ):
    
    ((cx, cy), r) = circ
    results = np.zeros(ANGLE_SLICES)
    fns = [None for i in range(ANGLE_SLICES)]

    COS_MATCHED_FILTER_FREQ = 2.5

    def score_matched(trimx, trimy):
        # first, normalize this data
        newtrimx = np.linspace(0.0, 1.0, np.size(trimx))
        minimum = np.min(trimy)
        maximum = np.max(trimy) - minimum
        newtrimy = (trimy - minimum) / maximum

        filt = 1 - ((np.cos(COS_MATCHED_FILTER_FREQ*2*np.pi*newtrimx)) /
                    2 + (0.5))
        cr = correlate(newtrimy, filt, mode="same")
        return np.max(cr)

    for i in range(ANGLE_SLICES):
        trimx, trimy, trimcoords = get_line(roi, cx, cy, np.pi*i/ANGLE_SLICES)
        score2 = score_matched(trimx, trimy)
        results[i] = score2
        fns[i] = (UnivariateSpline(trimx, trimy), trimx, trimcoords)

    best = np.argmax(results)
    
    return (best * np.pi / ANGLE_SLICES, fns[best])


#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def find_threshold_point(best, best_fn):
    
    fn, trimx, trim_coords = best_fn
    dom = np.linspace(np.min(trimx), np.max(trimx), 1000)
    f = fn(dom)
    mins = argrelmin(f)

    closest_min = -1
    closest_dist = -1
    for m in np.nditer(mins):
        dist = np.abs(500 - m)
        if closest_min == -1 or closest_dist > dist:
            closest_min = m
            closest_dist = dist

    fnprime = fn.derivative()
    restrict = dom[np.max(closest_min-THRESHOLD_AREA, 0):
                   closest_min+THRESHOLD_AREA]
    f2 = fnprime(restrict)

    m1 = restrict[np.argmax(f2)]
    mean = fn(m1)

    idx = np.min([int(np.floor(m1))+1, len(trim_coords)-1])
    
    return (mean, trim_coords, idx)

#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def get_closest_slice(rois):
    
    ctrd = get_centroid(rois)
    closest_slice = int(np.round(ctrd[0]))
    
    return closest_slice


#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def locate_lv_component(img_bin, coords, idx):
    
    #------------------------------------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------
        
    labeled, num = label(img_bin)
    x, y = coords[idx]

    count = 0
    #------------------------------------------------------------------------------------------------------------------------------------------------
    # Look along the line for a component. If one isn't found within a certain
    # number of indices, just spit out the original coordinate.
    #------------------------------------------------------------------------------------------------------------------------------------------------        

    while labeled[y][x] == 0:
        idx += 1
        count += 1
        x, y = coords[idx]
        if count > COMPONENT_INDEX_TOLERANCE:
            idx -= count
            x, y = coords[idx]
            break

    if count <= COMPONENT_INDEX_TOLERANCE:
        component = np.transpose(np.nonzero(labeled == labeled[y][x]))
    else:
        component = np.array([[y, x]])


    return labeled, component, count


#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def locate_lv_blood_pool(images, rois, circles, closest_slice, time):
    
    #print "locate_lv_blood_pool: " + str(closest_slice)


    #------------------------------------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------
    
    best, best_fn = find_best_angle(rois[closest_slice], circles[closest_slice])
    mean, coords, idx = find_threshold_point(best, best_fn)

    #print "Best angle: " + str( (best * 360) / (2 * np.pi) )
    
#    name = "test/roi_time%02d_slice%02d_color.png" % (time, closest_slice)
#    save_image_gray(name, rois[closest_slice])

#    name = "test/circle_time%02d_slice%02d_color.png" % (time, closest_slice)   
#    image.imsave(name, circles[closest_slice])    


#    #------------------------------------------------------------------------------------------------------------------------------------------------
#    #------------------------------------------------------------------------------------------------------------------------------------------------
#
#    img = images[closest_slice, time]
#    
#    name = "imgOrig/img_time%02d_slice%02d_color.png" % (time, closest_slice)
#    
#    image.imsave(name, img)


    #------------------------------------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------
    
    #imgf = img.astype(np.float32)    
#
#    name = "imgFloat32/float_time%02d_slice%02d_color.png" % (time, closest_slice)
#    
#    image.imsave(name, imgf)


#    #------------------------------------------------------------------------------------------------------------------------------------------------
#    #------------------------------------------------------------------------------------------------------------------------------------------------
#    
#    img_gray16 = img.astype(np.uint16) 
#    
#    name = "imgGray/gray16_time%02d_slice%02d_color.png" % (time, closest_slice)
#    
#    save_image_gray(name, img_gray16)


#    #------------------------------------------------------------------------------------------------------------------------------------------------
#    #------------------------------------------------------------------------------------------------------------------------------------------------
#    
#    img_gray8 = get_image_gray(img) # cv2.convertScaleAbs(img_gray16)
#    
#    name = "imgGray/gray8_time%02d_slice%02d_color.png" % (time, closest_slice)
#    
#    save_image_gray(name, img_gray8)

 
    
    #------------------------------------------------------------------------------------------------------------------------------------------------
    # Threshold
    #------------------------------------------------------------------------------------------------------------------------------------------------
    
    #print "Threshold: " + str(mean)
    
    #img_8UC1 = img.astype(np.uint8)
    #img_8UC1 = img_gray8.copy()
  
    #thresh, img_bin = cv2.threshold(imgf, mean, 255.0, cv2.THRESH_BINARY)  
    #thresh, img_bin = cv2.threshold(img_8UC1, mean, 255.0, cv2.THRESH_BINARY)
    #thresh, img_bin = cv2.threshold(img_gray8, mean, 255.0, cv2.THRESH_BINARY)    
    #thresh, img_bin = cv2.threshold(img_gray16, mean, 255.0, cv2.THRESH_BINARY)
    
    #img_bin = cv2.adaptiveThreshold(img_8UC1, 255.0, cv2.ADAPTIVE_THRESH_MEAN_C    , cv2.THRESH_BINARY, 11, 2)
    #img_bin = cv2.adaptiveThreshold(img_8UC1, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)   
    #thresh, img_bin = cv2.threshold(img_8UC1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
   
    thresh, img_bin = cv2.threshold(images[closest_slice, time].astype(np.float32), mean, 255.0, cv2.THRESH_BINARY)   


#    #------------------------------------------------------------------------------------------------------------------------------------------------
#    #------------------------------------------------------------------------------------------------------------------------------------------------
#
#    name = "imgGray/bin_time%02d_slice%02d_color.png" % (time, closest_slice)
#    
#    save_image_gray(name, img_bin)



#    #------------------------------------------------------------------------------------------------------------------------------------------------
#    #------------------------------------------------------------------------------------------------------------------------------------------------
#
#    restrictToRoi = False
#    
#    if restrictToRoi:
#        
#        print "Restrict to ROI..."
#        
#        #roi16 = rois[closest_slice].astype(np.uint16)
#        #roi8 = cv2.convertScaleAbs(roi16)
#        #maskimage = cv2.inRange(roi8, 1, 255)
#    
#        roi = rois[closest_slice] 
#        
#        name = "imgGray/mask0_time%02d_slice%02d_color.png" % (time, closest_slice)
#        save_image_gray(name, maskimage.astype(uint16))
#            
#        
#        maxVal = np.max(roi)
#        
#        print maxVal
#        
#        maskimage = cv2.inRange(roi, 1, 255)
#    
#    
#        
#        name = "imgGray/mask_time%02d_slice%02d_color.png" % (time, closest_slice)
#        save_image_gray(name, maskimage)
#        
#        
#        
#        img_bin = cv2.bitwise_and(img_bin, img_bin, mask=maskimage)
#    
#        name = "thresholds/thresh_time%02d_slice%02d_color.png" % (time, closest_slice)
#        
#        image.imsave(name, img_bin)



#    #------------------------------------------------------------------------------------------------------------------------------------------------
#    #------------------------------------------------------------------------------------------------------------------------------------------------
#    
#    gray = img_8UC1.copy()
# 
#    #cimg = img_8UC1.copy()
#    
#    gray = cv2.bilateralFilter(gray, 11, 17, 17)
#    
#    edged = cv2.Canny(gray, 100, 200, 10)
#
#    maskimage = cv2.inRange(rois[closest_slice], 1, 255)
#    
#    edged = cv2.bitwise_and(edged, edged, mask=maskimage)


#
#    #------------------------------------------------------------------------------------------------------------------------------------------------
#    #------------------------------------------------------------------------------------------------------------------------------------------------
#
#    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#    
#    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
#
#    name = "contour_time%02d_slice%02d_color.png" % (time, closest_slice)
#    
#    save_image_contours(name, edged, contours)



    #------------------------------------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------  

    labeled, component, count = locate_lv_component(img_bin, coords, idx)
          
    #labeled, num = label(img_bin)

#    
#    #------------------------------------------------------------------------------------------------------------------------------------------------
#    #------------------------------------------------------------------------------------------------------------------------------------------------    
#
#    img_comp = img_bin.copy()*0
#    
#    for point in component:
#        
#        x, y = point[0], point[1]
#        
#        img_comp[x][y] = 255
#    
#    
#    #
#    roiArea = np.count_nonzero(rois[closest_slice])
#    lvArea  = np.count_nonzero(img_comp)
#    ratio = float(lvArea) / float(roiArea)
#    
#    #rint "roiArea: " + str(roiArea)
#    #print "lvArea:  " + str(lvArea)
#
#    erode = False
#    
#    if (erode and ratio > 0.11):
#        
#        print "eroding..."
#        
#        #element = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))    
#        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
#        img_comp = cv2.erode(img_comp, element)    
#    
#        #labeled, num = label(img_comp)
#        
#        #print "Updated Connected Components: " + str(num)
#            
##        contours, hierarchy = cv2.findContours(img_comp.copy().astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
##        
##        print "Updated Connected Components: " + str(len(contours))
##        

    
    
#    
#        #------------------------------------------------------------------------------------------------------------------------------------------------
#        #------------------------------------------------------------------------------------------------------------------------------------------------  
#    
#        labeled, component, count = locate_lv_component(img_comp, coords, idx)

#
#    name = "components/component_time%02d_slice%02d_color.png" % (time, closest_slice)
#                
#    image.imsave(name, img_comp)





    #------------------------------------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------        

    hull = cv2.convexHull(component)
    squeezed = hull
    if count <= COMPONENT_INDEX_TOLERANCE:
        squeezed = np.squeeze(squeezed)
    hull = np.fliplr(squeezed)

    mask = np.zeros_like(labeled)
    cv2.drawContours(mask, [hull], 0, 255, thickness=-1)

    coords = coords[idx:idx+1]

    return mask, mean, best_fn, coords

#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def propagate_segments(images, rois, base_mask, mean, closest_slice, time):
    
    def propagate_segment(i, mask):
        
        thresh, img_bin = cv2.threshold(images[i, time].astype(np.float32), mean, 255.0, cv2.THRESH_BINARY)



        #--------------------------------------------------------------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------------------------------------------------------------

        img_gray8 = cv2.convertScaleAbs(images[i, time].astype(np.uint16))

        #thresh, img_bin = cv2.threshold(img_gray8, mean, 255.0, cv2.THRESH_BINARY)


        #--------------------------------------------------------------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------------------------------------------------------------

        maskimage = cv2.inRange(rois[i], 1, 255)

        #img_bin = cv2.bitwise_and(img_bin, img_bin, mask=maskimage)

        #--------------------------------------------------------------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------------------------------------------------------------    

        labeled, features = label(img_bin)

        region1 = mask == 255
        max_similar = -1
        max_region = 0
        for j in range(1, features+1):
            region2 = labeled == j
            intersect = np.count_nonzero(np.logical_and(region1, region2))
            union = np.count_nonzero(np.logical_or(region1, region2))
            similar = float(intersect) / union
            if max_similar == -1 or max_similar < similar:
                max_similar = similar
                max_region = j

        if max_similar == 0:
            component = np.transpose(np.nonzero(mask))
        else:
            component = np.transpose(np.nonzero(labeled == max_region))
        hull = cv2.convexHull(component)
        hull = np.squeeze(hull)
        if hull.shape == (2L,):
            hull = np.array([hull])
        hull = np.fliplr(hull)

        newmask = np.zeros_like(img_bin)

        cv2.drawContours(newmask, [hull], 0, 255, thickness=-1)

        return newmask

    (rois_depth, _, _) = rois.shape
    newmask = base_mask
    masks = {}
    areas = {}
    masks[closest_slice] = base_mask
    areas[closest_slice] = np.count_nonzero(base_mask)


    #-------------------------------------------------------------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------------------------------------------------

    #for i in range(0, closest_slice-1):
    for i in range(closest_slice-1, -1, -1):

        newmask = propagate_segment(i, newmask)        
        newarea = np.count_nonzero(newmask)
        roiarea = np.count_nonzero(rois[i])
        ratio   = float(newarea)/float(roiarea)

#        print "mask: " + str(i)
#        print "area: " + str(newarea)
#        print "roi:  " + str(roiarea)
#        print "ratio: " + str(ratio)     


#        Correct = False
#        
#        if (Correct == True):
#            
#            print "Correcting..."
#            
#            if (ratio > 0.6):
#                
#                ind = i + 1    
#                    
#                masks[i] = masks[ind]
#                areas[i] = areas[ind]
#                
#            else:
#                
#                masks[i] = newmask
#                areas[i] = newarea
#
#
#            print "corrected area: " + str(areas[i])      
#            print ""     
#            print "--------------------"        
#        
#        
#        else:

        masks[i] = newmask
        areas[i] = newarea


    #-------------------------------------------------------------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------------------------------------------------

    newmask = base_mask
    
    for i in range(closest_slice+1, rois_depth):

        newmask = propagate_segment(i, newmask)        
        newarea = np.count_nonzero(newmask)
        roiarea = np.count_nonzero(rois[i])
        ratio   = float(newarea)/float(roiarea)
#        
#        print "mask: " + str(i)
#        print "area: " + str(newarea)
#        print "roi:  " + str(roiarea)
#        print "ratio: " + str(ratio)

#        Correct = False
#        
#        print "Correcting..."
#        
#        if (Correct == True):
#                
#            if (ratio > 0.6):
#                
#                masks[i] = masks[i-1]
#                areas[i] = areas[i-1]
#                
#            else:
#                
#                masks[i] = newmask
#                areas[i] = newarea
#
#            print "corrected area: " + str(areas[i])
#            print ""  
#        
#        
#        else:
            
        masks[i] = newmask
        areas[i] = newarea

    return masks, areas


