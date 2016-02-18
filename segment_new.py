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

from multiprocessing import *

from subprocess import *
from threading import *
from time import *

from helpers import *
import time

from optparse import *
from argparse import *

#PARALLEL = False
PARALLEL = True

#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def async_proc(x):
    
    print "starting thread " + str(x)
    time.sleep(5)
    print "finished thread " + str(x)
    
    return x

#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def pool_test():
  
    nProcs = 1000
    
    resutls = []
    
    pool = Pool(processes=nProcs)         # start 4 worker processes
    res = pool.map_async(async_proc, range(nProcs))
    pool.close()
    pool.join()

    print res.get()
    
    #print results

    #results = []
    
#    for int i in range (0, nProcs):
#        
#        result = pool.apply_async(async_proc, (i,))   # evaluate "f(10)" asynchronously in a single process
#        result.get()
#        results.append(result)
   
    #print result.get(timeout=1)           # prints "100" unless your computer is *very* slow

    #res = pool.map(async_proc, range(nProcs))          # prints "[0, 1, 4,..., 81]"
    #res.get()
    
    #print res
    
    #it = pool.imap(f, range(10))
    #print it.next()                       # prints "0"
    #print it.next()                       # prints "1"
    #print it.next(timeout=1)              # prints "4" unless your computer is *very* slow

    #result = pool.apply_async(time.sleep, (10,))
    #print result.get(timeout=1)           # raises multiprocessing.TimeoutError

#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def async_test():

    #------------------------------------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------

    #print "CPUs: " + str(multiprocessing.cpu_count())

    nProcs = 10
    procs = []    
    
    for i in range(0, nProcs):
        
        proc = Process(target=async_proc, args=(i,))
        proc.start()
        procs.append(proc)


    #------------------------------------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------

    for i in range(0, len(procs)):
        
        procs[i].join()    

    
    #exit(0)
    
    return
    
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------
    
def get_comma_separated_args(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))

d = ""
    
def auto_segment_all_datasets():
    
    start_time = time.time()

    #
    #pool_test()
    #async_test()
    #exit(0)
    
    
    #------------------------------------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------
    
    global d
    d = sys.argv[1]

    #print "Data directory: " + d

    studies = next(os.walk(os.path.join(d, "train")))[1] + next(os.walk(os.path.join(d, "validate")))[1]
    #studies = next(os.walk(os.path.join(d, "validate")))[1]
    #studies = next(os.walk(os.path.join(d, "train")))[1]
        
    #print studies

#
#    #------------------------------------------------------------------------------------------------------------------------------------------------
#    #------------------------------------------------------------------------------------------------------------------------------------------------
#  
#    path = os.path.join(d, "train.csv")
#
#    #print path
#
#    labels = np.loadtxt(path, delimiter=",", skiprows=1)
#
#    #print labels
#
#    label_map = {}
#    for l in labels:
#        label_map[l[0]] = (l[2], l[1])
#

    
    #------------------------------------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------
     
    num_samples = None
    
    parser = OptionParser()
    parser.add_option("-n", "--num_samples"    , type="int", dest="num_samples"    , help="number of samples to process")
    #parser.add_option("-l", "--list_of_samples", dest="list_of_samples", help="list of samples to process")
    parser.add_option('-l', '--list_of_samples', type='string', action='callback', callback=get_comma_separated_args, dest="list_of_samples")
                  
    (options, args) = parser.parse_args()
    
    
    if (options.num_samples and options.list_of_samples):
        print "\nError: Specify num_samples OR list_of_samples AND not both\n"
        exit(0)

    elif (options.num_samples):        
  
        #random.seed(0)
        #studies = random.sample(studies, options.num_samples)
        #studies.sort()
        
        studies_int = [int(x) for x in studies]
        studies_int.sort() 
        studies_int = studies_int[0:options.num_samples]
        studies = [str(x) for x in studies_int]        

        
    elif (options.list_of_samples):        
    
        studies = options.list_of_samples
      
#    else:
#        print "else"
       
 
    num_samples = len(studies)        
    #studies.sort()       
        
    #print options
    #print args
    #print num_samples
    #print ""
    #print studies
    
    #return

    
#    #------------------------------------------------------------------------------------------------------------------------------------------------
#    # subsample
#    #------------------------------------------------------------------------------------------------------------------------------------------------
#
#
#    if len(sys.argv) > 2:
#
#        arg = sys.argv[2]
#        argtype = type(arg)
#    
#        print argtype        
#        
#        
#        if (argtype == type(str)):
#    
#            num_samples = int(arg)
#            random.seed(0)
#            studies = random.sample(studies, num_samples)
#
#        else:
#            print "Unexpected 2nd argument: " + arg
#
#    print studies 
        

    
    #------------------------------------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------

    if os.path.exists("output"):
        shutil.rmtree("output")

    if os.path.exists("middle_slices"):
        shutil.rmtree("middle_slices")
        
    os.mkdir("output")
    os.mkdir("middle_slices")


    #------------------------------------------------------------------------------------------------------------------------------------------------
    # Initialize file paths & writer headers
    #------------------------------------------------------------------------------------------------------------------------------------------------

    submit_csv, accuracy_csv = init_output("submit.csv", "accuracy.csv")
    

    #------------------------------------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------

    PARALLEL = False
    
    if not PARALLEL:    
        
        for s in studies:
    
            #dset = get_dataset(d, s)
            
            #path = get_path(d, s)
            #dset = Dataset(path, s)
    
            #index = int(s)
            #process_dataset(dset, index)

            dset = process_dataset(s)            
            save_output(dset, d, submit_csv, accuracy_csv)


    #------------------------------------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------
    
    else:

        nProcs = 8
        chunks = 1        

        #print "Total datasets: " + str(studies)


        
        #for a in args:
        #    print a

        args = (s for s in studies)
        
        #for a in args:
        #    print "a: " + str(a)
        
        pool = Pool(processes=nProcs)
        async_result = pool.map_async(process_dataset, args)#, chunksize=chunks)
        pool.close()
        pool.join()

        #exit(0)
        
        
        results = async_result.get()
    
    
        print "Saving output for " + str(len(results)) + " datasets..."
        
        for dset in results:
                        
            save_output(dset, d, submit_csv, accuracy_csv)
    
            print "Index = " + str(dset.index) + ", Dataset = " + str(dset.name) + ", ESV = " + str(dset.esv) + ", EDV = " + str(dset.edv)
    
    
    #exit(0)

#        length = len(studies)        
#        
#        nGroups = 8
#        
#        mod = length / nGroups        
#        rem = length % nGroups
#        #rem = len(studies) - mod*nGroups
#        
#        print "mod: " + str(mod)
#        print "rem: " + str(rem)
#        print "len: " + str(length)
#        print studies        
#        
#  
#
#        
#        
#        #--------------------------------------------------------------------------------------------------------------------------------------------
#        #--------------------------------------------------------------------------------------------------------------------------------------------  
#        
#        for i0 in range (0, length, 8):
#
#            j = (rem if i0 + nGroups > length else nGroups)
#         
#            dsets = []
#            procs = []  
#
#            #----------------------------------------------------------------------------------------------------------------------------------------
#            #----------------------------------------------------------------------------------------------------------------------------------------  
#                        
#            for i in range(0, j):
#                
#                index = i0 + i
#                
#                dset = get_dataset(d, studies[index]) 
#                dsets.append(dset)
#
#                print "group = " + str(i0) + "\tgroup = " + str(i0/nGroups) + "\tsub = " + str(i) + "\tind = " + str(index) + "\tdataset = " + dset.name                
#
#                proc = Process(target=process_dataset, args=(dsets[i], dsets[i].name,))                
#                proc.start()
#                procs.append(proc)
#
#
#
#
#
#
#            #----------------------------------------------------------------------------------------------------------------------------------------
#            #----------------------------------------------------------------------------------------------------------------------------------------  
#                                 
#            for i in range(0, len(procs)):
#                
#                procs[i].join()
#                print "EDV = " + str(dsets[i].esv) + "\tESV = " + str(dsets[i].esv)
#
#            #----------------------------------------------------------------------------------------------------------------------------------------
#            #----------------------------------------------------------------------------------------------------------------------------------------                   
#                
#            for i in range(0, len(dsets)):       
#                
#                save_output(dsets[i], d, submit_csv, accuracy_csv)
  
    
    
#        mod = (mod + 1 if rem > 0 else mod)
#
#        
#        for i in range(0, mod):
#            
#            print "group " + str(i)   
#         
#            for j in range(0, nGroups):
#
#                index = i*nGroups+j
#                
#                dset = get_dataset(d, studies[index]) 
#            
#                print "   index = " + str(index) + "\t\tdataset = " + dset.name 
     
     
#        #------------------------------------------------------------------------------------------------------------------------------------------------
#        #------------------------------------------------------------------------------------------------------------------------------------------------
#
#        print "remainder group"   
#        
#        for j in range(0, rem):            
#                
#            index = mod*nGroups+j
#                
#            dset = get_dataset(d, studies[index]) 
#            dsets.append(dset)
#            
#            print "   index = " + str(index) + "\t\tdataset = " + dset.name 
#         
#            proc = Process(target=process_dataset, args=(dset, dset.name,))
#            proc.start()
#            procs.append(proc)
         
         
    
#            
#        #------------------------------------------------------------------------------------------------------------------------------------------------
#        #------------------------------------------------------------------------------------------------------------------------------------------------
#      
#        for i in range(0, len(procs)):
#                
#            procs[i].join()
#
#
#        #------------------------------------------------------------------------------------------------------------------------------------------------
#        #------------------------------------------------------------------------------------------------------------------------------------------------
  
            
         
#        for s in studies:
#    
#            dset = get_dataset(d, s)        
#            index = int(s)
#
#            #------------------------------------------------------------------------------------------------------------------------------------------------
#            #------------------------------------------------------------------------------------------------------------------------------------------------
#                                  
#            nProcs = 1
#        
#    
#            procs = []    
#                
#            for i in range(0, nProcs):
#                
#                proc = Process(target=process_dataset, args=(dset, index,))
#                proc.start()
#                procs.append(proc)
#        
#            for i in range(0, len(procs)):
#                
#                procs[i].join()
            
            
        
    #------------------------------------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------

    accuracy_csv.close()
    submit_csv.close()

    #
    stop_time = time.time()
    dt = (stop_time - start_time)/60    

    print "Run time (min): " + str(dt)
    
    return


#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

#def process_dataset(dset):             
#def process_dataset(path, s):        
def process_dataset(s): 
    
    global d
    
    path = get_path(d, s)
    dset = Dataset(path, s)                  

    #dset = Dataset(s)  
        
    print "\nProcessing dataset %s (Index = %s)..." % ( dset.name, str(dset.index) )

    #return dset

    #------------------------------------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------
    
    try:
     
        dset.load()
        
    except Exception as e:
        
        log(dset.index, "***ERROR***: Exception %s loading dataset %s" % (str(e), dset.name), 0)


    #------------------------------------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------
         
    try:
        
        segment_dataset(dset, dset.index)

    
    except Exception as e:
        
        log(dset.index, "***ERROR***: Exception %s segmenting dataset %s" % (str(e), dset.name), 0)


    #------------------------------------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------ 
 
    print "Finished processing dataset %s..." % dset.name
    print "EDV = " + str(dset.edv) + "\tESV = " + str(dset.esv)
            
    return dset


#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------
        
def save_output(dset, d, submit_csv, accuracy_csv):  

    print "Saving results for dataset %s..." % dset.name
    
    maxVol = 600
    
    if dset.edv >= maxVol:
        msg = "EDV prediction too large: %s" % str(dset.edv)
        print msg
        #raise Exception(msg)
        #dset.edv = 0.0 #maxVol

    if dset.esv >= maxVol:
        msg = "ESV prediction too large: %s" % str(dset.esv)
        print msg
        #raise Exception(msg)
        #dset.esv = 0.0 #maxVol
        
    if (dset.edv >= maxVol or dset.esv >= maxVol):
        dset.edv = 0.0
        dset.esv = 0.0        
        
    p_edv = dset.edv
    p_esv = dset.esv
    

    #--------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------------------------------
        
    submit_csv.write("%d_systolic," % int(dset.name))
    
    for i in range(0, 600):
        
        if i < p_esv:
            submit_csv.write("0.0")
        else:
            submit_csv.write("1.0")
            
        if i == 599:
            submit_csv.write("\n")
        else:
            submit_csv.write(",")


    #--------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------------------------------
  
    submit_csv.write("%d_diastolic," % int(dset.name))
    
    for i in range(0, 600):
        
        if i < p_edv:
            submit_csv.write("0.0")
        else:
            submit_csv.write("1.0")
            
        if i == 599:
            submit_csv.write("\n")
        else:
            submit_csv.write(",")


    #------------------------------------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------
  
    path = os.path.join(d, "train.csv")
    labels = np.loadtxt(path, delimiter=",", skiprows=1)
    label_map = {}
    
    for l in labels:
        label_map[l[0]] = (l[2], l[1])
    
    
    #--------------------------------------------------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------------------------------------------------
                  
    (edv, esv) = label_map.get(int(dset.name), (None, None))
    
    if edv is not None:
        accuracy_csv.write("%s,%f,%f,%f,%f\n" % (dset.name, edv, esv, p_edv, p_esv))
    
    return

    
#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------
        
def init_output(submit, accuracy):  
    
    #------------------------------------------------------------------------------------------------------------------------------------------------
    # Initialize file paths & writer headers
    #------------------------------------------------------------------------------------------------------------------------------------------------

    accuracy_csv = open(accuracy, "w")
    accuracy_csv.write("Dataset,Actual EDV,Actual ESV,Predicted EDV,Predicted ESV\n")
    
    submit_csv = open(submit, "w")
    submit_csv.write("Id,")
    
    for i in range(0, 600):
        
        submit_csv.write("P%d" % i)
        
        if i != 599:
            submit_csv.write(",")
            
        else:
            submit_csv.write("\n")
                      
    return submit_csv, accuracy_csv
    

#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------
        
def get_path(d, s):  
    
    if int(s) <= 500:
        full_path = os.path.join(d, "train", s)
        
    else:
        full_path = os.path.join(d, "validate", s)

    return full_path

    
##-----------------------------------------------------------------------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------------------------------------------------------------------
#        
#def get_dataset(d, s):  
#    
#    if int(s) <= 500:
#        full_path = os.path.join(d, "train", s)
#        
#    else:
#        full_path = os.path.join(d, "validate", s)
#
#    return Dataset(full_path, s)


#------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------

class ImageData(object):
    
    class MetaData(object):
        
        def __init__(self):
            
            self.SliceLocation = None
            self.patCoords = None
            
    def __init__(self):
      
        self.img = None
        self.meta = self.MetaData()

        
        return
#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

class Dataset(object):
    dataset_count = 0

#    def __init__(self, subdir):
#        
#        self.index = int(subdir)
#        
#        self.edv = float(self.index)
#        self.esv = float(self.index)
#
##        self.directory = directory
##        self.time = sorted(times)
##        self.slices = sorted(slices)
##        self.slices_map = slices_map
##        
##        Dataset.dataset_count += 1
#
#        self.name = subdir
#        
#        return
        
        
 
    #------------------------------------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------

    def __init__(self, directory, subdir):
        
        # deal with any intervening directories
        while True:
            subdirs = next(os.walk(directory))[1]
            if len(subdirs) == 1:
                directory = os.path.join(directory, subdirs[0])
            else:
                break

        slices = []
        for s in subdirs:
            m = re.match("sax_(\d+)", s)
            if m is not None:
                slices.append(int(m.group(1)))

        slices_map = {}
        first = True
        times = []
        for s in slices:
            files = next(os.walk(os.path.join(directory, "sax_%d" % s)))[2]
            offset = None

            for f in files:
                m = re.match("IM-(\d{4,})-(\d{4})\.dcm", f)
                if m is not None:
                    if first:
                        times.append(int(m.group(2)))
                    if offset is None:
                        offset = int(m.group(1))

            first = False
            slices_map[s] = offset

        self.index = int(subdir)
        
        self.edv = 0.0
        self.esv = 0.0

        self.directory = directory
        self.time = sorted(times)
        self.slices = sorted(slices)
        self.slices_map = slices_map
        Dataset.dataset_count += 1
        self.name = subdir


    #------------------------------------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------

    def _filename(self, s, t):
        
        return os.path.join(self.directory,"sax_%d" % s, "IM-%04d-%04d.dcm" % (self.slices_map[s], t))


    #------------------------------------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------

    def test(self, d):
        
        pos = d.PatientPosition
        
        rad2deg = float(360) / (float(2)*np.pi)
        

        #--------------------------------------------------------------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------------------------------------------------------------
        
        row_dcosX = d.ImageOrientationPatient[0]
        row_dcosY = d.ImageOrientationPatient[1]
        row_dcosZ = d.ImageOrientationPatient[2]        

        col_dcosX = d.ImageOrientationPatient[3]
        col_dcosY = d.ImageOrientationPatient[4]
        col_dcosZ = d.ImageOrientationPatient[5] 
        
        nrm_dcosX = row_dcosY* col_dcosZ - row_dcosZ * col_dcosY; 
        nrm_dcosY = row_dcosZ* col_dcosX - row_dcosX * col_dcosZ; 
        nrm_dcosZ = row_dcosX* col_dcosY - row_dcosY * col_dcosX; 
        
        
        rot = np.matrix([
                [ row_dcosX, row_dcosY, row_dcosZ ],
                [ col_dcosX, col_dcosY, col_dcosZ ],
                [ nrm_dcosX, nrm_dcosY, nrm_dcosZ ]
        ])


        #--------------------------------------------------------------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------------------------------------------------------------

        rowvec = np.matrix([1, 1, 1])
        colvec = np.matrix([ [1], [1], [1] ])
        
        #print rowvec
        #print colvec
        
        
        #--------------------------------------------------------------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------------------------------------------------------------

        pat_unitX = np.matrix([ [1], [0], [0] ])
        pat_unitY = np.matrix([ [0], [1], [0] ])
        pat_unitZ = np.matrix([ [0], [0], [1] ])
        
        
        #--------------------------------------------------------------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------------------------------------------------------------
        
        patX_img = rot * pat_unitX
        patY_img = rot * pat_unitY
        patZ_img = rot * pat_unitZ        

        
        patX_img_proj = [ patX_img.item(0, 0), patX_img.item(1, 0) ]
        patY_img_proj = [ patY_img.item(0, 0), patY_img.item(1, 0) ]
        patZ_img_proj = [ patZ_img.item(0, 0), patZ_img.item(1, 0) ]
        
        #patY_img_proj = [ [patY_img[0]], [patY_img[1]] ]
        #patZ_img_proj = [ [patZ_img[0]], [patZ_img[1]] ]
            
        #print patX_img_proj
        #print patY_img_proj
        #print patZ_img_proj        

        #hollow_mask = roi*0                
    
        #pt1 = cv2.Point(0, 0)
        
        #cv2.line(hollow_mask, )
    
        #exit(1)
    
        ret = [ patX_img_proj, patY_img_proj, patZ_img_proj ]  
        #print np.multiply(patX_img_proj, 10)
        
        #print ret[0]
        
        #exit(0)
        
        return ret
        
    #------------------------------------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------

    def _read_dicom_image(self, filename):
        
        d = dicom.read_file(filename)
        #print d
        #print filename
        #print "Slice: " + str(s) + " Location: " + str(d.SliceLocation)
        
        #exit(1)

        img = d.pixel_array

        data = ImageData()
        data.img = np.array(img)

        data.meta = ImageData.MetaData()
        data.meta.SliceLocation = d.SliceLocation        
        
        
        #print d

            
        data.meta.patCoords = self.test(d)
        
#        print filename + ", " + str(d.PhotometricInterpretation) + ", " + str(d.ImageType) 
#
#        if (d.PhotometricInterpretation != "MONOCHROME2"):
#            print "Unexpted: " + str(d.PhotometricInterpretation)

       
        
        #rot = np.matrix([row_costheta_x, row_costheta_y, 0])
        
#                 [col_costheta_x, col_costheta_y, 0],
#                 [0, 0, 
#        ])
        
        #print filename + "z = " + str(z) + ", pos = " + str(pos)
        #print filename + ": theta_x = " + str(theta_x) + ", theta_y = " + str(theta_y) + ", theta_z = " + str(theta_z)
        
        #print filename + ", " + str(d.SmallestImagePixelValue) + ", " +  str(d.LargestImagePixelValue)             

        #print d.dir("rientation")
        
        #print d
        
        #exit(0)
        
        return data
        #return np.array(img)#, d #, d.SliceLocation, d.SliceThickness


    #------------------------------------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------

    def _read_all_dicom_images(self):
        
        f1 = self._filename(self.slices[0], self.time[0])
        d1 = dicom.read_file(f1)
        (x, y) = d1.PixelSpacing
        (x, y) = (float(x), float(y))
        f2 = self._filename(self.slices[1], self.time[0])
        d2 = dicom.read_file(f2)

        # try a couple of things to measure distance between slices
        try:
            
            dist = np.abs(d2.SliceLocation - d1.SliceLocation)
            
        except AttributeError:
            
            try:
                
                dist = d1.SliceThickness
                
            except AttributeError:
                
                dist = 8  # better than nothing...



        #
        self.images = np.array([[self._read_dicom_image(self._filename(d, i)).img for i in self.time] for d in self.slices])
        self.metas  = np.array([[self._read_dicom_image(self._filename(d, i)).meta for i in self.time] for d in self.slices])


#        print type(self.images)
#        print self.images.shape
#        print len(self.time)
#        print len(self.slices)

#        self.images = np.zeros( (len(self.slices), len(self.time), 0, 0) )
#        
#        print self.images.shape         
#        
#        #self.images.shape = 
#
#        
#        for i in self.time:
#            
#            id = 0
#            
#            for d in self.slices:
#                
#                img = self._read_dicom_image(self._filename(d, i))
#                
#                print img.shape
#
#                #print d
#                #print self.images[:,:,:,:].shape                
#                print self.images[id, i-1,:,:].shape
#                
#                self.images[id, i-1,:,:].shape = img.shape
#                
#                print "--"
#                
#                id += 1
#                
#                #.append(file)
#                
#                #print str(d) + ": " + str(loc) + " " + str(t)
#            #print ""
#            
#
#        print self.images.shape
#        
#        exit(1)
        
        self.dist = dist
        self.area_multiplier = x * y
#        
#        #exit(1)
        

   
    #------------------------------------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------
        
    def load(self):
        
        self._read_all_dicom_images()




#-----------------------------------------------------------------------------------------------------------------------------------------------------
# assumes dataset is loaded, call dataset.load()
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def segment_dataset(dataset, index):
    
    images = dataset.images
    metas = dataset.metas
    dist = dataset.dist
    areaMultiplier = dataset.area_multiplier
    
    #return
    
    # shape: num slices, num snapshots, rows, columns
    log(index, "Calculating rois...", 1)
    rois, circles = calc_rois(images, index)
    
    log(index, "Calculating areas...", 1)
    all_masks, all_areas, all_lines, all_coords, closest = calc_all_areas(images, rois, circles, index)

    log(index, "Calculating volumes...", 1)
    area_totals = [calc_total_volume(a, areaMultiplier, dist) for a in all_areas]
    
    log(index, "Calculating ef...", 1)
    edv = max(area_totals)
    esv = min(area_totals)
    ef = (edv - esv) / edv
    log(index, "Done, ef is %f" % ef, 1)

    log(index, "Saving masks...", 1)
    save_masks_to_dir(dataset, all_masks, rois, circles, all_lines, all_coords, closest, metas)
    log(index, "Done", 1)
    
    output = {}
    output["edv"] = edv
    output["esv"] = esv
    output["ef"] = ef
    output["areas"] = all_areas.tolist()
    f = open("output/%s/output.json" % dataset.name, "w")
    json.dump(output, f, indent=2)
    f.close()
    
    print "EDV = " + str(edv) + "\tESV = " + str(esv)
    
    dataset.edv = edv
    dataset.esv = esv
    dataset.ef = ef


#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    random.seed()
    auto_segment_all_datasets()
