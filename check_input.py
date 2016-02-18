import os
import numpy as np

mydir = "output"

case_dirs = os.listdir(mydir)

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

for i in case_dirs:

  nDupCases = 0
  
  if int(i) >= 500:
    continue

  case_path = mydir + "/" + i
  
  if os.path.isfile(case_path):
    continue

  #print case_path

  time_dirs = os.listdir(case_path)

  CaseHasDuplicates = False

  #--------------------------------------------------------------------------------------------------
  #--------------------------------------------------------------------------------------------------

  NumTimes = 0
  NumTimeDuplicates = 0

  for j in time_dirs:

    time_path = case_path + "/" + j

    if os.path.isfile(time_path):
      continue

    #print "   " + time_path 

    slices = os.listdir(time_path)

    TimeHasDuplicates = False


    #--------------------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------

    locs = []

    for k in slices:

      fpath = time_path + "/" + k

      if not os.path.isfile(fpath):
        continue
   
      #print "      " + k

      z = k.split('_')[1]

      z = float(z)

      locs.append(z)

      #print "      " + str(z)


      #--------------------------------------------------------------------------------------------------
      #--------------------------------------------------------------------------------------------------

      nDup = 0
      
      for l in slices:

        fpath2 = time_path + "/" + l

        if not os.path.isfile(fpath2):
          continue

        z2 = l.split('_')[1]

        z2 = float(z2)

        if (z == z2):
          nDup += 1

      if nDup > 1:

        TimeHasDuplicates = True
        CaseHasDuplicates = True

        #print "      " + str(nDup) + " duplicates for slice " + k + " " + str(z) + " " + time_path

    if TimeHasDuplicates:

      NumTimeDuplicates += 1
      #print "   " + time_path + " has duplicates"

    else:

      locs.sort()
      print locs
      print np.unique(locs) 
      print ""

      med = np.median(locs)
      
      ind = locs.index(med)

      print ind


    NumTimes += 1




  if CaseHasDuplicates:

    print "   " + case_path + " has duplicates in " + str(NumTimeDuplicates) + " of " + str(NumTimes) + " time directories"

