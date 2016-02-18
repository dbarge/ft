import sys
from numpy import *
import pandas as p
import matplotlib.pyplot as plt
import argparse


sys.path.append('../../git')
#sys.path.append('../../git/pythonUtils')
#sys.path.append('../../git/pythonUtils/histUtils')

#from pythonUtils import *
import pythonUtils.histUtils as h



#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def main():

  parser = argparse.ArgumentParser()

  parser.add_argument('infile')

  args = parser.parse_args()

  analyze_results(args.infile)



#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def analyze_results(infile):

  df = p.read_csv(infile)
  
  df2 = df
  df2 = df.loc[df['Predicted EDV'] != 0] 
  df2 = df.loc[df['Predicted ESV'] != 0]

  print "Predictions: " + str(len(df2))      


  #-------------------------------------------------------------------------------------------------------------------------------------------------
  # EDV
  #-------------------------------------------------------------------------------------------------------------------------------------------------


  rng_edv_abs = [-300, 300]
  rng_esv_abs = rng_edv_abs

  rng_edv_rel = [-3, 3]
  rng_esv_rel = rng_edv_rel

  edv_err_abs = (df2['Predicted EDV'] - df2['Actual EDV'])
  esv_err_abs = (df2['Predicted ESV'] - df2['Actual ESV'])

  edv_err_rel = (df2['Predicted EDV'] - df2['Actual EDV'])/df2['Actual EDV']
  esv_err_rel = (df2['Predicted ESV'] - df2['Actual ESV'])/df2['Actual ESV']

  edv_err_abs = h.bound(edv_err_abs, rng_edv_abs[0], rng_edv_abs[1])
  esv_err_abs = h.bound(esv_err_abs, rng_esv_abs[0], rng_esv_abs[1])

  edv_err_rel = h.bound(edv_err_rel, rng_edv_rel[0], rng_edv_rel[1])
  esv_err_rel = h.bound(esv_err_rel, rng_esv_rel[0], rng_esv_rel[1])


  #-------------------------------------------------------------------------------------------------------------------------------------------------
  # EF
  #-------------------------------------------------------------------------------------------------------------------------------------------------

  rng_ef_abs = [-1, 1]
  rng_ef_rel = [-1, 1]

  #rng_ef_abs = [-.5, .5]
  #rng_ef_rel = [-.5, .5]

  ef_true = (df2['Actual EDV']    - df2['Actual ESV']   ) / df2['Actual EDV']
  ef_pred = (df2['Predicted EDV'] - df2['Predicted ESV']) / df2['Predicted EDV']

  ef_err_abs = (ef_pred - ef_true)
  ef_err_rel = (ef_pred - ef_true)/ef_true


  ef_err_abs = h.bound(ef_err_abs, rng_ef_abs[0], rng_ef_abs[1])
  ef_err_rel = h.bound(ef_err_rel, rng_ef_rel[0], rng_ef_rel[1])


  print "Mean EDV Error = %f" % mean(abs(edv_err_abs))
  print "Mean ESV Error = %f" % mean(abs(esv_err_abs))
  print "Mean EF  Error = %f" % mean(abs(ef_err_abs))

  print "Mean Rel EDV Error = %f" % mean(abs(edv_err_rel))
  print "Mean Rel ESV Error = %f" % mean(abs(esv_err_rel))
  print "Mean Rel EF  Error = %f" % mean(abs(ef_err_rel))

  #plt.figure()
  #plt.hist(edv_err, bins=100, range=rng) #, label=labels[0], color=colors[0], normed=True)

  #plt.figure()
  #plt.hist(esv_err, bins=100, range=rng) #, label=labels[0], color=colors[0], normed=True)

  #rng = [-1, 1]
  #plt.figure()
  #plt.hist(ef_err, bins=100, range=rng) #, label=labels[0], color=colors[0], normed=True)


  #-------------------------------------------------------------------------------------------------------------------------------------------------
  # EF
  #-------------------------------------------------------------------------------------------------------------------------------------------------

  nbins = 100

  f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
  f.set_size_inches(16, 8) 
  ax1.hist(edv_err_abs, bins=nbins, range=rng_edv_abs, label="Predicted - True EDV") #, color=colors[0], normed=True)
  ax2.hist(esv_err_abs, bins=nbins, range=rng_esv_abs, label="Predicted - True ESV") #, color=colors[0], normed=True)
  ax3.hist(ef_err_abs , bins=nbins, range=rng_ef_abs , label="Predicted - True EF ") #, color=colors[0], normed=True)
  ax4.hist(edv_err_rel, bins=nbins, range=rng_edv_rel, label="Rel Error EDV") #, color=colors[0], normed=True)
  ax5.hist(esv_err_rel, bins=nbins, range=rng_esv_rel, label="Rel Error ESV") #, color=colors[0], normed=True)
  ax6.hist(ef_err_rel , bins=nbins, range=rng_ef_rel , label="Rel Error EF" ) #, color=colors[0], normed=True)


  #-------------------------------------------------------------------------------------------------------------------------------------------------
  # EF
  #-------------------------------------------------------------------------------------------------------------------------------------------------

  plt.show()
  
  return





#-----------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
  
  main()

