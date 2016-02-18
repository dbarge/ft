#
# PARAMETERS
#

# number of bins to use in histogram for gaussian regression
NUM_BINS = 100

# number of standard deviations past which we will consider a pixel an outlier
STD_MULTIPLIER = 2

# number of points of our interpolated dataset to consider when searching for
# a threshold value; the function by default is interpolated over 1000 points,
# so 250 will look at the half of the points that is centered around the known
# myocardium pixel
THRESHOLD_AREA = 250

# number of pixels on the line within which to search for a connected component
# in a thresholded image, increase this to look for components further away
COMPONENT_INDEX_TOLERANCE = 20

# number of angles to search when looking for the correct orientation
ANGLE_SLICES = 36