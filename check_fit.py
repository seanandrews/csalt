from csalt.fit import *

postfile = 'storage/posteriors/naif/fiducial_long_pure_naif_natv.h5'

truths = [40, 130, 1.0, 250., 0.2669, 1.25, 150, -0.5, 20, 297, 
          np.log10(2000.), -1., 5000, 0, 0]


post_analysis(postfile, burnin=0, 
              autocorr=False, corner_plot=False, truths=truths)
