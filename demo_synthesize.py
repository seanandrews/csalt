'''
    This is a demonstration of how to use csalt to generate a synthetic ALMA 
    dataset from scratch, based on the configuration file 'gen_simple-demo.py'.

    The make_data() function creates a set of blank (u, v, freq) tracks using 
    the standard CASA functionality (simobserve), and then populates them with 
    the FT of an input (calculated here) model cube.  It accounts for both the 
    ALMA "Doppler setting" issue and the spectral response function of the 
    correlator (see the csalt paper).  The outputs are a (concatenated) CASA 
    measurement set and an HDF5 file that contains the relevant subset of that 
    information (for easier manipulation outside of CASA using other csalt 
    functionality).  If desired, the _raw_ model cube can be output as a FITS 
    file also.  These are all located in

        outputbase_dir/reduced_dir/basename/

    as specified in 'gen_simple-demo.py'.  Once the visibilities are generated, 
    you can image them or model them as shown in other demo files.

    This code can be run on the command-line as

        % python demo_synthesize.py

    or by pasting the below into a Jupyter notebook, etc.  It assumes you're 
    operating in the main csalt directory, and that 'gen_simple-demo.py' is 
    located in the 'configs/' subdirectory.  
'''

from csalt.synthesize import make_data
from csalt.dmr import dmr, img_cube

make_data('simple-demo', mtype='csalt', make_raw_FITS=True)

dmr('simple-demo', mtype='csalt', make_raw_FITS=True)

img_cube('simple-demo', cubetype='DATA', mask_name=None)


