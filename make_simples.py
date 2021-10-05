import os, sys
import numpy as np
from csalt.synthesize import make_data
from csalt.dmr import *


make_data('simple2-default')
foo = img_cube('simple2-default')

#bar = dmr('simple2-default')
#poo = img_cube('simple2-default', cubetype='MOD', makemask=False)
#pee = img_cube('simple2-default', cubetype='RES', makemask=False)



