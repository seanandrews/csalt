import os, sys, importlib
from csalt.synthesize import make_template, make_data
sys.path.append('configs/')


cfg = 'fiducial_std_sspdemo1'


# only need to do this **one time**.
make_template(cfg)


