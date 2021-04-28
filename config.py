"""
    This is the main control file, used to generate synthetic data or to model
    real (or synthetic) datasets.
"""

# file naming
basename = 'test'
uvfile = 'uvtest'


# controls
gen_uv = False
gen_data = False
fit_data = False



# Simulated observations parameters

# spectral settings
dfreq0   = 61.035e3    # in Hz
restfreq = 230.538e9          # in Hz
vsys     = 4.0e3             # in m/s
vspan    = 10e3             # in m/s
sosampl  = 3     

# spatial settings
RA   = '04:30:00.00'
DEC  = '25:00:00.00'
HA   = '0.0h'
date = '2021/12/01'

# observation settings
config = '5'
ttotal = '20min'
integ  = '30s'







