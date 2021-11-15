"""
    This CASA script (optionally) reduces an available (concatenated) MS by 
    time-averaging and sub-selecting a given velocity range.  It is called 
    inside csalt.synthesize.make_data(), or can be used as a standalone script 
    for a real dataset as

        casa -c format_data.py configs/gen_<cfg_file> <arg1>

    where <cfg_file> is the relevant part of the configuration input filename 
    and <arg1> is an *optional* argument that contains a (string) filename 
    extension (usually "pure" or "noisy" in the csalt.synthesize framework).  
    (This *will* change when we update to full CASA v6.x.)

    This script will output ...

"""

import os, sys
import numpy as np
import scipy.constants as sc
import h5py


"""
    Parse inputs and load relevant information.
"""
# Ingest input arguments
if len(sys.argv) == 3:
    cfg_file = sys.argv[-1]
    _ext = ''
else:
    cfg_file = sys.argv[-2]
    _ext = '_'+sys.argv[-1]

# Make sure the configuration file exists
if os.path.exists(cfg_file+'.py'):
    execfile(cfg_file+'.py')
else:
    print('Could not find input configuration file!')
    sys.exit()

# Make sure outdir exists
if reduced_dir[-1] != '/': reduced_dir += '/'
outdir = reduced_dir+basename+'/'
if not os.path.exists(outdir):
    os.system('mkdir '+outdir)

# Load the "raw" MS datafile contents
in_MS += _ext
if not os.path.exists(in_MS+'.ms'):
    print('Could not find the input "raw" MS file!')
    print('"'+in_MS+'"'+' does not seem to exist.')
    sys.exit()
tb.open(in_MS+'.ms')
spw_col = tb.getcol('DATA_DESC_ID')
obs_col = tb.getcol('OBSERVATION_ID')
field_col = tb.getcol('FIELD_ID')
tb.close()

# Identify the unique EBs inside the MS datafile
obs_ids = np.unique(obs_col)
nEB = len(obs_ids)


"""
    Separate the individual EBs and time-average as specified by user.
    The individual MS files are only stored temporarily during manipulations.
"""
for EB in range(nEB):
    spws = np.unique(spw_col[np.where(obs_col == obs_ids[EB])])
    if len(spws) == 1:
        spw_str = str(spws[0])
    else:
        spw_str = "%d~%d" % (spws[0], spws[-1])

    fields = np.unique(field_col[np.where(obs_col == obs_ids[EB])]) 
    if len(fields) == 1:
        field_str = str(fields[0])
    else:
        field_str = "%d~%d" % (fields[0], fields[-1])

    os.system('rm -rf '+dataname+'_tmp'+str(EB)+'.ms*')
    split(vis=in_MS+'.ms', outputvis=dataname+'_tmp'+str(EB)+'.ms', 
          spw=spw_str, field=field_str, datacolumn='data', timebin=tavg[EB], 
          keepflags=False)


# Create a data dictionary
#data_dict = {'nobs': nEB, 
#             'original_MS': in_MS, 
#             'V_bounds_V': V_bounds,
#             'tavg': tavg}
#np.save(dataname, data_dict)
#
# Create an HDF5 file, and populate the top-level group with basic info
os.system('rm -rf '+dataname+_ext+'.h5')
f = h5py.File(dataname+_ext+'.h5', "w")
f.attrs["nobs"] = nEB
f.attrs["original_MS"] = in_MS+'.ms'
f.attrs["V_bounds"] = V_bounds
f.attrs["tavg"] = tavg
f.close()


# Loop through each EB
concat_files = []
for EB in range(nEB):

    # Get data
    tb.open(dataname+'_tmp'+str(EB)+'.ms')
    data_all = np.squeeze(tb.getcol('DATA'))
    u, v = tb.getcol('UVW')[0,:], tb.getcol('UVW')[1,:]
    wgt_all = tb.getcol('WEIGHT')
    times = tb.getcol('TIME')
    tb.close()

    # Parse timestamps
    tstamps = np.unique(times)
    tstamp_ID = np.empty_like(times)
    for istamp in range(len(tstamps)):
        tstamp_ID[times == tstamps[istamp]] = istamp

    # Get TOPO frequencies
    tb.open(dataname+'_tmp'+str(EB)+'.ms/SPECTRAL_WINDOW')
    nu_TOPO_all = np.squeeze(tb.getcol('CHAN_FREQ'))
    tb.close()

    # Calculate LSRK frequencies for each timestamp
    nu_LSRK_all = np.empty((len(tstamps), len(nu_TOPO_all)))
    ms.open(dataname+'_tmp'+str(EB)+'.ms')
    for istamp in range(len(tstamps)):
        nu_LSRK_all[istamp,:] = ms.cvelfreqs(mode='channel', outframe='LSRK', 
                                             obstime=str(tstamps[istamp])+'s')
    ms.close()

    # Identify channel boundaries for the requested LSRK range
    V_LSRK_all = sc.c * (1 - nu_LSRK_all / nu_rest)
    chslo = np.argmin(np.abs(V_LSRK_all - V_bounds[0]), axis=1)
    chshi = np.argmin(np.abs(V_LSRK_all - V_bounds[1]), axis=1)
    if np.diff(nu_TOPO_all)[0] < 0:
        chlo, chhi = chslo.min(), chshi.max()
    else:
        chlo, chhi = chshi.min(), chslo.max()	

    # Slice out the data of interest
    nu_TOPO = nu_TOPO_all[chlo-bounds_pad:chhi+bounds_pad+1]
    nu_LSRK = nu_LSRK_all[:,chlo-bounds_pad:chhi+bounds_pad+1]
    data = data_all[:,chlo-bounds_pad:chhi+bounds_pad+1,:]
    if wgt_all.shape == data_all.shape: 
        wgt = wgt_all[:,chlo-bounds_pad:chhi+bounds_pad+1,:]
    else:
        wgt = wgt_all

    # Pack the data into the HDF5 output file
    f = h5py.File(dataname+_ext+'.h5', "a")
    f.create_dataset("EB"+str(EB)+"/um", u.shape)[:] = u
    f.create_dataset("EB"+str(EB)+"/vm", v.shape)[:] = v
    f.create_dataset("EB"+str(EB)+"/data_re", data.shape)[:,:,:] = data.real
    f.create_dataset("EB"+str(EB)+"/data_im", data.shape)[:,:,:] = data.imag
    f.create_dataset("EB"+str(EB)+"/weights", wgt.shape)[:,:] = wgt
    f.create_dataset("EB"+str(EB)+"/tstamp_ID", tstamp_ID.shape)[:] = tstamp_ID
    f.create_dataset("EB"+str(EB)+"/nu_TOPO", nu_TOPO.shape)[:] = nu_TOPO
    f.create_dataset("EB"+str(EB)+"/nu_LSRK", nu_LSRK.shape)[:,:] = nu_LSRK
    f.close()
    #os.system('rm -rf '+dataname+_ext+'_EB'+str(EB)+'.npz')
    #np.savez_compressed(dataname+_ext+'_EB'+str(EB), data=data, um=u, vm=v, 
    #                    weights=wgt, tstamp_ID=tstamp_ID, 
    #                    nu_TOPO=nu_TOPO, nu_LSRK=nu_LSRK)

    # Split off a MS with the "reduced" data from this EB
    if not os.path.exists(reduced_dir+basename+'/subMS'):
        os.system('mkdir '+reduced_dir+basename+'/subMS')
    sub_ = reduced_dir+basename+'/subMS/'+basename+_ext+'_EB'+str(EB)+'.DAT.ms'
    os.system('rm -rf '+sub_)
    spwtag = '0:'+str(chlo-bounds_pad)+'~'+str(chhi+bounds_pad)
    split(vis=dataname+'_tmp'+str(EB)+'.ms', outputvis=sub_,
          datacolumn='data', spw=spwtag)
    concat_files += [sub_]


# Concatenate the MS files
os.system('rm -rf '+dataname+_ext+'.ms')
if len(concat_files) > 1:
    concat(vis=concat_files, concatvis=dataname+_ext+'.ms', dirtol='0.1arcsec',
           copypointing=False)
else:
    os.system('cp -r '+concat_files[0]+' '+dataname+_ext+'.ms')


# Cleanup
os.system('rm -rf '+dataname+'_tmp*.ms*')
os.system('rm -rf *.last')
