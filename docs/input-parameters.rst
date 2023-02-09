Now we can proceed with configuring a 'from-scratch' simulated .MS from a model 
you generate (for this example, not a csalt model).  This model should be in 
the form of a FITS cube.  Ideally, this cube is sampled at the native channel 
spacing for exoALMA data, which is about 15 kHz (anything coarser will be less 
accurate).  It should also be sampled on appropriate pixel scales for the 
exoALMA resolution: >5x smaller than the synthesized beam, at least (e.g., 10 
mas is suitable).  The FITS cube should have units of Jy per pixel (area).  

The standard FITS header variables are required: {CDELTi, CRPIXi, CRVALi, 
NAXISi} where i = 1, 2, 3 and i = 1 is RA (in degrees), i = 2 is DEC (in 
degrees), and i = 3 is FREQUENCY (channels, in Hz).  



And then also here I can subdivide by code (I will only be able to do mcfost stuff) and list the parameters that I am fitting for?
