import os, sys

# run CASA script to process data into modeling formats
os.system('rm -rf CASA_logs/format_data_'+sys.argv[-1]+'.log')
os.system('casa --nologger --logfile CASA_logs/format_data_'+ \
          sys.argv[-1]+'.log -c CASA_scripts/format_data.py '+sys.argv[-1])
