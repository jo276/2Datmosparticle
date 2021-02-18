### This deploys a parameter study on HPC

import os
import numpy as np

no_arad = True
runner_template = "runner_HJ2.py"


Kzz_grid = np.logspace(5.,10.,6)
Haze_flux_grid = np.logspace(-16.,-10.,7)


chdir = os.getcwd()

for i in range(np.size(Kzz_grid)):
    for j in range(np.size(Haze_flux_grid)):

        wk_dir = "run_dir%d_%d"%(i,j)

        os.mkdir(wk_dir)
        os.chdir(wk_dir)
        os.system("git clone https://github.com/jo276/2Datmosparticle.git")
        os.chdir("2Datmosparticle")
        
        # now re-write runner file
        f =open(runner_template,'r')
        filedata = f.read()
        f.close()

        newfiledata = filedata.replace("Haze_flux = 1e-13","Haze_flux = %e" %(Haze_flux_grid[j]))
        newfiledata = newfiledata.replace("Kzz = 1e6","Kzz = %e" %(Kzz_grid[i]))

        if (no_arad):
            newfiledata = newfiledata.replace("Arad = True","Arad = False")

        f = open("runner_use.py",'w')
        f.write(newfiledata)
        f.close()

        os.system("qsub submit.sh")
            
        os.chdir(chdir)




