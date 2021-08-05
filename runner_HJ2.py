#### THIS IS THE RUNNER FILE FOR A less MASSIVE HJ 0.5 Mjup 1.4Rjup

import numpy as np
import grid as grid
import field as field
import particle_updates as pu
import boundary as bd
import integrator as integrator
import source as sc
import analysis as an
import Q_fits as Qfit
import time

from scipy.interpolate import InterpolatedUnivariateSpline


#### Simulation parameters
Nsteps = 20000001 # total number of timesteps to run
Ndump = 2500 # output every this number of timesteps
Nrat = 20 # update radiative transfer this number of time-steps
Short_Fric = False ## whether to use short friction time approx or not

Arad = True
Haze_flux = 1e-14
Kzz = 1e6

Tstar = 5777.



#### initialise the grid 

gd = grid.grid(9.9e9,1.181e10,0.01,np.pi-0.01,225,2000,2.)
ry = grid.rays(gd,400,3.)
fd = field.field(gd,1,1.25)
sy = field.system(9.5e29,3.83e33,0.03*1.5e13,5e6,0.,2.35)

fd.setup_iso_atm(sy,gd,True)

fd.short_friction = Short_Fric

fd.Tstar = Tstar

sy.kappa_star = 4e-3 ### value from Guillot et al. (2010)

ry.get_tau_grid_analytic(gd,sy)

#### Below is for testing ray-tracing scheme on analytic gas profile
#ry.do_ray_trace(fd.gas_dens*sy.kappa_star) ### using analytic gas optical depth calculation
#pu.update_tau_b_gas(fd,gd,ry,sy)
### Now initialise the initial conditions

fd.par_K[:] = Kzz

Sdot = Haze_flux

fd.par_dens[:,:,0] = 1e-40

fd.gas_vth[1:-1,:] = 0.

bd.update_boundary(gd,fd)

kappa_bol = sy.kappa_star


Pstar = 1e-6 * 1e6
sigma_P = 0.5
a_init = 1e-7

# calculate optical depth for removal of haze production

get_tau_haze = InterpolatedUnivariateSpline(fd.gas_P[::-1,gd.NTH//2+1],ry.tau_b[::-1,gd.NTH//2+1])

tau_haze = get_tau_haze(Pstar + 2.*sigma_P)

fd.par_size[:] = a_init
source_args = (Sdot,Pstar,sigma_P,a_init,tau_haze)


#### Now run code 
# initial dt
dt =5.
start_time = time.time()
if (Arad):
    sim_time, dt = integrator.runner_semi_implicit_numba(0.45,Nsteps,Ndump,Nrat,dt,gd,fd,ry,sy,source_args=source_args,get_Qpr=Qfit.get_Qpr_soot,get_Qext=Qfit.get_Qext_soot)
else:
    sim_time, dt = integrator.runner_semi_implicit_numba(0.45,Nsteps,Ndump,Nrat,dt,gd,fd,ry,sy,source_args=source_args,get_Qpr=Qfit.get_Qpr_none,get_Qext=Qfit.get_Qpr_none)

print("Execution time %s s" % (time.time()-start_time))