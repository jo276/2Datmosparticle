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

#### star properties
Tstar = 5777.
Lstar = 3.83e33

#### planet properties
beta_actual = 9.17 ## beta of actual planets at 0.1microns with silicates
a_actual = 0.03*1.5e13
Mp = 4.*5.97e27
Rp = 6.*6.371e8



#### Simulation parameters
Nsteps = 20000001# total number of timesteps to run
Ndump = 2500 # output every this number of timesteps # output every this number of timesteps
Nrat = 20 # update radiative transfer this number of time-steps
Short_Fric = False ## whether to use short friction time approx or not

Arad = True
Haze_flux = 1e-14
Kzz = 1e6




beta_want = 3.
a_want = a_actual * np.sqrt(beta_actual/beta_want)
Mdot_actual = 0.#1.7e10
Mdot_use = Mdot_actual * (a_actual/a_want)**2.



Fbol = Lstar / (4. * np.pi * a_actual**2.)

Tequil = (Fbol/4./5.6704e-5)**(0.25)

#### initialise the grid 

gd = grid.grid(1.5e+9,1.5e10,0.01,np.pi-0.01,302,370,1.1)
ry = grid.rays(gd,200,3.)
fd = field.field(gd,1,1.25)
sy = field.system(Mp,Rp,a_want,1e6,Mdot_use,2.35,Tequil)

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

#### source setup
Pstar = 1e-6 * 1e6
sigma_P = 0.5
a_init = 1e-7
stype = 2 ## cloud-like
cloud_width = 0.05

# calculate optical depth for removal of haze production

get_tau_haze = InterpolatedUnivariateSpline(fd.gas_P[::-1,gd.NTH//2+1],ry.tau_b[::-1,gd.NTH//2+1])

tau_haze = get_tau_haze(Pstar + 2.*sigma_P)
fd.par_size[:] = a_init
source_args = (stype,Sdot,Pstar,sigma_P,a_init,tau_haze,cloud_width)


#### Now run code 
# initial dt
dt =5.
start_time = time.time()
if (Arad):
    sim_time, dt = integrator.runner_semi_implicit_numba(0.45,Nsteps,Ndump,Nrat,dt,gd,fd,ry,sy,source_args=source_args,get_Qpr=Qfit.get_Qpr_sil,get_Qext=Qfit.get_Qext_sil)
else:
    sim_time, dt = integrator.runner_semi_implicit_numba(0.45,Nsteps,Ndump,Nrat,dt,gd,fd,ry,sy,source_args=source_args,get_Qpr=Qfit.get_Qpr_none,get_Qext=Qfit.get_Qpr_none)

print("Execution time %s s" % (time.time()-start_time))