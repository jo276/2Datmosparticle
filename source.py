import numpy as np
import math
from numba import jit, int32, float64, prange, types, void

# source update term

def get_source(grid,field,system,args=()):

    # source update using term like Equation 6 of Ormel et al. (2019)

    G = 6.67e-8

    Sigma_dot = args[0]
    Pstar = args[1]
    sigma_P = args[2]
    tdamp = args[3]

    g_r = G * system.Mp / grid.Rb2d**2.

    in_exp = (1./2./sigma_P**2.)*(np.log(field.gas_P/Pstar))**2.

    in_exp [in_exp > 50.] = 50. # prevent underflow

    log_normal = 1./(sigma_P * Pstar * np.sqrt(2.*np.pi)) * np.exp(-in_exp)

    field.par_source[:,:grid.NTH//2+1,0] = (g_r * field.gas_dens * log_normal)[:,:grid.NTH//2+1]
    #field.par_source[:,:,0] = (g_r * field.gas_dens * log_normal)


    field.par_source[:,:,0]*= Sigma_dot

    #field.par_source[-4:-1,:,0] = -field.par_dens[-4:-1,:,0] /  tdamp # damp edges to prevent reflection

    return


def update_source(grid,field,system,dt,args=()):

    # get source term
    get_source(grid,field,system,args=args)

    field.par_dens += field.par_source * dt

    # set particle size in region to 1e-5cm

    a_init = args[3]

    field.par_size[field.par_source > 0.5 * np.amax(field.par_source)] = a_init

    field.par_dens[field.par_dens < 1e-40] = 1e-40 # floor density

    # switch off gr

    return

def update_source_jd(grid,field,rays,system,dt,args=()):

    update_source_numba(grid.ii,grid.io,grid.ji,grid.jo,field.Nparticles,system.Mp,system.mmw,grid.Rb,field.par_source,
                    field.par_size,field.par_dens,field.par_dens_in,field.gas_P,field.gas_dens,field.gas_T,rays.tau_b_gas,
                    field.par_vr_drift,field.par_vth_drift,field.par_ar,field.par_ath,dt,args)

    #a_init = args[3]

    #field.par_size[field.par_source > 0.5 * np.amax(field.par_source)] = a_init

    field.par_dens[field.par_dens < 1e-40] = 1e-40 # floor density


    return

# choose not to use signature here as args size needs to be specified in signature and speed up is small in entire integration
#void(int32,int32,int32,int32,int32,float64,float64[:],float64[:,:,:],float64[:,:,:],float64[:,:,:],float64[:,:],float64[:,:],float64,types.UniTuple(float64,5))
@jit(nopython=True,parallel=True)
def update_source_numba(ii,io,ji,jo,Nparticles,Mp,mmw,Rb,source,ps,pd,pid,gP,gd,gT,tau_b,vrd,vtd,ar,at,dt,args):

    G = 6.67e-8
    kb = 1.38e-16
    mh = 1.67e-24

    Sigma_dot = args[0]
    Pstar = args[1]
    sigma_P = args[2]
    ainsert = args[3]
    tau_haze = args[4]

    sqrt_2pi = math.sqrt(2.*math.pi)

    for i in prange(ii,io+1):
        g_r = G * Mp / Rb[i]**2.
        for j in range(ji,jo+1):
            in_exp = (1./2./sigma_P**2.)*(math.log(gP[i,j]/Pstar))**2.

            vth = math.sqrt(8. * kb * gT[i,j]/ (math.pi * mmw * mh))

            if in_exp > 50:
                in_exp = 50. # prevent underflow

            log_normal = 1./(sigma_P * Pstar * sqrt_2pi) * math.exp(-in_exp)
            for k in range(Nparticles):

                source[i,j,k] = (g_r * gd[i,j] * log_normal) * Sigma_dot * math.exp(-tau_b[i,j]/tau_haze)

                # add pertubation to source term
                #source[i,j,k] *= 10.**np.random.normal(loc=0,scale=1.)

                drho = source[i,j,k] * dt

                # drho is the density of small particles added over the time-step
                # we want to adjust the size based on this 

                # first thing we need to do is work out whether a newly inserted particle is more likely to collide
                # with another small particle or a big particle
                m_par_small = 4./3. * math.pi * pid[i,j,k] * ainsert**3.
                mpar = 4./3. * math.pi *pid[i,j,k] * ps[i,j,k]**3. # mass of big particle


                l_mfp_small = m_par_small / (math.sqrt(2.)*drho*(2.*ainsert)**2.)
                l_mfp_big   = mpar / (pd[i,j,k]*(math.pi * ps[i,j,k]**2.))

                if (l_mfp_small < l_mfp_big):
                    # more likely to collide with small particle so small particles grow together
                    accl_average = math.sqrt(ar[i,j,k]**2.+at[i,j,k]**2.)

                    tgrow = 4./3. * vth * gd[i,j]/(drho * accl_average) # this is the growth time of just the small particles introduced assuming short-friction time
                    asmall_grow = ainsert * math.exp(dt/tgrow) # size the small particles have grown to over the timestep

                    if (dt/tgrow>2.):
                        # flag error
                        print("Warnning timestep over growth timescale of small particles too large in source update")

                    # now do a mass weighted average between asmall and abig
                    anew = (pd[i,j,k]*ps[i,j,k] + drho*asmall_grow) / (pd[i,j,k]+drho) 

                else:
                    # more likely to collide with big particle

                    # calculate the size the big particles grow over timestep
                    macc_per_particles = math.pi * ps[i,j,k]**2. * math.sqrt(vrd[i,j,k]**2.+vtd[i,j,k]**2.) * drho * dt
                
                    abig_grow = ps[i,j,k]*(1. + 1./3.*macc_per_particles/mpar) # size the big particles grow to by accreting all the small particles  

                    # now check abig_grow is not bigger than if all of the small particles are added to the big particles over the timestep
                    Npar = pd[i,j,k] / mpar

                    mpar_new_all = (pd[i,j,k]+drho)/Npar
                    abig_all = (mpar_new_all / (4./3.*math.pi*pid[i,j,k]))**(1./3.)

                    abig = min(abig_grow,abig_all)

                    # small particles just swept-up by big
                    anew = abig
                

                ps[i,j,k] = anew

                pd[i,j,k] += drho



    return