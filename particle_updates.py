import numpy as np
from numba import jit, prange, void, float64, types, int32
import grid as gd 
import math as math
import numexpr as ne
#### written by JO Dec 2020 ###'#
#### This contains a spherical finite volume grid 
#### This grid is based on the staggered grids, ala zeus (Stone & Norman 1992)
#### The grid will include one ghost cell (2nd order method)
#### This file contains routines to update the particles with advection and diffusion

#### Each routine contains two verisions as pure numpy one for debugging and small grids and
#### a parallel verision using numba, will only be much faster for large grids, the interface
#### has a _jd name and the heavy lifting is done through the _numba file


##### functions defined to calculate flux limiters

@jit(float64(float64,float64),nopython=True)
def minmod(a,b):

    if (a*b>0.):
        if np.fabs(b) > np.fabs(a):
            ans = a
        else:
            ans = b
    else: 
        ans = 0. 

    return ans

@jit(float64(float64,float64),nopython =True)
def maxmod(a,b):
    if (a*b>0.):
        if np.fabs(b) > np.fabs(a):
            ans = b
        else:
            ans = a
    else: 
        ans = 0. 

    return ans


#### advection update
def advection_update(grid,field,dt):

    # 2nd order explicit advection update use unsplit update

    field.par_dens_star_r, field.par_dens_star_th = get_qstar(grid.ii,grid.io,grid.ji,grid.jo,
                                        grid.NR,grid.NTH,field.Nparticles,field.par_dens,
                                        field.par_vr,field.par_vth,grid.dRb,grid.dTb,
                                        grid.dRa,grid.dTa,grid.g2b,dt)

    # now calculate the mass fluxes

    field.M_r = field.par_dens_star_r[:,:-1,:]  * field.par_vr 
    field.M_t = field.par_dens_star_th[:-1,:,:] * field.par_vth

    field.M_r3D = (field.M_r.T * grid.A_r[:,:-1].T).T
    field.M_t3D = (field.M_t.T * grid.A_th[:-1,:].T).T

    delta_F_r  = (field.M_r3D[grid.ii:grid.io+1,grid.ji:grid.jo+1]-field.M_r3D[grid.ii+1:grid.io+2,grid.ji:grid.jo+1])
    delta_F_t  = (field.M_t3D[grid.ii:grid.io+1,grid.ji:grid.jo+1]-field.M_t3D[grid.ii:grid.io+1,grid.ji+1:grid.jo+2])

    div_F  = (delta_F_r.T * (np.outer(1./(grid.dvRa[grid.ii:grid.io+1]),np.ones(grid.NTH))).T).T
    div_F += (delta_F_t.T * (np.outer(1./(grid.dvRa[grid.ii:grid.io+1]),1./(grid.dvTa[grid.ji:grid.jo+1]))).T).T


    # do advection update
    field.par_dens[grid.ii:grid.io+1,grid.ji:grid.jo+1] += div_F * dt

    return

def advection_update_jd(grid,field,dt):

    # 2nd order explicit advection update use unsplit update

    field.par_dens_star_r, field.par_dens_star_th = get_qstar(grid.ii,grid.io,grid.ji,grid.jo,
                                        grid.NR,grid.NTH,field.Nparticles,field.par_dens,
                                        field.par_vr,field.par_vth,grid.dRb,grid.dTb,
                                        grid.dRa,grid.dTa,grid.g2b,dt)

    advection_flux_update_jit(grid.ii,grid.io,grid.ji,grid.jo,field.Nparticles,grid.dvRa,grid.dvTa,grid.A_r,grid.A_th,
                field.par_dens_star_r,field.par_dens_star_th,field.par_dens,field.par_vr,field.par_vth,dt)


@jit(void(int32,int32,int32,int32,int32,float64[:],float64[:],float64[:,:],float64[:,:],float64[:,:,:],float64[:,:,:],float64[:,:,:],float64[:,:,:],float64[:,:,:],float64),nopython=True,parallel=True)
def advection_flux_update_jit(ii,io,ji,jo,Nparticles,dvRa,dvTa,Ar,At,dstarr,dstarth,pd,vr,vth,dt):
    # fast verision of flux update
    # get fluxes on a grid
    M_r3D = np.zeros((io+2,jo+2,Nparticles))
    M_t3D = np.zeros((io+2,jo+2,Nparticles))
    for i in prange(ii,io+2):
        for j in range(ji,jo+2):
            for k in range(Nparticles):
                    M_r3D[i,j,k] = dstarr[i,j,k] * vr[i,j,k] * Ar[i,j]
                    M_t3D[i,j,k] = dstarth[i,j,k]*vth[i,j,k] * At[i,j]


    # now update particle density
    for i in prange(ii,io+1):
        for j in range(ji,jo+1):
            for k in range(Nparticles):
                delta_Fr = M_r3D[i,j,k] - M_r3D[i+1,j,k]
                delta_Ft = M_t3D[i,j,k] - M_t3D[i,j+1,k]

                div_F =  delta_Fr/dvRa[i]  + delta_Ft/(dvRa[i]*dvTa[j])

                pd[i,j,k] += div_F * dt

    return

#### calculate upwinded densities
@jit(types.UniTuple(float64[:,:,:],2)(int32, int32, int32, int32, int32, int32, int32,float64[:,:,:],float64[:,:,:],float64[:,:,:],float64[:],float64[:],float64[:],float64[:],float64[:],float64),nopython = True,parallel=True)
def get_qstar(ii,io,ji,jo,NR,NTH,Nparticles,q,ur_par,uth_par,dRb,dTb,dRa,dTa,g2b,dt):

    dq  = np.zeros((NR+3,NTH+3,Nparticles))
    qr  = np.zeros((NR+3,NTH+3,Nparticles))
    qth = np.zeros((NR+3,NTH+3,Nparticles))
    # do r first
    # limiter superbee
    for i in prange(ii,io+2):
        for j in range(ji,jo+1):
            for k in range(Nparticles):
                if (i==io+1):
                    Dq_p = 0. # force first-order in boundary
                else:
                    Dq_p = (q[i+1,j,k]-q[i,j,k])/dRb[i+1]
                Dq_m = (q[i,j,k]-q[i-1,j,k])/dRb[i]
                
                sigma1 = minmod(Dq_p,2.*Dq_m)
                sigma2 = minmod(2.*Dq_p,Dq_m)

                dq[i,j,k] = maxmod(sigma1,sigma2)


    # calculate limiter - vanleer
    #for i in range(ii,io+2):
    #    for j in range(ji,jo+1):
    #        for k in range(Nparticles):
    #            if (i==io+1):
    #                Dq_p = 0. # force first-order in boundary
    #            else:
    #                Dq_p = (q[i+1,j,k]-q[i,j,k])/dRb[i+1]
     #           Dq_m = (q[i,j,k]-q[i-1,j,k])/dRb[i]
     #           if (Dq_p*Dq_m > 0.):
     #               dq[i,j,k] = 2. * (Dq_p*Dq_m) / (Dq_m+Dq_p)
     #           else:
      #              dq[i,j,k] = 0.
    

    # get fluxes
    for i in prange(ii,io+2):
        for j in range(ji,jo+1):
            for k in range(Nparticles):
                if (ur_par[i,j,k] >=0.):
                    if ( i > io):
                        # use donnor cell at boundary
                        qr[i,j,k] = q[i-1,j,k]
                    else:
                        qr[i,j,k] = q[i-1,j,k] + (dRa[i-1]-ur_par[i,j,k]*dt)*dq[i-1,j,k]/2.
                else:
                    #qr[i,j,k] = q[i,j,k]
                    qr[i,j,k] = q[i,j,k] - (dRa[i]+ur_par[i,j,k]*dt)*dq[i,j,k]/2.

    # do th now
    # calculate limiter - super-bee
    for i in prange(ii,io+1):
        for j in range(ji,jo+2):
            for k in range(Nparticles):
                if (j==jo+1):
                    Dq_p = 0. # force first-order in boundary
                else:
                    Dq_p = (q[i,j+1,k]-q[i,j,k])/dTb[j+1]
                Dq_m = (q[i,j,k]-q[i,j-1,k])/dTb[j]

                sigma1 = minmod(Dq_p,2.*Dq_m)
                sigma2 = minmod(2.*Dq_p,Dq_m)

                dq[i,j,k] = maxmod(sigma1,sigma2)
                

    # calculate limiter - van-leer
    #for i in range(ii,io+1):
    #    for j in range(ji,jo+2):
    #        for k in range(Nparticles):
    #            if (j==jo+1):
    #                Dq_p = 0. # force first-order in boundary
    #            else:
    #                Dq_p = (q[i,j+1,k]-q[i,j,k])/dTb[j+1]
    #            Dq_m = (q[i,j,k]-q[i,j-1,k])/dTb[j]
    #            if (Dq_p*Dq_m > 0.):
    #                dq[i,j,k] = 2. * (Dq_p*Dq_m) / (Dq_m+Dq_p)
    #            else:
    #                dq[i,j,k] = 0.

    # get fluxes
    for i in prange(ii,io+1):
        for j in range(ji,jo+2):
            for k in range(Nparticles):
                if (uth_par[i,j,k] >=0.):
                    if (i > io-1):
                        qth[i,j,k] = q[i,j-1,k]
                    else:
                        qth[i,j,k] = q[i,j-1,k] + (dTa[j-1]-uth_par[i,j,k]*dt/g2b[i])*dq[i,j-1,k]/2.

                else:
                    if (i > io-1):
                        qth[i,j,k] = q[i,j,k]
                    else:
                        qth[i,j,k] = q[i,j,k] - (dTa[j]+uth_par[i,j,k]*dt/g2b[i])*dq[i,j,k]/2.

    return qr,qth


def diffusion_update(grid,field,dt):
    # we use a finite volume diffusion update
    # find concentration
    field.Conc = (field.par_dens.T * (1./field.gas_dens).T).T

    # find diffusion constant
    diffusion_bgrid = (field.par_K.T * field.gas_dens.T).T 
    
    field.div_Fdiff = get_diff_F(field.Conc,diffusion_bgrid,grid.g2a,grid.g31a,grid.dvRa,grid.dvRb,
        grid.g32a,grid.g2b,grid.dvTa,grid.dvTb,grid.NR,grid.NTH,grid.ii,grid.io,grid.ji,grid.jo,field.Nparticles)

    # update due to diffusion term

    field.par_dens[grid.ii:grid.io+1,grid.ji:grid.jo+1] -= field.div_Fdiff * dt
        

def diffusion_update_jd(grid,field,dt):

    diffusion_update_numba(field.Conc,field.par_diff_b,field.div_Fdiff_numba,field.par_dens,field.gas_dens,field.par_K,
                grid.g2a,grid.g31a,grid.dvRa,grid.dvRb, grid.g32a,grid.g2b,grid.dvTa,grid.dvTb,
                grid.ii,grid.io,grid.ji,grid.jo,field.Nparticles,dt)

    return 

@jit(void(float64[:,:,:],float64[:,:,:],float64[:,:,:],float64[:,:,:],float64[:,:],float64[:,:,:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],int32,int32,int32,int32,int32,float64),nopython=True,parallel=True)
def diffusion_update_numba(Conc,diff,diff_F,pd,gd,pK,g2a,g31a,dvRa,dvRb,g32a,g2b,dvTa,dvTb,ii,io,ji,jo,Nparticles,dt):

    # first calculate concentration and diffusion constant on b grid
    for i in prange(ii-1,io+2):
        for j in range(ji-1,jo+2):
            for k in range(Nparticles):
                Conc[i,j,k] = pd[i,j,k]/gd[i,j]
                diff[i,j,k] = pK[i,j,k] * gd[i,j]

    for i in prange(ii,io+1):
        for j in range(ji,jo+1):
            for k in range(Nparticles):
                diff1_ip1 = (diff[i+1,j,k] + diff[i,j,k])/2.
                diff1_im1 = (diff[i,j,k] + diff[i-1,j,k])/2. 

                diff2_jp1 = (diff[i,j+1,k] + diff[i,j,k])/2.
                diff2_jm1 = (diff[i,j,k] + diff[i,j-1,k])/2.

                diff_F[i,j,k] =  ( - (g2a[i+1]*g31a[i+1])**2./dvRa[i]*diff1_ip1*(Conc[i+1,j,k]-Conc[i,j,k])/dvRb[i+1] 
                              +(g2a[i]*g31a[i])**2./dvRa[i]*diff1_im1*(Conc[i,j,k]-Conc[i-1,j,k])/dvRb[i] 
                              -(g32a[j+1])**2./(g2b[i]**2.*dvTa[j])*diff2_jp1*(Conc[i,j+1,k]-Conc[i,j,k])/dvTb[j+1]
                              +(g32a[j])**2./(g2b[i]**2.*dvTa[j])*diff2_jm1*(Conc[i,j,k]-Conc[i,j-1,k])/dvTb[j] )

                pd[i,j,k] -= diff_F[i,j,k] * dt

    return


@jit(nopython=True)
def get_diff_F(Conc,diff,g2a,g31a,dvRa,dvRb,g32a,g2b,dvTa,dvTb,NR,NTH,ii,io,ji,jo,Nparticles):

    diff_F = np.zeros((NR+2,NTH+2,Nparticles))

    for i in range(ii,io+1):
        for j in range(ji,jo+1):
            for k in range(Nparticles):
                diff1_ip1 = (diff[i+1,j,k] + diff[i,j,k])/2.
                diff1_im1 = (diff[i,j,k] + diff[i-1,j,k])/2. 

                diff2_jp1 = (diff[i,j+1,k] + diff[i,j,k])/2.
                diff2_jm1 = (diff[i,j,k] + diff[i,j-1,k])/2.

                diff_F[i,j,k] =  ( - (g2a[i+1]*g31a[i+1])**2./dvRa[i]*diff1_ip1*(Conc[i+1,j,k]-Conc[i,j,k])/dvRb[i+1] 
                              +(g2a[i]*g31a[i])**2./dvRa[i]*diff1_im1*(Conc[i,j,k]-Conc[i-1,j,k])/dvRb[i] 
                              -(g32a[j+1])**2./(g2b[i]**2.*dvTa[j])*diff2_jp1*(Conc[i,j+1,k]-Conc[i,j,k])/dvTb[j+1]
                              +(g32a[j])**2./(g2b[i]**2.*dvTa[j])*diff2_jm1*(Conc[i,j,k]-Conc[i,j-1,k])/dvTb[j] )


    return diff_F[ii:io+1,ji:jo+1,:]


def get_timestep(CFL,grid,field):

    # calculates the timestep

    # advection timestep:
    dt_adv_r = 1e50
    dt_adv_t = 1e50
    dt_diff_r = 1e50
    dt_diff_t = 1e50
    dt_grow = 1e50

    dis_r_2d = np.outer(grid.dRa[grid.ii:grid.io+2],np.ones(grid.NTH))
    dis_th_2d = np.outer(grid.Rb[grid.ii:grid.io+1],grid.dTa[grid.ji:grid.jo+2])

    dt_r_3d = (dis_r_2d.T/(field.par_vr[grid.ii:grid.io+2,grid.ji:grid.jo+1]+1e-50).T).T 
    dt_t_3d = (dis_th_2d.T/(field.par_vth[grid.ii:grid.io+1,grid.ji:grid.jo+2]+1e-50).T).T

    dt_adv_r = np.amin(np.fabs(dt_r_3d))
    dt_adv_t = np.amin(np.fabs(dt_t_3d))

    # diffusion timestep - assumes spherically symmetirc K distrubtion
    dt_diff_r = np.amin(0.25 * (grid.dRb[grid.ii:grid.io+1]**2./(field.par_K[grid.ii:grid.io+1,grid.ji,0])))
    dt_diff_t = np.amin(0.25 * ((grid.dTb[grid.ji:grid.jo+1]*grid.Rb[grid.ii])**2./(field.par_K[grid.ii,grid.ji:grid.jo+1,0]))) 

    dt_grow = 0.1*np.amin(field.par_tgrow[grid.ii:grid.io+1,grid.ji:grid.jo+1,:])

    # add in quadrature
    dt = CFL * 1./np.sqrt(dt_adv_r**(-2.)+dt_adv_t**(-2.) + dt_diff_r**(-2.)+dt_diff_t**(-2.)+dt_grow**(-2.))

    return dt

def get_timestep_jd(CFL,grid,field):

    dt = get_timestep_numba(CFL,grid.ii,grid.io,grid.ji,grid.jo,field.Nparticles,grid.dRa,grid.dTa,grid.dRb,grid.dTb,grid.Rb,
                    field.par_vr,field.par_vth,field.par_K,field.par_tgrow)

    return dt
@jit(float64(float64,int32,int32,int32,int32,int32,float64[:],float64[:],float64[:],float64[:],float64[:],float64[:,:,:],float64[:,:,:],float64[:,:,:],float64[:,:,:]),nopython=True,parallel=True)
def get_timestep_numba(CFL,ii,io,ji,jo,Nparticles,dRa,dTa,dRb,dTb,Rb,vr,vth,pK,tgrow):

    # calculates the timestep

    # advection timestep:
    dt_adv_r = 1e50
    dt_adv_t = 1e50
    dt_diff_r = 1e50
    dt_diff_t = 1e50
    dt_grow = 1e50

    # loop through grid and calculate all timesteps
    for i in prange(ii,io+2):
        for j in range(ji,jo+2):
            for k in range(Nparticles):

                dis_r_2d = dRa[i]
                dis_th_2d = Rb[i]*dTa[j]

                dt_r_3d = math.fabs(dis_r_2d/(vr[i,j,k]+1e-50)) 
                dt_t_3d = math.fabs(dis_th_2d/(vth[i,j,k]+1e-50))

                dt_adv_r = min(dt_adv_r,dt_r_3d)
                dt_adv_t = min(dt_adv_t,dt_t_3d)

                 # diffusion timestep - assumes spherically symmetirc K distrubtion
                dt_diff_r = min(dt_diff_r,0.25 * (dRb[i]**2./(pK[i,j,k])))
                dt_diff_t = min(dt_diff_t,0.25 * ((dTb[j]*Rb[i])**2./(pK[i,j,k]))) 

                dt_grow = min(dt_grow,0.1*tgrow[i,j,k])

    
                
    # add in quadrature
    dt = CFL * 1./math.sqrt(dt_adv_r**(-2.)+dt_adv_t**(-2.) + dt_diff_r**(-2.)+dt_diff_t**(-2.)+dt_grow**(-2.))


    return dt



def get_par_acc(grid,field,rays,system,getQ):

    # this routine calculates the accleration of the particles 
    G = 6.67e-8
    clight = 2.9979e10

    field.Q = getQ(field.par_size + 1e-10)

    # zero previous accleration
    field.par_ar[:] = 0.
    field.par_ath[:] = 0.
    
    # spherical gravity first
    field.par_ar = ((field.par_ar).T +(- G * system.Mp / grid.Ra2d[:,grid.ji-1:grid.jo+2]**2.).T).T

    # radiation pressure
    field.kappa = 0.75 * Q / (field.par_dens_in * field.par_size)

    # radiation accleraton is in minus z direction
    field.Fstar_b = system.Fbol * np.exp(-(rays.tau_b)) # stellar flux on b grid
    arad_minus_z_rw = -0.5*(((field.kappa.T*field.Fstar_b.T).T)[grid.ii-1:grid.io+1,:,:]+((field.kappa.T*field.Fstar_b.T).T)[grid.ii:grid.io+2,:,:])/clight
    arad_minus_z_tw = -0.5*(((field.kappa.T*field.Fstar_b.T).T)[:,grid.ji-1:grid.jo+1,:]+((field.kappa.T*field.Fstar_b.T).T)[:,grid.ji:grid.jo+2:,:])/clight


    # now project onto r and theta directions
    field.par_ar[grid.ii:grid.io+2,grid.ji:grid.jo+1,:] +=  (np.cos(grid.Tb2d[grid.ii:grid.io+2,grid.ji:grid.jo+1]).T * arad_minus_z_rw[:,grid.ji:grid.jo+1,:].T).T
    field.par_ath[grid.ii:grid.io+1,grid.ji:grid.jo+2,:]+= -(np.sin(grid.Ta2d[grid.ii:grid.io+1,grid.ji:grid.jo+2]).T * arad_minus_z_tw[grid.ii:grid.io+1,:,:].T).T



    return

def get_par_acc_jd(grid,field,rays,system,getQ):

    #field.Q[:] = 1. 
    field.Q = getQ(field.par_size + 1e-10)

    get_par_acc_numba(grid.ii,grid.io,grid.ji,grid.jo,field.Nparticles,system.Mp,system.Fbol,field.Fstar_b,field.Q,field.par_dens_in,field.par_size,rays.tau_b,
                field.par_ar,field.par_ath,field.kappa,grid.Ra,grid.Tb,grid.Ta)

    return

@jit(void(int32,int32,int32,int32,int32,float64,float64,float64[:,:],float64[:,:,:],float64[:,:,:],float64[:,:,:],float64[:,:],float64[:,:,:],float64[:,:,:],float64[:,:,:],float64[:],float64[:],float64[:]),nopython=True,parallel=True)
def get_par_acc_numba(ii,io,ji,jo,Nparticles,Mp,Fbol,Fstar_b,Q,pid,ps,tau,ar,ath,kappa,Ra,Tb,Ta):

    G = 6.67e-8
    clight = 2.9979e10
    # compute kappa and attenuated flux on b grid
    for i in prange(ii-1,io+2):
        for j in range(ji-1,jo+2):
            Fstar_b[i,j] = Fbol * math.exp(-(tau[i,j]))
            for k in range(Nparticles):
                kappa[i,j,k] = 0.75 * Q[i,j,k] / (pid[i,j,k] * ps[i,j,k])

    # compute accleration (radiation pressure in in -ve Z-direction by defn)
    for i in prange(ii,io+2):
        for j in range(ji,jo+2):
            for k in range(Nparticles):
                # spherical gravity first
                ar[i,j,k] = (- G * Mp / Ra[i]**2.)
                # now radiation pressure
                arad_minus_z_rw = -0.5*(((kappa[i-1,j,k]*Fstar_b[i-1,j]))+((kappa[i,j,k]*Fstar_b[i,j])))/clight
                arad_minus_z_tw = -0.5*(((kappa[i,j-1,k]*Fstar_b[i,j-1]))+((kappa[i,j,k]*Fstar_b[i,j])))/clight
                
                # now project onto r and theta directions
                ar[i,j,k] +=  (math.cos(Tb[j]) * arad_minus_z_rw)
                ath[i,j,k] = -(math.sin(Ta[j]) * arad_minus_z_tw)


    return


def get_par_vel_terminal_velocity(grid,field,system):

    # this routine calculates the velocity of the particles assuming the terminal velocity approx

    # terminal velocity solution ==> v_p = v_g + tstop * a_ext

    # first calculate tstop

    kb = 1.38e-16
    mh = 1.67e-24

    cs_gas = np.sqrt(kb * field.gas_T / (mh * system.mmw))

    tstop_b = ((field.par_dens_in * field.par_size).T * (1./(field.gas_dens * cs_gas)).T).T 
    # now adjust to reduce velocity in cells with close to floor density
    field.tstop_bgrid = tstop_b

    # calculate vp on a grid
    field.par_vr_drift[grid.ii:grid.io+2,grid.ji:grid.jo+1,:] = ((field.tstop_bgrid[grid.ii-1:grid.io+1,grid.ji:grid.jo+1,:]+field.tstop_bgrid[grid.ii:grid.io+2,grid.ji:grid.jo+1,:])/2. *
                field.par_ar[grid.ii:grid.io+2,grid.ji:grid.jo+1,:])

    field.par_vr = ((field.par_vr_drift).T+(field.gas_vr).T).T

    field.par_vth_drift[grid.ii:grid.io+1,grid.ji:grid.jo+2,:] = ((field.tstop_bgrid[grid.ii:grid.io+1,grid.ji-1:grid.jo+1,:]+field.tstop_bgrid[grid.ii:grid.io+1,grid.ji:grid.jo+2,:])/2. *
                field.par_ath[grid.ii:grid.io+1,grid.ji:grid.jo+2,:])

    field.par_vth = ((field.par_vth_drift).T+(field.gas_vth).T).T

    # now call diffusive velocities and add

    #get_par_vel_diff(grid,field)

    #field.par_vr += field.par_vr_diff
    #field.par_vth+= field.par_vt_diff


    return

def get_velocity_semi_implicit(grid,field,system,dt):

    # this uses the semi-implict update of Rosotti et al. (2016)

    # first calculate tstop

    kb = 1.38e-16
    mh = 1.67e-24

    cs_gas = np.sqrt(kb * field.gas_T / (mh * system.mmw))

    tstop_b = ((field.par_dens_in * field.par_size).T * (1./(field.gas_dens * cs_gas)).T).T 
    # now adjust to reduce velocity in cells with close to floor density
    field.tstop_bgrid = tstop_b

    implicit_factor_r = 1.- np.exp(-dt/((field.tstop_bgrid[grid.ii-1:grid.io+1,grid.ji:grid.jo+1,:]+field.tstop_bgrid[grid.ii:grid.io+2,grid.ji:grid.jo+1,:])/2.))
    implicit_factor_t = 1.- np.exp(-dt/((field.tstop_bgrid[grid.ii:grid.io+1,grid.ji-1:grid.jo+1,:]+field.tstop_bgrid[grid.ii:grid.io+1,grid.ji:grid.jo+2,:])/2.))

    # calculate vp on a grid
    field.par_vr_drift[grid.ii:grid.io+2,grid.ji:grid.jo+1,:] = ((field.tstop_bgrid[grid.ii-1:grid.io+1,grid.ji:grid.jo+1,:]+field.tstop_bgrid[grid.ii:grid.io+2,grid.ji:grid.jo+1,:])/2. *
                field.par_ar[grid.ii:grid.io+2,grid.ji:grid.jo+1,:])

    field.par_vr[grid.ii:grid.io+2,grid.ji:grid.jo+1,:] = (((field.par_vr_drift).T+(field.gas_vr).T).T)[grid.ii:grid.io+2,grid.ji:grid.jo+1,:] * implicit_factor_r

    field.par_vth_drift[grid.ii:grid.io+1,grid.ji:grid.jo+2,:] = ((field.tstop_bgrid[grid.ii:grid.io+1,grid.ji-1:grid.jo+1,:]+field.tstop_bgrid[grid.ii:grid.io+1,grid.ji:grid.jo+2,:])/2. *
                field.par_ath[grid.ii:grid.io+1,grid.ji:grid.jo+2,:])

    field.par_vth[grid.ii:grid.io+1,grid.ji:grid.jo+2,:] = (((field.par_vth_drift).T+(field.gas_vth).T).T)[grid.ii:grid.io+1,grid.ji:grid.jo+2,:] * implicit_factor_t

    return 

def get_velocity_semi_implicit_jd(grid,field,system,dt):

    get_velocity_semi_implicit_numba(grid.ii,grid.io,grid.ji,grid.jo,field.Nparticles,system.mmw,field.gas_T,field.gas_dens,field.par_dens_in,
                    field.par_size,field.tstop_bgrid,field.par_ar,field.par_ath,field.par_vr_drift,field.par_vth_drift,field.par_vr,field.par_vth,
                    field.gas_vr,field.gas_vth,dt)

    return

@jit(void(int32,int32,int32,int32,int32,float64,float64[:,:],float64[:,:],float64[:,:,:],float64[:,:,:],float64[:,:,:],float64[:,:,:],float64[:,:,:],float64[:,:,:],float64[:,:,:],float64[:,:,:],float64[:,:,:],float64[:,:],float64[:,:],float64),nopython=True,parallel=True)
def get_velocity_semi_implicit_numba(ii,io,ji,jo,Nparticles,mmw,gT,gd,pid,ps,tstop,ar,ath,vrdrift,vtdrift,vr,vth,gvr,gvth,dt):

    # looped verision of semi-implict velocity update
    kb = 1.38e-16
    mh = 1.67e-24

    for i in prange(ii-1,io+2):
        for j in range(ji-1,jo+2):
            cs_gas = math.sqrt(kb * gT[i,j] / (mh * mmw))
            for k in range(Nparticles):
                tstop[i,j,k] = ((pid[i,j,k] * ps[i,j,k]) * (1./(gd[i,j] * cs_gas)))

    for i in prange(ii,io+2):
        for j in range(ji,jo+2):
            for k in range(Nparticles):
                tstop_ar = (tstop[i-1,j,k]+tstop[i,j,k])/2.
                tstop_at = (tstop[i,j-1,k]+tstop[i,j,k])/2.
                implicit_factor_r = 1.- math.exp(-dt/tstop_ar)
                implicit_factor_t = 1.- math.exp(-dt/tstop_at)

                # calculate vp on a grid
                vrdrift_store = tstop_ar * ar[i,j,k] 
                vtdrift_store = tstop_at * ath[i,j,k]

                vr[i,j,k] = (vrdrift_store + gvr[i,j]) * implicit_factor_r
                vrdrift[i,j,k] = vr[i,j,k] -gvr[i,j]

                vth[i,j,k] = (vtdrift_store + gvth[i,j]) * implicit_factor_t
                vtdrift[i,j,k] = vth[i,j,k] - gvth[i,j]


    return
                
     





def get_par_vel_diff(grid,field):

    # calculate the diffusive velocity
    # vdiff = - Diff * (1/conc) * grad (Conc)

    #field.Conc = ((field.par_dens).T*(1./field.gas_dens).T).T

    field.Conc,field.grad_Conc_R, field.grad_Conc_T = get_grad_conc(field.par_dens,field.gas_dens,grid.g2a,grid.g31a,grid.dvRb,grid.g32a,grid.g2b,grid.dvTb,grid.NR,grid.NTH,field.Nparticles,grid.ii,grid.io,grid.ji,grid.jo)

    field.par_vr_diff[grid.ii:grid.io+2,grid.ji:grid.jo+1,:] = - ((field.par_K)[grid.ii-1:grid.io+1,grid.ji:grid.jo+1,:] +
                    (field.par_K)[grid.ii:grid.io+2,grid.ji:grid.jo+1,:])/2.  * field.grad_Conc_R[grid.ii:grid.io+2,grid.ji:grid.jo+1,:]

    field.par_vt_diff[grid.ii:grid.io+1,grid.ji:grid.jo+2,:] = - ((field.par_K)[grid.ii:grid.io+1,grid.ji-1:grid.jo+1,:] +
                    (field.par_K)[grid.ii:grid.io+1,grid.ji:grid.jo+2,:])/2.  * field.grad_Conc_T[grid.ii:grid.io+1,grid.ji:grid.jo+2,:]


def get_par_vel_diff_jd(grid,field):

    field.Conc,field.grad_lgConc_R, field.grad_lgConc_T = get_grad_conc(field.par_dens,field.gas_dens,grid.g2a,grid.g31a,grid.dvRb,grid.g32a,grid.g2b,grid.dvTb,grid.NR,grid.NTH,field.Nparticles,grid.ii,grid.io,grid.ji,grid.jo)

    get_par_vel_diff_numba(field.par_K,field.Conc,field.grad_lgConc_R,field.grad_lgConc_T,field.par_vr_diff,field.par_vt_diff,grid.ii,grid.io,grid.ji,grid.jo,field.Nparticles)

    return

@jit(nopython = True, parallel=True)
def get_par_vel_diff_numba(K,C,glgCR,glgCT,vrd,vtd,ii,io,ji,jo,Nparticles):

    for i in prange(ii,io+3):
        for j in range(ji,jo+3):
            for k in range(Nparticles):
                if (i < io+2):
                    vtd[i,j,k] = -0.5*(K[i,j-1,k]+K[i,j,k])*glgCT[i,j,k]
                if (j < jo+2):
                    vrd[i,j,k] = -0.5*(K[i-1,j,k]+K[i,j,k])*glgCR[i,j,k]
    

    return

@jit(nopython = True,parallel=True)
def get_grad_conc(pd,gd,g2a,g31a,dvRb,g32a,g2b,dvTb,NR,NTH,Nparticles,ii,io,ji,jo):

    # calculate the conentration graient using volume derivatives

    grad_lgC_R = np.zeros((NR+3,NTH+2,Nparticles))
    grad_lgC_T = np.zeros((NR+2,NTH+3,Nparticles))
    Conc = np.zeros((NR+2,NTH+2,Nparticles))

    for i in prange(ii-1,io+2):
        for j in range(ji-1,jo+2):
            for k in range(Nparticles):
                Conc[i,j,k] = pd[i,j,k]/gd[i,j]
    

    for i in prange(ii,io+3):
        for j in range(ji,jo+3):
            for k in range(Nparticles):
                if (j < jo+2):
                    grad_lgC_R[i,j,k] = g2a[i]*g31a[i] * (math.log(Conc[i,j,k])-math.log(Conc[i-1,j,k])) / dvRb[i]
                if (i < io +2):
                    grad_lgC_T[i,j,k] = g32a[j]/g2b[i] * (math.log(Conc[i,j,k])-math.log(Conc[i,j-1,k])) / dvTb[j]

    return Conc,grad_lgC_R, grad_lgC_T


## growth update

def get_tgrow(grid,field,system):

    kb = 1.38e-16
    mh = 1.67e-24
    sigma_gas = 2e-15 # molecular gas collision cross-section

    # this finds the growth timescale

    #  get cell centered differential velocity
    vr_b = (field.par_vr_drift[grid.ii+1:grid.io+2,grid.ji:grid.jo+1,:] + field.par_vr_drift[grid.ii:grid.io+1,grid.ji:grid.jo+1,:])/2.
    vt_b = (field.par_vth_drift[grid.ii:grid.io+1,grid.ji+1:grid.jo+2,:] + field.par_vth_drift[grid.ii:grid.io+1,grid.ji:grid.jo+1,:])/2.

    delta_v = np.sqrt((vr_b**2.+vt_b**2.) / 2.) # average differentail velocity between to particles of same size

    par_num_dens = field.par_dens[grid.ii:grid.io+1,grid.ji:grid.jo+1,:]/(4./3.*np.pi*field.par_dens_in[grid.ii:grid.io+1,grid.ji:grid.jo+1,:]*field.par_size[grid.ii:grid.io+1,grid.ji:grid.jo+1,:]**3.)

    # growth due to differential velocity
    inv_tgrow = (0.5 * par_num_dens * np.pi * (2. * field.par_size[grid.ii:grid.io+1,grid.ji:grid.jo+1,:])**2. * delta_v)
    # growth due to brownian motion
    # Brownian motion velcity
    Vbm = np.sqrt(16. * kb * field.gas_T[grid.ii:grid.io+1,grid.ji:grid.jo+1]/(np.pi * system.mmw * mh))
    # Mean Thermal velocity
    Vth = Vbm / np.sqrt(2.)
    # molecular visocsity

    ngas = field.gas_dens[grid.ii:grid.io+1,grid.ji:grid.jo+1] / (system.mmw * mh)
    l_mfp = 1./ (ngas * sigma_gas * np.sqrt(2.))

    nu_mol = 0.5 * l_mfp * Vth

    eta = nu_mol * field.gas_dens[grid.ii:grid.io+1,grid.ji:grid.jo+1] # dynamic vicosity

    D_bm = (((kb * field.gas_T[grid.ii:grid.io+1,grid.ji:grid.jo+1] / (6. *np.pi * eta)).T * (1./field.par_size[grid.ii:grid.io+1,grid.ji:grid.jo+1,:]).T).T)

    VbmXa = (Vbm.T * field.par_size[grid.ii:grid.io+1,grid.ji:grid.jo+1,:].T).T

    min_diff = np.minimum(VbmXa,D_bm)

    inv_tgrow += 0.5*4.*np.pi * min_diff * field.par_size[grid.ii:grid.io+1,grid.ji:grid.jo+1,:] * par_num_dens

    field.par_tgrow[grid.ii:grid.io+1,grid.ji:grid.jo+1,:] = 1./inv_tgrow

    return

def get_tgrow_jd(grid,field,system):

    get_tgrow_numba(field.par_vr_drift,field.par_vth_drift,field.par_dens,field.par_dens_in,field.par_size,field.gas_T,
            field.gas_dens,field.par_tgrow,system.mmw,grid.ii,grid.io,grid.ji,grid.jo,field.Nparticles)


    return
@jit(void(float64[:,:,:],float64[:,:,:],float64[:,:,:],float64[:,:,:],float64[:,:,:],float64[:,:],float64[:,:],float64[:,:,:],float64,int32,int32,int32,int32,int32),nopython=True,parallel=True)
def get_tgrow_numba(vrd,vtd,pd,pid,pa,gT,gd,tgrow,mmw,ii,io,ji,jo,Nparticles):

    # this numba loop calculates the growth timescale on b grid

    kb = 1.38e-16
    mh = 1.67e-24
    sigma_gas = 2e-15 # molecular gas collision cross-section

    for i in prange(ii,io+1):
        for j in range(ji,jo+1):
            for k in range(Nparticles):
                # cell centred drift velocities 
                vr_b = 0.5* (vrd[i,j,k] + vrd[i+1,j,k])
                vt_b = 0.5* (vtd[i,j,k] + vtd[i,j+1,k])

                delta_v = math.sqrt((vr_b**2. + vt_b**2.)/2.) # RMS deltav

                par_num_dens = pd[i,j,k]/(4./3.*math.pi*pid[i,j,k]*pa[i,j,k]**3.)

                # growth due to differential velocity
                inv_tgrow = (0.5 * par_num_dens * math.pi * (2. * pa[i,j,k])**2. * delta_v)
                # growth due to brownian motion
                # Brownian motion velcity
                Vbm = math.sqrt(16. * kb * gT[i,j]/(math.pi * mmw * mh))
                # Mean Thermal velocity
                Vth = Vbm / math.sqrt(2.)
                # molecular visocsity

                ngas = gd[i,j] / (mmw * mh)
                l_mfp = 1./ (ngas * sigma_gas * math.sqrt(2.))

                nu_mol = 0.5 * l_mfp * Vth

                eta = nu_mol * gd[i,j] # dynamic vicosity

                D_bm = (((kb * gT[i,j] / (6. * math.pi * eta)) * (1./pa[i,j,k])))

                VbmXa = (Vbm * pa[i,j,k])

                min_diff = min(VbmXa,D_bm)

                inv_tgrow += 0.5*4.*math.pi * min_diff * pa[i,j,k] * par_num_dens

                tgrow[i,j,k] = 1./inv_tgrow

    return


def size_update(grid,field,dt):

    # this performs an update for the growth term using opperator splitting

    # first growth step 

    field.par_size[grid.ii:grid.io+1,grid.ji:grid.jo+1,:] += (field.par_size[grid.ii:grid.io+1,grid.ji:grid.jo+1,:] /
                                                field.par_tgrow[grid.ii:grid.io+1,grid.ji:grid.jo+1,:] ) * dt

    # now advection step - using MUSCL scheme
    field.par_a_star_r, field.par_a_star_th = get_qstar(grid.ii,grid.io,grid.ji,grid.jo,
                                        grid.NR,grid.NTH,field.Nparticles,field.par_size,
                                        field.par_vr,field.par_vth,grid.dRb,grid.dTb,
                                        grid.dRa,grid.dTa,grid.g2b,dt)

    field.M_r_size = field.par_a_star_r[:,:-1,:]  * field.par_vr 
    field.M_t_size = field.par_a_star_th[:-1,:,:] * field.par_vth

    field.M_r3D_size = (field.M_r_size.T * grid.A_r[:,:-1].T).T
    field.M_t3D_size = (field.M_t_size.T * grid.A_th[:-1,:].T).T

    delta_F_r  = (field.M_r3D_size[grid.ii:grid.io+1,grid.ji:grid.jo+1]-field.M_r3D_size[grid.ii+1:grid.io+2,grid.ji:grid.jo+1])
    delta_F_t  = (field.M_t3D_size[grid.ii:grid.io+1,grid.ji:grid.jo+1]-field.M_t3D_size[grid.ii:grid.io+1,grid.ji+1:grid.jo+2])

    div_F  = (delta_F_r.T * (np.outer(1./(grid.dvRa[grid.ii:grid.io+1]),np.ones(grid.NTH))).T).T
    div_F += (delta_F_t.T * (np.outer(1./(grid.dvRa[grid.ii:grid.io+1]),1./(grid.dvTa[grid.ji:grid.jo+1]))).T).T

    # now div_v term
    div_v = ((np.outer(grid.g2a[grid.ii+1:grid.io+2]*grid.g31a[grid.ii+1:grid.io+2]/grid.dvRa[grid.ii:grid.io+1],np.ones(grid.NTH)).T * (field.par_vr[grid.ii+1:grid.io+2,grid.ji:grid.jo+1,:]).T).T - 
             (np.outer(grid.g2a[grid.ii:grid.io+1]*grid.g31a[grid.ii:grid.io+1]/grid.dvRa[grid.ii:grid.io+1],np.ones(grid.NTH)).T * (field.par_vr[grid.ii:grid.io+1,grid.ji:grid.jo+1,:]).T).T )
    div_v +=((np.outer(1./grid.g2b[grid.ii:grid.io+1],grid.g32a[grid.ji+1:grid.jo+2]/grid.dvTa[grid.ji:grid.jo+1]).T * (field.par_vth[grid.ii:grid.io+1,grid.ji+1:grid.jo+2,:]).T).T -
             (np.outer(1./grid.g2b[grid.ii:grid.io+1],grid.g32a[grid.ji:grid.jo+1]/grid.dvTa[grid.ji:grid.jo+1]).T * (field.par_vth[grid.ii:grid.io+1,grid.ji:grid.jo+1,:]).T).T )

    # update size due to advection
    field.par_size[grid.ii:grid.io+1,grid.ji:grid.jo+1,:] += (field.par_size[grid.ii:grid.io+1,grid.ji:grid.jo+1,:] * div_v) *dt # a * div(v) term
    field.par_size[grid.ii:grid.io+1,grid.ji:grid.jo+1,:] += div_F * dt # (div (a * u) term)


    return

def size_update_jd(grid,field,dt):

    #field.par_size[grid.ii:grid.io+1,grid.ji:grid.jo+1,:] += (field.par_size[grid.ii:grid.io+1,grid.ji:grid.jo+1,:] /
    #                                            field.par_tgrow[grid.ii:grid.io+1,grid.ji:grid.jo+1,:] ) * dt

    # get the diffusive velocity 
    get_par_vel_diff_jd(grid,field)

    # now advection step - using MUSCL scheme
    field.par_a_star_r, field.par_a_star_th = get_qstar(grid.ii,grid.io,grid.ji,grid.jo,
                                        grid.NR,grid.NTH,field.Nparticles,field.par_size,
                                        field.par_vr+field.par_vr_diff,field.par_vth+field.par_vt_diff,grid.dRb,grid.dTb,
                                        grid.dRa,grid.dTa,grid.g2b,dt)

    size_update_numba(grid.ii,grid.io,grid.ji,grid.jo,field.Nparticles,field.par_a_star_r,field.par_a_star_th,field.par_size,
    field.par_vr+field.par_vr_diff,field.par_vth+field.par_vt_diff,field.par_tgrow,grid.A_r,grid.A_th,grid.dvRa,grid.dvTa,grid.g2a,grid.g31a,grid.g2b,grid.g32a,dt)


    return

@jit(void(int32,int32,int32,int32,int32,float64[:,:,:],float64[:,:,:],float64[:,:,:],float64[:,:,:],float64[:,:,:],float64[:,:,:],float64[:,:],float64[:,:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64),nopython=True,parallel=True)
def size_update_numba(ii,io,ji,jo,Nparticles,pastarr,pastart,ps,vr,vth,tgrow,Ar,At,dvRa,dvTa,g2a,g31a,g2b,g32a,dt):

    M_r3D_size = np.zeros((io+2,jo+2,Nparticles))
    M_t3D_size = np.zeros((io+2,jo+2,Nparticles))
    
    for i in prange(ii,io+2):
        for j in range(ji,jo+2):
            for k in range(Nparticles):
                M_r_size = pastarr[i,j,k]  * vr[i,j,k] 
                M_t_size = pastart[i,j,k]  * vth[i,j,k]

                M_r3D_size[i,j,k] = M_r_size * Ar[i,j]
                M_t3D_size[i,j,k] = M_t_size * At[i,j]

    for i in prange(ii,io+1):
        for j in range(ji,jo+1):
            for k in range(Nparticles):
                delta_F_r  = (M_r3D_size[i,j,k]-M_r3D_size[i+1,j,k])
                delta_F_t  = (M_t3D_size[i,j,k]-M_t3D_size[i,j+1,k])

                div_F  = delta_F_r/dvRa[i]
                div_F += delta_F_t/(dvRa[i]*dvTa[j])

                # now div_v term
                div_v = (((g2a[i+1]*g31a[i+1]/dvRa[i]) * (vr[i+1,j,k])) - 
                ((g2a[i]*g31a[i]/dvRa[i]) * (vr[i,j,k])) )
                div_v += ((1./g2b[i]*g32a[j+1]/dvTa[j]) * (vth[i,j+1,k]) -
                ((1./g2b[i]*g32a[j]/dvTa[j]) * (vth[i,j,k])))

                # size update due to growth
                ps[i,j,k] += ps[i,j,k] / tgrow[i,j,k] * dt
                # update size due to advection
                ps[i,j,k] += (ps[i,j,k] * div_v) *dt # a * div(v) term
                ps[i,j,k] += div_F * dt # (div (a * u) term)


    return


def get_kappa_rho_particles(field,Qext):

    # this routine updates rho*kappa for the particles only
    # 

    field.Qext = Qext(field.par_size) # extinction efficiency
    kappa = 3./4.*field.Qext/field.par_dens_in/field.par_size
    kappa_rho = kappa * field.par_dens

    # now sum over all particle sizes
    field.par_rho_kap = np.sum(kappa_rho,axis=2)

    return

def update_tau_b_par(field,grid,rays,Qext):

    # this updates optical depth at cell centres due to particles
    # this is a SLOW update so will not be run every time-step

    get_kappa_rho_particles(field,Qext)

    # do ray trace
    rays.do_ray_trace(field.par_rho_kap)
    rays.get_tau_grid(grid)

    # update total optical depth

    rays.tau_b = rays.tau_b_gas + rays.tau_b_par


    return    


##### functions defined to calculate flux limiters

@jit(float64(float64,float64),nopython=True)
def minmod(a,b):

    if (a*b>0.):
        if np.fabs(b) > np.fabs(a):
            ans = a
        else:
            ans = b
    else: 
        ans = 0. 

    return ans

@jit(float64(float64,float64),nopython =True)
def maxmod(a,b):
    if (a*b>0.):
        if np.fabs(b) > np.fabs(a):
            ans = b
        else:
            ans = a
    else: 
        ans = 0. 

    return ans






    

