import numpy as np
#### written by JO Dec 2020 ###'#
#### This contains a spherical finite volume grid 
#### This grid is based on the staggered grids, ala zeus (Stone & Norman 1992)
#### The grid will include one ghost cell (2nd order method)
#### This file contains constructions for the field varaibles

class system:

    def __init__(self,Mplanet,Lstar,sep,P0,mdot_wind,mmw,Tequil):

        self.Mp   = Mplanet
        self.Ls   = Lstar
        self.sep  = sep
        self.Fbol = Lstar / (4. * np.pi * sep**2.)
        self.Teq  = Tequil 
        self.Mdot = mdot_wind
        self.Pbase= P0
        self.mmw  = mmw
        self.kappa_star = 4e-3
        self.gamma      = 0.25
        self.Tint       = 50.
        self.firr       = 1./4.


class field:

    def __init__(self,grid,Nparticles,dens_int = 1.):

        self.short_friction = True ## default to short friction time approx
        self.Tstar = 5777. # temperature of star in K
        
        # Nparticles is the number of particles in the grid

        self.Nparticles = Nparticles

        # gas properties
        self.gas_dens = np.zeros((grid.NR+2,grid.NTH+2))
        self.gas_P    = np.zeros((grid.NR+2,grid.NTH+2))
        self.gas_T    = np.zeros((grid.NR+2,grid.NTH+2))
        self.gas_vr   = np.zeros((grid.NR+3,grid.NTH+2)) # gas velocity in radial direction
        self.gas_vth  = np.zeros((grid.NR+2,grid.NTH+3)) # gas velocity in theta direction
        self.tau_IR   = np.zeros((grid.NR+2)) ## veritcal optical depth in IR -used for Guillot (2010) atmosphere construction

        # particle properties
        self.par_size    = np.zeros((grid.NR+2,grid.NTH+2,Nparticles))
        self.par_dens_in = np.zeros((grid.NR+2,grid.NTH+2,Nparticles)) + dens_int
        self.par_dens    = np.zeros((grid.NR+2,grid.NTH+2,Nparticles))
        self.par_vr      = np.zeros((grid.NR+3,grid.NTH+2,Nparticles))
        self.par_Sr      = np.zeros((grid.NR+3,grid.NTH+2,Nparticles))
        self.par_vr_st_r = np.zeros((grid.NR+3,grid.NTH+3,Nparticles))
        self.par_vr_st_t = np.zeros((grid.NR+3,grid.NTH+3,Nparticles))
        self.par_vr_drift= np.zeros((grid.NR+3,grid.NTH+2,Nparticles))
        self.par_vth_drift=np.zeros((grid.NR+2,grid.NTH+3,Nparticles))
        self.par_vr_diff = np.zeros((grid.NR+3,grid.NTH+2,Nparticles))
        self.par_vr_adv  = np.zeros((grid.NR+3,grid.NTH+2,Nparticles)) ### velcoity to use in advection scheme
        self.par_vt_adv  = np.zeros((grid.NR+2,grid.NTH+3,Nparticles))
        self.par_vt_diff = np.zeros((grid.NR+2,grid.NTH+3,Nparticles))
        self.par_vth     = np.zeros((grid.NR+2,grid.NTH+3,Nparticles))
        self.par_Sth     = np.zeros((grid.NR+2,grid.NTH+3,Nparticles))
        self.par_vt_st_r = np.zeros((grid.NR+3,grid.NTH+3,Nparticles))
        self.par_vt_st_t = np.zeros((grid.NR+3,grid.NTH+3,Nparticles))
        self.par_tstop   = np.zeros((grid.NR+2,grid.NTH+2,Nparticles))
        self.par_K       = np.zeros((grid.NR+2,grid.NTH+2,Nparticles))
        self.par_ar      = np.zeros((grid.NR+3,grid.NTH+2,Nparticles))
        self.par_ath     = np.zeros((grid.NR+2,grid.NTH+3,Nparticles))
        self.par_source  = np.zeros((grid.NR+2,grid.NTH+2,Nparticles))
        self.par_tgrow   = np.zeros((grid.NR+2,grid.NTH+2,Nparticles)) + 1e50
        self.tstop_bgrid = np.zeros((grid.NR+2,grid.NTH+2,Nparticles))
        self.par_diff_b  = np.zeros((grid.NR+2,grid.NTH+2,Nparticles)) # diffusion contant on b grid
        self.div_Fdiff_numba = np.zeros((grid.NR+2,grid.NTH+2,Nparticles)) 
        self.Conc        = np.zeros((grid.NR+2,grid.NTH+2,Nparticles))
        self.Q           = np.zeros((grid.NR+2,grid.NTH+2,Nparticles))
        self.Qext        = np.zeros((grid.NR+2,grid.NTH+2,Nparticles))
        self.par_rho_kap = np.zeros((grid.NR+2,grid.NTH+2))

        # optical depth and flux properties
        self.Fstar_rw = np.zeros((grid.NR+3,grid.NTH+2,Nparticles))
        self.Fstar_tw = np.zeros((grid.NR+2,grid.NTH+2,Nparticles))
        self.Fstar_b  = np.zeros((grid.NR+2,grid.NTH+2))
        self.kappa    = np.zeros((grid.NR+2,grid.NTH+2,Nparticles))

        return

    def setup_iso_atm(self, system, grid, with_spherical_wind = False):

        kb = 1.38e-16
        G  = 6.67e-8
        mh = 1.67e-24

        # setup hydrostatic spherical atmosphere
        cs2 = kb * system.Teq / (system.mmw * mh)
        b = G * system.Mp / grid.Rmin / cs2 # depth of potential

        self.gas_P    = system.Pbase * np.exp(b*(grid.Rmin/grid.Rb2d-1.))
        self.gas_dens = self.gas_P / cs2
        self.gas_T    = np.zeros((grid.NR+2,grid.NTH+2)) + system.Teq

        grid.scale_height_b = cs2 / (G * system.Mp/grid.Rb**2.)

        # check for suffient resolution - want 3 cells per H min
        Hbase = (G*system.Mp/grid.Ra[0]**2./cs2)**(-1.)
        if (Hbase < 3*grid.dRa[0]):
            print ("Error, insuffient radial resolution, require 3 cells per Scale Height")
            print ("H/deltaR is")
            print (Hbase/grid.dRa[0])


        if with_spherical_wind:
            # include spherical outflow
            rho_a = (self.gas_dens[:-1,:] + self.gas_dens[1:,:])/2.
            self.gas_vr[1:-1,:] = system.Mdot / (4. * np.pi * grid.Ra2d[1:-1,:-1]**2. * rho_a)
        
        return


    def setup_guillot_atm(self, system, grid, with_spherical_wind = False):

        ## this setups a Guillot (2010) style average global atmosphere with equal redistribution

        ###
        ### WARNING : UNTESTED
        ###

        kb = 1.38e-16
        G  = 6.67e-8
        mh = 1.67e-24

        # setup hydrostatic spherical atmosphere
        cs2 = kb * system.Teq / (system.mmw * mh)
        b = G * system.Mp / grid.Rmin / cs2 # depth of potential

        kappa_IR = system.kappa_star/system.gamma

        tau_base = system.Pbase * kappa_IR / (G * system.Mp / grid.Rmin**2.) # constant g formula
        
        T_base = guillot_Ttau(system.Tint,tau_base,system.Teq,system.firr,system.gamma)

        rho_base = system.Pbase / T_base * system.mmw * mh / kb

        print (kappa_IR,tau_base,T_base,rho_base,system.Pbase)

        #### Now first cell middle
        delta_tau_intial = rho_base * (grid.Rb[grid.ii]-grid.Ra[grid.ii]) * kappa_IR

        print (delta_tau_intial)

        self.tau_IR[grid.ii] = tau_base - delta_tau_intial

        print(self.tau_IR[grid.ii])

        self.gas_P[grid.ii,:] = system.Pbase - delta_tau_intial * (G * system.Mp / grid.Rmin **2. / kappa_IR)

        print (G * system.Mp / grid.Rmin **2.)

        self.gas_T[grid.ii,:] = guillot_Ttau(system.Tint,self.tau_IR[grid.ii],system.Teq,system.firr,system.gamma)

        self.gas_dens[grid.ii,:] = self.gas_P[grid.ii,:] / self.gas_T[grid.ii,:] * system.mmw * mh /kb

        ## loop cell down
        delta_tau = self.gas_dens[grid.ii,10] * (grid.Rb[grid.ii]-grid.Rb[grid.ii-1]) * kappa_IR ## assumed spherically symmetric
        self.tau_IR[grid.ii-1] = self.tau_IR[grid.ii] + delta_tau

        self.gas_P[grid.ii-1,:] = self.gas_P[grid.ii,:] + delta_tau_intial * (G * system.Mp / grid.Rb[grid.ii] **2.  / kappa_IR)

        self.gas_T[grid.ii-1,:] = guillot_Ttau(system.Tint,self.tau_IR[grid.ii-1],system.Teq,system.firr,system.gamma)

        self.gas_dens[grid.ii-1,:] = self.gas_P[grid.ii-1,:] / self.gas_T[grid.ii-1,:] * system.mmw * mh /kb

        ## now loop to the top of the grid
        for i in range(grid.ii+1,grid.io+2):
            delta_tau = self.gas_dens[i-1,10] * (grid.Rb[i]-grid.Rb[i-1]) * kappa_IR ## assumed spherically symmetric

            self.tau_IR[i] = self.tau_IR[i-1] - delta_tau

            lg_delta_tau = np.log(self.tau_IR[i-1]) - np.log(self.tau_IR[i])

            new_lgP = np.log(self.gas_P[i-1,:]) -  lg_delta_tau * (G * system.Mp * self.tau_IR[i-1] / grid.Rb[i-1] **2. / kappa_IR / self.gas_P[i-1,:])

            self.gas_P[i,:] = np.exp(new_lgP)

            self.gas_T[i,:] = guillot_Ttau(system.Tint,self.tau_IR[i],system.Teq,system.firr,system.gamma)

            self.gas_dens[i,:] = self.gas_P[i,:] / self.gas_T[i,:] * system.mmw * mh /kb

            cs2 = kb * self.gas_T[:,10] / (system.mmw * mh)

        grid.scale_height_b = cs2 / (G * system.Mp/grid.Rb**2.)

        # check for suffient resolution - want 3 cells per H min
        Hbase = (G*system.Mp/grid.Ra[0]**2./cs2[0])**(-1.)
        if (Hbase < 3*grid.dRa[0]):
            print ("Error, insuffient radial resolution, require 3 cells per Scale Height")
            print ("H/deltaR is")
            print (Hbase/grid.dRa[0])


        if with_spherical_wind:
            # include spherical outflow
            rho_a = (self.gas_dens[:-1,:] + self.gas_dens[1:,:])/2.
            self.gas_vr[1:-1,:] = system.Mdot / (4. * np.pi * grid.Ra2d[1:-1,:-1]**2. * rho_a)
        
        return

def guillot_Ttau(Tint,tau,Tirr,firr,gamma):

    ## evaluate the guillot T-tau relation

    T4 = 0.75 * Tint**4. * (2./3. + tau) + 0.75 *Tirr**4.*firr* (2./3.+1./(gamma*np.sqrt(3.))  + (gamma/np.sqrt(3.)-1./(gamma*np.sqrt(3.))) * np.exp(-gamma * tau / np.sqrt(3.))  )

    return T4**0.25