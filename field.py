import numpy as np
#### written by JO Dec 2020 ###'#
#### This contains a spherical finite volume grid 
#### This grid is based on the staggered grids, ala zeus (Stone & Norman 1992)
#### The grid will include one ghost cell (2nd order method)
#### This file contains constructions for the field varaibles

class system:

    def __init__(self,Mplanet,Lstar,sep,P0,mdot_wind,mmw):

        self.Mp   = Mplanet
        self.Ls   = Lstar
        self.sep  = sep
        self.Fbol = Lstar / (4. * np.pi * sep**2.)
        self.Teq  = (self.Fbol/4./5.6704e-5)**(0.25)
        self.Mdot = mdot_wind
        self.Pbase= P0
        self.mmw  = mmw
        self.kappa_star = 1e-2


class field:

    def __init__(self,grid,Nparticles,dens_int = 1.):

        # Nparticles is the number of particles in the grid

        self.Nparticles = Nparticles

        # gas properties
        self.gas_dens = np.zeros((grid.NR+2,grid.NTH+2))
        self.gas_P    = np.zeros((grid.NR+2,grid.NTH+2))
        self.gas_T    = np.zeros((grid.NR+2,grid.NTH+2))
        self.gas_vr   = np.zeros((grid.NR+3,grid.NTH+2)) # gas velocity in radial direction
        self.gas_vth  = np.zeros((grid.NR+2,grid.NTH+3)) # gas velocity in theta direction

        # particle properties
        self.par_size    = np.zeros((grid.NR+2,grid.NTH+2,Nparticles))
        self.par_dens_in = np.zeros((grid.NR+2,grid.NTH+2,Nparticles)) + dens_int
        self.par_dens    = np.zeros((grid.NR+2,grid.NTH+2,Nparticles))
        self.par_vr      = np.zeros((grid.NR+3,grid.NTH+2,Nparticles))
        self.par_vr_drift= np.zeros((grid.NR+3,grid.NTH+2,Nparticles))
        self.par_vth_drift=np.zeros((grid.NR+2,grid.NTH+3,Nparticles))
        self.par_vr_diff = np.zeros((grid.NR+3,grid.NTH+2,Nparticles))
        self.par_vt_diff = np.zeros((grid.NR+2,grid.NTH+3,Nparticles))
        self.par_vth     = np.zeros((grid.NR+2,grid.NTH+3,Nparticles))
        self.par_tstop   = np.zeros((grid.NR+2,grid.NTH+2,Nparticles))
        self.par_K     = np.zeros((grid.NR+2,grid.NTH+2,Nparticles))
        self.par_ar      = np.zeros((grid.NR+3,grid.NTH+2,Nparticles))
        self.par_ath     = np.zeros((grid.NR+3,grid.NTH+2,Nparticles))
        self.par_source = np.zeros((grid.NR+2,grid.NTH+2,Nparticles))
        self.par_tgrow = np.zeros((grid.NR+2,grid.NTH+2,Nparticles)) + 1e50
        self.tstop_bgrid = np.zeros((grid.NR+2,grid.NTH+2,Nparticles))
        self.par_diff_b = np.zeros((grid.NR+2,grid.NTH+2,Nparticles)) # diffusion contant on b grid
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
