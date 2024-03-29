import numpy as np
from numba import jit
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from scipy.integrate import quad
from joblib import Parallel, delayed
#### written by JO Dec 2020 ###'#
#### This contains a spherical finite volume grid 
#### This grid is based on the staggered grids, ala zeus (Stone & Norman 1992)
#### The grid will include one ghost cell (2nd order method)
#### This file contains constructions for the grids (FV and rays)

class grid:

    def __init__(self,Rmin,Rmax,theta_min,theta_max,NR,NTH,TH_PL):

        # Rmin in the inner spherical boundary
        # Rmax is the outer spherical boundary
        # theta_min is the theta min boundary
        # theta_max is the theta max boundary
        # NR is the number of radial grid cells
        # NTh is the number of angular grid cells

        self.NR = NR
        self.NTH = NTH

        self.Rmin = Rmin
        self.Rmax = Rmax
        self.Th_min = theta_min
        self.Th_max = theta_max

        if (self.Th_max <=np.pi):
            # use spherical polars with the pole pointing towards the star

            # we'll use logspace grid in R
            self.Ra = np.logspace(np.log10(self.Rmin),np.log10(self.Rmax),self.NR+1)
            # adjust resolution for theta to give square cells near terminator
            self.Ta = np.linspace((self.Th_min)**TH_PL,(np.pi/2.)**TH_PL,self.NTH//2+1)
            self.Ta = self.Ta**(1./TH_PL)

            # now print out dT over delta r/r at terminaotr
            print ("delta Theta compared to delta r /r at the terminator is")
            print (np.diff(self.Ta[-2:]))
            print (np.diff(self.Ra[:2])/self.Ra[0])
            # now append second part of Ta to array
            self.Ta = np.append(self.Ta[:-1],np.pi-self.Ta[::-1])



            # add ghost cells to a grid
            dRa = self.Ra[1]-self.Ra[0] 
            dTa = self.Ta[1]-self.Ta[0]
            self.Ra = np.append(self.Ra[0]-dRa,np.append(self.Ra,self.Ra[-1]+dRa)) # assumes linear grid
            self.Ta = np.append(self.Ta[0]-dTa,np.append(self.Ta,self.Ta[-1]+dTa)) # assumes linear grid

            # calculate b-grid
            self.Rb = self.Ra[:-1] + np.diff(self.Ra)/2.
            self.Tb = self.Ta[:-1] + np.diff(self.Ta)/2.

            # find differences and scale factors, we use the zeus-style indexing
            self.dRa = np.append(np.diff(self.Ra),0.) # add zero to make array same length
            self.dRb = np.append(0.,np.diff(self.Rb)) # ditto   
            self.dTa = np.append(np.diff(self.Ta),0.)
            self.dTb = np.append(0.,np.diff(self.Tb))

            # scale factors
            self.g2a = self.Ra
            self.g2b = self.Rb
            self.g31a = self.Ra
            self.g31b = self.Rb
            self.g32a = np.sin(self.Ta)
            self.g32b = np.sin(self.Tb)

            # volume differences
            self.dvRa = np.append(1./3.*np.diff(self.Ra**3.),0)
            self.dvRb = np.append(0.,1./3.*np.diff(self.Rb**3.))
            self.dvTa = np.append(np.diff(-np.cos(self.Ta)),0.)
            self.dvTb = np.append(0.,np.diff(-np.cos(self.Tb)))

            # 2d verisions of grids
            [self.Ra2d,self.Ta2d] = np.meshgrid(self.Ra,self.Ta,indexing='ij')
            [self.Rb2d,self.Tb2d] = np.meshgrid(self.Rb,self.Tb,indexing='ij')

            # 2d verisions with those that don't see a ray hidden
            theta_max_rays = np.pi - np.arcsin(self.Ra[0]/self.Ra[-1])

            self.index_theta_max = np.argmin(np.fabs(self.Tb-theta_max_rays))+1 # plus one for safety
            [self.Ra2dTb,self.Tb2dRa] = np.meshgrid(self.Ra,self.Tb,indexing='ij') # for Ray-tracing interpolation
            [self.Rb2dTa,self.Ta2dRb] = np.meshgrid(self.Rb,self.Ta,indexing='ij')


            # cartesian 
            self.Xa = self.Ra2d * np.sin(self.Ta2d)
            self.Za = self.Ra2d * np.cos(self.Ta2d)
            self.Xb = self.Rb2d * np.sin(self.Tb2d)
            self.Zb = self.Rb2d * np.cos(self.Tb2d)

            # those needed for ray-tracing
            self.Xrw = self.Ra2dTb * np.sin(self.Tb2dRa) # Xpositions on radial walls
            self.Zrw = self.Ra2dTb * np.cos(self.Tb2dRa) # Zpositions on radial walls
            self.Xtw = self.Rb2dTa * np.sin(self.Ta2dRb) # X psoitions on theta walls
            self.Ztw = self.Rb2dTa * np.cos(self.Ta2dRb) # Z positions on theta walls

            a,self.NTrw = np.shape(self.Tb2dRa)
            a,self.NTtw = np.shape(self.Ta2dRb)

            self.ii = 1
            self.io = NR
            self.ji = 1
            self.jo = NTH

            # area elements (Hayes et al. 2006)
            self.A_r = np.outer(self.g2a*self.g31a,np.ones(self.NTH+3))
            self.A_th = np.outer(np.append(self.g31b,0.)*self.dRa,self.g32a)

            # projected circular line-segement back to star
            self.dRpro_back = np.zeros((self.NR+2, self.NTH+2))
            # path length through cell
            self.cell_dZ = np.zeros((self.NR+2,self.NTH+2))
            for i in range(self.NR+2):
                for j in range(self.index_theta_max):
                    # estimate two projected areas in r and theta then pick smallest
                    # remember z-axis of co-ordinate system points towards the star
                    dr_pro = np.fabs(self.dRa[i]/(np.sin(self.Tb[j]) + 1e-10)) # to avoid overflow
                    dt_pro = np.fabs((self.dTa[j]*self.Rb[i])/(np.cos(self.Tb[j])+1e-10))

                    self.dRpro_back[i,j] = min(dr_pro,dt_pro)

                    # now calculate path lengths and pick smallest one
                    dr_z = np.fabs(self.dRa[i]/(np.cos(self.Tb[j]) + 1e-10)) # to avoid overflow
                    dt_z = np.fabs((self.dTa[j]*self.Rb[i])/(np.sin(self.Tb[j])+1e-10))

                    self.cell_dZ[i,j] = min(dr_z,dt_z)

        else:
            # 2pi version where "theta" is the "phi" of spherical polars and the pole points
            # perpendicular to the orbital plane
            # theta = 0 is the co-ordinate pointing towards the star (day-side)
            # we'll use logspace grid in R
            self.Ra = np.logspace(np.log10(self.Rmin),np.log10(self.Rmax),self.NR+1)
            # adjust resolution for theta to give square cells near terminator
            self.Ta = np.linspace((self.Th_min)**TH_PL,(np.pi/2.)**TH_PL,self.NTH//4+1)
            self.Ta = self.Ta**(1./TH_PL)

            # now print out dT over delta r/r at terminaotr
            print ("delta Theta compared to delta r /r at the terminator is")
            print (np.diff(self.Ta[-2:]))
            print (np.diff(self.Ra[:2])/self.Ra[0])
            # now append second part of Ta to array
            self.Ta = np.append(self.Ta[:-1],np.pi-self.Ta[::-1])
            self.Ta = np.append(self.Ta[:-1],2*np.pi-self.Ta[::-1])

            # add ghost cells to a grid
            dRa = self.Ra[1]-self.Ra[0] 
            dTa = self.Ta[1]-self.Ta[0]
            self.Ra = np.append(self.Ra[0]-dRa,np.append(self.Ra,self.Ra[-1]+dRa)) # assumes linear grid
            self.Ta = np.append(self.Ta[0]-dTa,np.append(self.Ta,self.Ta[-1]+dTa)) # assumes linear grid

            # calculate b-grid
            self.Rb = self.Ra[:-1] + np.diff(self.Ra)/2.
            self.Tb = self.Ta[:-1] + np.diff(self.Ta)/2.

            # find differences and scale factors, we use the zeus-style indexing
            self.dRa = np.append(np.diff(self.Ra),0.) # add zero to make array same length
            self.dRb = np.append(0.,np.diff(self.Rb)) # ditto   
            self.dTa = np.append(np.diff(self.Ta),0.)
            self.dTb = np.append(0.,np.diff(self.Tb))

            # scale factors
            self.g2a = self.Ra
            self.g2b = self.Rb
            self.g31a = self.Ra # here it is implicitly assumed that we are along the equator
            self.g31b = self.Rb # as above
            self.g32a = np.ones(np.size(self.Ta))
            self.g32b = np.ones(np.size(self.Tb))

            # volume differences
            self.dvRa = np.append(1./3.*np.diff(self.Ra**3.),0)
            self.dvRb = np.append(0.,1./3.*np.diff(self.Rb**3.))
            self.dvTa = np.append(np.diff((self.Ta)),0.)
            self.dvTb = np.append(0.,np.diff((self.Tb)))

            # 2d verisions of grids
            [self.Ra2d,self.Ta2d] = np.meshgrid(self.Ra,self.Ta,indexing='ij')
            [self.Rb2d,self.Tb2d] = np.meshgrid(self.Rb,self.Tb,indexing='ij')

            ## ranges
            self.ii = 1
            self.io = NR
            self.ji = 1
            self.jo = NTH

            # 2d verisions with those that don't see a ray hidden
            theta_max_rays = np.pi - np.arcsin(self.Ra[1]/self.Ra[-1])
            theta_max_rays2 = np.pi + np.arcsin(self.Ra[1]/self.Ra[-1])

            for j in range(NTH):
                if (self.Tb[j]>theta_max_rays):
                    self.index_theta_max = j
                    break
            
            for j in range(NTH,1,-1):
                if (self.Tb[j]<theta_max_rays2):
                    self.index_theta_max_2 = j
                    break

            [self.Ra2dTb,self.Tb2dRa] = np.meshgrid(self.Ra,self.Tb,indexing='ij') # for Ray-tracing interpolation
            [self.Rb2dTa,self.Ta2dRb] = np.meshgrid(self.Rb,self.Ta,indexing='ij')


            # cartesian - this is different to the "classical one" our z axis points towards the
            # star and the x axis is at the morning lim, the "y-axis" points in the direction
            # of the planets' orbital angular momentum
            self.Xa = self.Ra2d * np.sin(self.Ta2d)
            self.Za = self.Ra2d * np.cos(self.Ta2d)
            self.Xb = self.Rb2d * np.sin(self.Tb2d)
            self.Zb = self.Rb2d * np.cos(self.Tb2d)

            # those needed for ray-tracing
            self.Xrw = self.Ra2dTb * np.sin(self.Tb2dRa) # Xpositions on radial walls
            self.Zrw = self.Ra2dTb * np.cos(self.Tb2dRa) # Zpositions on radial walls
            self.Xtw = self.Rb2dTa * np.sin(self.Ta2dRb) # X psoitions on theta walls
            self.Ztw = self.Rb2dTa * np.cos(self.Ta2dRb) # Z positions on theta walls

            a,self.NTrw = np.shape(self.Tb2dRa)
            a,self.NTtw = np.shape(self.Ta2dRb)


            # area elements (Hayes et al. 2006)
            self.A_r = np.outer(self.g2a*self.g31a,np.ones(self.NTH+3))
            self.A_th = np.outer(np.append(self.g31b,0.)*self.dRa,self.g32a)

            # projected circular line-segement back to star -- NEVER USED??
            self.dRpro_back = np.zeros((self.NR+2, self.NTH+2))
            # path length through cell
            self.cell_dZ = np.zeros((self.NR+2,self.NTH+2))
            for i in range(self.NR+2):
                for j in range(self.index_theta_max):
                    # estimate two projected areas in r and theta then pick smallest
                    # remember z-axis of co-ordinate system points towards the star
                    dr_pro = np.fabs(self.dRa[i]/(np.sin(self.Tb[j]) + 1e-10)) # to avoid overflow
                    dt_pro = np.fabs((self.dTa[j]*self.Rb[i])/(np.cos(self.Tb[j])+1e-10))

                    self.dRpro_back[i,j] = min(dr_pro,dt_pro)

                    # now calculate path lengths and pick smallest one
                    dr_z = np.fabs(self.dRa[i]/(np.cos(self.Tb[j]) + 1e-10)) # to avoid overflow
                    dt_z = np.fabs((self.dTa[j]*self.Rb[i])/(np.sin(self.Tb[j])+1e-10))

                    self.cell_dZ[i,j] = min(dr_z,dt_z)


            #### calculate interpolation masks for the hemispherical interpolation of the ray-tracing grid
            self.mry1 = np.array(np.zeros(np.shape(self.Rb2d)),dtype=bool) # create empty masks set to all FALSE
            self.mry2 = np.array(np.zeros(np.shape(self.Rb2d)),dtype=bool)
            self.mns  = np.array(np.zeros(np.shape(self.Rb2d)),dtype=bool) # this is a nightside mask to set optical depth to highvalue

            for i in range(self.ii,self.io+2):
                for j in range(self.ji,self.index_theta_max+1):
                    ## determine whether index sits within hemisphere interpolation range or not
                    if (self.Tb[j]<np.pi/2.):
                        ## dayside so in range
                        self.mry1[i,j] = True
                    elif (self.Xb[i,j]>=self.Ra[1]):
                        ## night-side but can see a ray
                        self.mry1[i,j] = True

            for i in range(self.ii,self.io+2):
                for j in range(self.jo,self.index_theta_max_2-1,-1):
                    ## determine whether index sits within hemisphere interpolation range or not
                    if (self.Tb[j]>3.*np.pi/2.):
                        ## dayside so in range
                        self.mry2[i,j] = True
                    elif (self.Xb[i,j]<=-self.Ra[1]):
                    #    ## night-side but can see a ray
                        self.mry2[i,j] = True

            for i in range(self.io+1,self.ii-2,-1):
                for j in range(self.ji,self.jo):
                    # check whether on the nightside and set that mask to true
                    if ((self.Tb[j]>np.pi/2.) & (self.Tb[j]<np.pi)):
                        if ((self.Xb[i,j]<self.Ra[1])):
                            self.mns[i,j] = True
                    elif ((self.Tb[j]>np.pi) & (self.Tb[j]<3*np.pi/2.)):
                        if ((self.Xb[i,j]>-self.Ra[1])):
                            self.mns[i,j] = True

        return

#### This contains the positions of the rays through the grid

class rays:

    def __init__(self,grid,Nrays,PL):

        # grid is the 2D grid structure above
        # Nrays is the number of rays per hemisphere
        # We will space the height of the rays in theta

        theta_start = grid.Tb[grid.ji]
        theta_end1   = np.arcsin(grid.Ra[1]/grid.Ra[-1])
        theta_end    = np.pi/2.-0.00001

        zend1 = np.sin(theta_end1) * grid.Ra[-1]
        zend = np.sin(theta_end) * grid.Ra[-1]

        self.Nrays = Nrays

        Nr1 = Nrays // 2
        Nr2 = Nrays - Nrays // 2 +1

        self.id_terminator = Nr1 # id at which ther terminator starts

        theta_rays1 = (np.linspace(theta_start**(PL),theta_end1**(PL),Nr1))**(1./PL) 
        zrays_2 = np.linspace(zend1,zend,Nr2)
        theta_rays2 = np.arcsin(zrays_2/grid.Ra[-1])
        self.theta_rays = np.append(theta_rays1,theta_rays2[1:])


        #self.theta_rays = np.arccos(np.linspace(np.cos(theta_start),np.cos(theta_end),Nrays))

        self.Xrays = grid.Ra[-1]*np.sin(self.theta_rays)

        # now calculate the Z position that the rays enter the grid
        self.Zstart = grid.Ra[-1]*np.cos(self.theta_rays)

        self.Zpos = np.array([]) # Z position of all cell crossings
        self.Xpos = np.array([]) # X position of all cell crossings

        # now for each ray store the cell positions one moves through and the possition 
        # one enters/exits cell walls and store in list
        rays_i = [[] for i in range(Nrays)]
        rays_j = [[] for i in range(Nrays)]
        rays_Z = [[] for i in range(Nrays)]
        rays_dZ = [[] for i in range(Nrays)]
        ray_len = [1 for i in range(Nrays)] 

        for i in range(Nrays):
            # find starting position and index
            jstart = np.asarray(grid.Ta < self.theta_rays[i]).nonzero()[0][-1]
            istart = grid.NR+1 # enters in ghost cell

            rays_i[i].append(istart)
            rays_j[i].append(jstart)
            rays_Z[i].append(self.Zstart[i])

            ingrid = True
            while ingrid:

                # now move through grid calculating which cell is next
                # do this by calculating the r you cut the next theta 
                # and the theta you cut the next r  
                r_cut_theta = self.Xrays[i]/np.sin(grid.Ta[rays_j[i][-1]+1])

                if rays_Z[i][-1] > 0:
                    if r_cut_theta > grid.Ra[rays_i[i][-1]]:
                        # exit into theta +1 on theta axis
                        rays_i[i].append(rays_i[i][-1])
                        rays_j[i].append(rays_j[i][-1]+1)
                        rays_Z[i].append(self.Xrays[i]/np.tan(grid.Ta[rays_j[i][-1]]+1e-15)) # 1e-15 to prevent divide by inf
                        ray_len[i]+=1 

                    else:
                        # exit into r-1
                        rays_i[i].append(rays_i[i][-1]-1)
                        rays_j[i].append(rays_j[i][-1])
                        rays_Z[i].append(np.sqrt(grid.Ra[rays_i[i][-1]+1]**2.-self.Xrays[i]**2.))
                        ray_len[i]+=1
                        if (rays_i[i][-1]==-1):
                            # exitted the ghost cell at inner r boundary - ray finished
                            ingrid = False
                else:
                    if (r_cut_theta < grid.Ra[rays_i[i][-1]+1]):
                        # exit into theta +1 on theta axis
                        rays_i[i].append(rays_i[i][-1])
                        rays_j[i].append(rays_j[i][-1]+1)
                        rays_Z[i].append(self.Xrays[i]/np.tan(grid.Ta[rays_j[i][-1]]+1e-15))
                        ray_len[i]+=1
                    else: 
                        # exit into r+1
                        rays_i[i].append(rays_i[i][-1]+1)
                        rays_j[i].append(rays_j[i][-1])
                        rays_Z[i].append(-np.sqrt(grid.Ra[rays_i[i][-1]]**2.-self.Xrays[i]**2.))
                        ray_len[i]+=1
                        if (rays_i[i][-1]==(grid.NR+2)):
                            # exitted into ghost cell at inner r boundary - ray finished
                            ingrid = False

                rays_dZ[i].append(-(rays_Z[i][-1]-rays_Z[i][-2]))

            # append to cell crossings
            self.Xpos = np.append(self.Xpos,np.zeros(len(rays_Z[i]))+self.Xrays[i])
            self.Zpos = np.append(self.Zpos,rays_Z[i])

        # for symmetry append first ray 
        #self.Zpos = np.append(rays_Z[0],self.Zpos)
        #self.Xpos = np.append(np.zeros(len(rays_Z[0]))-self.Xrays[0],self.Xpos)



        self.ray_i = rays_i
        self.ray_j = rays_j
        self.ray_Z = rays_Z
        self.ray_length = ray_len
        self.ray_dZ = rays_dZ



        self.theta_rays2 = 2*np.pi - self.theta_rays #symmetric about the sub-stellar point


        #self.theta_rays = np.arccos(np.linspace(np.cos(theta_start),np.cos(theta_end),Nrays))

        self.Xrays2 = grid.Ra[-1]*np.sin(self.theta_rays2)

        # now calculate the Z position that the rays enter the grid
        self.Zstart2 = grid.Ra[-1]*np.cos(self.theta_rays2)

        self.Zpos2 = np.array([]) # Z position of all cell crossings
        self.Xpos2 = np.array([]) # X position of all cell crossings

        # now for each ray store the cell positions one moves through and the possition 
        # one enters/exits cell walls and store in list
        rays_i2 = [[] for i in range(Nrays)]
        rays_j2 = [[] for i in range(Nrays)]
        rays_Z2 = [[] for i in range(Nrays)]
        rays_dZ2 = [[] for i in range(Nrays)]
        ray_len2 = [1 for i in range(Nrays)] 

        for i in range(Nrays):
            # find starting position and index
            jstart = np.asarray(grid.Ta < self.theta_rays2[i]).nonzero()[0][-1]
            istart = grid.NR+1 # enters in ghost cell

            rays_i2[i].append(istart)
            rays_j2[i].append(jstart)
            rays_Z2[i].append(self.Zstart2[i])
            ingrid = True
            while ingrid:

                # now move through grid calculating which cell is next
                # do this by calculating the r you cut the next theta 
                # in this quadrant we move backwards through the theta array
                # and the theta you cut the next r  
                r_cut_theta = self.Xrays2[i]/np.sin(grid.Ta[rays_j2[i][-1]])

                if rays_Z2[i][-1] > 0:
                    if r_cut_theta > grid.Ra[rays_i2[i][-1]]:
                        # exit into theta -1 on theta axis
                        rays_i2[i].append(rays_i2[i][-1])
                        rays_j2[i].append(rays_j2[i][-1]-1)
                        rays_Z2[i].append(self.Xrays2[i]/np.tan(grid.Ta[rays_j2[i][-1]]+1e-15)) # 1e-15 to prevent divide by inf
                        ray_len2[i]+=1 

                    else:
                        # exit into r-1
                        rays_i2[i].append(rays_i2[i][-1]-1)
                        rays_j2[i].append(rays_j2[i][-1])
                        rays_Z2[i].append(np.sqrt(grid.Ra[rays_i2[i][-1]+1]**2.-self.Xrays2[i]**2.))
                        ray_len2[i]+=1
                        if (rays_i2[i][-1]==-1):
                            # exitted the ghost cell at inner r boundary - ray finished
                            ingrid = False
                else:
                    if (r_cut_theta < grid.Ra[rays_i2[i][-1]+1]):
                        # exit into theta -1 on theta axis
                        rays_i2[i].append(rays_i2[i][-1])
                        rays_j2[i].append(rays_j2[i][-1]-1)
                        rays_Z2[i].append(self.Xrays2[i]/np.tan(grid.Ta[rays_j2[i][-1]]+1e-15))
                        ray_len2[i]+=1
                    else: 
                        # exit into r+1
                        rays_i2[i].append(rays_i2[i][-1]+1)
                        rays_j2[i].append(rays_j2[i][-1])
                        rays_Z2[i].append(-np.sqrt(grid.Ra[rays_i2[i][-1]]**2.-self.Xrays2[i]**2.))
                        ray_len2[i]+=1
                        if (rays_i2[i][-1]==(grid.NR+2)):
                            # exitted into ghost cell at inner r boundary - ray finished
                            ingrid = False

                rays_dZ2[i].append(-(rays_Z2[i][-1]-rays_Z2[i][-2]))

            # append to cell crossings
            self.Xpos2 = np.append(self.Xpos2,np.zeros(len(rays_Z2[i]))+self.Xrays2[i])
            self.Zpos2 = np.append(self.Zpos2,rays_Z2[i])

        # append second set of rays
        #self.Zpos = np.append(self.Zpos2,self.Zpos)
        #self.Xpos = np.append(self.Xpos2,self.Xpos)



        self.ray_i2 = rays_i2
        self.ray_j2 = rays_j2
        self.ray_Z2 = rays_Z2
        self.ray_length2 = ray_len2
        self.ray_dZ2 = rays_dZ2

        self.tau_end1 = np.zeros(self.Nrays) # value of optical depth at end of each ray - hemisphere 1
        self.tau_end2 = np.zeros(self.Nrays) # value of optical depth at end of each ray - hemisphere 2
        self.tau_pos1 = [np.array([]) for i in range(self.Nrays+1)] # list of optical depths along each ray
        self.tau_pos2 = [np.array([]) for i in range(self.Nrays+1)] # list of optical depths along each ray
        self.tau_rw  = np.zeros((grid.NR+3,grid.NTH+3))
        self.tau_tw  = np.zeros((grid.NR+3,grid.NTH+3))
        self.tau_b   = np.zeros((grid.NR+2,grid.NTH+2)) + 1e-20 # total optical depth to star 
        self.tau_b_gas = np.zeros((grid.NR+2,grid.NTH+2)) + 1e-20 # optical depth to star from gas only
        self.tau_b_par = np.zeros((grid.NR+2,grid.NTH+2)) +1e-20 # optical depth to star from particles only

        self.tau_b_par[:,grid.index_theta_max+1:] = 100.

        ## now copy "sub-stellar" rays from one hemisphere to the other for boudaries

        self.Zpos1 = np.append(self.ray_Z2[0],self.Zpos)
        self.Xpos1 = np.append(np.zeros(len(rays_Z2[0]))+self.Xrays2[0],self.Xpos)

        self.Zpos2 = np.append(self.ray_Z[0],self.Zpos2)
        self.Xpos2 = np.append(np.zeros(len(rays_Z[0]))+self.Xrays[0],self.Xpos2)

        # compute delaunay tesselation of all ray-crossings
        self.delaunay1 = Delaunay(np.array([self.Xpos1,self.Zpos1]).T)    # hemisphere 1
        self.delaunay2 = Delaunay(np.array([self.Xpos2,self.Zpos2]).T)  # hemisphere 2



    def do_ray_trace(self,rho_kappa):
    
        # this routine loops through all the rays and peforms a ray_tracing calculation

        #rho_kappa is density * opacity


        # hemisphere 1
        for r in range(self.Nrays):
            tau_ray = trace_single_ray(np.array(self.ray_i[r]),np.array(self.ray_j[r]),np.array(self.ray_dZ[r]),rho_kappa,self.ray_length[r])
            self.tau_end1[r]= tau_ray[-1]
    
            self.tau_pos1[r+1] = tau_ray

        
            
        

        #hemisphere2

        for r in range(self.Nrays):
            tau_ray = trace_single_ray(np.array(self.ray_i2[r]),np.array(self.ray_j2[r]),np.array(self.ray_dZ2[r]),rho_kappa,self.ray_length2[r])
            self.tau_end2[r]= tau_ray[-1]
    
            self.tau_pos2[r+1] = tau_ray

        ## copy over "sub-stellar" rays from one hemisphere to other for boundaries
        
        self.tau_pos1[0] = self.tau_pos2[1]
        self.tau_pos2[0] = self.tau_pos1[1]

        self.flat_tau1 = np.array([item for sublist in self.tau_pos1 for item in sublist])    
        self.flat_tau2 = np.array([item for sublist in self.tau_pos2 for item in sublist])
            
        return


    def get_tau_grid(self,grid):

        # this routine computes the optical depth at the cell-walled positions required for 
        # velocities

        get_tau1 = LinearNDInterpolator(self.delaunay1,(self.flat_tau1),fill_value=1e-20)
        get_tau2 = LinearNDInterpolator(self.delaunay2,(self.flat_tau2),fill_value=1e-20)

        #self.tau_rw[:,:grid.NTrw] = get_tau((grid.Xrw,grid.Zrw))
        #self.tau_tw[:-1,:grid.NTtw] = get_tau((grid.Xtw,grid.Ztw))
        self.tau_b_par[:] = 1e-20
        self.tau_b_par[grid.mry1] = get_tau1((grid.Xb[grid.mry1],grid.Zb[grid.mry1])) # interpolation on morning hemisphere
        self.tau_b_par[grid.mry2] = get_tau2((grid.Xb[grid.mry2],grid.Zb[grid.mry2])) # interpolation on evening hemisphere
        self.tau_b_par[grid.mns] = 100. # night-side opticaldeth
        #self.tau_b_par[-1,grid.jo//2+1:grid.index_theta_max+1] = get_tau((grid.Xb[-1,grid.jo//2+1:grid.index_theta_max+1],grid.Zb[-1,grid.jo//2+1:grid.index_theta_max+1])) # do exiting angles of grid
        #self.tau_b_par[:,0] = self.tau_b_par[:,1] # by reflection symmetry

        # set upper radii to have low optical depth to remove interpolation error

        #self.tau_rw[-30:,:50] = 1e-15
        #self.tau_tw[-30:,:50] = 1e-15
        
        ## periodic boundary
        self.tau_b_par[:,0]=self.tau_b_par[:,-2]
        self.tau_b_par[:,-1] = self.tau_b_par[:,1]


        return

    def get_tau_grid_analytic(self,grid,system):

        # this routine does a numerical integration of an analytic density structure

        # do b grid
        for i in range(grid.NR+2):
            for j in range(grid.NTH+2):
                Xpos = grid.Xb[i,j]
                Zstart = np.sqrt(grid.Ra[-1]**2.-Xpos**2.)
                Zend = grid.Zb[i,j]
                # check if hidden
                if (grid.mns[i,j]):
                    # position in full shadow
                    self.tau_b_gas[i,j] = 150. 
                else:
                    # do integration    
                    self.tau_b_gas[i,j] = -quad(analytic_rho_kappa,Zstart,Zend,args=(Xpos,system,grid.Rmin,system.kappa_star))[0] + 1e-10
                    if (self.tau_b_gas[i,j]> 150):
                        self.tau_b_gas[i,j] = 150.

        # initialise total optical depth with gas only
        self.tau_b = np.copy(self.tau_b_gas)


        return





def analytic_rho_kappa(Z,x,system,Rmin,kappa):

    kb = 1.38e-16
    G  = 6.67e-8
    mh = 1.67e-24

    rsph = np.sqrt(Z**2.+x**2.)

    # setup hydrostatic spherical atmosphere
    cs2 = kb * system.Teq / (system.mmw * mh)
    b = G * system.Mp / Rmin / cs2 # depth of potential


    gas_P    = system.Pbase * np.exp(b*(Rmin/rsph-1.))
    gas_dens = gas_P / cs2

    kappa_rho = kappa * gas_dens

    return kappa_rho
    

@jit(nopython=True)
def trace_single_ray(i_index,j_index,dZ,rho,NZ):
    
    tau_end = 0
    tau_ray = np.zeros(NZ)
    
    for k in range(1,NZ):
        tau_ray[k] = tau_ray[k-1] + rho[i_index[k-1],j_index[k-1]]*dZ[k-1]
    
    return tau_ray   














