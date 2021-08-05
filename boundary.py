import numpy as np
# this is a boundary update file

def update_boundary(grid,field,args=()):

    # this routine updates the boundries

    # inner r
    field.par_dens[grid.ii-1,:,:] = field.par_dens[grid.ii,:,:]
    field.par_vth[grid.ii-1,:,:] = 0.
    #field.par_vr[grid.ii,:,:] = 0.
    field.par_size[grid.ii-1,:,:] = field.par_size[grid.ii,:,:]

    # outer r - outflow
    field.par_dens[grid.io+1,:,:] = 0.
    #field.par_dens[grid.io+1,:,:] = field.par_dens[grid.io,:,:]
    field.par_vr [grid.io+1,:,:] = field.par_vr[grid.io,:,:]
    field.par_vr [grid.io+2,:,:] = field.par_vr[grid.io+1,:,:]
    field.par_vr_diff [grid.io+1,:,:] = field.par_vr_diff[grid.io,:,:]
    field.par_vth [grid.io+1,:,:] = field.par_vth[grid.io,:,:]
    field.par_vt_diff [grid.io+1,:,:] = field.par_vt_diff[grid.io,:,:]

    # inner th and outer th assumed to be refection

    field.par_dens[:,grid.ji-1,:]=field.par_dens[:,grid.ji,:]
    field.par_vr[:,grid.ji-1,:] = field.par_vr[:,grid.ji,:]
    field.par_vth[:,grid.ji-1:grid.ji,:] = 0. 

    field.par_vr_diff[:,grid.ji-1,:] = field.par_vr_diff[:,grid.ji,:]
    field.par_vt_diff[:,grid.ji-1:grid.ji,:] = 0. 
    
    field.par_dens[:,grid.jo+1,:] = field.par_dens[:,grid.jo,:]
    field.par_vr[:,grid.jo+1,:] = field.par_vr[:,grid.jo,:]
    field.par_vth[:,grid.jo+1:,:] = 0.

    field.par_vr_diff[:,grid.jo+1,:] = field.par_vr_diff[:,grid.jo,:]
    field.par_vt_diff[:,grid.jo+1:,:] = 0.

    # floor
    field.par_dens[field.par_dens < 1e-40] = 1e-40

    return