import numexpr as ne
import numpy as np

# this function contains fits to evaluate variation radiation efficiencies

def get_Qpr_soot(size):
    
    # fit to the radiation pressure efficiency for soot 
    # use numexpr for rapid evaluation of large arrays
    # size is an array in cm

    a1 = 0.10547534
    a2 = 1.97684684
    a3 = 0.33246954
    a4 = 0.56791038
    a5 = 0.2357629
    a6 = 1.89598869
    a7 = 0.4833938
    
    Qpr = ne.evaluate("1./(1.+(size/(a1*1e-4))**(-1.)) + a2 /(exp(1e-4*a3/size)**a4+(size/(a5*1e-4))**a6)**a7")
   
    return Qpr

def get_Qext_soot(size):
    
    # fit to the extinction efficiency for soot 
    # use numexpr for rapid evaluation of large arrays
    # size is an array in cm

    a1 = 1.05371182e-01
    a2 = 2.98068059e+00
    a3 = 6.62107186e-01
    a4 = 3.80231376e+00
    a5 = 3.65485967e-03
    a6 = 2.65779924e+00
    a7 = 4.43209521e-02
    
    Qext = ne.evaluate("1./(1.+(size/(a1*1e-4))**(-1.)) + a2 /(exp(1e-4*a3/size)**a4+(size/(a5*1e-4))**a6)**a7")
   
    return Qext

def get_Qpr_tholins(size):
    
    # fit to the radiation pressure efficiency for tholins 
    # use numexpr for rapid evaluation of large arrays
    # size is an array in cm

    a1 = 1.15019735
    a2 = 1.94998013
    a3 = 0.60728454
    a4 = 1.01879291
    a5 = 0.57491265
    a6 = 1.9022088
    a7 = 0.25589791
    
    Qpr = ne.evaluate("1./(1.+(size/(a1*1e-4))**(-1.)) + a2 /(exp(1e-4*a3/size)**a4+(size/(a5*1e-4))**a6)**a7")
   
    return Qpr
    
def get_Qext_tholins(size):
    
    # fit to the extinction efficiency for soot 
    # use numexpr for rapid evaluation of large arrays
    # size is an array in cm

    a1 = 1.15021488
    a2 = 5.43945747
    a3 = 1.81603096
    a4 = 3.02707175
    a5 = 0.00908456
    a6 = 5.38647476
    a7 = 0.03847449
    
    Qext = ne.evaluate("1./(1.+(size/(a1*1e-4))**(-1.)) + a2 /(exp(1e-4*a3/size)**a4+(size/(a5*1e-4))**a6)**a7")
   
    return Qext
    


def get_Qpr_none(size):

    shape = np.shape(size)

    return np.zeros(shape)