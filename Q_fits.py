import numexpr as ne
import numpy as np

# this function contains fits to evaluate variation radiation efficiencies

def get_Qpr_soot(size,Tstar):
    
    # fit to the radiation pressure efficiency for soot 
    # use numexpr for rapid evaluation of large arrays
    # size is an array in cm

    a1 = 0.08903781
    a2 = 1.85504082
    a3 = 0.42252557
    a4 = 0.42735781
    a5 = 0.27721711
    a6 = 1.92360795
    a7 = 0.53452611
    a8 = 1.03883163

    Tfactor = Tstar / 5777.
    
    Qpr = ne.evaluate("1./(1.+(size*Tfactor/(a1*1e-4))**(-a8)) + a2 /(exp(1e-4*a3/size/Tfactor)**a4+(size*Tfactor/(a5*1e-4))**a6)**a7")
   
    return Qpr

def get_Qext_soot(size,Tstar):
    
    # fit to the extinction efficiency for soot 
    # use numexpr for rapid evaluation of large arrays
    # size is an array in cm

    a1 = 7.99220460e-02
    a2 = 2.97358796e+00 
    a3 = 3.06748456e+00
    a4 = 3.06755168e+00
    a5 = 2.92043294e-03
    a6 = 8.68508912e+00
    a7 = 1.29907879e-02
    a8 = 1.07572231e+00

    Tfactor = Tstar / 5777.
    
    Qext = ne.evaluate("1./(1.+(size*Tfactor/(a1*1e-4))**(-a8)) + a2 /(exp(1e-4*a3/size/Tfactor)**a4+(size*Tfactor/(a5*1e-4))**a6)**a7")
   
    return Qext


def get_Qpr_sil(size,Tstar):
    
    # fit to the radiation pressure efficiency for soot 
    # use numexpr for rapid evaluation of large arrays
    # size is an array in cm

    a1 = 9.25116964e+01
    a2 = 1.57191630e+00
    a3 = 3.21467286e+00
    a4 = 3.21464224e+00
    a5 = 2.26033616e-01
    a6 = 2.37602252e+01
    a7 = 1.52624394e-02
    a8 = 9.09503651e-01

    Tfactor = Tstar / 5777.
    
    Qpr = ne.evaluate("1./(1.+(size*Tfactor/(a1*1e-4))**(-a8)) + a2 /(exp(1e-4*a3/size/Tfactor)**a4+(size*Tfactor/(a5*1e-4))**a6)**a7")
   
    return Qpr

def get_Qext_sil(size,Tstar):
    
    # fit to the extinction efficiency for soot 
    # use numexpr for rapid evaluation of large arrays
    # size is an array in cm

    a1 = 1.03661207e-01
    a2 = 1.03085363e+01 
    a3 = 1.68585708e+00
    a4 = 1.73257963e+00
    a5 = 1.98631405e-07
    a6 = 8.30987144e-01
    a7 = 1.46615953e-01
    a8 = 3.83631723e+00

    Tfactor = Tstar / 5777.
    
    Qext = ne.evaluate("1./(1.+(size*Tfactor/(a1*1e-4))**(-a8)) + a2 /(exp(1e-4*a3/size/Tfactor)**a4+(size*Tfactor/(a5*1e-4))**a6)**a7")
   
    return Qext


def get_Qpr_al(size,Tstar):
    
    # fit to the radiation pressure efficiency for soot 
    # use numexpr for rapid evaluation of large arrays
    # size is an array in cm

    a1 = 0.85725765
    a2 = 1.71222671
    a3 = 0.64902685
    a4 = 0.57216299
    a5 = 0.46364614
    a6 = 1.91410129
    a7 = 0.50867936
    a8 = 1.10130509

    Tfactor = Tstar / 5777.
    
    Qpr = ne.evaluate("1./(1.+(size*Tfactor/(a1*1e-4))**(-a8)) + a2 /(exp(1e-4*a3/size/Tfactor)**a4+(size*Tfactor/(a5*1e-4))**a6)**a7")
   
    return Qpr

def get_Qext_al(size,Tstar):
    
    # fit to the extinction efficiency for soot 
    # use numexpr for rapid evaluation of large arrays
    # size is an array in cm

    a1 = 7.64180442e-01
    a2 = 5.48314885e+00 
    a3 = 4.31168459e+00
    a4 = 4.31209768e+00
    a5 = 4.35843575e-03
    a6 = 1.43032870e+01
    a7 = 1.31509925e-02
    a8 = 1.12640594e+00

    Tfactor = Tstar / 5777.
    
    Qext = ne.evaluate("1./(1.+(size*Tfactor/(a1*1e-4))**(-a8)) + a2 /(exp(1e-4*a3/size/Tfactor)**a4+(size*Tfactor/(a5*1e-4))**a6)**a7")
   
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
    


def get_Qpr_none(size,T):

    shape = np.shape(size)

    return np.zeros(shape)