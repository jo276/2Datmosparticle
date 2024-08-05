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
    
    # fit to the radiation pressure efficiency for MgSio4
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
    
    # fit to the extinction efficiency for MgSio4
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

def get_Qext_silFe(size,Tstar):
    
    # fit to the extinction efficiency for Iron Rich Silicates
    # use numexpr for rapid evaluation of large arrays
    # size is an array in cm

    a1 = 1.96102911e-01
    a2 = 5.56783052e+00
    a3 = 2.01369206e+00
    a4 = 1.64474071e+00
    a5 = 3.10609313e-04
    a6 = 2.32237854e+00
    a7 = 6.33446529e-02
    a8 = 1.33380840e+00

    Tfactor = Tstar / 5777.
    
    Qpr = ne.evaluate("1./(1.+(size*Tfactor/(a1*1e-4))**(-a8)) + a2 /(exp(1e-4*a3/size/Tfactor)**a4+(size*Tfactor/(a5*1e-4))**a6)**a7")
   
    return Qpr

def get_Qpr_silFe(size,Tstar):
    
    # fit to the radiation pressure efficiency for Iron Rich Silicates
    # use numexpr for rapid evaluation of large arrays
    # size is an array in cm

    a1 = 0.25250472
    a2 = 2.39379625
    a3 = 0.51461849
    a4 = 0.41928208
    a5 = 0.31280241
    a6 = 1.4482268
    a7 = 0.77682399
    a8 = 1.21686716

    Tfactor = Tstar / 5777.
    
    Qext = ne.evaluate("1./(1.+(size*Tfactor/(a1*1e-4))**(-a8)) + a2 /(exp(1e-4*a3/size/Tfactor)**a4+(size*Tfactor/(a5*1e-4))**a6)**a7")
   
    return Qext


def get_Qpr_al(size,Tstar):
    
    # fit to the radiation pressure efficiency for corundum
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
    
    # fit to the extinction efficiency for corundum 
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


def get_Qext_tholins(size,Tstar):
    
    # fit to the extinction efficiency for tholins
    # use numexpr for rapid evaluation of large arrays
    # size is an array in cm

    a1 = 2.38294902e-01
    a2 = 5.91737671e+00
    a3 = 2.58515448e+00
    a4 = 2.58515195e+00
    a5 = 1.45082334e-04
    a6 = 3.76766520e+00
    a7 = 3.73838983e-02
    a8 = 1.52945694e+00

    Tfactor = Tstar / 5777.
    
    Qext = ne.evaluate("1./(1.+(size*Tfactor/(a1*1e-4))**(-a8)) + a2 /(exp(1e-4*a3/size/Tfactor)**a4+(size*Tfactor/(a5*1e-4))**a6)**a7")
   
    return Qext


def get_Qpr_tholins(size,Tstar):

    # fit to the radiation pressure efficiency for tholins
    # use numexpr for rapid evaluation of large arrays
    # size is an array in cm

    a1 = 4.73264349
    a2 = 1.83301403
    a3 = 0.65091085
    a4 = 0.64829428
    a5 = 0.439601
    a6 = 1.96553458
    a7 = 0.36181274
    a8 = 0.73242457

    Tfactor = Tstar / 5777.
    
    Qpr = ne.evaluate("1./(1.+(size*Tfactor/(a1*1e-4))**(-a8)) + a2 /(exp(1e-4*a3/size/Tfactor)**a4+(size*Tfactor/(a5*1e-4))**a6)**a7")
   
    return Qpr
    


def get_Qpr_none(size,T):

    shape = np.shape(size)

    return np.zeros(shape)