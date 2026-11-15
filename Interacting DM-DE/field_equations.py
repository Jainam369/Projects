
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.interpolate import interp1d
from scipy import integrate
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import numpy as np
from numpy.linalg import inv, pinv
import multiprocessing





# DES-Dovekie datasets
DES_Data = np.loadtxt("DES_file (1).txt")
data_sn = np.loadtxt('DES_COV (1).txt')

# Desired Columns
Redshift = DES_Data[:,0]
Redshift_1 = DES_Data[:,1]
dist_mod =  DES_Data[:,2]
dist_mod_err = DES_Data[:,3]



# Building Covariance matrix
n = len(Redshift)
inv_cov_des = np.zeros((n, n))
inv_cov_des[np.triu_indices(n)] = data_sn[:]
i_lower = np.tril_indices(n, -1)
inv_cov_des[i_lower] = inv_cov_des.T[i_lower]
C = np.linalg.inv(inv_cov_des)





# DESI BAO DR2
z_desi_bao_eff = np.array([0.295,0.510, 0.706, 0.934, 1.321, 1.484, 2.330])  # this is the redshift we are using to calculate the observables. 
dat_bao=np.loadtxt('desi_mean.txt')
dat_bao = dat_bao[:,1]
cov_bao=np.loadtxt("desi_cov.txt") # this is covariance matrix. 
inv_cov_bao = inv(cov_bao) #Here, we are taking the inverse of it. 




# CC and SDSS BA0 Hubble data

data_H = np.loadtxt("hz_cc_sdss.txt")  
z_dataH = data_H[:, 0]
H_obs_data = data_H[:,1]
H_err = data_H[:, 2]
h_err_use= H_err**2  
H_diag=np.diag(h_err_use)
H_inv = inv(H_diag)    

# CMB Data
# Here the data structure is [100*Omega_{b0}h^2, 100 theta_* , R, Omega_{c0}h^2]
data_cmb = np.array([2.225,1.0414,1.7529,0.1184])

data_cmb_sigma_gen = np.array([0.0223,0.00054,0.0056, 0.0029])


inv_cmb_gen = np.array(
    [[ 8.36144553e+03, -1.94826450e+04,  2.42268487e+04, -2.68410558e+04],
 [-1.94826450e+04,  1.07036647e+07, -2.12229562e+05,  1.61294718e+06],
 [ 2.42268487e+04, -2.12229562e+05,  1.05554887e+05, -8.82933658e+04],
 [-2.68410558e+04 , 1.61294718e+06, -8.82933658e+04 , 4.35051678e+05]])



#Constants
c = 299792.458
Mpl = 1.0  # Reduced Planck mass                 M_pl^2 = 1/8PiG = 1
f0 = 1.0
Omega_b0 = 0.049
Omega_r0 = 9e-5
#Omega_m0 = 0.3153






#  Define Coupling, potential functions and their derrivatives

def f(phi,beta,gamma):

    return 1 + beta*phi - gamma*phi**2

def df(phi,beta,gamma):

    return beta - 2*gamma*phi

def V(phi,alpha,V0):

    return V0 + alpha*phi

def dV(phi,alpha):
    return alpha


# define Hubble Parameter as a funtion of N(=lna)

def H(phi,phiN,a,H0,Omega_m0,alpha,beta,gamma,obh,V0):


    Omega_dm0 = Omega_m0 - Omega_b0


    rho_crit0 = 3 * H0**2          # present critical density


    rho_dm0 = rho_crit0*Omega_dm0  # present dark matter density
    rho_b0  = rho_crit0*Omega_b0   # present baryona density
    rho_r0  = rho_crit0*Omega_r0   # present radiation density


    f0 = 1.0

    rho_dm = (
        rho_dm0

        *f(phi,beta,gamma)

        /f0

        /a**3
    )


    rho_b = rho_b0/a**3

    rho_r = rho_r0/a**4


    numerator = (

        V(phi,alpha,V0) + rho_dm + rho_b + rho_r
        
    )


    den = 3.0 * (1.0 - phiN**2 / 6.0)


    if den <=0:

        return np.nan


    arg = numerator / den


    if arg <=0:

        return np.nan


    return np.sqrt(arg)        

    


def epsilon_H(phi,phiN,a,H0,Omega_m0,alpha,beta,gamma,obh,V0):

    Omega_dm0 = Omega_m0 - Omega_b0


    rho_crit0 = 3 * H0**2 


    rho_dm0 = rho_crit0*Omega_dm0

    rho_b0  = rho_crit0*Omega_b0

    rho_r0  = rho_crit0*Omega_r0


    f0 = 1.0

    rho_dm = (
        rho_dm0

        *f(phi,beta,gamma)

        /f0

        /a**3
    )


    rho_b = rho_b0/a**3

    rho_r = rho_r0/a**4





    Hval = H(phi,phiN,a,H0,Omega_m0,alpha,beta,gamma,obh,V0)


    if not np.isfinite(Hval):

       return np.nan


    Omega_dm = rho_dm / ( 3 * Hval**2)

    Omega_b = rho_b / ( 3 * Hval**2)

    Omega_r = rho_r / ( 3 * Hval**2)


    f0 = 1


    Omega_m = Omega_dm + Omega_b

    


    return (

        0.5*phiN**2

        + 1.5*(Omega_m)

        + 2*Omega_r
    )



# define equations to be evaluate

def equations(N,y,H0,Omega_m0,alpha,beta,gamma,obh,V0):


    phi,phiN = y


    a = np.exp(N)


    Hval = H(phi,phiN,a,H0,Omega_m0,alpha,beta,gamma,obh,V0)


    if not np.isfinite(Hval):

        return [np.nan,np.nan]


    epsH = epsilon_H(phi,phiN,a,H0,Omega_m0,alpha,beta,gamma,obh,V0)


    if not np.isfinite(epsH):

        return [np.nan,np.nan]


    f0 = 1

    Omega_dm0 = Omega_m0 - Omega_b0

    rho_dm0 = (

        3*H0**2

        *Omega_dm0
    )


    phiNN = (

        -(3-epsH)*phiN

        -(dV(phi,alpha)

          + rho_dm0*df(phi,beta,gamma)/(f0 * a**3)

         )/Hval**2
    )


    if not np.isfinite(phiNN):

        return [np.nan,np.nan]


    return [phiN,phiNN]


# Solving the ODE

def integrate_model(phi_ini,H0,Omega_m0,alpha,beta,gamma,obh,V0):


    y0 = [phi_ini,0.0]   # initial condition


    if not np.all(np.isfinite(y0)):

        return None

    


    a_ini = 1e-7

    a_final = 1


    N_ini = np.log(a_ini)

    N_final = np.log(a_final)

    N_eval = np.linspace(N_ini, N_final, 500)


    try :

        sol = solve_ivp(

        lambda N,y: equations(N,y,H0,Omega_m0,alpha,beta,gamma,obh,V0),

        [N_ini,N_final],

        y0,

        t_eval=N_eval,method='RK45',

        rtol=1e-5,

        atol=1e-7)

    except Exception as e:


        print("Integration crashed:")
        print(e)


        return None    

    if (not sol.success or

        np.any(~np.isfinite(sol.y))):


        #print("Integration failed:")
        #print(sol.message)


        return None


    return sol             
    

    





def shooting(vars,

             H0, Omega_m0, 

             alpha, beta, gamma,obh):


    phi_ini, V0 = vars



    if not np.isfinite(phi_ini):
      return [1e30, 1e30]

    if not np.isfinite(V0):
      return [1e30, 1e30]

    if V0 < 0:
      return [1e30, 1e30]

    if abs(phi_ini) > 20:
      return [1e30, 1e30]


    Omega_de_0 = 1.0 - Omega_m0


    rho_DE_target = (

    3*H0**2*Omega_de_0            # Present normalised dark energy density
)


    sol = integrate_model(
        phi_ini,

        H0, Omega_m0,

        alpha, beta, gamma,obh,

        V0
    )


    if sol is None:

        return [1e30, 1e30]


    phi0 = sol.y[0, -1]

    phiN0 = sol.y[1, -1]


    H0_model = H(
        phi0,

        phiN0,

        1.0,

        H0, Omega_m0,

        alpha, beta, gamma,obh,

        V0
    )


    if not np.isfinite(H0_model):

        return [1e30, 1e30]

    rho_phi0 = (

        0.5 * H0_model**2 * phiN0**2

        + V(phi0, alpha, V0)
    )


    if not np.isfinite(rho_phi0):

        return [1e30, 1e30]


    return [
        phi0,

        rho_phi0 - rho_DE_target

    ]


    


_shooting_cache = {}


def solve_background(H0,Omega_m0, alpha, beta, gamma,obh):


    global _shooting_cache


    pid = multiprocessing.current_process().pid


    guess = _shooting_cache.get(
        pid,

        np.array([0.5, 9300])   # default for first call
    )

    if not np.all(np.isfinite(guess)):
        guess = np.array([0.5, 9300])


    try:
        trial = root(

            lambda vars: shooting(

                vars,

                H0, Omega_m0,
                alpha,

                beta,
                gamma,obh
            ),

            guess,

            method='hybr'
        )


    except Exception:

        return None, None


    if not trial.success:

        return None, None


    # Save successful solution

    _shooting_cache[pid] = trial.x.copy()


    phi_ini, V0 = trial.x

    sol = integrate_model(
        phi_ini,

        H0, Omega_m0,
        alpha,beta,
        gamma,obh,

        V0
    )


    if sol is None:

        return None, None


    return sol, V0



#print(solve_background(67,0.3153, -4.66, 0.180, 0.150,0.022))



def background(H0,Omega_m0,alpha,beta,gamma,obh):



    sol,V0 = solve_background(

        H0, Omega_m0,

        alpha,beta,gamma,obh
    )


    if sol is None:

      return None, None



    phi  = sol.y[0]

    phiN = sol.y[1]


    N = sol.t


    a = np.exp(N)


    z = 1/a - 1


    H_array = np.array([

        H(

            phi[i],

            phiN[i],

            a[i],

            H0, Omega_m0,

            alpha,beta,gamma,obh,

            V0
        )
        for i in range(len(a))

    ])


    idx = np.argsort(z)




    return z[idx],H_array[idx]

   





def Model_data(H0, Omega_m0,alpha,beta,gamma,obh):


    z_bg, H_bg = background(

    H0, Omega_m0,

    alpha,beta,gamma,obh)


    if z_bg is None:

      return -np.inf


    H_of_z = interp1d(

    z_bg,

    H_bg,

    kind='cubic',

    fill_value='extrapolate')


    mask = z_bg <= 5

    #print(H_of_z(1090))


    z_use = z_bg[mask]

    H_use = H_bg[mask]






    numerical_integration = integrate.cumulative_trapezoid(

    1/H_use,

    z_use,
    initial=0)


    dL = c*(1+z_use)*numerical_integration



    dL_of_z = interp1d(

    z_use,

    dL,

    kind='cubic',

    fill_value='extrapolate')


    def chi_sn_des(dL_of_z, redshift):   # defining chi-square for Sne



        res_sn = np.zeros(len(redshift))


        mu_values_sn = 5 * np.log10((1+Redshift_1) * dL_of_z(redshift)/(1+redshift)) + 25




        res_sn = dist_mod - mu_values_sn

        res_sum = (res_sn.T @ inv_cov_des @ res_sn)

        B = np.sum(res_sn @ inv_cov_des)

        C = np.sum(inv_cov_des)

        chi2 = res_sum - (B**2 / C) + np.log(C / (2 * np.pi))

        return - 0.5*chi2


    def desi_bao(dL_of_z, H_of_z):


        rd = 147  # Here, you can simply vary the rd in the uniform range [100,300].


        dm_for_dv = dL_of_z(0.29500000)/(1+0.29500000)

    

        dv0 = (0.29500000 * dm_for_dv**2 *2.99792458e5/H_of_z(0.29500000) )**(0.333)/rd


        dm05 = dL_of_z(0.51000000)/(1+0.51000000)/rd

        dh05 = 2.99792458e5/(rd* H_of_z(0.51000000))


        dm07 = dL_of_z(0.70600000)/(1+0.70600000)/rd

        dh07 = 2.99792458e5/(rd* H_of_z(0.70600000))


        dm09 = dL_of_z(0.93400000)/(1+0.93400000)/rd

        dh09 = 2.99792458e5/(rd* H_of_z(0.93400000))


        dm13 = dL_of_z(1.32100000)/(1+1.32100000)/rd

        dh13 = 2.99792458e5/(rd* H_of_z(1.32100000))


        dm14 = dL_of_z(1.48400000)/(1+1.48400000)/rd

        dh14 = 2.99792458e5/(rd* H_of_z(1.48400000))


        dm23 = dL_of_z(2.33000000)/(1+2.33000000)/rd

        dh23 = 2.99792458e5/(rd* H_of_z(2.33000000))

    

        val = np.array([dv0, dm05, dh05, dm07,dh07, dm09, dh09, dm13, dh13, dm14, dh14, dh23, dm23])



        res = dat_bao - val


        chi_bao = -0.5* (res @ inv_cov_bao @ res )    



        return chi_bao  

    def planck_chi(dL_of_z,H_of_z,H0,Omega_m0,alpha, beta, gamma,obh):

          # Here dmh \Omega_dm h^2

        hh = H0/100  
        #ob0 = obh/hh**2
        
        calc_obh =100* obh

        #omh = Omega_m0 * hh**2
        Omega_dm0 = Omega_m0 - Omega_b0
        dmh = Omega_dm0 * hh**2

        
        rs= 147

    

        zs = 1090  #this can either be fixed and must be set TT+EE+lowE data for flat LambdaCDM plank 18.  

    # https://doi.org/10.1016/j.dark.2024.101641

        da = dL_of_z(zs)/(1+zs)

    # la= (1+zs) * np.pi * da/rs  #accoustic length

        theta = rs / da 

        Rr = (da* H0* np.sqrt(Omega_m0)) / 2.99792458e5  #shift parameter


    
        shift_val = Rr 

        theta_val = 100 * theta
    
    # ob_val = ob0 * h**2

    # eval_val = np.array([shift_val, la_val, ob_val])

        eval_val = np.array([calc_obh, theta_val, shift_val, dmh])

    

        res9 = eval_val - data_cmb

    # res_cmb = solve_triangular(chol_cmb, res9, lower=True, check_finite=False)

        chi_cmb = -0.5 * (res9.T @inv_cmb_gen @ res9)
    
        return  chi_cmb
     


    res_hubble = np.zeros(len(z_dataH))


    res_hubble = H_of_z(z_dataH) -  H_obs_data


    chi_hubble = -0.5 * (res_hubble.T @ H_inv @ res_hubble)     





    chi_tot = chi_sn_des(dL_of_z, Redshift) + chi_hubble + desi_bao(dL_of_z,H_of_z) + planck_chi(dL_of_z,H_of_z,H0,Omega_m0,alpha, beta, gamma,obh)


    return chi_tot




def log_prob(H0, Omega_m0,alpha, beta, gamma,obh):


    try:

        res = Model_data(H0, Omega_m0, alpha, beta, gamma,obh)


    except Exception:

        return -np.inf


    if not np.isfinite(res):

        return -np.inf


    return float(res)

print(log_prob(67,0.3,-4.66,0.180,0.150,0.022) )   



























   










