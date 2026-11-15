import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, CubicSpline

from scipy.integrate import solve_ivp, simpson

from numpy.linalg import inv
import scipy.linalg as la

from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator


import DESI_BAO






# Hubble data
data_H = np.loadtxt("hz_cc_sdss.txt")  # Here, no-covariance matrix has been used. However, in this sample, 15 data points are highly correlated. I will update the data set and corresponding covariance matrix. 
z_dataH = data_H[:, 0]
H_obs_data = data_H[:,1]
H_err = data_H[:, 2]
h_err_use= H_err**2  # this must be 1/sigma^2 other wise it will be wrong. 
H_diag=np.diag(h_err_use)
H_inv = la.inv(H_diag)

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
inv_cov = np.zeros((n, n))
inv_cov[np.triu_indices(n)] = data_sn[:]
i_lower = np.tril_indices(n, -1)
inv_cov[i_lower] = inv_cov.T[i_lower]
C = np.linalg.inv(inv_cov)





def equation(t,variable,params):
    od,om,h,dl = variable   # od is dark energy density, om is dark matter density, ob is baryon energy density, ora is radiation energy density

    wd = -1

    doth = -3/2 * (1/h**2 * (wd * od) + 1)


    eqd =  - 3* (1+ wd) * od
    eqm = - 3 * om 
    
    

    eqh = h * doth   # h = H/H0

    eqdl = - dl - (1 / h) * 2.99792458e5 * np.exp(- 2* t)

    var = np.array([eqd, eqm, eqh, eqdl])  # doubt( How re these will be calculated?)

    return var


def ode_sol(params):
    om0, H0 = params


    

    

    od0 = 1- om0


    tmax = 900

    yi = np.array([od0, om0, 1, 0]) # doubt( explain the initial condition)

    sol = solve_ivp(lambda t, y: equation(t, y, params),
            [0, -6], yi, t_eval=np.linspace(0.0,-6,tmax), method='BDF', rtol = 1e-6, atol = 1e-3)

    t_sol = sol.t

    zz = np.exp(-t_sol)- 1

    odsol, omsol, h_sol, dlh0_sol = sol.y 

    hhsol = H0 * h_sol

    dlsol =  dlh0_sol/H0 


    H_val = CubicSpline(zz, hhsol)
    dl_val=CubicSpline(zz, dlsol)

    res_hubble = np.zeros(len(z_dataH))

    res_hubble = H_val(z_dataH) -  H_obs_data

    chi_hubble = -0.5 * (res_hubble.T @ H_inv @ res_hubble)
       


    def chi_sn_des(dl_val, redshift):   # defining chi-square for Sne


        res_sn = np.zeros(len(redshift))

        mu_values_sn = 5 * np.log10((1+Redshift_1) * dl_val(redshift)/(1+redshift)) + 25



        res_sn = dist_mod - mu_values_sn
        res_sum = (res_sn.T @ inv_cov @ res_sn)
        B = np.sum(res_sn @ inv_cov)
        C = np.sum(inv_cov)  # doubt
        chi2 = res_sum - (B**2 / C) + np.log(C / (2 * np.pi))
        return - 0.5*chi2

    chi_bao_desi = DESI_BAO.desi_bao(dl_val, H_val)

    chi_tot =  chi_bao_desi + chi_sn_des(dl_val, Redshift) + chi_hubble

    return chi_tot


def log_prob( params):   
    
    
    res = ode_sol(params)

    if np.any(np.isinf(res)) and np.any(np.isnan(res)):

        return -np.inf
    
    
    
    return  res