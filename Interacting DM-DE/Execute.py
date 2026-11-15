import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt

from scipy import integrate
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d 
from matplotlib import rc
plt.rcParams.update({'font.size': 12})
from multiprocessing import Pool, cpu_count
from multiprocessing import freeze_support

from tqdm import tqdm
import getdist
from getdist import plots, MCSamples

import arviz as az

import emcee



from datetime import datetime



import field_equations




data=np.loadtxt("DES_file (1).txt")
z_data_sn=data[:,0]

data_H = np.loadtxt("hz_cc_sdss.txt")
z_dataH = data_H[:, 0]


 

##############################

z_desi_bao =field_equations.z_desi_bao_eff

N= len(z_dataH) + len(field_equations.z_desi_bao_eff) + len(z_data_sn)

file_name= "DM-DE_BAO+CC+DES"
label_fig="CC+BAO+DES"




    





def liklihood(params):

    H0,Omega_m0,alpha, beta, gamma = params
    
    
    
    if not 30 < H0 < 100: 
        return -np.inf

    if not 0 < Omega_m0 < 0.7: 
        return -np.inf

    if not -7 < alpha < -1:
        return -np.inf       
    
    if not 0.1< beta < 0.5:
        return -np.inf

    if not 0.1 < gamma < 0.5:
        return -np.inf    
    
 
    
    res = field_equations.log_prob(H0,Omega_m0,alpha,beta,gamma)
    
    if not np.isfinite(res):
      return -np.inf

    return float(res)


ncpu = cpu_count()

print("{0} CPUs not all in used.".format(ncpu))



print(f"Now we are doing {file_name} analysis with LambdaCDm:", N)


name = ['H0','Omega_m0','alpha', 'beta', 'gamma']
labels1 = [ r'H_0',r'\Omega_m0' ,r'\alpha', r'\beta', r'\gamma']


if __name__ == "__main__":

        nwalker = 40
        ndim = len(name)
        niter = 10000
        best = np.array([70.0, 0.3,-4.0, 0.20, 0.20])

        # Small perturbations around it
        scatter = np.array([1.0, 0.01,0.2, 0.02, 0.02])

        p0 = best + scatter*np.random.randn(nwalker, ndim)
        #p0 = np.random.uniform(low=[30,0,-7,0.1,0.1], high=[100,0.7,-1,0.5,0.5], size=(nwalker, ndim))

        #print("Testing likelihood at initial guess...")
        #print(liklihood(best))

        move = [
        (emcee.moves.StretchMove(a=2.0),0.8),
        (emcee.moves.DEMove(), 0.2),
        (emcee.moves.DESnookerMove(), 0.0)
        # (emcee.moves.KDEMove(bw_method=None), 0.30)
                                                ]


        # Save chain automatically
                                              
        with Pool(processes=4) as pool:  #set the processes according to the number of cores you have. 
           sampler = emcee.EnsembleSampler(nwalker, ndim, liklihood,pool=pool,live_dangerously=None,  moves=move)
           sampler.run_mcmc(p0,niter,progress =True)

           print("\nAcceptance fractions:")
           print(sampler.acceptance_fraction)

           print("\nMean acceptance fraction:")
           print(np.mean(sampler.acceptance_fraction))




          

        dis = 3000
        thi = 1

        samples = sampler.get_chain(
           discard=dis,
           thin=thi,
           flat=True
                        )

        log_prob = sampler.get_log_prob(
        discard=dis,
        thin=thi,
        flat=True
                )

        np.savez(
           f"{file_name}_chains_10000e.npz",
        chain=sampler.get_chain(),
        log_prob=sampler.get_log_prob())         



        sample2 = MCSamples(samples=samples,names=name, labels=labels1)

        sample2.smooth_scale_1D = 0.8
        sample2.smooth_scale_2D = 0.8

        



        



        print(f"Here is the result for {file_name} on:")

        az_data = az.from_emcee(
            sampler,
            var_names=name
                             )

        r_hat = az.rhat(az_data)

        print("Gelman-Rubin statistic (R_hat):")
        print(r_hat)

        try:
            tau = sampler.get_autocorr_time(discard = dis)
            
            print("Autocorrelation time:")
            print(tau)

            eff_samples = (
               nwalker*(niter-dis)/tau
                )

            print("Effective samples:")
            print(eff_samples)
        except Exception:

            print(
                "Chain too short to estimate autocorrelation time."
                          )    



        imax = np.argmax(log_prob)

        max_loglike = log_prob[imax]

        max_likelihood_params = samples[imax]

        print("Maximum log likelihood:")
        print(max_loglike)

        print("Best-fit parameters:")
        print(max_likelihood_params)

        chi2val = -2* max_loglike
        print("Min chi_square: ", chi2val)

        reduced = chi2val/(N-ndim)

        print("Reduced chi_squre is: ", reduced)

        print(f"Parameters at Maximum Likelihood: {max_likelihood_params}\n")




        


        


   

        g = plots.get_subplot_plotter(width_inch=6)  # Adjust width of the plot in 

        g.settings.figure_legend_frame = True  # Remove frame around the legend
        g.settings.num_plot_contours = 2
        
        g.settings.alpha_filled_add = 0.6  # Adjust transparency of filled contours
        g.settings.title_limit_fontsize = 14  # Limit title fontsize to 14
        g.settings.axes_labelsize = 12  # Adjust label fontsize
        g.settings.legend_fontsize = 10  # Adjust legend fontsize
        g.settings.colorbar_axes_fontsize = 10  # Adjust colorbar fontsize
        g.triangle_plot(sample2, name, filled=True, legend_labels=[f'{label_fig}'],legend_loc='upper right', contour_colors=['orange'],title_limit=1)  

        g.export(f'figure/{file_name}_10000e.pdf')