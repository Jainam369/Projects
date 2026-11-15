import numpy as np
from getdist import plots, MCSamples
import matplotlib.pyplot as plt

file_name = "DM-DE_BAO+CC+DES"

name = ['H0','Omega_m0','alpha','beta','gamma']
labels = [r'H_0', r'\Omega_m0',r'\alpha', r'\beta', r'\gamma']

data = np.load(f"{file_name}_chains_10000n.npz")


chain = data["chain"]

log_prob = data["log_prob"]

dis = 4000
thin = 1







samples = chain[dis::thin,:,:].reshape(-1, len(name))

sample2 = MCSamples(
    samples=samples,
    names=name,
    labels=labels
)




print(log_prob.shape)

imax = np.argmax(log_prob)

#max_loglike = log_prob[imax]

max_likelihood_params = samples[imax]

print("Maximum log likelihood:")
#print(max_loglike)

print("Best-fit parameters:")
print(max_likelihood_params)
print(f"Parameters at Maximum Likelihood: {max_likelihood_params}\n")
#g.export(f"{file_name}.pdf")







label_fig="CC+BAO+DES"
#sample2.smooth_scale_1D = 1.0
sample2.smooth_scale_2D = 0.8


g = plots.get_subplot_plotter(width_inch=6)  # Adjust width of the plot in 

g.settings.figure_legend_frame = True  # Remove frame around the legend
g.settings.num_plot_contours = 2
        
g.settings.alpha_filled_add = 0.6  # Adjust transparency of filled contours
g.settings.title_limit_fontsize = 14  # Limit title fontsize to 14
g.settings.axes_labelsize = 12  # Adjust label fontsize
g.settings.legend_fontsize = 10  # Adjust legend fontsize
g.settings.colorbar_axes_fontsize = 10  # Adjust colorbar fontsize
g.triangle_plot(sample2, name, filled=True, legend_labels=[f'{label_fig}'],legend_loc='upper right', contour_colors=['orange'],title_limit=1)  

g.export(f'figure/{file_name}_check_5.pdf')


names = [r"$H_0$", r"$\Omega_m0$", r"$\alpha$", r"$\beta$", r"$\gamma$"]
fig, axes = plt.subplots(5, 1, figsize=(10, 8), sharex=True)

for i in range(5):

    ax = axes[i]

    for walker in range(chain.shape[1]):
        ax.plot(chain[:, walker, i],
                alpha=0.3,
                lw=0.5)

    ax.set_ylabel(names[i])

axes[-1].set_xlabel("Step number")

plt.tight_layout()
plt.show() 