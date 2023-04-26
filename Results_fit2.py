# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 01:57:21 2023

@author: Edoardo Giancarli
"""

import numpy as np
import pandas as pd
import Driver_LongTransient_Python_release8 as ltp

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=22, usetex=True)


#### path for sources and lfft
path_bsd_gout = "D:/Home/Universita'/Universita'/Magistrale/AA 2022 - 2023/Stage/Code_python/band295_300/gout_58633_58638_295_300.mat"
path_bsd_softInj_powlaw = "D:/Home/Universita'/Universita'/Magistrale/AA 2022 - 2023/Stage/Code_python/band295_300/bsd_softInj_LL_longtransient_powlawC01_GATED_SUB60HZ_O3_295_300_.mat"
path_bsd_softInj_exp = "D:/Home/Universita'/Universita'/Magistrale/AA 2022 - 2023/Stage/Code_python/band295_300/bsd_softInj_LL_longtransient_expC01_GATED_SUB60HZ_O3_295_300_.mat"


#### path for dataframe with candidates
amp = '10^-4'
lfft = 4096
y_edge = 2*lfft
path_dfpowlaw = "D:/Home/Universita'/Universita'/Magistrale/AA 2022 - 2023/Stage/Code_python/band295_300/295_300amp_" + amp + "/Output_candidates_powerlaw_database_295to300Hz_lfft" + str(lfft) + "_" + amp + ".csv"
path_dfexp = "D:/Home/Universita'/Universita'/Magistrale/AA 2022 - 2023/Stage/Code_python/band295_300/295_300amp_" + amp + "/Output_candidates_exp_database_295to300Hz_lfft" + str(lfft) + "_" + amp + ".csv"


#### get info from gout and injected signal
bsd_powerlaw = ltp.get_data(path_bsd_gout, path_bsd_softInj_powlaw, y_edge, key = 'gout_58633_58638_295_300', mat_v73 = True)
bsd_exp = ltp.get_data(path_bsd_gout, path_bsd_softInj_exp, y_edge, key = 'gout_58633_58638_295_300', mat_v73 = True)


#### load csv with output candidates and define the signal frequency
# powlaw
par_powlaw = ltp.display_candidates(path_dfpowlaw, thr = None, bsd = bsd_powerlaw, lfft = lfft, show_totfreq = False)


for x1, x2, y1, y2 in zip([-1.2], [0.01], [295], [300]):        # remove outliers
    
    try:
        df_powlaw = ltp.candidates_remover(df_powlaw, x_min = x1, x_max = x2, y_min = y1, y_max = y2)
    except:
        df_powlaw = ltp.candidates_remover(path_dfpowlaw, x_min = x1, x_max = x2, y_min = y1, y_max = y2)

par_powlaw = ltp.display_candidates(df_powlaw, thr = None, bsd = bsd_powerlaw, lfft = lfft,
                                    show_totfreq = False, save_plot = False, save_to_csv = False, freqband = [295, 300])

mu_powlaw = par_powlaw[0]                        
t_powlaw = par_powlaw[1]
CR_powlaw = par_powlaw[2]

# exp
par_exp = ltp.display_candidates(path_dfexp, fig_num = 2, thr = None, signal = 'exp', bsd = bsd_exp, lfft = lfft,
                                 show_totfreq = False)

for x1, x2, y1, y2 in zip([-1.2], [0.01], [295], [300]):       # remove outliers
    try:
        df_exp = ltp.candidates_remover(df_exp, x_min = x1, x_max = x2, y_min = y1, y_max = y2)
    except:
        df_exp = ltp.candidates_remover(path_dfexp, x_min = x1, x_max = x2, y_min = y1, y_max = y2)

par_exp = ltp.display_candidates(df_exp, fig_num = 2, thr = None, signal = 'exp', bsd = bsd_exp, lfft = lfft,
                                 show_totfreq = False, save_plot = False, save_to_csv = False, freqband = [295, 300])

mu_exp = par_exp[0]
t_exp = par_exp[1]
CR_exp = par_exp[2]



####  verify the matched filtering through scipy.optimize.curve_fit
from scipy.optimize import curve_fit

popt_pl, pcov_pl = curve_fit(ltp.freq_pow_law, t_powlaw, mu_powlaw)                 # curve fit for powlaw freq
f_pl_pred = ltp.freq_pow_law(t_powlaw, *popt_pl)                                    # predicted freq from fit 

popt_exp, pcov_exp = curve_fit(ltp.freq_exp, t_exp, mu_exp)                         # curve fit for exp freq
f_exp_pred = ltp.freq_exp(t_exp, *popt_exp)                                         # predicted freq from fit 

# -----> curve_fit is not able to reconstruct the signal from the output candidates


####  verify the matched filtering through linear regression (log scale)
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

## log values for fit
tau_days_pl = 1400000/86400      # tau for pow law freq in days (t_powlaw are in days)
ln_mu_pl = np.log(mu_powlaw)     # power law templates
ln_t_pl = np.log(1 + t_powlaw/tau_days_pl)

tau_days_exp = 5000000/86400     # tau for exp freq in days (t_exp are in days)
ln_mu_exp = np.log(mu_exp)       # exp templates
ln_t_exp = t_exp/tau_days_exp

## linear regression
regr = linear_model.LinearRegression()

regr.fit(ln_t_pl.reshape(len(ln_t_pl), 1), ln_mu_pl)                                    # power law freq fit
fit_templ_powlaw = regr.predict(ln_t_pl.reshape(len(ln_t_pl), 1))
a_fit, b_fit, c_fit = regr.coef_, regr.intercept_, r2_score(ln_mu_pl, fit_templ_powlaw) # regression coefficients
se_a, se_b = ltp.coeffs_errors(ln_t_pl, ln_mu_pl, fit_templ_powlaw)                     # standard errors of the regression coefficients
print('r2 score:', c_fit)
print('fit coeff:', a_fit[0], '+-', se_a, 'true coeff', -0.25)
print('fit intercept:', b_fit, '+-', se_b, 'true intercept', np.log(300.01))
# dictionary with info
pl_dict = {'r2 score' : c_fit,
           'fit coeff' : a_fit[0],
           'coeff error' : se_a,
           'fit intercept' : b_fit,
           'intercept error' : se_b}                                                              

regr.fit(ln_t_exp.reshape(len(ln_t_exp), 1), ln_mu_exp)                                 # exp freq fit
fit_templ_exp = regr.predict(ln_t_exp.reshape(len(ln_t_exp), 1))
d_fit, e_fit, f_fit = regr.coef_, regr.intercept_, r2_score(ln_mu_exp, fit_templ_exp)   # regression coefficients
se_d, se_e = ltp.coeffs_errors(ln_t_exp, ln_mu_exp, fit_templ_exp)                      # standard errors of the regression coefficients
print('r2 score:', f_fit)
print('fit coeff:', d_fit[0], '+-', se_d, 'true coeff', -1)
print('fit intercept:', e_fit, '+-', se_e, 'true intercept', np.log(300.01))
# dictionary with info
exp_dict = {'r2 score' : f_fit,
           'fit coeff' : d_fit[0],
           'coeff error' : se_d,
           'fit intercept' : e_fit,
           'intercept error' : se_e}                                                                            

####  residues

## power law residues
freq_pl_signal = ltp.freq_pow_law(300.01, tau_days_pl, 5, 58635, t_powlaw + 58635, log = True)    # freq pow law true signal 
res_powlaw1 = fit_templ_powlaw - ln_mu_pl                                                         # residues between fit and candidates
res_powlaw2 = fit_templ_powlaw - freq_pl_signal                                                   # residues between fit and true signal
res_powlaw3 = ln_mu_pl - freq_pl_signal                                                           # residues between candidates and true signal

# mean and std for residues
for res, j in zip([res_powlaw1, res_powlaw2, res_powlaw3], ['fit vs candidates', 'fit vs true signal', 'candidates vs true signal']):
    res_median, res_std = np.median(res), np.std(res)
    pl_dict[j + ' median'] = res_median
    pl_dict[j + ' std'] = res_std
    print(j + ' residues -', 'median:', res_median, 'std:', res_std)

# plot residues for power law frequency (fit vs candidates, fit vs true signal, candidates vs true signal)
ltp.residues_plot(t_powlaw, res_powlaw1, res_powlaw2, res_powlaw3)


## exp residue
freq_exp_signal = ltp.freq_exp(300.01, tau_days_exp, 58635, t_exp + 58635, log = True)       # freq pow law true signal 
res_exp1 = fit_templ_exp - ln_mu_exp                                                         # residues between fit and candidates
res_exp2 = fit_templ_exp - freq_exp_signal                                                   # residues between fit and true signal
res_exp3 = ln_mu_exp - freq_exp_signal                                                       # residues between candidates and true signal

# mean and std for residues
for res, j in zip([res_exp1, res_exp2, res_exp3], ['fit vs candidates', 'fit vs true signal', 'candidates vs true signal']):
    res_median, res_std = np.median(res), np.std(res)
    exp_dict[j + ' median'] = res_median
    exp_dict[j + ' std'] = res_std
    print(j + ' residues -', 'median:', res_median, 'std:', res_std)

# plot residues for power law frequency (fit vs candidates, fit vs true signal, candidates vs true signal)
ltp.residues_plot(t_exp, res_exp1, res_exp2, res_exp3, signal = 'exp')


#### dataframe with info about fit and residues
df_info = pd.concat([pd.DataFrame(pl_dict, index = [0]), pd.DataFrame(exp_dict, index = [0])])
df_info.index = ['power law', 'exp']
df_info.to_csv('295_300df_info_' + amp + '.csv')



#### plot

f3 = plt.figure(num=3, figsize=[16, 12], tight_layout = True)                    # powlaw freq plot
ax = f3.add_subplot(111)
ax.scatter(ln_t_pl, ln_mu_pl, c='m', alpha=0.5)                                  # plot candidates from dataframe         
ax.plot(ln_t_pl, a_fit[0]*ln_t_pl + b_fit, c='b', alpha=0.7)                     # plot t vs fit
ax.plot(ln_t_pl, (-0.25)*ln_t_pl + np.log(300.01), c='cyan', alpha=0.8)          # plot true signal
patch_data = mpatches.Patch(color='m', label='candidates')
patch_fit = mpatches.Patch(color='b', label='fit')
patch_true = mpatches.Patch(color='cyan', label='true signal')
plt.legend(handles=[patch_data, patch_fit, patch_true], loc='best')
ax.grid(True)
plt.title('Reconstruction for powlaw frequency')
plt.xlabel('log(1 + t/tau)')
plt.ylabel('log(freq)')
ax.label_outer()
ax.tick_params(which='both', direction='in',width=2)
ax.tick_params(which='major', direction='in',length=7)
ax.tick_params(which='minor', direction='in',length=4)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
plt.show()



f4 = plt.figure(num=4, figsize=[16, 12], tight_layout = True)                    # exp freq plot
ax = f4.add_subplot(111)
ax.scatter(ln_t_exp, ln_mu_exp, c='m', alpha=0.5)                                # plot candidates from dataframe         
ax.plot(ln_t_exp, d_fit[0]*ln_t_exp + e_fit, c='b', alpha=0.7)                   # plot t vs fit
ax.plot(ln_t_exp, (-1)*ln_t_exp + np.log(300.01), c='cyan', alpha=0.8)           # plot true signal
patch_data = mpatches.Patch(color='m', label='candidates')
patch_fit = mpatches.Patch(color='b', label='fit')
patch_true = mpatches.Patch(color='cyan', label='true signal')
plt.legend(handles=[patch_data, patch_fit, patch_true], loc='best')
ax.grid(True)
plt.title('Reconstruction for exp frequency')
plt.xlabel('t/tau')
plt.ylabel('log(freq)')
ax.label_outer()
ax.tick_params(which='both', direction='in',width=2)
ax.tick_params(which='major', direction='in',length=7)
ax.tick_params(which='minor', direction='in',length=4)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
plt.show()

