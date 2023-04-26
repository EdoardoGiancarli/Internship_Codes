# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 11:57:32 2023

@author: Edoardo Giancarli
"""

import numpy as np
import matplotlib.pyplot as plt
import Driver_LongTransient_Python_release8 as ltp
# import pandas as pd

# quick analysis + curve fit (in Results_fit2.py)
#
# quicker version of Matched_Filtering2 (more complete)

##################################           matched filtering (quick version)         #############################################

path_bsd_gout = "D:/Home/Universita'/Universita'/Magistrale/AA 2022 - 2023/Stage/Code_python/band295_300/gout_58633_58638_295_300.mat"
path_bsd_softInj_powlaw = "D:/Home/Universita'/Universita'/Magistrale/AA 2022 - 2023/Stage/Code_python/band295_300/bsd_softInj_LL_longtransient_powlawC01_GATED_SUB60HZ_O3_295_300_.mat"
path_bsd_softInj_exp = "D:/Home/Universita'/Universita'/Magistrale/AA 2022 - 2023/Stage/Code_python/band295_300/bsd_softInj_LL_longtransient_expC01_GATED_SUB60HZ_O3_295_300_.mat"


m = 12
lfft = 2**m        # m = 10 --> lfft = 1024
y_edge = 2*lfft    # edge for the inj data into the LIGO-L data 
step = 20          # N of templates in each TFFT
ncand = 32         # N output candidates for chunk in the matched filtering process
thr = 0            # threshold for the Critical Ratio (for candidates selection)


# the function Matched_Filtering gives directly the complete dataframe with the analysis in each selected chunk (it's a shortcut)

candidates_powlaw = ltp.Matched_Filtering(path_bsd_gout, path_bsd_softInj_powlaw,
                                          lfft, y_edge, step, ncand, thr, key = 'gout_58633_58638_295_300',
                                          mat_v73 = True)

candidates_exp = ltp.Matched_Filtering(path_bsd_gout, path_bsd_softInj_exp,
                                       lfft, y_edge, step, ncand, thr, signal = 'exp',
                                       key = 'gout_58633_58638_295_300', mat_v73 = True)

# dataframe with the output candidates whose critical ratio is over the threshold

output_powlaw = ltp.get_candidates(candidates_powlaw[0], candidates_powlaw[2], lfft, choosetem = True)
  
output_exp = ltp.get_candidates(candidates_exp[0], candidates_exp[2], lfft, choosetem = True)


## powlaw
par_powlaw = ltp.display_candidates(output_powlaw, thr = None, bsd = candidates_powlaw[2],
                                    lfft = lfft, show_totfreq = False, save_plot = False, save_to_csv = True, freqband = [295, 300])

## exp
par_exp = ltp.display_candidates(output_exp, fig_num = 2, thr = None, signal = 'exp', bsd = candidates_exp[2],
                                 lfft = lfft, show_totfreq = False, save_plot = False, save_to_csv = True, freqband = [295, 300])

