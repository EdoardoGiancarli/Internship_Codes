# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 01:51:58 2023


## modified on Thu Feb 28 20:17 2023 --- added residues_plot
                          22:10 2023 --- added candidates_remover (then modified)
## modified on Wed Mar 22 17:58 2023 --- added coeffs_errors


@author: Edoardo Giancarli
"""

###############      useful libraries      ###################


import numpy as np                 # operations
import pandas as pd                # dataframe  

from scipy.io import loadmat       # matlab datafile
import hdf5storage as hd           # matlab datafile -v7.3  
# import matlab.engine

import matplotlib.pyplot as plt                   # plotting
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=12, usetex=True)


######### Driver with function for Matched Filtering ##########
#
# mat_to_dict: converts matlab data file into dict (simpler to handle)
#
# freq_powlaw, freq_exp: frequency signals
#
# signal_freq: define the long transient frequency
#
# TFunc_gauss_DB: it builds the Gaussian templates database
#
# get_data: gets the data from gout, gsinj and source (e.g. for filter_data_chunk)
#
# CR: computes the Critical Ratio (for sel_candidates)
#
# sel_candidates: gives the candidates for each data chunk
#
# filter_data_chunk: performs the Matched Filtering using the database and the data from gout, gsinj and source
#
# get_candidates: It selects the output candidates from the total candidates list based on the chosen threshold.
#
# Matched_Filtering: quick method to obtain candidates (for now)
#
# display_templates: displays the templates of the selected candidates
#
# display_candidates: displays the selected candidates
#
# residue_plot: displays the selected residues
#
# coeffs_errors: computes the standard errors of the fit regression coefficients
#
# candidates_remover: it removes certain candidates from the input dataframe in a rectangular area [x1, x2] x [y1, y2]

    


###########################################               Functions                 ####################################




def mat_to_dict(path, key = 'goutL', mat_v73 = False):  # convert the data from MATLAB: it seems that is better to leave goutL as a dict
    
    """
    Conversion from MATLAB data file to dict.
    Parameters:
    -----------------------------------------------------------------------------------------------------------------
    path : (str) path of your MATLAB data file ---
    key : keyword with info from L, H or V interferometer (insert gout + interf. or gsinj + interf., default = goutL) ---
    mat_v73 : (bool) if the matlab datafile version is -v7.3 insert the 'True' value (default = 'False') ---
    
    -----------------------------------------------------------------------------------------------------------------
    return:
        
    data_dict: (dict) dict from MATLAB data file ---
    perczero: (float) percentage of total zero data in y (data from the Obs run)
    -----------------------------------------------------------------------------------------------------------------     
    """
    
    if mat_v73 == True:       
        
        mat = hd.loadmat(path)                                        # load mat-file -v7.3 

        mdata = mat[key]                                              # variable in mat file
        mdtype = mdata.dtype                                          # dtypes of structures are "unsized objects"
        data_dict = {n: mdata[n][0] for n in mdtype.names}            # express mdata as a dict

        y = data_dict['y']
        y = y.reshape(len(y))
        data_dict['y'] = y
        y_zero = np.where(y == 0)[0]
        perczero = len(y_zero)/len(y)                                 # perc of total zero data in y (data from the Obs run)

        cont = data_dict['cont']
        cont_dtype = cont.dtype
        cont_dict = {u: cont[str(u)] for u in cont_dtype.names}       # cont in data_dict is a structured array, I converted it in a dict
        data_dict['cont'] = cont_dict
        
        return data_dict, perczero
    
    else:
        
        mat = loadmat(path)                                               # load mat-file
        
        if key == 'sour':
    
            # SciPy reads in structures as structured NumPy arrays of dtype object
            # The size of the array is the size of the structure array, not the number-
            #  -elements in any particular field. The shape defaults to 2-dimensional.
            # For convenience make a dictionary of the data using the names from dtypes
            # Since the structure has only one element, but is 2-D, index it at [0, 0]                               
            
            mdata = mat['sour']                                           # variable in mat file
            mdtype = mdata.dtype                                          # dtypes of structures are "unsized objects"
            data_dict = {n: mdata[n][0, 0] for n in mdtype.names}         # express mdata as a dict
            
            return data_dict
        
        else:
            
            mdata = mat[key]                                              # variable in mat file
            mdtype = mdata.dtype                                          # dtypes of structures are "unsized objects"
            data_dict = {n: mdata[n][0, 0] for n in mdtype.names}         # express mdata as a dict
            
            y = data_dict['y']
            y = y.reshape(len(y))
            data_dict['y'] = y
            y_zero = np.where(y == 0)[0]
            perczero = len(y_zero)/len(y)                                 # perc of total zero data in y (data from the Obs run)
            
            cont = data_dict['cont']
            cont_dtype = cont.dtype
            cont_dict = {u: cont[str(u)] for u in cont_dtype.names}       # cont in data_dict is a structured array, I converted it in a dict
            data_dict['cont'] = cont_dict                                 # now we have a fully accessible dict
        
            return data_dict, perczero




########################################################################################################################



def freq_pow_law(fgw0, tau, nbreak, tcoes, t, log = False):
    
    """
    Power law frequency for the long transient signal.
    Parameters:
    -----------------------------------------------------------------------------------------------
    fgw0 : (float) initial frequency [Hz] ---
    tau : (int or float) characteristic time [s] (or same S.I. units as t) ---
    nbreak : (int or float) breaking index (= 5 from NS studies) ---
    tcoes : (int) coalescing time [s] (or same S.I. units as t)  ---
    t : (arrray) time vector [s] ---
    log : (bool) if you want the log of the frequency (default = False) ---
    
    -----------------------------------------------------------------------------------------------
    return:
    
    freq: (array) freq signal [Hz] ---
    -----------------------------------------------------------------------------------------------      
    """
    
    if log == True:
        return np.log(fgw0) + (1./(1 - nbreak))*np.log(1. + (t - tcoes)/tau)
    else:
        return fgw0*(1. + (t - tcoes)/tau)**(1./(1 - nbreak))




def freq_exp(fgw0, tau, tcoes, t, log = False):
    
    """
    Exponential frequency for the long transient signal.
    Parameters:
    -----------------------------------------------------------------------------------------------
    fgw0 : (float) initial frequency [Hz] ---
    tau : (int or float) characteristic time [s] (or same S.I. units as t) ---
    tcoes : (int) coalescing time [s] (or same S.I. units as t) ---
    t : (array) time vector [s] ---
    log : (bool) if you want the log of the frequency (default = False) ---
    
    -----------------------------------------------------------------------------------------------
    return:
    
    freq: (array) freq signal [Hz] ---
    -----------------------------------------------------------------------------------------------      
    """
    
    if log == True:
        return np.log(fgw0) - (t - tcoes)/tau
    else:
        return fgw0*np.exp(-(t - tcoes)/tau)





def signal_freq(bsd, lfft, signal = 'power law', show_freq = False):
    
    """
    Frequency of the long transient signal.
    Parameters:
    -----------------------------------------------------------------------------------------------
    bsd : (dict) bsd from get_data ---
    lfft : (int) fft lenght ---
    signal : (string) define the frequency signal ('power law' or 'exp', default = power law) ---
    show_freq : (bool) if show_freq is True, signal_freq return the complete frequency and time array (default = False) ---
    
    -----------------------------------------------------------------------------------------------
    return:
    
    freq: (array) freq signal in the chosen frequency interval ([Hz]) ---
    time: (array) time values of freq signal ([s]) 
    -----------------------------------------------------------------------------------------------      
    """
    
    dx = bsd['dx']                                          # sampling time [s]  
    source = bsd['source']                                  # source parameters
    TFFT = lfft*dx                                          # FFT time duration
    inifr = bsd['inifr']                                    # initial bin freq of data
    bandw = bsd['bandw']                                    # bandwidth  of the bin
    finfr = inifr + bandw                                   # final freq bin of data
    
    ## signal parameters
    
    tcoe = source['tcoe'][0, 0]                             # coalescing time [days]
    tcoe = np.int64(tcoe)
    tcoes = tcoe*86400                                      # coalescing time [s] (86400 # s in one day)
    fgw0 = source['f0'][0, 0]                               # initial frequency [Hz]
    
    ##### define frequency
    
    if signal == 'power law':                               # power law frequency
    
        tau = source['tau'][0, 0]                           # characteristic time [s]
        nbreak = source['n'][0, 0]                          # breaking index (= 5 from NS studies)
        days = source['days'][0, 0]                         # N of days after tcoes 
        days = np.int64(days)
        time = np.arange(tcoes, tcoes + days*86400)         # time vector [s]
        
        freq = freq_pow_law(fgw0, tau, nbreak, tcoes, time)
        
        f_ini = np.where(freq < finfr)[0][0]                           # first element of freq < finfr [Hz]
        f_fin = np.where(freq > inifr)[0][-1]                          # last element of freq > inifr [Hz]
        p, q = int((f_ini//TFFT + 1)*TFFT), int((f_fin//TFFT)*TFFT)    # p, q | len(freq) = n*TFFT with n natural number
        
        freq_resh = freq[p:q]                                          # freq_resh contains inifr < freq < finfr [Hz]
        time_resh = time[p:q]                                          # time_resh array [s] with respect to freq in (inifr, finfr) 
        
    elif signal == 'exp':                                   # exp frequency
    
        tau = source['tau'][0, 0]                           # characteristic time [s]
        days = source['days'][0, 0]                         # N of days after tcoes
        days = np.int64(days)
        time = np.arange(tcoes, tcoes + days*86400)         # time vector [s]
        
        freq = freq_exp(fgw0, tau, tcoes, time)
        
        f_ini = np.where(freq < finfr)[0][0]                           # first element of freq < finfr [Hz]
        f_fin = np.where(freq > inifr)[0][-1]                          # last element of freq > inifr [Hz]
        p, q = int((f_ini//TFFT + 1)*TFFT), int((f_fin//TFFT)*TFFT)    # p, q | len(freq) = n*TFFT with n natural number
        
        freq_resh = freq[p:q]                                          # freq_resh contains inifr < freq < finfr [Hz]
        time_resh = time[p:q]                                          # time_resh array [s] with respect to freq in (inifr, finfr) 
        
    # if not len(freq) % TFFT == 0:                              # control warning
    #     print('Freq is not in TFFT units: lenght of freq = {x}'.format(x = len(freq)/TFFT))
    
    if show_freq == True:
        return freq, time
    else:
        return freq_resh, time_resh
    




########################################################################################################################




def TFunc_gauss_DB(lfft, bsd, step = 1, signal = 'power law', save_to_csv = False):     # build the gaussian DataBase
    
    """
    Filter Database (Gaussian Template).
    Parameters:
    -----------------------------------------------------------------------------------------------
    lfft : (int) fft lenght
    bsd : (dict) dict from get_data with info ---
    step : (int) number of Gaussian template for bins of lenght TFFT = lfft*dx (default = 1) ---
    signal : (string) define the frequency signal ('power law' or 'exp', default = power law) ---
    save_to_csv : (bool) if you want to save the dataframe insert the 'True' value (default = 'False') ---
    
    -----------------------------------------------------------------------------------------------
    return:
    
    database: (pandas.core.frame.DataFrame) pandas dataframe with the Gaussian templates, Gaussian mean values and std values
    -----------------------------------------------------------------------------------------------      
    """
    
    dx = bsd['dx']                                       # sampling time [s]
    bandw = bsd['bandw']                                 # bandwidth  of the bin      
    dfr = 1./lfft*dx                                     # freq resolution
    freq = np.linspace(0, bandw, lfft)                   # freq interval [0, bandw)  
    TFFT = lfft*dx                                       # FFT time duration
    div = lfft//128                                      # sigma dividend
    
    # if dx >= 1:
    #     div = 2**(np.log2(lfft) - 7)                     # sigma dividend
    # else:
    #     div = 2**(np.log2(lfft) - 5)
    
    def ES_filter(f, Mu, Sigma):                         # define the transfer function (Gaussian template)
        return np.exp(-(f - Mu)**2/(2.0*Sigma**2))
    
    
    ##### define std for templates
    
    if signal == 'power law':                                                   # power law frequency
        
        f_pow = signal_freq(bsd, lfft)[0]
    
        sigma = np.array([np.abs(f_pow[int(i*TFFT)] - f_pow[int((i - 1)*TFFT)])/div 
                          for i in range(1, int(len(f_pow)/TFFT))])             # std for Gaussian templates
        
    elif signal == 'exp':                                                       # exp frequency
    
        f_exp = signal_freq(bsd, lfft, signal = signal)[0]
    
        sigma = np.array([np.abs(f_exp[int(i*TFFT)] - f_exp[int((i - 1)*TFFT)])/div 
                          for i in range(1, int(len(f_exp)/TFFT))])             # std for Gaussian templates

    ##### generate database
    
    mu = np.linspace(dfr, bandw - dfr, len(sigma)*step)                                              # mean values array
    
    gauss_db = list(ES_filter(freq, mu[i], sigma[i//step]) for i in range(0, len(mu)))[step:-step]   # list with the templates                             

    database = pd.DataFrame(index = range(len(freq)), columns = range(len(gauss_db)))                # creating empty dataframe with chosen dim
                                             
    for i in range(len(gauss_db)):                                                                   # inserting template in df columns
        database[i] = gauss_db[i]

    database.columns = list("mu = {value1}".format(value1 = v) for v in mu[step:-step])              # rename columns
    
    std = []
    for s in sigma[1:-1]:                            # modelling the sigma array (if step != 1 more mu have the same sigma)
        std += [s]*step
        
    par = pd.DataFrame([mu[step:-step], std], index = ['mu', 'std'], columns = database.columns)    
    database = pd.concat([database, par])                                                            # add mu and sigma to dataframe
    
    
    if save_to_csv == True:                          # save the dataframe as csv "Matched Filtering Gaussian database"
        name = 'MFGauss'
        
        if signal == 'power law':
            p_law = name + '_power_law_database.csv'
            database.to_csv(p_law)
            
        elif signal == 'exp':
            exp = name + '_exp_database.csv'
            database.to_csv(exp)
        
    return database




########################################################################################################################




def get_data(path_bsd_gout, path_bsd_gsinj, y_edge, key = 'goutL', mat_v73 = False):     # take noise + signal
    
    """
    It takes some info from bsd_gout (dx, n, y, t0, inifr, bandw) and from bsd_gsinj (y) to obtain noise + signal for the filtering.
    Parameters:
    -----------------------------------------------------------------------------------------------------------------
    path_bsd_gout : (str) bsd_gout containing the interferometer's noise ---
    path_bsd_gsinj : (str) _gsinj containing the injected signal ---
    y_edge : (int) number of y_gout elements which surround y_gsinj (recommended at least 1*lfft) ---
    key : (str) keyword with info from L, H or V interferometer (insert gout + interf. or gsinj + interf., default = goutL) ---
    mat_v73 : (bool) if the matlab datafile version is -v7.3 insert the 'True' value (default = 'False') ---
    
    -----------------------------------------------------------------------------------------------------------------
    return:
        
    bsd_out: (dict) bsd with the info to use for filter data chunk and for the database
    -----------------------------------------------------------------------------------------------------------------     
    """    
    
    bsd_gout, perczero_gout = mat_to_dict(path_bsd_gout, key = key, mat_v73 = mat_v73)     # gout and perczero of y_gout 
    bsd_gsinj, perczero_gsinj = mat_to_dict(path_bsd_gsinj, key = 'gsinjL')                # gsinj and perczero of y_gsinj
    source = mat_to_dict(path_bsd_gsinj, key = 'sour')                                     # source
    
    
    dx = bsd_gout['dx'][0, 0]                  # sampling time of the input data
    y_gout = bsd_gout['y']                     # data from the gout bsd
    
    y_gsinj = bsd_gsinj['y']                   # data from the gsinj bsd
    
    try:
        t0_gout = bsd_gout['cont']['t0'][0, 0]      # starting time of the gout signal [days]
    except:
        t0_gout = bsd_gout['cont']['t0'][0, 0, 0]
    
    tcoe = source['tcoe'][0, 0]                     # starting time of the signal [days]
    t0_gout = np.int64(t0_gout)
    tcoe = np.int64(tcoe)
    
    t_ind = int(((tcoe - t0_gout)*86400)/dx)       # index of gout data with respect to injected signal
    
    y_gout = y_gout[t_ind - y_edge:t_ind + len(y_gsinj) + y_edge]        # we take a chunk of y_gout that contain y_gsinj
    n_new = len(y_gout)                                                  # number of samples to consider for filter data chunk
    
    y_gsinj_resh = np.hstack([np.zeros(y_edge, dtype = complex), 
                              y_gsinj,
                              np.zeros(n_new - len(y_gsinj) - y_edge, dtype = complex)])     # reshaping y_sinj to sum with y_gout
    
                            
    amp = 1e-4                                 # amplitude factor to inject a signal of amplitude AMP
    y_tot = amp*y_gsinj_resh + y_gout          # signal + noise
    
    y_zero = np.where(y_tot == 0)[0]
    perczero = len(y_zero)/len(y_tot)          # perc of total zero data in y (data from the Obs run)
    
    try:
        inifr = bsd_gout['cont']['inifr'][0, 0][0, 0]       # initial bin freq
        bandw = bsd_gout['cont']['bandw'][0, 0][0, 0]       # bandwidth of the bin
    except:
        inifr = bsd_gout['cont']['inifr'][0, 0, 0]          # initial bin freq
        bandw = bsd_gout['cont']['bandw'][0, 0, 0]          # bandwidth of the bin
        
    
    bsd_out = {'dx': dx,
               'n': n_new,
               'y': y_tot,
               'perczero': perczero,
               'inifr': inifr,
               'bandw': bandw,
               'y_edge' : y_edge,
               'source': source}
    
    return bsd_out




########################################################################################################################




def CR(x):
    
    """
    It performs the Critical Ratio of the candidates using the median of input data.
    Parameters:
    ----------------------------------------------------------------------------------------------------------------
    x : (array) matched filtered data
    
    ----------------------------------------------------------------------------------------------------------------
    return:
        
    Crit Ratio : (array)
    ----------------------------------------------------------------------------------------------------------------   
    """
    
    m_1 = np.median(x)
    m_2 = np.median(np.abs(x - m_1))/0.6745
    
    try:
        np.seterr(invalid = 'ignore')          # it ignores the indefinites values which can occur in y (i.e. 0/0 or inf)
        y = np.abs(x - m_1)/m_2
    except:
        # print("x - m_1 = {v1}, m_2 = {v2}".format(v1 = x - m_1, v2 = m_2))
        y = -0.1
    
    return y





def sel_candidates(data, ncand, threshold, dx, t0, t_ini):
    
    """
    It lists the output candidates from matched filtered data.
    Parameters:
    ----------------------------------------------------------------------------------------------------------------
    data : (array) matched filtered data ---
    ncand : (int) number of searched candidates ---
    threshold : (float) threshold for the Critical Ratio ---
    dx : (float) sampling time ---
    t0 : (int) signal starting time ---
    t_ini : (float) starting time of data ---
    
    ----------------------------------------------------------------------------------------------------------------
    return:
        
    out_candidates : (list) output candidates
    ----------------------------------------------------------------------------------------------------------------   
    """
    
    dfstep = len(data)//ncand                                 # divide data in ncand sub-chunk
    Crit_R = CR(data)                                         # CR of data 
    out_candidates = []
    
    for i in range(0, len(data), dfstep):
        cr_max = np.max(Crit_R[i:i + dfstep])                 # max of CR in the sub-chunk
        ind = np.argmax(Crit_R[i:i + dfstep])                 # index of cr_max
        
        t_cand = t_ini + (i + ind)*dx/86400                   # time of the candidates (from t = 0, len(data) = lfft)
        
        if cr_max >= threshold:                                                  # candidates selection   
            a = [t_cand, cr_max, "sub_chunk = {k}".format(k = i)]
        else:
            a = "CR below threshold"
        
        out_candidates.append(a)
    
    return out_candidates

 


########################################################################################################################




def filter_data_chunk(bsd, lfft, ncand, threshold, BPFilter):
    
    """
    It performs the Matched Filtering throught the input gout and returns the output candidates.
    Parameters:
    ----------------------------------------------------------------------------------------------------------------
    bsd : (dict) dict from MATLAB data file ---
    lfft : (int) fft lenght ---
    ncand : (int) number of searched candidates ---
    threshold : (float) threshold for the Critical Ratio ---
    BPFilter : (pandas.core.frame.DataFrame) dataframe with filter templates ---
    
    ----------------------------------------------------------------------------------------------------------------
    return:
        
    database : (pandas.core.frame.DataFrame) output candidates with relative info: 
                time, critical ratio, mean values and std of the Gauss templates
    ----------------------------------------------------------------------------------------------------------------   
    """
    
    dx = bsd['dx']                             # sampling time of the input data
    TFFT = lfft*dx                             # FFT time duration
    n = bsd['n']                               # total number of samples in the input BSD
    perczero = bsd['perczero']                 # percentage of zeros above which a chunk is not analyzed
    
    ##### init:
    
    C_out = []                                 # list with matched filtered candidates
    y = bsd['y']                               # data from the bsd
    t0 = bsd['source']['tcoe'][0, 0]           # signal starting time
    n_lfft = (n//lfft)*lfft                    # number of samples in lfft unit
    
    ##### loop for chunks
    
    for j in range(0, n_lfft - lfft//2, lfft//2):
        
        y_chunk = y[j:j + lfft]                                # select a chunck between [j, j + lfft] in y_gout
        jzero = len(np.where(y_chunk == 0)[0])                 # find the N of zeros in the chunk
        
        # if not jzero <= perczero*lfft:                                     # to see if there are problem in y_chunk
        #     print("N zeros in y_chunk = {zero}".format(zero = jzero))
        
        
        if jzero <= perczero*lfft:                                  # FFTs removed if percentage of 0 > perczero   
            fft_y = np.fft.fft(y_chunk, n = lfft, norm = "ortho")*dx  # normalised FFT
            t_ini = (j//lfft)*(TFFT + dx)/86400                     # starting time of each data chunk (initial time = 0)
        
            for column in BPFilter.columns:
                T_Func = np.array(BPFilter[column][:-2], dtype = complex)
                DataMatched_Freq = np.conjugate(T_Func)*fft_y
                DataMatched_bound = np.real(np.fft.ifft(DataMatched_Freq, n = lfft, norm = "ortho"))
                DataMatched = DataMatched_bound[lfft//16: -lfft//16]                                  # remove boundary
                C_out.append(sel_candidates(DataMatched, ncand, threshold, dx, t0, t_ini))
            
            
        # if j % 50*lfft == 0:                                        # loop control
        #     print("Info: loop = {index}".format(index = j))
    
    
    col = [column for column in BPFilter.columns]                     # columns of BPFilter
    ncol = BPFilter.shape[1]                                          # N columns of BPFilter
    
    cand_out_tot = [C_out[i:i + ncol] 
                    for i in range(0, len(C_out), ncol)]              # join filter for the same chunk
    
    
    C_database = pd.DataFrame(cand_out_tot,                                                               # out list to database
                              index = ["chunk = {u}".format(u = i) for i in range(len(cand_out_tot))], 
                              columns = col)
    
    
    mu_db, std_db = np.array(BPFilter.iloc[-2]), np.array(BPFilter.iloc[-1])     # mu and std from BPFilter
    
    par = pd.DataFrame([mu_db, std_db], index = ['mu', 'std'], columns = col)    
    database = pd.concat([C_database, par])                                      # add mu and sigma to candidates dataframe
    
    
    return database






########################################################################################################################





def get_candidates(candidates, bsd, lfft, choosetem = True):
    
    """
    It selects the output candidates from the total candidates list based on the chosen threshold.
    Parameters:
    ----------------------------------------------------------------------------------------------------------------
    candidates : (pandas.core.frame.DataFrame) dataframe with tot candidates templates (also those with CR < threshold) ---
    bsd : (dict) dict from get_data with info ---
    lfft : (int) fft lenght ---
    choosetem : (bool) choose the template with higher CR at fixed frequency and time (default = False) ---
    
    ----------------------------------------------------------------------------------------------------------------
    return:
        
    output candidates : (pandas.core.frame.DataFrame) output candidates with relative info: time (from t = 0),
                        time (in days from start), critical ratio, mean value, std (of the Gauss template)
    ----------------------------------------------------------------------------------------------------------------   
    """
    
    output = []
    out_list = []
    out_list_2 = []

    for ind in candidates.index[:-2]:                 # extract candidates with CR >= threshold
        for column in candidates.columns:
            
            a = candidates[column][ind]
            
            for element in a:
                if not element == "CR below threshold":
                    output.append(element + [ind, candidates[column]['mu'], candidates[column]['std']])
    
    
    
    if choosetem == True:
        
        df = pd.DataFrame(output, 
                          columns = ['t_ini', 'CR', 'sub_chunk', 'chunk', 'mu', 'std'])    # candidates dataframe
    
        mu_db = np.array(candidates.iloc[-2], dtype = float)     # template mean values
        
        for h in mu_db:                                          # choose the template with higher CR (at fixed frequency)
            try:
                t = df.loc[df['mu'] == h]
                r = t.loc[t['CR'] == t['CR'].max()]
                out_list.append(r)
            except:
                pass
    
        df_out = pd.concat([u for u in out_list])
        
        for l in df_out['t_ini']:                                # choose the template with higher CR (at fixed time)
            try:
                z = df_out.loc[df_out['t_ini'] == l]
                x = z.loc[z['CR'] == z['CR'].max()]
                out_list_2.append(x)
            except:
                pass
        
        df_out_2 = pd.concat([u for u in out_list_2])
        
    else:
        df_out_2 = pd.DataFrame(output, 
                                columns = ['t_ini', 'CR', 'sub_chunk', 'chunk', 'mu', 'std'])    # candidates dataframe
    
    
    dx = bsd['dx']                                             # sampling time of the input data
    TFFT = lfft*dx                                             # FFT time duration
    y_edge = bsd['y_edge']                                     # number of y_gout elements which surround y_gsinj
    TFFT = lfft*dx
    time_correction = 1 + (y_edge//lfft)*(TFFT + dx)/86400     # time correction for y_edge and matlab to python-
                                                               # -conversion (which add a full day)
    
    tcoe = bsd['source']['tcoe'][0, 0]                         # signal coalescing time [days]
    df_out_2['t_ini'] = df_out_2['t_ini'] - time_correction
    df_out_2['mu'] = df_out_2['mu'] + bsd['inifr']
    df_out_2.insert(1, 't_days', df_out_2['t_ini'] + tcoe)
    
    return df_out_2




########################################################################################################################




def Matched_Filtering(path_gout, path_gsinj, lfft, y_edge, step, ncand, threshold,
                      signal = 'power law', key = 'goutL', mat_v73 = False):
    
    """
    It performs the Matched Filtering process. Inside this function get_data, TFunc_gauss_DB and filter_data_chunk
    are performed.
    Parameters:
    ----------------------------------------------------------------------------------------------------------------
    path_gout : (str) bsd_gout containing the interferometer's noise ---
    path_gsinj : (str) _gsinj containing the injected signal ---
    lfft : (int) fft lenght ---    
    y_edge : (int) number of y_gout elements which surround y_gsinj (recommended at least 1*lfft) ---
    step : (int) number of Gaussian template for bins of lenght TFFT = lfft*dx (default = 1) ---
    ncand : (int) number of searched candidates ---
    threshold : (float) threshold for the Critical Ratio ---
    signal : (string) frequency signal ('power law' or 'exp', default = power law) ---
    key : keyword with info from L, H or V interferometer (insert gout + interf. or gsinj + interf., default = goutL) ---
    mat_v73 : (bool) if the matlab datafile version is -v7.3 insert the 'True' value (default = 'False') ---
    
    ----------------------------------------------------------------------------------------------------------------
    return:
        
    candidates : (pandas.core.frame.DataFrame) output candidates with relative info: 
                time, critical ratio, mean values and std of the Gauss templates ---
    database: (pandas.core.frame.DataFrame) pandas dataframe with the Gaussian templates,
               Gaussian mean values and std values ---
    bsd_out: (dict) bsd with the info to use for filter data chunk and for the database
    ----------------------------------------------------------------------------------------------------------------   
    """
    
    bsd_data = get_data(path_gout, path_gsinj, y_edge, key = key, mat_v73 = mat_v73)    # get data for signal freq
    
    database = TFunc_gauss_DB(lfft, bsd_data, step, signal = signal)                    # TFunc_gauss_DB output
    
    candidates = filter_data_chunk(bsd_data, lfft, ncand, threshold,                    # filter data chunk
                                     database)
    
    return candidates, database, bsd_data
    
    
    

########################################################################################################################

    
    
    
def display_templates(mu, std, bandw, lfft):             # display the templates from the output candidates
    
    """
    It displays the Gaussian templates from the arrays mu (mean values) and sigma (std values).
    Parameters:
    ----------------------------------------------------------------------------------------------------------------
    mu : (array) dataframe with candidates (can be also the DF path) ---
    sigma : (array) initial bin freq of data ---
    bandw : (int) bandwidth of the considered frequency bin ---
    lfft : (int) fft lenght ---
    
    ----------------------------------------------------------------------------------------------------------------
    return:
        
    Plot with the templates
    ----------------------------------------------------------------------------------------------------------------   
    """

    def ES_filter(f, Mu, Sigma):                         # define the transfer function (Gaussian template)
        return np.exp(-(f - Mu)**2/(2.0*Sigma**2))

    f = np.linspace(0, bandw, lfft)
    
    fig = plt.figure(num = 1, figsize=(16, 12), tight_layout = True)    # templates plot
    ax = fig.add_subplot(111)                                         
    for i, j in zip(mu, std):
        template = ES_filter(f, i, j)
        ax.plot(f, template)
    plt.xlabel('freq [Hz]')
    plt.ylabel('template')
    plt.title('Gaussian templates')
    ax.grid(True)
    ax.label_outer()                   
    ax.tick_params(which='both', direction='in',width=2)
    ax.tick_params(which='major', direction='in',length=7)
    ax.tick_params(which='minor', direction='in',length=4)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    
    plt.show()
    
    
    

########################################################################################################################

    
    

def display_candidates(df_input, fig_num = 1, thr = None,
                       signal = 'power law', bsd = None, lfft= None, show_totfreq = False,
                       save_plot = False, save_to_csv = False, freqband = False):     # display the output candidates

    """
    It displays the output candidates and gives their info.
    Parameters:
    ----------------------------------------------------------------------------------------------------------------
    df_input : (pandas.core.frame.DataFrame or str) dataframe with candidates (can be also the DF path) ---
    fig_num : (int) number of the figure (default = 1) ---
    thr : (float) threshold for the Critical Ratio ---
    signal : (string) frequency signal ('power law' or 'exp', default = power law) ---
    bsd : (dict) dict from get_data with info (default = None) ---
    lfft : (int) fft lenght (default = None) ---
    show_totfreq : (bool) if True, display_candidates shows the complete frequency signal (default = False) ---
    save_plot : (bool) if you want to save the plot as PNG image ---
    save_to_csv : (bool) if you want to save the dataframe insert the 'True' value (default = 'False') ---
    freqband : (list) frequency bandwidth (info for the file name) ---
    
    ----------------------------------------------------------------------------------------------------------------
    return:
        
    mu, t, CR, std : (array) output info: templates mean values, time (in days from start),
                                           critical ratio, templates std values
    ----------------------------------------------------------------------------------------------------------------   
    """
    
    if isinstance(df_input, str):           # if df_input is the df path, this load the dataframe
        df_input = pd.read_csv(df_input)
    else:
        pass
    
    
    if not thr == None:
        
        df_thr = df_input.loc[df_input['CR'] >= thr]      # get candidates with CR over threshold 
        
        mu = np.array(df_thr['mu'])                       # info in the dataframe with CR over thr                
        t = np.array(df_thr['t_ini'])
        CR = np.array(df_thr['CR'])
        std = np.array(df_thr['std'])
    
    else:
        mu = np.array(df_input['mu'])                     # info in the dataframe                 
        t = np.array(df_input['t_ini'])
        CR = np.array(df_input['CR'])
        std = np.array(df_input['std'])
    
    
    if (bsd != None and lfft != None):                         # to display all the true freq signal
        
        if show_totfreq == True:
            freq, time = signal_freq(bsd, lfft, signal = signal, show_freq = True)
        else:
            freq, time = signal_freq(bsd, lfft, signal = signal)
    else:
        pass
            
        
    fig = plt.figure(num = fig_num, figsize=(16, 12), tight_layout = True)   ### plot
    ax = fig.add_subplot(111)
    f1 = ax.scatter(t, mu, c = CR, cmap = 'jet', alpha=0.8)                  # scatter plot of candidates
    
    try:                                                                     # plot true frequency signal
        tcoe = bsd['source']['tcoe'][0, 0]                                   # coalescing time [days]
        time_sc = time/86400 - tcoe
        ax.plot(time_sc, freq, color='OrangeRed', label='True frequency signal')
    except:
        pass
    
    plt.colorbar(f1, label = 'CR values')                          # colorbar for CR values  
    plt.legend(loc = 'best')                                       # legend
    
    # patch_freq = mpatches.Patch(color='OrangeRed', label='True frequency signal')
    # plt.legend(handles=[patch_data, patch_fit], loc='best')
    
    if signal == 'power law':                                      # title and labels
        plt.title('Power law frequency long transient')
    elif signal == 'exp':
        plt.title('Exp frequency long transient')
    
    plt.xlabel('t [days]')
    plt.ylabel('freq [Hz]')                                
    ax.grid(True)
    ax.label_outer()                                              # ticks options                     
    ax.tick_params(which='both', direction='in',width=2)
    ax.tick_params(which='major', direction='in',length=7)
    ax.tick_params(which='minor', direction='in',length=4)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    
    if save_plot == True:                                         # save plot as PNG image 
        plt.savefig("plot.png", format = 'png')
    
    plt.show()                                                    ### end plot
    
    
    if save_to_csv == True:                                       # save the dataframe as csv "output_candidates database"
        name = 'Output_candidates'
        name_lfft = 'lfft' + str(lfft)
        band = '_' + str(freqband[0]) + 'to' + str(freqband[1]) + 'Hz_'
        
        if signal == 'power law':
            db_powlaw = name + '_powerlaw_database' + band + name_lfft + '.csv'
            if not thr == None:
                df_thr.to_csv(db_powlaw)
            else:
                df_input.to_csv(db_powlaw)
            
        elif signal == 'exp':
            db_exp = name + '_exp_database' + band + name_lfft + '.csv'
            if not thr == None:
                df_thr.to_csv(db_exp)
            else:
                df_input.to_csv(db_exp)
    
    return mu, t, CR, std




########################################################################################################################




def residues_plot(x, y1, y2, y3, signal = 'power law'):
    
    """
    It displays the residues in y1, y2 and y3.
    Parameters:
    ----------------------------------------------------------------------------------------------------------------
    x : (array) x-axis data ---
    y1 : (array) y-axis data for 1st subplot ---
    y2 : (array) y-axis data for 2nd subplot ---
    y3 : (array) y-axis data for 3rd subplot ---
    signal : (string) frequency signal ('power law' or 'exp', default = power law) ---
    
    ----------------------------------------------------------------------------------------------------------------
    return:
        
    plot from matplotlib
    ----------------------------------------------------------------------------------------------------------------   
    """
    
    # create the figure and three subplots with shared x-axis
    fig, axs = plt.subplots(3, 1, figsize=(16, 12), sharex=True, tight_layout = True)
    
    # plot the data on each subplot and save the lines for the legend + grid + ticks options
    axs[0].scatter(x, y1, c = 'cyan')             # residues between fit and data
    axs[1].scatter(x, y2, c = 'b')                # residues between fit and true signal
    axs[2].scatter(x, y3, c = 'm')                # residues between data and true signal
    
    for ind in range(3):
        axs[ind].plot([-0.2, 1], [0, 0], c = 'r')
        axs[ind].grid(True)
        axs[ind].label_outer()                                              # ticks options                     
        axs[ind].tick_params(which='both', direction='in',width=2)
        axs[ind].tick_params(which='major', direction='in',length=7)
        axs[ind].tick_params(which='minor', direction='in',length=4)
        axs[ind].xaxis.set_ticks_position('both')
        axs[ind].yaxis.set_ticks_position('both')
    
    if signal == 'power law':                                               # add titles to each subplot
        axs[0].set_title('Residues for power law frequency')
    elif signal == 'exp':
        axs[0].set_title('Residues for exp frequency')
    axs[1].set_title('')
    axs[2].set_title('')
    
    axs[0].set_ylabel('residues')                                           # add y labels to the subplots
    axs[1].set_ylabel('residues')
    axs[2].set_ylabel('residues')
    axs[2].set_xlabel('time [days]')
    
    # add a legend to the bottom subplot and adjust its position
    patch1 = mpatches.Patch(color='cyan', label='fit vs candidates')
    patch2 = mpatches.Patch(color='b', label='fit vs true')
    patch3 = mpatches.Patch(color='m', label='candidates vs true')
    
    for ind, patch in zip([0, 1, 2], [patch1, patch2, patch3]):
        axs[ind].legend(handles=[patch], loc='best')
    
    # adjust the layout of the subplots to avoid overlapping
    fig.tight_layout()
    plt.show()




########################################################################################################################




def coeffs_errors(x, y, y_fit):
    
    """
    It computes the standard errors of the fit regression coefficients.
    Parameters:
    ----------------------------------------------------------------------------------------------------------------
    x : (array) x-axis data for fit ---
    y : (array) y-axis data for fit ---
    y_fit : (array) fitted y-axis data ---
    
    ----------------------------------------------------------------------------------------------------------------
    return:
        
    se_slope : (float)
    se_intercept : (float)
    ----------------------------------------------------------------------------------------------------------------   
    """
    
    # Calculate the sum of squared residuals
    ssr = np.sum((y - y_fit)**2)
    
    # Calculate the degrees of freedom
    n = len(x)
    p = 1
    dof = n - p - 1                      # p = 1 because we have 1D data
    
    # Estimate the variance of the error terms
    mse = ssr/dof
    
    # Calculate the standard errors of the slope and intercept
    se_slope = np.sqrt(mse/np.sum((x - np.mean(x))**2))
    se_intercept = np.sqrt(mse*(1/n + np.mean(x)**2/np.sum((x - np.mean(x))**2)))
    
    return se_slope, se_intercept




########################################################################################################################




def candidates_remover(df, x_min, x_max, y_min, y_max):
    
    """
    It removes the candidates within the rectangle defined by the
    values of x_min, x_max, y_min, y_max.
    Parameters:
    ---------------------------------------------------------------
    df : (pandas.core.frame.DataFrame) dataframe with the candidates
    x_min : (float) min value along x-axis
    x_max : (float) max value along x-axis
    y_min : (float) min value along y-axis
    y_max : (float) max value along y-axis
    ---------------------------------------------------------------
    return:
    df : (pandas.core.frame.DataFrame) dataframe with the candidates
          without those that were inside the rectangle
    ---------------------------------------------------------------   
    """
    
    if isinstance(df, str):               # if df_input is the df path, this load the dataframe
        df = pd.read_csv(df)
    else:
        pass

    mask = (df['t_ini'] > x_min) & (df['t_ini'] < x_max) & (df['mu'] > y_min) & (df['mu'] < y_max)
    df = df.loc[~mask]

    return df


