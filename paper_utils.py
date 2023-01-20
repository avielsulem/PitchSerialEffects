

import numpy as np
np.random.seed(0)
import pandas as pd
from matplotlib import pyplot as plt
import platform
from scipy.io import loadmat
import pickle
import scipy as sc
import statsmodels.api as sm
from matplotlib import cm
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde
import scipy
import rpy2.robjects as robjects
import feather
from scipy.stats import wilcoxon
from scipy.stats import kruskal

# Data path
path = "C:/Users/user/Dropbox (PPCA)/notebooks/notebooks_LabPC/Data"

import sys
sys.path
# sys.path.append('C:/Users/ilieder/Dropbox/phd/Thesis/chapter3/paper/\
# ilieder/AppData/Local/Continuum/miniconda3/envs/python2/Lib/\
# site-packages/rpy2/R/win-library/3.4/psyphy')
    
    
from utils_full import *

mgcv = importr('mgcv')
base = importr('base')
psyphy= importr('psyphy')
stats = importr('stats')
link = psyphy.probit_2asym(.05,.05)
fam = stats.binomial(link)

from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from rpy2.robjects.conversion import py2ri
from rpy2.robjects import pandas2ri
rpy2.robjects.numpy2ri.activate()
pandas2ri.activate()

def mgcv_fit(df_,model, columns, factors=False,newdata=[]):
    datar= base.data_frame(df_)
    if type(factors) == list:
        for fac in factors:
            datar[columns.index(fac)] = base.as_factor(datar[columns.index(fac)]) # declaring y1 into a factor
    b=mgcv.gam(ro.r(model),\
               data=datar,\
               family=fam,\
               optimizer='perf')

    if isinstance(newdata, pd.DataFrame):
        fs,es = mgcv.predict_gam(b,newdata=newdata,type="terms",se="True")
    else:
        fs,es = mgcv.predict_gam(b,type="terms",se="True", nthreads=8)
    fs = np.asarray(fs)
    es = np.asarray(es)
    return [fs,es]


def mgcv_fit_bam(df_,model,factors=False,newdata=[]):
    datar= base.data_frame(df_)
    if type(factors) == list:
        for fac in factors:
            datar[columns.index(fac)] = base.as_factor(datar[columns.index(fac)]) # declaring y1 into a factor
    b=mgcv.bam(ro.r(model),\
               data=datar,\
               family=fam)
    if isinstance(newdata, pd.DataFrame):
        fs,es = mgcv.predict_bam(b,newdata=newdata,type="terms",se="True")
    else:
        fs,es = mgcv.predict_bam(b,type="terms",se="True", discrete="True",
                                 nthreads=8)
    fs = np.asarray(fs)
    es = np.asarray(es)
    return [fs,es]

def draw_horizonal(ax,x1,x2,y):
    xx = np.linspace(x1,x2,1000)
    yy = np.ones_like(xx)*y
    ax.plot(xx,yy,'--k')
    
    
def draw_vertical(ax,y1,y2,x):
    yy = np.linspace(y1,y2,1000)
    xx = np.ones_like(yy)*x
    ax.plot(xx,yy,'-k',lw=2)
    
def box_plot(ax, data, xlabel = [], ylabel = [], title = '', alpha_face = 1, show_ind_dots = True, colors = []):
    
    l = len(data)
    y = np.arange(l)
    
    noise = []
    ys = []
    for i in range(l):
        ys.append(np.ones_like(data[i])*y[i])
        noise.append(np.random.randn(len(ys[i]))*.07)
    if show_ind_dots:
        for i in range(l):
            ax.plot(ys[i]+noise[i],data[i],'o',ms=2.5,color='k',alpha=.2)
        try:
            ax.plot(ys, data,  '-', color = 'k',alpha=.05)
            ax.plot(ys, np.mean(data,1), '-k',lw=2)
        except:
            pass
    
    bplot1 = ax.boxplot(data , notch = 'True', positions=y, vert=1, patch_artist=True, showmeans=False,\
                        meanprops={"markerfacecolor":"w", "markeredgecolor":"w"},\
                        showfliers = False, widths = 0.3, boxprops=dict(alpha=alpha_face))
    
    for patch,c in zip(bplot1['boxes'],colors):
        patch.set_facecolor(c)
        patch.set_alpha(alpha_face)
        
    for median in bplot1['medians']:
        median.set_color('firebrick')
        median.set(linewidth=0)
        

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.get_xaxis().tick_bottom()
    # ax.get_yaxis().tick_left()
    # ax.set_xticklabels(xlabel, rotation = 0, ha="right")
    # ax.set_xlabel(xlabel, fontsize=20)
    # ax.set_ylabel(ylabel, fontsize=35)    
    # ax.set_title(title, fontsize=20)
    # simpleaxis(ax,20)
    
    
def build_features(data_dict,alphas=[]):
    
    f1 = data_dict['f1'] 
    f1x2 = np.log2(2**f1*2)
    f1x3 = np.log2(2**f1*3)
    f1x4 = np.log2(2**f1*4)

    f2 = data_dict['f2'] 
    Y = data_dict['resp'] 
    acc = data_dict['acc'] 

    try:
        cgroup = data_dict['cgroup'] 
    except: 
        cgroup = np.ones_like(f1)
        
    try:
        flag = data_dict['flag'] 
    except:
        flag = np.ones_like(f1)    
        
    Ytrue = ((f1-f2)<0).astype(int) # what the subject should answer

    lag = 5
    Y1 = Y[:,lag-1:-1]
    d1 = f1[:,lag:]-0.5*(f1[:,lag-1:-1]+f1[:,lag-1:-1])
    d1x2 = f1[:,lag:]-f1x2[:,lag-1:-1]
    d1x3 = f1[:,lag:]-f1x3[:,lag-1:-1]
    d1x4 = f1[:,lag:]-f1x4[:,lag-1:-1]

    d12 = f1[:,lag:]-f2[:,lag-1:-1]
    d2 = f1[:,lag:]-0.5*(f1[:,lag-2:-2]+f2[:,lag-2:-2])
    d22 = f1[:,lag:]-f2[:,lag-2:-2]

    d3 = f1[:,lag:]-0.5*(f1[:,lag-3:-3]+f2[:,lag-3:-3])
    d4 = f1[:,lag:]-0.5*(f1[:,lag-4:-4]+f2[:,lag-4:-4])
    d5 = f1[:,lag:]-0.5*(f1[:,lag-5:-5]+f2[:,lag-5:-5])
    df = (f2-f1)[:,lag:]
    dinf = f1[:,lag:]-(f1.min()+0.5*(f1.max()-f1.min()))
    #dinf = f1[:,lag:]-f1.mean()

    if flag.shape != f1.shape:
        flag = np.ones_like(f1)

    flag = flag[:,lag:]
    acc_raw = acc[:,lag:]

    corr = (Y==Ytrue).astype(int) # whether subject responded correct or wrong
    corr1 = 2*(corr[:,lag-1:-1])-1 # correct or wrong
    corr =  2*(corr[:,lag:]).astype(int)-1 # correct
    Y  = Y[:,lag:]
    Ytrue1  = Ytrue[:,lag-1:-1]
    switch1 = corr1 * Y1 # correct * previous
    
    cgroup  = cgroup[:,lag:]

    dinf_g1 = np.copy(dinf)
    dinf_g2 = np.copy(dinf)
    dinf_g1[cgroup>0] = 0
    dinf_g2[cgroup<=0] = 0
    
    d1_g1 = np.copy(d1)
    d1_g2 = np.copy(d1)
    d1_g1[cgroup>0] = 0
    d1_g2[cgroup<=0] = 0
    
    side = np.copy(dinf)
    side[side<=0] = -1
    side[side>0] = 1

    macc = np.copy(acc[:,lag:])
    for i_s in range(acc.shape[0]):
        macc[i_s,:] = acc[i_s,:].mean()
        
    thresh = .75
    group = (macc < thresh).astype(int)*2-1
    subject = np.tile(np.arange(Y.shape[0]),[Y.shape[1],1]).T
    
    if alphas == []:
        alphas = np.zeros_like(f1[:,1:])
        
    alphas = alphas[:,lag-1:]

    trials = np.tile(np.linspace(0,d1.shape[1]-1,d1.shape[1]),(f1.shape[0],1))
    trials = trials*(macc.max()-macc.min())/trials.max() + macc.min()
    
    d1,d1x2,d1x3,d1x4,d12,d1_g1,d1_g2,d2,d22,d3,d4,d5,df,dinf,dinf_g1,dinf_g2,macc,acc_raw,alphas,flag,\
    Ytrue1,corr,corr1,switch1,group,side,cgroup,subject,Y,Y1 =\
    [u.reshape(1,-1) for u in [d1,d1x2,d1x3,d1x4,d12,d1_g1,d1_g2,d2,d22,d3,d4,d5,df,dinf,dinf_g1,dinf_g2,macc,acc_raw,\
                               alphas,flag,Ytrue1,corr,corr1,switch1,group,side,cgroup,subject,Y,Y1]]
    
    X = np.vstack( [df,d1,d1x2,d1x3,d1x4,d12,d1_g1,d1_g2,d2,d22,d3,d4,d5,dinf,dinf_g1,dinf_g2,macc,acc_raw,alphas,flag,\
                    Ytrue1,corr,corr1,switch1,group,side,cgroup,subject,Y1])
    D = np.vstack([X,Y]).T

    columns=['df','d1','d1x2','d1x3','d1x4','d12','d1_g1','d1_g2','d2','d22','d3','d4','d5','dinf','dinf_g1','dinf_g2','acc',\
             'acc_raw','alphas','flag','ytrue1','corr','corr1','switch1','group','side','cgroup','subject','y1','y']
    df_ = pd.DataFrame(D,columns=columns)
    return df_, columns, data_dict


def load_data():
    path = "C:/Users/user/Dropbox (PPCA)/notebooks/notebooks_LabPC/Data/"

    with open(path + "datasets.p", 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        datasets_dict = u.load()

    fnames = ['complex_pure_mixed_no_missing','complex_pure_mixed_missing_f'\
              ,'complex_pure_mixed_missing_f1']

    dist_names = fnames
    # dist_dict = {0:dist_names[0], 1:dist_names[1], 2:dist_names[2], 3:dist_names[3]}
    # colors_dict = {0:'b',1:'r',2:'g',3:'c'}

    key = 'Complex'

    thresh = 0.0089138
    # thresh = 2

    dfs = []
    datas = []

    for i,fname in enumerate(fnames):
        print(fname)
        data = datasets_dict[key][fname].copy()
        data['resp'] = data['resp']
        acc_lims=(.55,1.)
        data_dict,_ = filter_sub_trials(data,\
                          acc_lims=acc_lims,\
                          consistency_prm={'n_window': 10, 'w_window': 60, 'thresh': thresh},\
                          trial_lims = (0,600))

        alphas = fit_alphas(data_dict,infer='')

        df, columns, data = build_features(data_dict,alphas)
        dfs.append(df)
        datas.append(data)
    return dfs, datas, columns

def organize_data(dfs, columns):
    dfs_org = dfs
    dfs2 = []
    for df in dfs_org:

        df['flag'] = (df['flag']==1) + 0.

        flag_1 = np.hstack([1,df['flag'][1:]])
        flag_2 = np.hstack([1,df['flag'][:-1]])
        dinf_1 = np.hstack([1,df['dinf'][1:]])
        dinf_2 = np.hstack([1,df['dinf'][:-1]])
        dinf = df['dinf']
        df['flags_d1'] = df['flag']
        df['flags_d1'][(flag_1==1) & (flag_2==1)] = 0
        df['flags_d1'][(flag_1==0) & (flag_2==0)] = 1
        df['flags_d1'][(flag_1==1) & (flag_2==0) & (dinf<0)] = 2
        df['flags_d1'][(flag_1==1) & (flag_2==0) & (dinf>0)] = 3

        df['flags_d1'][(flag_1==0) & (flag_2==1) & (dinf_2<0)] = 4
        df['flags_d1'][(flag_1==0) & (flag_2==1) & (dinf_2>0)] = 5

        df['flags_d1'][(flag_1==1) & (flag_2==1) & (dinf_2<0)] = 6
        df['flags_d1'][(flag_1==1) & (flag_2==1) & (dinf_2>0)] = 7

        # df['flags_d1'][(flag_1==1) & (flag_2==0)] = 2

        # df['flags_d1'][(flag_1==0) & (flag_2==1)] = 3

        df['intercept'] = df['dinf']>-1000

        if 'flags_d1' not in columns:
            columns.append('flags_d1')

        if 'intercept' not in columns:
            columns.append('intercept')

        print(len(df))
        df = df[~((df['flags_d1']==1) & (df['d1']<-1))]
        df = df[~((df['flags_d1']==2) & (df['d1']<-1))]
        df = df[~((df['flags_d1']==3) & (df['d1']<0))]
        df = df[~((df['flags_d1']==4) & (df['d1']<-1.))]
        df = df[~((df['flags_d1']==5) & (df['d1']>0.))]
        df = df[~((df['flags_d1']==6) & (df['d1']<-1))]
        df = df[~((df['flags_d1']==6) & (df['dinf']>0) & (df['d1x2']<-1))]

        df = df[~((df['flags_d1']==7) & (df['d1']>1))]
        df = df[~((df['flags_d1']==7) & (df['dinf']<0) & (df['d1']>0))]

        df = df.set_index(np.arange(len(df)))
        dfs2.append(df)

    dfs = dfs2
    return dfs, columns

def combine_data(dfs):
    max_subs = []
    for df in dfs:
        max_subs.append(df['subject'].max())

    df1 = dfs[0].copy()
    df2 = dfs[1].copy()
    df3 = dfs[2].copy()

    df2['subject'] = df2['subject'] + int(max_subs[0])
    df3['subject'] = df3['subject'] + int(max_subs[0]) + int(max_subs[1])
    
    df_comb = df1.append(df2.append(df3))
    df_comb.set_index(np.arange(len(df_comb)))
    return df_comb, df1, df2, df3
