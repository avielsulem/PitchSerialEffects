

import numpy as np
np.random.seed(0)
import pandas as pd
from matplotlib import pyplot as plt
import platform
from scipy.io import loadmat
from sklearn import metrics
# from sklearn.cross_validation import KFold
import pickle
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
#import pandas.rpy.common as com
from rpy2.robjects.conversion import py2ri
from rpy2.robjects import pandas2ri
rpy2.robjects.numpy2ri.activate()
pandas2ri.activate()
import statsmodels.api as sm
import math
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

def draw_horizonal(ax,x1,x2,y):
    xx = np.linspace(x1,x2,1000)
    yy = np.ones_like(xx)*y
    ax.plot(xx,yy,'--k')
    
def text2int(textnum, numwords={}):
    if not numwords:
      units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
      ]

      tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

      scales = ["hundred", "thousand", "million", "billion", "trillion"]

      numwords["and"] = (1, 0)
      for idx, word in enumerate(units):    numwords[word] = (1, idx)
      for idx, word in enumerate(tens):     numwords[word] = (1, idx * 10)
      for idx, word in enumerate(scales):   numwords[word] = (10 ** (idx * 3 or 2), 0)

    current = result = 0
    for word in textnum.split():
        if word not in numwords:
          raise Exception("Illegal word: " + word)

        scale, increment = numwords[word]
        current = current * scale + increment
        if scale > 100:
            result += current
            current = 0

    return result + current

def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

from scipy.stats import zscore

'%%%%%%%%%%%%%%%% Sliding window %%%%%%%%%%%%%%%%'     
def sliding_window(target,n_window=10,w_window=60,info=False):
    nt = len(target)
    step = nt/n_window    
    I_w = [np.arange(step*i,np.min([step*i+w_window,nt])).astype(int) for i in range(n_window)]
    overlap =  len(set(I_w[0]) & set(I_w[1]))
    mtarget = np.array([target[I].mean() for I in I_w])
    if info:
        print('# windows: %d\n width: %d\n overlap %d')%(n_window, w_window, overlap)
    return mtarget, I_w

def filt_inconsistency(acc,n_window=10,w_window=60,thresh=2):
    ns,nt = acc.shape        
    varz = np.array([sliding_window(acc[i_s,:],n_window,w_window)[0].var() for i_s in range(ns)])
    I = varz<=thresh
    return I,np.array(varz)


def check_units(f1,f2,supress=False):
    if np.any(f1>500):
        f1 = np.log2(f1); f2 = np.log2(f2)
        if supress == False: 
            print('in HZ')
    elif np.any((2**f1)>500) and np.any((2**f1)<2500):
        if supress == False: 
            print('in log 2')
        pass
    elif np.any((10**f1)>500) and np.any((10**f1)<2500):
        f1 = f1/np.log(2)*np.log(10); f2 = f2/np.log(2)*np.log(10)
        if supress == False: 
            print('in log 10')
    elif np.any((np.exp(f1))>500):
        if supress == False: 
            print('in log e')
        f1 = f1/np.log(2); f2 = f2/np.log(2)
    return f1,f2


def fix_details(data):
    gender = data['gender']
    age = data['age']
    music = data['musical']
    age =  age.flatten()


    musical = np.zeros_like(age)
    for ii,m in enumerate(music.flatten()):
        for m_ in m[0].split():
            if m_.isdigit():
                musical[ii] = m_
                continue
            else:
                try:
                    musical[ii] = text2int(m_)
                    continue
                except:
                    pass
    musical =  musical.astype(np.float64)


    gender = np.matlib.repmat(gender,300,1).T
    age = np.matlib.repmat(age,300,1).T
    music = np.matlib.repmat(musical,300,1).T
    
    data['gender'] = gender
    data['age'] = age
    data['musical'] = music
    return data


def simpleaxis(ax,fs=15):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
#     ax.set_ylim([-1.1,1.1])
    ax.tick_params(axis='both', which='major', labelsize=fs)


def optimize_probit(df,resp,signed=0):  
    X = np.empty((len(df),2))
    X[:,0] = np.ones(len(df))
    X[:,1] = df
    res_newton = sm.Probit(resp.transpose()+0.,X)
    fit = res_newton.fit(disp=0)
    conf =  fit.conf_int()[0]
    conf2 =  fit.conf_int()[1]

    alpha = fit.params[1]
    #self.alpha = fit.params[0]*self.sigma            
    beta = fit.params[0]
    
    return alpha,beta,conf,conf2


'%%%%%%%%%%%%%%%% plotting tools %%%%%%%%%%%%%%%%'     

def sigmoid(x,slope):
    a = []
    for item in x:
        a.append(1/(1+math.exp(-slope*item)))
    return a


def set_axes(axarr):
    axarr.set_ylabel('Bias',fontsize=15)
    axarr.spines['top'].set_visible(False)
    axarr.spines['right'].set_visible(False)
    axarr.spines['left'].set_visible(False)
#     axarr.spines['buttom'].set_visibl.axis('off')e(False)
    axarr.axis('off')
    axarr.get_xaxis().tick_bottom()
    axarr.get_yaxis().tick_left()
    axarr.get_yaxis().set_ticks([])
    
def plot_response_test(s1,s2,resp,cov='global'):       
    if cov == 'global':
        d = s1[1:] - s1[1:].mean()
    elif cov == 'recency':
        d = s1[1:] - 0.5*(s1+s2)[:-1]

    s = np.argsort(d)
    d = d[s]
    df = (s1[1:] - s2[1:])[s]
    resp = (resp[1:])[s]
    y, I_w = sliding_window(d,n_window=20,w_window=2700)

    r = []
    x = []
    c1 = []
    c2 = []
    for I in I_w:
        sigma,alpha,conf,conf2 = optimize_probit(df[I],resp[I])
        r.append(alpha) 
        x.append(np.mean(d[I]))
        c1.append(conf[0])
        c2.append(conf[1])
    return x,r,c1,c2


def filter_sub_trials(data_dict,
                      acc_lims=(.0,1.),
                      consistency_prm={'n_window': 10, 'w_window': 60, 'thresh': 2},
                      trial_lims = [],
                      exc_sub = [],
                      exc_sub_post = [],
                      inc_sub = [],
                      supress_output = False):
    
    try:
        f1,f2 = check_units(data_dict['s1'],data_dict['s2'],supress=supress_output)
    except:
        f1,f2 = check_units(data_dict['f1'],data_dict['f2'],supress=supress_output)

    data_dict['f1'] = f1
    data_dict['f2'] = f2
    
    if trial_lims == []:
        trial_lims = (0,f1.shape[1])
    
    excludes = ['f1','f2','acc','resp','name','gender','age','musical','rt','rt_break','math','context_tone','onset','ratios','order','ISI','flag','vol']
    iterator_dict = data_dict.copy()
    
    for k in iterator_dict.keys():
        if k not in excludes:
            del data_dict[k]
    
    try:
        data_dict = fix_details(data_dict)
    except:
        pass
    
    try:
        if len(data_dict['name'][0][0][0])>1:
            name = []
            for ii in range(len(data_dict['name'])):
                name.append(data_dict['name'][ii][0][0])
            data_dict['name'] = np.array(name)
    except:
        pass

    
    
    # excluding particular subjects
    Is_sub = np.arange(f1.shape[0])
    Is_sub = np.delete(Is_sub,exc_sub)        
    for k,v in data_dict.items(): 
        try:
            data_dict[k] = v[Is_sub,:]
        except:
            print(v.shape)
            data_dict[k] = v[:,Is_sub]
    # including particular subjects
    if inc_sub != []:
        inc_sub = np.array(inc_sub)
        for k,v in data_dict.items():
            data_dict[k] = v[inc_sub]
        if supress_output == False:
            print('After acc filtering: ',data_dict['f1'].shape[0])
    

            
    nsub = data_dict['f1'].shape[0]
    data_dict['acc'] = data_dict['acc'].astype(float)
    data_dict['resp'] = data_dict['resp'].astype(float)
    
    if supress_output == False:
        print('before filtering: ',data_dict['f1'].shape[0])
        
    Ib = np.arange(trial_lims[0],trial_lims[1]) 
    for k,v in data_dict.items():
        try:
            data_dict[k] = v[:,Ib]
        except:
            pass
            
    acc = data_dict['acc']
    Is = np.where( (acc.mean(1) > acc_lims[0]) & (acc.mean(1) <= acc_lims[1]) )[0] # subselecting good enough subjects
    for k,v in data_dict.items():
        try:
            data_dict[k] = v[Is,:]
        except:
            data_dict[k] = v[:,Is]
            
    if supress_output == False:
        print('After acc filtering: ',data_dict['f1'].shape[0])
    
    Is_var,varz = filt_inconsistency(data_dict['acc'],consistency_prm['n_window'],\
                                consistency_prm['w_window'],consistency_prm['thresh'])
    for k,v in data_dict.items():
        try:
            data_dict[k] = v[Is_var,:]
        except:
            data_dict[k] = v[:,Is_var]
            
    if supress_output == False:
        print('After consistency filtering: ',data_dict['f1'].shape[0])

    # excluding particular subjects, POST other exclusions
    Is_sub = np.arange(data_dict['f1'].shape[0])
    Is_sub = np.delete(Is_sub,exc_sub_post)        
    for k,v in data_dict.items():
        try:
            data_dict[k] = v[Is_sub,:]
        except:
            print(v.shape)
            data_dict[k] = v[:,Is_sub]
    
    if supress_output == False:
        print('total left: ',str(data_dict['f1'].shape[0]) + '/' + str(nsub))
        print('total excluded: ',str((nsub-data_dict['f1'].shape[0])) + '/' + str(nsub))

        print('included: ' +  str(data_dict['f1'].shape[0]*100./nsub)[:4] + ' %' + ' ; excluded: ' \
              +  str((nsub-data_dict['f1'].shape[0])*100./nsub)[:4] + ' %')

    return data_dict,varz


def build_features(data_dict,alphas=[]):
    
    f1 = data_dict['f1'] 
    f1x2 = np.log2(2**f1*2)
    f1x3 = np.log2(2**f1*3)
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

    trials = np.matlib.repmat(np.linspace(0,d1.shape[1]-1,d1.shape[1]),f1.shape[0],1)
    trials = trials*(macc.max()-macc.min())/trials.max() + macc.min()
    
    d1,d1x2,d1x3,d12,d1_g1,d1_g2,d2,d22,d3,d4,d5,df,dinf,dinf_g1,dinf_g2,macc,alphas,trials,flag,\
    Ytrue1,corr,corr1,switch1,group,side,cgroup,subject,Y,Y1 =\
    [u.reshape(1,-1) for u in [d1,d1x2,d1x3,d12,d1_g1,d1_g2,d2,d22,d3,d4,d5,df,dinf,dinf_g1,dinf_g2,macc,alphas,trials,flag,\
                               Ytrue1,corr,corr1,switch1,group,side,cgroup,subject,Y,Y1]]
    
    X = np.vstack( [df,d1,d1x2,d1x3,d12,d1_g1,d1_g2,d2,d22,d3,d4,d5,dinf,dinf_g1,dinf_g2,macc,alphas,trials,flag,\
                    Ytrue1,corr,corr1,switch1,group,side,cgroup,subject,Y1])
    D = np.vstack([X,Y]).T

    columns=['df','d1','d1x2','d1x3','d12','d1_g1','d1_g2','d2','d22','d3','d4','d5','dinf','dinf_g1','dinf_g2','acc','alphas','trials','flag',\
             'ytrue1','corr','corr1','switch1','group','side','cgroup','subject','y1','y']
    df_ = pd.DataFrame(D,columns=columns)
    return df_, columns, data_dict


def build_features_ISI(data_dict,alphas=[]):
    
    f1 = data_dict['f1'] 
    f2 = data_dict['f2'] 
    Y = data_dict['resp'] 
    acc = data_dict['acc'] 
    ISI = data_dict['ISI'] 

    try:
        cgroup = data_dict['cgroup'] 
    except: 
        cgroup = np.ones_like(f1)
        
    Ytrue = ((f1-f2)<0).astype(int) # what the subject should answer

    lag = 5
    Y1 = Y[:,lag-1:-1]
    d1 = f1[:,lag:]-f1[:,lag-1:-1]
    d12 = f1[:,lag:]-f2[:,lag-1:-1]
    d2 = f1[:,lag:]-f1[:,lag-2:-2]
    d3 = f1[:,lag:]-f1[:,lag-3:-3]
    d4 = f1[:,lag:]-f1[:,lag-4:-4]
    d5 = f1[:,lag:]-f1[:,lag-5:-5]
    df = (f2-f1)[:,lag:]
    dinf = f1[:,lag:]-f1.mean()
    ISI = ISI[:,lag:]
    ISI[ISI==100] = 0
    ISI[ISI==300] = 0
    #ISI[ISI==700] = 1

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
    side[side<0] = -1
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

    trials = np.matlib.repmat(np.linspace(0,d1.shape[1]-1,d1.shape[1]),f1.shape[0],1)
    trials = trials*(macc.max()-macc.min())/trials.max() + macc.min()
    
    d1,d12,d1_g1,d1_g2,d2,d3,d4,d5,df,dinf,dinf_g1,dinf_g2,macc,alphas,trials,ISI,\
    Ytrue1,corr,corr1,switch1,group,side,cgroup,subject,Y,Y1 =\
    [u.reshape(1,-1) for u in [d1,d12,d1_g1,d1_g2,d2,d3,d4,d5,df,dinf,dinf_g1,dinf_g2,macc,alphas,trials,ISI,\
                               Ytrue1,corr,corr1,switch1,group,side,cgroup,subject,Y,Y1]]
    
    X = np.vstack( [df,d1,d12,d1_g1,d1_g2,d2,d3,d4,d5,dinf,dinf_g1,dinf_g2,macc,alphas,trials,ISI,\
                    Ytrue1,corr,corr1,switch1,group,side,cgroup,subject,Y1])
    D = np.vstack([X,Y]).T

    columns=['df','d1','d12','d1_g1','d1_g2','d2','d3','d4','d5','dinf','dinf_g1','dinf_g2','acc','alphas','trials','ISI',\
             'ytrue1','corr','corr1','switch1','group','side','cgroup','subject','y1','y']
    df_ = pd.DataFrame(D,columns=columns)
    return df_, columns, data_dict


def build_features_math(data_dict,alphas=[]):
    
    f1 = data_dict['f1'] 
    f2 = data_dict['f2'] 
    Y = data_dict['resp'] 
    acc = data_dict['acc'] 
    maths = data_dict['math']
    
    try:
        cgroup = data_dict['cgroup'] 
    except: 
        cgroup = np.ones_like(f1)
        
    Ytrue = ((f1-f2)<0).astype(int) # what the subject should answer

    lag = 5
    Y1 = Y[:,lag-1:-1]
    d1 = f1[:,lag:]-f1[:,lag-1:-1]
    d12 = f1[:,lag:]-f2[:,lag-1:-1]
    d2 = f1[:,lag:]-f1[:,lag-2:-2]
    d3 = f1[:,lag:]-f1[:,lag-3:-3]
    d4 = f1[:,lag:]-f1[:,lag-4:-4]
    d5 = f1[:,lag:]-f1[:,lag-5:-5]
    df = (f2-f1)[:,lag:]
    dinf = f1[:,lag:]-f1.mean()
    
    maths[maths==0] = 1
    maths[maths == 55] = 0
    pre_math = maths[:,lag-1:-1]
    maths = maths[:,lag:]
    
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
    side[side<0] = -1
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

    trials = np.matlib.repmat(np.linspace(0,d1.shape[1]-1,d1.shape[1]),f1.shape[0],1)
    trials = trials*(macc.max()-macc.min())/trials.max() + macc.min()
    
    d1,d12,d1_g1,d1_g2,d2,d3,d4,d5,df,dinf,dinf_g1,dinf_g2,macc,alphas,trials,maths,pre_math,\
    Ytrue1,corr,corr1,switch1,group,side,cgroup,subject,Y,Y1 =\
    [u.reshape(1,-1) for u in [d1,d12,d1_g1,d1_g2,d2,d3,d4,d5,df,dinf,dinf_g1,dinf_g2,macc,alphas,trials,maths,pre_math,\
                               Ytrue1,corr,corr1,switch1,group,side,cgroup,subject,Y,Y1]]
    
    X = np.vstack( [df,d1,d12,d1_g1,d1_g2,d2,d3,d4,d5,dinf,dinf_g1,dinf_g2,macc,alphas,trials,pre_math,\
                    Ytrue1,corr,corr1,switch1,group,side,cgroup,subject,Y1])
    D = np.vstack([X,Y]).T
    D = D[np.where((maths==0)[0])[0],:]
    columns=['df','d1','d12','d1_g1','d1_g2','d2','d3','d4','d5','dinf','dinf_g1','dinf_g2','acc','alphas','trials','pre_math',\
             'ytrue1','corr','corr1','switch1','group','side','cgroup','subject','y1','y']
    df_ = pd.DataFrame(D,columns=columns)

    return df_, columns, data_dict

def build_features_time(data_dict,alphas=[]):
    
    f1 = data_dict['f1'] 
    f2 = data_dict['f2'] 
    Y = data_dict['resp'] 
    acc = data_dict['acc'] 
    context_tone = data_dict['context_tone'] 
    onset = data_dict['onset'] 
    
    try:
        cgroup = data_dict['cgroup'] 
    except: 
        cgroup = np.ones_like(f1)
        
    Ytrue = ((f1-f2)<0).astype(int) # what the subject should answer

    lag = 5
    Y1 = Y[:,lag-1:-1]
    d1 = f1[:,lag:]-f1[:,lag-1:-1]
    d12 = f1[:,lag:]-f2[:,lag-1:-1]
    d2 = f1[:,lag:]-f1[:,lag-2:-2]
    d3 = f1[:,lag:]-f1[:,lag-3:-3]
    d4 = f1[:,lag:]-f1[:,lag-4:-4]
    d5 = f1[:,lag:]-f1[:,lag-5:-5]
    df = (f2-f1)[:,lag:]
    dinf = f1[:,lag:]-f1.mean()
    context_tone = f1[:,lag:] - np.log2(context_tone[:,lag:])
    onset = onset[:,lag:]

    
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
    side[side<0] = -1
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

    trials = np.matlib.repmat(np.linspace(0,d1.shape[1]-1,d1.shape[1]),f1.shape[0],1)
    trials = trials*(macc.max()-macc.min())/trials.max() + macc.min()
    
    d1,d12,d1_g1,d1_g2,d2,d3,d4,d5,df,dinf,dinf_g1,dinf_g2,macc,alphas,trials,context_tone,onset,\
    Ytrue1,corr,corr1,switch1,group,side,cgroup,subject,Y,Y1 =\
    [u.reshape(1,-1) for u in [d1,d12,d1_g1,d1_g2,d2,d3,d4,d5,df,dinf,dinf_g1,dinf_g2,macc,alphas,trials,context_tone,onset,\
                               Ytrue1,corr,corr1,switch1,group,side,cgroup,subject,Y,Y1]]
    
    X = np.vstack( [df,d1,d12,d1_g1,d1_g2,d2,d3,d4,d5,dinf,dinf_g1,dinf_g2,macc,alphas,trials,context_tone,onset,\
                    Ytrue1,corr,corr1,switch1,group,side,cgroup,subject,Y1])
    D = np.vstack([X,Y]).T

    columns=['df','d1','d12','d1_g1','d1_g2','d2','d3','d4','d5','dinf','dinf_g1','dinf_g2','acc','alphas','trials','context_tone',\
             'onset','ytrue1','corr','corr1','switch1','group','side','cgroup','subject','y1','y']
    df_ = pd.DataFrame(D,columns=columns)
    return df_, columns, data_dict

def build_features_personal(data_dict,alphas=[]):
    
    f1 = data_dict['f1'] 
    f2 = data_dict['f2'] 
    Y = data_dict['resp'] 
    acc = data_dict['acc'] 
    gender = data_dict['gender'] 
    age = data_dict['age'] 
    music = data_dict['musical'] 
  
    lag = 5
    Y1 = Y[:,lag-1:-1]
    Y  = Y[:,lag:]
    d1 = f1[:,lag:]-f1[:,lag-1:-1]
    d2 = f1[:,lag:]-f1[:,lag-2:-2]
    d3 = f1[:,lag:]-f1[:,lag-3:-3]
    d4 = f1[:,lag:]-f1[:,lag-4:-4]
    d5 = f1[:,lag:]-f1[:,lag-5:-5]
    df = (f2-f1)[:,lag:]
    dinf = f1[:,lag:]-f1.mean()
    
    side = np.copy(dinf)
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
    
    gender = gender[:,lag:]
    age = age[:,lag:]
    music = music[:,lag:]

    trials = np.matlib.repmat(np.linspace(0,d1.shape[1]-1,d1.shape[1]),f1.shape[0],1)
    trials = trials*(macc.max()-macc.min())/trials.max() + macc.min()
    
    d1,d2,d3,d4,d5,df,dinf,macc,alphas,trials,gender,age,music,group,side,subject,Y,Y1 =\
    [u.reshape(1,-1) for u in [d1,d2,d3,d4,d5,df,dinf,macc,alphas,trials,gender,age,music,group,side,subject,Y,Y1]]
    X = np.vstack( [df,d1,d2,d3,d4,d5,dinf,macc,alphas,trials,gender,age,music,group,side,subject,Y1])
    D = np.vstack([X,Y]).T

    columns=['df','d1','d2','d3','d4','d5','dinf','acc','alphas','trials',\
             'gender','age','music','group','side','subject','y1','y']
    df_ = pd.DataFrame(D,columns=columns)
    return df_, columns, data_dict


def build_features_subject(data_dict,subject):

    f1 = data_dict['f1'] 
    f2 = data_dict['f2']
    Y = data_dict['resp'] 
    acc = data_dict['acc'] 
    
    lag = 1
    Y1 = Y[subject,lag-1:-1]
    Y  = Y[subject,lag:]
    d1 = f1[subject,lag:]-f1[subject,lag-1:-1]
    df = (f2-f1)[subject,lag:]
    dinf = f1[subject,lag:]-f1.mean()
    trials = np.linspace(0,f1.shape[1]-2,f1.shape[1]-1)+1
    mf = 0.5*(f1[subject,lag:]+f2[subject,lag:])
    mf = np.round(mf)
#     mf = (mf-mf.min())
#     mf = mf/mf.max()

    d1,df,dinf,trials,mf,Y =\
    [u.reshape(1,-1) for u in [d1,df,dinf,trials,mf,Y]]

    df[df==0] = 0.5
    Y[df==0] = 1 
        
    X = np.vstack( [df,d1,dinf,trials,mf])
    D = np.vstack([X,Y]).T

    columns=['df','d1','dinf','trials','mf','y']
    df_ = pd.DataFrame(D,columns=columns)
    return df_, columns, data_dict


def plot_with_errors(ax,x,y,err,color,alpha=.4,label='_nolegend_'):
        ax.plot(x,y,'-',color=color,label='_nolegend_')
        ax.plot(x,y-err,'--',color=color,label='_nolegend_')
        ax.plot(x,y+err,'--',color=color,label='_nolegend_')  
        ax.fill_between(x,y-err,y+err,color=color,alpha=alpha,label=label)

        
        
def fit_alphas(data,infer='all'):
    ns,nt = data['f1'].shape
    dfs, datas = [],[]
    for i_sub in range(ns):
        df, columns, data = build_features_subject(data,i_sub)
        dfs.append(df)
        datas.append(data)
        
    mgcv = importr('mgcv')
    base = importr('base')
    psyphy= importr('psyphy')
    stats = importr('stats')
    link = psyphy.probit_2asym(.05,.05)
    fam = stats.binomial(link)

    if infer == 'all':
        model = 'y~s(trials,mf,by=df)'
    elif infer == 'trials':
        model = 'y~s(trials,by=df)'
    elif infer == 'mf':
        model = 'y~s(mf,by=df)'
    else:
        model = 'y~s(df)'


    A = []
    E = []
    for df in dfs:
        datar= base.data_frame(df)
        b=mgcv.gam(ro.r(model),\
                   data=datar,\
                   family=fam,\
                  optimizer='perf')

        fs,es = mgcv.predict_gam(b,type="terms",se="True")
        # se: standard error estimates
        A.append(np.asarray(fs))
        E.append(np.asarray(es))

    alphas = []
    for i in range(len(dfs)):
        alphas.append(np.squeeze(A[i])/dfs[i]['df'].values)
    alphas = np.array(alphas)
    return alphas


    
def multicolor_ylabel(ax,list_of_strings,list_of_colors,axis='x',size=30,anchorpad=0,**kw):
    """this function creates axes labels with multiple colors
    ax specifies the axes object where the labels should be drawn
    list_of_strings is a list of all of the text items
    list_if_colors is a corresponding list of colors for the strings
    axis='x', 'y', or 'both' and specifies which label(s) should be drawn"""
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

    # x-axis label
    if axis=='x' or axis=='both':
        boxes = [TextArea(text, textprops=dict(color=color, size=size, ha='left',va='bottom',**kw)) 
                    for text,color in zip(list_of_strings,list_of_colors) ]
        xbox = HPacker(children=boxes,align="center",pad=0, sep=5)
        anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=anchorpad,frameon=False,bbox_to_anchor=(0.3, -0.25),
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_xbox)

    # y-axis label
    if axis=='y' or axis=='both':
        boxes = [TextArea(text, textprops=dict(color=color, size=size, ha='left',va='bottom',rotation=90,**kw)) 
                     for text,color in zip(list_of_strings[::-1],list_of_colors[::-1]) ]
        ybox = VPacker(children=boxes,align="center", pad=0, sep=5)
        anchored_ybox = AnchoredOffsetbox(loc=3, child=ybox, pad=anchorpad, frameon=False, bbox_to_anchor=(-0.17, 0.1), 
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_ybox)
        
        
def build_features_bimodal(data_dict,alphas=[]):
    
    f1 = data_dict['f1'] 
    f2 = data_dict['f2'] 
    Y = data_dict['resp'] 
    acc = data_dict['acc'] 
  
    lag = 5
    Y1 = Y[:,lag-1:-1]
    Y  = Y[:,lag:]
    d1 = f1[:,lag:]-f1[:,lag-1:-1]
    d2 = f1[:,lag:]-f1[:,lag-2:-2]
    d3 = f1[:,lag:]-f1[:,lag-3:-3]
    d4 = f1[:,lag:]-f1[:,lag-4:-4]
    d5 = f1[:,lag:]-f1[:,lag-5:-5]
    df = (f2-f1)[:,lag:]
    dinf = f1[:,lag:]-f1.mean()
    
    dinf_plus = np.copy(dinf)
    dinf_plus[dinf_plus<0] = 0
    
    dinf_minus = np.copy(dinf)
    dinf_minus[dinf_minus>0] = 0
    
#     dinf_plus = np.copy(dinf)
#     dinf_plus[dinf_plus<0] = np.mean(dinf_plus[dinf_plus>0])
    
#     dinf_minus = np.copy(dinf)
#     dinf_minus[dinf_minus>0] = np.mean(dinf_plus[dinf_plus<0])

    macc = np.copy(acc[:,lag:])
    for i_s in range(acc.shape[0]):
        macc[i_s,:] = acc[i_s,:].mean()
        
    thresh = .75
    group = (macc < thresh).astype(int)*2-1
    subject = np.tile(np.arange(Y.shape[0]),[Y.shape[1],1]).T
    
    if alphas == []:
        alphas = np.zeros_like(f1[:,1:])
        
    alphas = alphas[:,lag-1:]

    trials = np.matlib.repmat(np.linspace(0,d1.shape[1]-1,d1.shape[1]),f1.shape[0],1)
    trials = trials*(macc.max()-macc.min())/trials.max() + macc.min()
    
    d1,d2,d3,d4,d5,df,dinf_plus,dinf_minus,macc,alphas,trials,group,subject,Y,Y1 =\
    [u.reshape(1,-1) for u in [d1,d2,d3,d4,d5,df,dinf_plus,dinf_minus,macc,alphas,trials,group,subject,Y,Y1]]
    
    X = np.vstack( [df,d1,d2,d3,d4,d5,dinf_plus,dinf_minus,macc,alphas,trials,group,subject,Y1])
    D = np.vstack([X,Y]).T

    columns=['df','d1','d2','d3','d4','d5','dinf_plus','dinf_minus','acc','alphas','trials',\
             'group','subject','y1','y']
    df_ = pd.DataFrame(D,columns=columns)
    return df_, columns, data_dict

def cross_validate(df_,ntrials,models,columns,factor=''):

    mgcv = importr('mgcv')
    base = importr('base')
    psyphy= importr('psyphy')
    stats = importr('stats')
    link = psyphy.probit_2asym(.05,.05)
    fam = stats.binomial(link)

    n_folds = 10

    ntrials = ntrials-5 # lag = 1
    nsub = int(df_.shape[0]/ntrials)
    print(nsub,nsub*ntrials,ntrials)
    AUCs = []
    for model in models:
        aucs = []
        kf = KFold(n=ntrials, n_folds=n_folds, shuffle=True, random_state=111)
        print(model)
        for train_index, test_index in kf:
            train_index_ = np.concatenate( [train_index + i*(ntrials)   for i in range(nsub)] )
            test_index_ = np.concatenate( [test_index + i*(ntrials)   for i in range(nsub)] )
            
            # train and test subsets
            df_train,df_test = df_.loc[train_index_], df_.loc[test_index_]
            y_true = df_test['y'].as_matrix()
            datar= base.data_frame(df_train)
            if type(factor)==list:
                for fac in factor:
                    datar[columns.index(fac)] = base.as_factor(datar[columns.index(fac)]) 
            
            b=mgcv.gam(ro.r(model),\
                  data=base.data_frame(datar),\

                  family=fam,\
                    optimizer='perf')
            # testing model on test set
            
            datar= base.data_frame(df_test)
            if type(factor)==list:
                for fac in factor:
                    datar[columns.index(fac)] = base.as_factor(datar[columns.index(fac)]) 
            y_pred = mgcv.predict_gam(b,\
                                      base.data_frame(datar),\
                                     type="response")
            y_pred = np.asarray(y_pred)
            # Computing AUC score
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
            roc_auc = metrics.auc(fpr, tpr)

            print(roc_auc)
            aucs.append(roc_auc)
        AUCs.append(aucs)
    return AUCs

# ====== Bimodal split code ===========
# f, axarr = plt.subplots(1,2,figsize=[8,3],sharey=True,sharex=True)
# for i,(d,title,d_latex) in enumerate(zip(['d1',['dinf_minus','dinf_plus']],['Recency bias','Long-term bias'],['$d_1$','$d_\infty$'])):
#     ax = axarr[i]
#     for p,c in zip(range(2),['b','g']):
        
#         if type(d)!=list:

#             s = np.argsort(dfs[p][d])
#             x = dfs[p][d][s]
#             y = A[p][:,i+1][s]
#             err = E[p][:,i+1][s]
#             plot_with_errors(ax,x,y,err,color=c)
            
#         else:
#             print d
#             s = np.argsort(dfs[p][d[0]])
#             s = s[np.where(dfs[p][d[0]][s]!=0)[0]]
#             x = dfs[p][d[0]][s]
#             y = A[p][:,i+1][s]
#             err = E[p][:,i+1][s]
#             plot_with_errors(ax,x,y,err,color=c)
            
#             s = np.argsort(dfs[p][d[1]])
#             s = s[np.where(dfs[p][d[1]][s]!=0)[0]]
#             x = dfs[p][d[1]][s]
#             y = A[p][:,i+2][s]
#             err = E[p][:,i+2][s]
#             plot_with_errors(ax,x,y,err,color=c)