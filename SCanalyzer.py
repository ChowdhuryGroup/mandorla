# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 09:18:15 2020

@author: Michael
"""

import opticstoolkit as otk
import orchard as orc
import labclerk as lck
import numpy as np
import numpy.ma as ma

def masklevel(array,dB): # change the mask level by some offset factor (in dB)
    offset=np.max(array)*10**(dB/10)
    narray=ma.exp(ma.log(array-offset))+offset
    return narray

def modspec(wavelengths,pump,bandwidth,coeffs=[],noise=True):
    """
    Create a model supercontinuum spectrum with or without noise
    """
    if len(coeffs)==0:
        center=np.random.randint(20,80)
        window=np.random.randint(5,20)
        coef=np.array((center,window))
        
    else:
        coef=coeffs
        
    fqs=otk.freq(wavelengths)
    pumpf=otk.freq(pump)
    bwf=otk.freq(pump-bandwidth/2)-otk.freq(pump+bandwidth/2)    
    def rundulate(x):
        x2=coef[1]/2*x+coef[0]
        y=np.cos(np.sqrt(2)*x2)+np.cos(np.sqrt(3)*x2)+np.cos(2*x2)+3
        return y
    maxval=bwf/2
    x=fqs-pumpf
    peaks=np.piecewise(x,[np.abs(x)<bwf,np.abs(x)>=bwf],[lambda x: rundulate(x/maxval),1e-25])
    if noise:
        noise=np.random.randn(len(peaks))*np.mean(peaks)*.5
    else:
        noise=np.zeros(len(peaks))
        
    spec=np.exp(-2**11*np.log(10)*np.abs((fqs-pumpf)/bwf)**10-peaks+noise)
    spec=spec/np.max(spec)+1e-4
    return spec
    
    

def model(wavelengths,powers,pump,pthres,maxbandwidth,rapidity=4,noise=True):
    """
    Generate a model supercontinuum power scan with the specified parameters. Noise is included.
    """
    bwmax=maxbandwidth
    bwmin=pump/20
    bw=(bwmax-bwmin)/2*np.tanh(rapidity*(powers-pthres))+(bwmax+bwmin)/2
    
    center=np.random.randint(20,100)
    window=np.random.randint(5,20)
    coef=np.array((center,window))
    hmap=[]
    for i in range(len(bw)):
        spec=modspec(wavelengths,pump,bw[i],coef,noise)
        hmap.append(spec)
    
    hmap=np.array(hmap)
    g=otk.graph()
    g.addheatmap('dB',otk.db(hmap),wavelengths*1e9,powers)
    
    return hmap
    
def openr():
    """
    Open raw data files following Andy's format
    """
    tab='\t'
    files=lck.filedlg('Select Automatic Power Scan Files')
    rdata=lck.loadfile(files[0],tab,10,True)
    power=rdata[0,1:].copy()
    wl=rdata[2:,0].copy()
    rdata=rdata[2:,1:]
    return rdata,wl,power

def openp():
    """
    Open raw data files following Andy's format
    """
    tab='\t'
    files=lck.filedlg('Select Automatic Power Scan Files')
    pdata=lck.loadfile(files[0],tab,10,True)
    power=pdata[0,1:].copy()
    wl=pdata[2:,0].copy()
    pdata=pdata[2:,1:]
    return pdata,wl,power

def oautops(): # Open automatic power scan files based on format set by Andy. 
    tab='\t'
    files=lck.filedlg('Select Automatic Power Scan Files')
    rdata=lck.loadfile(files[0],tab,10,True)
    pdata=lck.loadfile(files[1],tab,10,True)
    power=rdata[0,1:].copy()
    wl=rdata[2:,0].copy()
    rdata=rdata[2:,1:]
    pdata=pdata[:,1:]
    return rdata,pdata,wl,power

def ulbd(spectrum,wavelengths,level):
    """
    Find upper and lower bounds of spectrum at given level.
    """
    aspec=lck.mavg(spectrum,5)
    wl=wavelengths[2:-2]
    ind=np.where(otk.db(aspec)>=level)
    lwbd=np.min(ind)
    upbd=np.max(ind)
    
    return wl[lwbd],wl[upbd] 

def bwidth(spectrum,wavelengths,level):
    """
    Calculate the bandwidth of a spectral peak for a given level. Returns bandwidth in frequency.
    """
    bds=ulbd(spectrum,wavelengths,level)
    bw=otk.freq(bds[0])-otk.freq(bds[1])
    
    return bw

def sfm(spectrum): # calculate the spectral flatness measure of a spectrum. DO NOT include wavelength or frequency array
    gmean=ma.exp(ma.sum(ma.log(spectrum)/ma.size(spectrum)))
    mean=ma.mean(ma.exp(ma.log(spectrum)))
    sf=gmean/mean
    return sf

def sfmfull(spectra,powers,row_wise=True,plot=False): # calculate the SFM for entire power scan data set. 'pwr' is mainly used as an index
    sfmarray=[]
    if row_wise==True:
        for i in range(len(powers)):
            sfmarray.append(sfm(spectra[i,:]))
    else:
        for i in range(len(powers)):
            sfmarray.append(sfm(spectra[:,i]))
    
    if plot:
       g=orc.graph('Power','','S.F.M.').addcurve('Data',powers,sfmarray,'r')
    else:
        pass
    sfmarray=np.array(sfmarray)
    return sfmarray

def sgaus(frequency,centfrq,dBlevel,bandwidth,order): # super-Gaussian in dB form
    gaus=-1*dBlevel*np.abs(2*(frequency-centfrq)/bandwidth)**order
    return gaus

def modsgaus(frequency,centfrq,dBlevel,minfrq,maxfrq,lorder,rorder): # a modified super-Gaussian where the left and right halves of the curve have different orders
    leftmask=np.piecewise(frequency,[frequency<centfrq,frequency>=centfrq],[1,0])
    rightmask=(leftmask+1)%2
    
    left=leftmask*sgaus(frequency,centfrq,dBlevel,2*(centfrq-minfrq),lorder)
    right=rightmask*sgaus(frequency,centfrq,dBlevel,2*(maxfrq-centfrq),rorder)
    final=left+right
    
    return final

def sgausfit(wavelengths,intensities,pumpwl,bandwidth,dBlevel): # fit data to super-Gaussian 
    frqs=otk.freq(wavelengths) # convert quantities to frequencies
    pumpf=otk.freq(pumpwl)
    bwf=otk.freq(bandwidth)
    dbi=otk.db(intensities) #convert intensities to dB
    fit,covar=spo.curve_fit(sgaus,frqs,dbi,p0=[pumpf,dBlevel,bwf,2])
    return fit








