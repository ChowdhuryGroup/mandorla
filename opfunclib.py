# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 17:54:42 2020

@author: Michael
"""
import numpy as np
import scipy.optimize
import scipy.stats as sps

def sellmeier(wavelength,b1,b2,b3,c1,c2,c3): # calculate refractive index of given wavelength in microns
    wl=wavelength
    nsq=1+b1*wl**2/(wl**2-c1)+b2*wl**2/(wl**2-c2)+b3*wl**2/(wl**2-c3)
    n=np.sqrt(nsq)
    return n  

def gaussian(x,A,FWHM,x_off,y_off=0): # Gaussian function
    y=y_off+A*np.exp(-4*np.log(2)*(x-x_off)**2/FWHM**2)
    return y

def gaussfit(x,y): # perform statistics on lineout and use those as initial parameters for nonlinear Gaussian fit
    area=np.sum(y)
    expectval=np.sum(x*y)/area
    stdev=np.sqrt(np.sum((x-expectval)**2*y/area))
    fwhm=stdev*2*np.sqrt(2*np.log(2))
    fit,covar=scipy.optimize.curve_fit(gaussian,x,y,p0=[1,fwhm,expectval,0])
    return fit # by index of 'fit': 0:amplitude, 1:FWHM, 2:center, 3:vertical_offset  

def gausslinefit(x,y): # perform statistics on lineout and use those as initial parameters for nonlinear Gaussian+line fit
    area=np.sum(y)
    expectval=np.sum(x*y)/area
    stdev=np.sqrt(np.sum((x-expectval)**2*y/area))
    fwhm=stdev*2*np.sqrt(2*np.log(2))
    islope=(y[-1]-y[0])/(x[-1]-x[0])
    def gaussline(x,a,fwhm,center,y_off,slope):
        y=gaussian(x,a,fwhm,center,y_off)+x*slope
        return y
    fit,covar=scipy.optimize.curve_fit(gaussline,x,y,p0=[1,fwhm,expectval,y[0],islope])
    return fit # by index of 'fit': 0:amplitude, 1:FWHM, 2:center, 3:vertical_offset, 4:slope

def pseudoerf(x): # approximate of the error function suitable for knife-edge measurements. Taken from Abramowitz and Stegun
    x=np.array(x)
    
    p = 0.47047 
    a1 = 0.3480242 
    a2 = -0.0958798
    a3 = 0.7478556
    t=1/(1+p*np.abs(x))
    erf=np.sign(x)*(1-(a1*t+a2*t**2+a3*t**3)*np.exp(-1*x**2))
    return erf

def stdlinreg(x,y):
    """
    Perform a linear regression by first standardizing x and y.
    """
    x=np.array(x)
    y=np.array(y)
    
    sx=(x-np.mean(x))/np.std(x)
    sy=(y-np.mean(y))/np.std(y)
    fit,cov=np.polyfit(sx,sy,1,cov=True)
    r=fit[0]
    std_r=np.sqrt(cov[0,0])
    # r,pval=sps.pearsonr(x,y)
    
    slope=r*np.std(y)/np.std(x)
    errslope=std_r*np.std(y)/np.std(x)
    const=np.mean(y)-np.mean(x)*r*np.std(y)/np.std(x) 
    return slope,const,errslope




