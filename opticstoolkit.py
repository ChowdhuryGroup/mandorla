# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:18:24 2020

@author: Michael
"""
import numpy as np
import numpy.ma as ma
import numpy.fft as fft
import scipy.optimize as spo
import scipy.special as sps
import matplotlib.pyplot as plt
from orchard import graph
import opfunclib as ofl

def freq(wavelengths,angular=False): # convert wavelengths to frequencies with option to  use angular frequencies
    c=2.99792458e8 # units: m/s
    if angular:
        f=np.pi*2*c/wavelengths
    else:
        f=c/wavelengths
    return f
    
def stdcorr(n): #statistical correction factor for standard deviation of sample size n
    C=np.sqrt((n-1)/2)*sps.erf((n-1)/2)*sps.erf(n/2)
    return C

def knifeedge(position,power,plot=True): # calculates the estimated beam waist from a knife edge measurement. Units of output are same as input. Currently, it only handles case where power increases with position.
    position=np.array(position)
    power=np.array(power)
    
    def powerscale(x,x0,y0,P0,r0): # power relation for a knife edge measurement. x0 is the displacement, P0 is max power, and r0 is 1/e^2 radius of the beam
        
        p=y0+P0/2*(1+ofl.pseudoerf(np.sqrt(2)*(x-x0)/r0))
        return p
    
    r_guess=np.min(np.diff(position))
    
    output,covar=spo.curve_fit(powerscale,position,power,p0=[np.mean(position),np.min(power),np.max(power)/2,r_guess],maxfev=5000)
    
    stdev=np.sqrt(np.diag(covar))
    
    fitx=np.linspace(np.min(position),np.max(position),100)
    fit=powerscale(fitx,*output)
    
    if plot:
        g=graph('Position','','Power')
        g.addpoints('Data',position,power,'k')
        g.addcurve('Fit',fitx,fit,'r')
    else:
        g='No plot generated'
        
    return g,output,stdev   

def neff(no,ne,rads): # calculate the extraordinary (or effective) index as a function of angle in radians    
    n=1/np.sqrt(np.cos(rads)**2/no**2+np.sin(rads)**2/ne**2)
    return n

def feff(f1,f2,separation=0): # calculate effective focal length of compounded lens
    f=f2*(f1-separation)/(f1+f2-separation)
    return f

def paraWL(wl1,wl2): # input two wavelengths and return the third wavelength for a parametric process
        invwl3=np.abs(1/wl1-1/wl2)
        wl3=1/invwl3
        return wl3
 
def spotsize(iwaist,f,n,wavelength): # calculate waist size at focus of lens with focal length f and initial size w0. Waists are diameters.
    w0= iwaist/2
    waist=w0/2*np.sqrt(1+f**2/(np.pi*n*w0**2/wavelength)**2) 
    return 2*waist     
    
def db(data,maxval=None): # convert data to decibels using numpy.ma module which will automatically mask entries where x<=0.
    cdata=np.copy(data)
    ndata=10*ma.log10(cdata/ma.max(cdata))
    ndata.fill_value=ma.min(ndata)
    return ndata    

def kerrlens(avgPower,waist,Tpulse,n0,n2,threshold=0,reprate=1000): # an equation for the "focal length" of a beam undergoing self-focusing in a fiber. 'waist' and 'Tpulse' are FWHM quantities
    const=np.pi**(3/2)/np.log(2)**(5/2)/32
    fsq=const*n0*waist**4*Tpulse*reprate/(n2*(avgPower-threshold))
    f=np.sqrt(fsq)
    return f

def zf(wavelength,waist,power,pcrit):
    """
    Calculate the focus due to self-focusing. Waist is a FWHM quantity.
    Equation taken from Brodeur and Chin, JOSAB 16 (1999)
    """
    k=2*np.pi/wavelength
    a=waist/2/np.sqrt(np.log(2))
    z=0.367*k*a**2/np.sqrt((np.sqrt(power/pcrit)-0.852)**2-.0219)
    return z

def powerscale(x,y): # calculate the power scaling of a dataset
    logy=np.log(y)
    logx=np.log(x)
    def line(x,power,logconst): # line for a log-log plot
        y=power*x+logconst
        return y
    fit,var=spo.curve_fit(line,logx,logy)
    power=fit[0]
    const=np.exp(fit[1])
    return power,const

def ld(Tpulse,GVD): # classical characteristic dispersion length. 'Tpulse' is a FWHM quantity
    l=Tpulse**2/(4*np.log(2)*np.abs(GVD))
    return l

def lnl(P0,wavelength,n2,waist): # characteristic nonlinear length scale. 'waist' is a FWHM quantity
    c=299792458 # units: m/s
    w0=2*np.pi*c/wavelength
    l=np.pi*waist**2*c/(2*np.log(2)*w0*n2*P0)
    return l

def leff(length, alpha): # calcualte effective length for a fiber
    if alpha==0:
        l=length
    else:        
        l=(1-np.exp(-alpha*length))/alpha
    return l

def pcrit(wavelength,n0,n2):
    """
    Calculate critical power of a material
    """
    p=3.77*wavelength**2/(8*np.pi*n0*n2)
    return p

def fouriert(t2f,Efield,t_or_f,carrierwl=0,applyshift=False):
    """
    Perform a Fourier transform from time (t) to frequency (f) domain or vice versa. Direction specified by Boolean argument 't2f'.
    """
    if t2f:
        t=t_or_f
        dt=(np.max(t)-np.min(t)/len(t))
        ftE=fft.fft(Efield)
        f=fft.fftfreq(len(t),dt)+freq(carrierwl)
        domain=f
    else:
        f=t_or_f
        df=(np.max(f)-np.min(f)/len(f))
        ftE=fft.ifft(Efield)
        t=fft.fftfreq(len(f),df)
        domain=t
        
    if applyshift:
        ftE=fft.fftshift(ftE)
        domain=fft.fftshift(domain)
    else:
        pass
    
    return ftE,domain

def tlgauss(wavelengths,spectrum):
    """
    A function designed to take an input spectrum a calculate the transform limited pulse with an estimate of the pulse duration (FWHM).
    """
    wl=wavelengths
    f=freq(wl)
    s=spectrum
    
    #bandwidth approximation
    gfit=ofl.gaussfit(f,s)
    fwhm=0.441/gfit[1]
    
    s[s<0]=0 #remove negative values for taking square root and Fourier Transform
    
    E=np.sqrt(s) # recall that spectrum is square of field in frequency domain
    
    fts,time=fouriert(False,E,f,applyshift=True)**2 #recall that intensity is square of field in time domain
        
    tlgfit=ofl.gaussfit(time,fts)
        
    return fts,time,tlgfit[1],fwhm

def kwplot(spectra2d,x,y,x0y0,flength):
    """
    Create a k-omega plot from 2D profile spectra.
    """
    X,Y=np.meshgrid(x,y)
    sp=np.array(spectra2d)
    x0,y0=x0y0
    cnum=(X-x0)+1j*(Y-y0)
    radius=np.abs(cnum)
    angles=np.arctan(radius/flength)
    
    return angles

def fresnel(aoi,n1,n2,s_or_p,aoi_in_degrees=False):
   """
   Calculate the reflection and transmission coefficients according to Fresnel's equations. Return reflection and tranmission together as a tuple.
   """   
   if aoi_in_degrees:
       theta_i=np.radians(aoi)
   else:
       theta_i=aoi
       
   theta_t=np.sqrt(1-(n1/n2*np.sin(theta_i))**2)
   if s_or_p=='s':
       reflection=np.abs((n1*np.cos(theta_i)-n2*np.cos(theta_t))/(n1*np.cos(theta_i)+n2*np.cos(theta_t)))**2
       
   elif s_or_p=='p':
       reflection=np.abs((n1*np.cos(theta_t)-n2*np.cos(theta_i))/(n1*np.cos(theta_t)+n2*np.cos(theta_i)))**2
       
   else: 
       raise ValueError('s_or_p must be \'s\' or \'p\'')
       
   transmission=1-reflection    
   return reflection,transmission

class pmatchcond: # A class that takes input functions representing both sides of the phase-matching condition
    def __init__(self,f1,f2):      
        self.f1=f1
        self.f2=f2
        return
    
    def plot(self): # plot both sides of the phase-matching condition to see if there is a solution
        degs=np.linspace(0,90,100)
        
        plt.xlabel('Angle (deg.)')
        plt.ylabel('Mod. Index')
        self.plot=plt.plot(degs,self.f1(np.radians(degs)),degs,self.f2(np.radians(degs)))
        return     
             
    def optfunc(self,angle): # the function that is actually solved numerically
        fx=self.f1(angle)-self.f2(angle)
        return fx
    
    def solve(self,upbound,lowbound): #solve phase-matching condition with Brent method. Returns angle in degrees.
        soln=spo.brentq(self.optfunc,upbound,lowbound)
        return soln*180/np.pi
    
class consolver: # A general class that takes input functions representing both sides of an algebraic equation and provides tools to solve it. This is originally meant for things like phase matching or dispersion conditions found in optics.
    def __init__(self,f1,f2):      
        self.f1=f1
        self.f2=f2
        return
    
    def plot(self,varparam): # plot both sides of the phase-matching condition to see if there is a solution
        degs=np.linspace(0,90,100)
        
        plt.xlabel('Angle (deg.)')
        plt.ylabel('Mod. Index')
        self.plot=plt.plot(degs,self.f1(np.radians(degs)),degs,self.f2(np.radians(degs)))
        return     
             
    def optfunc(self,angle): # the function that is actually solved numerically
        fx=self.f1(angle)-self.f2(angle)
        return fx
    
    def solve(self,upbound,lowbound): #solve phase-matching condition with Brent method. Returns angle in degrees.
        soln=spo.brentq(self.optfunc,upbound,lowbound)
        return soln*180/np.pi     
    
        
    
        