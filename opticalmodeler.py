# -*- coding: utf-8 -*-
"""

@author: Michael Tripepi

This is the code used for HW #4 Problem 5
"""
#-- import necessary Python packages
import numpy as np # used for mathematical tools
import numpy.fft as fft # for Fourier transforms
import opticstoolkit as otk # for basic optics quantities
import opfunclib as ofl

sf1=(1.60865158,0.237725916,1.51530653,0.0119654879,0.0590589722,135.521676) #SF1 glass
sf2=(1.40301821, 0.231767504, 0.939056586, 0.0105795466, 0.0493226978, 112.405955) #SF2 glass
bk7=(1.03961212,2.31792344e-01,1.01046945,6.00069867e-3,2.00179144e-2,1.03560653E+02) #BK7 glass
yag=(2.282,3.27644,0,0.01185,282.734,0) # yttrium aluminum garnet
gan_o=(4.199,3.625,0,0.1753**2,17.05**2,0) # gallium nitride (ordinary)
gan_e=(4.347,2.964,0,0.1781**2,15.23**2,0) # gallium nitride (extraordinary)
sapph=(1.4313493,0.65054713,5.3414021,0.0726631**2,0.1193242**2,18.028251**2) # sapphire crystal (Al2O3)
baf=(0.643356,0.506762,3.8261,0.057789**2,0.10968**2,46.3864**2) # barium fluoride
caf=(0.5675888,0.4710914,3.8484723,0.050263605**2,0.1003909**2,34.649040**2) # calcium fluoride
mgf=(0.48755108,0.39875031,2.3120353,0.04338408**2,0.09461442**2,23.793604**2) # magnesium fluoride
lif=(0.92549,6.96747,0,0.07376**2,32.790**2,0) # lithium fluoride
znse=(4.45813734,0.467216334,2.89566290,0.200859853**2,0.391371166**2,47.1362108**2) # zinc selenide
fus_sio2=(0.6961663,0.0684043**2,0.4079426,0.1162414**2,0.8974794,9.896161**2) #fused silica

def bs(inputE,a=0.5): 
    """
    Beamsplitter with 'a' denoting the ratio of power transmitted through one arm. 'inoutE' is the amplitude of the input field and 'outputA' and 'outputB' are the output fields for the two arms of the beamsplitter.
    """
    b=1-a
    outputA=inputE*a
    outputB=inputE*b
    
    return outputA,outputB


    
#-- Define function to plot amplitude and phase on same graph
def ampphaseplot(subplot,time,amp,time2,phase,plottitle,xlab,ylab,y2lab,l1,l2,l3,l4):
    subplot.plot(time[l1:l2],amp[l1:l2])
    phaseplot=subplot.twinx()
    phaseplot.plot(time2[l3:l4],phase[l3:l4],'r--')
    subplot.set(title=plottitle,xlabel=xlab,ylabel=ylab)
    phaseplot.set(ylabel=y2lab)
    phaseplot.yaxis.label.set_color('red')
    phaseplot.tick_params(axis='y',colors='red')

#-- Define function to find when amplitude drops below a given threshold
#   to be used for phase blanking and calculate FWHM
def bounds(amplitude,threshold):
    find= [1 if  i >=threshold else 0 for i in amplitude]
    lowerbound=find.index(1)
    upperbound=find.index(0,lowerbound+1)
    return lowerbound,upperbound

#-- Define function to reorder FT data for cleaner plotting with arrays
# In Python, frequency array comes out as [0,1,2,3,...+max,-max, -max+1,...,-1]
# which looks terrible when unwrapping phases and plotting.    
def reorder(array):
    sliceindex=int(len(array)/2)
    partA=array[:sliceindex]
    partB=array[sliceindex:]
    newarray=np.hstack((partB,partA))
    return newarray



def dispersion(inputE,time,carrierwl,material,distance):
    """
    Computes the full effects of dispersion on input field (as a function of time) propagating through a material of a given length. 'material' must be the name given in the  library for the coefficients.
    'inputE' is assumed to be complex.
    'carrierwl' is the carrier wavelength in nm.
    """
    c=2.99792458e8 # units: m/s
    FTinitpulse,f=otk.fouriert(True,inputE,time,carrierwl,applyshift=True)
    FTinitPhase=np.unwrap(np.angle(FTinitpulse))
    FTinitAmp=np.abs(FTinitpulse)
    wl=c/f*1e6 #units: um

#-- Calculate dispersion for 'material' using Sellmeier coefficients
    mat=sellmeier1(*material)
    n=mat.n(wl)
    k=2*np.pi*n*f/c # units 1/m
    midpoint=int(np.ceil(len(k)/2)) #midpoint is the index for the carrier frequency
    dispPhase=k*distance
    dtheta_dk=(dispPhase[midpoint-1]-dispPhase[midpoint+1])/(k[midpoint-1]-k[midpoint+1])
    modPhase=dispPhase-(k[midpoint]-dtheta_dk*k)*distance # set phase to zero at 620 nm and remove linear phase term

#-- Calculate inverse FFT to get dispersed pulse
    dispSpectrum=FTinitAmp*np.exp(1j*(FTinitPhase+modPhase))
    dispPulse=np.fft.ifft(fft.ifftshift(dispSpectrum))
    
    return outputE


class matquants: # calculate quantities based on material index for refraction
 
    def groupv(self,wavelength): # group velocity. Units: m/s
        c=2.992792458e8
        wl=wavelength
        n=self.n(wl)
        v=c/n/(1-wl/n*self.n_p(wl))
        return v
    
    def pftilt(self,deg_incid,wavelength): # calculates pulse front tilt for a given angle of incidence
        ang=deg_incid*np.pi/180
        tan_a=-wavelength*self.n_p(wavelength)/(2*np.pi)/(self.n(wavelength)-wavelength*self.n_p(wavelength)/(2*np.pi))*np.sin(ang)/np.sqrt(self.n(wavelength)**2-np.sin(ang)**2)
        a=np.arctan(tan_a)
        return a
    
    def gvd(self,wavelength): # group velocity dispersion. Units: fs^2/mm
        GVD=1e-3*wavelength**3/(2*np.pi*9e-8)*self.n_pp(wavelength)
        return GVD
    
    def beta3(self,wavelength): # 3rd order dispersion. Units: fs^3/mm
        wl=wavelength
        c=3e-4 # units: mm/fs
        b3=wl**2*1e-6*(-3*wl*self.n_pp(wl)-wl**3*self.n_ppp(wl))/(4*np.pi**2*c**3)       
        return b3
    
    def solfiswl(self,wavelength): # calculate wavelength of soliton fission signature peak
        w1=0.3/(0.3/wavelength-3*self.gvd(wavelength)/self.beta3(wavelength)/2/np.pi)
        w2=0.3/(0.3/wavelength+3*self.gvd(wavelength)/self.beta3(wavelength)/2/np.pi)
        return w1,w2
    
    def aslb(self,pumpwl): # calculate dispersion condition for anti-Stokes light bullet peak
        c=2.992792458e8
        def dispcond(lbwl): # dispersion condition to be solved
            x=pumpwl*self.n(lbwl)-lbwl*self.n(pumpwl)-c/self.groupv(pumpwl)*(pumpwl-lbwl)
            return x
        
        lbwl=spo.brentq(dispcond,10,.3)
        
        return lbwl

class sellmeier1(matquants): 
    """
    create Sellmeier Equation from coefficients of the form:
        n^2 =1 + b1 * wl^2 / ( wl^2 - c1 ) + b2 * wl^2 / ( wl^2 - c2 ) + b3 * wl^2 / ( wl^2 - c3 )
    """
    def __init__(self,b1,b2,b3,c1,c2,c3):
        self.coeffs=[b1,b2,b3,c1,c2,c3]
        return
    
    def n(self,wavelength): # calculate refractive index of given wavelength in microns
        wl=wavelength
        nsq=1+self.coeffs[0]*wl**2/(wl**2-self.coeffs[3])+self.coeffs[1]*wl**2/(wl**2-self.coeffs[4])+self.coeffs[2]*wl**2/(wl**2-self.coeffs[5])
        n=np.sqrt(nsq)
        return n
 
    def n_p(self,wavelength): # 1st derivtive of n w.r.t wavelength in microns
        wl=wavelength
        dndl=-1*wl/self.n(wl)*(self.coeffs[0]*self.coeffs[3]/(wl**2-self.coeffs[3])**2+self.coeffs[1]*self.coeffs[4]/(wl**2-self.coeffs[4])**2+self.coeffs[2]*self.coeffs[5]/(wl**2-self.coeffs[5])**2)
        return dndl
    
    def n_pp(self,wavelength): # 2nd derivtive of n w.r.t wavelength in microns
        wl=wavelength
        t=-1*wl*((self.coeffs[0]*self.coeffs[3])/(self.coeffs[3] - wl**2)**2+(self.coeffs[1]*self.coeffs[4])/(self.coeffs[4] - wl**2)**2+(self.coeffs[2]*self.coeffs[5])/(self.coeffs[5] - wl**2)**2)
        dt=t/wl-wl**2*((4*self.coeffs[0]*self.coeffs[3])/(self.coeffs[3] - wl**2)**3-(4*self.coeffs[1]*self.coeffs[4])/(self.coeffs[4] - wl**2)**3-(4*self.coeffs[2]*self.coeffs[5])/(self.coeffs[5] - wl**2)**3)
        deriv=(self.n(wl)*dt-t*self.n_p(wl))/self.n(wl)**2
        return deriv
    
    def n_ppp(self,wavelength): # 3rd derivative of n w.r.t. wavelength in microns
        wl=wavelength
        n=self.n(wavelength)
        t=-1*wl*((self.coeffs[0]*self.coeffs[3])/(self.coeffs[3] - wl**2)**2+(self.coeffs[1]*self.coeffs[4])/(self.coeffs[4] - wl**2)**2+(self.coeffs[2]*self.coeffs[5])/(self.coeffs[5] - wl**2)**2)
        dt=t/wl-wl**2*((4*self.coeffs[0]*self.coeffs[3])/(self.coeffs[3] - wl**2)**3-(4*self.coeffs[1]*self.coeffs[4])/(self.coeffs[4] - wl**2)**3-(4*self.coeffs[2]*self.coeffs[5])/(self.coeffs[5] - wl**2)**3)
        ddt=-12*wl*(((self.coeffs[0]*self.coeffs[3]*(self.coeffs[3] + wl**2))/(self.coeffs[3] - wl**2)**4) - (self.coeffs[1]*self.coeffs[4]*(self.coeffs[4] + wl**2))/(self.coeffs[4] - wl**2)**4 -(self.coeffs[2]*self.coeffs[5]*(self.coeffs[5] + wl**2))/(self.coeffs[5] - wl**2)**4)
        deriv=(n*(self.n_p(wl)*dt+n*ddt)-2*self.n_p(wl)*(n*dt+t*self.n_p(wl)))/n**3
        return deriv
    
class birefmat: # calculate index for refraction for a birefringent material
    def __init__(self,ordinary,extraordinary):
        self.coeffs_o=ordinary
        self.coeffs_e=extraordinary
        return
    
    def n_o(self,wavelength): # calculate ordinary refractive index
        wl=wavelength
        nsq=1+self.coeffs_o[0]*wl**2/(wl**2-self.coeffs_o[3])+self.coeffs_o[1]*wl**2/(wl**2-self.coeffs_o[4])+self.coeffs_o[2]*wl**2/(wl**2-self.coeffs_o[5])
        n=np.sqrt(nsq)
        return n
    
    def n_e(self,wavelength): # calculate extraordinary refractive index
        wl=wavelength
        nsq=1+self.coeffs_e[0]*wl**2/(wl**2-self.coeffs_e[3])+self.coeffs_e[1]*wl**2/(wl**2-self.coeffs_e[4])+self.coeffs_e[2]*wl**2/(wl**2-self.coeffs_e[5])
        n=np.sqrt(nsq)
        return n
    
    def n(self,rads,wavelength): # calculate index of refraction at angle w.r.t. optic axis
        wl=wavelength
        no=self.n_o(wl)
        ne=self.n_e(wl)
        n=neff(no,ne,rads)
        return n
    
    def gensmcoef(self,angle,wlrange): # generate Sellmeier coefficients for give angle w.r.t. optic axis
        nfit=self.n(angle,wlrange)
        smcoeffs,covar=spo.curve_fit(ofl.sellmeier,wlrange,nfit,p0=self.coeffs_o,maxfev=5000)
        return smcoeffs