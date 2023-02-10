# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:28:18 2020
@author: Michael
"""

import numpy as np
import scipy.ndimage
import scipy.optimize
import scipy.special as scs
import matplotlib.pyplot as plt
import tkinter as tk
from labclerk import filedlg
import orchard as orc
import opfunclib as ofl
import scipy.interpolate as spi
import re


def getval (data,position):
    search= [1 if  i >=position else 0 for i in data[0,:]]
    upperbound=search.index(1)
    value=(data[1,upperbound]+data[1,upperbound-1])/2
    return value

def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.
    Format specification: http://netpbm.sourceforge.net/doc/pgm.html
    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))



def diodetrace(position,intensity):
    modpos=position-np.mean(position)
    def trace(x,A,FWHM):
        y=(1/(2*np.sqrt(1/FWHM**2)*np.sqrt(np.log(2))))*A*(0.8862269254527579*scs.erf(2*np.sqrt(1/FWHM**2)*(0.4999999995 - x)*np.sqrt(np.log(2))) + 0.8862269254527579*        scs.erf(2*np.sqrt(1/FWHM**2)*(0.5000000005 + x)*np.sqrt(np.log(2))))
        return y
    imax=np.amax(intensity)
    fit,covar=scipy.optimize.curve_fit(trace,modpos,intensity,p0=[imax,1])
    return fit
def uline(image):
    """
    Brings up image and user selects two points to define a line across. Returns coordinates of two points.
    """
    orc.imshow(image)
    pts=np.intc(plt.ginput(2,timeout=-1))
    plt.close()
    return pts
      
def extractlineout(image,x0,x1,y0,y1,num,width): # provides lineout of image with averaging over specified width
        if x1==x0:
            dx=width/2
            dy=0
        elif y1==y0:
            dx=0
            dy=width/2
        else:
            angle=np.arctan((y1-y0)/(x1-x0))
            dx=width/2*np.sin(angle)
            dy=width/2*np.cos(angle)
        x0s=np.linspace(x0-dx,x0+dx,width)
        x1s=np.linspace(x1-dx,x1+dx,width)
        y0s=np.linspace(y0-dy,y0+dy,width)
        y1s=np.linspace(y1-dy,y1+dy,width)
        length=np.sqrt((x1-x0)**2+(y1-y0)**2)
        dist=np.linspace(0,length,num)
        line=np.zeros(num)
        for i in range(width):
            x, y = np.linspace(x0s[i], x1s[i], num), np.linspace(y0s[i], y1s[i], num)   
            line = line+scipy.ndimage.map_coordinates(image, np.vstack((y,x)))/width # Extract the values along the line, using cubic interpolation. x & y flipped becuase of image format
        #data=np.vstack((dist,line))
        plt.close()
        return line,dist

def lineout(image,num,width):
    pt0,pt1=uline(image)
    x0,y0=pt0
    x1,y1=pt1
    line,dist=extractlineout(image,x0,x1,y0,y1,num,width)
    return line,dist,pt0,pt1
  
def integim(image):
    maxind=np.argmax(image[1:,1:]) # gives the flattened index of maximum ignoring the first row and column due to camera aberrations
    cols=np.size(image,axis=1) # find number of columns
    rows=np.size(image,axis=0)
    maxrow=maxind//(cols-1)+1
    maxcol=maxind%(cols-1)+1
    maxind=(maxrow,maxcol)
    
    rmax=np.min([maxrow,maxcol,rows-maxrow,cols-maxcol])
    radii=range(rmax)
    sums=[]
    modderiv=[]
    for r in radii:
        newsum=np.sum(image[maxrow-r:maxrow+r+1,maxcol-r:maxcol+r+1])
        sums.append(newsum)
        
        if r==0:
            modderiv.append(newsum)
        else:
            modderiv.append((sums[r]-sums[r-1])/(4*r))
         
    return maxind,np.array(sums),np.array(modderiv)   

def fluencefactor(image,threshold=5):
    """
    Calculates the fluence factor for focal spot images.
    """
    maxind,sums,deriv=integim(image)
    crop_radius=np.min(np.where(deriv<0.1))
    crop=image.copy()[maxind[0]-crop_radius:maxind[0]+crop_radius,maxind[1]-crop_radius:maxind[1]+crop_radius]
    
    orc.graph()
    plt.imshow(crop)
    
    crop[crop<np.max(image)*np.exp(-1*threshold)]=0
    
    f=np.max(image[1:,1:])/np.sum(crop)
    
    return f

def localfluence(image,pulseEnergy,pixelArea):
    """
    Take an image from a focal spot image and convert it from pixel value to local fluence (in J/cm^2). Originally used for TRSM analysis.
    """
    ffactor=fluencefactor(image)
    maxval=np.max(image[1:,1:])
    locflu=image*ffactor/maxval*pulseEnergy/pixelArea*1e-4
    return locflu

def matchlineouts(lineout1,x1,lineout2,x2,name1='lineout1',name2='lineout2',display_graphs=True):
    """
    Scale and shift 'lineout2' to match 'lineout1'. Originally used for TRSM analysis.
    This method fits the two lineouts to Gaussians then compares amplitudes and centers. Make sure x1 and x2 are on same scale.
    Next feature is to correlate the values of lineout 1 to lineout 2.
    """
    gfit1=ofl.gaussfit(x1,lineout1)
    gfit2=ofl.gaussfit(x2,lineout2)
    diffs=gfit1-gfit2
    diffCenter=diffs[2]
    diffFWHM=diffs[1]
    
    newx2=x2+diffCenter
    g=orc.graph('x','',name1)
    g.addcurve(name1,x1,lineout1,'r')
    g.addcurve(name2,newx2,lineout2,'b',diffscale=True)
    g.addlegend('best')
    g.fig.tight_layout()
    
    lo1=spi.interp1d(x1,lineout1)
    interp_lo1=lo1(newx2)
    imax=np.argmax(interp_lo1)
    g2=orc.graph(name1,'',name2)
    g2.addpoints('Top side',interp_lo1[0:imax],lineout2[0:imax],'r',marksize=2)
    g2.addpoints('Bottom side',interp_lo1[imax:],lineout2[imax:],'b',marksize=2)
    g2.addlegend('best')
    g2.fig.tight_layout()
    
    if display_graphs:
        pass
    else:
        plt.close()
        plt.close()
    
    return interp_lo1,diffCenter,diffFWHM,g,g2

    
class calibration:
    def __init__(self):
        self.cfiles=filedlg('Select Calibration Images')
        self.imgs=[]
        for im in self.cfiles:
            img=read_pgm(im)
            self.imgs.append(img)
            lineout(img,)
        
        return
       
    def plotline ():
        fig, axes = plt.subplots(nrows=2,figsize=(10,8))
        axes[0].imshow(image)
        axes[0].plot([x0, x1], [y0, y1], 'ro-')
        axes[0].axis('image')
        
        axes[1].axvline(x=expectval, color='r')
        axes[1].axvline(x=expectval-stdev,color='g')
        axes[1].axvline(x=expectval+stdev,color='g')
        axes[1].axvline(x=expectval-fwhm/2,color='m')
        axes[1].axvline(x=expectval+fwhm/2,color='m')
        axes[1].plot(dist,zi)
        axes[1].plot(dist,gauss)
        axes[1].plot(dist,gaussian(dist,*fit[0]))
    







