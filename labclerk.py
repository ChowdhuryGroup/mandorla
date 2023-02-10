# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 13:38:42 2020

@author: Michael
"""
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os
import json
import scipy.stats as sps
import orchard as orc


#-- Physical constants:
c=2.99e8
h=6.626e-34
k=1.38e-23

#----

def filedlg(title=''): 
    """
    Pulls up a dialog box to select multiple files. 'title' argument only labels the dialog box
    """
    
    root=tk.Tk()
    files=list(tk.filedialog.askopenfilenames(title='MANDORLA - '+title))
    root.destroy()
    return files

def savedlg():
    root=tk.Tk()
    fname=tk.filedialog.asksaveasfilename()
    root.destroy()
    return fname

def findheader(file,delim,maxrows,tries): 
    """
    Determines the header for a given file. A brute force method based on whether importing the file throws a ValueError in Numpy.loadtxt
    """
    
    head=0
    headerfound=False
    while head<tries:
        try:
            np.loadtxt(file,delimiter=delim,skiprows=head,max_rows=maxrows)
        except ValueError:
            head +=1
        else:
            headerfound=True
            break
    if headerfound==False:
        print('Header not found after '+str(tries)+' lines.')
    else:
       print('Header was found') 
    return head

def findfooter(file,delim,header,maxrows,tries): 
    """
    Determines the footer for a given file. A brute force method based on whether importing the file throws a ValueError in Numpy.loadtxt
    """
    
    foot=0
    footerfound=False
    while foot<tries:
        try:
            np.loadtxt(file,delimiter=delim,skiprows=header,max_rows=maxrows-foot)
        except ValueError:
            foot +=1
        else:
            footerfound=True
            break
    if footerfound==False:
        print('Footer not found after '+str(tries)+' lines.')
    else:
       print('Footer was found') 
    return foot

def flength(fname): 
    """
    Opens a file to determine the number of rows
    """
    
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i+1

def finddata(file,delim,tries): 
    """
    Determines file size, header, and footer of generic data file. Assumes footer is shorter than header.
    """
    
    totalrows=flength(file)
    header=findheader(file,delim,totalrows-tries,tries)
    maxrows=totalrows-header
    footer=findfooter(file,delim,header,maxrows,tries)
    return maxrows, header, footer

def loadfile(file,delim,tries,row_wise=False): 
    """
    Loads data from a file into a numpy array. This method automatically selects out the header and footer with the 'findheader' and 'findfooter' methods
    """
    mr,h,f=finddata(file,delim,tries)
    data=np.loadtxt(file,delimiter=delim,skiprows=h,max_rows=mr-f)
    if row_wise:
        data=data.T
    # loads the data in rows instead of columns instead of columns for easier manipulation
    else:
        pass    
    return data

def fnamerep(orig_txt='',corr_txt='',num=1):
    """
    Replace a text string in a file name.
    """
    fpaths=filedlg('Select Files to Rename')
    path=os.path.commonpath(fpaths)
    for fp in fpaths:
        fn=os.path.basename(fp)
        newname=fn.replace(orig_txt,corr_txt,num)
        newfp=path+'\\'+newname
        os.rename(fp,newfp)
    
    print('Files renamed to replace: \''+orig_txt+'\' with \''+corr_txt+'\'') 
    
    return

def simport(delim,tries,row_wise=False,correction=False,orig_txt='',corr_txt=''):
    """
    This method is for loading multiple files with a similar format (i.e. same delimiter and similae header/footer sizes). There's optional automatic correction of files using the 'filecorr' method.
    """
    
    files=filedlg('Select Files')
    data_list=[]
    for file in files:
        
        if correction:
            filecorr(file,orig_txt,corr_txt)
        else:
            pass
        
        data=loadfile(file,delim,tries,row_wise)
        data_list.append(data)
    return files,data_list

def filecorr(file,original_string,replacement_string):
    """
    Opens the file and overwrites original string with a replacement. This is for troubleshooting files that don't import properly.       
    """
    fin=open(file,'r')
    text=fin.read()
    corr_text=text.replace(original_string,replacement_string)
    fin.close()
    
    fin=open(file,'w')
    fin.write(corr_text)
    fin.close()
    return
    
def compress(simport_obj,columnindex,xdata_name=''):
    """
    Compresses data in objects imported from 'simport' method into a single numpy array with columnindex as the x-data. Currently, does not support data with different lengths.
    """
    n=columnindex
    so=simport_obj
    fnames=[xdata_name,*so[0]]
    xdata=so[1][0][:,n]
    arrayshape=(len(xdata),len(fnames))
    data=np.zeros(arrayshape)
    data[:,0]=xdata
    for i in range(len(fnames)-1): # fill in ydata
        data[:,i+1]=so[1][i][:,n+1]
        
    new_obj=(fnames,data)
    
    return new_obj
 
def mavg(data,n):
    """
    Calculates the moving average of 'n' data points.
    """
    ret = np.cumsum(data) # ret is just a stand-in array
    ret[n:] = ret[n:] - ret[:-n]
    avg=ret[n-1:]/n 
    return avg   

def dsort(array,column):
    """
    Sort array by a particular column. Presumes increasing order.
    """
    a=np.array(array)
    narray=array[a[:,column].argsort()]
    return narray
    

def polyfit(x,y,degree):
    """
    New alias for numpy.polynomial.polynomial.Polynomial.fit
    """
    output=np.polynomial.polynomial.Polynomial.fit(x,y,degree)
    return output

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx


def offset(data,minindex,maxindex):
    """
    Originally for spectra, this calculates the offset of data based on interval where there is no known signal.
    """
    offset=np.mean(data[minindex:maxindex])
    return offset

def fnameread(fname,unit_string,prefix=''):
    """
    Reads a filename for a number specfied by the 'unit_string'. Convention assumes that number is of the form: [prefix]000p000uu_ where 'uu' is the unit string, 'p' is the decimal point, and there are an arbitrary number of digits before or after the decimal point.
    """
    uindex=fname.find(unit_string+'_') #find end of the number
    
    if prefix=='': #sometimes the number you're looking for is at the very beginning so the prefix may be blank.
        lindex=0
    else:
        lindex=fname.find(prefix)+len(prefix)
    
    num_str=fname[lindex:uindex] #pick out number string
    num_str=num_str.replace('p','.') # replace 'p' with decimal point
    num=np.float64(num_str) # convert to float number
    
    return num

def oobgsub(data,data_inttime,bg,bg_inttime,offset_minindex,offset_maxindex):
    """
    Perform background subtraction algorithm that was found to be effective for Ocean Optics spectrometers. Signal scales linearly with integration time but background doesn't.
    """
    doffset=offset(data/data_inttime,offset_minindex,offset_maxindex)
    bgoffset=offset(bg/bg_inttime,offset_minindex,offset_maxindex)
    
    cdata=data/data_inttime-bg/bg_inttime/bgoffset*doffset

    return cdata
    
def blackbodcal(data,blackbodyspec,wavelengths,temperature):
    """
    Performs sensitivity calibration assuming that the data and blackbody spectra are already background subtracted. 
    """
    wl=wavelengths
    T=temperature
    
    bbcurve=2*h*c**2/(wl**5*(np.exp((h*c)/(wl*k*T))-1))
    cdata=data*bbcurve/blackbodyspec
    return cdata

def applycal(data,d_wl,calibration,cal_wl,division=True):
    """
    Divide generic calibration data (e.g. transmittance) to a data set. Separate wavelength arrays are used to apply an interpolation in case the data and calibration file are taken at different wavelengths. Calibration will be interpolated linearly onto data.
    """
    samewls=np.array_equal(d_wl,cal_wl)
        
    if samewls==True:
        calib=calibration
    else:
        print('Interpolating calibration data')
        calib=np.interp(d_wl,cal_wl,calibration)
        
    if division:
        cdata=data/calib
    else:
        cdata=data*calib
    
    return cdata
    
def normalize(data,minrange=0,maxrange=None): 
    """
    Normalizes data to max of a particular range. Behaviors and lifestyles are not supported data types at this time.
    """
    
    if maxrange==None:
        N=np.amax(data[minrange:len(data)])
    else:     
        N=np.amax(data[minrange:maxrange])
   
    normdata=data.copy()/N
    normdata[abs(normdata)>5]=0 # remove areas that have unrealistic signal due to low sensitivity.
    return normdata

def wlselect(wavelength_array,lowerwl,upperwl):
    """
    Given an array for the wavelengths. Find the indices that correspond to the lower and upper wavelength range.
    """
    wl=np.array(wavelength_array)
    lindex=np.max(np.where(wl<=lowerwl))
    uindex=np.min(np.where(wl>=upperwl))
    return lindex,uindex
    

def specmode(data2d,wavelength_array,lowerwl,upperwl):
    """
    Plot 2D modes of a given spectral range. Based on the nested listing format used by Andy.
    """
    l,u=wlselect(wavelength_array,lowerwl,upperwl)
    d=data2d
    intspec=np.zeros((len(d),len(d[0]))) # array for integrated spectrum
    for i in range(len(d)):
        for j in range(len(d[i])):
           intspec[i,j]=np.sum(d[i][j][l:u])
    return intspec

def oobgsub2d(data,data_inttime,bg,bg_inttime,offset_minindex,offset_maxindex):
    bg=np.array(bg)
    cdata=[]
    for i in range(len(data)):
        subcdata=[]
        for j in range(len(data[i])):
            d=np.array(data[i][j])
            cd=oobgsub(d,data_inttime,bg,bg_inttime,offset_minindex,offset_maxindex)
            subcdata.append(cd)
            
        cdata.append(subcdata)
        
    return cdata

def apply2d(data,function,*args):
    """
    Take an existing function and apply it to a 2D set of nested lists.
    """
    cdata=[]
    for i in range(len(data)):
        subcdata=[]
        for j in range(len(data[i])):
            d=np.array(data[i][j])
            cd=function(d,*args)
            subcdata.append(cd)
            
        cdata.append(subcdata)
        
    return cdata

def applycal2d(data,d_wl,calibration,cal_wl,division=True):
    cd=apply2d(data,applycal,d_wl,calibration,cal_wl,division)
    return cd

def totspec(data):
    d=np.array(data)
    tsum=np.sum(np.sum(d,axis=0),axis=0)    
    return tsum

def jsave(keys,data):
    """
    For large datasets. Use JSON to save multiple data sets with specified keys by serializing a dictionary.
    """
    jdict={}
    for i in range(len(keys)):
        jdict[keys[i]]=data[i]
    fname=savedlg()
    with open(fname+'.csv','w') as file:
        json.dump(jdict,file)
        
    print('Data saved')
    return

class Spectra:
    """
    A general spectrum class based on numpy arrays. It's intended to be a workbook to consolidate all the relevant info for a particular spectra dataset in one file.
It stores raw & processed spectra from a parameter scan (e.g. power or angle) along with calibrations that can be loaded and applied to the data. Also, it can hold prepended metadata and has methods to generate heatmaps and 2D scatterplots.
There are 2 subclasses for spectra collected from an imaging spectrometer and a linear spectrometer.

    """
    
    def __init__(self,filename):
        self.filename=filename
        self.prodata=[] # holds processed data
        self.bg=[]
        self.calibfiles=[] # holds file names for calibrations based on index
        self.calibs=[]
        self.appcals=[] # keeps track of which calibrations have been applied to processed data
        self.calINT=[]
        self.metadata=[]
        return
    
    def seldata (self): # select data files
        self.files=filedlg('Select Files')
        return
    
    def setcolumns(self,colentry): # set column values (e.g. pulse energies) in data file
        
        return
        
    def mrawdata (self,delim,inttimes,bgmin,bgmax,makeprodata=True): # make raw dataset
        nfiles=len(self.files)
        data=loadfile(self.files[0],delim,30)
        self.rawdata=np.zeros((nfiles,len(data[0])))
        self.wl=data[0] # wavelengths stored separately
        self.range=[bgmin,bgmax]
        self.doffsets=[]
        for i in range(nfiles):
            data=loadfile(self.files[i],delim,30)
            self.rawdata[i]=data[1]
            self.doffsets.append(offset(data[1],self.range[0],self.range[1]))
            
        self.dataINT=inttimes # make sure that the indices matc
        self.prodata=np.copy(self.rawdata)       
        return

    def rsprodata(self): # resets prodata to rawdata and clears applied calibration list
        self.prodata=np.copy(self.rawdata)
        self.appcals=[]
        print('Prodata reset')
        return
  
    def selcalib(self): # select a calibration file to the Spectra class
        self.calibfiles.append(filedlg('Select Calibration'))
        index=len(self.calibfiles)-1
        return self.calibfiles[index], index    
    
    def applycalib(self,index,delim,inttime=1,applyoffset=True): # apply calibration. Currently, assumes calibration and data have same indexing
        self.calINT.append(inttime)
        caldata=loadfile(self.calibfiles[index],delim,30)
        samewls=np.array_equal(self.wl,caldata[0])
        
        if samewls==True:
            calib=caldata[1]
        else:
            print('Interpolating calibration data')
            calib=np.interp(self.wl,caldata[0],caldata[1])
            
        self.calibs.append(calib)
        caloffset=offset(calib,self.range[0],self.range[1])
        modcal=(calib-self.bg*caloffset/self.bgoffset)/inttime   
        
        if applyoffset==False:
            self.prodata /= calib
            self.appcals.append('Cal'+str(index))
        else:
            self.prodata /= modcal
            
        self.prodata=self.prodata/calib
        self.appcals.append('modCal'+str(index))
        return

    def applybbcal(self,index,delim,inttime,temperature): # apply intensity calibration based on measured blackbody curve
        T=temperature
        wl=self.wl*1e-9
        self.bbcurve=2*h*c**2/(wl**5*(np.exp((h*c)/(wl*k*T))-1))
        self.applycalib(index,delim,inttime)
        self.appcals.append('Blackbody')        
        return 
    
    def selbg(self): #select a background file to the Spectra class
        self.bgfile=filedlg('Select Background')[0]
        return
    
    def applybg(self,delim,inttime,applyoffset=True):
        nfiles=len(self.files)
        self.bg=loadfile(self.bgfile,delim,30)[1]
        self.bgINT=inttime
        self.bgoffset=offset(self.bg,self.range[0],self.range[1])
        
        if applyoffset==False:
            self.prodata=self.rawdata-self.bg
            self.appcals.append('BG sub')
        else:
            for i in range(nfiles):
                self.prodata[i]=(self.rawdata[i]-self.bg*self.doffsets[i]/self.bgoffset)/self.dataINT[i]   
            self.appcals.append('BG sub. w. offset')
        return
    
    def notes (self): # pulls up notes for Spectra class
        
        return
    
    def metadata(self): # collect metadata to prepend to spectrum file
        
        return

    def save(self): # save entire Spectra workbook
        file=pd.concat([self.bg,self.calibs,self.rawdata,self.prodata],axis=1)
        folder=1
        file.to_csv(folder+self.filename)
        return
 
    
    
class imgSpectra(Spectra): # subclass for spectra obtained by imaging spectrometer
    def __init__(self, filename):
        super().__init__(filename)
        return
    
    def opensfmov(file):
        f = open(file, encoding="mbcs")
        content = f.read()
        width = int(content.split()
            [content.split().index('xPixls')+1])
        height = int(content.split()
            [content.split().index('yPixls')+1])
        frames_claimed = int(content.split()
            [content.split().index('NumDPs')+1])
        f.seek(content.index('DATA')+75, os.SEEK_SET)
        del content
        image = np.fromfile(f, dtype=np.uint16)
        f.close();
        image = np.reshape(image, (-1, height, width)).astype(float).sum(axis=0)
        return image
    
    def lineoutarea(self, sampledata): #select region of image to extract spectral lineout 
        
        return
    #you will want a function to select the part of the image with the spectrum and average over it and save this to a csv or text file FOR EACH IMAGE
    
    
    
class lineSpectra(Spectra): # subclass for spectra obtained by linear array (e.g. Ocean Optics)     
    def __init__(self, filename):
        super().__init__(filename)
        return
