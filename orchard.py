# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 17:24:44 2020
@author: Michael
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.interpolate as spip
import numpy as np
import pandas as pd

# set plot styling
def setstyle(): 
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['lines.markersize'] = 10
    mpl.rcParams['axes.linewidth'] = 2
    mpl.rcParams['axes.labelsize'] = 20
    mpl.rcParams['xtick.labelsize'] = 20
    mpl.rcParams['xtick.major.size'] = 10
    mpl.rcParams['xtick.major.width'] = 2
    mpl.rcParams['ytick.labelsize'] = 16
    mpl.rcParams['ytick.major.size'] = 10
    mpl.rcParams['ytick.major.width'] = 2
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'DejaVu Sans'
    mpl.rcParams['xtick.minor.size'] = 5
    mpl.rcParams['xtick.minor.width'] = 2
        
    mpl.rcParams['legend.fontsize'] = 16
    mpl.rcParams['legend.fancybox'] = False
    return

def rehydrate(dryfig): # opens a saved figure to further edit. Rehydrating a '.dryfig'! ;)
    df=pd.read_csv(dryfig,sep='\t')
    plottype=df[0][0] # I hope to used this later to specify how to unpack things
    gprops=df[0][1]
    
    g=graph()
    nlines=len(df.columns)
    for i in range(nlines):
        lprops=df[i][2]
        x=df[i][3]    
        g.sbplt.plot()
    
    g.sbplt.set(**gprops[0])
    
    
    
    return

def imshow(matrix,axes=True):
    """
    A function to quickly display an image in a separate figure using the existing 'graph' class.
    """
    g=graph()
    plt.imshow(matrix)
    if axes:
        pass
    else:
        plt.xticks([])
        plt.yticks([])
    
    return g   

def openim(fname,flip_x=False,flip_y=False,show_image=True):
    """"
    A wrapper to open an image and flip the x and/or y axis as needed. Image 
    """
    image=plt.imread(fname)
    
    if flip_x:
        image=np.flip(image,axis=1)
    else:
        pass
    
    if flip_y:
        image=np.flip(image,axis=0)
    else:
        pass
    if show_image:
        imshow(image)
    else: pass

    return image



class graph:
    def __init__(self,xlabel='',xunits='',ylabel='',yunits='',ptitle='',size=(8,5)):
        setstyle()
        plt.ion()
        self.fig, self.sbplt = plt.subplots(1,figsize=size)
        self.sbplt.set_xlabel(xlabel+' $\mathregular{'+xunits+'}$')
        self.sbplt.set_ylabel(ylabel+' $\mathregular{'+yunits+'}$')
        self.sbplt.set_title(ptitle)
        
        self.sbplt.axes.tick_params(direction='in',which='both')
       
        return     
    
    def addpoints(self,name,x,y,color,lines=False,line_color='k',fill_style='full',marksize=10):
        self.sbplt.plot(x,y,ls='None',marker='o',markeredgewidth=2,c=color,label=name,fillstyle=fill_style,markersize=marksize,zorder=3) 
        if lines==False:
            pass
        else: 
            xlim=self.sbplt.get_xlim()
            ylim=self.sbplt.get_ylim()
            self.sbplt.vlines(x,ylim[0],y,colors=line_color,linestyles='dashed',zorder=1)
            self.sbplt.hlines(y,xlim[0],x,colors=line_color,linestyles='dashed',zorder=1)
             
        return
    
    def addebars(self,name,x,y,color,y_err=None,x_err=None):
        self.sbplt.errorbar(x,y,y_err,x_err,c=color,label=name,fmt=color+'o',ecolor=color,capsize=5,capthick=2)
        return
    
    def addcurve(self,name,x,y,color,diffscale=False,**kwargs):
        if diffscale==False:
            self.sbplt.plot(x,y,c=color,label=name,zorder=2,**kwargs)
        else:
            self.sbplt.plot([],[],c=color,label=name,zorder=2,**kwargs) #quick and dirty but not proper way of doing things
            ncurve=self.sbplt.twinx()
            ncurve.plot(x,y,c=color,**kwargs)
            ncurve.set(ylabel=name)
            #ncurve.yaxis.label.set_color(color)
            ncurve.tick_params(axis='y',direction='in')
        return
    
    def maddcurve(self,names,xdata,ydata,colormap,rowwise=True):
        """
        Add multiple plots of the same type.
        """
        print(colormap)
        if rowwise:
            pass
        else:
            ydata=np.transpose(ydata)
            
        dsize=len(ydata)
        
        if len(names)==dsize:
            pass
        else:
            names=[names]*dsize
            
        if len(xdata)==dsize:
            pass
        else:
            xdata=[xdata]*dsize
        
        cindex=np.linspace(0,1,dsize)
        cmap=mpl.cm.get_cmap(colormap)
        colors=cmap(cindex)
        for i in range(dsize):
           self.addcurve(names[i],xdata[i],ydata[i],colors[i]) 
       
        return
        
    
    def addheatmap(self,zlabel,z,x=[],y=[],resx=1,resy=1,interp='linear',colorscale='viridis'): # create a heatmap with interpolation of data
        if len(x)==0:
            x=np.arange(np.size(z,axis=1))
        else:
            pass
        if len(y)==0:
            y=np.arange(np.size(z,axis=0))
        else:
            pass
        
        self.sbplt.axes.tick_params(direction='out')
        
        interpz=spip.interp2d(x,y,z,kind=interp) # interpolate data
        newx=np.linspace(np.min(x),np.max(x),int(x.size*resx)) # resample x & y coordinates
        newy=np.linspace(np.min(y),np.max(y),int(y.size*resy))
        dmap=interpz(newx,newy)
        X,Y=np.meshgrid(newx,newy)
        heatmap=self.sbplt.pcolormesh(X,Y,dmap, cmap=colorscale)
        cb=self.fig.colorbar(heatmap,ax=self.sbplt)
        cb.ax.set_title(zlabel,fontsize=20)
        return
    
    def fillarea(self,x,centerline,curve1,curve2,color,altcolor=''):
        """
        Fill section of plot between curve1 and curve2 with centerline.
        """
        if altcolor=='':
            altcolor=color
        else:
            pass
        
        self.sbplt.fill_between(x,curve1,curve2,color=altcolor,alpha=0.2)
        self.addcurve('centerline',x,centerline,color)
        
        return
    
    def addhline(self,yval,color):
        self.sbplt.axhline(y=yval,c=color)
        return
    
    def addvline(self,xval,color):
        self.sbplt.axvline(x=xval,c=color)
        return
    
    def adjy(self,ymin,ymax,rescalex=True): # adjust y-axis range
        self.sbplt.set_ylim(bottom=ymin,top=ymax)
        self.sbplt.set_xlim(auto=rescalex)
        return
    
    def adjx(self,xmin,xmax,rescaley=True): # adjust x-axis range
        self.sbplt.set_xlim(left=xmin,right=xmax)
        self.sbplt.set_ylim(auto=rescaley)
        return
    
    def addlegend(self,location): # adds a legend at the location on the plot
        self.sbplt.legend(loc=location)
        return
    
    def pulldata(self): # pulls data from each subplot line
        nlines=len(self.sbplt.lines)
        gdata=[]
        for i in range(nlines):
            gdata.append(self.sbplt.lines[i].get_xydata())
                
        return gdata
    
    def pullparams(self): # pulls style parameters from each subplot line
        l=self.sbplt.lines
        nlines=len(l)
        gparams=dict()
        gpropkeys=['title','xlabel','xlim','xscale','xticks','ylabel','ylim','yscale','yticks']
        
        gprops=self.sbplt.properties()
        for prop in gpropkeys:
            gparams[prop]=gprops[prop]
        
        glparams=[]
        glpropkeys=['color','fillstyle','label','linestyle','linewidth','marker','markeredgecolor','markeredgecolor','markeredgewidth','markerfacecolor','markerfacecoloralt','markersize','zorder']
        
        for i in range(nlines):
            props=l[i].properties()
            tempdict=dict()
            for prop in glpropkeys:
                tempdict[prop]=props[prop]
                
            glparams.append(tempdict)
                
        return gparams,glparams
    
    def save (self,filename):
        self.fig.savefig(filename,bbox_inches='tight')
        print('Figure saved.')
        return 
    
    def harvest(self,filename,plottype): # store important figure properties and data in a csv format using Pandas DataFrame. File extension is .dryfig 'cause why not.
        
        data=self.pulldata()
        gprops,dprops=self.pullparams()
        pltype=[plottype,]
        df=pd.DataFrame((pltype,gprops,dprops,data))
        df.to_csv(filename+'.dryfig',sep='\t',header=False,index=False)
        return