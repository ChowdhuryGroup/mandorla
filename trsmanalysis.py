# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 17:45:47 2020
@author: Michael
"""
import labclerk as lck
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.interpolate as spi
import scipy.optimize as spo
import orchard as orc
import json
import opfunclib as ofl
import beamsizelib as bsl
import numpy.ma as ma

e=1.6021766208e-19 # units: Coulombs
epsilon_0=8.854187817e-12 # units: F/m
c=2.99792458e8 # units: m/s

def delaycalc(fname,zero_pos):
    """
    Calculate delay from file name based on convention 00cm_0p000in for manual rail and imperial micrometer. zero_pos is (45,0.007) for first set of measurements.
    
    Parameters:
    -----------    
    fname: string
        File name of TRSM image.
    
    zero_pos: tuple (float,float)
        Location of zero delay on translation stage and rail.
        
    Returns:
    ----------
    delay: float64
        Delay between pump and probe pulses.     
    """
    
    rail_delay=lck.fnameread(fname,'cm','J/') #read delay from file name
    mmeter_delay=lck.fnameread(fname,'in','m_') #read delay from file name
    
     # the following is based on the orientation of the rail and micrometer stage. May need to be changed in later setups.
    delaycm=2*(zero_pos[0]-rail_delay) #delay on rail
    delayin=2*(mmeter_delay-zero_pos[1]) #delay on imperial micrometer
    
    delay=(delaycm/100+delayin*0.0254)/c #units: seconds
    
    return delay

def cropim(image,reverse=False):
    """
    Crop an image. Returns image selection or everything else (if 'reverse'=True). 
    """
    g=orc.imshow(image)
    ul,lr=np.intc(g.fig.ginput(2),timeout=-1) #User clicks on upper-left and lower-right corners from  the image
       
    if reverse:
        mask=np.ones(image.shape)
        mask[ul[1]:lr[1],ul[0]:lr[0]]=0
    else:
        mask=np.zeros(image.shape)
        mask[ul[1]:lr[1],ul[0]:lr[0]]=1
    
    ravmask=np.where(mask.ravel()==1)
    plt.close('all')
    return ul,lr,ravmask[0]

def makepf2dterms(image,xorder,yorder,saveterms=False):
    rows,columns=image.shape
    x=np.float64(np.arange(columns))
    y=np.float64(np.arange(rows))
    X,Y=np.meshgrid(x,y)
    XX,YY=X.ravel(),Y.ravel()
    
    terms=[]
    for i in range(xorder+1):
        for j in range(yorder+1):
            terms.append(XX**i*YY**j)
    if saveterms:
        fn=lck.savedlg()
        xx=XX.tolist()
        yy=YY.tolist()
        t=[]
        for i in terms:
            t.append(i.tolist())
        j={'terms':t,'XX':xx,'YY':yy}
        with open(fn,'w') as file:
            json.dump(t,file)

    else:
        pass
            
    return terms,XX,YY
    
def polyfit2d(image,xorder,yorder,ravel_mask=None, import_terms=None):
    
    # take the image and create an array of XY coordinates. Then flatten arrays to input into fitting method.
    if np.any(import_terms==None):
        terms,XX,YY=makepf2dterms(image,xorder,yorder)
    else:
        terms,XX,YY=import_terms
    
    Afull=np.array(terms[:]).T
    grid=np.vstack((XX,YY))
    flatimage=image.ravel() #flatten image to input into fitting method
   
    if np.all(ravel_mask==None):
        fimage=flatimage
        A=Afull
    else:
        fimage=np.delete(flatimage,ravel_mask)
        cropterms=[]
        for i in terms:
            cropterms.append(np.delete(i,ravel_mask))
            A=np.array(cropterms[:]).T

    fitterms,_,_,_= sp.linalg.lstsq(A,fimage)
    
    fit=np.reshape(np.dot(Afull,fitterms),image.shape)
    
    
    return fitterms,fit,grid

def fftfilter(image,x_width,y_width):
    """
    Applies a rectangular filter to the Fourier transform passing through low frequencies at the center.
    """
    
    fimage=np.fft.fft2(image)
    filterim=fimage.copy()
    a=x_width
    b=y_width
    filterim[b:-b,a:-a]=0
    newimage=np.abs(np.fft.ifft2(filterim))
    
    return newimage
    
def trsmproc(fnames,zero_pos,precrop=True,save=True,loadterms=True):
    """
    Process TRSM images by calculating the delay from file name and performing probe uniformity correction.
    """
    images=[]
    masks=[]
    procparams=[]
    if precrop:
        print('Select pre-crop area.')
        pcim=orc.openim(fnames[0],False,True)
        pcul,pclr,pcmask=cropim(pcim,reverse=True)
        pcshape=np.flip(pclr-pcul)
        
    else:
        pass
    
    for f in fnames:
        if precrop:
            pcimg=orc.openim(f,False,True).ravel()
            img=np.reshape(np.delete(pcimg,pcmask),pcshape)
        else:
            img=orc.openim(f,False,True)
            
        images.append(img)
        _,_,mask=cropim(img)
        masks.append(mask)
    if loadterms: #this feature doesn't seem to work for some reason
        fn=lck.filedlg('Select JSON File')[0]
        with open(fn,'r') as f:
            t=json.load(f)
        tm=[]
        for i in t['terms']: 
            tm.append(np.array(i))
        XX=np.array(t['XX'])
        YY=np.array(t['YY'])
        imprt_t=(tm,XX,YY)
    else:    
        imprt_t=makepf2dterms(img,5,5)
        
    for i in range(len(images)):
       print(str(i))
       f=fnames[i]
       delay=delaycalc(fnames[i],zero_pos) 
       fitterms,fit,grid=polyfit2d(images[i],5,5,masks[i],imprt_t)
       cimg=images[i]-fit
       deltaToverT=cimg/fit # relative change in transmission is defined as (original image - fit)/fit
       print('Final crop for saving images.')
       ul2,lr2,finalcrop=cropim(cimg)
       finalshape=np.flip(lr2-ul2) # from 'ginput' function, column is first then row so order of array needs to be flipped.
       procimg=np.reshape(cimg.ravel()[finalcrop],finalshape)
       dTimage=np.reshape(deltaToverT.ravel()[finalcrop],finalshape)
       procparams.append([f,delay,fit,cimg,deltaToverT,finalcrop,procimg,dTimage])
       
       if save:
          sfn=lck.savedlg('Select folder and basename')
           
    return procparams

def trsmproc2(fnames,zero_pos,imshow=False):
    """
    Process TRSM images by calculating the delay from file name and subtract a background image to flatten background. Background file must be same name as TRSM image but end in '_0'
    """
    proc=[]

    for f in fnames:
        img=orc.openim(f,show_image=imshow)
        d=delaycalc(f,zero_pos) #calculate pump-probe delay
        
        bfn=f.replace('_1.bmp','_0.bmp') #find corresponding background image
        bimg=orc.openim(bfn,show_image=imshow)
        
        cimg=img.astype(np.int32)-bimg.astype(np.int32) #bitmaps are usually imported as a np.uint8 type array so we need to convert data type for image subtraction
        deltaToverT=cimg/bimg
        
        proc.append([f,d,img,bimg,cimg,deltaToverT])
    
    return proc

def testtrsmproc(images):
    """
    Test trsmproc algorithm on image that is not saved
    """
    procparams=[]    
    for i in range(len(images)):
       print(str(i))

       fitterms,fit,grid=polyfit2d(images[i],5,5)
       cimg=images[i]-fit
       deltaToverT=cimg/fit # relative change in transmission is defined as (original image - fit)/fit
       print('Final crop for saving images.')
       ul2,lr2,finalcrop=cropim(cimg)
       finalshape=np.flip(lr2-ul2) # from 'ginput' function, column is first then row so order of array needs to be flipped.
       procimg=np.reshape(cimg.ravel()[finalcrop],finalshape)
       dTimage=np.reshape(deltaToverT.ravel()[finalcrop],finalshape)
       procparams.append([fit,cimg,deltaToverT,finalcrop,procimg,dTimage])
           
    return procparams

def lineoutfit(x,y,show_graphs=True):
    """
    Take lineout and fit two Gaussians for a background subtraction.
    """
    area=np.sum(y)
    expectval=np.sum(x*y)/area
    def paratop(x,a,fwhm,center):
        
        return y
    
    def gausstop(x,a,fwhm,center):
        cutoff=np.sqrt(np.log(10)/4/np.log(2)*fwhm**2)
        y=np.piecewise(x,[np.abs(x-center)<=cutoff,np.abs(x-center)>cutoff],[lambda x: ofl.gaussian(x,a,fwhm,center)-a/10,0])
        return y
    def doubgauss(x,a0,fwhm0,center0,y_off,slope,a1,fwhm1,center1):
        y=y_off+slope*x+a0*np.exp(-4*np.log(2)*(x-center0)**2/fwhm0**2)+ofl.gaussian(x,a1,fwhm1,center1)
        #gausstop(x,a1,fwhm1,center1)
        return y
    p0norm=[120,500,expectval,0,-20/1200,-100,200,expectval]
    p0norm2=[120,500,expectval,0,20/1200,-100,200,expectval]
    p036ps=[120,500,800,0,20/1200,-50,150,700] #initial params for 36ps trace
    
    def sumsq_error(fit_parameters):
        return np.sum((y-doubgauss(x,*fit_parameters))**2)
#    try:
#        fit_parameters,covar=sp.optimize.curve_fit(doubgauss,x,y,p0=p0norm)
#    except RuntimeError:
#       try: fit_parameters,covar=sp.optimize.curve_fit(doubgauss,x,y,p0=p0norm2)
#       except RuntimeError:
#           try: fit_parameters,covar=sp.optimize.curve_fit(doubgauss,x,y,p0=p036ps)
#           except RuntimeError:
#               print('Error in processing image')
        
    bounds=[(0,np.max(y)),
            (0,np.max(x)/2),
            (np.min(x),np.max(x)),
            (0,0),
            ((np.min(y)-np.max(y))/(np.max(x)-np.min(x)),(np.max(y)-np.min(y))/(np.max(x)-np.min(x))),
            (-np.max(y),0),
            (0,200),
            (np.min(x),np.max(x))]   
    
    initial_values=sp.optimize.differential_evolution(sumsq_error,bounds).x
    print(initial_values)
    fit_parameters,covar=sp.optimize.curve_fit(doubgauss,x,y,p0=initial_values)    
    fit=doubgauss(x,*fit_parameters) #complete double Gaussian fit
    probe_fit=ofl.gaussian(x,*fit_parameters[0:4])+x*fit_parameters[4]
    processed_trace=y-probe_fit
    g=orc.graph('Pixels','','Counts','(a.u.)')
    g.addcurve('Raw Trace',x,y,'r')
    g.addcurve('Fit',x,fit,'b')
    g.addcurve('Probe Background',x,probe_fit,'g')
    g.addhline(0,'y')
    g.addcurve('Processed Trace',x,processed_trace,'k')
    g.addlegend('best')
    if show_graphs:
        pass
    else: 
        plt.close()
    
    return processed_trace,fit_parameters,probe_fit,g

def lineprocess(fnames,x_coord=1120): 
    """
    Extract vertical lineouts from TRSM images and fit background to two gaussians to perform background subtraction.
    
    Parameters:
    ----------
    fnames: list (string)
        File names of TRSM images
    x_coord=1120
        Default x-coordinate to take vertica lineout from.
    
    Returns:
    ----------
    dicts: dictionary
        A dictionary containing the file names, delays, fit parameters, processed and unprocessed traces, and relative change in transmission for TRSM images.
    """
    xs=[]
    ys=[]
    dicts=[]
    graphs=[]
    folder=lck.savedlg()
    for f in fnames:
        image=orc.openim(f,flip_y=True)
        delay=delaycalc(f,(45,0.007))*1e12 # units in picoseconds
        x_coord,_=np.intc(plt.ginput(1,timeout=-1))[0]
        lineout,x_axis=bsl.extractlineout(image,x_coord,x_coord,0,image.shape[0]-1,image.shape[0],20)
        xs.append(x_axis)
        ys.append(lineout)
        processed_trace,fit_parameters,probe_fit,g=lineoutfit(x_axis,lineout)
        deltaToverT=processed_trace/probe_fit
        ddict={'file':f,'delay':delay,'xcoord':x_coord,'probe':probe_fit,'ptrace':processed_trace,'rtrace':lineout,'deltaToverT':deltaToverT,'yaxis':x_axis,'fitparams':fit_parameters}
        
        delay_name=str(np.around(delay))+'ps' # delay in picoseconds
        delay_name=delay_name.replace('.','p')
        filename=folder+'_'+delay_name+'_lineout.png'
        g.fig.tight_layout()
        g.fig.savefig(filename)
        graphs.append(g)
        dicts.append(ddict)
        print(f)
    return dicts,graphs

def lineproc2(fnames,delay_cutoff,zero_pos=(45,0.007),x_coord=1120): 
    """
    Process lineout using fit from before or crater images depending on delay cutoff. Needs refinement.
    """
    xs=[]
    ys=[]
    fxs=[]
    fys=[]
    dicts=[]
    
    for f in fnames:
        im=orc.openim(f,flip_y=True)
        delay=delaycalc(f,zero_pos)*1e12 # units in picoseconds
        x_coord,_=np.intc(plt.ginput(1,timeout=-1))[0]
        line,dist=bsl.extractlineout(im,x_coord,x_coord,0,im.shape[0]-1,1000,5)
        xs.append(dist)
        ys.append(line)
        plo,fit,background,g=lineoutfit(dist,line,False)
        deltaToverT=plo/background
        ddict={'file':f,'delay':delay,'xcoord':x_coord,'fit':fit,'ptrace':plo,'rtrace':line,'deltaToverT':deltaToverT,'yaxis':dist}
        
        if delay<delay_cutoff:
            ffn=f.replace('_1.bmp','_0.bmp') # use before image
        else:
            ffn=f.replace('_1.bmp','_2.bmp') # use crater image
        final_image=orc.openim(ffn,flip_y=True,show_image=False)    
        fline,fdist=bsl.extractlineout(final_image,x_coord,x_coord,0,im.shape[0]-1,1000,5)
        fxs.append(fdist)
        fys.append(fline)
        fplo,ffit,fbackground,fg=lineoutfit(fdist,fline,False)   
        fdeltaToverT=fplo/fbackground
        corrected_deltaToverT=(deltaToverT-fdeltaToverT)/(fdeltaToverT+1)
        
        ddict['deltaToverT_corr']=corrected_deltaToverT
        g=orc.graph('Pixels','',str(delay))
        g.addcurve('deltaT',dist,deltaToverT,'r')
        g.addcurve('Processed Crater',dist,fplo,'b')
        g.addcurve('corr. deltaT',dist,corrected_deltaToverT,'k')
        g.addhline(0,'y')
        g.fig.tight_layout()
        g.addlegend('best')
        
#        bim=orc.openim(bf,show_image=False)
#        bline,bdist=bsl.extractlineout(bim,x_coord,x_coord,0,bim.shape[0]-1,1000,5)
#        bfit=ofl.gausslinefit(bdist,bline)
#        b=ofl.gaussian(bdist,*bfit[0:4])+bdist*bfit[4]
#        g=orc.graph()
#        g.addcurve('',bdist,bline,'r')
#        g.addcurve('',bdist,b,'b')
#        
#        sub=im.astype(np.int32)-bim.astype(np.int32)
#        sline,sdist=bsl.extractlineout(sub,x_coord,x_coord,0,bim.shape[0]-1,1000,5)
#        def background(x,a1,center1,a2,fwhm2,center2):
#            y=ofl.gaussian(x,-1*bfit[0],*bfit[1:3])+ofl.gaussian(x,a1,bfit[1],center1)+ofl.gaussian(x,a2,fwhm2,center2)
#            return y
#        icenter2=sdist[np.argmin(sline)]
#        icenter1=sdist[((icenter2+np.argmax(sline))/2).astype(np.int16)]
#        sfit,covar=sp.optimize.curve_fit(background,sdist,sline,p0=[bfit[0],icenter1,np.min(sline),200,icenter2])
#        bgfit=background(sdist,*sfit)
#        g2=orc.graph()
#        g2.addcurve('',sdist,sline,'r')
#        g2.addcurve('',sdist,bgfit,'b')
#        g2.addhline(0,'y')
#            
#        trace= sline-ofl.gaussian(sdist,-1*bfit[0],*bfit[1:3])-ofl.gaussian(sdist,sfit[0],bfit[1],sfit[1])
#        g2.addcurve('',sdist,trace,'k')
        
        
    dicts.append(ddict)
    
    return dicts

def extractfrommap(interp_lineout,lineout,xs,numpoints):
    """
    Extract points from a correlation of a lineout and an interpolated lineout from 'beamsizelib.matchlineout' method. xs are the points along the interp_lineout and numpoints is the number of points to be averaged around (must be odd). Returns points with standard deviation of x and y values.
    Assumes the lineouts are roughly Gaussian so a top and bottom side are calculated and at the maximum
    """
    top=[]
    bottom=[]
    etop=[]
    ebottom=[]
    ilo=interp_lineout
    lo=lineout
    num=numpoints//2
    imax=np.argmax(interp_lineout)

    
    for x in xs:
        idxt=lck.find_nearest(ilo[0:imax],x)[1]
        idxb=lck.find_nearest(ilo[imax:],x)[1]+imax
        
        toppts=(ilo[idxt-num:idxt+num+1],lo[idxt-num:idxt+num+1])
        botpts=(ilo[idxb-num:idxb+num+1],lo[idxb-num:idxb+num+1])
        toppoint=(np.mean(toppts[0]),np.mean(toppts[1]))
        etoppoint=(np.std(toppts[0]),np.std(toppts[1]))
        botpoint=(np.mean(botpts[0]),np.mean(botpts[1]))
        ebotpoint=(np.std(botpts[0]),np.std(botpts[1]))
        
        top.append(toppoint)
        etop.append(etoppoint)
        bottom.append(botpoint)
        ebottom.append(ebotpoint)
        
    points=(ilo[imax-num:imax+num+1],lo[imax-num:imax+num+1])
    maxpoint=(np.mean(points[0]),np.mean(points[1]))
    emaxpoint=(np.std(points[0]),np.std(points[1]))
    top.append(maxpoint)
    etop.append(emaxpoint)
    bottom.append(maxpoint)
    ebottom.append(emaxpoint)
    
    return top,bottom,etop,ebottom

def transvtime(mapping,delays):
    """
    Take a list of mappings and the corresponding delays and reorganize the data into lists for easy plotting of relative transmission vs time.
    """
    
    dtot_top=[]
    dtot_bottom=[]
    dtot_etop=[]
    dtot_ebottom=[]
    for f in range(len(mapping[0][0])): #iterate over fluences 
        dtot_top.append([m[0][f][1] for m in mapping])
        dtot_etop.append([m[2][f][1] for m in mapping])
        dtot_bottom.append([m[1][f][1] for m in mapping])
        dtot_ebottom.append([m[3][f][1] for m in mapping])
    
    dt=np.vstack([delays,dtot_top])
    det=np.vstack([delays,dtot_etop])
    db=np.vstack([delays,dtot_bottom])
    deb=np.vstack([delays,dtot_ebottom])
    
    #sort with respect to delays
    dtop=lck.dsort(dt.T,0)
    detop=lck.dsort(det.T,0)
    dbot=lck.dsort(db.T,0)
    debot=lck.dsort(deb.T,0)
    
    return dtop,detop,dbot,debot

def create_mapping(trsm_dictionarylist,trsm_scale,focalspot_lineout,focalspot_xaxis,focalspot_scale,fluences):
    """
    Description here.
    """
    folder=lck.savedlg()
    mapping=[]
    delays= [td['delay'] for td in trsm_dictionarylist]
    for td in trsm_dictionarylist:
        try:
            newfocalspot_lineout,diffCenter,diffFWHM,graph_matchup,graph_mapping=bsl.matchlineouts(focalspot_lineout*td['factor'],focalspot_xaxis*focalspot_scale,-1*td['ptrace'],td['yaxis']*trsm_scale,'Focal spot lineout','TRSM lineout',False)
        except KeyError:
            print('No fluence correction applied.')
            newfocalspot_lineout,diffCenter,diffFWHM,graph_matchup,graph_mapping=bsl.matchlineouts(focalspot_lineout,focalspot_xaxis*focalspot_scale,-1*td['ptrace'],td['yaxis']*trsm_scale,'Focal spot lineout','TRSM lineout',False)
            
        delay_name=str(np.around(td['delay']))+'ps' # delay in picoseconds
        delay_name=delay_name.replace('.','p')
        graph_matchup.save(folder+'Fspot-ptrace_matchup_'+delay_name+'.png')
        graph_mapping.save(folder+'Fspot-ptrace_mapping_'+delay_name+'.png')
        mapping.append(extractfrommap(newfocalspot_lineout,td['deltaToverT'],fluences,5))
    rel_trans=transvtime(mapping,delays)
           
    return mapping, rel_trans

def topbotavg(relative_transmission):
    """
    Calculate the average of the top and bottom relative transmission points and their corresponding errors.
    """
    rt=relative_transmission
    AvgRelativeTransmission=[np.mean([rt[0],rt[2]],axis=0),np.mean([rt[1],rt[3]],axis=0)]
    
    return AvgRelativeTransmission

def epsilon_r(n,ne,meff,omega,tau):
    er = n**2*(1-ne*e**2/(n**2*epsilon_0*meff*(omega**2+1j*omega/tau)))
    return er

def ne_lefthand(n,ne,meff,omega,tau,l):
    er = epsilon_r(n,ne,meff,omega,tau)
    al = np.sqrt(2)*omega*l/c*np.sqrt(np.abs(er)-np.real(er))
    return al

def ne_righthand(n,ne,meff,omega,tau,deltaToverT):
    er = epsilon_r(n,ne,meff,omega,tau)
    rs = np.log(1-np.abs((np.sqrt(er)-n)/(np.sqrt(er)+n))**2)-np.log(1+deltaToverT)
    return rs 

def solve(n,meff,omega,tau,l,deltaToverT):
    x0=n**2*epsilon_0*meff/e**2*omega**2
    def func(ne):
        return ne_lefthand(n,ne,meff,omega,tau,l)-ne_righthand(n,ne,meff,omega,tau,deltaToverT)
    
    root=spo.newton(func,x0,tol=1e15)
    return root

def electrondensity(meff,n,tau,wavelength,thickness,relative_transmission):
    """
    Calculate carrier concentration transmission data using Drude model.
    """
    me=9.109e-31 #units: kg
    epsilon_0=8.8541878128e-12 #units: F m^-1
    c=299792458 # units: m/s
    omega=2*np.pi*c/wavelength #units: Hz
    m_eff=meff*me
    e=1.60217662e-19 #units: C
    
    deltaToverT=relative_transmission[:,1:] #take all but first column where delays are kept
    x0=n**2*epsilon_0*m_eff/e**2*omega**2
    
    N=relative_transmission.copy()
    for i in range(deltaToverT.shape[0]):
        for j in range(deltaToverT.shape[1]):
           
            rel_trans=deltaToverT[i,j]
            def func(ne):
                return ne_lefthand(n,ne,m_eff,omega,tau,thickness)-ne_righthand(n,ne,meff,omega,tau,rel_trans)
    
            root=spo.newton(func,x0,tol=1e15)
            N[i,1+j]=root*1e-6 #units: cm^-3
    return N

def ratefit(time,y,y_inf=0):
    """
    Fitting with a single decay rate.
    """
    
    def expdecay1(x,tau,y0): # y at infinity is set to y_inf
        y=y_inf+(y0-y_inf)*np.exp(x/tau)
        return y
    
    def expdecay2(x,tau,y0,yinf):# y at infinity is nonzero
        y=yinf+(y0-yinf)*np.exp(x/tau)
        return y
    
    itau=-1*np.abs((time[-1]-time[0])/(np.log(np.abs(y[-1]))-np.log(np.abs(y[0]))))
    print('Initial decay rate is: '+str(itau))
    
    fit1,var1=sp.optimize.curve_fit(expdecay1,time,y,p0=[itau,y[0]],bounds=(-1*np.inf,[0,np.inf]))
    fit2,var2=sp.optimize.curve_fit(expdecay2,time,y,p0=[itau,y[0],y[-1]],bounds=(-1*np.inf,[0,np.inf,np.inf]))
    
    time2=np.linspace(time[0],time[-1],1000)
    g=orc.graph('Time','')
    g.addpoints('Extracted Data',time,y,'r')
    g.addcurve('Exp. Decay1',time2,expdecay1(time2,*fit1),'b')
    g.addcurve('Exp. Decay2',time2,expdecay2(time2,*fit2),'g')
    g.addlegend('best')
    
    return fit1,fit2,var1,var2,g

def doubleratefit(time,y,trans_time,y_inf=0):
    """
    Fitting with two decay rates.
    """
    
    def expdecay1(x,tau1,tau2,A,y0): # y at infinity is set to y_inf
        y=y_inf+(y0-y_inf)*(A*np.exp(x/tau1)+(1-A)*np.exp(x/tau2))
        return y
    
    def expdecay2(x,tau1,tau2,A,y0,yinf):# y at infinity is nonzero
        y=yinf+(y0-yinf)*(A*np.exp(x/tau1)+(1-A)*np.exp(x/tau2))
        return y
    
    _,idx=lck.find_nearest(time,trans_time) #find index for point closest to transition time
    
    itau1=-1*np.abs((time[idx]-time[0])/(np.log(np.abs(y[idx]))-np.log(np.abs(y[0]))))
    print('Initial Tau_1 is: '+str(itau1))
    itau2=-1*np.abs((time[-1]-time[idx])/(np.log(np.abs(y[-1]))-np.log(np.abs(y[idx]))))
    print('Initial Tau_2 is: '+str(itau2))
    
    fit1,var1=sp.optimize.curve_fit(expdecay1,time,y,p0=[itau1,itau2,0.5,y[0]],bounds=([-1*np.inf,-1*np.inf,0,-1*np.inf],[0,0,1,np.inf]))
    fit2,var2=sp.optimize.curve_fit(expdecay2,time,y,p0=[itau1,itau2,0.5,y[0],y[-1]],bounds=([-1*np.inf,-1*np.inf,0,-1*np.inf,-1*np.inf],[0,0,1,np.inf,np.inf]))
    
    time2=np.linspace(time[0],time[-1],1000)
    g=orc.graph('Time','(ns)','Carrier Conc.','(cm^{-3})')
    g.addpoints('Extracted Data',time,y,'r')
    g.addcurve('Exp. Decay1',time2,expdecay1(time2,*fit1),'b')
    g.addcurve('Exp. Decay2',time2,expdecay2(time2,*fit2),'g')
    g.addlegend('best')
    g.fig.tight_layout()
    
    return fit1,fit2,var1,var2

def doubleratefit2(time,y,trans_region,y_inf=0):
    """
    Fitting with two decay rates, modified algorithm that linearizes the fast time to apply fit parameters.
    """
   
    _,idx1=lck.find_nearest(time,trans_region[0]) #find index for point closest to transition time
    _,idx2=lck.find_nearest(time,trans_region[1])
    
    slope,const,errs=ofl.stdlinreg(time[0:idx2],ma.log(y[0:idx2]))
    tau1=1/slope
    y0=np.exp(const)
    err_tau1=tau1**2*errs
    print('Tau_1 is: '+str(tau1)+' Error: '+str(err_tau1))
    
    itau2=-1*np.abs((time[-1]-time[idx1])/(np.log(np.abs(y[-1]))-np.log(np.abs(y[idx1]))))
    #print('Initial Tau_2 is: '+str(itau2))
    
    slope2,const2,errs2=ofl.stdlinreg(time[idx1:],ma.log(y[idx1:]))
    tau2=1/slope2
    y02=np.exp(const2)
    err_tau2=tau2**2*errs2

    def expdecay(x,A,tau1,tau2): # y at infinity is set to y_inf
        y=y_inf+(y0-y_inf)*(A*np.exp(x/tau1)+(1-A)*np.exp(x/tau2))
        return y
    
    print('Tau_2 is: '+str(tau2)+' Error: '+str(err_tau2))
    
    try:
        A,var2=sp.optimize.curve_fit(expdecay,time,y,p0=[0.5,tau1,tau2],bounds=([0,-np.inf,-np.inf],[1,0,0]))
        err_A=np.sqrt(np.diag(var2))/len(y)
    except ValueError: 
        A=[0,tau2,tau2]
        err_A=[0,err_tau2,err_tau2]
    
    
    time2=np.linspace(time[0],time[-1],1000)
    g=orc.graph('Time','(ps)','Carrier Conc.','(cm^{-3})')
    g.addpoints('Extracted Data',time,y,'r')
    g.addcurve('Exp. Decay (Slow)',time2,y02*np.exp(time2/tau2),'b',linestyle='--')
    g.addcurve('Exp. Decay (Fast)',time2,y0*np.exp(time2/tau1),'g',linestyle='--')
    g.addcurve('Biexpontential',time2,expdecay(time2,*A),'k')

    
    g.addlegend('best')
    g.fig.tight_layout()
    
    return tau1,tau2,err_tau1,err_tau2,A[0],A[1],A[2],err_A[0],err_A[1],err_A[2]

def stretchedexp_fit(time,y):
    """
    Fitting with a skewed exponential curve.
    """
   
    def stretched_exponential(x,A,beta): # y at infinity is set to y_inf
        tau=time[-1]*(-1*np.log(y[-1]/A))**(-1/beta)
        fx=A*np.exp(-1*(x/tau)**beta)
        return fx

    fit,covar=spo.curve_fit(stretched_exponential,time,y,p0=[np.max(y),1],bounds=[(np.min(y)/2,0),(np.inf,1)])
    A=fit[0]
    beta=fit[1]
    tau=time[-1]*(-1*np.log(y[-1]/A))**(-1/beta)
    
    errs=np.sqrt(np.diag(covar))/len(time)
    
    time2=np.linspace(time[0],time[-1],1000)
    g=orc.graph('Time','(ps)','Carrier Conc.','(cm^{-3})')
    g.addpoints('Extracted Data',time,y,'r')
    g.addcurve('Stretched Exponential',time2,stretched_exponential(time2,A,beta),'b')
#    try:
#        g.addcurve('Both',time2,expdecay(time2,*A),'m')
#    except UnboundLocalError:
#        pass
    
    g.addlegend('best')
    g.fig.tight_layout()
    
    return (A,beta,tau),errs

def calc_rates(rel_tranmission,meff,n,tau,wavelength,thickness):
    """
    Description here.
    """
    fit_params=[]
    avgrelT= topbotavg(rel_tranmission)
    Ne=electrondensity(meff,n,tau,wavelength,thickness,avgrelT[0])
    for i in range(Ne.shape[1]-1):
        fit=doubleratefit2(avgrelT[0][1:,0],Ne[1:,i+1],(200,250))
        fit_params.append(fit)
    
    return Ne,fit_params

def cratermatch(pic_dict,interp_los):
    """
    Matchup the crater pictures to the interpolated fluence lineouts.
    """
    ilo=interp_los
    d=pic_dict
    for i in range(len(pic_dict)):
        craterfile=d[i]['file'].replace('_1.bmp','_2.bmp')
        x_coord=d[i]['xcoord']
        craterim=orc.openim(craterfile,show_image=False)
        clo,dist=bsl.extractlineout(craterim,x_coord,x_coord,0,craterim.shape[0]-1,craterim.shape[0],5)
        profit=ofl.gausslinefit(dist,clo)
        bgfit=ofl.gaussian(dist,*profit[0:4])+dist*profit[4]
        pclo=clo-bgfit
        scale=np.max(ilo[i])/np.max(pclo) #scaling for crater lineout
        g=orc.graph()
        g.addcurve('Crater',dist,np.flip(clo)*scale,'r')
        g.addcurve('',dist,np.flip(pclo)*scale,'g')
        g.addcurve('Pump',dist,ilo[i],'b')
    
    return craterfile,clo
