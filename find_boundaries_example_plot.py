#SJL 2/17
#Script to find the CoRoL absed on extrapolating from corotating runs

###########################################################
###########################################################
###########################################################
import numpy as np
import scipy as sp
from scipy import constants as const
import sys
import os
import fileinput

#package to use wildcards 
import fnmatch

import csv

#plotting tools
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab
import matplotlib.gridspec as gridspec

#HERCULES_structures
sys.path.insert(0, '/n/home05/slock/Code_repository/HERCULES/HERCULES_v1.0/Analysis_scripts')
from HERCULES_structures import *

#import colormaps
sys.path.insert(0, '/n/gstore/Labs/Stewart_Lab/slock/lunar_disk/HSSL_impact_comparison/')
import colormaps as cmaps

###########################################################
###########################################################
###########################################################
#define some general functions
#Function to calculate the numerical gradient of unevenly spaced points to 2nd order
def gradient2(x,y):
    #define the difference between all the points
    dx=np.diff(x)
    #define the output array
    dydx=np.empty(np.size(y))
    #use centered difference for the middle points
    dydx[1:-1]=(y[2:]*(dx[0:-1]**2.0)-y[0:-2]*(dx[1:]**2.0)-\
                              y[1:-1]*((dx[0:-1]**2.0)-(dx[1:]**2.0)))/\
                             (dx[0:-1]*dx[1:]*(dx[1:]+dx[0:-1]))
    #use forward diff for first point
    dydx[0]=(y[2]*(dx[0]**2)-y[1]*((dx[1]+dx[0])**2)-y[0]*((dx[0]**2)-((dx[1]+dx[0])**2)))/\
        (dx[0]*(dx[0]+dx[1])*-dx[1])
    #use backwards difference for the last point
    dydx[-1]=(y[-3]*(dx[-1]**2)-y[-2]*((dx[-1]+dx[-2])**2)-y[-1]*((dx[-1]**2)-((dx[-1]+dx[-2])**2)))/\
        (dx[-1]*(dx[-1]+dx[-2])*dx[-2])
    return dydx

###########################################################
###########################################################
###########################################################
#PARAMS

#file containg CoRoL boundary information
CoRoL_file='HSSL_boundaries_paper_18_3_17.csv'

#new file to generate
New_CoRoL_file='HSSL_boundaries_paper_18_3_17_corrected.csv'

ind_plot=14

marker_size=10

#########################################################
###########################################################
###########################################################
#constants
LEM_H=3.53E34
LEM=3.5E34

###########################################################
###########################################################
###########################################################
#read in the CoRoL file
fID=open(CoRoL_file, 'rU')
reader = csv.reader(fID, delimiter=",")
header1=reader.next()
header2=reader.next()
params = list(reader)

#TEST
#params=params[0:10]

AM_CoRoL=-1*np.ones(len(params))
omg_CoRoL=-1*np.ones(len(params))
a_CoRoL=-1*np.ones(len(params))
aspect_CoRoL=-1*np.ones(len(params))

AM_old=-1*np.ones(len(params))
dL=np.zeros(len(params))

temp=np.arange(len(params))
for i in temp[ind_plot:(ind_plot+1)]:
    print(str(i)+'/'+str(len(params)))

    if params[i][0]!='':
        #find the name of the required directory
        dir='/n/gstore/Labs/Stewart_Lab/slock/lunar_disk/HERCULES/HSSL_finding/'+params[i][24]+'/'+params[i][23]
        print dir

        AM_old[i]=np.asarray((params[i][22]),dtype=np.float)
        dL[i]=np.asarray((params[i][21]),dtype=np.float)
 
        #find all the directories 
        runall=os.listdir(dir)
        runlist=[k for k in runall if  fnmatch.fnmatch(k, '*AM*[!constant_omega]') ]
        #runlist=[k for k in runall if  fnmatch.fnmatch(k, '*AM*') ]

        #initialise arrays to record results
        stop_flag=np.empty(np.size(runlist)) #0 for failed, 1 for ok, 2 for beyond boundary
        AM=np.empty(np.size(runlist))
        omega_Kep=np.empty(np.size(runlist))
        omega_rot=np.empty(np.size(runlist))
        a=np.empty(np.size(runlist))
        aspect=np.empty(np.size(runlist))

        for j in np.arange(np.size(runlist)):
            dir2=runlist[j]
            print dir2
    
            #find the final output for each AM 
            outall=os.listdir(dir+'/'+dir2+'/Output')
            outlist=[k for k in outall if  fnmatch.fnmatch(k, '*l1_0_1.5*final') ]

            #If the run has failed record the target properties
            if np.size(outlist)==0:
                startlist=[k for k in outall if  fnmatch.fnmatch(k, '*_0') ]
                start=startlist[0]
                #print start
                
                #read in start planet
                Hparams=HERCULES_parameters()
                p=HERCULES_planet()
                file = open(dir+'/'+dir2+'/Output/'+start, "rb")
                Hparams.read_binary(file)
                p.read_binary(file)

                stop_flag[j]=0
                AM[j]=p.Ltot_tar
                omega_Kep[j]=0.0
                omega_rot[j]=0.0
                a[j]=0.0
                aspect[j]=0.0

            #if not check whether it was above or below the boundary
            else:
                #read in final planet
                out=outlist[0]
                #print out
                Hparams=HERCULES_parameters()
                p=HERCULES_planet()
                file = open(dir+'/'+dir2+'/Output/'+out, "rb")
                Hparams.read_binary(file)
                p.read_binary(file)

                #extract the AM
                AM[j]=p.Ltot_tar
                
                #see if the run was above or below the boundary
                if (p.Ltot_tar==0.0):
                    stop_flag[j]=2
                elif ((((np.amin(p.press)-p.pmin)/p.pmin)>=0.0)&(np.abs((p.layers[0].omega-p.omega_rot)/p.omega_rot)<1E-14)&(np.abs((p.Mtot_tar-p.Mtot)/p.Mtot_tar)<1E-14)):
                    #find the radius of all the layers
                    radius=np.empty(p.Nlayer)
                    for k in np.arange(p.Nlayer):
                        radius[k]=p.layers[k].a
                    dVdr=gradient2(radius, p.Ulayers-(((radius*p.omega_rot)**2)/2.0))
                    temp=np.sqrt(-1.0*dVdr/radius)

                    #assign the values we need
                    stop_flag[j]=1
                    omega_Kep[j]=temp[0]
                    omega_rot[j]=p.omega_rot
                    a[j]=p.amax
                    aspect[j]=p.aspect
                    
                else:
                    stop_flag[j]=2
                    omega_Kep[j]=0.0
                    omega_rot[j]=0.0
                    a[j]=0.0
                    aspect[j]=0.0
                    #print 'NOT ALLOWED RUN'
                    #print (np.amin(p.press)-p.pmin)/p.pmin, (np.abs((p.layers[0].omega-p.omega_rot)/p.omega_rot)), np.amin(p.press), p.pmin

        #find the successful runs
        temp=np.where(stop_flag==1)[0]
        temp1=np.argsort(AM[temp])
        ind=temp[temp1]
        #ind=ind[0:-1]

        #check the resolution and issues with missing final runs
        if (((np.absolute(AM[ind[-2]]-AM[ind[-1]]))>(dL[i]+1E-8)*LEM_H)|(AM[ind[-1]]<(AM_old[i]-1E-5)*LEM_H)):
            print dir
            print AM[ind[-2]]/LEM_H, AM[ind[-1]]/LEM_H, dL[i], (np.absolute(AM[ind[-2]]-AM[ind[-1]]))/LEM_H, AM_old[i]
            sys.exit()
        else:
            print AM[ind[-2]]/LEM_H, AM[ind[-1]]/LEM_H, dL[i], AM_old[i]

        ###############################################################
        #use last two acceptable points to find intersection by using two straight lines
        m_Kep=(omega_Kep[ind[-2]]-omega_Kep[ind[-1]])/(AM[ind[-2]]-AM[ind[-1]])
        m_rot=(omega_rot[ind[-2]]-omega_rot[ind[-1]])/(AM[ind[-2]]-AM[ind[-1]])
        
        b_Kep=omega_Kep[ind[-2]]-AM[ind[-2]]*m_Kep
        b_rot=omega_rot[ind[-2]]-AM[ind[-2]]*m_rot

        #find teh AM and omega at CoRoL
        AM_CoRoL[i]=AM[ind[-2]]+(AM[ind[-2]]-AM[ind[-1]])*(omega_Kep[ind[-2]]-omega_rot[ind[-2]])/\
            ((omega_rot[ind[-2]]-omega_rot[ind[-1]])-(omega_Kep[ind[-2]]-omega_Kep[ind[-1]]))
        omg_CoRoL[i]=m_Kep*AM_CoRoL[i]+b_Kep

        #extrapolate to find the other properties we want
        m_a=(a[ind[-2]]-a[ind[-1]])/(AM[ind[-2]]-AM[ind[-1]])
        m_aspect=(aspect[ind[-2]]-aspect[ind[-1]])/(AM[ind[-2]]-AM[ind[-1]])

        b_a=a[ind[-2]]-AM[ind[-2]]*m_a
        b_aspect=aspect[ind[-2]]-AM[ind[-2]]*m_aspect

        a_CoRoL[i]=m_a*AM_CoRoL[i]+b_a
        aspect_CoRoL[i]=m_aspect*AM_CoRoL[i]+b_aspect

        deltaL=AM_CoRoL[i]/(LEM_H)-AM_old[i]
        delta_L_res=dL[i]*np.floor((deltaL/dL[i])-1E-8)

        #######################################################
        #plot an example figure
        fig = plt.figure(figsize=(4.2,8.5))
        gs = gridspec.GridSpec(3, 1)

        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax3 = plt.subplot(gs[2], sharex=ax1)

        plt.rc('text', usetex=True)
        plt.rc('font', family='sans-serif')
        plt.rcParams.update({'font.size': 12, 'legend.fontsize':12})

        mpl.rcParams['text.latex.preamble'] = [
            #r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
            
            #r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
            #r'\usepackage{times}',    # set the normal font here
            r'\usepackage{helvet}',    # set the normal font here
            r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
            r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
            ]  

        col_line='k'
        col2='r'
        col1=cmaps.parula(0.12)

        AM_plot=np.linspace(0.0*LEM_H, 2.0*LEM_H)
        ax1.plot(AM_plot/LEM_H, (m_Kep*AM_plot+b_Kep)*1E3, '-', color=col_line, linewidth=1.5)
        #AM_plot=np.linspace(1.2*LEM_H, 1.55*LEM_H)
        ax1.plot(AM_plot/LEM_H, (m_rot*AM_plot+b_rot)*1E3, '-', color=col_line, linewidth=1.5)

        ax1.plot(AM[ind]/LEM_H,omega_Kep[ind]*1E3, '+', color=col1, markersize=marker_size)
        ax1.plot(AM[ind]/LEM_H,omega_rot[ind]*1E3, '+', color=col2, markersize=marker_size)

        ax1.plot(AM_CoRoL[i]/LEM_H, omg_CoRoL[i]*1E3, 'o', markerfacecolor=col_line, markeredgecolor='none')

        ax1.set_xlim([0.9, 1.6])
        ax1.set_ylim([0.2, 0.45])
        ax1.set_ylabel('Ang. vel. [$10^{-3}$~rad~s$^{-1}$]')
        plt.setp(ax1.get_xticklabels(), visible=False)

        #check the other parameters
        #AM_plot=np.linspace(1.4*LEM_H, 1.52*LEM_H)
        ax2.plot(AM_plot/LEM_H, (m_a*AM_plot+b_a)/1E6, '-', color=col_line)
        ax2.plot(AM[ind]/LEM_H,a[ind]/1E6, '+', color=col2, markersize=marker_size)
        ax2.plot(AM_CoRoL[i]/LEM_H, a_CoRoL[i]/1E6, 'o', markerfacecolor=col_line, markeredgecolor='none')

        ax2.set_ylim([13, 18])
        ax2.set_ylabel('Radius, $a_0$ [$10^{6}$~m]')
        plt.setp(ax2.get_xticklabels(), visible=False)


        #AM_plot=np.linspace(1.4*LEM_H, 1.52*LEM_H)
        ax3.plot(AM_plot/LEM_H, m_aspect*AM_plot+b_aspect, '-', color=col_line)
        ax3.plot(AM[ind]/LEM_H,aspect[ind], '+', color=col2, markersize=marker_size)
        ax3.plot(AM_CoRoL[i]/LEM_H, aspect_CoRoL[i], 'o', markerfacecolor=col_line, markeredgecolor='none')

        ax3.set_ylim([0.6, 0.9])
        ax3.set_ylabel('Aspect ratio')
        ax3.set_xlabel('Angular momentum [$L_{EM}$]')


        ax1.text(0.94, 0.92, 'A', horizontalalignment='center',verticalalignment='center', fontsize=12,transform=ax1.transAxes, color='k')
        ax2.text(0.94, 0.92, 'B', horizontalalignment='center',verticalalignment='center', fontsize=12,transform=ax2.transAxes, color='k')
        ax3.text(0.94, 0.92, 'C', horizontalalignment='center',verticalalignment='center', fontsize=12,transform=ax3.transAxes, color='k')


        fig.tight_layout()
        plt.savefig('Example_finding_CoRoL.pdf', dpi=400)
        #sys.exit()
