import numpy as np
import os
from HERCULES_structures import *
from isentrope_planet import *
mant_mat_id = 403 #USER INPUT
core_mat_id = 402 #USER INPUT

class HERC_run:
    def __init__(self,planet=None):
        self.planet=None
        self.M = 0.
        self.Mcore = 0.
        self.Mmant = 0.
        self.r10 = 0.
        self.rCMB0 = 0.
        self.PCMB0 = 0.
        self.Pcore0 = 0.
        self.basename = ''
        if planet is not None:
            self.load_planet(planet)
        self.keys = np.array([])
        self.paramsdict = dict()
        self.planetdict = dict()
        self.Ltot = np.array([])
        self.PCMB = np.array([])
        self.Pcore = np.array([])
        self.r1 = np.array([])
        self.b1 = np.array([])
        self.rCMB = np.array([])
        self.bCMB = np.array([])
        self.aspect = np.array([])
        self.w = np.array([])
    
    def load_planet(self,planet):
        if planet.file is None:
            print('Error: tried to load empty planet')
            return
        planet.convert_to_mks()
        self.planet=planet # this is the planet_profile object that defines the HERCULES ICs
        self.M=self.planet.M
        indcore=np.where(self.planet.mat==core_mat_id)
        self.Mcore=self.planet.mass[indcore][-1]
        self.Mmant=self.M-self.Mcore
        self.r10=self.planet.rarr[-1]
        self.rCMB0=self.planet.rarr[indcore][-1]
        self.PCMB0=self.planet.pressure[indcore][-1]
        self.Pcore0=self.planet.pressure[0]
        self.Lzstar=self.planet.Lzstar
        #self.Lfinal=((self.M/M_earth)**2*5+1)*LEM
        # finding basename, expecting the file to be in the planet_profiles folder
        #print(np.char.split(planet.file,'/'))
        split=np.char.split(planet.file,'/').item()[2]
        print(split)
        self.basename=np.char.split(split,'N').item()[0] #assuming the same naming format as in MakePlanets
        
    def load_data(self,path='./'):
        for filename in os.listdir(path):
            if filename.startswith(self.basename) and filename.endswith('final'):
                print(filename)
                temp = filename.replace(self.basename+'_','')
                keyparams = temp.replace('_final','')
                tparams = HERCULES_parameters()
                tparams.flag_output_format=1
                tparams.thermo_var_output=['T','S','E']
                tplanet = HERCULES_planet()
                tplanet.flag_output_format=1
                tplanet.thermo_var_output=['T','S','E']
                try:   
                    try:
                        with open(path+filename, "rb") as file:
                            tparams.read_binary(file)
                            tplanet.read_binary(file,1,['T','S','E'])
                    except:
                        print('ERROR: READ WITH FORMAT 1 UNSUCCESSFUL')
                        tparams.flag_output_format=0
                        tplanet.flag_output_format=0
                        with open(path+filename, "rb") as file:
                            tparams.read_binary(file)
                            tplanet.read_binary(file,0,['T','S','E'])

                    tplanet.calc_pCMB()
                    print('Loaded HERCULES output ',filename)
                    self.keys=np.append(self.keys,keyparams)
                    self.paramsdict[keyparams]=tparams
                    self.planetdict[keyparams]=tplanet
                    self.Ltot=np.append(self.Ltot,tplanet.Ltot)
                    self.PCMB=np.append(self.PCMB,tplanet.pCMB)
                    self.Pcore=np.append(self.Pcore,tplanet.pcore)
                    self.r1=np.append(self.r1,tplanet.amax)
                    self.b1=np.append(self.b1,tplanet.layers[0].b)
                    #self.b1=np.append(self.b1,tplanet.layers[0].xi[-1]*tplanet.layers[0].a*tplanet.layers[0].mu[-1])
                    self.w=np.append(self.w,tplanet.omega_rot)
                    core_layer=tplanet.Nmaterial-1
                    #ind=np.where(tplanet.flag_material==core_layer)[0]
                    ind=np.where(tplanet.flag_material>0)[0][0]
                    #print(ind)
                    #print(tplanet.layers[ind])
                    self.rCMB=np.append(self.rCMB,tplanet.layers[ind].a)
                    print(tplanet.layers[ind].a)
                    self.bCMB=np.append(self.bCMB,tplanet.layers[ind].b)
                    #self.bCMB=np.append(self.bCMB,tplanet.layers[ind[0]].xi[-1]*tplanet.layers[ind[0]].a*tplanet.layers[ind[0]].mu[-1])
                    self.aspect=np.append(self.aspect,tplanet.aspect)
                except:
                    print('ERROR: READ UNSUCCESSFUL. SKIPPING FILE')
        sort=np.argsort(self.Ltot)
        self.keys = self.keys[sort]
        self.Ltot = self.Ltot[sort]
        self.PCMB = self.PCMB[sort]
        self.Pcore = self.Pcore[sort]
        self.r1 = self.r1[sort]
        self.rCMB = self.rCMB[sort]
        self.w = self.w[sort]

        self.b1 = self.b1[sort]
        #self.rCMB = self.rCMB[sort]
        self.bCMB = self.bCMB[sort]
        self.aspect = self.aspect[sort]
        #self.w = self.w[sort]