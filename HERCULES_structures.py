#SJL 11/15
#File containing the structures for dealing with HERCULES output

#import required modules
import numpy as np
import struct
import sys


###############################################################
###############################################################
###############################################################
#Parameters structure from HERCULES

class HERCULES_parameters(object):
    ###################################################
    def __init__(self):
        #Version of HERCULES
        self.flag_version=2
        self.flag_subversion=0
        self.flag_input_version=1
        self.input_file="HERCULES_input.yml"

        #verbosity
        self.flag_verbosity=0
        
        #Run name
        self.run_name_base=""
        self.flag_naming=1
        self.run_name=""

        #Format used for output
        self.flag_output_format=1
        self.thermo_var_output=[]

        #Output directory
        self.output_dir="Output/"

        #max itterations and tollerance
        self.nint_max=0
        self.toll=0.0
        
        #max itterations and tollerance for xi calculation
        self.xi_nint_max=0
        self.xi_toll=0.0
        #xi step for differential calc
        self.dxi=0.0

        #flags and their parameters
        self.flag_start=0
        self.start_file=""
        self.flag_format_start_file=1
        self.flag_Mconc=0
        self.flag_Lconc=0
        self.flag_iter_print=0

        #Run mode flag (0: single planet; 1: Increasing AM, 2: Increasing mass)
        self.flag_run_mode=0
        self.Ndiv_Xstep=0
        self.Xstep=0.
        self.Xfinal=0.
        self.flag_array_toll=0 #whether to use a different tollerance for array before the step size is decreased
        #tollerances for array calculation before step size is decreased
        self.toll_array=0.
        self.xi_toll_array=0. 

        #rotational profile properties
        self.omega_param=np.empty(3)

    ###################################################
    #function to read the parameters from a HERCULES output
    def read_binary(self, f):

        #Read in depends on type of file
        if self.flag_output_format==1:
            print('PARAM FORMAT 1')
            self.flag_version=struct.unpack('i', f.read(4))[0]
            #print('flag_version: ',self.flag_version)
            self.flag_subversion=struct.unpack('i', f.read(4))[0]
            #print('flag_subversion: ',self.flag_subversion)
            self.flag_input_version=struct.unpack('i', f.read(4))[0]
            #print('flag_input_version: ',self.flag_input_version)

            temp=struct.unpack('i', f.read(4))[0]
            #print('temp:',temp)
            temp2=struct.unpack('i', f.read(4))[0] #need to read and discard metadata for c++ size_t class (I think)
            #print('temp2:',temp2)
            self.input_file=f.read(temp2).decode('ascii')
            #print('input_file:',self.input_file)
            temp3=struct.unpack('i', f.read(4))[0]
            #print('temp3:',temp3)
            #temp3r=f.read(temp3).decode('ascii')

            self.flag_verbosity=struct.unpack('i', f.read(4))[0]
            #print('flag_verbosity:',self.flag_verbosity)

            temp=struct.unpack('i', f.read(4))[0]
            #print('temp:',temp)
            temp2=struct.unpack('i', f.read(4))[0]
            #print('temp2:',temp2)
            self.run_name_base=f.read(temp).decode('ascii')
            #print('run_name_base:',self.run_name_base)

            self.flag_naming=struct.unpack('i', f.read(4))[0]

            temp=struct.unpack('i', f.read(4))[0]
            temp2=struct.unpack('i', f.read(4))[0]
            self.run_name=f.read(temp).decode('ascii')

            self.flag_output_format=struct.unpack('i', f.read(4))[0]

            temp=struct.unpack('i', f.read(4))[0]
            self.thermo_var_output=[""]*temp
            for i in np.arange(temp):
                temp=struct.unpack('i', f.read(4))[0]
                temp2=struct.unpack('i', f.read(4))[0]
                self.thermo_var_output[i]=f.read(temp).decode('ascii')

            temp=struct.unpack('i', f.read(4))[0]
            temp2=struct.unpack('i', f.read(4))[0]
            self.output_dir=f.read(temp).decode('ascii')
                
            
            self.nint_max=struct.unpack('i', f.read(4))[0]
            self.toll=struct.unpack('d', f.read(8))[0]

            self.xi_nint_max=struct.unpack('i', f.read(4))[0]
            self.xi_toll=struct.unpack('d', f.read(8))[0]
            self.dxi=struct.unpack('d', f.read(8))[0]
            
            self.flag_start=struct.unpack('i', f.read(4))[0]
            self.flag_format_start_file=struct.unpack('i', f.read(4))[0]
            
            temp=struct.unpack('i', f.read(4))[0]
            temp2=struct.unpack('i', f.read(4))[0]
            self.start_file=f.read(temp).decode('ascii')
            
            self.flag_Mconc=struct.unpack('i', f.read(4))[0]
            self.flag_Lconc=struct.unpack('i', f.read(4))[0]
            self.flag_iter_print=struct.unpack('i', f.read(4))[0]

            self.flag_run_mode=struct.unpack('i', f.read(4))[0]
            self.Ndiv_Xstep=struct.unpack('i', f.read(4))[0]
            self.Xstep=struct.unpack('d', f.read(8))[0]
            self.Xfinal=struct.unpack('d', f.read(8))[0]
            self.flag_toll_array=struct.unpack('i', f.read(4))[0]
            self.toll_array=struct.unpack('d', f.read(8))[0]
            self.xi_toll_array=struct.unpack('d', f.read(8))[0]

            self.omega_param=np.asarray(struct.unpack('3d', f.read(3*8)))
            print('PARAM READ SUCCESSFUL WITH FORMAT 1')

        else:
            print('PARAM FORMAT 0')
            self.run_name=f.read(200).rstrip()
            #print('run_name:',self.run_name)
            self.nint_max=struct.unpack('i', f.read(4))[0]
            #print('nint_max:',self.nint_max)
            self.toll=struct.unpack('d', f.read(8))[0]
            #print('toll:',self.toll)

            self.xi_nint_max=struct.unpack('i', f.read(4))[0]
            #print('xi_nint_max:',self.xi_nint_max)
            self.xi_toll=struct.unpack('d', f.read(8))[0]
            #print('xi_toll:',self.xi_toll)
            self.dxi=struct.unpack('d', f.read(8))[0]
            #print('dxi:',self.dxi)
            
            self.flag_start=struct.unpack('i', f.read(4))[0]
            #print('flag_start:',self.flag_start)
            self.start_file=f.read(200).rstrip()
            #print('start_file:',self.start_file)
            self.flag_Mconc=struct.unpack('i', f.read(4))[0]
            #print('flag_Mconc:',self.flag_Mconc)
            self.flag_Lconc=struct.unpack('i', f.read(4))[0]
            #print('flag_Lconc:',self.flag_Lconc)
            self.flag_iter_print=struct.unpack('i', f.read(4))[0]
            #print('flag_iter_print:',self.flag_iter_print)

            self.omega_param=np.asarray(struct.unpack('3d', f.read(3*8)))
            #print('omega_param:',self.omega_param)
            print('PARAM READ SUCCESSFUL WITH FORMAT 0')

        
###############################################################
###############################################################
###############################################################
#Planet structure from HERCULES

class HERCULES_planet(object):
    ###################################################
    def __init__(self):
        #Number of mu points, number of layers, number of different materials
        self.Nmu=0
        self.Nlayer=0
        self.Nmaterial=0
        
        #max legendre degree to calculate to
        self.kmax=0
        
        #Mass, AM of planet
        self.Mtot=0.0
        self.Ltot=0.0
        
        #various properties of planet
        self.omega_rot=0.0    #solid body rotation rate
        self.pmin=0.0         #pressure of outermost layer
        self.amax=0.0         #maximum equitorial radius
        self.aspect=0.0       #aspect ratio
        self.ref_rho=0.0      #constant density layers if needed
        self.Mtot_tar=0.0     #mass target for itteration
        self.Ltot_tar=0.0     #AM target for itteration
        self.Ucore=0.0        #Potential at the centre of body
        self.pcore=0.0        #pressure at centre of body

        #Vectors of real density, pressure, dpdr and equitorial potetnail of the layers
        self.real_rho=np.empty(0)
        self.press=np.empty(0)
        self.dpdr=np.empty(0)
        self.Ulayers=np.empty(0)
        self.Mout=np.empty(0)
        self.Lout=np.empty(0)

        #vector of Js
        self.Js=np.empty(0)

        #Vectors of angles
        self.mu=np.empty(0)

        #Vector of material flags
        self.Mint_material=np.empty(0)
        self.flag_material=np.empty(0)
        self.material_lay=np.empty(0)
        
        #Vector of material masses
        self.M_materials=np.empty(0)

        #array of structures
        self.layers=[]

        #array of materials
        self.materials=[]

        #array of thermal profiles
        self.thermo_profiles=[]

        #SJL 1/16
        #properties that we might want to know about planets
        self.pCMB=0.0
        self.pf_layer0=0.0

        self.T=np.empty(0)
        self.S=np.empty(0)
        self.E=np.empty(0)

    ###################################################
    #function to read the structure from a HERCULES output
    def read_binary(self, f, flag_output_format, thermo_var_output):

        if flag_output_format==1:
        #read ints
            print('OUTPUT FORMAT 1')
            self.Nmu=struct.unpack('i', f.read(4))[0]
            #print('Nmu:',self.Nmu,' should be 400')
            self.Nlayer=struct.unpack('i', f.read(4))[0]
            #print('Nlayer:',self.Nlayer,' should be 150 or 200')
            self.Nmaterial=struct.unpack('i', f.read(4))[0]
            #print('Nmaterial:',self.Nmaterial,' should be 2')
            self.kmax=struct.unpack('i', f.read(4))[0]
            #print('kmax:',self.kmax,' should be 6?')

            #read doubles
            self.Mtot=struct.unpack('d', f.read(8))[0]
            #print('Mtot:',self.Mtot)
            self.Ltot=struct.unpack('d', f.read(8))[0]
            #print('Ltot:',self.Ltot)
            self.omega_rot=struct.unpack('d', f.read(8))[0]
            #print('omega_rot:',self.omega_rot)
            self.pmin=struct.unpack('d', f.read(8))[0]
            #print('pmin:',self.pmin)
            self.amax=struct.unpack('d', f.read(8))[0]
            #print('amax:',self.amax)
            self.aspect=struct.unpack('d', f.read(8))[0]
            #print('aspect:',self.aspect)
            self.ref_rho=struct.unpack('d', f.read(8))[0]
            #print('ref_rho:',self.ref_rho)
            self.Mtot_tar=struct.unpack('d', f.read(8))[0]
            #print('Mtot_tar:',self.Mtot_tar)
            self.Ltot_tar=struct.unpack('d', f.read(8))[0]
            #print('Ltot_tar:',self.Ltot_tar)
            self.Ucore=struct.unpack('d', f.read(8))[0]
            #print('Ucore:',self.Ucore)
            self.pcore=struct.unpack('d', f.read(8))[0]
            #print('pcore:',self.pcore)

            #read vectors
            #print(self.Nlayer)
            #print(struct.calcsize(str(self.Nlayer)+'d'))
            self.real_rho=np.asarray(struct.unpack(str(self.Nlayer)+'d', f.read(self.Nlayer*8)))
            self.press=np.asarray(struct.unpack(str(self.Nlayer)+'d', f.read(self.Nlayer*8)))
            self.dpdr=np.asarray(struct.unpack(str(self.Nlayer)+'d', f.read(self.Nlayer*8)))
            self.mu=np.asarray(struct.unpack(str(self.Nmu)+'d', f.read(self.Nmu*8)))
            self.Ulayers=np.asarray(struct.unpack(str(self.Nlayer)+'d', f.read(self.Nlayer*8)))
            self.Mint_materials=np.asarray(struct.unpack(str(self.Nlayer)+'d', f.read(self.Nlayer*8)))
            self.flag_material=np.asarray(struct.unpack(str(self.Nlayer)+'i', f.read(self.Nlayer*4)))
            self.M_materials=np.asarray(struct.unpack(str(self.Nmaterial)+'d', f.read(self.Nmaterial*8)))
            self.material_lay=np.asarray(struct.unpack(str(self.Nmaterial)+'i', f.read(self.Nmaterial*4)))
            self.Mout=np.asarray(struct.unpack(str(self.Nlayer)+'d', f.read(self.Nlayer*8)))
            self.Lout=np.asarray(struct.unpack(str(self.Nlayer)+'d', f.read(self.Nlayer*8)))
            self.Js=np.asarray(struct.unpack(str(self.kmax+1)+'d', f.read((self.kmax+1)*8)))


            #initialise layers and read in
            for i in np.arange(0,self.Nlayer):
                self.layers.append(HERCULES_concentric_layer())
                self.layers[i].read_binary(f,flag_output_format)


            #initialise materials and read in
            for i in np.arange(0,self.Nmaterial):
                self.materials.append(HERCULES_EOS())
                self.materials[i].read_binary(f,flag_output_format)

            #initialise thermal profiles and read in
            for i in np.arange(0,self.Nmaterial):
                self.thermo_profiles.append(HERCULES_thermal_profile())
                self.thermo_profiles[i].read_binary(f, flag_output_format)

            #Find out if there is any thermodynamic data to read in
            #Temperature
            for count in np.arange(np.size(thermo_var_output)):
                if ((thermo_var_output[count]=="T")|(thermo_var_output[count]=="t")|\
                    (thermo_var_output[count]=="temperature")|\
                    (thermo_var_output[count]=="Temperature")|\
                    (thermo_var_output[count]=="temp")|(thermo_var_output[count]=="Temp")):
                    write_var=1
                    self.T=np.asarray(struct.unpack(str(self.Nlayer)+'d', f.read((self.Nlayer)*8)))
                    break
                else:
                    write_var=0

            #Entropy
            for count in np.arange(np.size(thermo_var_output)):
                if ((thermo_var_output[count]=="S")|(thermo_var_output[count]=="s")|\
                    (thermo_var_output[count]=="entropy")|(thermo_var_output[count]=="Entropy")):
                    write_var=1
                    self.S=np.asarray(struct.unpack(str(self.Nlayer)+'d', f.read((self.Nlayer)*8)))
                    break
                else:
                    write_var=0

            #Internal energy
            for count in np.arange(np.size(thermo_var_output)):
                if ((thermo_var_output[count]=="U")|(thermo_var_output[count]=="u")|\
                    (thermo_var_output[count]=="E")|(thermo_var_output[count]=="e")|\
                    (thermo_var_output[count]=="internal energy")|\
                    (thermo_var_output[count]=="Inernal energy")):
                    write_var=1
                    self.E=np.asarray(struct.unpack(str(self.Nlayer)+'d', f.read((self.Nlayer)*8)))
                    break
                else:
                    write_var=0
            print('OUTPUT READ SUCCESSFUL WITH FORMAT 1')
                    
        else:
            print('OUTPUT FORMAT 0')
            #read ints
            self.Nmu=struct.unpack('i', f.read(4))[0]
            self.Nlayer=struct.unpack('i', f.read(4))[0]
            self.Nmaterial=struct.unpack('i', f.read(4))[0]
            self.kmax=struct.unpack('i', f.read(4))[0]

            #read doubles
            self.Mtot=struct.unpack('d', f.read(8))[0]
            self.Ltot=struct.unpack('d', f.read(8))[0]
            self.omega_rot=struct.unpack('d', f.read(8))[0]
            self.pmin=struct.unpack('d', f.read(8))[0]
            self.amax=struct.unpack('d', f.read(8))[0]
            self.aspect=struct.unpack('d', f.read(8))[0]
            self.ref_rho=struct.unpack('d', f.read(8))[0]
            self.Mtot_tar=struct.unpack('d', f.read(8))[0]
            self.Ltot_tar=struct.unpack('d', f.read(8))[0]
            self.Ucore=struct.unpack('d', f.read(8))[0]
            self.pcore=struct.unpack('d', f.read(8))[0]

            #read vectors
            self.real_rho=np.asarray(struct.unpack(str(self.Nlayer)+'d', f.read(self.Nlayer*8)))
            self.press=np.asarray(struct.unpack(str(self.Nlayer)+'d', f.read(self.Nlayer*8)))
            self.dpdr=np.asarray(struct.unpack(str(self.Nlayer)+'d', f.read(self.Nlayer*8)))
            self.mu=np.asarray(struct.unpack(str(self.Nmu)+'d', f.read(self.Nmu*8)))
            self.Ulayers=np.asarray(struct.unpack(str(self.Nlayer)+'d', f.read(self.Nlayer*8)))
            self.flag_material=np.asarray(struct.unpack(str(self.Nlayer)+'i', f.read(self.Nlayer*4)))
            self.M_materials=np.asarray(struct.unpack(str(self.Nmaterial)+'d', f.read(self.Nmaterial*8)))
            self.Mout=np.asarray(struct.unpack(str(self.Nlayer)+'d', f.read(self.Nlayer*8)))
            self.Lout=np.asarray(struct.unpack(str(self.Nlayer)+'d', f.read(self.Nlayer*8)))
            self.Js=np.asarray(struct.unpack(str(self.kmax+1)+'d', f.read((self.kmax+1)*8)))


            #initialise layers and read in
            for i in np.arange(0,self.Nlayer):
                self.layers.append(HERCULES_concentric_layer())
                self.layers[i].read_binary(f,flag_output_format)


            #initialise materials and read in
            for i in np.arange(0,self.Nmaterial):
                self.materials.append(HERCULES_EOS())
                self.materials[i].read_binary(f,flag_output_format)
            print('OUTPUT READ SUCCESSFUL WITH FORMAT 0')


    ###################################################
    #function to calculate CMB pressure assuming the core is the lowest layer
    def calc_pCMB(self):
        core_layer=self.Nmaterial-1
        temp=np.where(self.flag_material==core_layer)[0]
        ind=temp[0]
        
        self.pCMB=self.press[ind]

    ###################################################
    #function to calculate CMB pressure assuming the core is the lowest layer
    #pass the fraction of material to find the pressure for, from core up
    def calc_pf_layer0(self, f):
        #redefine f for ease of use
        f=1-f

        #find an array of cumulative masses
        temp=np.asarray([0])
        for i in np.arange(self.Nlayer-1):
            temp=np.append(temp, ((self.layers[i].Vol-self.layers[i+1].Vol)*self.real_rho[i]))
        mass=np.cumsum(temp)

        #find the layer below the fraction f of the mantle
        temp=np.where(mass>f*self.M_materials[0])[0]
        ind_max=temp[0]

        #linearly interpolare
        frac=(f*self.M_materials[0]-mass[ind_max-1])/(mass[ind_max]-mass[ind_max-1])
        pf=self.press[ind_max]*frac+self.press[ind_max-1]*(1-frac)

        #print pf/1E9, self.press[ind_max]/1E9, self.press[ind_max-1]/1E9
        
        self.pf_layer0=pf
        

###############################################################
###############################################################
###############################################################
#Concentric layer structure from HERCULES
class HERCULES_concentric_layer(object):
    ###################################################
    def __init__(self):
        #number of mu points and max legendre degree
        self.Nmu=0
        self.kmax=0

        #the rotation rate of layer, density (only of concentric layer), mass 
        self.omega=0.0
        self.rho=0.0
        self.M=0.0
        self.Vol=0.0

        #equitorial and polar radius
        self.a=0.0 
        self.b=0.0
        
        #Moment of inertia
        self.I=0.0

        #Vector of the angles and xi
        self.mu=np.empty(0)
        self.xi=np.empty(0)

        #vector of Js
        self.Js=np.empty(0)

    ###################################################
    def read_binary(self, f,flag_output_format):
        
        #read ints
        self.Nmu=struct.unpack('i', f.read(4))[0]
        self.kmax=struct.unpack('i', f.read(4))[0]

        #read doubles
        self.omega=struct.unpack('d', f.read(8))[0]
        self.rho=struct.unpack('d', f.read(8))[0]
        self.M=struct.unpack('d', f.read(8))[0]
        self.Vol=struct.unpack('d', f.read(8))[0]
        self.a=struct.unpack('d', f.read(8))[0]
        self.b=struct.unpack('d', f.read(8))[0]
        self.I=struct.unpack('d', f.read(8))[0]

        #read vectors
        self.mu=np.asarray(struct.unpack(str(self.Nmu)+'d', f.read(self.Nmu*8)))
        self.xi=np.asarray(struct.unpack(str(self.Nmu)+'d', f.read(self.Nmu*8)))
        self.Js=np.asarray(struct.unpack(str(self.kmax+1)+'d', f.read((self.kmax+1)*8)))


                        
###############################################################
###############################################################
###############################################################
#Concentric layer structure from HERCULES
class HERCULES_EOS(object):
    ###################################################
    def __init__(self):
        self.fname=""
        self.EOS_type=0

        #For newer versions
        self.std_fname=""
        self.ext_fname=""
        self.flag_EOS=0
        self.flag_ext=0
        self.flag_interp=0
        self.rho_interp_pmin=3E3

        self.Ndata=0
        self.Ndata2=0


    ###################################################
    def read_binary(self, f, flag_file_format):

        if flag_file_format==1:
            temp=struct.unpack('i', f.read(4))[0]
            temp2=struct.unpack('i', f.read(4))[0]
            self.std_fname=f.read(temp).decode('ascii')

            temp=struct.unpack('i', f.read(4))[0]
            temp2=struct.unpack('i', f.read(4))[0]
            if temp!=0:
                temp2=struct.unpack('i', f.read(4))[0]
                self.ext_fname=f.read(temp).decode('ascii')
            else:
                self.ext_fname=""
            
            self.EOS_type=struct.unpack('i', f.read(4))[0]
            
            self.flag_EOS=struct.unpack('i', f.read(4))[0]
            self.flag_ext=struct.unpack('i', f.read(4))[0]
            self.flag_interp=struct.unpack('i', f.read(4))[0]
            self.rho_interp_min=struct.unpack('d', f.read(8))[0]

            self.Ndata=struct.unpack('i', f.read(4))[0]
            self.Ndata2=struct.unpack('i', f.read(4))[0]

        else:
            self.fname=f.read(200).rstrip()
            self.EOS_type=struct.unpack('i', f.read(4))[0]


###############################################################
###############################################################
###############################################################
#Concentric layer structure from HERCULES
class HERCULES_thermal_profile(object):
    ###################################################
    def __init__(self):

        #Type of profile
        flag_type=0;

        #Number of data points
        Npoints=0;

        #File name
        file_name="";

        #arrays
        m=np.empty(0)
        S=np.empty(0)
        rho=np.empty(0)
        GPE=np.empty(0)
        U=np.empty(0)
        KE=np.empty(0)
        Etot=np.empty(0)

    ###################################################
    def read_binary(self, f, flag_file_format):

        if flag_file_format==1:

            self.flag_type=struct.unpack('i', f.read(4))[0]
            self.Npoints=struct.unpack('i', f.read(4))[0]

            temp=struct.unpack('i', f.read(4))[0]
            temp2=struct.unpack('i', f.read(4))[0]
            self.file_name=f.read(temp).decode('ascii')


        else:
            ValueError("Incorrect file format input option")
                
