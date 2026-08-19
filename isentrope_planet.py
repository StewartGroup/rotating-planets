import numpy as np

# calculate the structure for one planet
class isentrope_class:
    """Class to isentrope data extracted from EOS table."""  # this is a documentation string for this class
    def __init__(self,entropy=None): # self is the default name of the object for internal referencing of the variables in the class
        """A function to initialize the class object.""" # this is a documentation string for this function
        self.entropy = entropy
        self.ND = 0 # number of density points
        self.density     = []   
        self.pressure    = []
        self.temperature = []
        self.soundspeed  = []
        self.energy      = []
        # not going to use all the variables in the file
        self.units = '' # I like to keep a text note in a structure about the units

class planet_profile:
    def __init__(self,file=None):
        self.file=file
        self.M = 0.
        self.cf = 0.
        self.rarr = np.array([])
        self.density = np.array([])
        self.pressure = np.array([])
        self.temperature = np.array([])
        self.energy = np.array([])
        self.entropy = np.array([])
        self.mat = np.array([])
        self.mass = np.array([])
        self.K=0
        self.Lzstar=0
        self.units = 'cgs'
        self.iendcore=0
        if file is not None:
            self.load(file)
            self.calcK(file)
        
    def load(self,file='profile.txt'):
        self.rarr,self.density,self.temperature,self.pressure,self.energy,self.entropy,self.mass,self.mat=np.loadtxt(file,skiprows=1,unpack=True)
        self.M=self.mass[-1]
        self.file=file
        
    def convert_to_mks(self):
        if self.units!='mks':
            self.rarr = self.rarr * 1.e-2            #cm to m
            self.density = self.density * 1.e3       #g/cm^3 to kg/m^3
            self.pressure = self.pressure * 1.e-1    #dyne/cm^2 to Pa
            self.energy = self.energy * 1.e-4        #erg/g to J/kg
            self.entropy = self.entropy * 1.e-4      #erg/g/K to J/kg/K
            self.mass = self.mass * 1.e-3            #g to kg
            self.M = self.mass[-1]
            self.units = 'mks'
            
    def convert_to_cgs(self):
        if self.units!='cgs':
            self.rarr = self.rarr * 1.e2            #m to cm
            self.density = self.density * 1.e-3       #kg/m^3 to g/cm^3
            self.pressure = self.pressure * 1.e1    #Pa to dyne/cm^2
            self.energy = self.energy * 1.e4        #J/kg to erg/g
            self.entropy = self.entropy * 1.e4      #J/kg/K to erg/g/K
            self.mass = self.mass * 1.e3            #kg to g
            self.M = self.mass[-1]
            self.units = 'cgs'

    def calcK(self,startProfile='profile.txt'):
        [r0cgs,rho0cgs,T0,P0cgs,u0cgs,s0cgs,menclosedcgs,matID0]=np.loadtxt(startProfile,skiprows=1,unpack=True,delimiter=' ')
        r0=r0cgs/1.e2
        menclosed=menclosedcgs/1.e3
        Mt=menclosed[-1]
        Rt=r0[-1]
        Ishell=np.zeros(np.size(r0))
        for i in range(1,np.size(r0)):
            Ishell[i]=(2/5)*(menclosed[i]-menclosed[i-1])*(r0[i]**5-r0[i-1]**5)/(r0[i]**3-r0[i-1]**3)
        Itot=np.sum(Ishell,dtype='float64')
        K=Itot/Mt/(Rt**2)
        self.K=K
        self.Lzstar=self.K*(Mt)*(Rt)**2*np.sqrt(6.67408e-11*(Mt)/(Rt)**3)/3.5E34
        
            
    def write(self,file='profile.txt'):
        self.convert_to_cgs()
        with open(file,"w") as outfile: # open the file for writing
            outfile.write("#radius[cm] density[g/cm^3] temperature[K] pressure[dyne/cm^2] sp.energy[erg/g] sp.entropy[erg/g/K] menclosed[g] matID\n")
            for i in range(0,len(self.rarr)):
                outfile.write("{:.8e} {:.8e} {:.8e} {:.8e} {:.8e} {:.8e} {:.8e} {:g}\n".format(self.rarr[i],self.density[i],self.temperature[i],self.pressure[i],self.energy[i],self.entropy[i],self.mass[i],self.mat[i]))
        