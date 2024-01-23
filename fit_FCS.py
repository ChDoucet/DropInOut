

import numpy as np
import pandas as pd
from numpy import loadtxt
import linecache
from lmfit import minimize




def load_curves(Filename):

    File = loadtxt(Filename, delimiter=",", skiprows=13)
    
    return File


# Extract metadata
def extract_date(Filename):
    Date = pd.read_csv(Filename,delimiter=",", skiprows=3, nrows=1)

    return Date



def extract_metadata(Filename):

    MetaData = pd.read_csv(Filename,delimiter=",", usecols=[1],skiprows=5, nrows=5, dtype=np.float64()) 	    # File is a csv array

    return MetaData 	



def extract_CPS(Filename):

    cps = linecache.getline(Filename, 12)
    CPS = cps.split(",")
    CPS.pop(0)
    Cps = [float(x) for x in CPS]

    return Cps


class FCS_Csv:       
    """This script defines a class object named FCS_csv().
    Its rationale is to store and analyze data related to FCS curves.
    It was developed to extract and analyze multiposition datasets acquired with ISS.
    The functions load_curves, extract_date and extract_CPS will instanciate some parameters of the objects.
    It also contains methods to fit the auto-correlation curves with one or two species. """
        
      
    def __init__(self, Filename):
        
        Date = extract_date(Filename)
        name = Filename.split('/')[-1]
        
        MetaData = extract_metadata(Filename)
        CPS = extract_CPS(Filename)
        
        self.name = name
        self.data = load_curves(Filename)             #csv file
        self.laser_power = name.split('.')[0].split('-')[0] # laser power in uW 
        self.time = name.split('.')[-2].split('-')[-1]  # time after reaction initiation (minutes)
        self.date = Date
        self.cps = CPS
        self.nb_positions = MetaData.iloc[2,0]
        self.nb_times = MetaData.iloc[1,0]
        self.nb_curves = self.nb_times*self.nb_positions
        self.acquisition_time = MetaData.iloc[0,0]	# acquisition time in seconds
        self.fluo_peptide = []		
        self.cold_peptide = []
        self.RNA = []				
        self.PKA = []
        self.LPP = []
        self.MnCl2 = []
        self.height = []
        self.experimentator = []
        
    


# indique la représentation en chaîne de caractères d'un objet
    def __str__(self):
        attrs = vars(self)
        return(', '.join("%s: %s" % item for item in attrs.items()))
    
    

def func1(params,xdata,ydata):

    N = params['N']
    t = params['t']
    R = params['R']
    Offset = params['Offset']
    
    y_fit = ((1/N)*np.power((1+(xdata/t)), -1) *
            np.power((1+(1/R)*(1/R)*xdata/t), -0.5)+Offset)
    
    return y_fit-ydata
    


def func2(params,xdata,ydata):
    
    
    N = params['N']
    t1 = params['t1']
    t2 = params['t2']
    F1 = params['F1']
    R = params['R']
    Offset = params['Offset']

    y_fit = ((1/N)*(F1*np.power((1+(xdata/t1)), -1)*np.power((1+(1/R)*(1/R)*xdata/t1), -0.5) +
            (1-F1)*np.power((1+(xdata/t2)), -1)*np.power((1+(1/R)*(1/R)*xdata/t2), -0.5))+Offset)
    
    return y_fit-ydata
    


def fit_FCS(func,xdata,ydata,params):


    fitted_params = minimize(func, params, args=(xdata, ydata), method='least_squares')

    return fitted_params

    



def g1(xdata,params):
    N = params['N']
    t = params['t']
    R = params['R']
    Offset = params['Offset']
    
    g1 = ((1/N)*np.power((1+(xdata/t)), -1) *
             np.power((1+(1/R)*(1/R)*xdata/t), -0.5)+Offset)
    
    return g1

def g2(xdata,params):
    N = params['N']
    t1 = params['t1']
    t2 = params['t2']
    F1 = params['F1']
    R = params['R']
    Offset = params['Offset']

    g2 = ((1/N)*(F1*np.power((1+(xdata/t1)), -1)*np.power((1+(1/R)*(1/R)*xdata/t1), -0.5) +
             (1-F1)*np.power((1+(xdata/t2)), -1)*np.power((1+(1/R)*(1/R)*xdata/t2), -0.5))+Offset)
    
    return g2