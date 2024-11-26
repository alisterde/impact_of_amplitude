import math
from typing import Protocol
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 24})
import os
import scipy
from scipy import optimize
import copy

import pints

from numba import int32, float32, float64,  types, typed, boolean# import the types
from numba.experimental import jitclass

# from electrochemistry_modelling import riddlers_method

from electrochemistry_modelling import harmonics_and_fourier_transform

import copy

import time

# from electrochemistry_modelling import dispersion_class

kv_ty = (types.int64, types.unicode_type)


spec = [
    ('R', float64),
    ('temp', float64),
    ('F', float64),
    ('s', float64),
    ('v', float64),
    ('oxidising_direction', boolean),
    ('mu', float64),
    ('freq', float64),
    ('omega', float64),
    ('omega0', float64),
    ('epsilon', float64),
    ('epsilon_r', float64),
    ('row', float64),
    ('tau0', float64),
    ('n', float64),
     ('E0', float64),
    ('T0', float64),
    ('I0', float64),
    ('kappa0', float64),
    ('epsilon0', float64),
    ('alpha', float64),
    ('zeta', float64),
    ('dtheta_ox_dt',float64),
    ('theta_ox',float64),
    ('theta_ox_list', float64[:]),
    ('dtheta_ox_dt_list', float64[:]),
    ('epsilon_start', float64),
    ('epsilon_reverse', float64),
    ('deltaepislon', float64),
    ('startT', float64),
    ('revT', float64),
    ('dimlessRevT', float64),
    ('endT', float64),
    ('timeStepSize', float64),
    ('dimlessTimeStepSize', float64),
    ('i', float64[:]),
    ('episilon_without_drop', float64[:]),
    ('episilon_IR_drop_componenet', float64[:]),
    ('I_inital', float64),
    ('protocol', types.DictType(*kv_ty)),
    ('solver_option', types.DictType(*kv_ty)),
    ('gamma0', float64),
    ('gamma1', float64),
    ('gamma2', float64),
    ('gamma3', float64),
    ('gamma4', float64),
    ('gamma5', float64),
    ('gamma6', float64),
    ('gamma7', float64),
    ('gamma8', float64),
    ('gamma9', float64),
    ('permittivityFree', float64),
    ('boltzmann', float64),
    ('helmholtzConstant', float64),
    ('Z', float64),
    ('dielectricConstant', float64),
    ('numberConc', float64),
    ('pzc', float64),
    ('electrostaticPotential0', float64),
    ('gouyChapman_nd', float64),
    ('iterations', int32),
    ('appliedPotential', float64[:]),
    ('drivAppliedPotential', float64[:]),
    ('appliedPotentialDCVComponetent',float64[:]),
    ('modelling_faradaic_current', boolean),
    ('modeling_experimental_data', boolean),
    ('fit_alpha', boolean)

]

@jitclass(spec)
class newtonRaphsonCapFaradaicDip():
    '''
    This is a class to solve the mathematical model outlined in [1] written in base python
    using an implimentation of the Newton-Raphson methond
    [1] The Impact of Sinusoidal Amplitude on Visualising Thermodynamic Dispersion in Fourier Transformed AC Voltammetry.
    Alister R. Dale-Evans, Nicholas D. J. Yates, Rifael Z. Snitkoff-Sol, Lior Elbaz,
    Alan M. Bond, David J. Gavaghan, and Alison Parkin
    '''

    def __init__(self, timeStepSize: float, numberOfMeasurements: int, startPotential: float = -0.15, revPotential: float = -0.75,
                 rateOfPotentialChange: float = -22.35e-3, inital_current: float = 3.92061e-06,
                 freq: float = 8.95931721948, deltaepislon: float = 150.0E-3, electrode_area: float = 0.03, Z: float = 1.0,
                 numberConc: float = 6.0221409e+23*100*10**-3, fit_alpha: boolean = False):
                 
        
        #defining constants
        self.R = 8.314459848 #J / mol·K the perfect gas constant
        # self.temp = 5.0+273.15  # k temperature in kelvin
        # self.temp = 25.0+273.15  # k temperature in kelvin
        self.temp = 25.0+273.0
        self.F = 96485.3328959 # A.S.mol−1 Faraday constant

        #parameters for non-dimensionalisation
        self.s = electrode_area#E-4 # m^2 geometric area of the electrode
        rate = abs(rateOfPotentialChange)
        if startPotential < revPotential:
            self.v = rate #Vs-1 the rate at which the potential is swept over at
            self.oxidising_direction = True
        else:
            self.v = - rate #Vs-1 the rate at which the potential is swept over at
            self.oxidising_direction = False

        self.mu  = 0.0 #-0.031244092599793216 #phase
        self.freq = freq #Hz (0.1116/1564832 seconds per period insure data has even number of periods)
        print('frequency: ', self.freq)
        self.omega = 2.0*math.pi*self.freq # *self.T0 # dimensionless omega
        self.omega0 = 2.0*math.pi*self.freq# *self.T0 # dimensionless omega 
        self.epsilon = 0.0
        self.epsilon_r = 0.0
        self.row = 0.0 # 27.160770551*(self.I0/self.E0)# dimensionless uncompensated resistance
        self.tau0 = 1.0E-10 # initial estimate of surface coverage used for non_dim
        self.n = 1.0 # number of electron transfer's in one step (not this is 1 if no faradic signal is involved)

        # parameters for dimension removal
        if self.v != 0 or self.v != 0.0:
            self.E0 = (self.R*self.temp)/(self.F*self.n)
            # print('in 1 !!!!!!')
            self.T0 = (self.E0/self.v)
            self.I0 = (self.F*self.s*self.tau0)/self.T0
        else:
            # For protocols with no applied DCV current component

            self.E0 = (self.R*self.temp)/(self.F*self.n)
            self.T0 = (self.E0*self.freq)
            self.I0 = (self.F*self.s*self.tau0)/self.T0


        # defining faradaic parameters
        # self.kappa0 = 10000.0/self.T0
        # self.epsilon0 = ((startPotential + revPotential)/2.0)/self.E0
        # self.alpha = 0.5
        # self.zeta = 1.0
        # self.dtheta_X_dt = 0.0
        # self.theta_X = 1.0
        self.kappa0 = 100.0*self.T0
        self.epsilon0 = 0.0 # ((startPotential + revPotential)/2.0)/self.E0
        self.alpha = 0.5
        self.zeta = 7E-11/self.tau0
        self.dtheta_ox_dt = 0.0
        self.theta_ox = 0.0
        self.theta_ox_list = np.zeros(numberOfMeasurements, dtype = np.float64)
        self.dtheta_ox_dt_list = np.zeros(numberOfMeasurements, dtype = np.float64)

        # electode potential variables for epsilon
        self.epsilon_start = startPotential/self.E0 
        self.epsilon_reverse = revPotential/self.E0 

        self.deltaepislon = deltaepislon/self.E0 # V 
        #time interval
        if self.v != 0 or self.v != 0.0:
            # print('in 3 !!!!!!')
            self.startT = 0.0#specify in seconds
            self.revT =  abs((revPotential - startPotential)/(rateOfPotentialChange))#specify in seconds
            self.dimlessRevT = self.revT/self.T0#
            self.endT = self.revT*2.0
        self.timeStepSize = timeStepSize #self.revT/numberOfMeasurements # in seconds
        # print('time step size: ', timeStepSize)
        self.dimlessTimeStepSize = (self.timeStepSize)/self.T0
        # print('dimensionless time step size: ', self.dimlessTimeStepSize)

        self.i = np.zeros(numberOfMeasurements, dtype = np.float64)
        self.episilon_without_drop = np.zeros(numberOfMeasurements, dtype = np.float64)
        self.episilon_IR_drop_componenet = np.zeros(numberOfMeasurements, dtype = np.float64)

        self.I_inital = inital_current

        self.protocol = typed.Dict.empty(*kv_ty)
        self.solver_option = typed.Dict.empty(*kv_ty)

        # capacitance parameters for forward sweep
        self.gamma0 = 0.0
        self.gamma1 = 0.0
        self.gamma2 = 0.0
        self.gamma3 = 0.0

        # capacitance parameters for reverse sweep
        self.gamma4 = 0.0
        self.gamma5 = 0.0
        self.gamma6 = 0.0
        self.gamma7 = 0.0

        # capacitance parameters for 2d poly
        self.gamma8 = 0.0
        self.gamma9 = 0.0


        self.permittivityFree = 8.85418782E-12*10**2 # m-3 kg-1 s4 A2 i.e F/M converting to per cM
        self.boltzmann = 1.38064852E-23 #m2 kg s-2 K-1

        # constant if using hemholtz model
        self.helmholtzConstant = 0.0

        # gouy-chapman constants
        self.Z = Z # charge on ion storing capactiance
        self.dielectricConstant = 0.0 # dielectric constant of the medium
        self.numberConc = numberConc # number concentration of the ion
        self.pzc = 0.0 # point of zero charge
        self.electrostaticPotential0 = 0.0 # electrostatic potential at X=0 i.e next to electrode
        self.gouyChapman_nd = 0.0 # non_dimensionalisation constant for gouy_chapman model

        self.iterations = 0

        self.appliedPotential = np.zeros(numberOfMeasurements, dtype = np.float64)
        self.drivAppliedPotential = np.zeros(numberOfMeasurements, dtype = np.float64)
        self.appliedPotentialDCVComponetent= np.zeros(numberOfMeasurements, dtype = np.float64)

        self.modelling_faradaic_current = False
        self.modeling_experimental_data = False
        self.fit_alpha = False

    def set_experimental_protocol(self, protocol:str = 'FTACV'):
        '''setter function for the applied voltage protocol
           for use in the simulation.
           Currently the following experimental protocols are allowed:
            - FTACV
            - DCV
            - EIS
            - PSV
        '''
        if protocol == 'FTACV' or protocol == 'DCV' or protocol == 'EIS' or protocol == 'PSV':
            # if not using JIT class remove the [0]
            self.protocol[0] = protocol
        else:
            raise Exception('Invalid applied voltage protocol, please choice from: FTACV, DCV, EIS, or PSV')


    def set_solver_option(self, solver_option:str = 'my_newton_raphson'):
        '''setter function for the applied voltage protocol
           for use in the simulation.
           Currently the following experimental protocols are allowed:
            - FTACV
            - DCV
            - EIS
            - PSV
        '''
        if solver_option == 'my_newton_raphson' or solver_option == 'scipy_newton_raphson' or solver_option == 'scipy_brent' or solver_option ==  'my_riddlers':
            # if not using JIT class remove the [0]
            self.solver_option[0] = solver_option
        else:
            raise Exception('Invalid solver option, please choice from: my_newton_raphson, scipy_newton_raphson, scipy_brent,  my_riddlers, or ....')
        
    def find_epsilon(self, time_: float, index: int, IR_drop: boolean = True):
        ''' finding epsilon and epsilon_r
            as described in ref [1]
            Currently this works for the following experimental protocols:
            - r-FTACV
            - DCV
            - EIS
            - PSV
        '''
        
        # defining the experimental voltage protocol to use
        # if not using JIT class remove the [0]
        protocol = self.protocol[0]

        dc, ac = False, False
        if protocol == 'FTACV' or protocol == 'DCV':
            dc = True
        if protocol == 'FTACV' or protocol == 'EIS' or protocol == 'PSV':
            ac = True

        # adding ac component
        if ac is True:
            self.epsilon = (self.deltaepislon)*math.sin(self.omega*time_ + self.mu)
        else:
            self.epsilon = 0.0

        # adding dc component
        if dc is True:
            if abs(self.T0) == self.T0:
                if time_ < self.dimlessRevT:
                    # epsilon before dc current reversal
                    self.epsilon += self.epsilon_start + time_
                    self.appliedPotentialDCVComponetent[index] = (self.epsilon_start + time_)*self.E0

                elif time_ >= self.dimlessRevT:
                    # epsilon after dc current reversal
                    self.epsilon += self.epsilon_reverse - time_ + self.dimlessRevT
                    self.appliedPotentialDCVComponetent[index] = (self.epsilon_reverse - time_ + self.dimlessRevT)*self.E0
            else:
                # taking into account changes in logic if T0 is negative 
                if time_ > self.dimlessRevT:
                    # epsilon before dc current reversal
                    self.epsilon += self.epsilon_start + time_
                    self.appliedPotentialDCVComponetent[index] = (self.epsilon_start + time_)*self.E0
                    # print('logic 3')

                elif time_ <= self.dimlessRevT:
                    # epsilon after dc current reversal
                    self.epsilon += self.epsilon_reverse - time_ + self.dimlessRevT
                    self.appliedPotentialDCVComponetent[index] = (self.epsilon_reverse - time_ + self.dimlessRevT)*self.E0
                    # print('logic 4')

        # adding ohmic drop
        self.appliedPotential[index] = self.epsilon*self.E0
        if IR_drop is True:
            self.epsilon_r = self.epsilon - self.row*self.i[int(index-1)]
            self.episilon_without_drop[index] =  self.epsilon
            self.episilon_IR_drop_componenet[index] = self.row*self.i[int(index-1)]
        else:
            self.epsilon_r = self.epsilon



    def find_dev_epsilon(self, i_n, t, i_n1):
        '''helper function to calaculate depsilon_rdt and d2epsilon_rdidt.
            Currently this works for the following experimental protocols:
            - FTACV
            - DCV
            - EIS
            - PSV
        '''
        
        # defining the experimental voltage protocol to use
        # if not using JIT class remove the [0]
        protocol = self.protocol[0]
        dc, ac = False, False
        if protocol == 'FTACV' or protocol == 'DCV':
            dc = True
        if protocol == 'FTACV' or protocol == 'EIS' or protocol == 'PSV':
            ac = True

        # adding ac component
        if ac is True:
            depsilon_rdt = self.omega*self.deltaepislon*math.cos(self.omega*t + self.mu)
        else:
            depsilon_rdt = 0.0
        

         # adding dc component
        if dc is True:
            if abs(self.T0) == self.T0:
                if t < self.dimlessRevT:
                    # epsilon before dc current reversal
                    # adding dc component
                    depsilon_rdt += 1.0
                elif t >= self.dimlessRevT:
                    # epsilon after dc current reversal
                    # adding dc component
                    depsilon_rdt += -1.0
            else:
                # taking into account changes in logic if T0 is negative 
                if t > self.dimlessRevT:
                    # epsilon before dc current reversal
                    # adding dc component
                    depsilon_rdt += 1.0
                elif t <= self.dimlessRevT:
                    # epsilon after dc current reversal
                    # adding dc component
                    depsilon_rdt += -1.0 

        # adding ohmic drop
        return depsilon_rdt - (self.row*(i_n1 - i_n ))/self.dimlessTimeStepSize
    
    def find_dtheta_X_dt(self):
        '''
            finding  dtheta_X/dt i.e solving stoichiometry and using the Butler-volmer model of electron-transfer
            dtheta_X/dt = k0*((1.0 - theta_X)*exp((1.0 - alpha)*(epsilon_r - epsilon0))
                - theta_X*exp(-alpha*(epsilon_r - epsilon0)))
        '''
        self.dtheta_ox_dt = self.kappa0*((1.0 - self.theta_ox)*math.exp((1.0 - self.alpha)*(self.epsilon_r - self.epsilon0))
                - self.theta_ox*math.exp(-self.alpha*(self.epsilon_r - self.epsilon0)))
        
    def backwards_euler(self):
        '''
        applies the backwards euler method to theta_X
        as f(thetax(n+1),time(n+1))
        '''
        A = math.exp((1.0 - self.alpha)*(self.epsilon_r - self.epsilon0))
        B = math.exp(-self.alpha*(self.epsilon_r - self.epsilon0))
       
        denominator = (1/self.dimlessTimeStepSize) + A*self.kappa0 + B*self.kappa0
        numerator = A*self.kappa0 +  (self.theta_ox/self.dimlessTimeStepSize)

        self.theta_ox = numerator/denominator

    def current_function(self, i_n1, t, i_n, capOption:str):
        ''' 
        def current_function(self, i_n, t, i_n1, capOption:str):
        solving the current function described in ref [1] rearraged to equal zero 
        note the backwards euler is used for di/dT
        '''

        # input protocols this might be best done with a helper function
        depsilon_rdt = self.find_dev_epsilon(i_n, t, i_n1)

        if capOption == 'doublePoly':

            if abs(self.T0) == self.T0:
                if t < self.dimlessRevT:
                    # capacitance polynomial before dc current reversal
                    gamma0 = self.gamma0
                    gamma1 = self.gamma1
                    gamma2 = self.gamma2
                    gamma3 = self.gamma3
                elif t >= self.dimlessRevT:
                    # capacitance polynomial after dc current reversal
                    gamma0 = self.gamma4
                    gamma1 = self.gamma5
                    gamma2 = self.gamma6
                    gamma3 = self.gamma7
            else:
                # taking into account changes in logic if T0 is negative 
                if t > self.dimlessRevT:
                    # capacitance polynomial before dc current reversal
                    gamma0 = self.gamma0
                    gamma1 = self.gamma1
                    gamma2 = self.gamma2
                    gamma3 = self.gamma3
                elif t <= self.dimlessRevT:
                    # capacitance polynomial after dc current reversal
                    gamma0 = self.gamma4
                    gamma1 = self.gamma5
                    gamma2 = self.gamma6
                    gamma3 = self.gamma7

            I_cap = (gamma0 + gamma1*self.epsilon_r + gamma2*math.pow(self.epsilon_r, 2.0 ) + gamma3*math.pow(self.epsilon_r, 3.0 ))*depsilon_rdt

        elif capOption == 'poly':
            
            gamma0 = self.gamma0
            gamma1 = self.gamma1
            gamma2 = self.gamma2
            gamma3 = self.gamma3
            gamma4 = self.gamma4
            gamma5 = self.gamma5
        
            I_cap = (gamma0 + gamma1*self.epsilon_r + gamma2*math.pow(self.epsilon_r, 2.0 ) + gamma3*math.pow(self.epsilon_r, 3.0 )
                     + gamma4*math.pow(self.epsilon_r, 4.0 ) + gamma5*math.pow(self.epsilon_r, 5.0 ))*depsilon_rdt
    
        elif capOption == 'helmholtz':
            
            I_cap = self.helmholtzConstant*depsilon_rdt

        elif capOption == 'gouyChapman':
            # FIXME: temporary electrostaticPotential0
            self.electrostaticPotential0 = (self.epsilon_r - self.pzc)# /self.E0 # is the problem with this???? 
            e = 1.60217662E-19
            A = math.sqrt((2.0*math.pow(self.Z,2.0)*math.pow(e,2.0)*self.dielectricConstant*self.permittivityFree*self.numberConc)/(self.boltzmann*self.temp))
            # if self.iterations < 1:
            #     print('self.epsilon_r: ', self.epsilon_r)
            #     print('self.pzc: ', self.pzc/self.E0)
            #     print('A: ', A)
            #     print('Z: ', self.Z)
            #     print('e: ', e)
            #     print('electrstatic: ', self.electrostaticPotential0)
            #     print('boltzmann: ', self.boltzmann)
            #     print('temp: ', self.temp)
            #     print('X: ', (self.Z*e*self.electrostaticPotential0)/(2.0*self.boltzmann*self.temp))
            #     print('cosh: ', math.cosh(((self.Z)*(e)*self.electrostaticPotential0/(2.0*self.boltzmann*self.temp))))
            #     self.iterations += 1
            A = A*math.cosh((self.Z*e*self.electrostaticPotential0)/(2.0*self.boltzmann*self.temp))
            
            I_cap = self.gouyChapman_nd*A*depsilon_rdt
        
        elif capOption == 'gouyChapman_dilute_solution':
            pass
        
        elif capOption == 'stern':
            
            self.helmholtzConstant
            # FIXME: temporary electrostaticPotential0
            self.electrostaticPotential0 = (self.epsilon_r - self.pzc)/self.E0
            e = 1.60217662E-19
            C_D = math.sqrt((2.0*math.pow(self.Z,2.0)*math.pow(e,2.0)*self.dielectricConstant*self.permittivityFree*self.numberConc)/(self.boltzmann*self.temp))
            C_D = C_D*self.gouyChapman_nd
            C_D = C_D*math.cosh((self.Z*e*self.electrostaticPotential0)/(2.0*self.boltzmann*self.temp))
            C_d_inv = 1.0/self.helmholtzConstant + 1.0/C_D
            c_d = 1.0/C_d_inv
            # c_d = self.helmholtzConstant*C_D*(1.0/(self.helmholtzConstant+C_D))
            I_cap = c_d*depsilon_rdt
        
        elif capOption == '2dPoly':
            
            gamma0 = self.gamma0
            gamma1 = self.gamma1
            gamma2 = self.gamma2
            gamma3 = self.gamma3
            gamma4 = self.gamma4
            gamma5 = self.gamma5
            # gamma6 = self.gamma6
            # gamma7 = self.gamma7
            # gamma8 = self.gamma8
            # gamma9 = self.gamma9

            I_cap =(gamma0 + gamma1*self.epsilon_r + gamma2*depsilon_rdt + gamma3*depsilon_rdt*self.epsilon_r
                    + gamma4*math.pow(self.epsilon_r, 2.0 )+ gamma5*math.pow(depsilon_rdt, 2.0 ))*depsilon_rdt

            # return (-i_n1 + (gamma0 + gamma1*self.epsilon_r + gamma2*depsilon_rdt + gamma3*depsilon_rdt*self.epsilon_r
            #         )*depsilon_rdt)
        
            # return (-i_n1 + (gamma0 + gamma1*self.epsilon_r + gamma2*depsilon_rdt + gamma3*depsilon_rdt*self.epsilon_r
            #         + gamma4*math.pow(self.epsilon_r, 2.0 )+ gamma5*math.pow(depsilon_rdt, 2.0 )
            #         + gamma6*math.pow(self.epsilon_r, 2.0 )*depsilon_rdt  + gamma7*math.pow(self.epsilon_r, 3.0 )
            #         + gamma8*math.pow(depsilon_rdt, 2.0 )*self.epsilon_r + gamma9*math.pow(depsilon_rdt, 3.0 ))*depsilon_rdt)

        else:
            raise Exception('Invalid current option selected please choice from: doublePoly, poly, helmholtz, gouyChapman, gouyChapman_dilute_solution, stern, or 2dPoly')
        
        if self.modelling_faradaic_current == True:
            I_f = self.n*self.zeta*(self.dtheta_ox_dt)
            return -i_n1  + I_cap + I_f
        else:
            return -i_n1 + I_cap

    
    def deriv_current_function(self, i_n1, t, i_n, capOption:str):
        ''' def deriv_current_function(self, i_n, t, i_n1, capOption:str):
        solving the differential WRT i current function described in ref [1] rearraged to equal zero 
            note the backwards euler is used for di/dT
        '''

        # input protocalls this might be best done with a helper function

        # calculating depsilon_rdt
        depsilon_rdt = self.find_dev_epsilon(i_n, t, i_n1)

        if capOption == 'doublePoly':
            if abs(self.T0) == self.T0:
                if t < self.dimlessRevT:
                    # capacitance polynomial before dc current reversal
                    gamma0 = self.gamma0
                    gamma1 = self.gamma1
                    gamma2 = self.gamma2
                    gamma3 = self.gamma3
                elif t >= self.dimlessRevT:
                    # capacitance polynomial after dc current reversal
                    gamma0 = self.gamma4
                    gamma1 = self.gamma5
                    gamma2 = self.gamma6
                    gamma3 = self.gamma7
            else:
                # taking into account changes in logic if T0 is negative 
                if t > self.dimlessRevT:
                    # capacitance polynomial before dc current reversal
                    gamma0 = self.gamma0
                    gamma1 = self.gamma1
                    gamma2 = self.gamma2
                    gamma3 = self.gamma3
                elif t <= self.dimlessRevT:
                    # capacitance polynomial after dc current reversal
                    gamma0 = self.gamma4
                    gamma1 = self.gamma5
                    gamma2 = self.gamma6
                    gamma3 = self.gamma7

            d2epsilon_rdidt = -self.row/self.dimlessTimeStepSize

            return(-1.0 + (-gamma1*self.row - 2.0*gamma2*self.row*(self.epsilon - self.row*i_n1) - 3.0*gamma3*self.row*math.pow((self.epsilon - self.row*i_n1),2.0))*depsilon_rdt 
                    + (gamma0 + gamma1*self.epsilon_r + gamma2*math.pow(self.epsilon_r, 2.0 ) + gamma3*math.pow(self.epsilon_r, 3.0 ))*d2epsilon_rdidt)

        elif capOption == 'poly':
            # capacitance polynomial parameters
            gamma0 = self.gamma0
            gamma1 = self.gamma1
            gamma2 = self.gamma2
            gamma3 = self.gamma3
            gamma4 = self.gamma4
            gamma5 = self.gamma5

            d2epsilon_rdidt = -self.row/self.dimlessTimeStepSize

            return(-1.0 + (-gamma1*self.row - 2.0*gamma2*self.row*(self.epsilon - self.row*i_n1) - 3.0*gamma3*self.row*math.pow((self.epsilon - self.row*i_n1),2.0))*depsilon_rdt 
                    + (gamma0 + gamma1*self.epsilon_r + gamma2*math.pow(self.epsilon_r, 2.0 ) + gamma3*math.pow(self.epsilon_r, 3.0 )
                       + gamma4*math.pow(self.epsilon_r, 4.0 ) + gamma5*math.pow(self.epsilon_r, 5.0 ))*d2epsilon_rdidt)

        elif capOption == 'helmholtz':

            d2epsilon_rdidt = -self.row/self.dimlessTimeStepSize
            return(-1.0 + self.helmholtzConstant*d2epsilon_rdidt)

        elif capOption == 'gouyChapman':
            
            # FIXME: temporary electrostaticPotential0

            # self.electrostaticPotential0 = (self.epsilon_r - self.pzc) # /self.E0
            self.electrostaticPotential0 = (self.epsilon_r - self.pzc) # /self.E0

            d2epsilon_rdidt = -self.row/self.dimlessTimeStepSize
            e = 1.60217662E-19 

            A = math.sqrt((2.0*math.pow(self.Z,2.0)*math.pow(e,2.0)*self.dielectricConstant*self.permittivityFree*self.numberConc)/(self.boltzmann*self.temp))
            B = A*math.cosh((self.Z*e*self.electrostaticPotential0)/(2.0*self.boltzmann*self.temp))
            # C = A*((self.Z)*(e)/(2.0*self.boltzmann*self.temp))
            
            return(-1.0 + self.gouyChapman_nd*B*d2epsilon_rdidt)
            #+self.gouyChapman_nd*C*math.sinh(((self.Z)*(math.e)*self.electrostaticPotential0/(2.0*self.boltzmann*self.temp)

        elif capOption == 'stern':

            d2epsilon_rdidt = -self.row/self.dimlessTimeStepSize
            
            self.helmholtzConstant
            # FIXME: temporary electrostaticPotential0
            self.electrostaticPotential0 = (self.epsilon_r - self.pzc)/self.E0
            e = 1.60217662E-19
            C_D = math.sqrt((2.0*math.pow(self.Z,2.0)*math.pow(e,2.0)*self.dielectricConstant*self.permittivityFree*self.numberConc)/(self.boltzmann*self.temp))
            C_D = C_D*self.gouyChapman_nd
            C_D = C_D*math.cosh((self.Z*e*self.electrostaticPotential0)/(2.0*self.boltzmann*self.temp))
            C_d_inv = 1.0/self.helmholtzConstant + 1.0/C_D
            c_d = 1.0/C_d_inv
            # c_d = self.helmholtzConstant*C_D*(1.0/(self.helmholtzConstant+C_D))
            return(-1.0 + c_d*d2epsilon_rdidt)

        elif capOption == '2dPoly':
            
            gamma0 = self.gamma0
            gamma1 = self.gamma1
            gamma2 = self.gamma2
            gamma3 = self.gamma3
            gamma4 = self.gamma4
            gamma5 = self.gamma5
            # gamma6 = self.gamma6
            # gamma7 = self.gamma7
            # gamma8 = self.gamma8
            # gamma9 = self.gamma9

            d2epsilon_rdidt = -self.row/self.dimlessTimeStepSize
        
            # return (-1.0 + (gamma0 + gamma1*self.epsilon_r + gamma2*depsilon_rdt + gamma3*depsilon_rdt*self.epsilon_r
            #         + gamma4*math.pow(self.epsilon_r, 2.0 )+ gamma5*math.pow(depsilon_rdt, 2.0 )
            #         + gamma6*math.pow(self.epsilon_r, 2.0 )*depsilon_rdt  + gamma7*math.pow(self.epsilon_r, 3.0 )
            #         + gamma8*math.pow(depsilon_rdt, 2.0 )*self.epsilon_r + gamma9*math.pow(depsilon_rdt, 3.0 ))*d2epsilon_rdidt)
            

            return(-1.0 + (gamma0 + gamma1*self.epsilon_r + gamma2*depsilon_rdt + gamma3*depsilon_rdt*self.epsilon_r
                    + gamma4*math.pow(self.epsilon_r, 2.0 )+ gamma5*math.pow(depsilon_rdt, 2.0 ))*d2epsilon_rdidt)

            # return (-1.0 + (gamma0 + gamma1*self.epsilon_r + gamma2*depsilon_rdt + gamma3*depsilon_rdt*self.epsilon_r
            #         )*d2epsilon_rdidt)

        else:
            raise Exception('Invalid current option selected please choice from: doublePoly, poly, helmholtz, gouyChapman, stern, or 2dPoly')
        
    def analytical_current_solution(self, time_, index: int, capOption: str):
        '''implementation of the newton-raphson method to solve for the current i at the next time step
        '''

        solver_option = self.solver_option[0]
        if solver_option == 'my_newton_raphson':
            x0 = self.i[int(index-1)]
            x1 = self.i[int(index-1)]

            if time_ == 0.0 or time_ == -0.0:
                print('inital didT: ', self.deriv_current_function(x1, time_, x0, capOption))

            h = self.current_function(x1, time_, x0, capOption)/self.deriv_current_function(x1, time_, x0, capOption)

            iterations = 0
            max_iterations = 1E6
            tolerance = 1E-8
            # max_iterations = 1E6
            # tolerance = 1E-20
            check_1 = 2
            check_2 = 1
            # if x1>0:
            #     while abs(h) >= tolerance and iterations <= max_iterations and abs(x1-(h/x1))>tolerance*x1:

            #         h = self.current_function(x1, time, x0, capOption)/self.deriv_current_function(x1, time, x0, capOption)

            #         # x(i+1) = x(i) - f(x) / f'(x) 

            #         x1 = x1 - h
            #         iterations = iterations +1

            #         check_2 = tolerance*x1*x1

            #         check_1 = abs(x1*x1-h)
            # else:
            while abs(h) >= tolerance and iterations <= max_iterations:

                h = self.current_function(x1, time_, x0, capOption)/self.deriv_current_function(x1, time_, x0, capOption)

                # x(i+1) = x(i) - f(x) / f'(x) 

                x1 = x1 - h
                iterations = iterations +1

            self.i[index] = x1

        # elif solver_option == 'scipy_newton_raphson':

        #     x0 = self.i[int(index-1)]
        #     x1 = self.i[int(index-1)]

        #     params = (time_, x0, capOption)
        #     max_iterations = int(1E6)
        #     tol = 1.48e-8

        #     # print('brent')

        #     root = scipy.optimize.newton(self.current_function, x0, fprime=self.deriv_current_function, args=params, tol=tol, maxiter=max_iterations, fprime2=None, x1=None, rtol=0.0, full_output=False, disp=True)
        #     self.i[index] = root


        # elif solver_option == 'scipy_brent':

        #     print('minimize_scalar:')

        #     x0 = self.i[int(index-1)]
        #     x1 = self.i[int(index-1)]
        #     print('x0: ', x0)

        #     params = (time, x0, capOption)
        #     max_iterations = int(1E6)

        #     # (func, xa=0.0, xb=1.0, args=(), grow_limit=110.0, maxiter=1000)[source]#
        #     # xa, xb, xc, fa, fb, fc, funcalls = scipy.optimize.bracket(self.current_function,  xa=0.0, xb=1.0, grow_limit=x0*1.2, args=params, maxiter=max_iterations)

        #     # # print('bracket: ', xa, xb, xc)

        #     # res = scipy.optimize.minimize_scalar(self.current_function, bracket=(xa, xb, xc), bounds=None, args=params, method='brent', tol=None, options={'maxiter':max_iterations,})
        #     res = scipy.optimizse.minimize_scalar(self.current_function, bracket=(x0*0.8,x0*1.2), bounds=None, args=params, method='brent', tol=None, options={'maxiter':max_iterations,})
        #     print(res.fun)
        #     # print(res.x)

        #     print('done')
        #     self.i[index] = float64(res.fun)

        elif solver_option == 'my_riddlers':
            
            # def fun(i_n1, time_, x0, capOption):
            #     return i_n1 - self.current_function(i_n1, time_, x0, capOption)
            # def fun(i_n1, time_, x0, capOption):
            #     return self.current_function(i_n1, time_, x0, capOption)

            x0 = self.i[int(index-1)]
            bound_val = x0*0.1
            lower = x0-bound_val
            upper = x0+bound_val
            if lower > upper:
                upper = x0-bound_val
                lower = x0+bound_val
            # interval = (x0+bound_val, x0-bound_val)
            # riddler_solver = riddlers_method(starting_interval= interval, function=fun, time_ = time_, in0 = x0, capOption = capOption)
            root = self.riddler_iteration(lower,upper,time_=  time_, in0 = x0, capOption = capOption)
            self.i[index] = root

        # elif solver_option == 'my_ITP':
        #     pass

    def propose_x1_and_d(self, x0, x2):
        '''
        d = x2-x1 = x1-x0
        therefore x1 is the midpoint of the
        proposed interval

        return x1, d
        '''

        x1 = (x0+x2)*0.5
        d = x2-x1
        return x1, d

    def sign(self, x):
        if (x<0):
            return -1
        elif (x == 0):
            return 0
        elif (x>0):
            return 1
        
    def regulu_falsi(self,d,x0,x1,x2, F_x0, F_x1, F_x2):
        '''
        x3 = x1 + d*((F(x1)/F(x0))/sqrt((F(x1)/F(x0))^2 - (F(x2)/F(x0))))
        '''

        F_x1_F_x0 = F_x1/F_x0
        F_x2_F_x0 = F_x2/F_x0

        denominator = math.sqrt(abs(math.pow(F_x1_F_x0,2.0) - F_x2_F_x0))

        if denominator == 0.0:
            raise Exception('denominator of the regulu falsi in riddler\'s method is zero, aborting.....')

        else: 
            x3 = x1 + d*(F_x1_F_x0/denominator)

            return x3

        # sign = self.sign(F_x0)

        # denominator = math.sqrt(abs(math.pow(F_x1,2.0) - F_x0*F_x2))

        # if denominator == 0.0:
        #     raise Exception('denominator of the regulu falsi in riddler\'s method is zero, aborting.....')

        # else: 
        #     x3 = x1 + (sign*F_x1*d)/denominator

        #     return x3
    
    def riddler_iteration(self,x0,x2, time_, in0, capOption):
        '''
        A single iteration of riddler's method
        '''

        # tic = time.perf_counter()

        iteration = 0
        tolerance = 1.48e-8 # scipy tends to use 1.48e-08
        max_iter =1E6

        # TODO:FIXME:TODO:
        # introduce a check as if either bracket is the root 
        # there is no point continuing with the algorithm.

        # function_to_solve = self.current_function
        F_x0 = self.current_function(x0, time_, in0 , capOption)

        if tolerance > abs(F_x0):
            # print('final root: ', x0)
            # print('F_x0: ', F_x0)
            # print ('iteraions: ', iteration)
            # print ('less than tol: True')

            return x0

        F_x2 = self.current_function(x2, time_, in0 , capOption)

        if tolerance > abs(F_x2):
            # print('final root: ', x2)
            # print('F_x0: ', F_x2)
            # print ('iteraions: ', iteration)
            # print ('less than tol: True')

            return x0

        # if (self.function_to_solve(x0, self.time_, self.in0 , self.capOption)*self.function_to_solve(x2, self.time_, self.in0 , self.capOption)<=0.0):

        # inital F_x3
        F_x3 = tolerance*10.0
        while iteration<max_iter and tolerance < abs(F_x3):
            
            #  defining mid-point and distance between 
            # mid-point and bracket edges
            x1, d = self.propose_x1_and_d(x0, x2)

            # evaluating function at 3 bracket points
            F_x0 = self.current_function(x0, time_, in0 , capOption)
            F_x1 = self.current_function(x1, time_, in0 , capOption)
            F_x2 = self.current_function(x2, time_, in0 , capOption)

            # proposing new point using the regulu_falsi
            x3 = self.regulu_falsi(d,x0,x1,x2, F_x0, F_x1, F_x2)


            F_x3 = self.current_function(x3, time_, in0 , capOption)

            # print ('iteraion: ', iteration)
            # print ('x3: ', x3)
            # print ('F(x3): ', F_x3)

            iteration += 1

            # updating bracket
            # TODO: currently only updating one side based on this,
            #  should also update other side based on x1
            if F_x0>0:
                if F_x3 < 0:
                    x2=x3
                    if self.sign(x0) == self.sign(x1):
                        x0 = x1
                else:
                    x0=x3
                    if self.sign(x1) == self.sign(x2):
                        x2 = x1
            else:
                if F_x3 > 0:
                    x2=x3
                    if self.sign(x0) == self.sign(x1):
                        x0 = x1
                else:
                    x0=x3 
                    if self.sign(x1) == self.sign(x2):
                        x2 = x1

        # toc = time.perf_counter()

        # print(f"root found in {toc - tic:0.10f} seconds")

        # print('\n' +10*'*')
        # print('final root: ', x3)
        # print('F(x3): ', F_x3)
        # print ('iteraions: ', iteration)
        tol_meet = False
        if tolerance > abs(F_x3):
            tol_meet = True
        # print ('less than tol: ', tol_meet)

        return x3
        
        # else:
        #     raise Exception('starting bracket invalid, f(x0)*f(x2)<0 i.e have opposite signs. ')


    def set_capacitance_params(self, cap_params, capOption: str = 'doublePoly'):
        '''
        takes a list of capasiance parameters and sets these for the model
        :param: cap_params = [gamma0, gamma1, gamma2, gamma3, gamma4,
                              gamma5, gamma6, gamma7, row omega, mew]
        '''
        #FIXME: make me more elegant!!!! this is not a nice way of setting parameters.

        protocol = self.protocol[0]
        if capOption == 'doublePoly':
            non_dimensiosation_constant = self.E0*self.s/(self.T0*self.I0)
            self.gamma0 = (cap_params[0]*non_dimensiosation_constant)
            self.gamma1 = (cap_params[1]*self.E0)*non_dimensiosation_constant
            self.gamma2 = (cap_params[2]*math.pow(self.E0,2.0))*non_dimensiosation_constant
            self.gamma3 = (cap_params[3]*math.pow(self.E0,3.0))*non_dimensiosation_constant
            self.gamma4 = (cap_params[4]*non_dimensiosation_constant)
            self.gamma5 = (cap_params[5]*self.E0)*non_dimensiosation_constant
            self.gamma6 = (cap_params[6]*math.pow(self.E0,2.0))*non_dimensiosation_constant
            self.gamma7 = (cap_params[7]*math.pow(self.E0,3.0))*non_dimensiosation_constant
            self.row = cap_params[8]*(self.I0/self.E0)
            if protocol != 'DCV':
                self.omega = cap_params[9]*2.0*math.pi*self.T0
                if protocol == 'PSV':
                    self.mu = cap_params[10] + (3/2)*math.pi
                else:
                    self.mu = cap_params[10]

        elif capOption == 'poly':
            non_dimensiosation_constant = self.E0*self.s/(self.T0*self.I0)
            self.gamma0 = (cap_params[0]*non_dimensiosation_constant)
            self.gamma1 = (cap_params[1]*self.E0)*non_dimensiosation_constant
            self.gamma2 = (cap_params[2]*math.pow(self.E0,2.0))*non_dimensiosation_constant
            self.gamma3 = (cap_params[3]*math.pow(self.E0,3.0))*non_dimensiosation_constant
            self.gamma4 = (cap_params[4]*math.pow(self.E0,4.0))*non_dimensiosation_constant
            self.gamma5 = (cap_params[5]*math.pow(self.E0,5.0))*non_dimensiosation_constant
            self.row = cap_params[6]*(self.I0/self.E0)
            if protocol != 'DCV':
                self.omega = cap_params[7]*2.0*math.pi*self.T0
                if protocol == 'PSV':
                    self.mu = cap_params[8] + (3/2)*math.pi
                else:
                    self.mu = cap_params[8]

            # print('parameters in solver: ', cap_params)
            # print('self.gamma0: ', self.gamma0)
            # print('self.gamma1: ', self.gamma1)
            # print('self.gamma2: ', self.gamma2)
            # print('self.gamma3: ', self.gamma3)


        elif capOption == 'helmholtz':
            non_dimensiosation_constant = self.E0*self.s/(self.T0*self.I0)
            # permittivitySpaceRatio = cap_params[0] # (permittivityMedium/interplateSpace)*self.permittivityFree
            # self.helmholtzConstant = permittivitySpaceRatio*self.permittivityFree
            # self.helmholtzConstant = self.helmholtzConstant*non_dimensiosation_constant
            self.helmholtzConstant = (cap_params[0]*non_dimensiosation_constant)
            self.row = cap_params[1]*(self.I0/self.E0)
            # print('protocol in wrapped class: ', self.protocol)
            if protocol != 'DCV':
                self.omega = cap_params[2]*2.0*math.pi*self.T0
                if protocol == 'PSV':
                    self.mu = cap_params[3] + (3/2)*math.pi
                else:
                    self.mu = cap_params[3]

        elif capOption == 'gouyChapman':
            # self.gouyChapman_nd = self.E0*self.s*(10**4)/(self.T0*self.I0)
            self.gouyChapman_nd = self.E0*self.s/(self.T0*self.I0)
            self.dielectricConstant = cap_params[0] # dielectric constant of the medium
            # self.numberConc = cap_params[1] # number concentration of the ion
            self.pzc = cap_params[1]*self.E0 # point of zero charge of the electrode
            self.row = cap_params[2]*(self.I0/self.E0)
            if protocol != 'DCV':
                self.omega = cap_params[3]*2.0*math.pi*self.T0
                if protocol == 'PSV':
                    self.mu = cap_params[4] + (3/2)*math.pi
                else:
                    self.mu = cap_params[4]

        elif capOption == 'stern':
            self.gouyChapman_nd = self.E0*self.s/(self.T0*self.I0)
            self.dielectricConstant = cap_params[0] # dielectric constant of the medium
            interplateSpace = cap_params[1]
            # self.numberConc = cap_params[2] # number concentration of the ion
            self.pzc = cap_params[2]*self.E0 # point of zero charge of the electrode
            self.row = cap_params[3]*(self.I0/self.E0)
            self.helmholtzConstant = ((self.dielectricConstant*self.permittivityFree)/interplateSpace)
            self.helmholtzConstant = self.helmholtzConstant*self.gouyChapman_nd
            if protocol != 'DCV':
                self.omega = cap_params[4]*2.0*math.pi*self.T0
                if protocol == 'PSV':
                    self.mu = cap_params[5] + (3/2)*math.pi
                else:
                    self.mu = cap_params[5]
        
        elif capOption == '2dPoly':

            non_dimensiosation_constant = self.E0*self.s/(self.T0*self.I0)
            no_dim_diff = self.E0/self.T0
            self.gamma0 = cap_params[0]*non_dimensiosation_constant
            self.gamma1 = cap_params[1]*self.E0*non_dimensiosation_constant
            self.gamma2 = cap_params[2]*no_dim_diff*non_dimensiosation_constant
            self.gamma3 = cap_params[3]*self.E0*no_dim_diff*non_dimensiosation_constant
            self.gamma4 = cap_params[4]*math.pow(self.E0,2.0)*non_dimensiosation_constant
            self.gamma5 = cap_params[5]*math.pow(no_dim_diff,2.0)*non_dimensiosation_constant
            # self.gamma6 = cap_params[6]*math.pow(self.E0,2.0)*no_dim_diff*non_dimensiosation_constant
            # self.gamma7 = cap_params[7]*math.pow(self.E0,3.0)*non_dimensiosation_constant
            # self.gamma8 = cap_params[8]*math.pow(no_dim_diff,2.0)*self.E0*non_dimensiosation_constant
            # self.gamma9 = cap_params[9]*math.pow(no_dim_diff,3.0)*non_dimensiosation_constant
            self.row = cap_params[6]*(self.I0/self.E0)
            if protocol != 'DCV':
                self.freq = cap_params[7]
                self.omega = cap_params[7]*2.0*math.pi*self.T0
                if protocol == 'PSV':
                    self.mu = cap_params[8] + (3/2)*math.pi
                else:
                    self.mu = cap_params[8]


        else:
            raise Exception('Invalid current option selected please choice from: doublePoly, poly, helmholtz, gouyChapman, stern, or 2dPoly, 4th_order_poly, or 5th_order_poly')
        
    
    def set_faradaic_params(self, parameters):
        '''
         kappa0, epsilon0, freq, mew, resistance, surface_coverage
        '''

        protocol = self.protocol[0]

        self.kappa0 = parameters[0]*self.T0
        self.epsilon0 = parameters[1]/self.E0
        # self.freq =  self.freq #parameters[2]
        # self.omega =  parameters[2]*2.0*math.pi*self.T0
        # self.omega =  self.freq*2.0*math.pi*self.T0
        if protocol == 'PSV':
            # phase is demnsionless
            self.mu = parameters[2] + (3/2)*math.pi
            # self.mu =0.0+ (3/2)*math.pi
        else:
            # phase is demnsionless
            self.mu = parameters[2]
            # self.mu =0.0
            # self.mu = parameters[2]

        # self.row = parameters[3]*(self.I0/self.E0)
        # self.row = parameters[3]*(self.I0/self.E0)
        # self.row = 2.74802881816768547e+01*(self.I0/self.E0)
        self.zeta = parameters[3]/self.tau0
        # self.zeta = parameters[4]/self.tau0
        # self.zeta = 4e-11/self.tau0
        # self.zeta = parameters[5]/self.tau0 #(self.F*self.s*parameters[5]/(self.T0*self.I0)) # surface coverage
        if self.fit_alpha == True:
            self.alpha = parameters[-1]



    def solve(self, times: float64, capOption : str = 'doublePoly'):
        '''Steps through and solves the system
        '''
        # print('In base python simulator')
        t = times[1:]
        # non dimensioanless times
        t=t/self.T0

        if self.oxidising_direction ==  True:
            # startPotential < revPotential:
            self.theta_ox = 0.0 # i.e all of molecule starts in reduced state.
        else:
            self.theta_ox = 1.0 # i.e all of molecule starts in oxidised state.

        self.appliedPotential[0] = self.epsilon_start*self.E0
        self.appliedPotentialDCVComponetent[0] = self.epsilon_start*self.E0

        # specifying initial value of the current
        if self.modeling_experimental_data == True:
            self.i[0] = self.I_inital/self.I0
            self.theta_ox_list[0] = self.theta_ox


            self.dtheta_ox_dt_list[0] = self.dtheta_ox_dt

        else:
            # IR_drop = False
            self.find_epsilon(time_ = times[0], index = 0, IR_drop=False)
            self.backwards_euler()
            self.find_dtheta_X_dt()
            self.theta_ox_list[0] = self.theta_ox
            self.dtheta_ox_dt_list[0] = self.dtheta_ox_dt
            self.i[0] = self.current_function(0.0,times[0],0.0,capOption)
    
        index = 1
        # print('inital i: ',  self.i[index - 1])
        # print('self.modelling_faradaic_current: ', self.modelling_faradaic_current)
        # print('self.modeling_experimental_data: ', self.modeling_experimental_data)
        # IR_drop = True
        for time in t:
            # print('find_epsilon')
            self.find_epsilon(time, index, IR_drop=True)
            # finding current at next time step
            # print('analytical_current_solution')
            if self.modelling_faradaic_current == True:
                # updating theta_X
                self.backwards_euler()
                # updating dtheta_X_dt
                self.find_dtheta_X_dt()
                self.theta_ox_list[index] = self.theta_ox
                self.dtheta_ox_dt_list[index] = self.dtheta_ox_dt
            self.analytical_current_solution(time, index, capOption)
            self.drivAppliedPotential[index] = self.find_dev_epsilon(self.i[index-1], time, self.i[index])
            index = index + 1
        
        return self.i

    def get_applied_potential_and_driv(self):
        return (self.appliedPotential, self.appliedPotentialDCVComponetent,self.drivAppliedPotential, self.episilon_IR_drop_componenet, self.episilon_without_drop)
    
    def set_modelling_faradaic_current(self, modelling_faradaic_current = False):
        self.modelling_faradaic_current = modelling_faradaic_current
    
    def set_modeling_experimental_data(self, modeling_experimental_data = False):
        self.modeling_experimental_data = modeling_experimental_data


# adjust me so pure capacitance is no longer regional

class wrappedNewtonCapFaradaicDip(pints.ForwardModel):
    def __init__(self, times: float, startPotential: float = -0.15, revPotential: float = -0.75, rateOfPotentialChange: float = -22.35e-3,
                inital_current: float = 3.92061e-06, freq: float = 8.95931721948, deltaepislon: float = 150E-3,
                electrode_area: float = 0.07, Z: float = 1.0, protocol = 'FTACV', numberConc: float = 6.0221409e+23*100*10**-3,
                endPureCapatianceFor: float = 0.3, beingPureCapitanceto: float = 0.02, fit_alpha: boolean = False,
                resistance_cap_probability_likelhood: boolean = False,
                nodes = 16):

        self.startPotential = startPotential
        self.revPotential = revPotential
        self.rateOfPotentialChange = rateOfPotentialChange

        self.times = times
       
        length = times.shape
        self.numberOfMeasurements = int(length[0])
        self.half_of_measuremnts = math.ceil(self.numberOfMeasurements/2)
        self.capOption = 'doublePoly'

        if self.numberOfMeasurements % 2.0 == 0.0:

            self.potentialRange = np.linspace(startPotential, revPotential, self.numberOfMeasurements )
            reversed_potentialRange = np.flip(self.potentialRange)
            fullPotentialRange = np.hstack((self.potentialRange, reversed_potentialRange[1:]))
            self.fullPotentialRange = fullPotentialRange[::2]

        else:

            self.potentialRange = np.linspace(startPotential, revPotential, self.half_of_measuremnts)
            reversed_potentialRange = np.flip(self.potentialRange)
            self.fullPotentialRange = np.hstack((self.potentialRange, reversed_potentialRange[1:]))

        # As the first time is at 0.0s we take one of the numberOfMeasurements
        # to split total time evenly and get the most accurate timeStepSize
        self.timeStepSize = times[-1]/(self.numberOfMeasurements - 1)

        # parameters to pass to main model
        self.inital_current=inital_current
        self.freq=freq
        self.deltaepislon=deltaepislon
        self.electrode_area=electrode_area

        self.Z = Z
        self.numberConc = numberConc

        # experimental voltage protocol
        self.protocol = protocol

        # adding regioning function

        self.endPureCapatianceFor = int(endPureCapatianceFor*int(self.numberOfMeasurements/2))
        self.beingPureCapitanceto = int(beingPureCapitanceto*int(self.numberOfMeasurements/2))
        print('self.endPureCapatianceFor: ',self.endPureCapatianceFor)
        print('self.beingPureCapitanceto: ',self.beingPureCapitanceto)
        removed_measures_to_account_for = 0.0


        self.endCap = self.numberOfMeasurements - self.beingPureCapitanceto + int(removed_measures_to_account_for)

        self.midCapLow = int(self.numberOfMeasurements/2)-self.endPureCapatianceFor
        self.midCaphigh = int(self.numberOfMeasurements/2)+self.endPureCapatianceFor


        self.appliedPotential = np.zeros(self.numberOfMeasurements, dtype = np.float64)
        self.drivAppliedPotential = np.zeros(self.numberOfMeasurements, dtype = np.float64)
        self.appliedPotentialDCVComponetent = np.zeros(self.numberOfMeasurements, dtype = np.float64)

        self.modelling_faradaic_current = False
        self.modeling_experimental_data = False

        length = times.shape
        self.numberOfMeasurements = int(length[0])
        self.half_of_measuremnts = math.ceil(self.numberOfMeasurements/2)
        # to split total time evenly and get the most accurate timeStepSize
        self.timeStepSize = times[-1]/(self.numberOfMeasurements - 1)

        self.first_harm = 4
        self.last_harm = 12
        self.fitting_range = 0.5

        if self.freq != 0.0:
                # frequencies from FFT of times
                self.full_frequencies = np.fft.fftfreq(self.numberOfMeasurements, d=times[1])
                self.half_frequencies=self.full_frequencies [:self.half_of_measuremnts]
                # index location of harmonic 4-12
                x = np.where(self.half_frequencies  < (self.freq*3.5))
                self.lower_index_harm_fit = x[0][-1]
                x = np.where(self.half_frequencies < (self.freq*12.5))
                self.upper_index_harm_fit = x[0][-1]

        self.modeling_dispersion_e = False
        self.modeling_dispersion_k = False
        self.modeling_dispersion_alpha = False
        self.modeling_dispersion = False
        self.fit_alpha = fit_alpha
        self.fit_faradaic_and_cap = False

        self.resistance_cap_probability_likelhood = resistance_cap_probability_likelhood

        self.solver_option = 'my_newton_raphson'

        self.nodes = nodes

    
    def set_dispersion_modeling(self, e_dip = False, k_dip= False, alpha_dip= False):
        self.modeling_dispersion_e = e_dip
        self.modeling_dispersion_k = k_dip
        self.modeling_dispersion_alpha = alpha_dip

        if self.modeling_dispersion_alpha == False and self.modeling_dispersion_e == False and self.modeling_dispersion_k == False:
            self.modeling_dispersion = False
        else:
            self.modeling_dispersion = True

        if self.modeling_dispersion_alpha == True:
            self.fit_alpha = True

    def set_modelling_faradaic_current(self, modelling_faradaic_current = False):
        self.modelling_faradaic_current = modelling_faradaic_current

    def set_fitting_faradaic_and_capactiance_current(self, faradaic_and_capactiance_current = False):
        self.fit_faradaic_and_cap = faradaic_and_capactiance_current

    def set_modeling_experimental_data(self, modeling_experimental_data = False):
        self.modeling_experimental_data = modeling_experimental_data


    def reshape_to_cap_regions(self, array):

        raw = np.asarray(array)
        a = raw[:self.beingPureCapitanceto]
        b = raw[self.midCapLow:self.midCaphigh]
        c = raw[self.endCap:]
        reshaped = np.hstack((a,b,c))
        return reshaped
    
    def set_resistance_likelihood_scan(self, resistance_likelihood_scan):

        self.resistance_likelihood_scan = resistance_likelihood_scan

    def n_outputs(self):
        """ 
        See :meth:`pints.ForwardModel.n_outputs()`.
        number of outputs of the model
        """
        # current I
        return 1
    
    def n_parameters(self):
        """ See :meth:`pints.ForwardModel.n_parameters()`. 
        :return: dimensions of parameter vector
        """
        # [gamma0, gamma1, gamma2, gamma3, gamma4, gamma5, gamma6,
        #  gamma7, mew, omega, uncompensated_resistance]
        
        models = ['doublePoly', '3rd_order_poly', 'helmholtz', 'gouyChapman', 'stern', '2dPoly', '1st_order_poly', '2nd_order_poly','4th_order_poly', '5th_order_poly']
        if self.modelling_faradaic_current == True and self.fit_faradaic_and_cap == False:
            order = self.faradaic_parameter_order()
            print('order: ', order)
            length = len(order)
            print('parameter length: ', length)
            output = length -1 # removing 1 for sigma
        else:
             order = self.parameter_order()
             length = len(order)
             output = length - 1 # i.r. dropping sigma
             if self.protocol == 'DCV':
                # removing mew and omega
                output = output - 2
        
        return output
        
        # else:
        #     n = [11,7,4,5,6,9,5,6]
        #     index = models.index(self.capOption)
        #     output = n[index]
        #     if self.protocol == 'DCV':
        #         # removing mew and omega
        #         output = output - 2
        # return output

    def get_cap_model(self):
        """displays and returns the capacitance model that is currently being used
        """
        print('capacitance model: ', self.capOption)
        return self.capOption
    
    def get_protocol(self):
        """displays and returns the capacitance model that is currently being used
        """
        print('protocol: ', self.protocol)
        return self.protocol
    
    def set_cap_model(self, cap_model:str):
        """sets the capacitance model to be used in the simulation the options
            avaliable are:
            doublePoly - a double 3rd order polynomial model of capacitance
            3rd_order_poly - a single 3rd order polynomial model of capacitance
            helmholtz - the Helmholtz model of capacitance
            gouyChapman - the Gouy-Chapman model of capacitance
            stern - the Gouy-Chapman-Stern model of capacitance
            """

        models = ['doublePoly', '3rd_order_poly', 'helmholtz', 'gouyChapman', 'stern', '2dPoly', '1st_order_poly', '2nd_order_poly', '4th_order_poly', '5th_order_poly']

        if cap_model in models:
            self.capOption = cap_model
        else:
            raise Exception('Invalid current option selected please choice from: doublePoly, 3rd_order_poly, helmholtz, gouyChapman, stern, 2dPoly, 1st_order_poly ,  2nd_order_poly, 4th_order_poly, or 5th_order_poly')
        
    def set_solver_option(self, solver_option:str = 'my_newton_raphson'):
        '''setter function for the applied voltage protocol
           for use in the simulation.
           Currently the following experimental protocols are allowed:
            - FTACV
            - DCV
            - EIS
            - PSV
        '''
        if solver_option == 'my_newton_raphson' or solver_option == 'scipy_newton_raphson' or solver_option == 'scipy_brent' or solver_option ==  'my_riddlers':
            # if not using JIT class remove the [0]
            self.solver_option = solver_option
        else:
            raise Exception('Invalid solver option, please choice from: my_newton_raphson, scipy_newton_raphson, scipy_brent,  my_riddlers, or ....')
    
    def _simulate(self, parameters, times, reduce):
        """
        Private helper function that uses ``scipy.integrate.odeint`` to
        simulate a model (with or without sensitivities).
        """
               
        # ensuring time and parameters are numpy array
        times = np.asarray(times)
        parameters = np.asarray(parameters)

        # print('PARAMETERS PASSED TO _SIMULATE: ',parameters )

        # creating instance of newtonRaphsonCap

        solver = newtonRaphsonCapFaradaicDip(timeStepSize=self.timeStepSize, numberOfMeasurements=self.numberOfMeasurements,
                                  startPotential=self.startPotential, revPotential=self.revPotential,
                                  rateOfPotentialChange=self.rateOfPotentialChange, inital_current=self.inital_current,
                                  freq=self.freq, deltaepislon=self.deltaepislon,
                                  electrode_area =self.electrode_area, Z= self.Z, numberConc=self.numberConc,
                                  fit_alpha = self.fit_alpha)
        # print('cap option in warpper: ', self.capOption)
        solver.set_experimental_protocol(self.protocol)
        solver.set_solver_option(self.solver_option)
        solver.set_modelling_faradaic_current(self.modelling_faradaic_current)
        solver.set_modeling_experimental_data(self.modeling_experimental_data)

        # solver.set_modelling_faradaic_current(self.modelling_faradaic_current)
        if self.modelling_faradaic_current == False and self.fit_faradaic_and_cap == False:
            solver.set_modelling_faradaic_current(False)

            if self.capOption == '1st_order_poly':
                cap_model = 'poly'
                cap_parameters_copy = copy.deepcopy(parameters)
                cap_parameters = np.zeros([9], dtype=float)
                cap_parameters[:2] = cap_parameters_copy[:2]
                cap_parameters[-3:] = cap_parameters_copy[-3:]
            elif self.capOption == '2nd_order_poly':
                cap_model = 'poly'
                cap_parameters_copy = copy.deepcopy(parameters)
                cap_parameters = np.zeros([9], dtype=float)
                cap_parameters[:3] = cap_parameters_copy[:3]
                cap_parameters[-3:] = cap_parameters_copy[-3:]
            elif self.capOption == '3rd_order_poly':
                cap_model = 'poly'
                cap_parameters_copy = copy.deepcopy(parameters)
                cap_parameters = np.zeros([9], dtype=float)
                cap_parameters[:4] = cap_parameters_copy[:4]
                cap_parameters[-3:] = cap_parameters_copy[-3:]
            elif self.capOption == '4th_order_poly':
                cap_model = 'poly'
                cap_parameters_copy = copy.deepcopy(parameters)
                cap_parameters = np.zeros([9], dtype=float)
                cap_parameters[:5] = cap_parameters_copy[:5]
                cap_parameters[-3:] = cap_parameters_copy[-3:]
            elif self.capOption == '5th_order_poly':
                cap_model = 'poly'
                cap_parameters_copy = copy.deepcopy(parameters)
                cap_parameters = np.zeros([9], dtype=float)
                cap_parameters[:6] = cap_parameters_copy[:6]
                cap_parameters[-3:] = cap_parameters_copy[-3:]
            else:
                cap_model = self.capOption
            
                if self.resistance_cap_probability_likelhood == False:
                    cap_parameters = parameters
                else:
                    cap_parameters_copy = copy.deepcopy(parameters)
                    shape_wanted = cap_parameters_copy.shape[0]
                    shape_wanted = shape_wanted + 1
                    cap_parameters = np.zeros([shape_wanted], dtype=float)
                    if self.protocol != 'DCV':
                        cap_parameters[:-3] = cap_parameters_copy[:-2]
                        cap_parameters[-2:] = cap_parameters_copy[-2:]
                        cap_parameters[-3] = self.resistance_likelihood_scan
                    else:
                        cap_parameters[:-2] = cap_parameters_copy[:]
                        cap_parameters[-1] = self.resistance_likelihood_scan

            

            # print('PARAMETERS PASSED TO NEWTON: ',cap_parameters )
            solver.set_capacitance_params(cap_parameters, cap_model)
            # print('ABOUT TO RUN NEWTON: ',cap_parameters )
            i = solver.solve(times, cap_model)
        else:
            solver.set_modelling_faradaic_current(True)
            if self.modelling_faradaic_current == True and self.fit_faradaic_and_cap == False:
                if self.capOption == '1st_order_poly':
                    cap_model = 'poly'
                    cap_parameters_copy = copy.deepcopy(self.cap_parameters)
                    cap_parameters = np.zeros([9])
                    cap_parameters[:2] = cap_parameters_copy[:2]
                    cap_parameters[-3:] = cap_parameters_copy[-3:]
                elif self.capOption == '2nd_order_poly':
                    cap_model = 'poly'
                    cap_parameters_copy = copy.deepcopy(self.cap_parameters)
                    cap_parameters = np.zeros([9])
                    cap_parameters[:3] = cap_parameters_copy[:3]
                    cap_parameters[-3:] = cap_parameters_copy[-3:]
                elif self.capOption == '3rd_order_poly':
                    cap_model = 'poly'
                    cap_parameters_copy = copy.deepcopy(self.cap_parameters)
                    cap_parameters = np.zeros([9], dtype=float)
                    cap_parameters[:4] = cap_parameters_copy[:4]
                    cap_parameters[-3:] = cap_parameters_copy[-3:]
                elif self.capOption == '4th_order_poly':
                    cap_model = 'poly'
                    cap_parameters_copy = copy.deepcopy(self.cap_parameters)
                    cap_parameters = np.zeros([9], dtype=float)
                    cap_parameters[:5] = cap_parameters_copy[:5]
                    cap_parameters[-3:] = cap_parameters_copy[-3:]
                elif self.capOption == '5th_order_poly':
                    cap_model = 'poly'
                    cap_parameters_copy = copy.deepcopy(self.cap_parameters)
                    cap_parameters = np.zeros([9], dtype=float)
                    cap_parameters[:6] = cap_parameters_copy[:6]
                    cap_parameters[-3:] = cap_parameters_copy[-3:]
                else:
                    cap_model = self.capOption
                    cap_parameters = self.cap_parameters

                faradaic_parameters = parameters

                # print('OG cap parameters: ', cap_parameters)
                # print('OG faradaic parameters: ', parameters)

            
            else:
                # doublePoly, poly, helmholtz, gouyChapman, stern, 2dPoly, 1st_order_poly, or 2nd_order_poly
                number_of_parameters = self.n_parameters()
                # faradaic_parameters
                order = self.faradaic_parameter_order()
                length = len(order)
                length = length - 1 # i.r. dropping sigma 
                faradaic_parameters = np.zeros([length])
                faradaic_copy= copy.deepcopy(parameters)
                faradaic_parameters[0] = faradaic_copy[0]
                faradaic_parameters[1] = faradaic_copy[1]
                faradaic_parameters[2] = faradaic_copy[-1]
                faradaic_parameters[3] = faradaic_copy[-3]
                faradaic_parameters[4] = faradaic_copy[2]
                # print('faradaic_parameters: ',faradaic_parameters)
                # print('SUGGEST faradaic_parameters: ',self.suggested_faradaic_parameters())
                # capcaitance parameters
                # -3 as 2 parameters are shared between cap and faradaic parameters (resistance and phase)
                # print('parameters: ',parameters)
                # print('parameters[3:]: ',parameters[3:])
                cap_parameters = np.zeros([number_of_parameters-3])
                cap_copy= copy.deepcopy(parameters)
                cap_parameters[:] = cap_copy[3:]
                if self.capOption == '1st_order_poly':
                    cap_model = 'poly'
                    cap_parameters_copy = copy.deepcopy(cap_parameters)
                    cap_parameters = np.zeros([9])
                    cap_parameters[:2] = cap_parameters_copy[:2]
                    cap_parameters[-3:] = cap_parameters_copy[-3:]
                elif self.capOption == '2nd_order_poly':
                    cap_model = 'poly'
                    cap_parameters_copy = copy.deepcopy(cap_parameters)
                    cap_parameters = np.zeros([9])
                    cap_parameters[:3] = cap_parameters_copy[:3]
                    cap_parameters[-3:] = cap_parameters_copy[-3:]
                elif self.capOption == '3rd_order_poly':
                    cap_model = 'poly'
                    cap_parameters_copy = copy.deepcopy(parameters)
                    cap_parameters = np.zeros([9], dtype=float)
                    cap_parameters[:4] = cap_parameters_copy[:4]
                    cap_parameters[-3:] = cap_parameters_copy[-3:]
                elif self.capOption == '4th_order_poly':
                    cap_model = 'poly'
                    cap_parameters_copy = copy.deepcopy(parameters)
                    cap_parameters = np.zeros([9], dtype=float)
                    cap_parameters[:5] = cap_parameters_copy[:5]
                    cap_parameters[-3:] = cap_parameters_copy[-3:]
                elif self.capOption == '5th_order_poly':
                    cap_model = 'poly'
                    cap_parameters_copy = copy.deepcopy(parameters)
                    cap_parameters = np.zeros([9], dtype=float)
                    cap_parameters[:6] = cap_parameters_copy[:6]
                    cap_parameters[-3:] = cap_parameters_copy[-3:]
                else:
                    cap_model = self.capOption

                # print('cap_parameters: ',cap_parameters)
                # print('SUGGEST cap_parameters: ',self.suggested_capacitance_params())
                # print('SUGGEST parameters: ',self.suggested_parameters())
                # print('SUGGEST suggested_faradaic_and_capactiance_params: ',self.suggested_faradaic_and_capactiance_params())

            
            solver.set_capacitance_params(cap_parameters, cap_model)

            if self.modeling_dispersion is False:
                solver.set_faradaic_params(faradaic_parameters)
                i = solver.solve(times, cap_model)
                self.theta_ox_list = solver.theta_ox_list
                self.dtheta_ox_dt_list = solver.dtheta_ox_dt_list

            else:
                from electrochemistry_modelling import dispersion_class
                dis_model = dispersion_class()

                std_dev_counter = -1
                
                params = copy.deepcopy(faradaic_parameters)
                if self.modeling_dispersion_e is True:
                    # self.nodes = 16
                    self.E_std = parameters[std_dev_counter]
                    self.E_mean=parameters[1]
                    E_dip_val,E_dip_weight = dis_model.normal(self.nodes, self.E_std, self.E_mean)

                    print('E_dip_val: ', E_dip_val)
                    print('E_dip_weight: ', E_dip_weight)
                    # print('E_dip_weight sum: ', np.sum(E_dip_weight))

                    weighted_i = 0
                    weighted_theta_ox_list = np.zeros(self.numberOfMeasurements, dtype = np.float64)
                    weighted_dtheta_ox_dt_list = np.zeros(self.numberOfMeasurements, dtype = np.float64)

                    for index in range(self.nodes):

                        params[1] = E_dip_val[index]
                        # print('index: ', index)
                        # print('E_dip_val evaluated: ', E_dip_val[index])
                        # print('E_dip_weight evaluated: ', E_dip_weight[index])
                        solver.set_faradaic_params(params)
                        i = solver.solve(times, cap_model)
                        weighted_i += i*E_dip_weight[index]
                        theta_ox_list = solver.theta_ox_list
                        dtheta_ox_dt_list = solver.dtheta_ox_dt_list
                        weighted_theta_ox_list += theta_ox_list*E_dip_weight[index]
                        weighted_dtheta_ox_dt_list += dtheta_ox_dt_list*E_dip_weight[index]

                    i = weighted_i
                    std_dev_counter -=1
                    self.theta_ox_list = weighted_theta_ox_list
                    self.dtheta_ox_dt_list = weighted_dtheta_ox_dt_list

                if self.modeling_dispersion_k is True:
                    # self.nodes = 16
                    self.k_std = parameters[std_dev_counter]
                    self.k_mean=parameters[0]
                    k_dip_val,k_dip_weight = dis_model.log_normal(self.nodes, self.k_std, self.k_mean)

                    print('k_dip_val: ', k_dip_val)
                    # print('k_dip_weight: ', k_dip_weight)
                    # print('k_dip_weight sum: ', np.sum(k_dip_weight))


                    weighted_i = 0

                    for index in range(self.nodes):

                        params[0] = k_dip_val[index]
                        # print('index: ', index)
                        # print('k_dip_val evaluated: ', k_dip_val[index])
                        # print('k_dip_weight evaluated: ', k_dip_weight[index])
                        solver.set_faradaic_params(params)
                        i = solver.solve(times, cap_model)
                        weighted_i += i*k_dip_weight[index]

                    i = weighted_i
                    std_dev_counter -=1

                if self.modeling_dispersion_alpha is True:
                    # print('parameters: ', parameters)
                    # print('alpha: ', parameters[5])
                    # self.nodes = 16
                    self.alpha_std = parameters[std_dev_counter]
                    # print('alpha_std: ',  self.alpha_std)
                    self.alpha_mean=parameters[5]
                    alpha_dip_val,alpha_dip_weight = dis_model.normal(self.nodes, self.alpha_std, self.alpha_mean)

                    print('alpha_dip_val: ', alpha_dip_val)
                    # print('alpha_dip_weight: ', alpha_dip_weight)
                    # print('alpha_dip_weight sum: ', np.sum(alpha_dip_weight))


                    weighted_i = 0

                    for index in range(self.nodes):

                        params[5] = alpha_dip_val[index]
                        # print('index: ', index)
                        # print('alpha_dip_val evaluated: ', alpha_dip_val[index])
                        # print('alpha_dip_weight evaluated: ', alpha_dip_weight[index])
                        solver.set_faradaic_params(params)
                        i = solver.solve(times, cap_model)
                        weighted_i += i*alpha_dip_weight[index]
                    
                    i = weighted_i
                    std_dev_counter -=1
                


        potential = solver.get_applied_potential_and_driv()
        self.appliedPotential = potential[0]
        self.appliedPotentialDCVComponetent = potential[1]
        self.drivAppliedPotential = potential[2]
        self.episilon_IR_drop_componenet = potential[3]
        self.episilon_without_drop = potential[4]

        if reduce == True:
            output = self.reshape_to_cap_regions(i)
            return output
        else:
            return i


    def simulate(self, parameters, times):
        """ See :meth:`pints.ForwardModel.simulate()`. """

        i = self._simulate(parameters, times, False)
        I = np.asarray(i)
        # non_dim_constants = self.get_non_dimensionality_constants()
        # I0 = non_dim_constants[-1]
        # I = I*I0

        return I

    def simulate_fitting_regions(self, parameters, times):
       
        i = self._simulate(parameters, times, True)
        I = np.asarray(i)
        return I
    
    def set_capacitance_params(self, cap_params):
        '''
        takes a list of capasiance parameters and sets these for the model
        :param: cap_params = [gamma0, gamma1, gamma2, gamma3, gamma4,
                              gamma5, gamma6, gamma7, omega, mu, row]
        '''
        cap_params = np.asarray(cap_params)
        shape = cap_params.shape
        print('shape cap parameters: ', shape)
        
        if self.capOption == 'doublePoly':
            if self.protocol != 'DCV':
                if shape == np.asarray([11]):
                    self.cap_parameters = cap_params
                else:
                    raise Exception('Invalid cap_params size, doublePoly should have 11 parameters: gamma0, gamma1, gamma2, gamma3, gamma4, gamma5, gamma6, gamma7, resistance, omega, mew')
            else:
                if shape == np.asarray([9]):
                    self.cap_parameters = cap_params
                else:
                    raise Exception('Invalid cap_params size, doublePoly should have 9 parameters: gamma0, gamma1, gamma2, gamma3, gamma4, gamma5, gamma6, gamma7, resistance')

        elif self.capOption == '3rd_order_poly':
            if self.protocol != 'DCV':
                if shape == np.asarray([7]):
                    self.cap_parameters = cap_params
                else:
                    raise Exception('Invalid cap_params size, poly should have 7 parameters: gamma0, gamma1, gamma2, gamma3, resistance, omega, mew')
            else:
                if shape == np.asarray([5]):
                    self.cap_parameters = cap_params
                else:
                    raise Exception('Invalid cap_params size, poly should have 5 parameters: gamma0, gamma1, gamma2, gamma3, resistance')
            
        elif self.capOption == '4th_order_poly':
            if self.protocol != 'DCV':
                if shape == np.asarray([8]):
                    self.cap_parameters = cap_params
                else:
                    raise Exception('Invalid cap_params size, poly should have 8 parameters: gamma0, gamma1, gamma2, gamma3, gamma4, resistance, omega, mew')
            else:
                if shape == np.asarray([6]):
                    self.cap_parameters = cap_params
                else:
                    raise Exception('Invalid cap_params size, poly should have 6 parameters: gamma0, gamma1, gamma2, gamma3, gamma4, resistance')
            
        elif self.capOption == '5th_order_poly':
            if self.protocol != 'DCV':
                if shape == np.asarray([9]):
                    self.cap_parameters = cap_params
                else:
                    raise Exception('Invalid cap_params size, poly should have 9 parameters: gamma0, gamma1, gamma2, gamma3, gamma4, gamma5, resistance, omega, mew')
            else:
                if shape == np.asarray([7]):
                    self.cap_parameters = cap_params
                else:
                    raise Exception('Invalid cap_params size, poly should have 7 parameters: gamma0, gamma1, gamma2, gamma3, gamma4, gamma5, resistance')    
            
        elif self.capOption == 'helmholtz':
            if self.protocol != 'DCV':
                if shape == np.asarray([4]):
                    self.cap_parameters = cap_params
                else:
                    raise Exception('Invalid cap_params size, helmholtz should have 4 parameters: permittivitySpaceRatio, resistance, omega, mew')
            else:
                if shape == np.asarray([2]):
                    self.cap_parameters = cap_params
                else:
                    raise Exception('Invalid cap_params size, helmholtz should have 2 parameters: permittivitySpaceRatio, resistance')

        elif self.capOption == 'gouyChapman':
            if self.protocol != 'DCV':
                if shape == np.asarray([5]):
                    self.cap_parameters = cap_params
                else:
                    raise Exception('Invalid cap_params size, gouyChapman should have 5 parameters: dielectricConstant, pzc, resistance, omega, mew')
            else:
                if shape == np.asarray([3]):
                    self.cap_parameters = cap_params
                else:
                    raise Exception('Invalid cap_params size, gouyChapman should have 3 parameters: dielectricConstant, pzc, resistance')

        elif self.capOption == 'stern':
            if self.protocol != 'DCV':
                if shape == np.asarray([6]):
                    self.cap_parameters = cap_params
                else:
                    raise Exception('Invalid cap_params size, gouyChapman should have 6 parameters: dielectricConstant, interplateSpace,  pzc, resistance, omega, mew')
            else:
                if shape == np.asarray([4]):
                    self.cap_parameters = cap_params
                else:
                    raise Exception('Invalid cap_params size, gouyChapman should have 4 parameters: dielectricConstant, interplateSpace,  pzc, resistance')
        
        elif self.capOption == '2dPoly':
            if self.protocol != 'DCV':
                if shape == np.asarray([9]):
                    self.cap_parameters = cap_params
                else:
                    raise Exception('Invalid cap_params size, poly should have 9 parameters: gamma0, gamma1, gamma2, gamma3, gamma4, gamma5, resistance, omega, mew')
            else:
                if shape == np.asarray([7]):
                    self.cap_parameters = cap_params
                else:
                    raise Exception('Invalid cap_params size, poly should have 7 parameters: gamma0, gamma1, gamma2, gamma3, gamma4, gamma5, resistance')
            
        elif self.capOption == '1st_order_poly':
            if self.protocol != 'DCV':
                if shape == np.asarray([5]):
                    self.cap_parameters = cap_params
                else:
                    raise Exception('Invalid cap_params size, poly should have 5 parameters: gamma0, gamma1, resistance, omega, mew')
            else:
                if shape == np.asarray([3]):
                    self.cap_parameters = cap_params
                else:
                    raise Exception('Invalid cap_params size, poly should have 3 parameters: gamma0, gamma1, resistance')
            
        elif self.capOption == '2nd_order_poly':
            if self.protocol != 'DCV':
                if shape == np.asarray([6]):
                    self.cap_parameters = cap_params
                else:
                    raise Exception('Invalid cap_params size, poly should have 76 parameters: gamma0, gamma1, gamma2, resistance, omega, mew')
            else:
                if shape == np.asarray([4]):
                    self.cap_parameters = cap_params
                else:
                    raise Exception('Invalid cap_params size, poly should have 4 parameters: gamma0, gamma1, gamma2, resistance')
            
        else:
            raise Exception('Invalid current option selected please choice from: doublePoly, poly, helmholtz, gouyChapman, stern, 2dPoly, 1st_order_poly, 2nd_order_poly')



    def suggested_capacitance_params(self):
        """Returns a list with suggestsed capacitance parameters for the model with dimension
        return: [gamma0, gamma1, gamma2, gamma3, gamma4, gamma5, gamma6,
                 gamma7, omega, mew, uncompensated_resistance]
        """

        if self.capOption == 'doublePoly':

            if self.deltaepislon == 300E-3:

                gamma0 = 1.75863380728210474e-05
                gamma1 = 7.93921078590131314e-06
                gamma2 = 1.12013976531365102e-05
                gamma3 = -6.90042986767856747e-06
                gamma4 =  1.94076119947446042e-05 
                gamma5 = 4.40653034129777181e-06
                gamma6 = 8.67255407443448707e-06
                gamma7 =  -4.03703922384588028e-06
                # resistance = 5.22609315334568805e+02
                resistance = 5.50e+02

            else:

                gamma0 = 1.46317932258782763e-05 
                gamma1 = 6.37695325874081814e-08 
                gamma2 = 5.22600504148165723e-06
                gamma3 = 3.22007594244348196e-06 
                gamma4 =  1.48102232617370742e-05
                gamma5 = -1.66320856939575039e-06 
                gamma6 = 6.08627865194329848e-06 
                gamma7 =  2.99172212300043323e-06 
                # resistance = 7.75409989510109995e+02 
                resistance = 5.50e+02

            output = [gamma0, gamma1, gamma2, gamma3, gamma4, gamma5, gamma6,
                    gamma7, resistance ]

        
        elif self.capOption == '3rd_order_poly':
            # gamma0 = 1.69e-04 
            # gamma1 = -3.19e-04
            # gamma2 = -4.89e-04
            # gamma3 = -3.02e-04 
            if self.deltaepislon == 300E-3:
            
                gamma0 =  1.85071072619651155e-05
                gamma1 =  6.20059590854927160e-06
                gamma2 =  9.92367826638328590e-06
                gamma3 = -5.55238888434982238e-06
                # resistance = 5.58729755769305370e+02
                resistance = 5.00e+02

            elif self.deltaepislon == 150E-3:
                gamma0 = 1.071261e-05
                gamma1 = -2.175661e-07
                gamma2 = 5.095181e-06
                gamma3 = 6.592396e-07
                # resistance = 7.73588959680028438e+02
                resistance = 5.00e+02
            
            else:
                gamma0 =  9.736447e-06
                gamma1 = -5.785937e-08
                gamma2 = 5.043966e-06
                gamma3 = -1.501332e-07
                # resistance = 7.73588959680028438e+02
                resistance = 5.00e+02

            output = [gamma0, gamma1, gamma2, gamma3,
                      resistance]
            if self.resistance_cap_probability_likelhood == True:
                output = [gamma0, gamma1, gamma2, gamma3]

        elif self.capOption == '4th_order_poly':
            # gamma0 = 1.69e-04 
            # gamma1 = -3.19e-04
            # gamma2 = -4.89e-04
            # gamma3 = -3.02e-04 
            if self.deltaepislon == 300E-3:
            
                gamma0 =  1.85071072619651155e-05
                gamma1 =  6.20059590854927160e-06
                gamma2 =  9.92367826638328590e-06
                gamma3 = -5.55238888434982238e-06
                gamma4 = 1e-05
                # resistance = 5.58729755769305370e+02
                resistance = 5.50e+02

            else:
                gamma0 = 1.47201819370179903e-05
                gamma1 = -8.19989743760371676e-07 
                gamma2 = 5.61946210605756829e-06
                gamma3 = 3.18881138399119474e-06
                gamma4 = 1e-05
                # resistance = 7.73588959680028438e+02
                resistance = 5.50e+02
            output = [gamma0, gamma1, gamma2, gamma3, gamma4,
                      resistance]
            if self.resistance_cap_probability_likelhood == True:
                output = [gamma0, gamma1, gamma2, gamma3, gamma4]

        elif self.capOption == '5th_order_poly':
            # gamma0 = 1.69e-04 
            # gamma1 = -3.19e-04
            # gamma2 = -4.89e-04
            # gamma3 = -3.02e-04 
            if self.deltaepislon == 300E-3:
            
                gamma0 =  1.85071072619651155e-05
                gamma1 =  6.20059590854927160e-06
                gamma2 =  9.92367826638328590e-06
                gamma3 = -5.55238888434982238e-06
                gamma4 = 1e-05
                gamma5 = 1e-05
                # resistance = 5.58729755769305370e+02
                resistance = 5.50e+02

            else:
                gamma0 = 1.47201819370179903e-05
                gamma1 = -8.19989743760371676e-07 
                gamma2 = 5.61946210605756829e-06
                gamma3 = 3.18881138399119474e-06
                gamma4 = 1e-05
                gamma5 = 1e-05
                # resistance = 7.73588959680028438e+02
                resistance = 5.50e+02
            output = [gamma0, gamma1, gamma2, gamma3, gamma4, gamma5,
                      resistance]
            if self.resistance_cap_probability_likelhood == True:
                output = [gamma0, gamma1, gamma2, gamma3, gamma4, gamma5]


        elif self.capOption == 'helmholtz':
            permittivityMedium = 10.0 # 78 approximately water at 25 celsius
            # interplateSpace = 0.5E-9*10**2 # 0.5 nM, note interplate spacing is in cm
            permittivityMedium = 78.0
            interplateSpace = 0.5E-6*10**2

            if self.deltaepislon == 300E-3:
                gamma0 = 1.97574508240218726e-05
                # resistance = 9.99999999999985221e+02
                resistance = 4.50e+02
            else:
                gamma0 = 1.58537716727587013e-05
                # resistance = 1.00000000000000000e+03
                resistance = 5.50e+02


            output = [gamma0,
                      resistance]

        elif self.capOption == 'gouyChapman':
            dielectricConstant = 70.0 # approximately water at 20 celsius
            # numberConc = 6.0221409e+23*2*10**6 #i.e a 2 molar solution converting to per cm^3
            pzc = 0.0E-3 # point of zero charge in volts
            output = [dielectricConstant, pzc,
                      79.9]
        
        elif self.capOption == 'stern':
            dielectricConstant = 70.0 # 78 approximately water at 25 celsius
            interplateSpace = 0.5E-9*10**2 # 0.5 nM, note interplate spacing is in cm
            dielectricConstant = 70.0 # approximately water at 20 celsius
            # numberConc = 6.0221409e+23*2*10**6 #i.e a 2 molar solution converting to per cm^3
            # pzc = 1.0E-3 # point of zero charge in volts

            dielectricConstant = 70.0
            interplateSpace = 0.5E-6*10**2
            pzc = 0.0E-3

            output = [dielectricConstant, interplateSpace, pzc,
                      79.9]
        
        elif self.capOption == '2dPoly':

            if self.deltaepislon == 300E-3:

                gamma0 = 1.75863380728210474e-05
                gamma1 = 7.93921078590131314e-06 
                gamma2 = 1.12013976531365102e-05
                gamma3 = -6.90042986767856747e-06
                gamma4 = 1.94076119947446042e-05 
                gamma5 = 4.40653034129777181e-06

                # resistance = 5.22609315334568805e+02
                resistance = 5.50e+02

            else:

                pass



            output = [gamma0, gamma1, gamma2, gamma3,
                      gamma4, gamma5, gamma6, gamma7,
                      resistance]
            
        elif self.capOption == '1st_order_poly':
            # gamma0 = 1.69e-04 
            # gamma1 = -3.19e-04
            # gamma2 = -4.89e-04
            # gamma3 = -3.02e-04 
            if self.deltaepislon == 300E-3:

                gamma0 =  1.99886352827815220e-05
                gamma1 =  7.33142590812881527e-06
                # resistance = 6.47291302069237872e+02
                resistance = 5.50e+02

            else:

                gamma0 =  1.49960874915495244e-05
                gamma1 =  3.57094637219130753e-06
                # resistance = 1.00000000032769343e+01 
                resistance = 5.50e+02

            output = [gamma0, gamma1,
                       resistance]
            
        elif self.capOption == '2nd_order_poly':
            # gamma0 = 1.69e-04 
            # gamma1 = -3.19e-04
            # gamma2 = -4.89e-04
            # gamma3 = -3.02e-04 
            
            if self.deltaepislon == 300E-3:

                gamma0 =  1.85093024943017783e-05
                gamma1 =  4.52256813112485882e-06
                gamma2 =  7.71210345224108554e-06
                # resistance = 5.07361394174784152e+02
                resistance = 5.50e+02

            else:

                gamma0 =  1.47641080061587817e-05
                gamma1 =  -2.32526402314847814e-08
                gamma2 =  6.72649818538665569e-06
                # resistance = 7.82047033456383247e+02
                resistance = 5.50e+02

            output = [gamma0, gamma1, gamma2,
                       resistance]
            
        else:
            raise Exception('Invalid current option selected please choice from: doublePoly, poly, helmholtz, gouyChapman, stern, 2dPoly, 1st_order_poly, 2nd_order_poly')

        if self.protocol != 'DCV':
            # output.append(2.0*math.pi*solver.freq)
            # output.append(-0.03*math.pi)
            if self.capOption == 'doublePoly':

                if self.deltaepislon == 300E-3:

                    output.append(8.97784886690163297e+00)
                    # output.append(-4.71238898037763143e-01)
                else:
                    output.append(8.95927934895605382e+00)
                    # output.append(-4.71238898035948872e-01)
            
            elif self.capOption == '3rd_order_poly':
                if self.deltaepislon == 300E-3:

                    output.append(8.97785841843812626e+00)
                    # output.append(-4.71238898037768028e-01)

                else:

                    output.append(8.95927772626955843e+00 )
                    # output.append(-4.71238898037763088e-01)

            elif self.capOption == 'helmholtz':

                if self.deltaepislon == 300E-3:

                    output.append(8.97795544262888079e+00)
                    # output.append(-4.71238898038221832e-01)
                
                else:

                    output.append(8.95931535334210594e+00)
                    # output.append(-4.71238898038218668e-01)

            elif self.capOption == '1st_order_poly':

                if self.deltaepislon == 300E-3:

                    output.append(8.97788044891481718e+00)
                    # output.append(-4.71238898038450038e-01)
                
                else:

                    output.append(8.91805072054293468e+00)
                    # output.append(1.46096496446918134e-01 )
                
            elif self.capOption == '2nd_order_poly':

                if self.deltaepislon == 300E-3:

                    output.append(8.97784417447677185e+00)
                    # output.append(-4.71238898038220000e-01 )
                
                else:
                    output.append(8.95927942187071480e+00)
                    # output.append(-4.71238898038222498e-01)

            else:
                output.append(5.66441667236186319e+01/(2.0*math.pi))
                # output.append(-5.99345033706255026e-02)

            output.append(0.1)
        return output

    def suggested_capacitance_bounds(self):
            """Returns a list with suggested capacitance parameters for the model with dimension
            return: [gamma0, gamma1, gamma2, gamma3, gamma4, gamma5, gamma6,
                    gamma7, omega, mew, uncompensated_resistance]
            """

            if self.capOption == 'doublePoly':

                lower = [0.0, -1E-3, -1E-3, -1E-3, 0.0, -1E-3, -1E-3,
                        -1E-3, 1.0]
                if self.protocol != 'DCV':
                    lower.append(0.98*self.freq)
                    lower.append(-0.5*math.pi)

                upper = [1.0E-3, 1.0E-3, 1.0E-3, 1.0E-3, 1.0E-3, 1.0E-3, 1.0E-3,
                        1.0E-3, 1000.0]
                if self.protocol != 'DCV':
                    upper.append(1.02*self.freq)
                    upper.append(0.5*math.pi)

            
            elif self.capOption == '3rd_order_poly':


                lower = [0.0, -5.0E-4, -5.0E-4, -5.0E-4,
                        1.0]
                if self.protocol != 'DCV':
                    lower.append(0.98*self.freq)
                    lower.append(-0.5*math.pi)

                upper = [1.0E-3, 5.0E-4, 5.0E-4, 5.0E-4,
                         1000.0]
                if self.protocol != 'DCV':
                    upper.append(1.02*self.freq)
                    upper.append(0.5*math.pi)

                if self.resistance_cap_probability_likelhood == True:
                     
                    lower = [0.0, -5.0E-4, -5.0E-4, -5.0E-4,]
                    if self.protocol != 'DCV':
                        lower.append(0.98*self.freq)
                        lower.append(-0.5*math.pi)

                    upper = [1.0E-3, 5.0E-4, 5.0E-4, 5.0E-4]
                    if self.protocol != 'DCV':
                        upper.append(1.02*self.freq)
                        upper.append(0.5*math.pi)


            elif self.capOption == '4th_order_poly':


                lower = [0.0, -5.0E-5, -5.0E-5, -5.0E-5, -5.0E-5,
                        1.0]
                if self.protocol != 'DCV':
                    lower.append(0.98*self.freq)
                    lower.append(-0.25*math.pi)

                upper = [1.0E-3, 5.0E-5, 5.0E-5, 5.0E-5, 5.0E-5,
                         1000.0]
                if self.protocol != 'DCV':
                    upper.append(1.02*self.freq)
                    upper.append(0.25*math.pi)

                if self.resistance_cap_probability_likelhood == True:
                     
                    lower = [0.0, -5.0E-4, -5.0E-4, -5.0E-4, -5.0E-4]
                    if self.protocol != 'DCV':
                        lower.append(0.98*self.freq)
                        lower.append(-0.25*math.pi)

                    upper = [1.0E-3, 5.0E-4, 5.0E-4, 5.0E-4, 5.0E-4]
                    if self.protocol != 'DCV':
                        upper.append(1.02*self.freq)
                        upper.append(0.25*math.pi)

            elif self.capOption == '5th_order_poly':


                lower = [0.0, -5.0E-5, -5.0E-5, -5.0E-5, -5.0E-5, -5.0E-5,
                        1.0]
                if self.protocol != 'DCV':
                    lower.append(0.98*self.freq)
                    lower.append(-0.25*math.pi)

                upper = [1.0E-3, 5.0E-5, 5.0E-5, 5.0E-5, 5.0E-5, 5.0E-5,
                         1000.0]
                if self.protocol != 'DCV':
                    upper.append(1.02*self.freq)
                    upper.append(0.25*math.pi)

                if self.resistance_cap_probability_likelhood == True:
                     
                    lower = [0.0, -5.0E-6, -5.0E-6, -5.0E-6, -5.0E-6, -5.0E-6]
                    if self.protocol != 'DCV':
                        lower.append(0.98*self.freq)
                        lower.append(-0.25*math.pi)

                    upper = [1.0E-3, 5.0E-4, 5.0E-4, 5.0E-4, 5.0E-4, 5.0E-4]
                    if self.protocol != 'DCV':
                        upper.append(1.02*self.freq)
                        upper.append(0.25*math.pi)



            elif self.capOption == 'helmholtz':
                dielectricConstantlow = 1.0
                dielectricConstanthigh = 1000.0
                interplateSpacelow = 5E-11*10**2 # note interplate spacing is in cm
                interplateSpacehigh = 5E-6*10**2

                lower = [0.0,
                         1.0]
                if self.protocol != 'DCV':
                    lower.append(0.98*self.freq)
                    lower.append(-0.25*math.pi)
                print('lower: ', lower)
                upper = [5e-3,
                         1000.0]
                if self.protocol != 'DCV':
                    upper.append(1.02*self.freq)
                    upper.append(0.25*math.pi)
                print('upper: ', upper)

            elif self.capOption == 'gouyChapman':
                
                pzcLow = -1.1E-3
                pzcHigh = 1.1E3
                # fitting between a 0.01 ans 2.5 molar solution
                lower = [1.0, pzcLow,
                         10.0]
                if self.protocol != 'DCV':
                    lower.append(0.95*self.freq)
                    lower.append(-0.25*math.pi)

                upper = [100.0, pzcHigh,
                         140.0]
                if self.protocol != 'DCV':
                    upper.append(1.05*self.freq)
                    upper.append(0.25*math.pi)
            
            elif self.capOption == 'stern':
                dielectricConstantlow = 1.0
                dielectricConstanthigh = 100.0
                interplateSpacelow = 0.5E-11
                interplateSpacehigh = 0.5E-7
                # numberConclow = 6.0221409e+23*0.01*10**6
                # numberConchigh = 6.0221409e+23*2.5*10**6
                pzcLow = -1.1E-3
                pzcHigh = 1.1E3

                lower = [dielectricConstantlow, interplateSpacelow, pzcLow,
                         1.0]
                if self.protocol != 'DCV':
                    lower.append(0.95*self.freq)
                    lower.append(-0.25*math.pi)

                upper = [dielectricConstanthigh, interplateSpacehigh, pzcHigh,
                         140.0]
                if self.protocol != 'DCV':
                    upper.append(1.05*self.freq)
                    upper.append(0.25*math.pi)
            
            elif self.capOption == '2dPoly':

                # lower = [1.0E-5, -1.0E-6, 1.0E-9, 1.0E-8, 1.0E-6, -1.0E-8,
                #          0.95*solver.omega0, -0.15*math.pi, 1.0]

                # upper = [1.0E-4, -1.0E-5, 1.0E-8, 1.0E-7, 1.0E-5, -9.99E-6,
                #          1.05*solver.omega0, 0.15*math.pi, 140.0]

                # lower = [1.0E-4, 1.0E-5, 1.0E-8, 1.0E-6, 1.0E-5, -9.0E-6,  -9.0E-7,  1.0E-6,  -9.0E-7, -9.0E-7,
                #          0.01]

                div_bounds_lower = -1.0E-7
                div_bounds_upper = 1.0E-7

                lower = [1.0E-5, 1.0E-5 , div_bounds_lower, div_bounds_lower, 1.0E-5,
                         div_bounds_lower, 10]
                # lower = [0.0, -1.0E-3, -1.0E-3, -1.0E-3, -1.0E-3, -1.0E-3, -1.0E-3, -1.0E-3, -1.0E-3, -1.0E-3,
                #          0.01]
                if self.protocol != 'DCV':
                    lower.append(0.95*self.freq)
                    lower.append(-0.25*math.pi)

                # upper = [4.0E-4, 9.0E-3, 5.0E-9, 4.0E-7, 9.0E-3, -9.0E-7,  -9.0E-8,  9.0E-4,  -5.0E-9,  -5.0E-9,
                #          100.0]
                upper = [9.0E-3, 9.0E-3, div_bounds_upper, div_bounds_upper, 9.0E-3,
                         div_bounds_upper, 1000.0]
                # upper = [9.0E-4, 1.0E-3, 1.0E-3, 1.0E-3, 1.0E-3, 1.0E-3, 1.0E-3, 1.0E-3, 1.0E-3, 1.0E-3,
                #          100.0]
                if self.protocol != 'DCV':
                    upper.append(1.05*self.freq)
                    upper.append(0.25*math.pi)

                
            # (gamma0 + gamma1*self.epsilon_r + gamma2*depsilon_rdt + gamma3*depsilon_rdt*self.epsilon_r
            #         + gamma4*math.pow(self.epsilon_r, 2.0 )+ gamma5*math.pow(depsilon_rdt, 2.0 )
            #         + gamma6*math.pow(self.epsilon_r, 2.0 )*depsilon_rdt  + gamma7*math.pow(self.epsilon_r, 3.0 )
            #         + gamma8*math.pow(depsilon_rdt, 2.0 )*self.epsilon_r + gamma9*math.pow(depsilon_rdt, 3.0 ))*depsilon_rdt)


            # gamma0 + gamma1*self.epsilon_r + gamma2*math.pow(self.epsilon_r, 2.0 ) + gamma3*math.pow(self.epsilon_r, 3.0 ))*depsilon_rdt)
                    
            elif self.capOption == '1st_order_poly':

                lower = [0.0, -1.0E-4,
                        1.0]
                if self.protocol != 'DCV':
                    lower.append(0.98*self.freq)
                    lower.append(-0.25*math.pi)

                upper = [1.0E-3, 1.0E-4,
                         1000.0]
                if self.protocol != 'DCV':
                    upper.append(1.02*self.freq)
                    upper.append(0.25*math.pi)
            
            elif self.capOption == '2nd_order_poly':

                lower = [0.0, -1.0E-4, -1.0E-4,
                        1.0]
                if self.protocol != 'DCV':
                    lower.append(0.98*self.freq)
                    lower.append(-0.25*math.pi)

                upper = [1.0E-3, 1.0E-4, 1.0E-4,
                         1000.0]
                if self.protocol != 'DCV':
                    upper.append(1.02*self.freq)
                    upper.append(0.25*math.pi)


            else:
                raise Exception('Invalid current option selected please choice from: doublePoly, poly, helmholtz, gouyChapman, stern, 2dPoly, 1st_order_poly, or 2nd_order_poly')

            return (lower, upper)

    def reshape_to_cap_regions(self, array):

        raw = np.asarray(array)
        a = raw[:self.beingPureCapitanceto]
        b = raw[self.midCapLow:self.midCaphigh]
        c = raw[self.endCap:]
        reshaped = np.hstack((a,b,c))
        return reshaped


    def get_omega_0(self):

        solver = newtonRaphsonCapFaradaicDip(timeStepSize=self.timeStepSize, numberOfMeasurements=self.numberOfMeasurements,
                                  startPotential=self.startPotential, revPotential=self.revPotential,
                                  rateOfPotentialChange=self.rateOfPotentialChange, inital_current=self.inital_current,
                                  freq=self.freq, deltaepislon=self.deltaepislon,
                                  electrode_area =self.electrode_area, Z=self.Z, numberConc = self.numberConc,
                                  fit_alpha = self.fit_alpha)

        return solver.omega0

    def get_non_dimensionality_constants(self):

        solver = newtonRaphsonCapFaradaicDip(timeStepSize=self.timeStepSize, numberOfMeasurements=self.numberOfMeasurements,
                                  startPotential=self.startPotential, revPotential=self.revPotential,
                                  rateOfPotentialChange=self.rateOfPotentialChange, inital_current=self.inital_current,
                                  freq=self.freq, deltaepislon=self.deltaepislon,
                                  electrode_area =self.electrode_area, Z=self.Z, numberConc = self.numberConc,
                                  fit_alpha = self.fit_alpha)
        return [solver.E0, solver.T0, solver.I0]
    
    def parameter_order(self):

        if self.fit_faradaic_and_cap == True:
            return self.faradaic_and_capactiance_parameter_order()
        elif self.modelling_faradaic_current == True:
            return self.faradaic_parameter_order()
        else:
            return self.capacitance_parameter_order()

    def capacitance_parameter_order(self):
        """Returns a list with the names and order of the capacitance parameters for the model being used
        """


        if self.capOption == 'doublePoly':
            output = ['gamma0', 'gamma1', 'gamma2', 'gamma3', 'gamma4', 'gamma5', 'gamma6',
                      'gamma7','resistance', 'omega', 'phase', 'sigma']

        
        elif self.capOption == '3rd_order_poly':
            output = ['gamma0', 'gamma1', 'gamma2', 'gamma3','resistance', 'omega', 'phase', 'sigma']
            if self.resistance_cap_probability_likelhood == True:
                output = ['gamma0', 'gamma1', 'gamma2', 'gamma3', 'omega', 'phase', 'sigma']

        elif self.capOption == '4th_order_poly':
            output = ['gamma0', 'gamma1', 'gamma2', 'gamma3', 'gamma4', 'resistance', 'omega', 'phase', 'sigma']
            if self.resistance_cap_probability_likelhood == True:
                output = ['gamma0', 'gamma1', 'gamma2', 'gamma3', 'gamma4', 'omega', 'phase', 'sigma']

        elif self.capOption == '5th_order_poly':
            output = ['gamma0', 'gamma1', 'gamma2', 'gamma3', 'gamma4',  'gamma5', 'resistance', 'omega', 'phase', 'sigma']
            if self.resistance_cap_probability_likelhood == True:
                output = ['gamma0', 'gamma1', 'gamma2', 'gamma3', 'gamma4',  'gamma5', 'omega', 'phase', 'sigma']

        elif self.capOption == 'helmholtz':
            output = ['permittivityMedium (unitless)/interplateSpace (cm)',
                      'resistance', 'omega', 'phase', 'sigma']

        elif self.capOption == 'gouyChapman':
            output = ['dielectricConstant (unitless)', 'PZC',
                      'resistance', 'sigma', 'omega', 'phase', 'sigma']

        elif self.capOption == 'stern':
            output = ['dielectricConstant (unitless)', 'interplateSpace (cm)', 'PZC',
                      'resistance', 'omega', 'phase', 'sigma']
        
        elif self.capOption == '2dPoly':
            output = ['gamma0', 'gamma1', 'gamma2', 'gamma3', 'gamma4', 'gamma5',
                      'resistance', 'omega', 'phase', 'sigma']
            
        elif self.capOption == '1st_order_poly':
            output = ['gamma0', 'gamma1','resistance', 'omega', 'phase', 'sigma']

        elif self.capOption == '2nd_order_poly':
            output = ['gamma0', 'gamma1', 'gamma2','resistance', 'omega', 'phase', 'sigma']

        else:
            raise Exception('Invalid current option selected please choice from: doublePoly, poly, helmholtz, gouyChapman, stern, 2dPoly, 1st_order_poly, or 2nd_order_poly')


        return output
    
    
    def faradaic_parameter_order(self):
        """Returns a list with the names and order of the capacitance parameters for the model being used
        """

        output = ['kappa0', 'epsilon0', 'mu', 'surface_coverage']

        # output = ['kappa0', 'epsilon0', 'mu', 'resistance', 'surface_coverage']

        if self.fit_alpha == True:
            output.append('alpha')

        if self.modelling_faradaic_current == True:

                if self.modeling_dispersion_e is True:

                    output.append('epsilon0_std')
                    output[1]= 'epsilon0_mean'

                if self.modeling_dispersion_k is True:
                    output.append('kappa0_std')
                    output[0]= 'kappa0_mean'

                if self.modeling_dispersion_alpha is True:
                    output.append('alpha_std')
                    output[5]= 'alpha_mean'

        output.append('sigma')

        return output
    
    def faradaic_and_capactiance_parameter_order(self):
        """Returns a list with the names and order of the capacitance parameters for the model being used
        """


        output = ['kappa0', 'epsilon0', 'surface_coverage']

        if self.fit_alpha == True:
            output.append('alpha')

        if self.modelling_faradaic_current == True:

                if self.modeling_dispersion_e is True:

                    output.append('epsilon0_std')
                    output[1]= 'epsilon0_mean'

                if self.modeling_dispersion_k is True:
                    output.append('kappa0_std')
                    output[0]= 'kappa0_mean'

                if self.modeling_dispersion_alpha is True:
                    output.append('alpha_std')
                    output[5]= 'alpha_mean'


        if self.capOption == 'doublePoly':
            
            output.append('gamma0')
            output.append('gamma1')
            output.append('gamma2')
            output.append('gamma3')
            output.append('gamma4')
            output.append('gamma5')
            output.append('gamma6')
            output.append('gamma7')
            output.append('resistance')
            output.append('omega')
            output.append('phase')

        
        elif self.capOption == '3rd_order_poly':

            output.append('gamma0')
            output.append('gamma1')
            output.append('gamma2')
            output.append('gamma3')
            output.append('resistance')
            output.append('omega')
            output.append('phase')

        elif self.capOption == '4th_order_poly':

            output.append('gamma0')
            output.append('gamma1')
            output.append('gamma2')
            output.append('gamma3')
            output.append('gamma4')
            output.append('resistance')
            output.append('omega')
            output.append('phase')

        elif self.capOption == '5th_order_poly':

            output.append('gamma0')
            output.append('gamma1')
            output.append('gamma2')
            output.append('gamma3')
            output.append('gamma4')
            output.append('gamma5')
            output.append('resistance')
            output.append('omega')
            output.append('phase')

        elif self.capOption == 'helmholtz':
            
            output.append('gamma0')
            output.append('resistance')
            output.append('omega')
            output.append('phase')

        elif self.capOption == 'gouyChapman':
            
            output.append('dielectricConstant (unitless)')
            output.append('PZC')
            output.append('resistance')
            output.append('omega')
            output.append('phase')
            


        elif self.capOption == 'stern':
            
            output.append('dielectricConstant (unitless)')
            output.append('interplateSpace (cm)')
            output.append('PZC')
            output.append('resistance')
            output.append('omega')
            output.append('phase')
        
        elif self.capOption == '2dPoly':
            
            output.append('gamma0')
            output.append('gamma1')
            output.append('gamma2')
            output.append('gamma3')
            output.append('gamma4')
            output.append('gamma5')
            output.append('resistance')
            output.append('omega')
            output.append('phase')
            
        elif self.capOption == '1st_order_poly':

            output.append('gamma0')
            output.append('gamma1')
            output.append('resistance')
            output.append('omega')
            output.append('phase')

        elif self.capOption == '2nd_order_poly':
            output.append('gamma0')
            output.append('gamma1')
            output.append('gamma2')
            output.append('resistance')
            output.append('omega')
            output.append('phase')

        else:
            raise Exception('Invalid current option selected please choice from: doublePoly, poly, helmholtz, gouyChapman, stern, 2dPoly, 1st_order_poly, or 2nd_order_poly')
        

        output.append('sigma')
        return output
    
    def suggested_faradaic_parameters(self):
        '''kappa0, epsilon0, freq, mu, resistance, surface_coverage
        '''
       # return np.asarray([100,200.0E-3,self.freq, 0.01, 2.74802881816768547e+01, 4.0E-11])
        # theritcal_E0 = 240.0e-3 + 171.0e-3 - 179.0e-3

        # output = [100.0, 300.0e-3, 1.0e-01, 550.0, 1.0E-10]
        # output = [120, 0.270, 3.20138487053213794e-01, 6.000000e+02, 1.35e-10]
        output = [120, 0.270, 3.20138487053213794e-01,1.35e-10]

        # output = [120, 0.270, 3.20138487053213794e-012, 1.35e-10]
        # output = [100,theritcal_E0,-math.pi*0.1, 4.0E-11]
        # output = [100,theritcal_E0,-4.71238898038009502e-01, 4.0E-11, 2.74802881816768547e+01]
        # return np.asarray([100,200.0E-3,0.01, 2.74802881816768547e+01, 4.0E-11])
        # return np.asarray([100,200.0E-3, self.freq])
        # return np.asarray([100,200.0E-3, 8.941])
        # return np.asarray([100,200.0E-3, 0.01, 200.0, 4.0E-11])
        # return np.asarray([100,200.0E-3])

        if self.fit_alpha == True:
            output.append(0.5)
            # output.append(0.433)

        if self.modelling_faradaic_current == True:

                if self.modeling_dispersion_e is True:

                    output.append(45.0E-3)

                if self.modeling_dispersion_k is True:
                    # output.append(math.exp(0.5))
                    # output.append(math.exp(1.5))
                    output.append(1.5)

                if self.modeling_dispersion_alpha is True:
                   output.append(0.163)

        return np.asarray(output)

    def psv_paper_faradaic_parameters(self):
        '''kappa0, epsilon0, freq, mu, resistance, surface_coverage
        '''
        # return np.asarray([100,200.0E-3,self.freq, 0.01, 2.74802881816768547e+01, 4.0E-11])
        theritcal_E0 = 229.0e-3 + 171.0e-3 - 179.0e-3

        output = [123,theritcal_E0,873.0, 3.0820188438853916e-11]

        if self.fit_alpha == True:
            output.append[0.433]
    
        if self.modelling_faradaic_current == True:

                if self.modeling_dispersion_e is True:

                    output.append(50.0E-3)

                if self.modeling_dispersion_k is True:
                    output.append(math.exp(0.5))

                if self.modeling_dispersion_alpha is True:
                   output.append(0.1)

        return np.asarray(output)
    
    def suggested_faradaic_bounds(self):
        """Returns a list with suggested faradaic parameters for the model with dimension
        return: [kappa0, epsilon0, freq, mu, resistance, surface_coverage]
        """
        
        # 0.433-0.191 = 0.242
        theritcal_E0 = 229.0e-3 + 171.0e-3 - 242.0e-3
        E0_bound = 50.0E-3
        theritcal_E0 = 270.0E-3

        # E0_bound = 10.0E-3
        # theritcal_E0 = 277.0E-3

        # if self.deltaepislon == 150E-3:
        #     upper = [50.0,theritcal_E0+E0_bound, -math.pi*0.15, 250.0, 6E-11]
            
        #     lower = [320.0,theritcal_E0-E0_bound, math.pi*0.15, 1000.0, 1E-11]
        # elif self.deltaepislon == 295E-3:
        #     upper = [50.0,theritcal_E0+E0_bound,f -math.pi*0.15, 250.0, 6E-11]
            
        #     lower = [320.0,theritcal_E0-E0_bound, math.pi*0.15, 1000.0, 1E-11]

        # lower = [50.0,theritcal_E0-E0_bound, -math.pi*0.05, 10.0, 5.0E-11]
        
        # upper = [500.0,theritcal_E0+E0_bound, math.pi*0.15, 1000.0, 5E-10]

        # lower = [10.0,theritcal_E0-E0_bound, -math.pi*0.25, 10.0, 5E-11]
        
        # upper = [500.0,theritcal_E0+E0_bound, math.pi*0.25, 1000, 5E-10]
        # lower = [10.0,theritcal_E0-E0_bound, -math.pi*0.15, 10.0, 5E-11]
        
        # upper = [800.0,theritcal_E0+E0_bound, math.pi*0.15, 1500, 5E-10]

        # lower = [10.0,theritcal_E0-E0_bound, -math.pi*0.25, 5E-11]
        
        # upper = [500.0,theritcal_E0+E0_bound, math.pi*0.25, 5E-10]

        # E0_bound = 10.0E-3
        # theritcal_E0 = 292.7E-3

        upper = [500.0,theritcal_E0+E0_bound, math.pi*0.25,  5E-10]
        lower = [10.0,theritcal_E0-E0_bound, -math.pi*0.25, 5E-11]
        # upper = [500.0,theritcal_E0+E0_bound, math.pi*0.25, 1000.0, 5E-10]
        # lower = [10.0,theritcal_E0-E0_bound, -math.pi*0.25, 1.0, 5E-11]

        if self.fit_alpha == True:
            lower.append(0.2)
            upper.append(0.8)


        if self.modelling_faradaic_current == True:

                if self.modeling_dispersion_e is True:
                    E0_bound = 30.0E-3
                    theritcal_E0 = 285.0E-3
                    lower = [50.0,theritcal_E0-E0_bound, math.pi*-0.25, 1.0, 5E-10]
                    upper = [500.0,theritcal_E0+E0_bound, math.pi*0.25, 1000.0, 5E-11]
                    # lower = [50.0,theritcal_E0-E0_bound, math.pi*-0.25, 5E-10]
                    # upper = [500.0,theritcal_E0+E0_bound, math.pi*0.25, 5E-11]
                    lower.append(10.0E-3)
                    upper.append(100.0E-3)

                if self.modeling_dispersion_k is True:
                   lower.append(0.01)
                   upper.append(1.6)

                if self.modeling_dispersion_alpha is True:
                   lower.append(0.01)
                   upper.append(0.4)

        return (lower, upper)
    
    def suggested_faradaic_and_capactiance_bounds(self):

        theritcal_E0 = 229.0e-3 + 171.0e-3 - 242.0e-3
        theritcal_E0 = 310.0E-3
        E0_bound = 50.0E-3

        lower = [10.0,theritcal_E0-E0_bound,  5E-11]
        
        upper = [1000.0,theritcal_E0+E0_bound, 5E-10]

        if self.fit_alpha == True:
            lower.append(0.2)
            upper.append(0.8)

        if self.modelling_faradaic_current == True:
                
                if self.modeling_dispersion_e is True:
                    lower.append(10.0E-3)
                    upper.append(80.0E-3)

                if self.modeling_dispersion_k is True:
                   lower.append(0.01)
                   upper.append(1.6)

                if self.modeling_dispersion_alpha is True:
                   lower.append(0.01)
                   upper.append(0.4)

        if self.capOption == 'doublePoly':

            lower.append(0.0)
            lower.append(-1E-3)
            lower.append(-1E-3)
            lower.append(-1E-3)
            lower.append(0.0)
            lower.append(-1E-3)
            lower.append(-1E-3)
            lower.append(-1E-3)
            lower.append(1.0)
            
            if self.protocol != 'DCV':
                lower.append(0.98*self.freq)
                lower.append(-0.25*math.pi)
            
            upper.append(1.0E-3)
            upper.append(1.0E-3)
            upper.append(1.0E-3)
            upper.append(1.0E-3)
            upper.append(1.0E-3)
            upper.append(1.0E-3)
            upper.append(1.0E-3)
            upper.append(1.0E-3)
            upper.append(1.0E3)
            
            if self.protocol != 'DCV':
                upper.append(1.02*self.freq)
                upper.append(0.25*math.pi)

        
        elif self.capOption == '3rd_order_poly':

            lower.append(0.0)
            lower.append( -1.0E-5)
            lower.append( -1.0E-5)
            lower.append( -1.0E-5)
            lower.append(1.0)
            
            if self.protocol != 'DCV':
                lower.append(0.98*self.freq)
                lower.append(-0.25*math.pi)
            
            upper.append(1.0E-3)
            upper.append( 1.0E-5)
            upper.append( 1.0E-5)
            upper.append( 1.0E-5)
            upper.append(1.0E3)
            
            if self.protocol != 'DCV':
                upper.append(1.02*self.freq)
                upper.append(0.25*math.pi)


        elif self.capOption == 'helmholtz':
            lower.append(0.0)
            lower.append(1.0)
            
            if self.protocol != 'DCV':
                lower.append(0.98*self.freq)
                lower.append(-0.25*math.pi)
            
            upper.append(5.0E-3)
            upper.append(1.0E3)
            
            if self.protocol != 'DCV':
                upper.append(1.02*self.freq)
                upper.append(0.25*math.pi)

        elif self.capOption == 'gouyChapman':

            # FIXME:
            raise Exception('gouyChapman bounds need defining')
            
        
        elif self.capOption == 'stern':
             # FIXME:
            raise Exception('stern bounds need defining')
        
        elif self.capOption == '2dPoly':
            
            div_bounds_lower = -1.0E-7
            div_bounds_upper = 1.0E-7
            
            lower.append( 1.0E-5)
            lower.append( 1.0E-5)
            lower.append( div_bounds_lower)
            lower.append( div_bounds_lower)
            lower.append(  1.0E-5)
            lower.append(1.0)
            
            if self.protocol != 'DCV':
                lower.append(0.98*self.freq)
                lower.append(-0.25*math.pi)
            
            upper.append(9.0E-3)
            upper.append(9.0E-3)
            upper.append(div_bounds_upper)
            upper.append(div_bounds_upper)
            upper.append(9.0E-3)
            upper.append(1.0E3)
            
            if self.protocol != 'DCV':
                upper.append(1.02*self.freq)
                upper.append(0.25*math.pi)

                
        elif self.capOption == '1st_order_poly':

            lower.append(0.0)
            lower.append( -1.0E-5)
            lower.append(1.0)
            
            if self.protocol != 'DCV':
                lower.append(0.98*self.freq)
                lower.append(-0.25*math.pi)
            
            upper.append(1.0E-3)
            upper.append( 5.0E-5)
            upper.append(1.0E3)
            
            if self.protocol != 'DCV':
                upper.append(1.02*self.freq)
                upper.append(0.25*math.pi)
        
        elif self.capOption == '2nd_order_poly':

            lower.append(0.0)
            lower.append( -1.0E-5)
            lower.append( -1.0E-5)
            lower.append(1.0)
            
            if self.protocol != 'DCV':
                lower.append(0.98*self.freq)
                lower.append(-0.25*math.pi)
            
            upper.append(1.0E-3)
            upper.append( 5.0E-5)
            upper.append( 1.0E-5)
            upper.append(1.0E3)
            
            if self.protocol != 'DCV':
                upper.append(1.02*self.freq)
                upper.append(0.25*math.pi)

        else:
            raise Exception('Invalid current option selected please choice from: doublePoly, poly, helmholtz, gouyChapman, stern, 2dPoly, 1st_order_poly, or 2nd_order_poly')

        return (lower, upper)
        

    def suggested_bounds(self):

        if self.fit_faradaic_and_cap == True:
            return self.suggested_faradaic_and_capactiance_bounds()
        elif self.modelling_faradaic_current == True:
            return self.suggested_faradaic_bounds()
        else:
            return self.suggested_capacitance_bounds()
        
    
    def suggested_parameters(self):

        if self.fit_faradaic_and_cap == True:
            return self.suggested_faradaic_and_capactiance_params()
        elif self.modelling_faradaic_current == True:
            return self.suggested_faradaic_parameters()
        else:
            return self.suggested_capacitance_params()
        
    def suggested_faradaic_and_capactiance_params(self):
        '''kappa0, epsilon0, freq, mu, resistance, surface_coverage
        '''
         
        theritcal_E0 = 240.0e-3 + 171.0e-3 - 179.0e-3

        output = [100,300e-3, 1.0E-10] 
    
        if self.fit_alpha == True:
            output.append(0.5)
            # output.append(0.433)

        if self.modelling_faradaic_current == True:

                if self.modeling_dispersion_e is True:

                    output.append(50.0E-3)

                if self.modeling_dispersion_k is True:
                    # output.append(math.exp(0.5))
                    output.append(math.exp(1.5))

                if self.modeling_dispersion_alpha is True:
                   output.append(0.163)

        if self.capOption == 'doublePoly':


            if self.deltaepislon == 300E-3:

                output.append(1.75863380728210474e-05)
                output.append(7.93921078590131314e-06)
                output.append(1.12013976531365102e-05)
                output.append(-6.90042986767856747e-06)
                output.append(1.94076119947446042e-05)
                output.append(4.40653034129777181e-06)
                output.append(8.67255407443448707e-06)
                output.append( -4.03703922384588028e-06)
                output.append(450.0)
                # output.append(5.22609315334568805e+02)

            else:

                output.append(1.46317932258782763e-05)
                output.append(6.37695325874081814e-08)
                output.append(5.22600504148165723e-06)
                output.append(3.22007594244348196e-06)
                output.append(1.48102232617370742e-05)
                output.append(-1.66320856939575039e-06)
                output.append(6.08627865194329848e-06)
                output.append(2.99172212300043323e-06)
                output.append(450.0)
                # output.append(7.75409989510109995e+02) 

        
        elif self.capOption == '3rd_order_poly':
          
            if self.deltaepislon == 300E-3:
            
                output.append(1.85071072619651155e-05)
                output.append(6.20059590854927160e-06)
                output.append(9.92367826638328590e-06)
                output.append(-5.55238888434982238e-06)
                output.append(450.0)
                # output.append(5.58729755769305370e+02)

            else:
                output.append(1.47201819370179903e-05)
                output.append(-8.19989743760371676e-07) 
                output.append(5.61946210605756829e-06)
                output.append(3.18881138399119474e-06)
                output.append(450.0)
                # output.append(7.73588959680028438e+02)

        elif self.capOption == 'helmholtz':
            permittivityMedium = 10.0 # 78 approximately water at 25 celsius
            # interplateSpace = 0.5E-9*10**2 # 0.5 nM, note interplate spacing is in cm
            permittivityMedium = 78.0
            interplateSpace = 0.5E-6*10**2

            if self.deltaepislon == 300E-3:
                output.append(1.97574508240218726e-05)
                output.append(450.0)
                # output.append(9.99999999999985221e+02)
            else:
                output.append(1.58537716727587013e-05)
                output.append(450.0)
                # output.append(1.00000000000000000e+03)

        elif self.capOption == 'gouyChapman':
            asdf
        elif self.capOption == 'stern':
            asdf
        
        elif self.capOption == '2dPoly':

            if self.deltaepislon == 300E-3:

                output.append(1.75863380728210474e-05)
                output.append(7.93921078590131314e-06) 
                output.append(1.12013976531365102e-05)
                output.append( -6.90042986767856747e-06)
                output.append(1.94076119947446042e-05) 
                output.append(4.40653034129777181e-06)

                
                output.append(450.0)
                #output.append(5.22609315334568805e+02)

            else:

                pass
            
        elif self.capOption == '1st_order_poly':
            if self.deltaepislon == 300E-3:

                output.append(1.99886352827815220e-05)
                output.append(7.33142590812881527e-06)
                output.append(450.0)
                #output.append(6.47291302069237872e+02)

            else:

                output.append(1.49960874915495244e-05)
                output.append(3.57094637219130753e-06)
                output.append(450.0)
                #output.append(1.00000000032769343e+01)

            
        elif self.capOption == '2nd_order_poly':
            
            if self.deltaepislon == 300E-3:

                output.append(1.85093024943017783e-05)
                output.append(4.52256813112485882e-06)
                output.append(7.71210345224108554e-06)
                output.append(550.0)
                #output.append(5.07361394174784152e+02)

            else:

                output.append(1.47641080061587817e-05)
                output.append(-2.32526402314847814e-08)
                output.append(6.72649818538665569e-06)
                output.append(550.0)
                #output.append(7.82047033456383247e+02)

            
        else:
            raise Exception('Invalid current option selected please choice from: doublePoly, poly, helmholtz, gouyChapman, stern, 2dPoly, 1st_order_poly, 2nd_order_poly')

        if self.protocol != 'DCV':
            # output.append(2.0*math.pi*solver.freq)
            # output.append(-0.03*math.pi)
            if self.capOption == 'doublePoly':

                if self.deltaepislon == 300E-3:

                    output.append(8.97784886690163297e+00)
                    # output.append(-4.71238898037763143e-01)
                else:
                    output.append(8.95927934895605382e+00)
                    # output.append(-4.71238898035948872e-01)
            
            elif self.capOption == '3rd_order_poly':
                if self.deltaepislon == 300E-3:

                    output.append(8.97785841843812626e+00)
                    # output.append(-4.71238898037768028e-01)

                else:

                    output.append(8.95927772626955843e+00 )
                    # output.append(-4.71238898037763088e-01)

            elif self.capOption == 'helmholtz':

                if self.deltaepislon == 300E-3:

                    output.append(8.97795544262888079e+00)
                    # output.append(-4.71238898038221832e-01)
                
                else:

                    output.append(8.95931535334210594e+00)
                    # output.append(-4.71238898038218668e-01)

            elif self.capOption == '1st_order_poly':

                if self.deltaepislon == 300E-3:

                    output.append(8.97788044891481718e+00)
                    # output.append(-4.71238898038450038e-01)
                
                else:

                    output.append(8.91805072054293468e+00)
                    # output.append(1.46096496446918134e-01 )
                
            elif self.capOption == '2nd_order_poly':

                if self.deltaepislon == 300E-3:

                    output.append(8.97784417447677185e+00)
                    # output.append(-4.71238898038220000e-01 )
                
                else:
                    output.append(8.95927942187071480e+00)
                    # output.append(-4.71238898038222498e-01)

            else:
                output.append(5.66441667236186319e+01/(2.0*math.pi))
                # output.append(-5.99345033706255026e-02)

            

            output.append(0.1)
        return output


    
    def _FT(self, Data, half = True):
        """Fourier transforms given data

        param: Data data to Fourier transform and reduce
        return: numpy array contain fourier transformed data for harmonics 3 -12
        """
        sp = np.fft.fft(Data)
        # print('self.half_of_measuremnts: ', self.half_of_measuremnts)
        if half is True:
            sp = 2.0*sp # doubling amplitudes as it is split between -ve and +ve frequencies and we are going to discard negative frequencies
            sp_reduced = np.asarray(sp[:self.half_of_measuremnts],  dtype=complex)#discarding -ve frequencies
            return sp_reduced
        else:
            return np.asarray(sp)
        
    def _select_harmonic(self, Data, harmonic, spacing_ratio = 0.5):
        """Fourier transforms given data and reduces it to harmonics

        param: Data data to Fourier transform and reduce
        harmonic: The harmonic to be selected
        spacing_ratio: The amount of data either side of the harmonic to select
        as a multiple i.e 0.5 give 3.5 to 4.5
        return: numpy array contain fourier transformed data for harmonics 3 -12
        """
        
        if harmonic != 0 or harmonic != 0.0:
            sp = self._FT(Data, half = True)
        else:
            sp = self._FT(Data, half = False)

        # index location of harmonic 4-12
        x = np.where(self.half_frequencies  < (self.freq*(harmonic - spacing_ratio)))
        lower_index = x[0][-1]
        x = np.where(self.half_frequencies < (self.freq*(harmonic + spacing_ratio)))
        upper_index = x[0][-1]


        return sp[lower_index:upper_index]
    
    def harmonic_selector(self, exp_data, sim_data, harmonic, spacing_ratio = 0.5):
        """Fourier transforms given data and adds zero's to it to maintain
            the initial size of the data

        param: Data data to Fourier transform and reduce
        return: numpy array contain fourier transformed data for harmonics 3 -12
        """
        
        for index in range(0,2,1):

            data = [exp_data, sim_data]
            data = data[index]
            if harmonic != 0 or harmonic != 0.0:
                sp = self._FT(data)
            else:
                sp = self._FT(data, half = False)

            # index location of harmonic 4-12
            x = np.where(self.half_frequencies  < (self.freq*(harmonic - spacing_ratio)))
            lower_index = x[0][-1]
            x = np.where(self.half_frequencies < (self.freq*(harmonic + spacing_ratio)))
            upper_index = x[0][-1]
            x = np.where(self.half_frequencies < (self.freq*(harmonic)))
            mid_index = x[0][-1]

            if ((upper_index-lower_index)%2) == 0:
                if(mid_index-lower_index)>abs(upper_index-mid_index):
                    upper_index += 1
                else:
                    lower_index -= 1
      
            sp_reduced = sp[lower_index:upper_index]
            temp =  self.half_of_measuremnts - math.ceil(sp_reduced.shape[0]/2)

            # print('number added to each half of array: ', temp)
            # print('number of points in harmonic: ', sp_reduced.shape[0])
            mid_upper_sim_plot = sp[mid_index:upper_index]
            mid_upper_sim_plot = np.hstack((mid_upper_sim_plot, np.zeros(temp, dtype=complex)))
            lower_sim_plot = sp[lower_index:mid_index]
            lower_sim_plot = np.hstack((np.zeros(temp, dtype=complex), lower_sim_plot))
            
            array_for_iFFT = np.hstack((mid_upper_sim_plot, lower_sim_plot))
       
            if index == 0:
                exp_harmonic = np.fft.ifft(array_for_iFFT)
            if index == 1:
                sim_harmonic = np.fft.ifft(array_for_iFFT)


        return exp_harmonic, sim_harmonic
    
    def FT_and_reduce_to_fitted_harmonics(self, Data):
        """Fourier transforms given data and reduces it to harmonics 3 to 12

        param: Data data to Fourier transform and reduce
        return: numpy array contain fourier transformed data for harmonics 3 -12
        """
        sp_reduced = self._FT(Data=Data, half = True)
        # print('half of fourier transform size: ', sp_reduced.shape)
        output = np.asarray(sp_reduced[self.lower_index_harm_fit:self.upper_index_harm_fit],  dtype=complex)# reducing to harmonics 4 - 12
        # print('self.lower_indexOf4to12: ', self.lower_indexOf4to12)
        # print('self.upper_indexOf4to12: ', self.upper_indexOf4to12)
        # print('output size: ', output.shape)
        del(sp_reduced)
        return output
    
    def FT_reduce_to_fitted_harmonics_and_iff(self,Data):
        """Fourier transforms given data and reduces it to top hat filtred harmonics

        AND ifft back into time domain for fitting, as I think leaving it the
        frequency domain impacts the noise parameter (it should as it is
        frequency domain noise vs time domain noise)

        param: Data data to Fourier transform and reduce
        return: numpy array contain fourier transformed data for harmonics 3 -12
        """

        lower_true_harm_freq=(self.first_harm-self.fitting_range)*self.freq
        upper_true_harm_freq=(self.last_harm +self.fitting_range)*self.freq

        top_hat=np.where((self.full_frequencies<upper_true_harm_freq) & (self.full_frequencies>lower_true_harm_freq))

        sp = np.fft.fft(Data)
        sp = sp*2.0

        # top_hat filtering
        zeroed_array = np.zeros_like(Data, dtype="complex")
        # zeroed_array = np.zeros(Data.shape, dtype="complex")
        zeroed_array[top_hat] = sp[top_hat] 

        # ifft
        time_domian_fit_region = np.fft.ifft(zeroed_array)

        # plt.plot(self.times, np.real(time_domian_fit_region))
        # plt.show()

        return time_domian_fit_region

    def frequencies_for_fitted_harmonics(self):
        """Fourier transforms given data and reduces it to harmonics 3 to 12

        param: Data data to Fourier transform and reduce
        return: numpy array contain fourier transformed data for harmonics 3 -12
        """
        freq = self.half_frequencies[self.lower_index_harm_fit:self.upper_index_harm_fit] # reducing to harmonics 4 - 12
        return freq
    
    def set_harmonic_fitting_range(self,first_harm = 4, last_harm = 12, fitting_range = 0.25):
        """The upper and lower bounds of the harmonics to fit to.
        these can be the same harmonic. Note this can be zero, it will extended into negative frequencies
        and allow the fundamental harmonic to be selected.
        These should be intergers.

        first_harm: first/lowest desired harmonic
        last_harm: last/highest desired harmonic
        fitting_range: the space either side of the boundary harmonics as frequency*fitting_range
        """

        self.first_harm = first_harm
        self.last_harm = last_harm
        self.fitting_range = fitting_range

        # if self.freq != 0.0:
        #         # frequencies from FFT of times
        #         self.full_frequencies = np.fft.fftfreq(self.numberOfMeasurements, d=self.timeStepSize)
        #         self.half_frequencies=self.full_frequencies[:self.half_of_measuremnts]
        #         # index location of harmonic first_harm to last_harm
        #         x = np.where(self.full_frequencies  < (self.freq*(first_harm-fitting_range)))
        #         self.lower_index_harm_fit = x[0][-1]
        #         x = np.where(self.full_frequencies < (self.freq*(last_harm-fitting_range)))
        #         self.upper_index_harm_fit = x[0][-1]

        if self.freq != 0.0:
                # frequencies from FFT of times
                self.full_frequencies = np.fft.fftfreq(self.numberOfMeasurements, d=self.timeStepSize)
                self.half_frequencies=self.full_frequencies[:self.half_of_measuremnts]
                # index location of harmonic first_harm to last_harm
                x = np.where(self.half_frequencies  < (self.freq*(first_harm-fitting_range)))
                self.lower_index_harm_fit = x[0][-1]
                x = np.where(self.half_frequencies < (self.freq*(last_harm+fitting_range)))
                self.upper_index_harm_fit = x[0][-1]
    
    def plot_harmonic_region_fitted_for(self, Data, save_to = None):
        '''
        highlights the fitted region of the fourier transformed data
        '''
        full_spec = self._FT(Data, False)
        half_spec = self._FT(Data, True)
        fitting_region = self.FT_and_reduce_to_fitted_harmonics(Data)

        plt.figure(figsize=(18,10))
        plt.title('full spectrum fitting region')
        plt.plot(self.full_frequencies,abs(full_spec), 'r', label = 'full spectrum')
        plt.plot(self.frequencies_for_fitted_harmonics(),
                 abs(fitting_region/2.0), 'b', label = 'fitted region')
        plt.legend(loc='best')
        if save_to is None:
            plt.show()
        else:
            plt.savefig(os.path.join(save_to, ('_fitting_region_full_spec_.png')))
        plt.close()

        # if self.first_harm > 0:
        #     plt.figure(figsize=(18,10))
        #     plt.title('half spectrum fitting region')
        #     plt.plot(self.half_frequencies,abs(half_spec), 'r', label = 'half spectrum')
        #     plt.plot(self.full_frequencies[self.lower_index_harm_fit:self.upper_index_harm_fit],
        #             abs(2*full_spec[self.lower_index_harm_fit:self.upper_index_harm_fit]), 'b', label = 'fitted region')
        #     plt.legend(loc='best')
        #     if save_to is None:
        #         plt.show()
        #     else:
        #         plt.savefig(os.path.join(save_to, ('_fitting_region_half_spec_.png')))
        #     plt.close()

        plt.figure(figsize=(18,10))
        plt.title('half spectrum fitting region')
        plt.plot(self.half_frequencies,abs(half_spec), 'r', label = 'half spectrum')
        plt.plot(self.frequencies_for_fitted_harmonics(),
                abs(fitting_region), 'b', label = 'fitted region')
        plt.legend(loc='best')
        if save_to is None:
            plt.show()
        else:
            plt.savefig(os.path.join(save_to, ('_fitting_region_half_spec_.png')))
        plt.close()

        self.half_frequencies=self.full_frequencies[:self.half_of_measuremnts]
        # index location of harmonic first_harm to last_harm
        x = np.where(self.half_frequencies  < (self.freq*(self.last_harm+self.fitting_range*3.0)))
        len_to_plot = x[0][-1]

        plt.figure(figsize=(18,10))
        plt.title('reduced spectrum fitting region')
        plt.plot(self.half_frequencies[0:len_to_plot],abs(half_spec[0:len_to_plot]), 'r', label = 'reduced spectrum')
        plt.plot(self.frequencies_for_fitted_harmonics(),
                abs(fitting_region), 'b', label = 'fitted region')
        plt.legend(loc='best')
        if save_to is None:
            plt.show()
        else:
            plt.savefig(os.path.join(save_to, ('_fitting_region_reduced_spec_.png')))
        plt.close()



def change_cap_params_to_same_as_paper(parameter_list):
    """
    docstring
    """
    params = np.asarray(parameter_list)
    
    C_dl = params[:,0]
    params[:,1] = params[:,1]/C_dl
    params[:,2] = params[:,2]/C_dl
    params[:,3] = params[:,3]/C_dl

    return params

if __name__ == "__main__":
     
    # declaring fake times
    # delacring know model paramters


    # rateOfPotentialChange = 33.35E-3
    # freq = 8.88
    # deltaepislon = 150.0E-3

    # electrode_area = 0.07


    # inital_current =  -7.735012e-05
    # startPotential= -325.00E-3
    # # startPotential= 500.00E-3
    # inital_current = startPotential
    # revPotential = 725.00E-3
    # # revPotential = -500.00E-3

    rateOfPotentialChange = 23.35E-3
    freq = 8.88
    deltaepislon = 150.0E-3

    electrode_area = 0.07


    inital_current =  -7.735012e-05
    startPotential= -325.00E-3
    # startPotential= 500.00E-3
    inital_current = startPotential
    revPotential = 725.00E-3
    # revPotential = -500.00E-3

    # DCV limits fo stern
    # startPotential= 1.00E-3*-23.0
    # inital_current = startPotential
    # revPotential = 1.00E-3*23.0
    # FTacV limits fo stern
    # startPotential= 1.00E-3*-13.0
    # inital_current = startPotential
    # revPotential = 1.00E-3*13.0
    # deltaepislon = 1.00E-3*10.0

    # DCV limits for gouy-chapman
    # startPotential= 1.00E-3*-6.0
    # inital_current = startPotential
    # revPotential = 1.00E-3*6.0
    # ftacv limits for gouy-chapman
    # startPotential= 1.00E-3*-6.0
    # inital_current = startPotential
    # revPotential = 1.00E-3*6.0
    # deltaepislon = 1.00E-3*3.0

    numberConc = 6.0221409e+23*0.1*10**-3 # 6.0221409e+23*0.01*10**6
    #6.0221409e+23*2*10**6 #i.e a 2 molar solution converting to per cm^3

    measurements = 200000
    startT = 0.0#specify in seconds
    revT =  abs((revPotential - startPotential)/(rateOfPotentialChange))#specify in seconds
    endT = revT*2.0
    first_half_times = np.linspace(startT, revT, measurements)
    last_half_times = np.linspace(revT, endT, measurements)
    times = np.hstack((first_half_times, last_half_times[1:]))

    length = times.shape
    numberOfMeasurements = int(length[0])

    timeStepSize = times[-1]/(numberOfMeasurements - 1)

    fit_alpha = False

    # solver=newtonRaphsonCapFaradaicDip(timeStepSize=timeStepSize, numberOfMeasurements=numberOfMeasurements, startPotential=startPotential,
    #                         revPotential=revPotential, rateOfPotentialChange=rateOfPotentialChange, deltaepislon = deltaepislon,
    #                          inital_current=inital_current, numberConc=numberConc, electrode_area = electrode_area, freq = freq,
    #                          fit_alpha = fit_alpha)

    # cap_options_avaliable = ['doublePoly', 'poly', 'helmholtz', 'gouyChapman', 'stern', '2dPoly', '1st_order_poly', '2nd_order_poly']
    # cap = cap_options_avaliable[2]
    protocol = 'FTACV'
    # protocol = 'DCV'
    # solver.set_experimental_protocol(protocol)
    # solver.set_solver_option('my_newton_raphson')

    # model = wrappedNewtonCapFaradaicDip(times, startPotential=startPotential, revPotential=revPotential,
    #                          rateOfPotentialChange=rateOfPotentialChange, deltaepislon = deltaepislon,
    #                          inital_current=inital_current, protocol= protocol, electrode_area = electrode_area,
    #                          numberConc=numberConc, freq = freq)
    # model.set_cap_model(cap)
    # parameters = np.asarray(model.suggested_capacitance_params())
    # modelling_faradaic_current = True
    # model.set_modelling_faradaic_current(modelling_faradaic_current)
    # model.set_modeling_experimental_data(False)

    # # model.set_solver_option('scipy_newton_raphson')
    # # model.set_solver_option('my_riddlers')
    # model.set_solver_option('my_newton_raphson')

    # print('parameters: ', parameters)

    # if model.modelling_faradaic_current == True:
    #     cap_parameters = np.asarray(model.suggested_capacitance_params())
    #     model.set_capacitance_params(cap_parameters)
    #     parameters = model.suggested_faradaic_parameters()

    #     # model.set_dispersion_modeling(True)
    
    # print('capoption: ',model.capOption)


    # non_dim_constants = model.get_non_dimensionality_constants()
    
    # print('I0: ', non_dim_constants[2])
    # print('T0: ', non_dim_constants[1])
    # print('E0: ', non_dim_constants[0])


    # if cap == '1st_order_poly':
    #     cap_model = 'poly'
    # elif cap == '2nd_order_poly':
    #     cap_model = 'poly'
    # else:
    #     cap_model = cap

    # # solver.set_capacitance_params(parameters, cap_model)
    # # solving using newtonRaphsonFT
    # # og_i = solver.solve(times, cap)
    # # og_potential = solver.get_applied_potential_and_driv()
    # # og_appliedPotential = og_potential[0]
    # # og_drivAppliedPotential = og_potential[1]
        
    # print('my newton solver: ')
    # tic = time.perf_counter()
    # og_i = model.simulate(parameters, times)
    # toc = time.perf_counter()
    # print(f"newton solver completed in : {toc - tic:0.10f} seconds")
    # og_appliedPotential = model.appliedPotential
    # og_drivAppliedPotential = model.drivAppliedPotential

    # # print('RUNNING SECOND SIM')
    # # parameters = [1.850711e-05, 6.200596e-06, 5.587298e+02, 8.977858e+00, -4.712389e-01]
    # # parameters = [1.850711e-05, -1.0e-05, 5.587298e+02, 8.977858e+00, -4.712389e-01]
    # # print('parameters: ', parameters)
    # # og_i = model.simulate(parameters, times)

    # # base_directory_output = os.path.dirname(os.path.realpath(__file__))
    # # file = 'trail_faradaic_fit.txt'
    # # np.

    # print('cap_and_faradaic_toghter solver: ')
    # model.set_fitting_faradaic_and_capactiance_current(True)
    # parameters = model.suggested_parameters()
    # tic = time.perf_counter()
    # cap_and_faradaic_toghter_og_i = model.simulate(parameters, times)
    # toc = time.perf_counter()
    # print(f"cap_andfaradaic_toghter solver completed in : {toc - tic:0.10f} seconds")


    # plt.figure()
    # plt.plot(times[:], og_i[:], 'r', label = 'just faradaic')
    # plt.plot(times[:], cap_and_faradaic_toghter_og_i[:], 'b', label = 'cap_and_faradaic_toghter_og_i')
    # plt.xlabel("times/s")
    # plt.ylabel("current/A")
    # plt.show()
    # plt.close()

    # plt.figure()
    # plt.plot(times[:], cap_and_faradaic_toghter_og_i[:], 'b', label = 'cap_and_faradaic_toghter_og_i')
    # plt.plot(times[:], og_i[:], 'r', label = 'just faradaic')
    # plt.xlabel("times/s")
    # plt.ylabel("current/A")
    # plt.show()
    # plt.close()

      # ******** testing riddlers below this *********

    model = wrappedNewtonCapFaradaicDip(times, startPotential=startPotential, revPotential=revPotential,
                             rateOfPotentialChange=rateOfPotentialChange, deltaepislon = deltaepislon,
                             inital_current=inital_current, protocol= protocol, electrode_area = electrode_area,
                             numberConc=numberConc, freq = freq)
    cap_options_avaliable = ['doublePoly', '3rd_order_poly', 'helmholtz', 'gouyChapman', 'stern', '2dPoly', '1st_order_poly', '2nd_order_poly', '4th_order_poly', '5th_order_poly']
    cap = cap_options_avaliable[1]
    model.set_cap_model(cap)
    parameters = np.asarray(model.suggested_capacitance_params())
    parameters = [1.082e-05,-1.653e-07,4.823e-06,5.939e-07,100.0, 8.959e+00,-3.014e-01]
    # parameters = [0.0, 0.0, 0.0,0.0, 100.0, 8.977926e+00, -3.628170e-01]
    modelling_faradaic_current = False
    model.set_modelling_faradaic_current(modelling_faradaic_current)
    model.set_modeling_experimental_data(False)

    # model.set_solver_option('scipy_newton_raphson')
    # model.set_solver_option('my_riddlers')
    model.set_solver_option('my_newton_raphson')

    # parameters[1] = 550.0
    # print('parameters: ', parameters)

    # if model.modelling_faradaic_current == True:
    #     cap = cap_options_avaliable[1]
    #     model.set_cap_model(cap)
    #     cap_parameters = np.asarray(model.suggested_capacitance_params())
    #     model.set_capacitance_params(cap_parameters)
    #     parameters = model.suggested_faradaic_parameters()

    #     # model.set_dispersion_modeling(True)
    
    # print('capoption: ',model.capOption)


    # non_dim_constants = model.get_non_dimensionality_constants()
    
    # print('I0: ', non_dim_constants[2])
    # print('T0: ', non_dim_constants[1])
    # print('E0: ', non_dim_constants[0])


    # # if cap == '1st_order_poly':
    # #     cap_model = 'poly'
    # # elif cap == '2nd_order_poly':
    # #     cap_model = 'poly'
    # # else:
    # #     cap_model = cap

    # print('my newton solver: ')
    # # model.set_solver_option('my_riddlers')
    # tic = time.perf_counter()
    # og_i = model.simulate(parameters, times)
    # toc = time.perf_counter()
    # print(f"newton poly 3rd order solver completed in : {toc - tic:0.10f} seconds")
    # og_appliedPotential = model.appliedPotential
    # og_drivAppliedPotential = model.drivAppliedPotential

    # # np.save(file = 'poly_3rd_order', arr = og_i)
    # og_poly_3rd_order = np.load(file = 'poly_3rd_order.npy')

    # plt.figure()
    # plt.plot(times[:], og_i[:], 'r', label = 'new')
    # plt.plot(times[:], og_poly_3rd_order[:], 'b', label = 'og')
    # plt.xlabel("times/s")
    # plt.ylabel("current/A")
    # plt.show()
    # plt.close()

    # plt.figure()
    # plt.plot(times[:], og_poly_3rd_order[:], 'b', label = 'og')
    # plt.plot(times[:], og_i[:], 'r', label = 'new')
    # plt.xlabel("times/s")
    # plt.ylabel("current/A")
    # plt.show()
    # plt.close()

    # if model.modelling_faradaic_current == True:
    #     cap = cap_options_avaliable[-4]
    #     model.set_cap_model(cap)
    #     cap_parameters = np.asarray(model.suggested_capacitance_params())
    #     model.set_capacitance_params(cap_parameters)
    #     parameters = model.suggested_faradaic_parameters()

    #     # model.set_dispersion_modeling(True)
    
    # print('capoption: ',model.capOption)


    # non_dim_constants = model.get_non_dimensionality_constants()
    
    # print('I0: ', non_dim_constants[2])
    # print('T0: ', non_dim_constants[1])
    # print('E0: ', non_dim_constants[0])


    # # if cap == '1st_order_poly':
    # #     cap_model = 'poly'
    # # elif cap == '2nd_order_poly':
    # #     cap_model = 'poly'
    # # else:
    # #     cap_model = cap

    # print('my '+cap+' solver: ')
    # # model.set_solver_option('my_riddlers')
    # tic = time.perf_counter()
    # og_i = model.simulate(parameters, times)
    # toc = time.perf_counter()
    # print(f"newton poly 1st order solver completed in : {toc - tic:0.10f} seconds")
    # og_appliedPotential = model.appliedPotential
    # og_drivAppliedPotential = model.drivAppliedPotential

    # # np.save(file = 'poly_1st_order', arr = og_i)
    # poly_1st_order = np.load(file = 'poly_1st_order.npy')

    # plt.figure()
    # plt.plot(times[:], og_i[:], 'r', label = 'new')
    # plt.plot(times[:], poly_1st_order[:], 'b', label = 'og')
    # plt.xlabel("times/s")
    # plt.ylabel("current/A")
    # plt.show()
    # plt.close()

    # plt.figure()
    # plt.plot(times[:], poly_1st_order[:], 'b', label = 'og')
    # plt.plot(times[:], og_i[:], 'r', label = 'new')
    # plt.xlabel("times/s")
    # plt.ylabel("current/A")
    # plt.show()
    # plt.close()

    # if model.modelling_faradaic_current == True:
    #     cap = cap_options_avaliable[-3]
    #     model.set_cap_model(cap)
    #     cap_parameters = np.asarray(model.suggested_capacitance_params())
    #     model.set_capacitance_params(cap_parameters)
    #     parameters = model.suggested_faradaic_parameters()

    #     # model.set_dispersion_modeling(True)
    
    # print('capoption: ',model.capOption)


    # non_dim_constants = model.get_non_dimensionality_constants()
    
    # print('I0: ', non_dim_constants[2])
    # print('T0: ', non_dim_constants[1])
    # print('E0: ', non_dim_constants[0])


    # # if cap == '1st_order_poly':
    # #     cap_model = 'poly'
    # # elif cap == '2nd_order_poly':
    # #     cap_model = 'poly'
    # # else:
    # #     cap_model = cap

    # print('my '+cap+' solver: ')
    # # model.set_solver_option('my_riddlers')
    # tic = time.perf_counter()
    # og_i = model.simulate(parameters, times)
    # toc = time.perf_counter()
    # print(f"newton poly 2nd order solver completed in : {toc - tic:0.10f} seconds")
    # og_appliedPotential = model.appliedPotential
    # og_drivAppliedPotential = model.drivAppliedPotential

    # # np.save(file = 'poly_2nd_order', arr = og_i)
    # poly_2nd_order = np.load(file = 'poly_2nd_order.npy')

    # plt.figure()
    # plt.plot(times[:], og_i[:], 'r', label = 'new')
    # plt.plot(times[:], poly_2nd_order[:], 'b', label = 'og')
    # plt.xlabel("times/s")
    # plt.ylabel("current/A")
    # plt.show()
    # plt.close()

    # plt.figure()
    # plt.plot(times[:], poly_2nd_order[:], 'b', label = 'og')
    # plt.plot(times[:], og_i[:], 'r', label = 'new')
    # plt.xlabel("times/s")
    # plt.ylabel("current/A")
    # plt.show()
    # plt.close()

    if model.modelling_faradaic_current == True:
        cap = cap_options_avaliable[1]
        model.set_cap_model(cap)
        # cap_parameters = np.asarray(model.suggested_capacitance_params())
        cap_parameters = parameters
        model.set_capacitance_params(cap_parameters)
        parameters = model.suggested_faradaic_parameters()

        # model.set_dispersion_modeling(True)
    
    print('capoption: ',model.capOption)


    non_dim_constants = model.get_non_dimensionality_constants()
    
    print('I0: ', non_dim_constants[2])
    print('T0: ', non_dim_constants[1])
    print('E0: ', non_dim_constants[0])


    # if cap == '1st_order_poly':
    #     cap_model = 'poly'
    # elif cap == '2nd_order_poly':
    #     cap_model = 'poly'
    # else:
    #     cap_model = cap

    print('my '+cap+' solver: ')
    # model.set_solver_option('my_riddlers')
    tic = time.perf_counter()
    og_i = model.simulate(parameters, times)
    toc = time.perf_counter()
    print(f"newton poly 2nd order solver completed in : {toc - tic:0.10f} seconds")
    og_appliedPotential_resistnace_is_100 = model.appliedPotential
    og_drivAppliedPotential_resistnace_is_100 = model.drivAppliedPotential
    og_episilon_IR_drop_componenet_resistnace_is_100 = model.episilon_IR_drop_componenet
    og_episilon_without_drop_resistnace_is_100 = model.episilon_without_drop

    # plt.figure()
    # plt.plot(times[:], og_i[:]* non_dim_constants[0], 'r', label = 'new')
    # plt.xlabel("times/s")
    # plt.ylabel("current/A")
    # plt.show()
    # plt.close()
    # plt.figure()
    # plt.plot(times[1:], og_episilon_IR_drop_componenet[1:], 'r', label = 'IR drop componenet')
    # plt.plot(times[1:], og_episilon_without_drop[1:], 'b', label = 'Applied Potenetial')
    # plt.xlabel("Time/s")
    # plt.ylabel('non-dimensional')
    # plt.legend(loc = 'best')
    # plt.show()
    # plt.close()
    # plt.figure()
    # plt.plot(times[1:], og_episilon_without_drop[1:]*non_dim_constants[0], 'b', label = 'og_episilon_without_drop')
    # plt.xlabel("times/s")
    # plt.ylabel("current/A")
    # plt.show()
    # plt.close()

    # if model.modelling_faradaic_current == True:
    #     cap = cap_options_avaliable[-1]
    #     model.set_cap_model(cap)
    #     cap_parameters = np.asarray(model.suggested_capacitance_params())
    #     model.set_capacitance_params(cap_parameters)
    #     parameters = model.suggested_faradaic_parameters()

    #     # model.set_dispersion_modeling(True)
    
    # print('capoption: ',model.capOption)


    # non_dim_constants = model.get_non_dimensionality_constants()
    
    # print('I0: ', non_dim_constants[2])
    # print('T0: ', non_dim_constants[1])
    # print('E0: ', non_dim_constants[0])


    # # if cap == '1st_order_poly':
    # #     cap_model = 'poly'
    # # elif cap == '2nd_order_poly':
    # #     cap_model = 'poly'
    # # else:
    # #     cap_model = cap

    # print('my '+cap+' solver: ')
    # # model.set_solver_option('my_riddlers')
    # tic = time.perf_counter()
    # og_i = model.simulate(parameters, times)
    # toc = time.perf_counter()
    # print(f"newton poly 2nd order solver completed in : {toc - tic:0.10f} seconds")
    # og_appliedPotential = model.appliedPotential
    # og_drivAppliedPotential = model.drivAppliedPotential

    # plt.figure()
    # plt.plot(times[:], og_i[:], 'r', label = 'new')
    # plt.xlabel("times/s")
    # plt.ylabel("current/A")
    # plt.show()
    # plt.close()

    modelling_faradaic_current = True
    model.set_modelling_faradaic_current(modelling_faradaic_current)
    model.set_modeling_experimental_data(False)

    if model.modelling_faradaic_current == True:
        cap = cap_options_avaliable[1]
        model.set_cap_model(cap)
        cap_parameters = parameters
        model.set_capacitance_params(cap_parameters)
        parameters = model.suggested_faradaic_parameters()

    print('my '+cap+' solver: ')
    # model.set_solver_option('my_riddlers')
    tic = time.perf_counter()
    faradaic_og_i = model.simulate(parameters, times)
    toc = time.perf_counter()
    print(f"newton poly 2nd order solver completed in : {toc - tic:0.10f} seconds")
    faradaic_og_appliedPotential_resistnace_is_100 = model.appliedPotential
    faradaic_og_drivAppliedPotential_resistnace_is_100 = model.drivAppliedPotential
    faradaic_og_episilon_IR_drop_componenet_resistnace_is_100 = model.episilon_IR_drop_componenet
    faradaic_og_episilon_without_drop_resistnace_is_100 = model.episilon_without_drop

    plt.figure()
    # plt.plot(times[1:], faradaic_og_appliedPotential[1:], 'g', label = 'farradaic applied potenetial')
    plt.plot(times[1:], og_appliedPotential_resistnace_is_100[1:], 'k', label = 'applied potenetial res 100')
    plt.plot(times[1:], faradaic_og_episilon_IR_drop_componenet_resistnace_is_100[1:], 'b', label = 'IR_drop_componenet_cap_and_faradaic_model res 100')
    plt.plot(times[1:], og_episilon_IR_drop_componenet_resistnace_is_100[1:], 'r', label = 'IR_drop_componenet_cap_only_model res 100')
    plt.xlabel("Time/s")
    plt.ylabel('non-dimensional')
    plt.legend(loc = 'best')
    plt.show()
    plt.close()


    model = wrappedNewtonCapFaradaicDip(times, startPotential=startPotential, revPotential=revPotential,
                             rateOfPotentialChange=rateOfPotentialChange, deltaepislon = deltaepislon,
                             inital_current=inital_current, protocol= protocol, electrode_area = electrode_area,
                             numberConc=numberConc, freq = freq)
    cap_options_avaliable = ['doublePoly', '3rd_order_poly', 'helmholtz', 'gouyChapman', 'stern', '2dPoly', '1st_order_poly', '2nd_order_poly', '4th_order_poly', '5th_order_poly']
    cap = cap_options_avaliable[1]
    model.set_cap_model(cap)
    parameters = np.asarray(model.suggested_capacitance_params())
    parameters = [1.082e-05,-1.653e-07,4.823e-06,5.939e-07,1000.0, 8.959e+00,-3.014e-01]
    # parameters = [0.0, 0.0, 0.0,0.0, 100.0, 8.977926e+00, -3.628170e-01]
    modelling_faradaic_current = False
    model.set_modelling_faradaic_current(modelling_faradaic_current)
    model.set_modeling_experimental_data(False)

    # model.set_solver_option('scipy_newton_raphson')
    # model.set_solver_option('my_riddlers')
    model.set_solver_option('my_newton_raphson')



   


    if model.modelling_faradaic_current == True:
        cap = cap_options_avaliable[1]
        model.set_cap_model(cap)
        # cap_parameters = np.asarray(model.suggested_capacitance_params())
        cap_parameters = parameters
        model.set_capacitance_params(cap_parameters)
        parameters = model.suggested_faradaic_parameters()

        # model.set_dispersion_modeling(True)
    
    print('capoption: ',model.capOption)


    non_dim_constants = model.get_non_dimensionality_constants()
    
    print('I0: ', non_dim_constants[2])
    print('T0: ', non_dim_constants[1])
    print('E0: ', non_dim_constants[0])

    print('my '+cap+' solver: ')
    # model.set_solver_option('my_riddlers')
    tic = time.perf_counter()
    og_i = model.simulate(parameters, times)
    toc = time.perf_counter()
    print(f"newton poly 2nd order solver completed in : {toc - tic:0.10f} seconds")
    og_appliedPotential_resistnace_is_1000 = model.appliedPotential
    og_drivAppliedPotential_resistnace_is_1000 = model.drivAppliedPotential
    og_episilon_IR_drop_componenet_resistnace_is_1000 = model.episilon_IR_drop_componenet
    og_episilon_without_drop_resistnace_is_1000 = model.episilon_without_drop


    modelling_faradaic_current = True
    model.set_modelling_faradaic_current(modelling_faradaic_current)
    model.set_modeling_experimental_data(False)

    if model.modelling_faradaic_current == True:
        cap = cap_options_avaliable[1]
        model.set_cap_model(cap)
        cap_parameters = parameters
        model.set_capacitance_params(cap_parameters)
        parameters = model.suggested_faradaic_parameters()

    print('my '+cap+' solver: ')
    # model.set_solver_option('my_riddlers')
    tic = time.perf_counter()
    faradaic_og_i = model.simulate(parameters, times)
    toc = time.perf_counter()
    print(f"newton poly 2nd order solver completed in : {toc - tic:0.10f} seconds")
    faradaic_og_appliedPotential_resistnace_is_1000 = model.appliedPotential
    faradaic_og_drivAppliedPotential_resistnace_is_1000 = model.drivAppliedPotential
    faradaic_og_episilon_IR_drop_componenet_resistnace_is_1000 = model.episilon_IR_drop_componenet
    faradaic_og_episilon_without_drop_resistnace_is_1000 = model.episilon_without_drop

    plt.figure()
    # plt.plot(times[1:], faradaic_og_appliedPotential[1:], 'g', label = 'farradaic applied potenetial')
    plt.plot(times[1:], og_appliedPotential_resistnace_is_1000[1:], 'k', label = 'applied potenetial res 1000')
    plt.plot(times[1:], faradaic_og_episilon_IR_drop_componenet_resistnace_is_1000[1:], 'b', label = 'IR_drop_componenet_cap_and_faradaic_model res 1000')
    plt.plot(times[1:], og_episilon_IR_drop_componenet_resistnace_is_1000[1:], 'r', label = 'IR_drop_componenet_cap_only_model res 1000')
    plt.xlabel("Time/s")
    plt.ylabel('non-dimensional')
    plt.legend(loc = 'best')
    plt.show()
    plt.close()


    plt.figure()
    # plt.plot(times[1:], faradaic_og_appliedPotential[1:], 'g', label = 'farradaic applied potenetial')
    plt.plot(times[1:], og_appliedPotential_resistnace_is_1000[1:], 'k', label = 'applied potenetial res 1000')
    plt.plot(times[1:], og_appliedPotential_resistnace_is_100[1:], 'm', label = 'applied potenetial res 100')

    plt.plot(times[1:], faradaic_og_episilon_IR_drop_componenet_resistnace_is_1000[1:], 'b', label = 'IR_drop_componenet_cap_and_faradaic_model res 1000')
    plt.plot(times[1:], og_episilon_IR_drop_componenet_resistnace_is_1000[1:], 'r', label = 'IR_drop_componenet_cap_only_model res 1000')
    plt.plot(times[1:], faradaic_og_episilon_IR_drop_componenet_resistnace_is_100[1:], 'g', label = 'IR_drop_componenet_cap_and_faradaic_model res 100')
    plt.plot(times[1:], og_episilon_IR_drop_componenet_resistnace_is_100[1:], 'orange', label = 'IR_drop_componenet_cap_only_model res 100')
    plt.xlabel("Time/s")
    plt.ylabel('non-dimensional')
    plt.legend(loc = 'best')
    plt.show()
    plt.close()

    