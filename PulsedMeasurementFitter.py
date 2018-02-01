#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 19:11:55 2018

@author: shaun
"""

import numpy as np
from StochasticModelSolver import continuousStochasticSolver
import csv
import matplotlib.pyplot as plt
from scipy import optimize




class NV5LevelFitter(object):
    
    """
    Class for fitting the pulsed measurement experiments.
    photodynamics of the NV centre can be observed through the short-time (3us)
    fluorescence of the NV centre under laser. The system can be considered to stochastically
    change from different energy levels, and it's population dynamics
    evolve according to a transition matrix. Different energy levels 
    in the NV centre have different temporal and fluorescent properties (some have long lifetimes 
    , some don't fluoresce). By considering the populationdynamics, the fluorescence of the NV centre 
    can be found and the collected pulsed measurement experiment can be fitted and the corresponding 
    parameters associated with the initial energy levels of the NV centre prior to laser irradiation
    can be found.
    """
    
    def __init__(self,file):
        
        
        self.t=np.array([])
        self.y=np.array([])
        """
        Data comes as two separate pulses, _pulsedMeasurementFormatter() instantiates ti and yi
        where i represents the pulse index
        """
        self.t0 = None
        self.y0 = None
        self.t1 = None
        self.y1 = None
        
        """
        Define transition rates, rates taken from:
        Nitrogen-vacancy center in diamond: Model of the electronic structure and associated dynamics
        https://journals.aps.org/prb/abstract/10.1103/PhysRevB.74.104303
        """
        self.gamma = 77.0
        self.epsilon = 3.0/77.0
        self.GamD = 3.3
        self.k53 =60.0
        
        with open(file,"r") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:

                self.t = np.append(self.t,np.array([np.float64(row[0])/1000]))
                self.y = np.append(self.y,np.array([np.float64(row[1])]))
                
        self._pulsedMeasurementFormatter()     
        
                
    def _pulsedMeasurementFormatter(self):
        "Find edges using differences"
        edges = abs(self.y[2:]-self.y[:-2])
        edges = np.concatenate((np.array([0]),edges,np.array([0])))
        "Expect 4 edges since there are two pulses"
        edgesindex=np.sort(np.argsort(edges)[-4:])
        "self.t0, and self.t1 are an array representing the cutout timebins of the whole pulsed measurement"
        self.tstep=(self.t[1]-self.t[0])
        self.t0=self.t[edgesindex[0]+1:edgesindex[1]-1]
        self.y0=self.y[edgesindex[0]+1:edgesindex[1]-1]
        self.t1=self.t[edgesindex[2]+1:edgesindex[3]-1]
        self.y1=self.y[edgesindex[2]+1:edgesindex[3]-1]
        
                
    
                
    def _defineMatrix(self,power):
        """
        
        4 ---- <- 
                  Excited state spins
        3 ---- <- 
        
        
                5 ---- <- Singlet state
                    
                
                
                
                
                
        2 ---- <- 
                  Ground state spins
        1 ---- <- 
        
        
        self.mat[i,j] = K_ij = TransitionRate(i->j)
        
        """
        
        R = 77*power
    
        self.mat=np.array([[-R*(1+self.epsilon), self.gamma, self.GamD, 0, self.gamma*self.epsilon], \
        [R, -self.gamma*(1+self.epsilon), 0, self.epsilon*R, 0], \
        [0, 0, -self.GamD, 0, self.k53], \
        [0, self.gamma*self.epsilon, 0, -R*(1+self.epsilon), self.gamma], \
        [R*self.epsilon, 0, 0, R, -self.gamma*(1 + self.epsilon) - self.k53 ]])

        self.mat=np.array([[-R*(1+self.epsilon),0,self.gamma,self.gamma*self.epsilon,self.GamD],\
                            [0,-R*(1+self.epsilon),self.gamma*self.epsilon,self.gamma,0],\
                            [R,R*self.epsilon,-self.gamma*(1+self.epsilon),0,0],\
                            [R*self.epsilon,R,0,-self.gamma*(1+self.epsilon)-self.k53,0],\
                            [0,0,0,self.k53,-self.GamD]])

        self.n = np.shape(self.mat)[0]
        

        
    
    def fluorescenceFunction(self,params,t, plot = False):

        "params = [power,initstatangle,fluorescenceFactor]"
        power = params[0]
        initstateangle = params[1]
        pop=[np.cos(initstateangle)**2,np.sin(initstateangle)**2,0,0,0]
        """
        note leastsq performs a flatten function on params
        representing the population as an initial state angle in the ground energy level i.e 
        [x,y,0,0,0] and parameterising it as an angle seems to be the best solution
        """
        
        fluorescenceFactor = params[2]
        timeStep = t[1]-t[0]
        time = t[-1]-t[0]


        self._defineMatrix(power)
        tranMat = self.mat
        t,pop = continuousStochasticSolver(tranMat,pop,time,timeStep)
        
        """
        Fluorescence originates from the 3rd and 4th energy levels 
        Where the fluorescence rates are proportional to the population in the energy levels
        The fluorescence is also dependent on the branching ratio associated with other non-fluorescing
        decay paths found in the transition matrix tranMat, these are associated with the
        spin initialisation process found in the NV centre.

        """
        fluorescence = fluorescenceFactor*(pop[2]+pop[3]*(self.gamma*(1+self.epsilon))/(self.gamma*(1+self.epsilon)+self.k53))

        if plot:
            fig = plt.figure()
            ax=fig.add_subplot(111)
            ax.plot(t,fluorescence)
            plt.show()
        
        return fluorescence
    
    def pulseMeasurementFunction(self,params,t,plot=False):
        "params = [power,initstateangle0,initstateangle1,fluorescenceFactor,backgroundfluorescence]"
        power = params[0]
        initstateangle0=params[1]
        initstateangle1=params[2]
        fluorescenceFactor=params[3]
        backgroundfluorescence=params[4]
        
        """
        The pulsed measurement procedure is comprised of two optical pulses, where the first pulse
        represents the initialised state and the second pulse represent the resultant state,
        both pulses are energy level/spin state dependent. In the initialised state, the 
        initstateangle0 ~ 0 whereas in the resultant state the initstateangle ~ np.pi/2
        The full pulsed measurement function can then be produced by using two arrays of 
        fluorescenceFunction, where the surrounding times about the pulses are set to 0, and then
        summing the arrays and adding the background fluorescence to account for the dark counts
        visible in the data.
        """

        pulsetimes=[i*(self.t0[1]-self.t0[0]) for i in range(len(self.t0))]
        firstPulse = self.fluorescenceFunction([power,initstateangle0,fluorescenceFactor],pulsetimes)
        firstPulse = np.pad(firstPulse, (len((np.where((t<self.t0[0]))[0])),len(np.where(t>self.t0[-1])[0])),'constant', constant_values=(0,0))
        secondPulse = self.fluorescenceFunction([power,initstateangle1,fluorescenceFactor],pulsetimes)
        secondPulse = np.pad(secondPulse, (len(np.where(t<self.t1[0])[0]),len(np.where(t>self.t1[-1])[0])), 'constant', constant_values=(0,0))
        pulseMeasurement = secondPulse+firstPulse+backgroundfluorescence
        
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(t,pulseMeasurement)
            plt.show()
        return pulseMeasurement
        
    
    def _pulseMeasurementErrorFunction(self, params,t,y):
        return np.abs(self.y-self.pulseMeasurementFunction(params,t))
    
    
        
    def pulseMeasurementFitter(self,params,plot=False):

        leastsq = optimize.least_squares(self._pulseMeasurementErrorFunction,params,bounds=([0,0,0,0,0],[np.inf,np.pi,np.pi,np.inf,np.inf]), args=(self.t,self.y))
        p = leastsq.x

        if plot:
            fig = plt.figure()
            ax=fig.add_subplot(111)
            ax.plot(self.t,self.pulseMeasurementFunction(p,self.t))
            ax.plot(self.t,self.y)
            plt.show()
        return p


        
        
if __name__ == "__main__":
    

    fitter = NV5LevelFitter("/Users/shaun/Dropbox/Python/Markov Chain Monte Carlo/SpinReadout.csv")
    "params = [power,initstatangle0,initstateangle1,fluorescenceFactor,backgroundfluorescence]"
    fitter.pulseMeasurementFunction([.5,0,np.pi/2,12000,254],fitter.t,plot=True)
    
    
    p = fitter.pulseMeasurementFitter([.05,0,np.pi/2,120000,240], plot=True)
    print "power = "+str(p[0]) 
    print "initstateangle0 = "+str(p[1])
    print "initstateangle1 = "+str(p[2])
    print "fluorescenceFactor = "+str(p[3])   
    print "backgroundfluorescence = "+str(p[4]) 
    
    """
    There is a discrepancy between the simulated and collected pulses, there is
    an associated rise time with each of the experimental data pulses
    which is not represented in the simulated data. The background is also poorly fitted
    """