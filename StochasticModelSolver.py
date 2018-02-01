#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 16:29:43 2017

@author: Shaun
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import os
import sys
import csv


def gillespieStochasticSolver(squareArray,population, time, timeStep, ensemble = 1000, plot = False):
    """
    A Discrete Stochastic Solver, the Gillespie Algorithm see
    
    'A General Method for Numerically Simulating the Stochastic Time Evolution of Coupled Chemical Reactions'
    and
    'Calculation of quantum-dot blinking using the Gillespie Monte Carlo algorithm'
    
    This implementation reformalises stochastic system into an event-based algorithm
    For many systems the naive discrete stochastic solver works well, but is limited
    by the fastest transition rate from one state to another, 
    where the time step must be greater than 1/max(rate).
    
    If there is a large difference between transition rates, then the Gillespie Algorithm
    performs better since the time step is not limited by the transition rate
    but by how long the state remains in a particular state.
    """
    shape = np.shape(squareArray)
    if shape[0]!=shape[1] or len(shape)!=2:
        raise TypeError("Matrix input needs to be a square matrix N x N")
    elif shape[0]!=len(population):
        raise TypeError("Matrix input needs to be same length as population input")
        



    length=shape[0]
    
    timeStepNo = int(time/timeStep)+1
    p = np.zeros((timeStepNo,length))
    
    
    for run in range(ensemble):
        tau=0
        timeStepIndex=0
        t_i=timeStepIndex*timeStep
        newprobabilities = population
        
        
        
        while tau<time:
            projPopIndex = np.random.choice(range(length),p=newprobabilities)
            projPop = np.zeros(length)
            projPop[projPopIndex]=1
            a_i = np.dot(squareArray,projPop)
            a_i[projPopIndex]=0
            a_0 = sum(a_i)
            newprobabilities = a_i/a_0
            dtau = 1.0/a_0*np.log(1.0/np.random.random())
            tau+=dtau
            
            
            while t_i<tau:
                p[timeStepIndex]+=projPop
                timeStepIndex+=1
                t_i=timeStepIndex*timeStep
                if t_i>time:
                    break
                
    p=p.transpose()/ensemble
    t=np.linspace(0,time,timeStepNo)
    
    if plot == True:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(length):
            ax.plot(t,p[0],label="%d" % i)

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()
        
        
    return t,p
    


        
    
def discreteStochasticSolver(squareArray, population, time, timeStep, ensemble = 4000,plot=False):
    
    """
    Monte Carlo method for solving stochastic model
    """
    
    
    shape = np.shape(squareArray)
    if shape[0]!=shape[1] or len(shape)!=2:
        raise TypeError("Matrix input needs to be a square matrix N x N")
    elif shape[0]!=len(population):
        raise TypeError("Matrix input needs to be same length as population input")
    """
    Stochastic Matrix to Linear Expansion Matrix A-> e^(Adt) ~ I+Adt
    
    Solution to dP/dt=A P -> P=e^(Adt)P0
    P is the probability distribution after evolution e^(Adt)
    First Order Expansion of e^(Adt)=I+Adt+...
    
    [-(c1+c2),c1,c2]
    [c0,-(c0+c2),c2]
    [c0,c1,-(c0+c2)]
    
    
    [1-(c1t+c2t),c1t,c2t]
    [c0t,1-(c0t+c2t),c2t]
    [c0t,c1t,1-(c0t+c2t)]
    
    """
    squareArray = squareArray*timeStep+np.identity(shape[0])
    flatArray = squareArray.flatten()
    if any(element<0 for element in flatArray):
        raise TypeError("Time step input needs to be shorter than the fastest transition rate, otherwise there are negative probabilities.")    
    
    """
    Want data to start at 0 and end at "time" 
    """
    timeStepNo = int(time/timeStep)+1
    time +=timeStep
    length = shape[0]
    

    p=np.zeros((timeStepNo,length))
    t = np.arange(0,time,timeStep)
    """
    "p" = histogram for all states in "length" and timebins in "timeStepNo"
    Histogram is performed over an "ensemble" of experiments
    """
    
    for run in range(ensemble):
        newprobabilities = population
        
        """
        Project population using a random weight choice and then calculate the next population weights
        """
    
        for i in range(timeStepNo):
            newprobabilities/=newprobabilities.sum()
            projPopIndex = np.random.choice(range(length),p=newprobabilities)
            projPop = np.zeros(length)
            projPop[projPopIndex]=1
            p[i]+=projPop
            newprobabilities = np.dot(squareArray,projPop)
    
    p = p.transpose()/ensemble
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(length):
            ax.plot(t,p[0],label="%d" % i)

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()
    
    
    return t, p

            
    
        
    

    
def continuousStochasticSolver(squareArray, population, time, timeStep, plot = False):
    
    """
    Solve:
        
     p'(t) = A p(t)
    ---------------
    Diagonalise:
     
     A = SDS'
    
    Substitue p(t) with q(t):
        
     q(t)  = S'p(t)
     q'(t) = S'p'(t)
     
    Therefore:
    
     q'(t) = D q(t) where D is diagonal
     
    First Order Linear Differential Equation:
        
     q(t) = e^(Dt) q(0)
     p(t) = S e^(Dt) q(0)  
     p(t) = Se^(Dt)S^ p(0)
     where we can consider q(0) = S^ p(0)
        
    """
    
    shape = np.shape(squareArray)
    length = shape[0]
    if shape[0]!=shape[1] or len(shape)!=2:
        raise TypeError("Matrix input needs to be a square matrix N x N")
    elif shape[0]!=len(population):
        raise TypeError("Matrix input needs to be same length as population input")
        
    """
    Note to self, scipy contains an inverse function numpy does not.
    (Remember, scipy is an extension of numpy)
    """
    
    eigVal,eigVec =sp.linalg.eig(squareArray)
    eigVecT= sp.linalg.inv(eigVec)
    

    timeStepNo = int(round(time/timeStep))+1

    timeStepDiag = np.diag(np.exp(eigVal * timeStep))
    timeStepTransform = eigVec.dot(timeStepDiag).dot(eigVecT)

    
    """
    p(2t) = Se^(2Dt)S^ p(0)
    or 
    p(2t) = Se^(Dt)S^ p(t) = Se^(Dt)S^ Se^(Dt)S^ p(0) = Se^(2Dt)S^ p(0)
    
    
    """
    
    p=np.array([population])
    t=np.array([0])
    
    for timeIndex in range(1,timeStepNo):
                

        pi = p[-1]
        pj = [timeStepTransform.dot(pi)]
        p = np.append(p,pj, axis = 0)
        t=np.append(t,np.array([timeIndex*timeStep]))
        
    
    p = p.transpose()
    
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(length):
            ax.plot(t,p[0],label="%d" % i)

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()
    
    return t,p
    
    

    
        
        
    



def initPopGenerator(n, init= "Random"):
    """ 
    Generates an 'n' length array representing the probability distribution 
    for states within the system
    [1,0,0,0,0,0] represents a 6 state system with the initial state in the
    zeroth state.
    This only needs to sum to one so can support both coherent/incoherent 
    dynamics. 
    """
    if init=="Random":
        indexList = range(n)
        initIndex = np.random.choice(indexList)
        popList = np.zeros(n)
        popList[initIndex]=1
    elif type(init)== int:
        if init>n:
            raise IndexError("Initial state index greater than the number of states")
        else:
            popList = np.zeros(n)
            popList[init]=1
    else:
        raise TypeError("Initial state should be an integer less than 'n' ")
            
            
        
    return popList

def stochasticMatrixGenerator(n,load = False):
    
    """
    Returns stochastic matrix, stochastic matrix is either loaded or generated
    and saved to StochasticMatrix.csv which is found in the same directory as StochasticModelSolver.py
    """
    
    path = os.path.dirname(sys.argv[0])
    filename = r"StochasticMatrix.csv"
    newpath = os.path.join(path,filename)
    if load:
        "Load stochastic matrix"
        mat=np.zeros((n,n))
        with open(newpath,"r") as csvfile:
            reader = csv.reader(csvfile,delimiter = "," )
            i=0
            for row in reader:
                mat[i]= row
                i+=1

    else:
        "Create new random stochastic matrix"
        mat = np.random.random((n,n))
        mat[(range(n),range(n))]=0    
        for i in range(n):
            mat[(i,i)]=-np.sum(mat[i])

        with open(newpath,"w") as csvfile:
            writer = csv.writer(csvfile,delimiter = ",")
            for i in range(n):
                writer.writerow(mat[i])
    
    return mat
            
            
                

    
    
if __name__ == "__main__":
    """
    Population transition rates taken from:
    Nitrogen-vacancy center in diamond: Model of the electronic structure and associated dynamics
    https://journals.aps.org/prb/abstract/10.1103/PhysRevB.74.104303
    Describes the population change in NV energy level under optical excitation
    Power is arbitrary
    """
    
    Power = .1
    R = 77*Power
    gam = 77.0
    ep = 2 * 1.5/77.0
    GamD = 3.3
    k53 = 2 * 30.0
    k35 = 0
    k23 = 0
    k32 = 0
    
    tranMat = np.array([[-R*(1+ep), gam, GamD, 0, gam*ep], \
    [R, -gam*(1+ep) - k23, k32, ep*R, 0], \
    [0, k23, -(GamD+k32+k35), 0, k53], \
    [0, gam*ep, 0, -R*(1+ep), gam], \
    [R*ep, 0, k35, R, -gam*(1 + ep) - k53 ]])
    b = [[1,1.0j],[1.0j,1]]
    

    
    initPop = initPopGenerator(5,init=0)
    continuousStochasticSolver(tranMat,initPop,10,.1,plot=True)
    
    """
    initPop = initPopGenerator(5,init = 0)
    #tranMat = stochasticMatrixGenerator(30,load=False)
    t = tm.time()
    #discreteStochasticSolver(tranMat,initPop,.2,.01,ensemble= 30000,plot=True)
    continuousStochasticSolver(tranMat,initPop,10,.01,plot = True)
    print "continuous"
    print tm.time()-t
    t= tm.time()

    #gillespieStochasticSolver(tranMat,initPop,.2,.01,ensemble = 100000, plot=True)
    print "gillespie"
    print tm.time()-t
    """
    



