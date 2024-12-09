#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:46:32 2023

@author: javier
"""

# ---------------------------- IMPORT PACKAGES --------------------------------
import numpy as np
import ncpol2sdpa as ncp
from numba import jit
import sympy as sp
from math import sqrt, log2, log, pi, cos, sin
from sympy.physics.quantum.dagger import Dagger
import mosek
import chaospy
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import minimize
import qutip as qtp
from scipy.special import factorial
from math import floor
import cvxopt
from scipy.io import savemat
import copy
params = {'text.usetex': True}
plt.rcParams.update(params)
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
mpl.rcParams['text.latex.preamble']=[r"\usepackage{xcolor}"]

# ------------------------- MEASUREMENT OPERATORS -----------------------------
# --> Measurement operators defined in Alexia's paper
def meas_operators(d,m,x,y):
    # Define the small omega
    omega = np.exp(1j*2*np.pi/d)
    # Define the Omega matrix
    Omega = np.diag([omega**i for i in range(d)])
    # Define F matrix
    F = np.zeros((d,d),dtype = "complex")
    for i in range(d):
        for j in range(d):
            F[i,j] = (1/np.sqrt(d))*omega**(i*j)
    # Define Ux matrix
    thetax = (x-0.5)/m
    Ux = np.diag([omega**(thetax*j) for j in range(d)])
    # Define Vy matrix
    zetay = y/m
    Vy = np.diag([omega**(zetay*j) for j in range(d)])
    # Alice matrix
    Alice_op = np.conjugate(np.transpose(Ux)).dot(F).dot(Omega).dot(np.conjugate(np.transpose(F))).dot(Ux)
    Bob_op = Vy.dot(np.conjugate(np.transpose(F))).dot(Omega).dot(F).dot(np.conjugate(np.transpose(Vy)))
    return np.kron(Alice_op,Bob_op)
 
# --> Define probability distributions
def new_probs_for_settings(d,m,x,y,gammas):
    # Define the quantum state
    state = 0
    for i in range(d):
        state+=gammas[i]*qtp.ket(str(i)+str(i))
    norm = state.norm()
    state = np.array(state/norm)
    # Define the measurement operator
    AtimesB = meas_operators(d,m,x,y)
    # Compute the probabilities
    probs = np.abs(AtimesB.dot(state))**2
    return probs.reshape(d,d)

# --> Define probabilities ideal
def new_probs_ideal(d,m,gammas):
    # Define lists for x and y
    x_list = list(range(1,m+1))
    y_list = list(range(1,m+1))
    # Define dictionary for saving data
    all_probs = {}
    for x in x_list:
        for y in y_list:
            all_probs[str(x)+str(y)] = new_probs_for_settings(d,m,x,y,gammas)
    return all_probs

# --> New probabilities with noise
def new_probs_noise(d,m,eta,gammas):
    # Define lists for x and y
    x_list = list(range(1,m+1))
    y_list = list(range(1,m+1))
    # White noise matrix
    white_noise = (1/(d**2))*np.ones((d,d))
    # Define dictionary for saving data
    all_probs = {}
    for x in x_list:
        for y in y_list:
            all_probs[str(x)+str(y)] = (eta*new_probs_for_settings(d,m,x,y,gammas)
                                        +(1-eta)*white_noise)
    return all_probs
    
# ----------------------- PROBABILITY DISTRIBUTIONS ---------------------------
# --> Define elements for the probability
@jit
def elements_probability(gammas,d,m,x,y,a,b):
    el = 0
    thetax = (x-0.5)/m
    zetay = y/m
    for q in range(0,d):
        el += gammas[q]*np.exp((1j*2*np.pi/d)*q*(a-b-thetax+zetay))
    return np.abs((1/d)*el)**2

# --> Define probability wehn Alice and Bob perform the same measurement
def elements_probability_equal(gammas,d,m,x,eta):
    # Some parameters
    thetax = x/m
    # List of measurements
    a_list = np.arange(0,d,1)
    b_list = np.arange(0,d,1)
    # Probability matrix
    probs = np.zeros((d,d))
    # Normalize gammas
    gammas = np.array(gammas)/np.sqrt(np.sum(np.abs(gammas)**2))
    for a in a_list:
        for b in b_list:
            el = 0
            for q in range(0,d):
                el += gammas[q]*np.exp(-(1j*2*np.pi/d)*q*(a+b-2*thetax))
            probs[a,b] = eta*np.abs((1/d)*el)**2 + (1-eta)*(1/(d**2))
    return probs

# --> Define ideal probability distributions 
def ideal_prob_for_settings(d,m,x,y,gammas = []):
    # Define coefficients for the states
    if d == 3 and len(gammas) == 0:
        tgamma = 0.5*(np.sqrt(11)-np.sqrt(3))
        gammas = [1/np.sqrt(2+tgamma**2),tgamma/np.sqrt(2+tgamma**2),
                  1/np.sqrt(2+tgamma**2)]
        #gammas = 1/np.sqrt(3)*np.ones(3)
    elif d==2 and len(gammas) == 0:
        gammas = [1/np.sqrt(2),1/np.sqrt(2)]
    tgamma = 1/np.sum(np.abs(gammas)**2)
    gammas = np.sqrt(tgamma)*np.array(gammas)
    # Define list for outputs
    a_list = np.arange(0,d,1)
    b_list = np.arange(0,d,1)
    # Define array where saving the elements of probability
    prob = np.zeros((len(a_list),len(b_list)))
    # Compute the elements of the probability
    for a in a_list:
        for b in b_list:
            prob[a,b] = elements_probability(gammas,d,m,x,y,a,b)
    return prob


# --> All ideal probabilities
def all_probs_ideal(d,m,gammas = []):
    # Define lists for x and y
    x_list = list(range(1,m+1))
    y_list = list(range(1,m+1))
    # Define dictionary for saving data
    all_probs = {}
    for x in x_list:
        for y in y_list:
            all_probs[str(x)+str(y)] = ideal_prob_for_settings(d,m,x,y,gammas)
    # Add the case
    return all_probs

# --> All ideal probabilities with white noise
def all_probs_noise(d,m,eta,gammas=[]):
    # Define lists for x and y
    x_list = list(range(1,m+1))
    y_list = list(range(1,m+1))
    # White noise matrix
    white_noise = (1/(d**2))*np.ones((d,d))
    # Define dictionary for saving data
    all_probs = {}
    for x in x_list:
        for y in y_list:
            all_probs[str(x)+str(y)] = (eta*ideal_prob_for_settings(d,m,x,y,gammas)
                                        +(1-eta)*white_noise)
    return all_probs

# --> Obtain marginal probabilities
def marginal_probabilities(m,all_probs):
    # Define empty lists for Alice and for Bob
    P_Alice = {}
    P_Bob = {}
    for i in range(1,m+1):
        P_Bob[str(i)]= np.sum(all_probs[str(i)+str(i)],axis=0).tolist()
        P_Alice[str(i)] = np.sum(all_probs[str(i)+str(i)],axis=1).tolist()
    return P_Alice,P_Bob

# --> Probabilities for Toni's noise model
def Toni_noise_probabilities(d,m,x,y,eta,gammas = []):
    # Define coefficients for the states
    if d == 3 and len(gammas) == 0:
        tgamma = 0.5*(np.sqrt(11)-np.sqrt(3))
        gammas = [1/np.sqrt(2+tgamma**2),tgamma/np.sqrt(2+tgamma**2),
                  1/np.sqrt(2+tgamma**2)]
        #gammas = 1/np.sqrt(3)*np.ones(3)
    elif d==2 and len(gammas) == 0:
        gammas = [1/np.sqrt(2),1/np.sqrt(2)]
    tgamma = 1/np.sum(np.abs(gammas)**2)
    gammas = np.sqrt(tgamma)*np.array(gammas)
    # Define list for outputs
    a_list = np.arange(0,d,1)
    b_list = np.arange(0,d,1)
    # Define array where saving the elements of probability
    prob = np.zeros((len(a_list),len(b_list)))
    # Compute the elements of the probability
    for a in a_list:
        for b in b_list:
            prob[a,b] = elements_probability(gammas,d,m,x,y,a,b)
    # Add extra rows and columns to the ideal probability distribution
    prob = np.hstack((prob,np.zeros((d,1))))
    prob = np.vstack((prob,np.zeros((1,d+1))))
    
    # Characterize the new elements added to the probability distribution
    for a in a_list:
        prob[d,a] = eta*(1-eta)*np.sum(prob[:,a])
        prob[a,d] = eta*(1-eta)*np.sum(prob[a,:])
        
    # Add the [d,d] element
    prob[d,d] = (1-eta)**2
    
    # Modify the elements in the middle
    prob[:d,:d] = eta**2*(prob[:d,:d])
    return prob

# --> Gives all the probabilities according to Toni's noise model
def all_probs_Toni_noise(d,m,eta,gammas = []):
    # Define lists for x and y
    x_list = list(range(1,m+1))
    y_list = list(range(1,m+1))
    # Define dictionary for saving data
    all_probs = {}
    for x in x_list:
        for y in y_list:
            all_probs[str(x)+str(y)] = Toni_noise_probabilities(d,m,x,y,eta,gammas)
    # Add the case
    return all_probs

# --> Define probability wehn Alice and Bob perform the same measurement (according to Toni's model)
def elements_probability_equal_Toni_noise(gammas,d,m,x,eta):
    # Some parameters
    thetax = x/m
    # List of measurements
    a_list = np.arange(0,d,1)
    b_list = np.arange(0,d,1)
    # Probability matrix
    probs = np.zeros((d,d))
    # Normalize gammas
    gammas = np.array(gammas)/np.sqrt(np.sum(np.abs(gammas)**2))
    for a in a_list:
        for b in b_list:
            el = 0
            for q in range(0,d):
                el += gammas[q]*np.exp((1j*2*np.pi/d)*q*(a+b-2*thetax))
            probs[a,b] = np.abs((1/d)*el)**2
    # Add extra rows and columns to the ideal probability distribution
    probs = np.hstack((probs,np.zeros((d,1))))
    probs = np.vstack((probs,np.zeros((1,d+1))))
    
    # Characterize the new elements added to the probability distribution
    for a in a_list:
        probs[d,a] = eta*(1-eta)*np.sum(probs[:,a])
        probs[a,d] = eta*(1-eta)*np.sum(probs[a,:])
        
    # Add the [d,d] element
    probs[d,d] = (1-eta)**2
    
    # Modify the elements in the middle
    probs[:d,:d] = eta**2*(probs[:d,:d])
    return probs

# ---------------------------- ENKY'S NOISE MODEL -----------------------------
# --> Define noisy state (pass through a BS and eliminate vacuum from environment)
def noise_state(d,theta,gammas):
    # Define noise parameter
    t = np.cos(theta)
    r = np.sin(theta)
    # Define creation and annihilation operators
    adag_t = qtp.tensor(qtp.create(d),qtp.qeye(d),qtp.qeye(d),qtp.qeye(d))
    adag_r = qtp.tensor(qtp.qeye(d),qtp.create(d),qtp.qeye(d),qtp.qeye(d))
    bdag_t = qtp.tensor(qtp.qeye(d),qtp.qeye(d),qtp.create(d),qtp.qeye(d))
    bdag_r = qtp.tensor(qtp.qeye(d),qtp.qeye(d),qtp.qeye(d),qtp.create(d))
    
    # Define the quantum state after the BS
    state = 0
    for q in range(d):
        state += gammas[q]*((((t*adag_t+r*adag_r)**q/np.sqrt(factorial(q))))*((t*bdag_t+r*bdag_r)**q/np.sqrt(factorial(q))))*qtp.ket("0000",dim=d)
    # Normalize the state
    state = state.unit()
    # Trace out the noise
    rho = state.ptrace((0,2))
    return rho

# --> Use to benchmark the results obtained before
def check_noise_state(d,theta,gammas):
    # Define noise parameter
    t = np.cos(theta)
    r = np.sin(theta)
    # Define creation and annihilation operators
    adag_t = qtp.tensor(qtp.create(d),qtp.qeye(d))
    adag_r = qtp.tensor(qtp.qeye(d),qtp.create(d))
        
    # Define the quantum state after the BS
    state = 0
    for q in range(d):
        state += gammas[q]*((((t*adag_t+r*adag_r)**q/np.sqrt(factorial(q)))))*qtp.ket("00",dim=d)
    return state.unit()

# --> Define the exact state
def exact_state(d,gammas):
    # Define the exact state
    state = 0
    for q in range(d):
        state += gammas[q]*qtp.ket(str(q)+str(q),dim=d)
    # Normalize the state
    state = state.unit()
    return state

# --> Alice and Bob measurement operators
# Here, I'm using the notation of PRL 97, 170409 (2006)
def AB_proj_operators(d,m,a,b,x,y):
    # Define some parameters
    alpha_x = (x-0.5)/m
    beta_y = y/m
    # Define eigenstates of Alice and Bob measurements
    eigstate_A = 0
    eigstate_B = 0
    for q in range(0,d):
        eigstate_A += np.sqrt(1/d)*np.exp((2*np.pi*1j/d)*q*(a-alpha_x))*qtp.ket(str(q),dim=d)
        eigstate_B += np.sqrt(1/d)*np.exp(-(2*np.pi*1j/d)*q*(b-beta_y))*qtp.ket(str(q),dim=d)
    # Define the operators
    A_op = eigstate_A.proj()
    B_op = eigstate_B.proj()
    return qtp.tensor(A_op,B_op)

# --> Probability distribution for fixed settings
def Enky_noise_prob_distrib_settings(d,m,x,y,theta,gammas):
    # Define the quantum state
    rho = noise_state(d,theta,gammas)
    # Define the list of outcomes
    a_list = list(range(0,d))
    b_list = list(range(0,d))
    # For each outcome compute the probability
    prob = np.zeros((d,d),dtype="complex")
    for a in a_list:
        for b in b_list:
            # Define projective operator
            AB = AB_proj_operators(d,m,a,b,x,y)
            # Compute probability
            prob[a,b] = (rho*AB).tr()
    return prob

# --> Compute probability distribution for all settings
def Enky_noise_all_probs(d,m,theta,gammas):
    # Define the set of measurement settings
    x_list = list(range(1,m+1))
    y_list = list(range(1,m+1))
    # Define dictionary for saving probs
    all_probs = {}
    # Compute the probabilities for the different settings
    for x in x_list:
        for y in y_list:
            all_probs[str(x)+str(y)] = np.real(Enky_noise_prob_distrib_settings(d,m,x,y,theta,gammas))
    return all_probs

# --> Alice and Bob measurement operators for equal settings
def AB_proj_operators_equal_settings(d,m,a,b,y):
    # Define some parameters
    beta_y = y/m
    # Define eigenstates of Alice and Bob measurements
    eigstate_A = 0
    eigstate_B = 0
    for q in range(0,d):
        eigstate_A += np.sqrt(1/d)*np.exp(-(2*np.pi*1j/d)*q*(a-beta_y))*qtp.ket(str(q),dim=d)
        eigstate_B += np.sqrt(1/d)*np.exp(-(2*np.pi*1j/d)*q*(b-beta_y))*qtp.ket(str(q),dim=d)
    # Define the operators
    A_op = eigstate_A.proj()
    B_op = eigstate_B.proj()
    return qtp.tensor(A_op,B_op)

# --> Compute probability distribution for equal settings
def Enky_noise_prob_equal_settings(d,m,y,theta,gammas):
    # Define the quantum state
    rho = noise_state(d,theta,gammas)
    # Define the list of outcomes
    a_list = list(range(0,d))
    b_list = list(range(0,d))
    # For each outcome compute the probability
    prob = np.zeros((d,d),dtype="complex")
    for a in a_list:
        for b in b_list:
            # Define projective operator
            AB = AB_proj_operators_equal_settings(d,m,a,b,y)
            # Compute probability
            prob[a,b] = (rho*AB).tr()
    return np.real(prob)


# -------------------- PROBABILITY AMPLITUDES FOR CHSH OPT --------------------
# --> Define elements for the probability
@jit
def elements_probability_opt(gammas,d,m,alpha,beta,a,b):
    el = 0
    for q in range(0,d):
        el += gammas[q]*np.exp((1j*2*np.pi/d)*q*(a-b-alpha+beta))
    return np.abs((1/d)*el)**2

# --> Define ideal probability distributions 
def opt_prob_for_settings(d,m,alpha,beta,y,gammas = []):
    # Define coefficients for the states
    if d == 3 and len(gammas) == 0:
        tgamma = 0.5*(np.sqrt(11)-np.sqrt(3))
        gammas = [1/np.sqrt(2+tgamma**2),tgamma/np.sqrt(2+tgamma**2),
                  1/np.sqrt(2+tgamma**2)]
        #gammas = 1/np.sqrt(3)*np.ones(3)
    elif d==2 and len(gammas) == 0:
        gammas = [1/np.sqrt(2),1/np.sqrt(2)]
    tgamma = 1/np.sum(np.abs(gammas)**2)
    gammas = np.sqrt(tgamma)*np.array(gammas)
    # Define list for outputs
    a_list = np.arange(0,d,1)
    b_list = np.arange(0,d,1)
    # Define array where saving the elements of probability
    prob = np.zeros((len(a_list),len(b_list)))
    # Compute the elements of the probability
    for a in a_list:
        for b in b_list:
            prob[a,b] = elements_probability(gammas,d,m,alpha,beta,a,b)
    return prob

# --> All ideal probabilities
def opt_probs_ideal(d,m,alpha_list,beta_list,gammas):
    # Define dictionary for saving data
    all_probs = {}
    x=0    
    for alpha in alpha_list:
        y=0
        for beta in beta_list:
            all_probs[str(x)+str(y)] = opt_prob_for_settings(d,m,alpha,beta,gammas)
            y+=1
        x+=1
    return all_probs

# --> CHSH computation
def CHSH_val(d,m,params,gammas):
    # Compute probabilities
    alpha_list = params[:2]
    beta_list = params[2:]
    all_probs = opt_probs_ideal(d,m,alpha_list,beta_list,gammas)
    # Compute individal correlators
    AB = []
    for i in range(m):
        for j in range(m):
            corr = np.sum(all_probs[str(i)+str(j)].flatten()*np.array([1,-1,-1,1]))
            AB.append(corr)
    # List of coefficients for CHSH
    coeffs = [[1,1,1,-1]
              ,[1,1,-1,1]
              ,[1,-1,1,1]
              ,[-1,1,1,1]]
    CHSH = []
    for coeff in coeffs:
        CHSH.append(np.sum(np.array(coeff)*np.array(AB)))
    return np.max(np.abs(CHSH))

# --> CHSH optimizer
def CHSH_opt(gammas = [1,1],x0 = [0.5,0.5,0.5,0.5]):
    # Define function to optimize
    func = lambda x: (-1)*CHSH_val(2,2,x,[1,1])
    # Optimize function
    result = minimize(func,x0,method="Nelder-Mead", options = {"maxiter":100})
    return result.x, (-1)*func(result.x)

# ------------------- MEASUREMENT SETTINGS AS VARIABLES -----------------------  
# --> Alice and Bob projectors
def AB_proj_settings(d,m,a,b,x,y,alpha,beta):
    # Define Alice and Bob's eigenstates of the projector
    eigenstate_A = 0
    eigenstate_B = 0
    # Compute the eigenstates
    for q in range(0,d):
        # Define the Fock component
        state = qtp.ket(str(q),dim=d)
        eigenstate_A += np.sqrt(1/d)*np.exp((2*np.pi*1j/d)*(q-1)*(a+alpha))*state
        eigenstate_B += np.sqrt(1/d)*np.exp(-(2*np.pi*1j/d)*(q-1)*(b+beta))*state
    # Compute the projectors
    A_op = eigenstate_A*eigenstate_A.dag()
    B_op = eigenstate_B*eigenstate_B.dag()
    # Compute the total projector
    operator = qtp.tensor(A_op,B_op)
    return operator

# --> Ideal probabilities for fixed settings
def ideal_prob_free_settings(d,m,x,y,gammas,alpha,beta):
    # Define the quantum state
    state = 0
    for q in range(0,d):
        basis_el = qtp.ket(str(q)+str(q),dim=d)
        state += gammas[q]*basis_el
    # Normalize the state
    state = state.unit()
    # Compute the probability
    prob = np.zeros((d,d),dtype="complex")
    for a in range(0,d):
        for b in range(0,d):
            # Get projector
            op = AB_proj_settings(d,m,a,b,x,y,alpha,beta)
            prob[a,b] = qtp.expect(op,state)
    return np.real(prob)

# --> WHITE NOISE
# --> All probabilities for white noise
def all_probs_noise_settings(d,m,eta,gammas,alphas,betas):
    # Define dictionary for saving data
    all_probs = {}
    # Define white noise
    white_noise = (1/(d**2))*np.ones((d,d))
    # For each measurement setting, we compute the probabilities
    for x in range(1,m+1):
        for y in range(1,m+1):
            all_probs[str(x)+str(y)] = (eta*ideal_prob_free_settings(d,m,x,y,gammas,alphas[x-1],betas[y-1])
                                        + (1-eta)*white_noise)
    return all_probs    

# --> Probability of extra measurement for Bob
def extra_meas_Bob_settings(d,m,eta,gammas,alpha,beta):
    # Compute the ideal probability
    ideal_prob = ideal_prob_free_settings(d,m,1,1,gammas,alpha,beta)
    # Note that here we set the meas. settings to (1,1), but this is irrelevant
    # because the ideal_prob function does not change with x,y
    
    # Define white noise
    white_noise = (1/(d**2))*np.ones((d,d))
    # Obtain noisy probability
    noisy_prob = eta*ideal_prob + (1-eta)*white_noise
    return noisy_prob

# --> ENKY'S MODEL (PHOTON ABSORPTION)
# --> Probability distribution for fixed settings using Enky's noise model
def Enky_noise_prob_distrib_optsettings(d,m,theta,gammas,alpha,beta):
    # Define the quantum state
    rho = noise_state(d,theta,gammas)
    # Compute the probability
    prob = np.zeros((d,d),dtype="complex")
    for a in range(0,d):
        for b in range(0,d):
            # Get projector (x and y are not defined within the fucntion so set to one)
            op = AB_proj_settings(d,m,a,b,1,1,alpha,beta)
            # Compute probability
            prob[a,b] = qtp.expect(op,rho)
    return np.real(prob)

# --> All probability distributions for Enky's noise
def all_probs_Enky_noise_settings(d,m,theta,gammas,alphas,betas):
    # Define dictionary for saving data
    all_probs = {}
    # Compute the probabilities for the different settings
    for x in range(1,m+1):
        for y in range(1,m+1):
            all_probs[str(x)+str(y)] = Enky_noise_prob_distrib_optsettings(d,m,theta,gammas,alphas[x-1],betas[y-1])
    return all_probs

# ------------------------- GENERALIZED PROJECTORS ----------------------------        
# -------> CASE OF d = 2
# --> Alice and Bob projectors for d=2
def generalized_projectors_d2(d,m,theta):
    # Define the eigenstates
    eigenstate_1 = np.cos(theta)*qtp.ket("0",dim=d) + np.sin(theta)*qtp.ket("1",dim=d)
    eigenstate_2 = np.sin(theta)*qtp.ket("0",dim=d) - np.cos(theta)*qtp.ket("1",dim=d)
    # Compute the projectors
    Op_1 = eigenstate_1*eigenstate_1.dag()
    Op_2 = eigenstate_2*eigenstate_2.dag()
    return [Op_1,Op_2]

# --> Compute ideal probabilities
def ideal_probs_generalized_d2(d,m,A_setting,B_setting,gammas):
    # Define our quantum state
    state = 0
    for q in range(0,d):
        basis_el = qtp.ket(str(q)+str(q),dim=d)
        state += gammas[q]*basis_el
    # Normalize the state
    state = state.unit()
    # Define the projectors
    A_ops = generalized_projectors_d2(d,m,A_setting)
    B_ops = generalized_projectors_d2(d,m,B_setting)
    # Compute the probabilities
    probs = np.zeros((d,d))
    for a in range(len(A_ops)):
        for b in range(len(B_ops)):
            # Define tensor product operator
            AB_op = qtp.tensor(A_ops[a],B_ops[b])
            probs[a,b] = qtp.expect(AB_op,state)
    return probs

# --> All probabilities
def noisy_probs_generalized_d2(d,m,A_settings,B_settings,eta,gammas):
    # Define dictionary for saving data
    all_probs = {}
    # Define white noise
    white_noise = (1/(d**2))*np.ones((d,d))
    # Compute the corresponding probabilities
    for x in range(m):
        for y in range(m):
            all_probs[str(x+1)+str(y+1)] = (eta*ideal_probs_generalized_d2(d,m,A_settings[x],B_settings[y],gammas)
                                        +(1-eta)*white_noise)
    return all_probs
            
# --> Compute ideal probabilities
def extra_probs_generalized_d2(d,m,eta,A_setting,B_setting,gammas):
    # Define our quantum state
    state = 0
    for q in range(0,d):
        basis_el = qtp.ket(str(q)+str(q),dim=d)
        state += gammas[q]*basis_el
    # Normalize the state
    state = state.unit()
    # Define the projectors
    A_ops = generalized_projectors_d2(d,m,A_setting)
    B_ops = generalized_projectors_d2(d,m,B_setting)
    # Compute the probabilities
    probs = np.zeros((d,d))
    for a in range(len(A_ops)):
        for b in range(len(B_ops)):
            # Define tensor product operator
            AB_op = qtp.tensor(A_ops[a],B_ops[b])
            probs[a,b] = qtp.expect(AB_op,state)
    # Define white noise
    white_noise = (1/(d**2))*np.ones((d,d))
    # Compute probabilities
    noise_probs = eta*probs + (1-eta)*white_noise
    return noise_probs 

# --> Enky's probability for fixed settings with the generalized projectors
def Enky_noise_prob_distrib_generalized_d2(d,m,theta,gammas,A_setting,B_setting):
    # Define the quantum state
    rho = noise_state(d,theta,gammas)
    # Compute measurement projectors
    A_op = generalized_projectors_d2(d,m,A_setting)
    B_op = generalized_projectors_d2(d,m,B_setting)
    # Compute the probability
    prob = np.zeros((d,d),dtype="complex")
    for a in range(0,d):
        for b in range(0,d):
            # Get projector (x and y are not defined within the fucntion so set to one)
            op = qtp.tensor(A_op[a],B_op[b])
            # Compute probability
            prob[a,b] = qtp.expect(op,rho)
    return np.real(prob)

# --> All probability distributions for Enky's noise
def all_probs_Enky_noise_generalized_d2(d,m,theta,gammas,A_setting,B_setting):
    # Define dictionary for saving data
    all_probs = {}
    # Compute the probabilities for the different settings
    for x in range(1,m+1):
        for y in range(1,m+1):
            all_probs[str(x)+str(y)] = Enky_noise_prob_distrib_generalized_d2(d,m,theta,gammas,A_setting[x-1],B_setting[y-1])
    return all_probs

# --> Define objective function for optimizing probability settings
def optimize_settings_Alexia_objective(d,m,gammas,settings):
    # Define Alexia's settings
    Alexia_settings = []
    for x in range(1,m+1):
        Alexia_settings.append((x-0.5)/m)
    Alexia_settings.append(0.5/m)
    for y in range(1,m+1):
        Alexia_settings.append(y/m)
    
    # Compute Alexia's probabilities
    alexia_probs = all_probs_noise_settings(d,m,1,gammas,Alexia_settings[:m],Alexia_settings[-(m):])
    # Compute our probabilities
    our_probs = noisy_probs_generalized_d2(d,m,settings[:m],settings[-m:],1,gammas)
    # Compute the distance between probs
    obj = 0
    for x in range(1,m+1):
        for y in range(1,m+1):
            obj += np.sum(np.abs(alexia_probs[str(x)+str(y)] - our_probs[str(x)+str(y)]))
    return obj

# --> Optimize the trace distance between probabilities
def optimize_settings_Alexia(d,m,gammas,error = 1e-3):
    val = 1
    while val > error:
        # Define initial settings at random
        init_settings = np.random.rand(2*m)
        # Define objective function as lambda function
        func = lambda x: optimize_settings_Alexia_objective(d,m,gammas,x)
        # Define bounds for the optimization
        bnds =[]
        for i in range(0,2*m):
            bnds.append([0,2*np.pi])
        
        # Do the optimization
        res = minimize(func, init_settings, method = "Nelder-Mead", bounds = bnds, options = {"maxiter":1000,"fatol":0.001, "disp":True})
        # Print the value
        val = func(res.x)
    return res.x, func(res.x)
    
# -------> CASE OF d = 3
# --> Alice and Bob projectors for d=3
def generalized_projectors_d3(d,m,theta,phi):
    # Define the eigenstates
    eigenstate_1 = (np.cos(theta)*np.sin(phi)*qtp.ket("0",dim=d) 
                    + np.sin(theta)*np.sin(phi)*qtp.ket("1",dim=d)
                    + np.cos(phi)*qtp.ket("2",dim=d))
    eigenstate_2 = (np.cos(theta)*np.cos(phi)*qtp.ket("0",dim=d) 
                    + np.sin(theta)*np.cos(phi)*qtp.ket("1",dim=d)
                    - np.sin(phi)*qtp.ket("2",dim=d))
    # Compute the projectors
    Op_1 = eigenstate_1*eigenstate_1.dag()
    Op_2 = eigenstate_2*eigenstate_2.dag()
    Op_3 = qtp.qeye(d) - Op_1 - Op_2
    return [Op_1,Op_2,Op_3]  

# --> Compute ideal probabilities for d=3
def ideal_probs_generalized_d3(d,m,A_setting,B_setting,gammas):
    # Define our quantum state
    state = 0
    for q in range(0,d):
        basis_el = qtp.ket(str(q)+str(q),dim=d)
        state += gammas[q]*basis_el
    # Normalize the state
    state = state.unit()
    # Define the projectors
    A_ops = generalized_projectors_d3(d,m,A_setting[0],A_setting[1]) # [0] for theta [1] for phi
    B_ops = generalized_projectors_d3(d,m,B_setting[0],B_setting[1])
    # Compute the probabilities
    probs = np.zeros((d,d))
    for a in range(len(A_ops)):
        for b in range(len(B_ops)):
            # Define tensor product operator
            AB_op = qtp.tensor(A_ops[a],B_ops[b])
            probs[a,b] = qtp.expect(AB_op,state)
    return probs

# --> Noisy probabilities for d=3
def noisy_probs_generalized_d3(d,m,A_settings,B_settings,eta,gammas):
    # Define dictionary for saving data
    all_probs = {}
    # Define white noise
    white_noise = (1/(d**2))*np.ones((d,d))
    # Compute the corresponding probabilities
    for x in range(m):
        for y in range(m):
            all_probs[str(x+1)+str(y+1)] = (eta*ideal_probs_generalized_d3(d,m,A_settings[x],B_settings[y],gammas)
                                        +(1-eta)*white_noise)
    return all_probs

# --> Compute iextra probabilities for d=3
def extra_probs_generalized_d3(d,m,eta,A_setting,B_setting,gammas):
    # Define our quantum state
    state = 0
    for q in range(0,d):
        basis_el = qtp.ket(str(q)+str(q),dim=d)
        state += gammas[q]*basis_el
    # Normalize the state
    state = state.unit()
    # Define the projectors
    A_ops = generalized_projectors_d3(d,m,A_setting[0],A_setting[1])
    B_ops = generalized_projectors_d3(d,m,B_setting[0],B_setting[1])
    # Compute the probabilities
    probs = np.zeros((d,d))
    for a in range(len(A_ops)):
        for b in range(len(B_ops)):
            # Define tensor product operator
            AB_op = qtp.tensor(A_ops[a],B_ops[b])
            probs[a,b] = qtp.expect(AB_op,state)
    # Define white noise
    white_noise = (1/(d**2))*np.ones((d,d))
    # Compute probabilities
    noise_probs = eta*probs + (1-eta)*white_noise
    return noise_probs 

# --> Define objective function for optimizing probability settings for d=3
def optimize_settings_Alexia_objective_d3(d,m,gammas,settings):
    # Define Alexia's settings
    Alexia_settings = []
    for x in range(1,m+1):
        Alexia_settings.append((x-0.5)/m)
    Alexia_settings.append(0.5/m)
    for y in range(1,m+1):
        Alexia_settings.append(y/m)
    
    # Reshape settings
    settings = np.reshape(settings,(2*m,2))
    # Compute Alexia's probabilities
    alexia_probs = all_probs_noise_settings(d,m,1,gammas,Alexia_settings[:m],Alexia_settings[-(m):])
    # Compute our probabilities
    our_probs = noisy_probs_generalized_d3(d,m,settings[:m],settings[-m:],1,gammas)
    # Compute the distance between probs
    obj = 0
    for x in range(1,m+1):
        for y in range(1,m+1):
            obj += np.sum(np.abs(alexia_probs[str(x)+str(y)] - our_probs[str(x)+str(y)]))
    return obj

# --> Optimize the trace distance between probabilities for d=3
def optimize_settings_Alexia_d3(d,m,gammas,error = 0.1):
    val = 1
    while val > error:
        # Define initial settings at random
        init_settings = np.random.rand(2*2*m)
        
        # Define objective function as lambda function
        func = lambda x: optimize_settings_Alexia_objective_d3(d,m,gammas,x)
        # Define bounds for the optimization
        bnds =[]
        for i in range(0,2*2*m):
            bnds.append([0,2*np.pi])
        
        # Do the optimization
        res = minimize(func, init_settings, method = "Nelder-Mead", bounds = bnds, options = {"maxiter":1000,"fatol":0.001, "disp":True})
        # Print the value
        val = func(res.x)
    return res.x, func(res.x)

# ---------------------- MOST GENERAL MEASUREMENTS (d=2) ----------------------
# ---> Define general projectors for d=2
def general_projectors_d3(d,settings):
    # Split settings
    ph0 = settings[0]
    th0 = settings[1]
    ph1 = settings[2]
    th1 = settings[3]
    al0 = settings[4]
    be0 = settings[5]
    al1 = settings[6]
    be1 = settings[7]
    # Define the eigenstate
    a1 = (np.cos(ph0)*np.sin(th0)*qtp.ket("0",dim=d)
          + np.exp(1)**((1j*(-2/3))*al0*np.pi)*np.sin(ph0)*np.sin(th0)*qtp.ket("1",dim=d)
          + np.exp(1)**((1j*(-4/3))*be0*np.pi)*np.cos(th0)*qtp.ket("2",dim=d)).unit()
    # Generate projector
    A1 = a1*a1.dag();
    # Define the second eigenstate
    a2 = ((np.cos(ph1)*(np.cos(th0)**2+np.sin(ph0)**2*np.sin(th0)**2)*np.sin(th1)
          +(-1)*np.cos(ph0)*np.sin(th0)*(np.exp(1)**((1j*(4/3))*(1+be0+(-1)*be1)*np.pi)*np.cos(th0)*np.cos(th1)
                                         +np.exp(1)**((1j*(2/3))*(1+al0+(-1)*al1)*np.pi)*np.sin(ph0)*np.sin(ph1)*np.sin(th0)*np.sin(th1)))*qtp.ket("0",dim=d)
          +(np.exp(1)**((1j*(-2/3))*al0*np.pi)*((-1)**(1/3)*np.exp(1)**((1j*(4/3))*(be0+(-1)*be1)*np.pi)*np.cos(th0)*np.cos(th1)*np.sin(ph0)*np.sin(th0)+(-1)**(2/3)*np.exp(1)**((1j*(2/3))*(al0+(-1)*al1)*np.pi)*np.cos(th0)**2*np.sin(ph1)*np.sin(th1)+np.cos(ph0)*((-1)*np.cos(ph1)*np.sin(ph0)+(-1)**(2/3)*np.exp(1)**((1j*(2/3))*(al0+(-1)*al1)*np.pi)*np.cos(ph0)*np.sin(ph1))*np.sin(th0)**2*np.sin(th1)))*qtp.ket("1",dim=d)
          +((-1)*np.exp(1)**((1j*(-4/3))*be0*np.pi)*np.sin(th0)*((-1)**(1/3)*np.exp(1)**((1j*(4/3))*(be0+(-1)*be1)*np.pi)*np.cos(th1)*np.sin(th0)+np.cos(th0)*(np.cos(ph0)*np.cos(ph1)+(-1)**(2/3)*np.exp(1)**((1j*(2/3))*(al0+(-1)*al1)*np.pi)*np.sin(ph0)*np.sin(ph1))*np.sin(th1)))*qtp.ket("2",dim=d)).unit()
    # Generate projector
    A2 = a2*a2.dag();
    # Compute the last projector
    A3 = qtp.qeye(3) - A1-A2
    return [A1,A2,A3]

# ---> Define general projectors for d3
def general_projectors_d2(d,settings):
    # Split settings
    th0 = settings[0];
    ph0 = settings[1];
    # Define the eigenstate
    a1 = (np.cos(th0)*qtp.ket("0") + np.exp(1j*np.pi*(0-ph0))*np.sin(th0)*qtp.ket("1")).unit()
    # Generate projector
    A1 = a1*a1.dag();
    # Compute A2
    A2 = qtp.qeye(2) - A1
    return [A1,A2]

# --> Ordering the parameters for Alice and Bob
def order_parameters(m,allsettings):
    # Define dictionary
    settingsA = {}
    settingsB = {}
    # Define length of the parameters
    lengthparams=2;
    # Order the parameters
    for i in range(m):
        settingsA[str(i+1)] = allsettings[(i*lengthparams):(i+1)*lengthparams]
        settingsB[str(i+1)] = allsettings[((2+i)*lengthparams):(2+i+1)*lengthparams]
    return settingsA, settingsB

# --> Ordering the parameters for Alice and Bob
def order_parameters_d3(m,allsettings):
    # Define dictionary
    settingsA = {}
    settingsB = {}
    # Define length of the parameters
    lengthparams=8;
    # Order the parameters
    for i in range(m):
        settingsA[str(i+1)] = allsettings[(i*lengthparams):(i+1)*lengthparams]
        settingsB[str(i+1)] = allsettings[((2+i)*lengthparams):(2+i+1)*lengthparams]
    return settingsA, settingsB

# ---> Define the noisy measurements
def AB_proj_operators_noise_d2(d,m,a,b,x,y,settingsA,settingsB,eta):
    # Compute projectors
    projA = general_projectors_d2(d,settingsA)
    projB = general_projectors_d2(d,settingsB)
    ident = qtp.qeye(2)
    # Add the noise
    A_op_noisy = eta*projA[a] + (1-eta)*ident/d
    B_op_noisy = eta*projB[b] + (1-eta)*ident/d
    # Compute the total projector
    operator = qtp.tensor(A_op_noisy,B_op_noisy)
    return A_op_noisy,B_op_noisy,operator

# ---> Define the noisy measurements
def AB_proj_operators_noise_d3(d,m,a,b,x,y,settingsA,settingsB,eta):
    # Compute projectors
    projA = general_projectors_d3(d,settingsA)
    projB = general_projectors_d3(d,settingsB)
    ident = qtp.qeye(3)
    # Add the noise
    A_op_noisy = eta*projA[a] + (1-eta)*ident/d
    B_op_noisy = eta*projB[b] + (1-eta)*ident/d
    # Compute the total projector
    operator = qtp.tensor(A_op_noisy,B_op_noisy)
    return A_op_noisy,B_op_noisy,operator

# --> Generate quantum state
def quantum_state_d2(d,state):
    # Define quantum state
    qstate = 0*qtp.ket("00")
    k=0
    for i in range(d):
        for j in range(d):
            qstate += float(state[k])*qtp.ket(str(i)+str(j))
            k+=1
    return qstate.unit()

# --> Generate quantum state
def quantum_state_d3(d,state):
    # Define quantum state
    qstate = 0*qtp.ket("00",dim=d)
    k=0
    for i in range(d):
        for j in range(d):
            qstate += float(state[k])*qtp.ket(str(i)+str(j),dim=d)
            k+=1
    return qstate.unit()
       
# --> Define the single probabilities
def single_prob_for_settings_d2(d,m,x,y,state,settingsA,settingsB,eta):
    # Define the ket state
    ketstate = quantum_state_d2(d,state)
    # Define the bra state
    brastate = ketstate.dag()
    # Define an array for saving the probabilities
    probs = np.zeros((d,d))
    for a in range(d):
        for b in range(d):
            # obtain the quantum operators
            A_op,B_op,operator = AB_proj_operators_noise_d2(d,m,a,b,x,y,settingsA,settingsB,eta)
            # Compute the mean value
            mel = brastate*operator*ketstate
            probs[a,b]=float(np.real(np.array(mel)))
    return probs

# --> Define the single probabilities
def single_prob_for_settings_d3(d,m,x,y,state,settingsA,settingsB,eta):
    # Define the ket state
    ketstate = quantum_state_d3(d,state)
    # Define the bra state
    brastate = ketstate.dag()
    # Define an array for saving the probabilities
    probs = np.zeros((d,d))
    for a in range(d):
        for b in range(d):
            # obtain the quantum operators
            A_op,B_op,operator = AB_proj_operators_noise_d3(d,m,a,b,x,y,settingsA,settingsB,eta)
            # Compute the mean value
            mel = brastate*operator*ketstate
            probs[a,b]=float(np.real(np.array(mel)))
    return probs

# --> Define all the probabilities
def all_probs_noise_d2(d,m,eta,allsettings,state):
    # First we organize the thetas and the phis
    settingsA,settingsB = order_parameters(m,allsettings)
    # Define dictionary for saving data
    all_probs = {}
    # Compute the probabilities
    for x in range(1,m+1):
        for y in range(1,m+1):
            all_probs[str(x)+str(y)] = single_prob_for_settings_d2(d,m,x,y,state,settingsA[str(x)],settingsB[str(y)],eta)
    return all_probs

# --> Define all the probabilities
def all_probs_noise_d3(d,m,eta,allsettings,state):
    # First we organize the thetas and the phis
    settingsA,settingsB = order_parameters_d3(m,allsettings)
    # Define dictionary for saving data
    all_probs = {}
    # Compute the probabilities
    for x in range(1,m+1):
        for y in range(1,m+1):
            all_probs[str(x)+str(y)] = single_prob_for_settings_d3(d,m,x,y,state,settingsA[str(x)],settingsB[str(y)],eta)
    return all_probs

# --> Define marginal probabilities
def marginal_probs_v2(m,all_probs):
    # Define dictionary for saving Alice and Bob probabilities
    P_Alice = {}
    P_Bob = {}
    # Compute the marginal probabilities
    for i in range(1,m+1):
        P_Alice[str(i)] = np.sum(all_probs[str(i)+str(i)],1)
        P_Bob[str(i)] = np.sum(all_probs[str(i)+str(i)],0)
    return P_Alice,P_Bob
# ------------------------- SEMIDEFINITE PROGRAMMING --------------------------
# --> Objective function for NPA relaxation of arXiv:2106.13692
def objective(A,B,Z,ti):
    # Define initial objective
    obj = 0.0
    # Get Alice operators
    F = ncp.flatten([A[0], 1- np.sum(A[0])])
    #F = ncp.flatten(A[0])
    # Obtain objective function (Gauss-Radau quadratures introduced later)
    for a in range(len(F)):
        obj += F[a] * (Z[a] + Dagger(Z[a]) + (1-ti)*Dagger(Z[a])*Z[a]) + ti*Z[a]*Dagger(Z[a])
    return obj

# --> Objective function with the noise model Toni proposed
def objective_Toni(A,B,Z,ti,eta):
    # Define initial objective
    obj = 0.0
    # Get Alice operators
    F = [eta*Ai for Ai in ncp.flatten(A[0])]
    F+= [eta*(1-np.sum(A[0]))]
    F+= [1-np.sum(F)]
    # Obtain objective function (Gauss-Radau quadratures introduced later)
    for a in range(len(F)):
        obj += F[a] * (Z[a] + Dagger(Z[a]) + (1-ti)*Dagger(Z[a])*Z[a]) + ti*Z[a]*Dagger(Z[a])
    return obj

# --> Substitutions to introduce in certificate
def get_subs(A,B,Z):
    """
        Returns any substitution rules to use with ncpol2sdpa. E.g. projections and
        commutation relations.
    """
    # Define dictionary where we load the substitutions
    subs = {}
    # Get Alice and Bob's projective measurement constraints
    subs.update(ncp.projective_measurement_constraints(A,B))

    # Finally we note that Alice and Bob's operators should All commute with Eve's ops
    for a in ncp.flatten([A,B]):
        for z in Z:
            subs.update({z*a : a*z, Dagger(z)*a : a*Dagger(z)})
    return subs

# --> Define extra monomials to introduce in the certificate
def get_extra_monomials(A,B,Z):
    """
        Returns additional monomials to add to sdp relaxation.
    """
    # Define list where we introduce monomials
    monos = []
    # Add ABZ
    ZZ = Z + [Dagger(z) for z in Z]
    Aflat = ncp.flatten(A)
    Bflat = ncp.flatten(B)
    for a in Aflat:
        for b in Bflat:
            for z in ZZ:
                monos += [a*b*z]
    # Add monos appearing in objective function
    for z in Z:
        monos += [A[0][0]*Dagger(z)*z]
    return monos[:]

# --> Define equality constrains based on the probabilities
def get_prob_eqs(A,B,Prob_joints):
    """
        Returns the equalities for the moments defined in our certificate
    """
    eqs = []
    
    # Obtain marginal probabilities
    P_Alice,P_Bob = marginal_probs_v2(len(A),Prob_joints)
    

    # Marginal probabilities for Alice
    for i in range(1,np.shape(A)[0]+1):
        for j in range(1,np.shape(A)[1]+1):
            eqs.append(A[i-1][j-1] - float(P_Alice[str(i)][j-1]))
        
    # Marginal probabilities for Bob
    for i in range(1,np.shape(B)[0]+1):
        for j in range(1,np.shape(B)[1]+1):
            eqs.append(B[i-1][j-1] - float(P_Bob[str(i)][j-1]))
 
    # Joint probabilities
    for i in range(1,len(A)+1):
        for j in range(1,len(B)+1):
            prob = Prob_joints[str(i)+str(j)]
            for k in range(0,len(A[0])):
                for l in range(0,len(B[0])):
                    eqs.append(A[i-1][k]*B[j-1][l] - float(prob[k][l]))
                    #eqs.append((1-A[i-1][k])*B[j-1][l] - float(prob[k+1][l]))
                    #eqs.append(A[i-1][k]*(1-B[j-1][l]) - float(prob[k][l+1]))
                    #eqs.append((1-A[i-1][k])*(1-B[j-1][l]) - float(prob[k+1][l+1]))
    return eqs

# --> Compute entropy
def compute_entropy(SDP,T,W,A,B,Z,KEEP_M=0):
    """
    Computes lower bound on H(A|X=0,E) using the fast (but less tight) method
        SDP -- sdp relaxation object
    """
    ck = 0.0        # kth coefficient
    ent = 0.0        # lower bound on H(A|X=0,E)

    # We can also decide whether to perform the final optimization in the sequence
    # or bound it trivially. Best to keep it unless running into numerical problems
    if KEEP_M:
        num_opt = len(T)
    else:
        num_opt = len(T) - 1

    for k in range(num_opt):
        ck = W[k]/(T[k] * log(2))

        # Get the k-th objective function
        new_objective = objective(A,B,Z,T[k])

        SDP.set_objective(new_objective)
        SDP.solve("mosek")
        ent += ck * (1 + SDP.dual)
    return ent

# --> Compute entropy with Toni's model of noise
def compute_entropy_Toni(SDP,T,W,eta,A,B,Z,KEEP_M=0):
    """
    Computes lower bound on H(A|X=0,E) using the fast (but less tight) method
        SDP -- sdp relaxation object
    """
    ck = 0.0        # kth coefficient
    ent = 0.0        # lower bound on H(A|X=0,E)

    # We can also decide whether to perform the final optimization in the sequence
    # or bound it trivially. Best to keep it unless running into numerical problems
    if KEEP_M:
        num_opt = len(T)
    else:
        num_opt = len(T) - 1

    for k in range(num_opt):
        ck = W[k]/(T[k] * log(2))

        # Get the k-th objective function
        new_objective = objective_Toni(A,B,Z,T[k],eta)

        SDP.set_objective(new_objective)
        SDP.solve('mosek')
        ent += ck * (1 + SDP.dual)
    return ent

# --> Optimize entropy
def optimize_entropy(SDP,cons,A,B,Z,T,W,Prob_joint,KEEP_M = 0):
    # Modify the constraints of the SDP
    SDP.process_constraints(equalities = cons[1][:],
        inequalities = cons[2][:],
        momentequalities = get_prob_eqs(A,B,Prob_joint),
        momentinequalities = cons[0])
    
    # Compute the entropy
    ent = compute_entropy(SDP,T,W,A,B,Z)
    return ent

# --> Optimize entropy with Toni's noise
def optimize_entropy_Toni(SDP,cons,A,B,Z,T,W,eta,Prob_joint,KEEP_M = 0):
    # Modify the constraints of the SDP
    SDP.process_constraints(equalities = cons[1][:],
        inequalities = cons[2][:],
        momentequalities = get_prob_eqs(A,B,Prob_joint),
        momentinequalities = cons[0])
    
    # Compute the entropy
    ent = compute_entropy_Toni(SDP,T,W,eta,A,B,Z)
    return ent

# --> SDP definer
def sdp_definer(m,d,Prob_joint,KEEP_M = 0):
    """
        Given a probability distribution set, it defines an sdp relaxation
    """

    LEVEL = 2                        # Level of the NPA hierarchy                      
    VERBOSE = 1                      # If you need a lot of info
    
    # Defining the configurations for Alice and Bob
    A_config = m*[d]
    B_config = m*[d]
    
    # Operators for Alice, Bob and Eve
    A = [Ai for Ai in ncp.generate_measurements(A_config, 'A')]
    B = [Bj for Bj in ncp.generate_measurements(B_config, 'B')]
    Z = ncp.generate_operators('Z', d, hermitian=0)
    
    # Constraints list
    substitutions = {}   # substitutions to be made (e.g. projections)
    moment_ineqs = []    # Moment inequalities (e.g. Tr[rho CHSH] >= c)
    moment_eqs = []      # Moment equalities (not needed here)
    op_eqs = []          # Operator equalities (not needed here)
    op_ineqs = []        # Operator inequalities (e.g. Id - A00 >= 0 -- we don't include for speed)
    extra_monos = []     # Extra monomials to add to the relaxation beyond the level.

    # Constraints from the functions
    substitutions = get_subs(A,B,Z)
    moment_eqs = get_prob_eqs(A,B,Prob_joint)
    extra_monos = get_extra_monomials(A,B,Z)
    
    # Define the objective function (changed later)
    obj = objective(A,B,Z,1)
    
    # Finally defining the sdp relaxation in ncpol2sdpa
    ops = ncp.flatten([A,B,Z])
    sdp = ncp.SdpRelaxation(ops, verbose = VERBOSE-1, normalized=True, parallel=0)
    sdp.get_relaxation(level = LEVEL,
        equalities = op_eqs[:],
        inequalities = op_ineqs[:],
        momentequalities = moment_eqs[:],
        momentinequalities = moment_ineqs[:],
        objective = obj,
        substitutions = substitutions,
        extramonomials = extra_monos)
    
    return sdp,[moment_ineqs,op_eqs,op_ineqs,extra_monos],A,B,Z

# --> SDP definer for Toni's noise model (we have an extra input)
def sdp_definer_Toni(m,d,eta,Prob_joint,KEEP_M = 0):
    """
        Given a probability distribution set, it defines an sdp relaxation
    """

    LEVEL = 2                        # Level of the NPA hierarchy                      
    VERBOSE = 1                      # If you need a lot of info
    
    # Defining the configurations for Alice and Bob
    A_config = m*[d]
    B_config = m*[d]
    
    # Operators for Alice, Bob and Eve
    A = [Ai for Ai in ncp.generate_measurements(A_config, 'A')]
    B = [Bj for Bj in ncp.generate_measurements(B_config, 'B')]
    Z = ncp.generate_operators('Z', d+1, hermitian=0)
    
    # Constraints list
    substitutions = {}   # substitutions to be made (e.g. projections)
    moment_ineqs = []    # Moment inequalities (e.g. Tr[rho CHSH] >= c)
    moment_eqs = []      # Moment equalities (not needed here)
    op_eqs = []          # Operator equalities (not needed here)
    op_ineqs = []        # Operator inequalities (e.g. Id - A00 >= 0 -- we don't include for speed)
    extra_monos = []     # Extra monomials to add to the relaxation beyond the level.

    # Constraints from the functions
    substitutions = get_subs(A,B,Z)
    moment_eqs = get_prob_eqs(A,B,Prob_joint)
    extra_monos = get_extra_monomials(A,B,Z)
    
    # Define the objective function (changed later)
    obj = objective_Toni(A,B,Z,1,eta)
    
    # Finally defining the sdp relaxation in ncpol2sdpa
    ops = ncp.flatten([A,B,Z])
    sdp = ncp.SdpRelaxation(ops, verbose = VERBOSE-1, normalized=True, parallel=0)
    sdp.get_relaxation(level = LEVEL,
        equalities = op_eqs[:],
        inequalities = op_ineqs[:],
        momentequalities = moment_eqs[:],
        momentinequalities = moment_ineqs[:],
        objective = obj,
        substitutions = substitutions,
        extramonomials = extra_monos)
    
    return sdp,[moment_ineqs,op_eqs,op_ineqs,extra_monos],A,B,Z
    
# ------------------------ GAUSS-RADAU QUADRATURES ----------------------------
# --> Coeficients of the Jacobi matrix, obtained from the 3-term recurrence coeffs
def betafun(n):
    beta = np.zeros(n-1)
    for idex in range(n-1):
        beta[idex] = 0.5/sqrt(1-(2*(idex+1))**(-2))
    return beta

# --> Jacobian matrix for Gauss quadrature (M=0)
def jacobian_m(n):   
    bet_aux = betafun(n)
    J = np.diag(bet_aux, 1) + np.diag(bet_aux, -1) 
    return J

# --> Jacobian matrix for the Gauss Radau method
def jacobian_GR(N):
    n = N - 1
    a = -1 # Lower limit of [-1,1], which is where Gauss (M=0) is solved
    jac = jacobian_m(n+1)
    red_jac = jac[:n,:n] 
    mat = red_jac - a*np.eye(n)
    constant = np.zeros(n)
    constant[n-1] = jac[n-1][n]**2
    res = np.linalg.solve(mat, constant)
    w_extra = res[n-1] + a
    jac[n][n] = w_extra
    return jac

# --> Calculate the nodes and weights via the GW algorithm
def GWel(n):
    interval = [-1,0]                 # Equivalent to [0,1] for how we've written the code  
    if n>1:
        J = jacobian_GR(n)            # Jacobian matrix for Gauss-Radau
        t_GW, v_GW = np.linalg.eig(J) # The nodes are just the eigenvalues    

        # We calculate the weights
        idex_sort = np.argsort(t_GW)
        t_GW = np.sort(t_GW)
        v_0 = v_GW[0,idex_sort]       # Fisrt element of v sorted after ascending t       
        w_GW = np.zeros(n)
        for idex in range(n):
            w_GW[idex] = 2*v_0[idex]**2

        # We correct for the input interval [cite]
        dab = np.diff(interval, n=1, axis=0)
        t_GW = (t_GW+1)/2*dab + interval[0]
        t_GW = -t_GW[::-1] # Such that t_m = 1 is the last element of the array
        w_GW = dab*w_GW/2  
        w_GW = w_GW[::-1]  # Such that w_m = 1/m**2 is the last element of the array
        
    else: # Trivial case
        t_GW = [1]
        w_GW = [1]
    
    return t_GW,w_GW

# ----------------------------- ENTROPY FUNCTIONS -----------------------------
# --> Relative entropy
def relative_entropy(d,p_joint,p_marg):
    if p_marg == None:
        p_marg = np.sum(p_joint,axis=0)
    hjoint, hb = 0.,0.
    for prob in p_joint.flatten():
        if 0 < prob < 1:
            hjoint += -prob*log(prob,d)
            
    for prob in p_marg:
        if 0 < prob < 1:
            hb += -prob*log(prob,d)
    
    return hjoint - hb

# --> All computation of relative entropy
def all_relative(d,m,gammas):
    # Compute the probabilities
    all_probs = all_probs_ideal(d,m,gammas)
    # Compute marginal
    P_Alice,P_Bob = marginal_probabilities(m,all_probs)
    # Compute conditional entropy
    ent = relative_entropy(d,all_probs["11"].flatten(),P_Bob)
    return ent

# --> Optimize entropy
def optimize_relative(d,m,initial_guess = [0.5,0.5,0.5]):
    func = lambda x: all_relative(d,m,x)
    res = minimize(func,initial_guess,method = "Nelder-Mead",options = {"disp":True})
    return func(res.x)

# --> Define objective function
def objective_relative_settings(d,m,eta,gammas,A_settings,opt_settings):
    # Define the function to optimize
    probs = extra_meas_Bob_settings(d,m,eta,gammas,A_settings[0],opt_settings)
    func = relative_entropy(2, probs, None)
    return func

# --> Optimize settings in relative entropy
def optimize_relative_settings(d,m,eta,gammas,A_settings,init_settings):
    # Define function to optimize
    func = lambda settings: objective_relative_settings(d,m,eta,gammas,A_settings,float(settings))
    # Perform the optimization
    res = minimize(func,init_settings,method = "Nelder-Mead",options = {"disp":True})
    # Compute the entropy
    HAB = func(res.x)
    return HAB

# ----------------------------- KEY RATE FUNCTIONS ----------------------------
# --> WHITE NOISE (EVEN WITH GENERALIZED MEASUREMENTS)
# --> Given some probability distribution, computes the settings
def compute_key_rate(SDP,cons,A,B,Z,T,W,Prob_joint,extra_prob,KEEP_M = 0):
    # Compute the relative entropy between Alice and Bob
    HAB = relative_entropy(2,extra_prob,None)
    # Compute the relative entropy between Alice and Eve
    HAE = optimize_entropy(SDP,cons,A,B,Z,T,W,Prob_joint,KEEP_M = KEEP_M)
    print(HAE)
    # Compute the key rate
    r = HAE - HAB
    return r

# --> Compute the key rate given some settings
def compute_key_with_settings(SDP,cons,A,B,Z,T,W,d,m,eta,gamma,settings,KEEP_M=0):
    # Order the settings as corresponds
    A_settings = settings[0:m]
    B_settings = settings[(m+1):]
    extra_B_settings = settings[m]
    # Compute the probabilities
    all_probs = all_probs_noise_settings(d,m,eta,gamma,A_settings,B_settings)
    extra_prob = extra_meas_Bob_settings(d,m,eta,gamma,A_settings[0],extra_B_settings)  
    # Compute key rate
    key = compute_key_rate(SDP,cons,A,B,Z,T,W,all_probs,extra_prob,KEEP_M = 0)
    return key

# --> Compute the key rate given some settings
def compute_key_generalized_d2(SDP,cons,A,B,Z,T,W,d,m,eta,gammas,settings,KEEP_M=0):
    # Order the settings as corresponds
    A_settings = settings[0:m]
    B_settings = settings[(m+1):]
    extra_B_settings = settings[m]
    # Compute probabilities  
    all_probs = noisy_probs_generalized_d2(d,m,A_settings,B_settings,eta,gammas)
    extra_prob = extra_probs_generalized_d2(d,m,eta,A_settings[0],extra_B_settings,gammas)
    # Compute key rate
    key = compute_key_rate(SDP,cons,A,B,Z,T,W,all_probs,extra_prob,KEEP_M = 0)
    return key

# --> Compute the key rate given some settings with Enky's noise
def compute_key_generalized_Enky_d2(SDP,cons,A,B,Z,T,W,d,m,theta,gammas,settings,KEEP_M=0):
    # Order the settings as corresponds
    A_settings = settings[0:m]
    B_settings = settings[(m+1):]
    extra_B_settings = settings[m]
    # Compute probabilities  
    all_probs = all_probs_Enky_noise_generalized_d2(d,m,theta,gammas,A_settings,B_settings)
    extra_prob = Enky_noise_prob_distrib_generalized_d2(d,m,theta,gammas,A_settings[0],extra_B_settings)
    # Compute key rate
    key = compute_key_rate(SDP,cons,A,B,Z,T,W,all_probs,extra_prob,KEEP_M = 0)
    return key

# --> Optimize key rate
def optimize_rate(SDP,cons,A,B,Z,T,W,d,m,eta,gamma,settings,KEEP_M=0):
    # Define lambda function over which to optimize
    func = lambda x: (-1)*compute_key_with_settings(SDP,cons,A,B,Z,T,W,d,m,eta,gamma,x,KEEP_M=KEEP_M)
    # Define bounds for the optimization
    bnds =[]
    for i in range(0,2*m+1):
        bnds.append([0,2*np.pi])
    # Do the optimization
    res = minimize(func, settings, method = "Nelder-Mead", bounds = bnds, options = {"maxiter":50,"fatol":0.001, "disp":True})
    return res.x, (-1)*func(res.x)

# --> Optimize key rate with generalized settings (for d2)
def optimize_rate_generalized_d2(SDP,cons,A,B,Z,T,W,d,m,eta,gammas,settings,KEEP_M=0):
    # Define lambda function over which to optimize
    func = lambda x: (-1)*compute_key_generalized_d2(SDP,cons,A,B,Z,T,W,d,m,eta,gammas,x,KEEP_M=KEEP_M)
    # Define bounds for the optimization
    bnds =[]
    for i in range(0,2*m+1):
        bnds.append([0,2*np.pi])
    # Do the optimization
    res = minimize(func, settings, method = "Nelder-Mead", bounds = bnds, options = {"maxiter":1000,"fatol":0.001, "disp":True})
    return res.x, (-1)*func(res.x)

# --> Optimize key rate with generalized settings (for d2 and Enky's noise)
def optimize_rate_generalized_Enky_d2(SDP,cons,A,B,Z,T,W,d,m,theta,gammas,settings,KEEP_M=0):
    # Define lambda function over which to optimize
    func = lambda x: (-1)*compute_key_generalized_Enky_d2(SDP,cons,A,B,Z,T,W,d,m,theta,gammas,x,KEEP_M=KEEP_M)
    # Define bounds for the optimization
    bnds =[]
    for i in range(0,2*m+1):
        bnds.append([0,2*np.pi])
    # Do the optimization
    res = minimize(func, settings, method = "Nelder-Mead", bounds = bnds, options = {"maxiter":1000,"fatol":0.001, "disp":True})
    return res.x, (-1)*func(res.x)

# --> ENKY'S NOISE (PHOTON ABSORPTION)
# --> Compute the key rate given some settings for Enky's noise model
def compute_key_with_settings_Enky_noise(SDP,cons,A,B,Z,T,W,d,m,theta,gamma,settings,KEEP_M=0):
    # Order the settings as corresponds
    A_settings = settings[0:m]
    B_settings = settings[(m+1):]
    extra_B_settings = settings[m]
    # Compute the probabilities
    all_probs = all_probs_Enky_noise_settings(d,m,theta,gamma,A_settings,B_settings)
    extra_prob = Enky_noise_prob_distrib_optsettings(d,m,theta,gamma,A_settings[0],extra_B_settings)
    # Compute key rate
    key = compute_key_rate(SDP,cons,A,B,Z,T,W,all_probs,extra_prob,KEEP_M = 0)
    return key

# Optimize rate
def optimize_rate_with_settings_Enky_noise(SDP,cons,A,B,Z,T,W,d,m,theta,gamma,settings,KEEP_M=0):
    # Define lambda function over which to optimize
    func = lambda x: (-1)*compute_key_with_settings_Enky_noise(SDP,cons,A,B,Z,T,W,d,m,theta,gamma,x,KEEP_M=KEEP_M)
    # Define bounds for the optimization
    bnds =[]
    for i in range(0,2*m+1):
        bnds.append([0,2*np.pi])
    # Do the optimization
    res = minimize(func, settings, method = "Nelder-Mead", bounds = bnds, options = {"maxiter":1000,"fatol":0.001, "disp":True})
    return res.x, (-1)*func(res.x)


"""
# --> Optimize key rate for Enky's noise model
def optimize_rate_Enky_noise(SDP,cons,A,B,Z,T,W,d,m,eta,settings,KEEP_M=0):
    # Define lambda function over which to optimize
    func = lambda x: (-1)*compute_key_with_settings_Enky_noise(SDP,cons,A,B,Z,T,W,d,m,theta,x,KEEP_M=KEEP_M)
    # Define bounds for the optimization
    bnds =[]
    for i in range(0,2*m+1):
        bnds.append([0,2*np.pi])
    # Do the optimization
    res = minimize(func, settings, method = "Nelder-Mead", bounds = bnds, options = {"maxiter":50,"fatol":0.001, "disp":True})
    return res.x, (-1)*func(res.x)
"""  
# ----------------------------- ALEXIA'S INEQUALITIES -------------------------
# --> Define alpha and beta coefficients
def alpha_beta_func(d,m,k):
    # Define g function
    g = lambda x: 1/np.tan(np.pi*(x+1/(2*m))/d)
    # Compute the coefficients
    alphak = (1/(2*d))*np.tan(np.pi/(2*m))*(g(k) - g(np.floor(d/2)))
    betak = (1/(2*d))*np.tan(np.pi/(2*m))*(g(k+1-1/m)+g(np.floor(d/2)))
    return alphak,betak

# --> Define the a coefficients
def a_coefficient(d,m,l):
    # Define parameters
    w = np.exp(1j*2*np.pi/d)
    # Compute the coefficient
    coeff = 0
    for k in range(0,floor(d/2)):
        alphak,betak = alpha_beta_func(d,m,k)
        coeff += alphak*w**(-k*l) - betak*w**((k+1)*l)
    return coeff

# --> Define A and B correlators
def AB_corr(x,y,k,l,all_probs):
    # Define the dimension
    d = np.shape(all_probs["11"])[0]
    # Define w
    w = np.exp(1j*2*np.pi/d)
    # Compute the correlator
    corr = 0
    for a in range(0,d):
        for b in range(0,d):
            corr += w**(a*k+b*l)*all_probs[str(x)+str(y)][a,b]
    return corr

# --> Define A and \bar{B} correlators
def ABbar_corr(d,m,i,l,all_probs):
    # Define parameters
    a_coeff = a_coefficient(d,m,l)
    w = np.exp(1j*2*np.pi/d)
    # Consider the two possible cases cases
    if i == 1: # If i == 1, then B has a particular definition
        first = AB_corr(i,1,l,d-l,all_probs)
        second = AB_corr(i,m,l,d-l,all_probs)
        corr = a_coeff*first + np.conjugate(a_coeff)*(w**l)*second
    else:
        first = AB_corr(i,i,l,d-l,all_probs)
        second = AB_corr(i,i-1,l,d-l,all_probs)
        corr = a_coeff*first + np.conjugate(a_coeff)*second
    return corr

# --> Compute Alexia's inequality
def Alexia_inequality(d,m,all_probs):
    # Compute inequality
    ineq = 0
    for i in range(1,m+1):
        for l in range(1,d):
            ineq += ABbar_corr(d,m,i,l,all_probs)
    return ineq

# --> Classical bound
def classical_bound(d,m):
    # Define g function
    g = lambda x: 1/np.tan(np.pi*(x+1/(2*m))/d)
    # Compute bound
    bound = 0.5*np.tan(np.pi/(2*m))*((2*m-1)*g(0)-g(1-1/m))-m
    return bound

# --> Tsirelson bound
def Tsirelson_bound(d,m):
    return m*(d-1)

# --> Optimize Alexia's inequality
def opt_Alexia_inequality(d,m,eta,gammas):
    # Define probabilities
    all_probs = all_probs_noise(d,m,eta,gammas)
    # Compute inequality
    ineq = Alexia_inequality(d,m,all_probs)
    return (-1)*np.abs(ineq)
    
# ------------------------ EXPORT FILES TO MATLAB -----------------------------
# --> Function Cristian defined in order to save the files
def ncpol_export_to_matlab(sdp: ncp.SdpRelaxation,
                           filename: str) -> None:
    """Helper function to export an SDP to MATLAB format.


    Parameters
    ----------
    sdp : ncpol2sdpa.SdpRelaxation
        Instance of SdpRelaxation class
    filename : str
        The filename to save the MATLAB file to.
    """
    
    assert len(sdp.block_struct) == 1, "Export function currently works only for single block SDPs"  # <-- I have no idea how to handle multiple blocks
    mat_dim = sdp.block_struct[0]
    nof_vars = sdp.n_vars + 1  # <-- +1 for the constant variable
    print("Generating the MATLAB file...")
    # Monomial index to monomial string mapping
    monidx_to_monstring = {**{0: '0', 1: '1'},
                           **{idx + 1: str(mon)
                              for mon, idx in sdp.monomial_index.items()}}
    
    # Var with idx 0 has value 0, and idx 1 has value 1
    known_vars = np.array([[0, 0.0],
                           [1, 1.0]])
    
    # Find objective as a linear combination of monomial indices
    # ! Warning: ncpol2sdpa removes the constant offset from the objective
    objective = np.array([[monidx + 2, coeff]
                          for monidx, coeff in zip(monidx_to_monstring.keys(),
                                                   sdp.obj_facvar)
                          if not np.isclose(coeff, 0)])
    
    # An SDP variable F can be written as F = \sum_i x_i F_i where x_i are 
    # scalar variables, and F_i are coefficient matrices such that F_i(i, j)  
    # is coefficient of the variable x_i at F(i, j). Now we compute these.
    F_i = {}
    for idx in tqdm(range(nof_vars)):
        mat = sdp.F[:, idx].reshape(mat_dim, mat_dim)
        # Ncpol2sdpa only specifies the upper triangular, add lower triangular
        F_i[idx + 1] = mat.A + np.triu(mat.A, 1).T
    
    # ! Index matrix only makes sense if there is a single variable in at (i,j)
    IndexMatrix = sum([i*Fi for i, Fi in F_i.items()])

    # For completeness, find where the moment matrix is zero
    F_0 = np.zeros((mat_dim, mat_dim))
    F_0[np.where(IndexMatrix == 0)] = 1
    F_i[0] = F_0

    # Make idx start from 1 for MATLAB compatibility, i.e., idx=1 is the 0
    # variable, idx=2 is the '1' variable, and rest are the monomials
    known_vars[:, 0] += 1
    objective[:, 0] += 1 
    IndexMatrix += 1
    savemat(filename, {'IndexMatrix': IndexMatrix,
                       'idx2monomial': np.array([[idx, name]
                                                 for idx, name in monidx_to_monstring.items()],
                                                 dtype=object),
                       'known_vars': known_vars,
                       'objective': objective})
    
# --> Define the SDP relaxation
def sdpRelaxation_MATLAB(d,m):
    LEVEL = 2                        # Level of the NPA hierarchy                      
    VERBOSE = 1                      # If you need a lot of info
    
    # Defining the configurations for Alice and Bob
    A_config = m*[d]
    B_config = m*[d]
    
    # Operators for Alice, Bob and Eve
    A = [Ai for Ai in ncp.generate_measurements(A_config, 'A')]
    B = [Bj for Bj in ncp.generate_measurements(B_config, 'B')]
    Z = ncp.generate_operators('Z', d, hermitian=0)
    
    # Constraints list
    substitutions = {}   # substitutions to be made (e.g. projections)
    moment_ineqs = []    # Moment inequalities (e.g. Tr[rho CHSH] >= c)
    moment_eqs = []      # Moment equalities (not needed here)
    op_eqs = []          # Operator equalities (not needed here)
    op_ineqs = []        # Operator inequalities (e.g. Id - A00 >= 0 -- we don't include for speed)
    extra_monos = []     # Extra monomials to add to the relaxation beyond the level.

    # Constraints from the functions
    substitutions = get_subs(A,B,Z)
    extra_monos = get_extra_monomials(A,B,Z)
    
    # Define the objective function (changed later)
    obj = objective(A,B,Z,1)
    
    # Finally defining the sdp relaxation in ncpol2sdpa
    ops = ncp.flatten([A,B,Z])
    sdp = ncp.SdpRelaxation(ops, verbose = VERBOSE, normalized=True, parallel=0)
    sdp.get_relaxation(level = LEVEL,
        equalities = op_eqs[:],
        inequalities = op_ineqs[:],
        momentequalities = moment_eqs[:],
        momentinequalities = moment_ineqs[:],
        objective = obj,
        substitutions = substitutions,
        extramonomials = extra_monos)
    
    return sdp

# --> Save the SDP in a file readable with Matlab
def save_to_mat(d,m,path = "/Users/javier/Documents/PhD/Projects/DIKQD/Big dimensions/Codes/Matlab/Alexia's paper/SDP matlab"):
    # Save old dirrectory
    oldir = os.getcwd()
    # Define SDP
    SDP = sdpRelaxation_MATLAB(d,m)
    # Move to the new directory
    os.chdir(path)
    # Define filename
    filename = "SDP_relax_d"+str(d)+"_m"+str(m)
    # Save the SDP as a file readable with MATLAB
    ncpol_export_to_matlab(SDP,filename)
    print("File saved successfully!")
    # Return to the old directory
    os.chdir(oldir)
    
    

    
    