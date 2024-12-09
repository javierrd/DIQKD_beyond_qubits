#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 10:19:22 2023

@author: javier
"""

from main import *
# ---------------------------- DEFINING THE SDP -------------------------------
# Define parameters
M=16 # Number of Gauss-Radau quadratures
d=2 # Dimension (and therefore number of outcomes)
m=2 # Number of measurements for Alice and Bob
gammas = [1,1]
# ------------------------------------------------------------------------------
# ---> EXTRA
# Here we are going to add a plot to the one we have right now, which is just a
# comparison with the case where we have optimized over the measurement settings

# Move to the proper directory
os.chdir("/Users/javier/Documents/PhD/Projects/DIKQD/Big dimensions/Results/Brute_force_opt/d"+str(d)+"_m"+str(m))

# Load all the files
files = os.listdir()
files_txt = [file for file in files if file.endswith('20.txt')]
files_key = [file for file in files_txt if file.startswith('Key_rate')]
files_params = [file for file in files_txt if file.startswith('Optimal_params')]

#files_v2_txt = [file for file in files if file.endswith('v2.txt')]
#files_v2_key = [file for file in files_v2_txt if file.startswith('Key_rate')]
#files_v2_params = [file for file in files_v2_txt if file.startswith('Optimal_params')]

# Define lists for saving keys and etas
keys_opt = []
keys_opt_v2 = []
etas_opt = []
etas_opt_v2 = []
etas_params = []
opt_params = {}
opt_params_v2 = {}
for file in files_key:
    # Define eta
    etas_opt.append(float(file[13:-14]))
    # Look where in the files_params do we have such value of eta
    loc_params = np.where(np.array(files_params) == 'Optimal_params_eta_'+file[13:-14]+'_length_20.txt')[0]
    file_params = files_params[int(loc_params)]
    # Load and save the best key
    all_keys = np.loadtxt(file)
    keys_opt.append(np.max(all_keys))
    # Load the corresponding params 
    loc_max_key = np.where(all_keys == keys_opt[-1])[0]
    opt_settings = np.loadtxt(file_params)
    opt_params[etas_opt[-1]] = opt_settings[loc_max_key]
    #opt_params[etas_opt[-1]] = opt_settings

# For the other initializations    
#for file in files_v2_key:
#    # Define eta
#    etas_opt_v2.append(float(file[13:-18]))
#    # Look where in the files_params do we have such value of eta
#    loc_params = np.where(np.array(files_v2_params) == 'Optimal_params_eta_'+file[13:-17]+'_length_20_v2.txt')[0]
#    file_params = files_v2_params[int(loc_params)]
    # Load and save the best key
#    all_keys = np.loadtxt(file)
#    keys_opt_v2.append(np.max(all_keys))
    # Load the corresponding params 
#    loc_max_key = np.where(all_keys == keys_opt_v2[-1])[0]
#    opt_settings = np.loadtxt(file_params)
#    opt_params_v2[etas_opt_v2[-1]] = opt_settings[loc_max_key]


# -----------------------------------------------------------------------------
# Define initial probabilities
all_probs = all_probs_noise_settings(d,m,1,gammas,[0.25,0.75],[0.5,1])

# Define quadrature weights and SDP
T,W = GWel(M)
print("-------------------------------------------")
print("             PREPARING SDP")
print("-------------------------------------------")

SDP,cons,A,B,Z = sdp_definer(m,d,all_probs)

# List of etas
etas = np.linspace(0.8,1,30) # \eta = 1 means no-noise in this code

# List for saving results
key_rate = []
key_rate_opt = []
key_rate_opt_v2 = []

# We add the initial settings of Alexia
Alexia_settings = []
for x in range(1,m+1):
    Alexia_settings.append((x-0.5)/m)
Alexia_settings.append(0.5/m)
for y in range(1,m+1):
    Alexia_settings.append(y/m)

print("-------------------------------------------")
print("             RUNNING SDP")
print("-------------------------------------------")
for eta in tqdm(etas_opt):
    # --> Alexia's case
    # Define settings
    settings = copy.deepcopy(Alexia_settings)
    A_settings = settings[0:m]
    B_settings = settings[(m+1):]
    extra_B_settings = settings[m]
    # Compute probabilities    
    all_probs = all_probs_noise_settings(d,m,eta,gammas,A_settings,B_settings)
    extra_prob = extra_meas_Bob_settings(d,m,eta,gammas,A_settings[0],extra_B_settings)    
    # Compute key rate
    key = compute_key_rate(SDP,cons,A,B,Z,T,W,all_probs,extra_prob,KEEP_M = 0)
    # Before saving the results we convert to the good units
    key_rate.append([eta,key])
    
  
    # --> Optimal case
    settings = opt_params[eta].flatten()
    A_settings = settings[0:m]
    B_settings = settings[(m+1):]
    extra_B_settings = settings[m]
    # Compute probabilities    
    all_probs = all_probs_noise_settings(d,m,eta,gammas,A_settings,B_settings)
    extra_prob = extra_meas_Bob_settings(d,m,eta,gammas,A_settings[0],extra_B_settings)    
    # Compute key rate
    key = compute_key_rate(SDP,cons,A,B,Z,T,W,all_probs,extra_prob,KEEP_M = 0)
    # Before saving the results we convert to the good units
    key_rate_opt.append([eta,key])

    
    # --> Optimal case v2
    #settings = opt_params_v2[eta].flatten()
    #A_settings = settings[0:m]
    #B_settings = settings[(m+1):]
    #extra_B_settings = settings[m]
    # Compute probabilities    
    #all_probs = all_probs_noise_settings(d,m,eta,[1,1],A_settings,B_settings)
    #extra_prob = extra_meas_Bob_settings(d,m,eta,[1,1],A_settings[0],extra_B_settings)    
    # Compute key rate
    #key = compute_key_rate(SDP,cons,A,B,Z,T,W,all_probs,extra_prob,KEEP_M = 0)
    # Before saving the results we convert to the good units
    #key_rate_opt_v2.append([eta,key])
    print("Status SDP:",SDP.status)
    
# Add the plots
key_rate = np.array(key_rate)
key_rate_opt = np.array(key_rate_opt)
#key_rate_opt_v2 = np.array(key_rate_opt_v2)
fig, ax = plt.subplots()
ax.plot(key_rate[:,0],key_rate[:,1], marker = "o",ls = "",label="Alexia's params")
ax.plot(key_rate_opt[:,0],key_rate_opt[:,1], marker = "s",ls = "", label = "Optimized params")
#ax.plot(etas_opt,keys_opt, marker = "o", ls = "", label = "Optimized params")
#ax.plot(key_rate_opt_v2[:,0],key_rate_opt_v2[:,1], marker = "s",ls = "", label = "Optimized params v2")
ax.axhline(0,ls = "--", c = "grey",alpha = 0.4)
ax.set_xlabel("White noise, $(1-\eta)$",fontsize = 30)
ax.set_ylabel("Key rate (bits)", fontsize = 30)
ax.tick_params(labelsize = 30)
ax.legend(loc = "best",fontsize = 20)
#ax.plot(etas_opt,keys_opt,marker = "s",ls = "")

# Save the data
os.chdir("/Users/javier/Documents/PhD/Projects/DIKQD/Big dimensions/Results/Optimized_d2")
np.savetxt("Key_rate_fixed_d2m2.txt",key_rate)
np.savetxt("Key_rate_opt_d2m2.txt",key_rate_opt)






