function [HAB] = compute_HAB(d,m,eta,Asett,Bextrasett,gammas,state)
%% Relative entropy between Alice and Bob
% Define probability distribution for Alice and Bob
extraprob = extra_prob_noise(d,m,eta,Asett,Bextrasett,gammas,state);
% Compute the relative entropy between Alice and Bob
HAB = real(relative_entropy(extraprob));
end