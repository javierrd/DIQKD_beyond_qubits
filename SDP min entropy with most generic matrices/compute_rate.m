function [r] = compute_rate(d,m,ent,eta,settings,gammas,state)

%% Relative entropy between Alice and Bob
% Define probability distribution for Alice and Bob
extraprob = extra_prob_noise(d,m,eta,settings(1),settings(m+1),gammas,state);
% Compute the relative entropy between Alice and Bob
HAB = relative_entropy(extraprob);

% Compute rate
r = ent-HAB;
end