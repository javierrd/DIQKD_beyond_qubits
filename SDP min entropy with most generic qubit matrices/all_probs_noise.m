function [all_probs] = all_probs_noise(d,m,eta,allsettings,gammas,state)
% This function provides the ideal probabilities the case of having d
% outcomes and m dimensions

% We have to organize the thetas and the phis
[settingsA,settingsB] = order_parameters(m,allsettings);

% Define sort of fictionary for saving the data
all_probs = containers.Map;

% Go through all possible measurement settings and compute the
% corresponding probability
for x=1:1:m
    for y=1:1:m
        all_probs(string(x)+string(y)) = single_prob_for_settings(d,m,x,y,gammas,state,settingsA(string(x)),settingsB(string(y)),eta);
    end
end
end
