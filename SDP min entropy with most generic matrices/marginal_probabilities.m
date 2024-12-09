function [P_Alice,P_Bob] = marginal_probabilities(m,all_probs)
% Given a probability distribution, obtains the corresponding marginals

% Define sort of dictionary to save the probabilities
P_Alice = containers.Map;
P_Bob = containers.Map;

% Go through all the values of the measurement settings and compute the
% marginal probabilities
for i=1:1:m
    P_Alice(string(i)) = sum(all_probs(string(i)+string(i)),2);
    P_Bob(string(i)) = sum(all_probs(string(i)+string(i)),1);
end
end

