function [probs] = extra_prob_noise(d,m,eta,settingsA,extrasettingsB,gammas,state)
% Here we are going to compute the noisy probability distribution that we
% will use afterwards for computing the relative entropy between Alice and
% Bob (in general is the same but the output is different)

% Define the (normalized) quantum state
if length(state) == 0
    state = 0;
    norm = sum(gammas.^2);
    for i = 1:1:d
        ket = zeros(d,1);
        ket(i) = 1;
        state = state + (gammas(i)/sqrt(norm))*kron(ket,ket);
    end
end

% Normalize the state just in case
norm = sum(state.^2);
state = state/sqrt(norm);

% Define the bra state
brastate = ctranspose(state);

% Compute the probability for each of the possible outcomes
probs = zeros(d,d);
for a = 0:(d-1)
    for b = 0:(d-1)
        % Obtain the quantum operators
        [A_op,B_op,operator] = AB_proj_operators_noise(d,m,a+1,b+1,1,1,settingsA,extrasettingsB,eta);  
        % Compute the mean value
        mel = brastate*(operator*state);
        probs(a+1,b+1) = mel;
    end
end
end