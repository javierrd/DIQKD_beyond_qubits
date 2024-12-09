function [P_Alice,P_Bob] = marginal_probabilities_v2(d,m,settings,eta,gammas,state)
% Given a probability distribution, obtains the corresponding marginals

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

% Define sort of dictionary to save the probabilities
P_Alice = containers.Map;
P_Bob = containers.Map;

% Go through all the values of the measurement settings and compute the
% marginal probabilities
for i=1:1:m
    pa_int = [];
    pb_int = [];
    for a=1:1:d
    [Aop,Bop,useless] = AB_proj_operators_noise(d,m,a,a,i,i,settings(i),settings(m+1+i),eta);
    pa_int = [pa_int, real(ctranspose(state)*(kron(Aop,eye(d))*state))];
    pb_int = [pb_int, real(ctranspose(state)*(kron(eye(d),Bop)*state))];
    end
    P_Alice(string(i)) = pa_int;
    P_Bob(string(i)) = pb_int;
end
end

