function [Gammasubs,cons,ObjPositive,ObjNegative,Pjoint,PAlice,PBob] = load_SDP(d,m,eta,allsettings,gammas,state)
%% Load different stuff
% Set the dimensionality of the problem (LOOK CAREFULLY!!!!)
% Go to directory
cd d2_m2

% Run all the files in it
NPA_d2_m2_order_2 % Substitutions in SDP
Objective_positive_d2_m2_order_2 % "Positive" terms of objective function
Objective_negative_d2_m2_order_2 % "Negative" terms of objective function
Pjoint_d2_m2_order_2 % Joint probabilities substitutions
PAlice_d2_m2_order_2 % Marginal probabilities of Alice substitutions
PBob_d2_m2_order_2 % Marginal probabilities of Bob substitutions
if d>2
    Zeros_d3_m2_order_2
end
% Return to old directory
cd ..

%% Generate SDP variables and the moment matrix
vars = sdpvar(length(Gamma),length(Gamma),'full');
Gammasubs_full = vars(Gamma);
Gammasubs = (Gammasubs_full + Gammasubs_full')/2; % Turn to hermitian
%% Introduce constraints on the certificate
% Positivity constrain
cons = [Gammasubs >= 0];
% First element is equal to one 
cons = [cons, Gammasubs(1,1) == 1];

% Introduce the probability constraints
all_probs = all_probs_noise(d,m,eta,allsettings,gammas,state);
cons = probability_constraints(d,m,Gammasubs,all_probs,cons,Pjoint,PAlice,PBob);

% If d>2, then the projective measurement constraints give rise to some
% elements that are zero. We set them equal to zero in the matrix
% [d=3 --> 171, ...]

% --> Inefficient way of introducing zeros
%if d>2
%    for i=1:1:length(Zerosloc)
%        cons = [cons,Gammasubs(Zerosloc(i,1),Zerosloc(i,2))==real(0.0)];
%    end
%end

% --> Efficient way of introducing zeros
if d>2
    index1 = transpose(Zerosloc(:,1));
    index2 = transpose(Zerosloc(:,2));
    idx = sub2ind([length(Gamma),length(Gamma)],index1,index2);
    cons = [cons,Gammasubs(idx)==real(0.0)];
end
end