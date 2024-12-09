function [cons_dic] = prob_cons_dict(d,m,Gammasubs,eta,settings,gammas,Pjoint,PAlice,PBob)
%% Preliminaries
% Define dictionary for saving data
cons_dic = containers.Map;

% Compute the probability distributions
alphas = settings(1:m);
betas = settings(m+2:2*(m)+1);
all_probs = all_probs_noise(d,m,eta,alphas,betas,gammas);
%% Obtain and order the probabilities
% Obtain marginal probabilities
[PmargAlice,PmargBob] = marginal_probabilities(m,all_probs);

% Order the joint probabilities as required by the substitutions
Pjointorder = [];
for x = 1:1:m
    for y=1:1:m
        for a=1:1:(d-1)
            for b=1:1:(d-1)
                prob = all_probs(string(x)+string(y));
                Pjointorder = [Pjointorder, prob(a,b)];
            end
        end
    end
end

% Order the marginal probabilities as required by the substitutions
PAliceorder = [];
PBoborder = [];
for x = 1:1:m
    for a=1:1:(d-1)
        probAlice = PmargAlice(string(x));
        probBob = PmargBob(string(x));
        PAliceorder = [PAliceorder,probAlice(a)];
        PBoborder = [PBoborder,probBob(a)];
    end
end

%% Introduce the constraints
% Joint probability constraints
i=1;
for x = 1:1:m
    for y=1:1:m
        for a=1:1:(d-1)
            for b=1:1:(d-1)
                cons_dic(string(x)+string(y)+string(a)+string(b)) = [Gammasubs(Pjoint(i,1),Pjoint(i,2))==Pjointorder(i)];
                i=i+1;
            end
        end
    end
end

% Marginal probability constraints
i=1
for x = 1:1:m
    for a=1:1:(d-1)
        cons_dic("Alice"+string(x)+string(a)) = [Gammasubs(PAlice(i,1),PAlice(i,2)) == PAliceorder(i)];
        cons_dic("Bob"+string(x)+string(a)) = [Gammasubs(PBob(i,1),PBob(i,2)) == PBoborder(i)];
        i = i+1
    end
end
end