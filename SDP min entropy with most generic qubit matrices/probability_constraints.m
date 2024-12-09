function [cons] = probability_constraints(d,m,Gammasubs,all_probs,cons,Pjoint,PAlice,PBob)
% Add the equality constraints regarding the probability distributionto an
% already existing list of constraints

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
                Pjointorder = [Pjointorder, real(prob(a,b))];
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
        PAliceorder = [PAliceorder,real(probAlice(a))];
        PBoborder = [PBoborder,real(probBob(a))];
    end
end

%% Introduce the constraints
% Joint probability constraints
for i=1:1:length(Pjoint)
    cons = [cons,Gammasubs(Pjoint(i,1),Pjoint(i,2))==Pjointorder(i)];
end

% Marginal probability constraints
for i=1:1:length(PAlice)
    cons = [cons,Gammasubs(PAlice(i,1),PAlice(i,2)) == PAliceorder(i)];
    cons = [cons,Gammasubs(PBob(i,1),PBob(i,2)) == PBoborder(i)];
end
end







