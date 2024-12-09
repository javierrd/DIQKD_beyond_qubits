function [ent,Gammasubs,cons,Pjoint,PAlice,PBob] = compute_entropy(d,m,eta,allsettings,gammas,state)
% Computes the entropy according to Peter Brown's formula

% Establecer la precisión numérica
% Define the SDP
[Gammasubs,cons,ObjPositive,ObjNegative,Pjoint,PAlice,PBob] = load_SDP(d,m,eta,allsettings,gammas,state);
% Define objective function
obj = objective_fun(d,Gammasubs,ObjPositive,ObjNegative);
% Solve the SDP
optimize(cons,obj,sdpsettings('solver','mosek','verbose',1));
% Add the result
ent = -log((-1)*value(obj))/log(d);
end
