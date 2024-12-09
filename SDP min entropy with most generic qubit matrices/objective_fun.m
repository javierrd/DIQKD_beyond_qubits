function [obj] = objective_fun(d,Gammasubs,ObjPositive,ObjNegative)
% Defines Peter's Brown objective function

obj = -1.0;

% Introduce the "positive" elements of the objective
for i=1:1:length(ObjPositive)
    obj = obj + (-1)*Gammasubs(ObjPositive(i,1),ObjPositive(i,2));
end

% Introduce the "negative" elements of the objective 
for i=1:1:length(ObjNegative)
    obj = obj + (1)*Gammasubs(ObjNegative(i,1),ObjNegative(i,2));
end
end




