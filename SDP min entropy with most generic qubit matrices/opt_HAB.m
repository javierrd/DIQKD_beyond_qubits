function [optBobsettings] = opt_HAB(d,m,eta,settingsA,extrasettingsB,gammas,state)
%% Running the optimization
% Define function for the optimization
fun = @(v)compute_HAB(d,m,eta,settingsA,v,gammas,state);
% Definie initial points
N=200;
init_points = generate_random(N,extrasettingsB);
int_x = zeros(N,length(extrasettingsB));
int_fval = zeros(N,1);

% Set some options of the optimizer
options = optimoptions('fminunc', ...
                       'Display', 'off', ...
                       'MaxIter', 1000, ...
                       'TolFun', 1e-6, ...
                       'TolX', 1e-6);
% Run the optimization problem
parfor i=1:1:N
    x0 = init_points(i,:);
    [x,fval] = fminunc(fun,x0,options);
    int_x(i,:) = x;
    int_fval(i) = fval;
end
% Search the optimal values
loc = find(int_fval==min(int_fval));
optBobsettings = int_x(loc,:);
end