function [optsettings,optstate] = opt_Bell_fminunc(d,m,Bell_coeffs,allsettings,eta)
%% Running the optimization for the thetas
% Define function for the optimization
fun = @(v)opt_Bell_score(d,m,Bell_coeffs,v,eta);
% Definie initial points
N=200;
init_points = generate_random(N,allsettings);
int_x = zeros(N,length(allsettings));
int_fval = zeros(N,1);

% Set some options of the optimizer
options = optimoptions('fminunc', ...
                       'Display', 'off', ...
                       'MaxIter', 1000, ...
                       'TolFun', 1e-6, ...
                       'TolX', 1e-6);
parfor i=1:1:N
    x0 = init_points(i,:);
    [x,fval] = fminunc(fun,x0,options);
    int_x(i,:) = x;
    int_fval(i) = fval;
end
% Search the optimal values
loc = find(int_fval==min(int_fval));
optsettings = int_x(loc,:);

% Define optimal state
optstate = opt_Bell_state(d,m,Bell_coeffs,optsettings,eta);

end
