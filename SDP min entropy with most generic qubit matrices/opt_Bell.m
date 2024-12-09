function [optsettings,optstate] = opt_Bell(d,m,Bell_coeffs,allsettings,eta)
%% Optimization boundaries and points
% Define lower and uper boundaries for the optimization
lb = zeros(1,length(allsettings));
ub = ones(1,length(allsettings))*2*pi;

%% Running the optimization for the thetas
% Define function for the optimization
fun = @(v)opt_Bell_score(d,m,Bell_coeffs,v,eta);
%rng default % For reproducibility
% Run the optimization problem
opts = optimoptions(@fmincon,'Algorithm','sqp');
problem = createOptimProblem('fmincon','objective',...
        fun,'x0',allsettings,'lb',lb,'ub',ub,'options',opts);
ms = MultiStart('Display','final');
%gs = GlobalSearch;
[x,fval] = run(ms,problem,50);
%[x,fval] = run(gs,problem);
optsettings = x;

% Define optimal state
optstate = opt_Bell_state(d,m,Bell_coeffs,optsettings,eta);

end
