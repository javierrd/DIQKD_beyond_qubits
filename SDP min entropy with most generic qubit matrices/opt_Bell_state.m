function [maxState] = opt_Bell_state(d,m,Bell_coeffs,allsettings,eta)
% Finds the the state that leads to the maximal score of a Bell operator

% Define Bell Operator
Bell_op = Bell_ineq(d,m,Bell_coeffs,allsettings,eta);
% Compute eigenvalues and eigenvectors
[V,D] = eig(Bell_op);

% Look for maximum score
col = find(ismember(diag(D), min(diag(D))));
% Obtain the state
maxState = V(:,col);
end
