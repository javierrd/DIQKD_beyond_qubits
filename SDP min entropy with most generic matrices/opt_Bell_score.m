function [maxScore] = opt_Bell_score(d,m,Bell_coeffs,allsettings,eta)
% Finds the maximum score of a given Bell operator

% Define Bell Operator
Bell_op = Bell_ineq(d,m,Bell_coeffs,allsettings,eta);
% Compute eigenvalues and eigenvectors
[V,D] = eig(Bell_op);
% Look for maximum score
maxScore = real(min(diag(D)));
end
