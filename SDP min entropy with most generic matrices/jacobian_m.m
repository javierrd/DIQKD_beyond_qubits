function [J] = jacobian_m(n)
% Jacobian matrix for Gauss quadrature (M=0)

% This function is a Matlab version of the functions Eva prepared

betaux = betafun(n);
J = diag(betaux,1)+diag(betaux,-1);
end
