function [jac] = jacobian_GR(N)
% Jacobian matrix for the Gauss-Radau method

% This function is a Matlab version of the functions Eva prepared

n = N-1;
a=-1;
jac = jacobian_m(n+1);
red_jac = jac(1:n,1:n);
mat = red_jac - a*eye(n);
constant = zeros(n,1);
constant(n) = jac(n,n+1).^2;
res = linsolve(mat,constant);
w_extra = res(n)+a;
jac(n+1,n+1) = w_extra;
end
