function [beta] = betafun(n)
% Coefficients of the Jacopi matrix, obtained from the 3-term recurrence
% coefficients

% This function is a Matlab version of the functions Eva prepared

beta = zeros(1,n-1);
for idex=1:1:(n-1)
    beta(1,idex) = 0.5/sqrt(1-(2*(idex))^(-2));
end
end