function [A_op_noisy,B_op_noisy,operator] = AB_proj_operators_noise(d,m,a,b,x,y,settingsA,settingsB,eta)
% This function defines the coefficients the projective operators that
% define Alice's and Bob's measurements

% Compute projectors
projA = general_projectors(d,settingsA);
projB = general_projectors(d,settingsB);
% Add the noise
A_op_noisy = eta*projA{a} + (1-eta)*eye(d)/d;
B_op_noisy = eta*projB{b} + (1-eta)*eye(d)/d;
% Compute the total projector
operator = kron(A_op_noisy,B_op_noisy);
end