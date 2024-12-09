function [t_GW,w_GW] = GWel(n)
% Calculate the nodes and weights via the GW algorithm

% This function is a Matlab version of the functions Eva prepared

interval = [-1, 0];
if n>1
    J = jacobian_GR(n);
    [v_GW,t_GW] = eig(J);
    t_GW = diag(t_GW); % These are the nodes

    % Compute the weights
    v_0 = v_GW(1,:);
    w_GW = zeros(1,n);
    for idex=1:1:n
        w_GW(idex) = 2*v_0(idex)^2;
    end
    % Correct for the input interval
    dab = diff([-1,0]);
    t_GW = (t_GW+1)/2*dab + interval(1);
    t_GW = transpose(-flip(t_GW));
    w_GW = flip(dab*w_GW/2);

else
    t_GW = 1;
    w_GW = 1;
end
end
