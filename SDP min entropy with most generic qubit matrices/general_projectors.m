function [proj] = general_projectors(d,settings)
% Generates general projectors
if d == 2
    % Split settings
    th0 = settings(1);
    ph0 = settings(2);
    
    %% First projector
    % Define the eigenstate
    a1 = [cos(th0);exp(1j*pi*(0-ph0))*sin(th0)];
    norm = sum(abs(a1).^2);
    a1 = a1/sqrt(norm);
    
    % Compute the projector
    A1 = a1*ctranspose(a1);

  % Define the projectors
  proj = {A1,eye(d)-A1};
end
end