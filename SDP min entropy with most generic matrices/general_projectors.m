function [proj] = general_projectors(d,settings)
% Generates general projectors
if d == 3
    % Split settings
    ph0 = settings(1);
    th0 = settings(2);
    ph1 = settings(3);
    th1 = settings(4);
    al0 = settings(5);
    be0 = settings(6);
    al1 = settings(7);
    be1 = settings(8);
    
    %% First projector
    % Define the eigenstate
    a1 = [cos(ph0).*sin(th0);exp(1).^((sqrt(-1)*(-2/3)).*al0.*pi).*sin(ph0) ...
  .*sin(th0);exp(1).^((sqrt(-1)*(-4/3)).*be0.*pi).*cos(th0)];
    norm = sum(abs(a1).^2);
    a1 = a1/sqrt(norm);
    
    % Compute the projector
    A1 = a1*ctranspose(a1);

    %% Second projector
    % Define the eigenstate
    a2 = [cos(ph1).*(cos(th0).^2+sin(ph0).^2.*sin(th0).^2).*sin(th1)+(-1).* ...
  cos(ph0).*sin(th0).*(exp(1).^((sqrt(-1)*(4/3)).*(1+be0+(-1).*be1) ...
  .*pi).*cos(th0).*cos(th1)+exp(1).^((sqrt(-1)*(2/3)).*(1+al0+(-1).* ...
  al1).*pi).*sin(ph0).*sin(ph1).*sin(th0).*sin(th1)); ...
  exp(1).^((sqrt(-1)*(-2/3)).*al0.*pi).*((-1).^(1/3).*exp(1).^(( ...
sqrt(-1)*(4/3)).*(be0+(-1).*be1).*pi).*cos(th0).*cos(th1).*sin( ...
ph0).*sin(th0)+(-1).^(2/3).*exp(1).^((sqrt(-1)*(2/3)).*(al0+(-1).* ...
  al1).*pi).*cos(th0).^2.*sin(ph1).*sin(th1)+cos(ph0).*((-1).*cos( ...
  ph1).*sin(ph0)+(-1).^(2/3).*exp(1).^((sqrt(-1)*(2/3)).*(al0+(-1).* ...
  al1).*pi).*cos(ph0).*sin(ph1)).*sin(th0).^2.*sin(th1));
  (-1).*exp(1).^((sqrt(-1)*(-4/3)).*be0.*pi).*sin(th0).*((-1).^(1/3) ...
  .*exp(1).^((sqrt(-1)*(4/3)).*(be0+(-1).*be1).*pi).*cos(th1).*sin( ...
  th0)+cos(th0).*(cos(ph0).*cos(ph1)+(-1).^(2/3).*exp(1).^((sqrt(-1) ...
  *(2/3)).*(al0+(-1).*al1).*pi).*sin(ph0).*sin(ph1)).*sin(th1))];
  norm = sum(abs(a2).^2);
  a2 = a2/sqrt(norm);
    
  % Compute the projector
  A2 = a2*ctranspose(a2);

  % Define the projectors
  proj = {A1,A2,eye(d)-A1-A2};
end