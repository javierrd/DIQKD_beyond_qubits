%% Define dimensions of our system and settings
d = 2;
m = 2;
% Introducing Alexia's settings
allsettings = [];
th0 = pi/4;
for x=1:m
    allsettings = [allsettings,[th0,(x-0.5)/m]];
end
for y=1:m
    allsettings = [allsettings,[th0,-y/m]];
end

% Define the initial state
%gammas = [1,1,1];
%state = [];


etalist = linspace(0,1,40);
HAB2 = [];
for eta=etalist
    HAB2 = [HAB2,compute_HAB(d,m,eta,allsettings(1:2),allsettings(3:4),[1,1],[])];
end
