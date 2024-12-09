%% Define dimensions of our system and settings
d = 3;
m = 2;
%allsettings = 2*pi*rand(1,5*2*m);
allsettings = [];
th0 = acos(1/sqrt(3));
ph0 = pi/4;
for x=1:m
    allsettings = [allsettings,[ph0,th0,ph0,th0,(x-0.5)/m,(x-0.5)/m,(x-0.5)/m,(x-0.5)/m]];
end
for y=1:m
    allsettings = [allsettings,[ph0,th0,ph0,th0,y/m,y/m,y/m,y/m]];
end

% Define the initial state
gammas = [1,1,1];
state = [];


etalist = linspace(0,1,40);
HAB3 = [];
for eta=etalist
    HAB3 = [HAB3,compute_HAB(d,m,eta,allsettings(1:8),allsettings(9:16),gammas,[])];
end
