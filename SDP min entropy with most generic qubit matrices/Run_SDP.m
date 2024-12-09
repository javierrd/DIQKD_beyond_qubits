% This function performs an SDP optimization

%% Define dimensions of our system and settings
d = 2;
m = 2;

% Initial measurement settings
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
gammas = [1,1];
state = [];

%% Compute min entropy
%etalist = linspace(sqrt(0.85),1.,50);
etalist = [1];
h = waitbar(0,'Processing...');
old_entropy = [];
old_rate = [];
for i=1:length(etalist)
    waitbar(i/length(etalist),h,sprintf('Processing eta = %d...',etalist(i)));
    [ent,Gammasubs,cons,Pjoint,PAlice,PBob] = compute_entropy(d,m,etalist(i),allsettings,gammas,state);
    old_entropy = [old_entropy,ent];
    %old_rate = [old_rate,compute_rate(d,m,ent,etalist(i),allthetas,allphis,gammas,state)];
end
close(h)
ent
