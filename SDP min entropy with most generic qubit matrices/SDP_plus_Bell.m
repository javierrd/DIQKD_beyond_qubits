% This code performs the back and forth between the SDP and Bell inequality

%% Define dimensions of our system and settings
d = 2;
m = 2;
%allsettings = 2*pi*rand(1,5*2*m);

% Introducing Alexia's settings
allsettings = [];
th0 = pi/4;
for x=1:m
    allsettings = [allsettings,[th0,(x-0.5)/m]];
end
for y=1:m
    allsettings = [allsettings,[th0,-y/m]];
end

% Define initial state
gammas = ones(1,d);
state = [];
%% Compute min entropy
npoints = 40;
etalist = linspace(0.5,1,npoints);
etalist = flip(etalist.^(1/2),2);
h = waitbar(0,'Processing...');
opt_entropy = zeros(1,npoints);
opt_rate = zeros(1,npoints);
opt_settings = zeros(npoints,length(allsettings));
opt_settingsB = zeros(npoints,length(allsettings)/(2*m));

for i=1:length(etalist)
    eta = etalist(i);
    waitbar(i/length(etalist),h,sprintf('Processing eta = %d...',etalist(i)));
    diff = 10;
    [new_ent,Gammasubs,cons,Pjoint,PAlice,PBob] = compute_entropy(d,m,eta,allsettings,gammas,state);
    while diff > 1e-4
        ent = new_ent;
        % Define Bell coefficients
        Bell_coeffs = Bell_ineq_coeffs(d,m,cons);
        % Optimize the obtained Bell inequality
        [allsettings,state] = opt_Bell(d,m,Bell_coeffs,allsettings,eta);
        % Compute new entropy
        [new_ent,Gammasubs,cons,Pjoint,PAlice,PBob] = compute_entropy(d,m,eta,allsettings,[],state);
        disp("--------------------------")
        disp("Old entropy "+string(ent))
        disp("New entropy "+string(new_ent))
        diff = new_ent - ent;
    end
    % Optimize H(A|B)
    extrasettingsBob = allsettings(1:length(allsettings)/(2*m));
    optextraBob = opt_HAB(d,m,eta,allsettings(1:8),extrasettingsBob,gammas,state);
    HAB = real(compute_HAB(d,m,eta,allsettings(1:8),optextraBob(1,:),[],state));
    disp("H(A|B) = "+string(HAB))
    opt_entropy(i) = new_ent;
    opt_settings(i,:) = allsettings(1,:);
    opt_rate(i) = real(new_ent - HAB);
    opt_settingsB(i,:) = optextraBob(1,:);
    saveBob = optextraBob(1,:);
    % Save the data
    cd Results/d2m2_random
    filename = sprintf('data_%d.mat',eta);
    save(filename,'new_ent','allsettings','saveBob','HAB','state');
    cd ..
    cd ..
end
