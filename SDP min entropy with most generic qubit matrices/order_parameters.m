function [settingsA,settingsB] = order_parameters(m,allsettings)
% Define dictionary
settingsA = containers.Map;
settingsB = containers.Map;

% Define length of the parameters
lengthparams = 2;
% Order the parameters
for i=0:m-1
    settingsA(string(i+1)) = allsettings((i*lengthparams+1):(i+1)*lengthparams);
    settingsB(string(i+1)) = allsettings((2+i)*lengthparams+1:((2+i+1)*lengthparams));
end
end