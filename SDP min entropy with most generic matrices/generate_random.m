function [M] = generate_random(N,allsettings)
% Generate an N times the length of all settings matrix of random numbers
% between 0 and 2*pi
M = 2*pi*rand(N-1, length(allsettings));
M = [allsettings;M];
end