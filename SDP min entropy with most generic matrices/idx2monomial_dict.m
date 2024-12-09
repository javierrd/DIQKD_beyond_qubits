function [dict] = idx2monomial_dict(idx2monomial)
% Turns the idx2monomial cell into a dictionary

% Define the values and the keys
values_cell = idx2monomial(:,1);
keys_cell = idx2monomial(:,2);

% Define lists for saving the keys and values
values = zeros(1,length(values_cell));
keys = strings(1,length(keys_cell));

% Save the data
for idex=1:1:length(keys)
    values(idex)=cell2mat(values_cell(idex));
    keys(idex)=cell2mat(keys_cell(idex));
end

% Define a dictionary with all these things
dict = containers.Map(keys,values);
end