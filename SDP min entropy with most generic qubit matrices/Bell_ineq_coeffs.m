function [coeffs] = Bell_ineq_coeffs(d,m,cons)
% Define dictionary
coeffs = containers.Map;

%% Compute Bell coefficients
% Joint probability coefficients
i=3;
for x = 1:1:m
    for y=1:1:m
        for a=1:1:(d-1)
            for b=1:1:(d-1)
                coeffs("Joint"+string(x)+string(y)+string(a)+string(b)) = dual(cons(i));
                i=i+1;
            end
        end
    end
end

% Marginal probability coefficients
for x = 1:1:m
    for a=1:1:(d-1)
        coeffs("Alice"+string(x)+string(a)) = dual(cons(i));
        i = i+1;
        coeffs("Bob"+string(x)+string(a)) = dual(cons(i));
        i = i+1;
    end
end

end