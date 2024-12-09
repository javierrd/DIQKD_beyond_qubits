function [real_score] = Bell_ineq(d,m,Bell_coeffs,allsettings,eta)
% Computes the Bell inequality given the coefficients obtained from the
% dual of the SDP

% Order the parameters
[settingsA,settingsB] = order_parameters(m,allsettings);

% Finally, compute the Bell inequality
score = 0.0;
% --> Joint probability terms
for x = 1:1:m
    for y=1:1:m
        for a=1:1:(d-1)
            for b=1:1:(d-1)
                [A_op,B_op,operators] = AB_proj_operators_noise(d,m,a,b,x,y,settingsA(string(x)),settingsB(string(y)),eta);
                score = score + Bell_coeffs("Joint"+string(x)+string(y)+string(a)+string(b))*operators;
            end
        end
    end
end
% --> Marginal probability terms
for x = 1:1:m
    for a=1:1:(d-1)
        [A_op,useless,useless2] = AB_proj_operators_noise(d,m,a,1,x,1,settingsA(string(x)),settingsB(string(1)),eta);
        [useless,B_op,useless2] = AB_proj_operators_noise(d,m,1,a,1,x,settingsA(string(1)),settingsB(string(x)),eta);
        score = score + Bell_coeffs("Alice"+string(x)+string(a))*kron(A_op,eye(d));
        score = score + Bell_coeffs("Bob"+string(x)+string(a))*kron(eye(d),B_op);
    end
end
real_score = real(score);
end