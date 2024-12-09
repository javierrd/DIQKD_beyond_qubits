function [rel_ent] = relative_entropy(probjoint)
% Given a probability distribution, computed the relative entropy

% First, compute the marginal probability
probmarg = sum(probjoint,1);
d = length(probmarg);
% Flatten the input probabilities
probjoint = reshape(probjoint,1,[]);

% Compute the entropies
hjoint = 0.0;
for prob=probjoint
    if prob > 0
        hjoint = hjoint - prob*log2(prob)/log2(d);
    end
end
hmarg = 0.0;
for prob=probmarg
    if prob>0
        hmarg = hmarg - prob*log2(prob)/log2(d);
    end
end

% Compute the relative entropy
rel_ent = hjoint - hmarg;
end


