function L = loss(name, w, D, v)
    % Compute loss for a weight vector w over a dataset D
    %
    % Inputs
    % name      - String, name of the loss to use
    %               - Implemented : 'binary' and 'hinge'
    % D         - Input dataset
    %               - each row represents a feature/label tuple
    %               - feature is a vector composed by the 
    %                 e-1 first elements, e being horizontal size
    %               - label is the last (eth) element
    % w         - weight vector
    % v         - value of (+) label if not 1, ~v will then be (-)
    %
    % Output
    % l         - normalized loss
    if(~exist('v','var')); v = 1; end % optional arg
    [N,Y] = size(D);
    L = 0;
    for i = 1:N
        xi = D(i, 1:Y-1); % feature
        ti = D(i, Y); % label
        if(ti ~= v); ti = -1; else; ti = 1; end % relabelisation
        if(strcmpi(name ,'binary'))
            L = L + (1 * (sign(dot(w, xi)) ~= ti)); % pointwise binary loss
        elseif(strcmpi(name, 'hinge'))
            L = L + max(0, 1 - (ti * dot(w, xi))); % pointwise hinge loss
        end
    end
    L = L/N; % normalize
end