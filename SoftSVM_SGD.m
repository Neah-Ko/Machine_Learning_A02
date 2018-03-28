function [w, his, bis] = SoftSVM_SGD(D, T, lambda, n)
    % Implements Stochastic Gradient Descent for Support Vector Machine
    % optimisation
    %
    % Inputs
    % D         - Input dataset
    %               - each row represents a feature/label tuple
    %               - feature is a vector composed by the
    %                 e-1 first elements, e being horizontal size
    %               - label is the last (eth) element
    % T         - Number of iterations
    % lambda    - regularization parameter
    % n         - number of last elements we sum up to approximate minimum
    %
    % Output
    % w         - Weight vector
    
    [N,Y] = size(D); % get dimensions
    thetai = zeros(1, Y-1); % build variables
    ws = zeros(T, Y-1);
    his = zeros(T, 1); % hinge losses vector
    bis = zeros(T, 1); % binary losses vector
    
    for i = 1:T
        wi = (1 / (lambda * i)) * thetai; % compute wi
        ws(i, :) = wi; % add to list
        his(i, 1) = loss('hinge', wi, D); % Keep track of hinge losses
        bis(i, 1) = loss('binary', wi, D); % similarly for binary
        j = round(random('unif', 1, N)); % random from 1 to N uniform distribution
        
        xj = D(j, 1:Y-1); % feature
        tj = D(j, Y); % label
        
        if((tj * dot(wi, xj)) < 1)
            thetai = thetai + tj * xj; % update theta
        end
    end
    
    w = 1/n * sum(ws((T-1):T,:)); % sum last n iterations to get minimum
end