%% Init
D = load('bg.txt'); % load initial dataset
[X, ~] = size(D); % get x dimension
D = [D, ones(X, 1)]; % add bias
l = ones(X, 1); l(101:200, 1) = -1;
D = [D, l]; % add label dimension
[X, Y] = size(D); % recomputing dimensions
DS = load('seeds_dataset.txt'); % 'seeds' dataset
[P, Q] = size(DS);
l = DS(:, Q);
DS(:, Q) = ones(P, 1); % bias on last index
DS = [DS, l]; % replacing labels
[P, Q] = size(DS); % dims
%% Q3.b
lambda = [100, 10, 1, .1, .01, .001];
wx = zeros(length(lambda),Y-1);
figure;
for i = 1:length(lambda)
    [w, his, bis] = SoftSVM_SGD(D, 200, lambda(i), 1);
    subplot(floor(length(lambda)/3),3,i);  % plotting
    set(gca, 'YScale', 'log');
    title('Losses over T ');
    semilogy(his, 'y', 'color', 'red'); % logarithmic scale on y axis
    hold on;
    semilogy(bis, 'y', 'color', 'blue');
    plot([0,0],'visible', 'off');
    legend('hinge loss','binary loss', ['lambda = ' num2str(lambda(i))]);
    ylim([0, max(max(his),max(bis))]);
    hold off; grid on; xlabel('i = 1..T'); ylabel('Loss of w^i');
end

%% Q3.e
bp = [0,70,140,210]; % breakpoints
ix = [2, 3, 4]; % sections represented by indexes of bp
lbin = zeros(3, 1); % binary losses
ws = zeros(3, Q-1); % weight vectors
min = 1; wsmin = ws; lbinmin = lbin;
for k = 1:50 % minimize multiclass
    for i = 1:length(ix)
       hi = bp(ix(i)); % area of interest
       lo = bp(ix(i) - 1) + 1;
       for j = 1:P % separate +1 and -1 labels to call our previous function
            if(j > hi || j < lo); DS(j, Q) = -1;
            else; DS(j, Q) = 1; end
       end
       [w, ~, ~] = SoftSVM_SGD(DS, 210, 0.001, 1);
       lbin(i) = loss('binary', w, DS);
       ws(i, :) = w';
    end
    DS(1:70, Q) = 1; DS(71:140, Q) = 2; DS(141:210, Q) = 3; % restore original labels
    DRC = DS(:,Q); % comparison vector
    for i = 1:P
        ti = DS(i,Q); % label
        xi = DS(i,1:Q-1); % feature  
        [~, I] = max([dot(ws(1,:),xi), dot(ws(2,:),xi), dot(ws(3,:),xi)]);
        DRC(i, 1) = I; % reclassify
    end
    DRC = DS(:,Q) ~= DRC(:,1); % logical difference
    mc = sum(double(DRC)) / P; % normalized sum
    if(mc < min)
        min = mc; wsmin = ws; lbinmin = lbin;
    end
end; min % display min