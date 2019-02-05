clc;
clear;
close all;

outpath = 'C:\Users\TakuyaIwase\Documents\MATLAB\bat_algorithm\SSCI\NSBA\';

for fnc = 1:4
        if exist([path 'F1'],'dir') == 0
            mkdir([outpath 'F1']);
        end
        if exist([path 'F2'],'dir') == 0
            mkdir([outpath 'F2']);
        end
        if exist([path 'F3'],'dir') == 0
            mkdir([outpath 'F3']);
        end
        if exist([path 'F4'],'dir') == 0
            mkdir([outpath 'F4']);
        end
end

NP = 50;
maxRun = 30;
seed = 0:maxRun-1;
fprintf('---------------------------------------------------------------\n');

for fnc = 1:4
    D = 2;
    MaxFes = 10000;
    x_lb = getLb(fnc);
    x_ub = getUb(fnc);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
%     xx1 = -5:0.1:5;
%     xx2 = -5:0.1:5;
%     for j = 1:length(xx1)
%         for k = 1:length(xx2)
%             y(:,k) = funcFit([xx1(j) xx2(k)],4);
%         end
%         fit(:,j) = y;
%     end
%     
%     figure;
%     contour(xx1,xx2,fit,'Fill','on');
%     xlabel('x1');
%     ylabel('x2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    for run = 1:maxRun
        alpha = 0.9;    % damping coef.
        gamma = 0.9;    % damping coef.
        A = ones(NP,1); % Loudness  (constant or decreasing) 
        r = ones(NP,1); % Pulse rate (constant or decreasing)
        % This frequency range determines the scalings
        fmin = 0; % Frequency minimum
        fmax = 1; % Frequency maximum
        f = zeros(NP,1);   % Frequency
        v = zeros(NP,D);
        s = 0;
        pop = zeros(NP,D);
        Srnd = zeros(NP,D);
        fitness = zeros(NP,1);
        % Initialize Population
        for i = 1:NP
            s = i + seed(run);
            rng(s);
            pop(i,:) = [x_lb(1) + (x_ub(1) - x_lb(1)) * rand, x_lb(2) + (x_ub(2) - x_lb(2)) * rand];
            fitness(i,:) = funcFit(pop(i,:),fnc);
            r(i) = r(i) * unifrnd(0,1);
            f(i) = fmin + (fmax - fmin) * rand;
        end
        pbest = pop;
        pcost = fitness;
        [gcost, idx] = min(pcost);
        gbest = pop(idx,:);
        
        for iter = 1:MaxFes
            % Calculate Dynamic Niche
            eq = zeros(NP,D);
            d = zeros(NP,D);
            for i = 1:NP
                dij = zeros(NP,D);
                dist = zeros(NP,1);
                for j = 1:NP
                    if i == j
                        continue;
                    end
                    dij(j,:) = pbest(i,:) - pop(j,:);
                    dist(j,:) = norm(dij(j,:));
                    eq(j,:) = dij(j,:)/(dist(j,:)^2);
                end
                d(i,:) = sum(eq)/NP;
            end
            % popution movement
            for i = 1:NP
                v(i,:) = v(i,:) + d(i,:) * f(i,:);
                pop(i,:) = pop(i,:) + v(i,:);
                % Apply simple bounds/limits
                pop(i,:) = Bounds(pop(i,:), fnc);
            end

            % Local search
            for i = 1:NP 
                Sloc = NaN(NP,D);
                if rand > r(i)      
                    Sloc(i,:) = pbest(i,:) + A(i) * unifrnd(0,1,[1,D]);
                    Sloc(i,:) = Bounds(Sloc(i,:), fnc);
                end
            end
            % random search
%             for i = 1:NP                      
%                 Srnd(i,:) = pop(i,:) + unifrnd(-m(i,:),m(i,:),[1,D]); 
%                 Srnd(i,:) = max(Srnd(i,:),x_lb);
%                 Srnd(i,:) = min(Srnd(i,:),x_ub);
%             end
            
            for i = 1:NP
                if  rand < A(i) & fitness(i,:) < pcost(i,:)
                    A(i) = alpha * A(i);
                    r(i) = r(i) * norm(1-exp(-gamma*iter));
                    pbest(i,:) = pop(i,:);
                    pcost(i,:) = fitness(i,:);
                elseif rand < A(i) & funcFit(Sloc(i,:),fnc) < pcost(i,:) % & Sloc(i,:)~=[0 0]
                    A(i) = alpha * A(i);
                    r(i) = r(i) * norm(1-exp(-gamma*iter));
                    pbest(i,:) = Sloc(i,:);
                    pcost(i,:) = funcFit(Sloc(i,:),fnc);
%                 elseif rand < A(i) & funcFit(Srnd(i,:),fnc) > pcost(i,:)% & Darray(2,1) < min(rdist(i))
%                     A(i) = alpha * A(i);
%                     r(i) = r(i) * norm(1-exp(-gamma*iter));
%                     pbest(i,:) = Srnd(i,:);
%                     pcost(i,:) = funcFit(Srnd(i,:),fnc);
                end     

                if pcost(i) <= gcost
                    gbest = pbest(i,:);
                    gcost = pcost(i);
                end
            end
            pop = pbest;
            fprintf(['seed: (' num2str(run) '/' num2str(maxRun) ')    The number of Evaluations: (' num2str(iter) '/' num2str(MaxFes) ') \n']);
            fprintf(['gbest = ' num2str(gbest) '\n']);
            fprintf(['gcost = ' num2str(gcost) '\n']); 
        end
        outData = horzcat(pop, fitness);
        csvwrite([outpath 'F' num2str(fnc) '\F' num2str(fnc) '_' num2str(run) '_pbest.csv'], outData);
    end
end

function bounds = Bounds(x, fnc)
ub = getUb(fnc);
lb = getLb(fnc);
if x(1) > ub(1)
    x(1) = ub(1);
elseif x(1) < lb(1)
    x(1) = lb(1);
end
if x(2) > ub(2)
    x(2) = ub(2);
elseif x(2) < lb(2)
    x(2) = lb(2);
end
bounds = [x(1) x(2)];
end

function lb = getLb(fnc)
Lb_arr = [-10 -10; -2 -1; 0.25 0.25; -5 -5];
lb = Lb_arr(fnc,:);
end

function ub = getUb(fnc)
Ub_arr = [10 10; 2 1; 4 4; 5 5];
ub = Ub_arr(fnc,:);
end

function fit = funcFit(x, fnc)
persistent fname;
if fnc == 1
    fname = str2func('Griewank');
elseif fnc == 2
    fname = str2func('SixHump');
elseif fnc == 3
    fname = str2func('Michalewicz');
elseif fnc == 4
    fname = str2func('Himmelblau');
end
fit = feval(fname, x);
end

function fit = Griewank(x)
fit = (x(1)^2 + x(2)^2) / 4000 - (cos(x(1)) * cos(x(2) / sqrt(2))) + 1;
end

function fit = SixHump(x)
fit = (4 - 2.1 * x(1)^2 + (x(1)^4) / 3) * x(1)^2 + x(1) * x(2) + (-4 + 4 * x(2)^2) * x(2)^2;
end

function fit = Michalewicz(x)
m = 10;
fit = - (sin(x(1)) * sin(x(1)^2/pi)^(2 * m) + sin(x(2)) * sin((2*(x(2))^2)/pi)^(2 * m));
end

function fit = Himmelblau(x)
fit = (x(1)^2 + x(2) - 11)^2 + (x(1) + x(2)^2 - 7)^2;
end
