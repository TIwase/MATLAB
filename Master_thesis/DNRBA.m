clc;
clear;
close all;
%DO NOT FORGET
global initial_flag; % the global flag used in test suite 

outpath = 'C:\Users\TakuyaIwase\Documents\MATLAB\Git\MATLAB\SICESSI2018\DNRBA\';

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
fprintf('---------------------------------------------------------------\n');

max_run = 50;
seed = 0:max_run-1;
NP = 50;
cnt_all = zeros(max_run,5);

for fnc = 1:4
	% DO NOT FORGET
	initial_flag = 0; % should set the flag to 0 for each run, each function 
	D = 2;
	MaxFes = 10000;
    fgoptima = get_fgoptima(fnc);
    x_lb = getLb(fnc);
    x_ub = getUb(fnc);
    % Calculate Niche radius
    R = sqrt((x_ub(1) - x_lb(1))^2) / 2;
    nr = R / NP^(1/D);
    
    parfor run = 1:max_run
        % Randomize population within optimization bounds 
        % (here dummy initialization within [0,1] ONLY for DEMO)    
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
        
        for i = 1:NP
            s = i + seed(run);
            rng(s);
            pop(i,:) = unifrnd(x_lb, x_ub, [1,D]);
            fitness(i,:) = Fun(pop(i,:),fnc);
            r(i) = r(i) * unifrnd(0,1);
            f(i) = fmin + (fmax - fmin) * rand;
        end
        
        pbest = pop;
        pcost = fitness;
        [gcost, idx] = min(pcost);
        gbest = pop(idx,:);
        
        for iter = 1:MaxFes
            % Calculate Dynamic Niche
            m = zeros(NP,1);
            for i = 1:NP
                dij = zeros(NP,D);
                dist = zeros(NP,1);
                sh = 0;
                for j = 1:NP
                    dij(j,:) = pop(i,:) - pop(j,:);
                    dist(j,:) = norm(dij(j,:));
                    if dist(j,:) < nr && i~=j
                        sh = sh + 1 - (dist(j,:)/nr);
                    else
                        sh = sh + 0;
                    end
                end
                m(i,:) = sh;
                if m(i,:) < nr
                    m(i,:) = nr;
                else
                    continue;
                end
            end
            % popution movement
            for i = 1:NP
                for j = 1:NP 
                    if dist(j,:) < m(j,:)
                        if fitness(i,:) < fitness(j,:)
                            v(j,:) = v(j,:) - dij(j,:) * f(j,:);
                            pop(j,:) = pop(j,:) + v(j,:);
                            % Apply simple bounds/limits
                            pop(j,:) = Bounds(pop(j,:),fnc);
                        else
                            continue
                        end   
                    end
                    fitness(j,:) = Fun(pop(j,:),fnc);
                end
            end
            
            m = zeros(NP,1);
            for i = 1:NP
                dij = zeros(NP,D);
                dist = zeros(NP,1);
                sh = 0;
                for j = 1:NP
                    dij(j,:) = pop(i,:) - pop(j,:);
                    dist(j,:) = norm(dij(j,:));
                    if dist(j,:) < nr && i~=j
                        sh = sh + 1 - (dist(j,:)/nr);
                    else
                        sh = sh + 0;
                    end
                end
                m(i,:) = sh;
                if m(i,:) < nr
                    m(i,:) = nr;
                else
                    continue;
                end
            end
            % Local search
            for i = 1:NP 
                Sloc = NaN(NP,D);
                if rand > r(i)      
                    for j = 1:NP 
                        if dist(j,:) < m(j,:)
                            if fitness(i,:) <= fitness(j,:)
                                Sloc(j,:) = pop(i,:) + A(i) * unifrnd(-m(j,:),m(j,:),[1,D]);
                                Sloc(j,:) = Bounds(Sloc(j,:),fnc);
                            end
                        end
                    end
                end
            end
            % random search
            for i = 1:NP                      
                Srnd(i,:) = pop(i,:) + unifrnd(-m(i,:),m(i,:),[1,D]); 
                Srnd(i,:) = Bounds(Srnd(i,:),fnc);
            end
            
            for i = 1:NP
                if  rand < A(i)
                    if fitness(i,:) < pcost(i,:)
                        A(i) = alpha * A(i);
                        r(i) = r(i) * norm(1 - exp(-gamma * iter));
                        pbest(i,:) = pop(i,:);
                        pcost(i,:) = fitness(i,:);
                    elseif Fun(Sloc(i,:),fnc) < pcost(i,:) % & Sloc(i,:)~=[0 0]
                        A(i) = alpha * A(i);
                        r(i) = r(i) * norm(1 - exp(-gamma * iter));
                        pbest(i,:) = Sloc(i,:);
                        pcost(i,:) = Fun(Sloc(i,:),fnc);
                    elseif Fun(Srnd(i,:),fnc) < pcost(i,:)% & Darray(2,1) < min(rdist(i))
                        A(i) = alpha * A(i);
                        r(i) = r(i) * norm(1 - exp(-gamma * iter));
                        pbest(i,:) = Srnd(i,:);
                        pcost(i,:) = Fun(Srnd(i,:),fnc);
                    end
                end     

                if pcost(i) <= gcost
                    gbest = pbest(i,:);
                    gcost = pcost(i);
                end
            end
            pop = pbest;
            fprintf(['Function: ' num2str(fnc) '  seed: (' num2str(run) '/' num2str(max_run) ')    The number of Evaluations: (' num2str(iter) '/' num2str(MaxFes) ') \n']);
            fprintf(['gbest = ' num2str(gbest) '\n']);
            fprintf(['gcost = ' num2str(gcost) '\n']); 
        end

        fit_all = horzcat(pop, pcost);
        csvwrite([outpath 'F' num2str(fnc) '\' 'F' num2str(fnc) '_' num2str(run) '_pbest.csv'],fit_all,0,0);
    end
end
% I—¹‰¹
[y, Fs] = audioread('C:\Users\TakuyaIwase\Music\datadelete.mp3');
soundsc(y, Fs);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

function fit = Fun(x, fnc)
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
