clc;
clear;
close all;

outpath = 'C:\Users\TakuyaIwase\Documents\MATLAB\Git\MATLAB\SICESSI2018\BA\';

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
maxRun = 1;
seed = 0:maxRun-1;
fprintf('---------------------------------------------------------------\n');

for fnc = 3:3
    D = 2;
    MaxFes = 100;
    x_lb = getLb(fnc);
    x_ub = getUb(fnc);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
%     xx1 = -5:0.1:5;
%     xx2 = -5:0.1:5;
%     for j = 1:length(xx1)
%         for k = 1:length(xx2)
%             y(:,k) = Fun([xx1(j) xx2(k)],4);
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
            % popution movement
            for i = 1:NP
                v(i,:) = v(i,:) + (gbest - pop(i,:)) * f(i,:);
                pop(i,:) = pop(i,:) + v(i,:);
                % Apply simple bounds/limits
                pop(i,:) = Bounds(pop(i,:), fnc);
            end

            % Local search
            Sloc = NaN(NP,D);
            for i = 1:NP 
                if rand > r(i)      
                    Sloc(i,:) = gbest + A(i) * unifrnd(-1,1,[1,D]);
                    Sloc(i,:) = Bounds(Sloc(i,:), fnc);
                end
            end
            % random search
            for i = 1:NP                      
                Srnd(i,:) = unifrnd(x_lb, x_ub, [1,D]); 
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
            fprintf(['Function: ' num2str(fnc) '  seed: (' num2str(run) '/' num2str(maxRun) ')    The number of Evaluations: (' num2str(iter) '/' num2str(MaxFes) ') \n']);
            fprintf(['gbest = ' num2str(gbest) '\n']);
            fprintf(['gcost = ' num2str(gcost) '\n']); 
            
            filename = 'iter.gif'; % Specify the output file name
%             for idx = 1:nImages
                figure;
                xx1 = x_lb:0.01:x_ub;
                xx2 = x_lb:0.01:x_ub;
                for i = 1:length(xx1)
                    for j = 1:length(xx2)
                        y(j) = Fun([xx1(i),xx2(j)],fnc);
                    end
                    fit(:,i) = y;
                end
                title(['iteration: ' num2str(iter)]);
                contour(xx1,xx2,fit);
                hold on;
                scatter(pbest(:,1),pbest(:,2),'r');
                title(['Generation: ' num2str(iter)]);
                grid on;
                
%                 F = getframe;
%                 [X,map] = rgb2ind(F.cdata,256);
%                 if iter == 1
%                     imwrite(X,map,[outpath filename],'LoopCount',Inf,'DelayTime',0.2);
%                 else
%                     imwrite(X,map,[outpath filename],'WriteMode','append','DelayTime',0.2);
%                 end
                saveas(gcf,[outpath 'iter' num2str(iter) '.png']);
%             end
                close;
        end
        outData = horzcat(pop, fitness);
%         csvwrite([outpath 'F' num2str(fnc) '\F' num2str(fnc) '_' num2str(run) '_pbest.csv'], outData);
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
