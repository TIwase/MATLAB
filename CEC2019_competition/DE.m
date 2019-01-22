
clc;
clear;

format long;
format compact;

max_eval = 10000;
max_run = 5;
max_func = 10;
seed = 0:max_run-1;
fnc_all = [];
% rand('seed', sum(100 * clock));
outpath = 'C:\Users\TakuyaIwase\Documents\MATLAB\bat_algorithm\CEC2019_competition\results\';

fhd=@cec19_func;

digit_1_reach =  10^(0);
digit_2_reach =  10^(-1);
digit_3_reach =  10^(-2);
digit_4_reach =  10^(-3);
digit_5_reach =  10^(-4);
digit_6_reach =  10^(-5);
digit_7_reach =  10^(-6);
digit_8_reach =  10^(-7);
digit_9_reach =  10^(-8);
digit_10_reach =  10^(-9);

for func = 3 : max_func
    lu = [];
    
    if func == 1
        dim = 9;
        x_ub = 8192.0;
        x_lb = -8192.0;
        lu = [x_lb * ones(1, dim); x_ub * ones(1, dim)];
        
    elseif func == 2
        dim = 16;
        x_ub = 16384.0;
        x_lb = -16384.0;
        lu = [x_lb * ones(1, dim); x_ub * ones(1, dim)];
        
    elseif func == 3
        dim = 18;
        x_ub = 4.0;
        x_lb = -4.0;
        lu = [x_lb * ones(1, dim); x_ub * ones(1, dim)];
        
    else
        dim = 10;
        x_ub = 100.0;
        x_lb = -100.0;
        lu = [x_lb * ones(1, dim); x_ub * ones(1, dim)];
    end
    
    optimum = 1.0;
    
    %% Record the best results
    outcome = [];
    digit_all = [];
    % Count correct digits
    cnt_1 = 0;
    cnt_2 = 0;
    cnt_3 = 0;
    cnt_4 = 0;
    cnt_5 = 0;
    cnt_6 = 0;
    cnt_7 = 0;
    cnt_8 = 0;
    cnt_9 = 0;
    cnt_10 = 0;
    
    fprintf('\n-------------------------------------------------------\n')
    fprintf('Function = %d, Dimension size = %d\n', func, dim)
    
    for run_id = 1 : max_run    
        %% Initialize the main population
        NP = 1500;
        pop = zeros(dim,NP);
        offspring = zeros(size(pop));
        s = 0;
        for i = 1:NP
            s = i+seed(run_id);
            rng(s);
            pop(:,i) = x_lb + (x_ub - x_lb) * rand(dim,1);
        end

        fitness = feval(fhd,pop,func);
        pbest = pop;
        pcost = fitness;
        [gcost, idx] = min(pcost);
        gbest = pop(:,idx);
        
        Cr = 0.5;       % Crossover Probability
        Fmax = 1;
        Fmin = 0;
        F = unifrnd(Fmin, Fmax, [dim, 1]);
        
        %% main loop
        for iter = 1:max_eval
            for i = 1:NP
                R_arr = randperm(NP);
                R_arr(R_arr==i) = [];

%                 offspring(:,i) = pop(:,R_arr(1)) + F.*(pop(:,R_arr(2))-pop(:,R_arr(3))); % rand/1/bin
                offspring(:,i) = gbest + F.*(pop(:,R_arr(2))-pop(:,R_arr(3))); % best/1/bin%
%                  offspring(:,i) = pop(:,i) + F.*(gbest-pop(:,i)) + F.*(pop(:,R_arr(1))-pop(:,R_arr(2))); % current-to-best/1

                offspring(:,i) = max(offspring(:,i), x_lb);
                offspring(:,i) = min(offspring(:,i), x_ub);
            end

            fitness = feval(fhd,offspring,func);
            jrnd = randi(dim,1);
            
            % Crossover
            for i = 1:NP
                for j = 1:dim
                    if j == jrnd || rand < Cr
                        newpop(j,i) = offspring(j,i);
                    else
                        newpop(j,i) = pbest(j,i);
                    end
                end
                newfit(:,i) = feval(fhd,newpop(:,i),func);
                if newfit(:,i) < pcost(:,i)
                    pcost(:,i) =newfit(:,i);
                    pbest(:,i) = newpop(:,i);
                end
                if pcost(:,i) < gcost
                    gcost = pcost(:,i);
                    gbest = pbest(:,i);
                end
            end
        
        fprintf(['The number of Evaluations: (' num2str(iter) '/' num2str(max_eval) ') \n']);
        fprintf(['gbest = ' num2str(gbest') '\n']);
        fprintf(['gcost = ' num2str(gcost) '\n']);     
        end
        
        %% record digits and FEs
        digit_1 = 0;
        digit_2 = 0;
        digit_3 = 0;
        digit_4 = 0;
        digit_5 = 0;
        digit_6 = 0;
        digit_7 = 0;
        digit_8 = 0;
        digit_9 = 0;
        digit_10 = 0;
        
        diff = gcost - optimum;

        if diff <= digit_10_reach  && digit_10 == 0
            fprintf('The first digit is caught at %d', max_eval); 
            digit_10 = 1;
            cnt_10 = cnt_10 + 1;
            break;

        elseif diff <= digit_9_reach  && digit_9 == 0
            fprintf('The first digit is caught at %d', max_eval); 
            digit_9 = 1;
            cnt_9 = cnt_9 + 1;
            break;

        elseif diff <= digit_8_reach  && digit_8 == 0
            fprintf('The first digit is caught at %d', max_eval); 
            digit_8 = 1;
            cnt_8 = cnt_8 + 1;
            break;

        elseif diff <= digit_7_reach  && digit_7 == 0
            fprintf('The first digit is caught at %d', max_eval); 
            digit_7 = 1;
            cnt_7 = cnt_7 + 1;
            break;

        elseif diff <= digit_6_reach  && digit_6 == 0
            fprintf('The first digit is caught at %d', max_eval); 
            digit_6 = 1;
            cnt_6 = cnt_6 + 1;
            break;

        elseif diff <= digit_5_reach  && digit_5 == 0
            fprintf('The first digit is caught at %d', max_eval); 
            digit_5 = 1;
            cnt_5 = cnt_5 + 1;
            break;

        elseif diff <= digit_4_reach  && digit_4 == 0
            fprintf('The first digit is caught at %d', max_eval); 
            digit_4 = 1;
            cnt_4 = cnt_4 + 1;
            break;
            
        elseif diff <= digit_3_reach  && digit_3 == 0
            fprintf('The first digit is caught at %d', max_eval); 
            digit_3 = 1;
            cnt_3 = cnt_3 + 1;
            break;
            
        elseif diff <= digit_2_reach  && digit_2 == 0
            fprintf('The first digit is caught at %d', max_eval); 
            digit_2 = 1;
            cnt_2 = cnt_2 + 1;
            break;
            
        elseif diff <= digit_1_reach  && digit_1 == 0
            fprintf('The first digit is caught at %d', max_eval); 
            digit_1 = 1;
            cnt_1 = cnt_1 + 1;
        end 
    
        %% 
        digit_run = vertcat(digit_1,digit_2,digit_3,digit_4,digit_5,digit_6,digit_7,digit_8,digit_9,digit_10);
        digit_all = [digit_all digit_run];
        
        fprintf('%d th run, best-so-far error value = %1.8e\n', run_id , diff)
%         outcome = [outcome; diff];

    end %% end 1 run
    
    csvwrite([outpath 'F' num2str(func) '_digit_accuracy.csv'],digit_all,0,0);
    
    digits_cnt = vertcat(cnt_1,cnt_2,cnt_3,cnt_4,cnt_5,cnt_6,cnt_7,cnt_8,cnt_9,cnt_10);
    fnc_all = [fnc_all digits_cnt];
    
    fprintf('\n')
%     fprintf('mean error value = %1.8e, std = %1.8e\n', mean(outcome), std(outcome))
end %% end 1 function run

csvwrite([outpath 'correct_digits_all.csv'],fnc_all,0,0);
% I—¹‰¹
[y, Fs] = audioread('mean_error.mp3');
soundsc(y, Fs);