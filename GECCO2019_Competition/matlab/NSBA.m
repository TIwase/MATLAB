clc;
clear;
close all;
%DO NOT FORGET
global initial_flag; % the global flag used in test suite 

path = 'C:\Users\takadamalab\Documents\MATLAB\results\';

if exist([path 'F1'],'dir') == 0
    mkdir C:\Users\takadamalab\Documents\MATLAB\results\ F1;
end
if exist([path 'F2'],'dir') == 0
    mkdir C:\Users\takadamalab\Documents\MATLAB\results\ F2;
end
if exist([path 'F3'],'dir') == 0
    mkdir C:\Users\takadamalab\Documents\MATLAB\results\ F3;
end
if exist([path 'F4'],'dir') == 0
    mkdir C:\Users\takadamalab\Documents\MATLAB\results\ F4;
end
if exist([path 'F5'],'dir') == 0
    mkdir C:\Users\takadamalab\Documents\MATLAB\results\ F5;
end
if exist([path 'F6'],'dir') == 0
    mkdir C:\Users\takadamalab\Documents\MATLAB\results\ F6;
end
if exist([path 'F7'],'dir') == 0
    mkdir C:\Users\takadamalab\Documents\MATLAB\results\ F7;
end
if exist([path 'F8'],'dir') == 0
    mkdir C:\Users\takadamalab\Documents\MATLAB\results\ F8;
end
if exist([path 'F9'],'dir') == 0
    mkdir C:\Users\takadamalab\Documents\MATLAB\results\ F9;
end
if exist([path 'F10'],'dir') == 0
    mkdir C:\Users\takadamalab\Documents\MATLAB\results\ F10;
end
if exist([path 'F11'],'dir') == 0
    mkdir C:\Users\takadamalab\Documents\MATLAB\results\ F11;
end
if exist([path 'F12'],'dir') == 0
    mkdir C:\Users\takadamalab\Documents\MATLAB\results\ F12;
end
if exist([path 'F13'],'dir') == 0
    mkdir C:\Users\takadamalab\Documents\MATLAB\results\ F13;
end
if exist([path 'F14'],'dir') == 0
    mkdir C:\Users\takadamalab\Documents\MATLAB\results\ F14;
end
if exist([path 'F15'],'dir') == 0
    mkdir C:\Users\takadamalab\Documents\MATLAB\results\ F15;
end
if exist([path 'F16'],'dir') == 0
    mkdir C:\Users\takadamalab\Documents\MATLAB\results\ F16;
end
if exist([path 'F17'],'dir') == 0
    mkdir C:\Users\takadamalab\Documents\MATLAB\results\ F17;
end
if exist([path 'F18'],'dir') == 0
    mkdir C:\Users\takadamalab\Documents\MATLAB\results\ F18;
end
if exist([path 'F19'],'dir') == 0
    mkdir C:\Users\takadamalab\Documents\MATLAB\results\ F19;
end
if exist([path 'F20'],'dir') == 0
    mkdir C:\Users\takadamalab\Documents\MATLAB\results\ F20;
end

fprintf('---------------------------------------------------------------\n');

max_run = 50;
seed = 0:max_run-1;
NP = 100;
cnt_all = zeros(max_run,5);

for fnc = 5:6
	% DO NOT FORGET
	initial_flag = 0; % should set the flag to 0 for each run, each function 
	D = get_dimension(fnc);
	MaxFes = get_maxfes(fnc);
    fgoptima = get_fgoptima(fnc);
    x_lb = get_lb(fnc);
    x_ub = get_ub(fnc);
    
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
%         Srnd = zeros(NP,D);
        fitness = zeros(NP,1);
        
        for i = 1:NP
            s = i + seed(run);
            rng(s);
            pop(i,:) = x_lb(1) + (x_ub(1) - x_lb(1)) * rand(1,D);
            fitness(i,:) = niching_func(pop(i,:),fnc);
            r(i) = r(i) * unifrnd(0,1);
            f(i) = fmin + (fmax - fmin) * rand;
        end
        
        pbest = pop;
        pcost = fitness;
        [gcost, idx] = max(pcost);
        gbest = pop(idx,:);
        
        for iter = 1:MaxFes
            % Calculate Dynamic Niche
            eq = zeros(NP,D);
            d = zeros(NP,D);
            for i = 1:NP
                dij = zeros(NP,D);
                dist = zeros(NP,1);
                for j = 1:NP
                    dij(j,:) = pop(i,:) - pop(j,:);
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
                pop(i,:) = max(pop(i,:),x_lb);
                pop(i,:) = min(pop(i,:),x_ub);               
                fitness(i,:) = niching_func(pop(i,:),fnc);
            end

            % Local search
            for i = 1:NP 
                Sloc = NaN(NP,D);
                if rand > r(i)      
                    Sloc(i,:) = pop(i,:) + A(i) * unifrnd(0,1,[1,D]);
                    Sloc(i,:) = max(Sloc(i,:),x_lb);
                    Sloc(i,:) = min(Sloc(i,:),x_ub);
                end
            end
            % random search
%             for i = 1:NP                      
%                 Srnd(i,:) = pop(i,:) + unifrnd(-m(i,:),m(i,:),[1,D]); 
%                 Srnd(i,:) = max(Srnd(i,:),x_lb);
%                 Srnd(i,:) = min(Srnd(i,:),x_ub);
%             end
            
            for i = 1:NP
                if  rand < A(i) & fitness(i,:) > pcost(i,:)
                    A(i) = alpha * A(i);
                    r(i) = r(i) * norm(1-exp(-gamma*iter));
                    pbest(i,:) = pop(i,:);
                    pcost(i,:) = fitness(i,:);
                elseif rand < A(i) & niching_func(Sloc(i,:),fnc) > pcost(i,:) % & Sloc(i,:)~=[0 0]
                    A(i) = alpha * A(i);
                    r(i) = r(i) * norm(1-exp(-gamma*iter));
                    pbest(i,:) = Sloc(i,:);
                    pcost(i,:) = niching_func(Sloc(i,:),fnc);
%                 elseif rand < A(i) & niching_func(Srnd(i,:),fnc) > pcost(i,:)% & Darray(2,1) < min(rdist(i))
%                     A(i) = alpha * A(i);
%                     r(i) = r(i) * norm(1-exp(-gamma*iter));
%                     pbest(i,:) = Srnd(i,:);
%                     pcost(i,:) = niching_func(Srnd(i,:),fnc);
                end     

                if pcost(i) >= gcost
                    gbest = pbest(i,:);
                    gcost = pcost(i);
                end
            end
            pop = pbest;
            fprintf(['seed: (' num2str(run) '/' num2str(max_run) ')    The number of Evaluations: (' num2str(iter) '/' num2str(MaxFes) ') \n']);
            fprintf(['gbest = ' num2str(gbest) '\n']);
            fprintf(['gcost = ' num2str(gcost) '\n']); 
        end
        % How many global optima have been found?
        acc_1 = 10^(-1);
        acc_2 = 10^(-2);
        acc_3 = 10^(-3);
        acc_4 = 10^(-4);
        acc_5 = 10^(-5);
        
        cnt_1 = 0;
        cnt_2 = 0;
        cnt_3 = 0;
        cnt_4 = 0;
        cnt_5 = 0;

        [count, goptima_found] = count_goptima(pop, fnc, acc_5);     
        if count ~= 0
            fprintf('f_%d, In the current population there are %d global optima!\n', fnc, count);
            goptima_found;
            for i = 1:size(goptima_found,1)
                val = niching_func(goptima_found(i,:), fnc);
                fprintf('F_p: %f, F_g:%f, diff: %f\n', val, get_fgoptima(fnc), abs(val - get_fgoptima(fnc)))
                fprintf('F_p - F_g <= %f : %d\n', acc_5, abs(val - get_fgoptima(fnc))<acc_5 )
            end
            cnt_5 = cnt_5 + count;
        end
        [count, goptima_found] = count_goptima(pop, fnc, acc_4); 
        if count ~= 0
            fprintf('f_%d, In the current population there are %d global optima!\n', fnc, count);
            goptima_found;
            for i = 1:size(goptima_found,1)
                val = niching_func(goptima_found(i,:), fnc);
                fprintf('F_p: %f, F_g:%f, diff: %f\n', val, get_fgoptima(fnc), abs(val - get_fgoptima(fnc)))
                fprintf('F_p - F_g <= %f : %d\n', acc_4, abs(val - get_fgoptima(fnc))<acc_4 )
            end
            cnt_4 = cnt_4 + count;
        end
        [count, goptima_found] = count_goptima(pop, fnc, acc_3);
        if count ~= 0
            fprintf('f_%d, In the current population there are %d global optima!\n', fnc, count);
            goptima_found;
            for i = 1:size(goptima_found,1)
                val = niching_func(goptima_found(i,:), fnc);
                fprintf('F_p: %f, F_g:%f, diff: %f\n', val, get_fgoptima(fnc), abs(val - get_fgoptima(fnc)))
                fprintf('F_p - F_g <= %f : %d\n', acc_3, abs(val - get_fgoptima(fnc))<acc_3 )
            end
            cnt_3 = cnt_3 + count;
        end
       [count, goptima_found] = count_goptima(pop, fnc, acc_2);     
        if count ~= 0
            fprintf('f_%d, In the current population there are %d global optima!\n', fnc, count);
            goptima_found;
            for i = 1:size(goptima_found,1)
                val = niching_func(goptima_found(i,:), fnc);
                fprintf('F_p: %f, F_g:%f, diff: %f\n', val, get_fgoptima(fnc), abs(val - get_fgoptima(fnc)))
                fprintf('F_p - F_g <= %f : %d\n', acc_2, abs(val - get_fgoptima(fnc))<acc_2 )
            end
            cnt_2 = cnt_2 + count;
        end
        [count, goptima_found] = count_goptima(pop, fnc, acc_1);
        if count ~= 0
            fprintf('f_%d, In the current population there are %d global optima!\n', fnc, count);
            goptima_found;
            for i = 1:size(goptima_found,1)
                val = niching_func(goptima_found(i,:), fnc);
                fprintf('F_p: %f, F_g:%f, diff: %f\n', val, get_fgoptima(fnc), abs(val - get_fgoptima(fnc)))
                fprintf('F_p - F_g <= %f : %d\n', acc_1, abs(val - get_fgoptima(fnc))<acc_1 )
            end
            cnt_1 = cnt_1 + count;
        end
        
        cnt_all(run,:) = horzcat(cnt_1, cnt_2, cnt_3, cnt_4, cnt_5);
        fit_all = horzcat(pop, pcost);
        csvwrite([path 'F' num2str(fnc) '\' 'F' num2str(fnc) '_' num2str(run) '_pbest.csv'],fit_all,0,0);
    end
    csvwrite([path 'F' num2str(fnc) '\' 'F' num2str(fnc) '_peak_accuracy.csv'],cnt_all,0,0);
    fprintf(['Wrote output F' num2str(fnc) ' CSV file \n']);
end
% I—¹‰¹
% [y, Fs] = audioread('C:\Users\TakuyaIwase\Music\datadelete.mp3');
% soundsc(y, Fs);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [fit] = get_dimension(nfunc)
Dims = [1 1 1 2 2 2 2 3 3 2 2 2 2 3 3 5 5 10 10 20]; % dimensionality of benchmark functions
fit = Dims(nfunc);
end

function [fit] = get_maxfes(nfunc)
Max_FEs = [50000*ones(1,5) 200000 200000 400000 400000 200000*ones(1,4) 400000*ones(1,7)]; 
fit = Max_FEs(nfunc);
end

function [fit] = get_fgoptima(nfunc)
fgoptima = [200.0 1.0 1.0 200.0 1.031628453489877 186.7309088310239 1.0 2709.093505572820 1.0 -2.0 zeros(1,10)];
fit = fgoptima(nfunc);
end

function [rho] = get_rho(nfunc)
rho_ = [0.01*ones(1,4) 0.5 0.5 0.2 0.5 0.2 0.01*ones(1,11)];
rho = rho_(nfunc);
end

function [no] = get_no_goptima(nfunc)
nopt = [2 5 1 4 2 18 36 81 216 12 6 8 6 6 8 6 8 6 8 8];
no = nopt(nfunc);
end

function [o] = get_copy_of_goptima(nfunc)
total_func_no = 20;

if nfunc > 10 & nfunc <= total_func_no
	load data/optima.mat; % saved the predefined optima, a 10*100 matrix;
	D = get_dimension(nfunc);
	o = o(:,1:D);
	return;
end

if nfunc == 1	    fname = 'data/F1_opt.dat';
elseif nfunc == 2	fname = 'data/F2_opt.dat';
elseif nfunc == 3	fname = 'data/F3_opt.dat';
elseif nfunc == 4	fname = 'data/F4_opt.dat';
elseif nfunc == 5	fname = 'data/F5_opt.dat';
elseif nfunc == 6	fname = 'data/F6_2D_opt.dat';
elseif nfunc == 7	fname = 'data/F6_3D_opt.dat';
elseif nfunc == 8	fname = 'data/F7_2D_opt.dat';
elseif nfunc == 9	fname = 'data/F7_3D_opt.dat';
elseif nfunc == 10	fname = 'data/F8_2D_opt.dat';
else
	fprintf('ERROR: Wrong function number: (%d).\n', nfunc);
	fprintf('       Please provide a function number in {1,2,...,%d}\n', total_func_no);
	fprintf('       For now function number == 1\n');
	fname = '';
end

o = load(fname); % saved the predefined optima
end

function ub = get_ub(fno)
dim = get_dimension(fno);
if (fno == 1 )
	ub = 30;
elseif (fno== 2 || fno== 3)
	ub = 1;
elseif (fno== 4)
	ub = 6*ones(1,2);
elseif (fno== 5)
	ub = [1.9 1.1];
elseif (fno== 6 || fno== 8)
	ub = 10*ones(1, dim);
elseif (fno== 7 || fno== 9)
	ub = 10*ones(1, dim);
elseif (fno== 10)
	ub = ones(1,2);
elseif (fno== 11 || fno== 12 || fno== 13)
	ub = 5*ones(1,dim);
elseif (fno== 14 || fno== 15)
	ub = 5*ones(1,dim);
elseif (fno== 16 || fno== 17)
	ub = 5*ones(1,dim);
elseif (fno== 18 || fno== 19)
	ub = 5*ones(1,dim);
elseif (fno== 20 )
	ub = 5*ones(1,dim);
else
	ub = [];
end
end

function lb = get_lb(fno)
dim = get_dimension(fno);
if (fno == 1 || fno== 2 || fno== 3)
	lb = 0;
elseif (fno== 4)
	lb = -6*ones(1,2);
elseif (fno== 5)
	lb = [-1.9 -1.1];
elseif (fno== 6 || fno== 8)
	lb = -10*ones(1, dim);
elseif (fno== 7 || fno== 9)
	lb = 0.25*ones(1, dim);
elseif (fno== 10)
	lb = zeros(1,2);
elseif (fno== 11 || fno== 12 || fno== 13)
	lb = -5*ones(1,dim);
elseif (fno== 14 || fno== 15)
	lb = -5*ones(1,dim);
elseif (fno== 16 || fno== 17)
	lb = -5*ones(1,dim);
elseif (fno== 18 || fno== 19)
	lb = -5*ones(1,dim);
elseif (fno== 20 )
	lb = -5*ones(1,dim);
else
	lb = [];
end
end

function [count, finalseeds] = count_goptima(pop, nfunc, accuracy)

% pop: NP, D
[NP, D] = size(pop);

% evaluate pop
fpop = zeros(1,NP);
for i=1:NP
	fpop(i) = niching_func(pop(i,:), nfunc);
end
fpoptmp = fpop;

% descent sorting
[B, IX] = sort(fpoptmp,'descend');

% Sort population based on its fitness values
% do not change the current populatio population. Work on cpop/cpopfits
cpop = pop(IX,:);
cpopfits = fpop(IX);

%get seeds
seeds = [];
seedsidx = [];

for i=1:NP
	found=0;
	[sNP,sD] = size(seeds);
	for j=1:sNP
		% Calculate distance from seeds
		dist = sqrt( sum( (seeds(j,:)-cpop(i,:)).^2,2) );
		% If the Euclidean distance is less than the radius
		if (dist <= get_rho(nfunc))
			found = 1;
			break;
		end
	end
	% If it is not similar to any other seed, then it is a new seed
	if (found == 0)
		seeds = [seeds;cpop(i,:)];
		seedsidx = [seedsidx; i];
	end
end

% Based on the accuracy: check which seeds are global optimizers
count = 0; finalseeds = [];
seedsfit = cpopfits(seedsidx);
[ idx ] = find(abs(seedsfit - get_fgoptima(nfunc))<=accuracy);
if (length(idx) > get_no_goptima(nfunc) )
	idx = idx(1:get_no_goptima(nfunc));
end
count = length(idx);
finalseeds = seeds(idx,:);
end

function fit = niching_func(x,func_num)
% Benchmark Functions for CEC'2013 Special Session and Competition on 
%        Niching Methods for Multimodal Function Optimization
%
% INPUT:  x :	 	is a 1xD input vector for evaluation
%         func_num : 	denotes the number of the objective function which 
%         		is going to be used.
%
% OUTPUT: fit : 	The objective function value of the x input vector.
%
% This benchmark set includes the following 12 multimodal test functions:
%F1 : Five-Uneven-Peak Trap (1D)
%F2 : Equal Maxima (1D)
%F3 : Uneven Decreasing Maxima (1D)
%F4 : Himmelblau (2D)
%F5 : Six-Hump Camel Back (2D)
%F6 : Shubert (2D, 3D)
%F7 : Vincent (2D, 3D)
%F8 : Modified Rastrigin - All Global Optima (2D)
%F9 : Composition Function 1 (2D)
%F10 : Composition Function 2 (2D)
%F11 : Composition Function 3 (2D, 3D, 5D, 10D)
%F12 : Composition Function 4 (3D, 5D, 10D, 20D)
%
% For more information please refer to the Technical Report of the 
% Special Session/Competition
% at: http://goanna.cs.rmit.edu.au/~xiaodong/cec13-niching/
%
% This source code is based on the following two works:
% P. N. Suganthan, N. Hansen, J. J. Liang, K. Deb, Y. P. Chen, A. Auger, and S. Tiwari, 
% "Problem definitions and evaluation criteria for the CEC 2005 special 
% session on real-parameter optimization," Nanyang Technological University 
% and KanGAL Report #2005005, IIT Kanpur, India., Tech. Rep., 2005.
% and
% B.-Y. Qu and P. N. Suganthan, "Novel multimodal problems and differential evolution 
% with ensemble of restricted tournament selection," in Proceedings of the 
% IEEE Congress on Evolutionary Computation, CEC 2010. Barcelona, Spain, 2010, pp. 1â€?.

persistent fname f_bias
total_func_no = 20;
MINMAN=1;      % Maximization

if func_num == 1	fname = str2func('five_uneven_peak_trap');
elseif func_num == 2	fname = str2func('equal_maxima');
elseif func_num == 3	fname = str2func('uneven_decreasing_maxima');
elseif func_num == 4	fname = str2func('himmelblau');
elseif func_num == 5	fname = str2func('six_hump_camel_back');
elseif func_num == 6	fname = str2func('shubert');
elseif func_num == 7	fname = str2func('vincent');
elseif func_num == 8	fname = str2func('shubert');
elseif func_num == 9	fname = str2func('vincent');
elseif func_num == 10	fname = str2func('modified_rastrigin_all');
elseif func_num == 11	fname = str2func('CF1');
elseif func_num == 12	fname = str2func('CF2');
elseif func_num == 13	fname = str2func('CF3');
elseif func_num == 14	fname = str2func('CF3');
elseif func_num == 15	fname = str2func('CF4');
elseif func_num == 16	fname = str2func('CF3');
elseif func_num == 17	fname = str2func('CF4');
elseif func_num == 18	fname = str2func('CF3');
elseif func_num == 19	fname = str2func('CF4');
elseif func_num == 20	fname = str2func('CF4');
else
	fprintf('ERROR: Wrong function number: (%d).\n', func_num);
	fprintf('       Please provide a function number in {1,2,...,%d}\n', total_func_no);
	fprintf('       For now function number == 1\n');
	fname = str2func('five_uneven_peak_trap');
end

f_bias = zeros(1,total_func_no);
fit = f_bias(func_num) + MINMAN*feval(fname,x);
end
%==============================================================================
% F1: Five-Uneven-Peak Trap
%==============================================================================
function fit = five_uneven_peak_trap(x)
% F1: Five-Uneven-Peak Trap
% Variable ranges: x in [0, 30]
% No. of global peaks: 2
% No. of local peaks:  3.

result = -1;
if (x >=0 & x < 2.5)
	result = 80*(2.5-x);
elseif (x >= 2.5 & x < 5.0)
	result = 64*(x-2.5);
elseif (x >= 5.0 & x < 7.5)
	result = 64*(7.5-x);
elseif (x >= 7.5 & x < 12.5)
	result = 28*(x-7.5);
elseif (x >= 12.5 & x < 17.5)
	result = 28*(17.5-x);
elseif (x >= 17.5 & x < 22.5)
	result = 32*(x-17.5);
elseif (x >= 22.5 & x < 27.5)
	result = 32*(27.5-x);
elseif (x >= 27.5 & x <= 30)
	result = 80*(x-27.5);
end
fit = result;
end
%==============================================================================
% F2: Equal Maxima
%==============================================================================
function fit = equal_maxima(x)
% F2: Equal Maxima
% Variable ranges: x in [0, 1]
% No. of global peaks: 5
% No. of local peaks:  0.

fit = sin( 5*pi*x ).^6;
end
%==============================================================================
% F3: Uneven Decreasing Maxima
%==============================================================================
function fit = uneven_decreasing_maxima(x)
% F3: Uneven Decreasing Maxima
% Variable ranges: x in [0, 1]
% No. of global peaks: 1
% No. of local peaks:  4.

fit = exp(-2*log(2)*((x-0.08)/0.854)^2)*sin(5*pi*(x^0.75-0.05))^6;
end
%==============================================================================
% F4: Himmelblau
%==============================================================================
function fit = himmelblau(x)
% F4: Himmelblau
% Variable ranges: x, y in [âˆ?, 6]
% No. of global peaks: 4
% No. of local peaks:  0.

fit = 200 - (x(1).^2 + x(2) - 11).^2 - (x(1) + x(2).^2 - 7).^2;
end
%==============================================================================
% F5: Six-Hump Camel Back
%==============================================================================
function fit = six_hump_camel_back(x)
% F5: Six-Hump Camel Back
% Variable ranges: x in [âˆ?.9, 1.9]; y in [âˆ?.1, 1.1]
% No. of global peaks: 2
% No. of local peaks:  2.

fit = -((4-2.1*x(1).^2+(x(1).^4)/3).*x(1).^2+x(1).*x(2)+(4*x(2).^2-4).*x(2).^2);
end
%==============================================================================
% F6: Shubert
%==============================================================================
function fit = shubert(x)
% F6: Shubert
% Variable ranges: x_i in  [âˆ?0, 10]^n, i=1,2,...,n
% No. of global peaks: n*3^n
% No. of local peaks: many.

[tmp,D] = size(x);
result = 1;
j = [1:5];
for i=1:D
	result = result * sum(j.*cos((j+1).*x(i)+j));
end
fit = -result;
end
%==============================================================================
% F7: Vincent
%==============================================================================
function fit = vincent(x)
% F7: Vincent
% Variable range: x_i in [0.25, 10]^n, i=1,2,...,n
% No. of global optima: 6^n
% No. of local optima:  0.

[tmp,D] = size(x);
fit = sum( sin( 10*log(x) ) )/D;
end
%==============================================================================
% F8: Modified Rastrigin - All Global Optima
%==============================================================================
function fit = modified_rastrigin_all(x)
% Variable ranges: x_i in [0, 1]^n, i=1,2,...,n
% No. of global peaks: \prod_{i=1}^n k_i
% No. of local peaks:  0.
MMP = 0;
[tmp,D] = size(x);
if D == 2
	MMP = [3 4];
elseif D == 8
	MMP = [1 2 1 2 1 3 1 4];
elseif D == 16
	MMP = [1 1 1 2 1 1 1 2 1 1 1 3 1 1 1 4];
end

fit = -sum( 10 + 9*cos( 2*pi*MMP.*x ) );
end
%==============================================================================
%1. Composition Function 1, n=6
%==============================================================================
function fit = CF1(x)
global initial_flag
persistent func_num func o sigma lambda bias M

[ps,D] = size(x);
func_num = 6;
lb = -5; ub = 5;
if initial_flag==0
	load data/optima.mat % saved the predefined optima
	if length( o(1,:) ) >= D
		o = o(:,1:D);
	else
		o = lb + (ub - lb) * rand(func_num,D);
	end
	initial_flag=1;
	func.f1 = str2func('FGriewank');
	func.f2 = str2func('FGriewank');
	func.f3 = str2func('FWeierstrass');
	func.f4 = str2func('FWeierstrass');
	func.f5 = str2func('FSphere');
	func.f6 = str2func('FSphere');
	bias = zeros(1,func_num);
	sigma = ones(1,func_num);
	lambda = [1; 1; 8; 8; 1/5; 1/5];
	lambda = repmat(lambda,1,D);
	for i = 1:func_num
		eval(['M.M' int2str(i) '= diag(ones(1,D));']);
	end
end
fit = hybrid_composition_func(x, func_num, func, o, sigma, lambda, bias, M);
end
%==============================================================================
%2. Composition Function 2, n=8
%==============================================================================
function fit = CF2(x)
global initial_flag
persistent func_num func o sigma lambda bias M

[ps,D] = size(x);
func_num = 8;
lb = -5; ub = 5;
if initial_flag==0
	initial_flag=1;
	load data/optima.mat % saved the predefined optima
	if length( o(1,:) ) >= D
		o = o(:,1:D);
	else
		o = lb + (ub - lb) * rand(func_num,D);
	end
	func.f1 = str2func('FRastrigin');
	func.f2 = str2func('FRastrigin');
	func.f3 = str2func('FWeierstrass');
	func.f4 = str2func('FWeierstrass');
	func.f5 = str2func('FGriewank');
	func.f6 = str2func('FGriewank');
	func.f7 = str2func('FSphere');
	func.f8 = str2func('FSphere');
	bias = zeros(1,func_num);
	sigma = ones(1,func_num);
	lambda = [1; 1; 10; 10; 1/10; 1/10; 1/7; 1/7];
	lambda = repmat(lambda,1,D);
	for i = 1:func_num
		eval(['M.M' int2str(i) '= diag(ones(1,D));']);
	end
end
fit = hybrid_composition_func(x, func_num, func, o, sigma, lambda, bias, M);
end
%==============================================================================
%3. Composition Function 3, n=6
%==============================================================================
function fit = CF3(x)
global initial_flag
persistent func_num func o sigma lambda bias M

[ps,D] = size(x);
func_num = 6;
lb = -5; ub = 5;
if initial_flag==0
	initial_flag=1;
	load data/optima.mat % saved the predefined optima, a 10*100 matrix;
	if length( o(1,:) ) >= D
		o = o(:,1:D);
	else
		o = lb + (ub - lb) * rand(func_num,D);
	end
	func.f1 = str2func('FEF8F2');
	func.f2 = str2func('FEF8F2');
	func.f3 = str2func('FWeierstrass');
	func.f4 = str2func('FWeierstrass');
	func.f5 = str2func('FGriewank');
	func.f6 = str2func('FGriewank');
	bias = zeros(1,func_num);
	sigma = [1,1,2,2,2,2];
	lambda = [1/4; 1/10; 2; 1; 2; 5];
	lambda = repmat(lambda,1,D);
	c = ones(1,func_num);
	if 	D==2,	load data/CF3_M_D2.mat,
	elseif 	D==3,	load data/CF3_M_D3.mat,
	elseif 	D==5,	load data/CF3_M_D5.mat,
	elseif 	D==10,	load data/CF3_M_D10.mat,
	elseif 	D==20,	load data/CF3_M_D20.mat,
	else 
		for i = 1:func_num
			%A = normrnd(0,1,D,D);
			%eval(['M.M' int2str(i) '= LocalGramSchmidt( A );']);
			eval(['M.M' int2str(i) '= RotMatrixCondition( D,c(i) );']);
		end
	end
end
fit = hybrid_composition_func(x,func_num,func,o,sigma,lambda,bias,M);
end
%==============================================================================
%4. Composition Function 4, n=8
%==============================================================================
function fit = CF4(x)
global initial_flag
persistent func_num func o sigma lambda bias M

[ps,D] = size(x);
func_num = 8;
lb = -5; ub = 5;
if initial_flag==0
	initial_flag=1;
	load data/optima.mat % saved the predefined optima, a 10*100 matrix;
	if length( o(1,:) ) >= D
		o = o(:,1:D);
	else
		o = lb + (ub - lb) * rand(func_num,D);
	end
	func.f1 = str2func('FRastrigin');
	func.f2 = str2func('FRastrigin');
	func.f3 = str2func('FEF8F2');
	func.f4 = str2func('FEF8F2');
	func.f5 = str2func('FWeierstrass');
	func.f6 = str2func('FWeierstrass');
	func.f7 = str2func('FGriewank');
	func.f8 = str2func('FGriewank');
	bias = zeros(1,func_num);
	sigma = [1,1,1,1,1,2,2,2];
	lambda = [ 4; 1; 4; 1; 1/10; 1/5; 1/10; 1/40];
	lambda = repmat(lambda,1,D);
	c = ones(1,func_num);
	if 	D==2,	load data/CF4_M_D2.mat,
	elseif 	D==3,	load data/CF4_M_D3.mat,
	elseif 	D==5,	load data/CF4_M_D5.mat,
	elseif 	D==10,	load data/CF4_M_D10.mat,
	elseif 	D==20,	load data/CF4_M_D20.mat,
	else 
		for i = 1:func_num
			%A = normrnd(0,1,D,D);
			%eval(['M.M' int2str(i) '= LocalGramSchmidt( A );']);
			eval(['M.M' int2str(i) '= RotMatrixCondition( D,c(i) );']);
		end
	end
end
fit = hybrid_composition_func(x, func_num, func, o, sigma, lambda, bias, M);
end
%==============================================================================
% Hybrid Composition General Framework
%==============================================================================
function res = hybrid_composition_func(x, func_num, func, o, sigma, lambda, bias, M)
[ps,D] = size(x);
for i = 1:func_num
	oo = repmat( o(i,:), ps, 1 );
	weight(:,i) = exp( -sum( (x-oo).^2, 2 )./2./( D * sigma(i)^2 ) );
end

[tmp,tmpid] = sort(weight,2);
for i = 1:ps
	weight(i,:) = (weight(i,:)==tmp(i,func_num)) .* weight(i,:) + (weight(i,:)~=tmp(i,func_num)) .* (weight(i,:).*(1-tmp(i,func_num).^10));
end
if sum(weight,2) == 0
	weight = weight + 1;
end
weight = weight ./ repmat( sum(weight,2), 1, func_num );
it = 0;
res = 0;
for i = 1:func_num
	oo = repmat(o(i,:),ps,1);
	eval(['f = feval(func.f' int2str(i) ',((x-oo)./repmat(lambda(i,:),ps,1))*M.M' int2str(i) ');']);
	x1 = 5*ones(1,D);
	eval(['f1 = feval(func.f' int2str(i) ',(x1./lambda(i,:))*M.M' int2str(i) ');']);
	fit1 = 2000 .* f ./ f1;
	res = res + weight(:,i) .* ( fit1 + bias(i) );
end
res = -res;
end
%==============================================================================
% Basic Functions
%==============================================================================

%------------------------------------------------------------------------------
% Sphere Function
%------------------------------------------------------------------------------
function f = FSphere(x)
%Please notice there is no use to rotate a sphere function, with rotation
%here just for a similar structure as other functions and easy programming
[ps,D] = size(x);
f = sum( x.^2, 2);
end
%------------------------------------------------------------------------------
% Griewank's Function
%------------------------------------------------------------------------------
function f = FGriewank(x)
[ps,D] = size(x);
f = 1;
for i = 1:D
	f = f.*cos( x(:,i)./sqrt(i) );
end
f = sum( x.^2, 2)./4000 - f + 1;
end
%------------------------------------------------------------------------------
% Rastrigin's Function
%------------------------------------------------------------------------------
function f = FRastrigin(x)
[ps,D] = size(x);
f = sum( x.^2-10.*cos( 2.*pi.*x )+10, 2 );
end
%------------------------------------------------------------------------------
% Weierstrass Function
%------------------------------------------------------------------------------
function f = FWeierstrass(x)
[ps,D] = size(x);
x = x+0.5;
a = 0.5;
b = 3;
kmax = 20;
c1(1:kmax+1) = a.^(0:kmax);
c2(1:kmax+1) = 2*pi*b.^(0:kmax);
f = 0;
c = -w(0.5,c1,c2);
for i = 1:D
	f = f + w( x(:,i)', c1, c2 );
end
f = f + c*D;

function y = w(x,c1,c2)
y = zeros(length(x),1);
for k = 1:length(x)
	y(k) = sum( c1.*cos( c2.*x(:,k) ) );
end
end
end
%------------------------------------------------------------------------------
% FEF8F2 Function
%------------------------------------------------------------------------------
function f = FEF8F2(x)
[ps,D] = size(x);
f = 0;
for i = 1:(D-1)
	f = f + F8F2( x(:,[i,i+1])+1 );     % (1,...,1) is minimum
end
f = f + F8F2( x(:,[D,1]) +1 );           % (1,...,1) is minimum
end
%------------------------------------------------------------------------------
% F8F2 Function
%------------------------------------------------------------------------------
function f = F8F2(x) 
f2 = 100.*(x(:,1).^2-x(:,2)).^2+(1-x(:,1)).^2; 
f = 1+f2.^2./4000-cos(f2); 
end
%------------------------------------------------------------------------------
% classical Gram Schmid  %TODO: why not use matlab's internal functions?
%------------------------------------------------------------------------------
function [q,r] = LocalGramSchmidt (A)
% computes the QR factorization of $A$ via
% classical Gram Schmid 

[n,m] = size(A); 
q = A;    
for j=1:m
    for i=1:j-1 
        r(i,j) = q(:,j)'*q(:,i);
    end
    for i=1:j-1   
      q(:,j) = q(:,j) -  r(i,j)*q(:,i);
    end
    t =  norm(q(:,j),2 ) ;
    q(:,j) = q(:,j) / t ;
    r(j,j) = t  ;
end 
end
%------------------------------------------------------------------------------
% Generates a D-dimensional rotation matrix with predifined Condition Number (c)
%------------------------------------------------------------------------------
function M = RotMatrixCondition(D,c)

% A random normal matrix
A = normrnd(0,1,D,D);

% P Orthogonal matrix
P = LocalGramSchmidt(A);

% A random normal matrix
A = normrnd(0,1,D,D);

% Q Orthogonal matrix
Q = LocalGramSchmidt(A);

% Make a Diagonal matrix D with prespecified Condition Number
u = rand(1,D);
D = c .^ ( (u-min(u))./(max(u)-min(u)) );
D = diag(D);

% M rotation matrix with Condition Number c
M = P * D * Q;
end

%==============================================================================
%============================= End of file ====================================
%==============================================================================
