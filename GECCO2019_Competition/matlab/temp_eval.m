clc;
clear;
close all;

fnc = 6;
max_run = 50;
% D = get_dimension(fnc);
% MaxFes = get_maxfes(fnc);
fgoptima = get_fgoptima(fnc);
% x_lb = get_lb(fnc);
% x_ub = get_ub(fnc);

for run = 1:max_run
%     for iter = 1:MaxFes
    path = 'C:\Users\TakuyaIwase\Documents\MATLAB\bat_algorithm\GECCO2018\matlab\results\F6\';
    infile = [path 'F6_' num2str(run) '_pbest.csv'];
    pop = csvread(infile, 0, 0, [0, 0, 99, 1]);
    pcost = csvread(infile, 0, 2, [0, 2, 99, 2]);
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
    csvwrite([path 'F' num2str(fnc) '_' num2str(run) '_pbest.csv'],fit_all,0,0);
end
csvwrite([path 'F' num2str(fnc) '_peak_accuracy.csv'],cnt_all,0,0);
fprintf(['Wrote output F' num2str(fnc) ' CSV file']);
% end


function [fit] = get_fgoptima(nfunc)
fgoptima = [200.0 1.0 1.0 200.0 1.031628453489877 186.7309088310239 1.0 2709.093505572820 1.0 -2.0 zeros(1,10)];
fit = fgoptima(nfunc);
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

function [rho] = get_rho(nfunc)
rho_ = [0.01*ones(1,4) 0.5 0.5 0.2 0.5 0.2 0.01*ones(1,11)];
rho = rho_(nfunc);
end

function [no] = get_no_goptima(nfunc)
nopt = [2 5 1 4 2 18 36 81 216 12 6 8 6 6 8 6 8 6 8 8];
no = nopt(nfunc);
end