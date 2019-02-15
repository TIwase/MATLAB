clc;
clear;
close all;

path = 'C:\Users\TakuyaIwase\Documents\MATLAB\Git\MATLAB\Master_thesis\NSBA\';
MaxRun = 30;

if exist([path 'results\'],'dir') == 0
    mkdir([path 'results']);
end
acc_1 = 0.1;
acc_2 = 0.01;
acc_3 = 0.001;

% cnt_0 = zeros(MaxRun,1);
% cnt_1 = zeros(MaxRun,1);
% cnt_2 = zeros(MaxRun,1);

% PR_all = zeros(length(), 6);
% PA_all = zeros(length(), 6);

for fnc = 1:4
    for seed = 1:MaxRun
        M = csvread([path 'F' num2str(fnc) '\F' num2str(fnc) '_' num2str(seed) '_pbest.csv']);
        pop = M(:,1:2);
        
        [cnt1(seed,:), optima_found1] = cntOptima(pop, fnc, acc_1);
        [cnt2(seed,:), optima_found2] = cntOptima(pop, fnc, acc_2);
        [cnt3(seed,:), optima_found3] = cntOptima(pop, fnc, acc_3);
    end
    
    [optima, len] = getOptima(fnc);
    
    PRmean1 = mean(cnt1) / len;
    PRstd1 = std(cnt1) / len;
    PRmean2 = mean(cnt2) / len;
    PRstd2 = std(cnt2) / len;
    PRmean3 = mean(cnt3) / len;
    PRstd3 = std(cnt3) / len;
    
    PR_all(fnc,:) = [PRmean1, PRstd1, PRmean2, PRstd2, PRmean3, PRstd3];
end

csvwrite([path 'results\PR.csv'], PR_all);
fprintf('DONE!!!\n');

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

function [result, num] = getOptima(fnc)

f1optima = [6.28 8.8769; 6.28 -8.8769; -6.28 8.8769; -6.28 -8.8769; 3.14 4.4385; 3.14 -4.4385; -3.14 4.4385; -3.14 -4.4385; 0 8.8769; 0 -8.8769; 6.28 0; -6.28 0; 9.42 4.4385; 9.42 -4.4385; -9.42 4.4385; -9.42 -4.4385; 0 0];
f2optima = [0.0898 -0.7126; -0.0898 0.7126; 1.704 -0.7965; -1.704 0.7965];
f3optima = [2.20 1.57; 2.203 2.7115];
f4optima = [3 2; -2.805118 3.283186; -3.779310 -3.283186; 3.584458 -1.848126];

if fnc == 1
    result = f1optima;
    num = length(f1optima);
elseif fnc == 2
    result = f2optima;
    num = length(f2optima);
elseif fnc == 3
    result = f3optima;
    num = length(f3optima);
elseif fnc == 4
    result = f4optima;
    num = length(f4optima);
end
end

function [count, optima_found] = cntOptima(x,fnc,accuracy)
count = 0;
k = 1;
[optima, num] = getOptima(fnc);

for i = 1:length(optima)
    for j = 1:length(x)
        diff(j,:) = norm( Fun(optima(i,:),fnc) - Fun(x(j,:),fnc) );
    end
    
    [minDiff, idx(i,:)] = min(diff);
    
    if minDiff < accuracy
        count = count + 1;
        cntID(i,:) = idx(i,:);
        optima_found(k,:) = x(cntID(i),:);
        k = k + 1;
    else
        optima_found(k,:) = NaN(1,2);
        k = k + 1;
    end
end
end
