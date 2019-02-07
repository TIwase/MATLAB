clc;
clear;
close all;

path = 'C:\Users\TakuyaIwase\Documents\MATLAB\Git\MATLAB\SICESSI2018\DNRBA\';

for nfnc = 1:1
    M = csvread([path 'F' num2str(nfnc) '\F' num2str(nfnc) '_15_pbest.csv']);
    pop = M(:,1:2);
    x_lb = getLb(nfnc);
    x_ub = getUb(nfnc);
    
    if nfnc == 2
        fun = @(x1,x2) (4 - 2.1 * x1^2 + (x1^4) / 3) * x1^2 + x1 * x2 + (-4 + 4 * x2^2) * x2^2;
        figure;
        fcontour(fun,'Fill','on');
        hold on;
        scatter(pop(:,1),pop(:,2),'r');
        xlim([x_lb(1) x_ub(1)]);
        ylim([x_lb(2) x_ub(2)]);
        grid on;
        saveas(gcf,[path 'F' num2str(nfnc) '.png']);
        continue;
        
    elseif nfnc == 3
        xx1 = x_lb(1):0.005:x_ub(1);
        xx2 = x_lb(2):0.005:x_ub(2);
    else
        xx1 = x_lb(1):0.01:x_ub(1);
        xx2 = x_lb(2):0.01:x_ub(2);
    end
    y = [];
    fit = [];
    for i = 1:length(xx1)
        for j = 1:length(xx2)
            y(j) = Fun([xx1(i),xx2(j)],nfnc);
        end
        fit(:,i) = y;
    end

    figure;
    contour(xx1,xx2,fit,'Fill','on');
    hold on;
    scatter(pop(:,1),pop(:,2),'r');
    xlim([x_lb(1) x_ub(1)]);
    ylim([x_lb(2) x_ub(2)]);
    grid on;
    saveas(gcf,[path 'F' num2str(nfnc) '3.png']);
%     end
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
