clc;
clear;
close all;

path = 'C:\Users\TakuyaIwase\Documents\MATLAB\Git\MATLAB\CEC2019\NSBA\';
for i = 1:4
    M = csvread([path 'F' num2str(i) '\F' num2str(i) '_1_pbest.csv']);
    pop = M(:,1:2);

    xx1 = -10:0.01:10;
    xx2 = -10:0.01:10;
    for j = 1:length(xx1)
        for k = 1:length(xx2)
            y(:,k) = Func([xx1(j) xx2(k)],i);
        end
        fit(:,j) = y;
    end

    figure;
    contour(xx1,xx2,fit,'Fill','on');
    title('iteration: 30000');
    xlabel('x1');
    ylabel('x2');
    colorbar;
    hold on;
    scatter(pop(:,1), pop(:,2),'w');
    saveas(gcf, [path 'F' num2str(i) '\F' num2str(i) '.png']);
    
end
function fit = Func(x, num)
persistent fname;
if num == 1
    fname = str2func('Griewank');
elseif num == 2
    fname = str2func('Shubert');
end
fit = feval(fname, x);
end

function fit = Griewank(x)
fit = (x(1)^2 + x(2)^2) / 4000 - (cos(x(1)) * cos(x(2) / sqrt(2))) + 1;
end

function fit = Shubert(x)
sm = 0;
prod = 1;
for i = 1:2
    for j = 1:5
        sm = sm + j * cos( (j+1) * x(i) + j);
    end
    prod = prod * sm;
    sm = 0;
end
fit = prod;
end