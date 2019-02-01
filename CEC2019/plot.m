clc;
clear;
close all;

path = 'C:\Users\TakuyaIwase\Documents\MATLAB\Git\MATLAB\CEC2019\NSBA\';
fNum = [4 6 7 10];
for i = 1:4
    M = csvread([path 'F' num2str(i) '\F' num2str(i) '_2_pbest.csv']);
    pop = M(:,1:2);
    xx1 = [get_lb(fNum(i)):0.01:get_ub(fNum(i))];
    xx2 = [get_lb(fNum(i)):0.01:get_ub(fNum(i))];
    y = NaN(1,length(xx1));
    fit = NaN(length(xx1),length(xx2));
    for j = 1:length(xx1)
        for k = 1:length(xx2)
            y(:,k) = niching_func([xx1(j) xx2(k)],fNum(i));
        end
        fit(:,j) = y;
    end

    figure;
    contour(xx1,xx2,fit,'Fill','on');
    title('iteration: 10000');
    xlabel('x1');
    ylabel('x2');
    colorbar;
    hold on;
    scatter(pop(:,1), pop(:,2),'r');
    saveas(gcf, [path 'results\F' num2str(i) '.png']);
    
end
