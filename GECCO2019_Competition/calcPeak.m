clc;
clear;
close all;

inPath = 'C:\Users\TakuyaIwase\Documents\MATLAB\Git\MATLAB\GECCO2019_Competition\matlab\NSBA\F';
outPath = 'C:\Users\TakuyaIwase\Documents\MATLAB\Git\MATLAB\GECCO2019_Competition\matlab\NSBA\';

for i = 1:6
    path_all = [inPath num2str(i) '\'];
    input = dlmread([path_all 'F' num2str(i) '_peak_accuracy.csv']);
    E1(i,:) = mean(input(:,1)/getNumPeak(i));
    E2(i,:) = mean(input(:,2)/getNumPeak(i));
    E3(i,:) = mean(input(:,3)/getNumPeak(i));
    E4(i,:) = mean(input(:,4)/getNumPeak(i));
    E5(i,:) = mean(input(:,5)/getNumPeak(i));
end

PR = horzcat(E1,E2,E3,E4,E5);
dlmwrite([outPath 'NSBA.dat'], PR);
fprintf('DONE!!!\n');

function Num = getNumPeak(i)
    Num_all = [2 5 1 4 2 18];
    Num = Num_all(i);
end