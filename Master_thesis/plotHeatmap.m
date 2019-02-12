clc;
clear;
close all;

path = 'C:\Users\TakuyaIwase\Documents\MATLAB\Git\MATLAB\SICESSI2018\';

BA = csvread([path 'PR_BA_2.csv']);
NSBA = csvread([path 'PR_NSBA.csv']);
NRBA = csvread([path 'PR_NRBA.csv']);
DNRBA = csvread([path 'PR_DNRBA.csv']);

row = horzcat(BA(:,1), NSBA(:,1), NRBA(:,1), DNRBA(:,1));
heatmap({'BA','NSBA','NRBA','DNRBA'},{'1','2','3','4'}, row, 'Colormap', parula);
saveas(gcf,[path 'results_comp.png']);

fprintf('DONE!!!\n');
