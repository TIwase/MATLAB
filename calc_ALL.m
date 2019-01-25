clc;
clear;
close all;

inPath = 'C:\Users\TakuyaIwase\Documents\MATLAB\Git\MATLAB\GECCO2019_Competition\NichingCompetition2013FinalData\';
outPath = 'C:\Users\TakuyaIwase\Documents\MATLAB\Git\MATLAB\GECCO2019_Competition\matlab\results_all\';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input state-of-the-art methods
NSGAII = dlmread([inPath 'ANSGAII_PR.dat']);
cma1 = dlmread([inPath 'cma1_PR.dat']);
CDE = dlmread([inPath 'CDE_PR.dat']);
dade1 = dlmread([inPath 'dade1_PR.dat']);
dade2 = dlmread([inPath 'dade2_PR.dat']);
decg = dlmread([inPath 'decg_PR.dat']);
delg = dlmread([inPath 'delg_PR.dat']);
dels_aj = dlmread([inPath 'dels_ajitter_PR.dat']);
denrand1 = dlmread([inPath 'denrand1_PR.dat']);
denrand2 = dlmread([inPath 'denrand2_PR.dat']);
ipop1 = dlmread([inPath 'ipop1_PR.dat']);
nea1 = dlmread([inPath 'nea1_PR.dat']);
nea2 = dlmread([inPath 'nea2_PR.dat']);
Molina = dlmread([inPath 'Molina_PR.dat']);
PNANSGAII = dlmread([inPath 'PNANSGAII_PR.dat']);
% My Proposed
NSBA = dlmread('C:\Users\TakuyaIwase\Documents\MATLAB\Git\MATLAB\GECCO2019_Competition\matlab\NSBA\NSBA.dat');
NRBA = dlmread('C:\Users\TakuyaIwase\Documents\MATLAB\Git\MATLAB\GECCO2019_Competition\matlab\NRBA\NRBA.dat');
DNRBA = dlmread('C:\Users\TakuyaIwase\Documents\MATLAB\Git\MATLAB\GECCO2019_Competition\matlab\DNRBA\DNRBA.dat');

xVal = {'NSGAII','CMA-ES','CDE','daDE/nrand/1','daDE/nrand/2','DECG','DELG','DELS-aj','DE/nrand/1','DE/nrand/2','IPOP-CMA-ES','NEA1','NEA2','N-VMO','PNA-NSGAII','NSBA','NRBA','DNRBA'};
yVal = {'1','2','3','4','5','6'};

figure;
E1 = horzcat(NSGAII(1:6,1),cma1(1:6,1),CDE(1:6,1),dade1(1:6,1),dade2(1:6,1),decg(1:6,1),delg(1:6,1),dels_aj(1:6,1),denrand1(1:6,1),denrand2(1:6,1),ipop1(1:6,1),nea1(1:6,1),nea2(1:6,1),Molina(1:6,1),PNANSGAII(1:6,1),NSBA(:,1),NRBA(:,1),DNRBA(:,1));
heatmap(xVal,yVal,E1,'Colormap',parula);
saveas(gcf,[outPath 'E-1.png']);
close;

figure;
E2 = horzcat(NSGAII(1:6,2),cma1(1:6,2),CDE(1:6,2),dade1(1:6,2),dade2(1:6,2),decg(1:6,2),delg(1:6,2),dels_aj(1:6,2),denrand1(1:6,2),denrand2(1:6,2),ipop1(1:6,2),nea1(1:6,2),nea2(1:6,2),Molina(1:6,2),PNANSGAII(1:6,2),NSBA(:,2),NRBA(:,2),DNRBA(:,2));
heatmap(xVal,yVal,E2,'Colormap',parula);
saveas(gcf,[outPath 'E-2.png']);
close;

figure;
E3 = horzcat(NSGAII(1:6,3),cma1(1:6,3),CDE(1:6,3),dade1(1:6,3),dade2(1:6,3),decg(1:6,3),delg(1:6,3),dels_aj(1:6,3),denrand1(1:6,3),denrand2(1:6,3),ipop1(1:6,3),nea1(1:6,3),nea2(1:6,3),Molina(1:6,3),PNANSGAII(1:6,3),NSBA(:,3),NRBA(:,3),DNRBA(:,3));
heatmap(xVal,yVal,E3,'Colormap',parula);
saveas(gcf,[outPath 'E-3.png']);
close;

figure;
E4 = horzcat(NSGAII(1:6,4),cma1(1:6,4),CDE(1:6,4),dade1(1:6,4),dade2(1:6,4),decg(1:6,4),delg(1:6,4),dels_aj(1:6,4),denrand1(1:6,4),denrand2(1:6,4),ipop1(1:6,4),nea1(1:6,4),nea2(1:6,4),Molina(1:6,4),PNANSGAII(1:6,4),NSBA(:,4),NRBA(:,4),DNRBA(:,4));
heatmap(xVal,yVal,E4,'Colormap',parula);
saveas(gcf,[outPath 'E-4.png']);
close;

figure;
E5 = horzcat(NSGAII(1:6,5),cma1(1:6,5),CDE(1:6,5),dade1(1:6,5),dade2(1:6,5),decg(1:6,5),delg(1:6,5),dels_aj(1:6,5),denrand1(1:6,5),denrand2(1:6,5),ipop1(1:6,5),nea1(1:6,5),nea2(1:6,5),Molina(1:6,5),PNANSGAII(1:6,5),NSBA(:,5),NRBA(:,5),DNRBA(:,5));
heatmap(xVal,yVal,E5,'Colormap',parula);
saveas(gcf,[outPath 'E-5.png']);
close;

fprintf('DONE!!!\n');