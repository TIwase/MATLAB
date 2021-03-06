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
BA = dlmread('C:\Users\TakuyaIwase\Documents\MATLAB\Git\MATLAB\GECCO2019_Competition\matlab\BA\BA.dat');
NSBA = dlmread('C:\Users\TakuyaIwase\Documents\MATLAB\Git\MATLAB\GECCO2019_Competition\matlab\NSBA\NSBA.dat');
NRBA = dlmread('C:\Users\TakuyaIwase\Documents\MATLAB\Git\MATLAB\GECCO2019_Competition\matlab\NRBA\NRBA.dat');
DNRBA = dlmread('C:\Users\TakuyaIwase\Documents\MATLAB\Git\MATLAB\GECCO2019_Competition\matlab\DNRBA\DNRBA.dat');

xVal = {'NSGAII','CMA-ES','CDE','daDE/nrand/1','daDE/nrand/2','DECG','DELG','DELS-aj','DE/nrand/1','DE/nrand/2','IPOP-CMA-ES','NEA1','NEA2','N-VMO','PNA-NSGAII','NSBA','NRBA','DNRBA'};
yVal = {'1','2','3','4','5','6'};

% figure;
% E1 = horzcat(NSGAII(1:6,i),cma1(1:6,i),CDE(1:6,i),dade1(1:6,i),dade2(1:6,i),decg(1:6,i),delg(1:6,i),dels_aj(1:6,i),denrand1(1:6,i),denrand2(1:6,i),ipop1(1:6,i),nea1(1:6,i),nea2(1:6,i),Molina(1:6,i),PNANSGAII(1:6,i),NSBA(:,1),NRBA(:,1),DNRBA(:,1));
% heatmap(xVal,yVal,E1,'Colormap',parula);
% saveas(gcf,[outPath 'E-1.png']);
% close;
for i = 1:5
    midNSGAII(:,i) = median(NSGAII(1:6,i));
    meanNSGAII(:,i) = mean(NSGAII(1:6,i));
    stdNSGAII(:,i) = std(NSGAII(1:6,i));
    midcma1(:,i) = median(cma1(1:6,i));
    meancma1(:,i) = mean(cma1(1:6,i));
    stdcma1(:,i) = std(cma1(1:6,i));
    midCDE(:,i) = median(CDE(1:6,i));
    meanCDE(:,i) = mean(CDE(1:6,i));
    stdCDE(:,i) = std(CDE(1:6,i));
    middade1(:,i) = median(dade1(1:6,i));
    meandade1(:,i) = mean(dade1(1:6,i));
    stddade1(:,i) = std(dade1(1:6,i));
    middade2(:,i) = median(dade2(1:6,i));
    meandade2(:,i) = mean(dade2(1:6,i));
    stddade2(:,i) = std(dade2(1:6,i));
    middecg(:,i) = median(decg(1:6,i));
    meandecg(:,i) = mean(decg(1:6,i));
    stddecg(:,i) = std(decg(1:6,i));
    middelg(:,i) = median(delg(1:6,i));
    meandelg(:,i) = mean(delg(1:6,i));
    stddelg(:,i) = std(delg(1:6,i));
    middels(:,i) = median(dels_aj(1:6,i));
    meandels(:,i) = mean(dels_aj(1:6,i));
    stddels(:,i) = std(dels_aj(1:6,i));
    middenrand1(:,i) = median(denrand1(1:6,i));
    meandenrand1(:,i) = mean(denrand1(1:6,i));
    stddenrand1(:,i) = std(denrand1(1:6,i));
    middenrand2(:,i) = median(denrand2(1:6,i));
    meandenrand2(:,i) = mean(denrand2(1:6,i));
    stddenrand2(:,i) = std(denrand2(1:6,i));
    midipop1(:,i) = median(ipop1(1:6,i));
    meanipop1(:,i) = mean(ipop1(1:6,i));
    stdipop1(:,i) = std(ipop1(1:6,i));
    midnea1(:,i) = median(nea1(1:6,i));
    meannea1(:,i) = mean(nea1(1:6,i));
    stdnea1(:,i) = std(nea1(1:6,i));
    midnea2(:,i) = median(nea2(1:6,i));
    meannea2(:,i) = mean(nea2(1:6,i));
    stdnea2(:,i) = std(nea2(1:6,i));
    midMolina(:,i) = median(Molina(1:6,i));
    meanMolina(:,i) = mean(Molina(1:6,i));
    stdMolina(:,i) = std(Molina(1:6,i));
    midPNANSGAII(:,i) = median(PNANSGAII(1:6,i));
    meanPNANSGAII(:,i) = mean(PNANSGAII(1:6,i));
    stdPNANSGAII(:,i) = std(PNANSGAII(1:6,i));
    midBA(:,i) = median(BA(1:6,i));
    meanBA(:,i) = mean(BA(1:6,i));
    stdBA(:,i) = std(BA(1:6,i));
    midNSBA(:,i) = median(NSBA(1:6,i));
    meanNSBA(:,i) = mean(NSBA(1:6,i));
    stdNSBA(:,i) = std(NSBA(1:6,i));
    midNRBA(:,i) = median(NRBA(1:6,i));
    meanNRBA(:,i) = mean(NRBA(1:6,i));
    stdNRBA(:,i) = std(NRBA(1:6,i));
    midDNRBA(:,i) = median(DNRBA(1:6,i));
    meanDNRBA(:,i) = mean(DNRBA(1:6,i));
    stdDNRBA(:,i) = std(DNRBA(1:6,i));
end

mmNSGAII = median(midNSGAII);
mNSGAII = mean(meanNSGAII);
sNSGAII = std(stdNSGAII);
mmcma1 = median(midcma1);
mcma1 = mean(meancma1);
scma1 = std(stdcma1);
mmCDE = median(midCDE);
mCDE = mean(meanCDE);
sCDE = std(stdCDE);
mmdade1 = median(middade1);
mdade1 = mean(meandade1);
sdade1 = std(stddade1);
mmdade2 = median(middade2);
mdade2 = mean(meandade2);
sdade2 = std(stddade2);
mmdecg = median(middecg);
mdecg = mean(meandecg);
sdecg = std(stddecg);
mmdelg = median(middelg);
mdelg = mean(meandelg);
sdelg = std(stddelg);
mmdels = median(middels);
mdels = mean(meandels);
sdels = std(stddels);
mmdenrand1 = median(middenrand1);
mdenrand1 = mean(meandenrand1);
sdenrand1 = std(stddenrand1);
mmdenrand2 = median(middenrand2);
mdenrand2 = mean(meandenrand2);
sdenrand2 = std(stddenrand2);
mmpop = median(midipop1);
mpop = mean(meanipop1);
spop = std(stdipop1);
mmnea1 = median(midnea1);
mnea1 = mean(meannea1);
snea1 = std(stdnea1);
mmnea2 = median(midnea2);
mnea2 = mean(meannea2);
snea2 = std(stdnea2);
mmMolina = median(midMolina);
mMolina = mean(meanMolina);
sMolina = std(stdMolina);
mmPNANSGAII = median(midPNANSGAII);
mPNANSGAII = mean(meanPNANSGAII);
sPNANSGAII = std(stdPNANSGAII);
mmBA = median(midBA);
mBA = mean(meanBA);
sBA = std(stdBA);
mmNSBA = median(midNSBA);
mNSBA = mean(meanNSBA);
sNSBA = std(stdNSBA);
mmNRBA = median(midNRBA);
mNRBA = mean(meanNRBA);
sNRBA = std(stdNRBA);
mmDNRBA = median(midDNRBA);
mDNRBA = mean(meanDNRBA);
sDNRBA = std(stdDNRBA);

output = [mmNSGAII, mNSGAII, sNSGAII; mmcma1, mcma1, scma1; mmCDE, mCDE, sCDE; mmdade1, mdade1, sdade1; mmdade2, mdade2, sdade2; mmdecg, mdecg, sdecg; mmdelg, mdelg, sdelg; mmdels, mdels, sdels; mmdenrand1, mdenrand1, sdenrand1; mmdenrand2, mdenrand2, sdenrand2; mmpop, mpop, spop; mmnea1, mnea1, snea1; mmnea2, mnea2, snea2; mmMolina, mMolina, sMolina; mmPNANSGAII, mPNANSGAII, sPNANSGAII; mmBA, mBA, sBA; mmNSBA, mNSBA, sNSBA; mmNRBA, mNRBA, sNRBA; mmDNRBA, mDNRBA, sDNRBA];
% out1 = friedman(output(1:18,:));

figure;
E1 = horzcat(BA(:,1),NSBA(:,1),NRBA(:,1),DNRBA(:,1));
heatmap({'BA','NSBA','NRBA','DNRBA'},{'1','2','3','4','5','6'},E1,'Colormap',parula);
saveas(gcf,[outPath 'E-1_1.0.5.png']);

% figure;
% E1 = horzcat(NSGAII(1:6,1),cma1(1:6,1),CDE(1:6,1),dade1(1:6,1),dade2(1:6,1),decg(1:6,1),delg(1:6,1),dels_aj(1:6,1),denrand1(1:6,1),denrand2(1:6,1),ipop1(1:6,1),nea1(1:6,1),nea2(1:6,1),Molina(1:6,1),PNANSGAII(1:6,1),NSBA(:,1),NRBA(:,1),DNRBA(:,1));
% heatmap(xVal,yVal,E1,'Colormap',parula);
% saveas(gcf,[outPath 'E-1.png']);
% 
% figure;
% E2 = horzcat(NSGAII(1:6,2),cma1(1:6,2),CDE(1:6,2),dade1(1:6,2),dade2(1:6,2),decg(1:6,2),delg(1:6,2),dels_aj(1:6,2),denrand1(1:6,2),denrand2(1:6,2),ipop1(1:6,2),nea1(1:6,2),nea2(1:6,2),Molina(1:6,2),PNANSGAII(1:6,2),NSBA(:,2),NRBA(:,2),DNRBA(:,2));
% heatmap(xVal,yVal,E2,'Colormap',parula);
% saveas(gcf,[outPath 'E-2.png']);
% close;
% 
% figure;
% E3 = horzcat(NSGAII(1:6,3),cma1(1:6,3),CDE(1:6,3),dade1(1:6,3),dade2(1:6,3),decg(1:6,3),delg(1:6,3),dels_aj(1:6,3),denrand1(1:6,3),denrand2(1:6,3),ipop1(1:6,3),nea1(1:6,3),nea2(1:6,3),Molina(1:6,3),PNANSGAII(1:6,3),NSBA(:,3),NRBA(:,3),DNRBA(:,3));
% heatmap(xVal,yVal,E3,'Colormap',parula);
% saveas(gcf,[outPath 'E-3.png']);
% close;
% 
% figure;
% E4 = horzcat(NSGAII(1:6,4),cma1(1:6,4),CDE(1:6,4),dade1(1:6,4),dade2(1:6,4),decg(1:6,4),delg(1:6,4),dels_aj(1:6,4),denrand1(1:6,4),denrand2(1:6,4),ipop1(1:6,4),nea1(1:6,4),nea2(1:6,4),Molina(1:6,4),PNANSGAII(1:6,4),NSBA(:,4),NRBA(:,4),DNRBA(:,4));
% heatmap(xVal,yVal,E4,'Colormap',parula);
% saveas(gcf,[outPath 'E-4.png']);
% close;
% 
% figure;
% E5 = horzcat(NSGAII(1:6,5),cma1(1:6,5),CDE(1:6,5),dade1(1:6,5),dade2(1:6,5),decg(1:6,5),delg(1:6,5),dels_aj(1:6,5),denrand1(1:6,5),denrand2(1:6,5),ipop1(1:6,5),nea1(1:6,5),nea2(1:6,5),Molina(1:6,5),PNANSGAII(1:6,5),NSBA(:,5),NRBA(:,5),DNRBA(:,5));
% heatmap(xVal,yVal,E5,'Colormap',parula);
% saveas(gcf,[outPath 'E-5.png']);
% close;

fprintf('DONE!!!\n');