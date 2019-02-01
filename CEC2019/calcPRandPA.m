clc;
clear;
close all;

path = 'C:\Users\TakuyaIwase\Documents\MATLAB\bat_algorithm\CEC&EvoCOP\NSBA\';
MaxRun = 30;
fncNum = [4 6 7 9];

acc_0 = 1.0;
acc_1 = 0.1;
acc_2 = 0.01;
cnt_0 = zeros(MaxRun,1);
cnt_1 = zeros(MaxRun,1);
cnt_2 = zeros(MaxRun,1);

PR_all = zeros(length(fncNum), 6);
PA_all = zeros(length(fncNum), 6);

for i = 1:4
    for seed = 1:MaxRun
        M = csvread([path 'F' num2str(i) '\F' num2str(i) '_' num2str(seed) '_pbest.csv']);
        pop = M(:,1:2);
        
        [cnt_0(seed,:), goptima_found0] = count_goptima(pop, fncNum(i), acc_0);
        [cnt_1(seed,:), goptima_found1] = count_goptima(pop, fncNum(i), acc_1);
        [cnt_2(seed,:), goptima_found2] = count_goptima(pop, fncNum(i), acc_2);
        
        if length(goptima_found0) ~= get_no_goptima(fncNum(i))
            for j = 1:get_no_goptima(fncNum(i))
                if length(goptima_found0) == get_no_goptima(fncNum(i))
                    break;
                end
                goptima_found0 = [goptima_found0; NaN(1,2)];
            end
        end
        if length(goptima_found1) ~= get_no_goptima(fncNum(i))
            for j = 1:get_no_goptima(fncNum(i))
                if length(goptima_found1) == get_no_goptima(fncNum(i))
                    break;
                end
                goptima_found1 = [goptima_found1; NaN(1,2)];
            end
        end
        if length(goptima_found2) ~= get_no_goptima(fncNum(i))
            for j = 1:get_no_goptima(fncNum(i))
                if length(goptima_found2) == get_no_goptima(fncNum(i))
                    break;
                end
                goptima_found2 = [goptima_found2; NaN(1,2)];
            end
        end
        diff0 = zeros(get_no_goptima(fncNum(i)),1);
        diff1 = zeros(get_no_goptima(fncNum(i)),1);
        diff2 = zeros(get_no_goptima(fncNum(i)),1);
        
        for j = 1:get_no_goptima(fncNum(i))
            if isnan(goptima_found0(j,:)) == 1
                continue;
            end
            diff0(j,:) = abs( get_fgoptima(fncNum(i)) - niching_func(goptima_found0(j,:), fncNum(i)) );
            if isnan(goptima_found1(j,:)) == 1
                continue;
            end
            diff1(j,:) = abs( get_fgoptima(fncNum(i)) - niching_func(goptima_found1(j,:), fncNum(i)) );
            if isnan(goptima_found2(j,:)) == 1
                continue;
            end
            diff2(j,:) = abs( get_fgoptima(fncNum(i)) - niching_func(goptima_found2(j,:), fncNum(i)) );
        end
        
        total_diff0(seed,:) = sum(diff0);
        total_diff1(seed,:) = sum(diff1);
        total_diff2(seed,:) = sum(diff2);
    end
    
    PRmean0 = mean(cnt_0);
    PRstd0 = std(cnt_0);
    PRmean1 = mean(cnt_1);
    PRstd1 = std(cnt_1);
    PRmean2 = mean(cnt_2);
    PRstd2 = std(cnt_2);
    
    PAmean0 = mean(total_diff0);
    PAstd0 = std(total_diff0);
    PAmean1 = mean(total_diff1);
    PAstd1 = std(total_diff1);
    PAmean2 = mean(total_diff2);
    PAstd2 = std(total_diff2);
    
    PR_all(i,:) = horzcat(PRmean0, PRstd0, PRmean1, PRstd1, PRmean2, PRstd2);
    PA_all(i,:) = horzcat(PAmean0, PAstd0, PAmean1, PAstd1, PAmean2, PAstd2);
end

csvwrite([path 'results\PR.csv'], PR_all);
csvwrite([path 'results\PA.csv'], PA_all);
fprintf('DONE!!!\n');