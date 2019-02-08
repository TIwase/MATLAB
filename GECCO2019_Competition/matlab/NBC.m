function results = NBC(pop,fitness,NP)
% Connect all edges between nearest better individuals
for i = 1:NP
    dist = zeros(NP,1);         
    for j = 1:NP
        dist(j,:) = norm(pop(i,:) - pop(j,:));
    end

    [dist, dsortID] = sort(dist);
    dmin = dist(2); % minimum distance
    flag = 0;
    
    for j = 2:NP
        if fitness(i,:) < fitness(dsortID(j),:)
            dmin = dist(j,:);
            flag = 1;
            break;
        end
    end
    if flag == 1 
        dominated_point(i,:) = dsortID(j,:);
        edge(i,:) = dmin;
    else
        dominated_point(i,:) = NaN;
        edge(i,:) = 0;
    end
end
%  Delete redundant edges 
edge_mean = mean(edge);
k = 1;
for i = 1:NP
    if edge(i,:) > edge_mean
        edge(i,:) = NaN;
        dominated_point(i,:) = NaN;
    end
    if isnan(dominated_point(i,:)) == 1
        lbestID(k,:) = i;
        k = k + 1;
    end
end
lb_size = NP;
n = 1;
% Divide individuals into clusters
for i = 1:NP
    clst = [];
    while isnan(dominated_point(n,:)) == 0
        clst = [clst n];
        n = dominated_point(n,:);
        if contains(num2str(ismember(clst, n)), '1') == 1
            break;
        end
    end
    clst = [clst n];
    clst_all(i,:) = [clst, zeros(1, lb_size - length(clst))];
    n = i + 1;

end
% Delete duplication clusters
n = 1;
for i = 1:NP
    flag = 0;
    rowNum = find(clst_all(i,:));
    for j = 1:NP
        curNum = find(clst_all(j,:));
        if i == j
            continue;

        elseif ismember(clst_all(i, 1:length(rowNum)), clst_all(j, 1:length(curNum))) == 1
            flag = 1;

        end
    end
    if flag == 0
        dup_clst(n,:) = clst_all(i,:);
        n = n + 1;
    end
end
% Integrate branched clusters
[row, col] = size(dup_clst);
for i = 1:row
    if row < i
        break;
    end
    
    flag = 0;
    for j = i + 1:row
        if row < j
            break;
        end

        if contains( num2str(ismember( dup_clst(i,1:length(find(dup_clst(i,:)))), dup_clst(j,1:length(find(dup_clst(j,:)))) ) ), '1') == 1
            tmp_row = unique([dup_clst(i,1:length(find(dup_clst(i,:)))), dup_clst(j,1:length(find(dup_clst(j,:))))]);
            dup_clst(i,:) = [tmp_row, zeros(1, lb_size - length(tmp_row))];
            dup_clst(j,:) = [];
            merged_clst(i,:) = dup_clst(i,:);
            [row, col] = size(dup_clst);
            flag = 1;
        end
    end

    if flag == 0
        merged_clst(i,:) = dup_clst(i,:);
        [row, col] = size(dup_clst);
    end
end
results = merged_clst;
end