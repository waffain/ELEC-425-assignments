function [membership, centres] = k_medoids(X, n_cluster)
% X: the data matrix, rows are data points and columns are features
% n_cluster: number of cluster

if n_cluster > 4
    disp ('You have set too many clusters.');
    disp ('Set the number of clusters to be 1-4.');
    disp ('The program and visualization allow for up to 4 clusters.');
    return;
end

% Initialize the figure
figure('position', [200, 200, 600, 500]);
    
% Get the number of data points and number of features
[n_sample, n_feat] = size(X); 



% Randomly initialize the starting medoids.
rng('shuffle');
medoid_indices = randperm(n_sample, n_cluster);
centres = X(medoid_indices, :);

disp('Start K-medoids clustering ... ');

% Initialization:
% In the begining, all data points are in cluster 1
% The "old_membership" variable is an n_sample-by-1 matrix.
% It saves the cluster id that each data point belongs to.
% Again, in the begining, all data points are in cluster 1
old_membership = ones(n_sample, 1);

% Display the initial cluster membership for all datapoints
% and the initial cluster centres
show(X, old_membership, n_cluster, centres, 'Cluster centres initialized!')

while true
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % You need to add code here.
    % Calculate cityblock distances
    % between every data point and every cluster centre.
    % Please put your results in an 
    % n_sample-by-n_cluster matrix named as "distance". 
    % You may find the Matlab function pdist2 to be useful here. 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    distance = pdist2(X, centres,"cityblock"); 
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % E step: Assign data points to closest clusters.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [~, membership] = min(distance, [], 2);
    
    %Show the result of the E step.
    show(X, membership, n_cluster, centres, 'E step finished: Datapoints re-assigned!')

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % M step: Update mediods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for j = 1:n_cluster
         cluster_points = X(membership == j, :);
         num_points = size(cluster_points, 1);
         %check empty clusters
         if isempty(cluster_points)
            centres(j, :) = X(randi(n_sample), :);
         else
             mindist = Inf;
             for i = 1:num_points
                 %find the sum of distances between that point and all
                 %other points
                 distance_sum = sum(pdist2(cluster_points(i, :), cluster_points, 'cityblock'));
                 %update medoid
                 if distance_sum < mindist
                     mindist = distance_sum;
                     centres(j, :) = cluster_points(i, :);
                 end
             end
         end
    end
    
    %Show the result of the M step.
    show(X, membership, n_cluster, centres, 'M step finished: Cluster centers updated!')
    
    % Stop if no more updates.
    if sum(membership ~= old_membership)==0
        show(X, membership, n_cluster, centres, 'Done! ');
        break;
    end
    
    old_membership = membership;
end
end

function show(X, c_pred, n_cluster, centres, txt)
    symbol = ['ro'; 'gp'; 'bd'; 'k^'; 'r*'];
    hold off;
        
    for i = 1:n_cluster
        marker = mod(i,5);
        if i > 4            
            disp('Total number of clusters exceeds 4, some symbols in the plot are reused!');
        end
        plot(X(c_pred==i, 1), X(c_pred==i, 2), symbol(marker,:));
        hold on;
        plot(centres(i, 1), centres(i, 2), symbol(marker,2), 'MarkerFaceColor',symbol(marker,1));
    end
    text(4.2, 5.4, txt);
    drawnow;
    
    %Pause some time here.
    %Used to show figure with enough time.
    %You can change the pause time.
    pause(2);
end
