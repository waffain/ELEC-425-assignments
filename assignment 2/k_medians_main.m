clear all;

% Load data
load data;

% Perform K-means clustering.
% The function provides visualization 
% to support up to 4 clusters.
k_medians(data, 2);
