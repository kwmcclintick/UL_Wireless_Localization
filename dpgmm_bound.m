clear all;
clc;
close all;

k = 10; % number of active clusters
n = 100; % number of samples
alpha = 1*(1/k);


% Generate observations


% Slice sampler (Kalli, Griffin, Walker 2011)
M=30000; % number of sampler iterations
for i=1:M
    % Step 1: pi(mu_j, sigma_j | ...)
    
    % Step 2: pi(lambda_j)
    
    % Step 3: pi(mu_i | ...)
    
    % Step 4: P(d_i = k | ...)
    
    % Step 5: pi(v)
    
end





function r = drchrnd(a,n)
% take a sample from a dirichlet distribution
p = length(a);
r = gamrnd(repmat(a,n,1),1,n,p);
r = r ./ repmat(sum(r,2),1,p);
end
