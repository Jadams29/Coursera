function [results] = TestingGradientDescent(X,y,theta)
%TESTINGGRADIENTDESCENT Summary of this function goes here
%   Detailed explanation goes here
prediction = theta'*X;

m = size(X,1);
results = ((1/m) * (prediction - y).* X);

end

