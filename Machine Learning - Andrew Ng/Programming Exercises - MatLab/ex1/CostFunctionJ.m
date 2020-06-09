function J = CostFunctionJ(X, y, theta)
%COSTFUNCTIONJ Summary of this function goes here
%   Detailed explanation goes here

rows = size(X,1);   % The size function takes in the matrix and if size(X) will return both number of rows and columns. If you specify which returned value you want size(X,1) you can get just rows, or size(X,2) to get just columns.
predictions = X*theta;  % Multiply X by theta
sqrErrors = (predictions-y).^2; % (Predictions - y) then ELEMENT-WISE squaring

J = 1/(2*rows) * sum(sqrErrors);    % 
end

