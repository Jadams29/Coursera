function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.


% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%

% ---------------------- Sample Solution ----------------------

% FORMULA
%   ((X'X)^-1) X'y
%
%

part1 = (X'*X);
part2 = X'*y;
theta = part1 \ part2;


% -------------------------------------------------------------


% ============================================================

end
