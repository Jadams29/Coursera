function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Step 1
h = sigmoid(X*theta);
% Step 3
red_circle = -y' * log(h);
% Step 4
blue_circle = (1-y)' * log(1-h);
% Step 5 & 6
unregularized_cost = (1/m)*(red_circle-blue_circle);
% Step 7
theta(1)=0;
% Step 8
sum_of_squares = theta' * theta;
% Step 9
regularized_cost = (lambda/(2*m)) * sum_of_squares;
% Step 10
J = unregularized_cost + regularized_cost;


% Gradient

% Step 2
unregularized_gradients = (1/m) * (X' * (h-y));
% Step 3
theta(1)=0;
% Step 4
regularized_gradients = (lambda/m)*theta;
% Step 5
grad = unregularized_gradients + regularized_gradients;

% =============================================================

end
