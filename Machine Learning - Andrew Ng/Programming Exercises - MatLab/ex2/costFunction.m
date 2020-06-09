function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

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

% Step 10
J = unregularized_cost;


% Gradient

% Step 2
unregularized_gradients = (1/m)*(X' * (h-y));
% Step 3
theta(1)=0;
% Step 4
% Step 5
results = unregularized_gradients;
grad = results;
% =============================================================

end
