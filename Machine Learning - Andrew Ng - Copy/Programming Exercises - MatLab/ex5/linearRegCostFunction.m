function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

predictions = X*theta;  % Multiply X by theta
sqrErrors = (predictions-y).^2; % (Predictions - y) then ELEMENT-WISE squaring

unregularized_cost = 1/(2*m) * sum(sqrErrors);    % 

% STEP 1:
h = X * theta;

% STEP 2:
errors = (h-y);

% STEP 3:    Theta_change is the gradient
unregularized_gradients = ((1/m)*(X' * errors));



theta(1) = 0;

sum_square_error = sum(theta.^2);
regularized_cost = ((lambda)/(2*m)) * sum_square_error;

% Step 10
J = unregularized_cost + regularized_cost;



% Gradient 

% Step 4
regularized_gradients = (lambda/m)*theta;
% Step 5
grad = unregularized_gradients + regularized_gradients;


% =========================================================================

grad = grad(:);

end
