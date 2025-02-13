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

H = X * theta - y; % size 12x1s

J = 1/(2*m) * H'*H; % without regularization

J = J + lambda/2/m * sum(theta(2:end).^2);

% grad = 1/m * (X' * H);
% %grad(1) = grad(1) + lambda/m * theta(1);
% grad(2) = grad(2) + lambda/m * theta(2);

grad(1) = grad(1) + (1/m) * sum(H);
grad(2:end) = grad(2:end) + (1/m) * (X(:,2:end)' * H) + (lambda/m * theta(2:end));







% =========================================================================

grad = grad(:);

end
