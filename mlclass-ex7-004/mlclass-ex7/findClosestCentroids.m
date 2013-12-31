function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

m = length(idx);
for i = 1:m
	p1 = X(i,:);
	c = centroids(1,:);
	dist = sum((p1-c).^2);
	idx(i) = 1;
	for j = 2:K
		c1 = centroids(j,:);
		dd = sum((p1-c1).^2);
		if dd < dist
			idx(i) = j;
			dist = dd;
		end
	end
end




% =============================================================

end

