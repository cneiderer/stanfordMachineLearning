function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

values = [.01, .03, .1, .3, 1, 3, 10, 30];
C_values = values;
sigma_values = values;
error = [];
% Iterate over possible C values
for C_idx = 1:length(C_values)
    % Iterate over possible sigma values
    for sigma_idx = 1:length(sigma_values)
        fprintf('Train and evaluate model with (C, sigma) = (%f, %f)\n', ...
            C_values(C_idx), sigma_values(sigma_idx));
        % Train model with possible (C, sigma) pair
        model = svmTrain(X, y, C_values(C_idx), ...
            @(x1, x2) gaussianKernel(x1, x2, sigma_values(sigma_idx)));
        % Evaluate model on validation dataset
        pred = svmPredict(model, Xval);
        % Calculate the error between the predictions and actuals
        pred_error = mean(double(pred ~= yval));
        fprintf('Prediction Error: %f\n\n', pred_error);
        % Store (C, sigma) pairs with their associated prediction error
        error = [error; C_values(C_idx), sigma_values(sigma_idx), pred_error];
    end
end
% Find min error index to select the optimal values of C and sigma
[minError, minErrorIdx] = min(error(1:end, 3));
C = error(minErrorIdx, 1);
sigma = error(minErrorIdx, 2);
fprintf('Optimal (C, sigma) pair is (%f, %f) with a prediction error of: %f\n', ...
    C, sigma, minError);

% =========================================================================

end
