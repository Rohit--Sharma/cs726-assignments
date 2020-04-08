function [outputArg1,outputArg2] = q3(n, n_iter, epsilon)
%Q3 Summary of this function goes here
%   Detailed explanation goes here

    rng('default'); % For reproducibility
    
    [M, b] = initializeMatrix(n);
    
    % Get L, f(x*)
%     evals = sort(eig(M));
%     L = evals(n);       % We can also prove L = 4 in this case
    L = 4;
    f_optimal = minimize(M, b);
    
    % Nesterov's Acceleration for smooth initializations:
    nesterov_x_k = zeros(n, 1);
    nesterov_A_k = 1;
    nesterov_v_k = nesterov_x_k - gradient(M, b, epsilon, nesterov_x_k) / L;
    nesterov_y_k = nesterov_v_k;
    nesterov_opt_gap = [];
    
    % Optimization loop
    for k = 1 : n_iter
        % Run Nesterov's acceleration step
        [nesterov_y_k, nesterov_v_k, nesterov_A_k] = nesterovsMethod(nesterov_y_k, nesterov_v_k, M, b, L, nesterov_A_k, epsilon);
        nesterov_f = evaluate_func(M, b, nesterov_y_k);
        nesterov_opt_gap = [nesterov_opt_gap, nesterov_f - f_optimal];
    end
    
    disp(nesterov_opt_gap);
end

% Run one step of steepest descent with constant step size = 1/L
function x_k = steepestDescent(x_k_prev, M, b, L)
    x_k = x_k_prev - (1 / L) * gradient(M, b, x_k_prev);
end

% Run one step of Nesterov's acceleration for smooth f
% TODO: Check if we need to implement Nesterov for smooth and str convex f
function [y_k, v_k, A_k] = nesterovsMethod(y_k_prev, v_k_prev, M, b, L, A_k_prev, epsilon)
    a_k = (1 + sqrt(1 + 4 * A_k_prev)) / 2;
    A_k = A_k_prev + a_k;
    
    x_k = (A_k_prev / A_k) * y_k_prev + (a_k / A_k) * v_k_prev;
    grad_x_k = gradient(M, b, epsilon, x_k);
    v_k = v_k_prev - (a_k / L) * grad_x_k;
    y_k = x_k - grad_x_k / L;
end

% Initialize M(nxn) and b(nx1) as required
function [M, b] = initializeMatrix(n)
    k = n;
    M = diag(2*[ones(k, 1); zeros(n-k, 1)], 0)...
        + diag([-ones(k-1, 1); zeros(n-k, 1)], -1)...
        + diag([-ones(k-1, 1); zeros(n-k, 1)], 1);
    b = zeros(n, 1);
    b(1) = b(1) + 1;
end

% Helper method to evaluate the value of function at a given input
function f_val = evaluate_func(M, b, x)
    f_val = (1/2) * dot(M*x, x) - dot(b, x);
end

% Helper method to evaluate the gradient of function at a given input
function grad = gradient(M, b, epsilon, x)
    grad = M * x - b + epsilon*normrnd(0, 1);
end

% Helper method to obtain the optimal value of the function (f(x*))
function f_optimal = minimize(M, b)
    x_optimal = pinv(M) * b;
    f_optimal = evaluate_func(M, b, x_optimal);
end
