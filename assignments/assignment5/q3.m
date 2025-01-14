% CS726 - Nonlinear Optimization
%   HW5
%   Author: Rohit Sharma (rohit.sharma@wisc.edu)
%
%Q3: Runs Nesterov's AGD, Steepest Descent and SGD
%   on a quadratic function with M or dimensions `nxn`, 
%   x of dimension `n` for `n_iter' iterations with a
%   noisy gradient with noise parameter `epsilon`.

function q3(n, n_iter, epsilon)
%Q3: Runs Nesterov's AGD, Steepest Descent and SGD
%   on a quadratic function with M or dimensions `nxn`, 
%   x of dimension `n` for `n_iter' iterations with a
%   noisy gradient with noise parameter `epsilon`.

    rng('default'); % For reproducibility
    
    [M, b] = initializeMatrix(n);
    
    % Get L, f(x*)
    L = 4;
    f_optimal = minimize(M, b);
    
    % Nesterov's Acceleration for smooth initializations:
    nesterov_x_k = zeros(n, 1);
    nesterov_A_k = 1;
    nesterov_v_k = nesterov_x_k - gradient(M, b, epsilon, nesterov_x_k) / L;
    nesterov_y_k = nesterov_v_k;
    nesterov_opt_gap = [];
    
    % SD:Const initializations
    sdconst_x_k = zeros(n, 1);
    sdconst_opt_gap = [];
    
    % SGD initializations:
    sgd_x_k = zeros(n, 1);
    sgd_x_k_out = sgd_x_k;
    sgd_A_k = 1/L;
    sgd_opt_gap = [];
    
    % Optimization loop
    for k = 1 : n_iter
        % Run Nesterov's acceleration step
        [nesterov_y_k, nesterov_v_k, nesterov_A_k] = nesterovsMethod(nesterov_y_k, nesterov_v_k, M, b, L, nesterov_A_k, epsilon);
        nesterov_f = evaluate_func(M, b, nesterov_y_k);
        nesterov_opt_gap = [nesterov_opt_gap, nesterov_f - f_optimal];
        
        % Run SD:Const step
        sdconst_x_k = steepestDescent(sdconst_x_k, M, b, L, epsilon);
        steep_desc_f = evaluate_func(M, b, sdconst_x_k);
        sdconst_opt_gap = [sdconst_opt_gap, steep_desc_f - f_optimal];
        
        % Run SGD step
        [sgd_x_k_out, sgd_x_k, sgd_A_k] = projectedSGDWithAvg(sgd_x_k_out, sgd_x_k, M, b, L, k, sgd_A_k, epsilon);
        sgd_f = evaluate_func(M, b, sgd_x_k_out);
        sgd_opt_gap = [sgd_opt_gap, sgd_f - f_optimal];
    end
    
    % Plot Optimality gap for SD, Nesterov and SGD
    figure
    plot(1:1:n_iter, sdconst_opt_gap)
    set(gca, 'YScale', 'log')
    hold on
    plot(1:1:n_iter, nesterov_opt_gap)
    hold on
    plot(1:1:n_iter, sgd_opt_gap)
    legend('SD', 'Nesterov', 'SGD')
    title(strcat('Analysis of Optimization algorithms, eps=', num2str(epsilon)))
    xlabel('Num iterations')
    ylabel('Optimality gap: f(x) - f(x*)');
end

% Run one step of steepest descent with constant step size = 1/L
function x_k = steepestDescent(x_k_prev, M, b, L, epsilon)
    x_k = x_k_prev - (1 / L) * gradient(M, b, epsilon, x_k_prev);
end

% Run one step of Nesterov's acceleration for smooth f
function [y_k, v_k, A_k] = nesterovsMethod(y_k_prev, v_k_prev, M, b, L, A_k_prev, epsilon)
    a_k = (1 + sqrt(1 + 4 * A_k_prev)) / 2;
    A_k = A_k_prev + a_k;
    
    x_k = (A_k_prev / A_k) * y_k_prev + (a_k / A_k) * v_k_prev;
    grad_x_k = gradient(M, b, epsilon, x_k);
    v_k = v_k_prev - (a_k / L) * grad_x_k;
    y_k = x_k - grad_x_k / L;
end

% Get the next iterates of SGD
function [x_k_out, x_k, A_k] = projectedSGDWithAvg(x_k_out_prev, x_k_prev, M, b, L, k_prev, A_k_prev, epsilon)
    a_k = 1 / (L*sqrt(k_prev+1));
    A_k = A_k_prev + a_k;
    
    x_k = x_k_prev - a_k * gradient(M, b, epsilon, x_k_prev);
    x_k_out = 1/A_k * (a_k*x_k + A_k_prev*x_k_out_prev);
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
    [n, ~] = size(b);
    grad = M * x - b + epsilon*normrnd(0, 1, [n, 1]);
end

% Helper method to obtain the optimal value of the function (f(x*))
function f_optimal = minimize(M, b)
    x_optimal = pinv(M) * b;
    f_optimal = evaluate_func(M, b, x_optimal);
end
