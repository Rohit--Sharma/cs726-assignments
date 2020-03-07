% CS726 - Nonlinear Optimization
%   HW3
%   Author: Rohit Sharma (rohit.sharma@wisc.edu)

function optimization(n, iter)
% OPTIMIZATION: Runs Steepest Descent variants (const step size,
%   exact line search, lagged step size) and Nesterov's method  
%   on a quadratic function defined by M of dimension nxn and b 
%   of dimension n for `iter' iterations.

    [M, b] = initializeMatrix(n);
    
    % Get L, f(x*)
    evals = sort(eig(M));
    L = evals(n);       % We can also prove L = 4 in this case
    f_optimal = minimize(M, b);
    
    % SD:Const initializations
    sdconst_x_k = zeros(n, 1);
    sdconst_opt_gap = [];
    
    % SD:Exact initializations
    sdexact_x_k = zeros(n, 1);
    sdexact_opt_gap = [];
    
    % SD:Lagged initializations
    sdlagged_x_k = zeros(n, 1);
    step_k = 1 / L;     % initialization of prev step size for lagged line search
    sdlagged_opt_gap = [];
    sdlagged_lowest_f = [];
    
    % Nesterov's Acceleration initializations:
    nesterov_x_k = zeros(n, 1);
    A_k = 1;
    v_k = nesterov_x_k - gradient(M, b, nesterov_x_k) / L;
    y_k = v_k;
    nesterov_opt_gap = [];
    nesterov_lowest_f = [];
    
    % Optimization loop
    for k = 1 : iter
        % Run SD:Const step
        sdconst_x_k = steepestDescent(sdconst_x_k, M, b, L);
        steep_desc_f = evaluate_func(M, b, sdconst_x_k);
        sdconst_opt_gap = [sdconst_opt_gap, steep_desc_f - f_optimal];
        
        % Run SD:Exact step
        sdexact_x_k = steepestDescentLineSearch(sdexact_x_k, M, b);
        line_search_f = evaluate_func(M, b, sdexact_x_k);
        sdexact_opt_gap = [sdexact_opt_gap, line_search_f - f_optimal];
        
        % Run SD:Lagged step
        [sdlagged_x_k, step_k] = laggedSteepestDescent(sdlagged_x_k, M, b, step_k);
        lagged_line_search_f = evaluate_func(M, b, sdlagged_x_k);
        sdlagged_opt_gap = [sdlagged_opt_gap, lagged_line_search_f - f_optimal];
        if isempty(sdlagged_lowest_f)
            sdlagged_lowest_f = lagged_line_search_f;
        else
            sdlagged_lowest_f = [sdlagged_lowest_f, min(sdlagged_lowest_f(end), lagged_line_search_f)];
        end
        
        % Run Nesterov's acceleration step
        [y_k, v_k, A_k] = nesterovsMethod(y_k, v_k, M, b, L, A_k);
        nesterov_f = evaluate_func(M, b, y_k);
        nesterov_opt_gap = [nesterov_opt_gap, nesterov_f - f_optimal];
        if isempty(nesterov_lowest_f)
            nesterov_lowest_f = nesterov_f;
        else
            nesterov_lowest_f = [nesterov_lowest_f, min(nesterov_lowest_f(end), nesterov_f)];
        end
    end
    
    % Plot part (i): Optimality gap for SD:Const, SD:Exact and Nesterov
    figure
    plot(1:1:iter, sdconst_opt_gap)
    set(gca, 'YScale', 'log')
    hold on
    plot(1:1:iter, sdexact_opt_gap)
    hold on
    plot(1:1:iter, nesterov_opt_gap)
    legend('SD:constant', 'SD:exact', 'Nesterov')
    title('Analysis of Optimization algorithms')
    xlabel('Num iterations')
    ylabel('Optimality gap: f(x) - f(x*)');
    
    % Plot part (ii): Optimality gap for Nesterov, SD:Lagged
    figure
    plot(1:1:iter, nesterov_opt_gap)
    set(gca, 'YScale', 'log')
    hold on
    plot(1:1:iter, sdlagged_opt_gap)
    legend('Nesterov', 'SD:lagged')
    title('Analysis of Optimization algorithms')
    xlabel('Num iterations')
    ylabel('Optimality gap: f(x) - f(x*)');
    
    % Plot part (iii): Lowest function value for Nesterov, SD:Lagged
    figure
    plot(1:1:iter, nesterov_lowest_f)
    set(gca, 'YScale', 'log')
    hold on
    plot(1:1:iter, sdlagged_lowest_f)
    legend('Nesterov', 'SD:lagged')
    title('Analysis of Optimization algorithms')
    xlabel('Num iterations')
    ylabel('Lowest f value attained');
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

% % Run one step of steepest descent with constant step size = 1/L
% function x_k = steepestDescent(x_k_prev, M, b, L)
%     x_k = x_k_prev - (1 / L) * gradient(M, b, x_k_prev);
% end
% 
% % Run one step of steepest descent with exact line search
% function x_k = steepestDescentLineSearch(x_k_prev, M, b)
%     desc_dirxn = -gradient(M, b, x_k_prev);
%     % Compute the step size using exact line search
%     step_size = dot(desc_dirxn, desc_dirxn) / dot(M*desc_dirxn, desc_dirxn);
%     x_k = x_k_prev + step_size * desc_dirxn;
% end
% 
% % Run one step of steepest descent with a lagged step size
% function [x_k, step_k] = laggedSteepestDescent(x_k_prev, M, b, step_k_prev)
%     desc_dirxn = -gradient(M, b, x_k_prev);
%     % Compute new step size
%     step_k = dot(desc_dirxn, desc_dirxn) / dot(M*desc_dirxn, desc_dirxn);
%     % Use old step size to update x as per steepest descent
%     x_k = x_k_prev + step_k_prev * desc_dirxn;
% end

% Run one step of Nesterov's acceleration
function [y_k, v_k, A_k] = nesterovsMethod(y_k_prev, v_k_prev, M, b, L, A_k_prev)
    a_k = (1 + sqrt(1 + 4 * A_k_prev)) / 2;
    A_k = A_k_prev + a_k;
    
    x_k = (A_k_prev / A_k) * y_k_prev + (a_k / A_k) * v_k_prev;
    grad_x_k = gradient(M, b, x_k);
    v_k = v_k_prev - (a_k / L) * grad_x_k;
    y_k = x_k - grad_x_k / L;
end

function nesterovForSmoothStronglyConvex()
    % Restarting AGD for n_iter = sqrt(8L/m)
    n_iter_nesterov = ceil(sqrt(8 * L / m));
    
    for k = 1 : n_iter_nesterov
        [y_k, v_k, A_k] = nesterovsMethod(y_k, v_k, M, b, L, A_k);
    end
end

function [x_k, p_k] = conjugateGradientMethod(M, b, x_k_prev, p_k_prev)
    gradient_x_k_prev = gradient(M, b, x_k_prev);
    
    % Exact line search for h_k_prev
    h_k_prev = - dot(gradient_x_k_prev, -p_k_prev) / dot(-M*p_k_prev, -p_k_prev);
    x_k = x_k_prev - h_k_prev * p_k_prev;
    
    beta_k_prev = dot(gradient_x_k_prev, gradient_x_k_prev) / dot(gradient(M, b, x_k) - gradient_x_k_prev, p_k_prev);
    p_k = gradient_x_k_prev - beta_k_prev * p_k_prev;
end

function x_k = heavyBallMethod(x_k_prev, x_k_prev_prev, M, b, L, m)
    alpha_1 = 4 / (sqrt(L) + sqrt(m))^2;
    alpha_2 = (sqrt(L) - sqrt(m))^2 / (sqrt(L) + sqrt(m))^2;
    
    x_k = x_k_prev - alpha_1 * gradient(M, b, x_k_prev) + alpha_2 * (x_k_prev - x_k_prev_prev);
end

% Helper method to evaluate the value of function at a given input
function f_val = evaluate_func(M, b, m, x)
    f_val = (1/2) * dot(M*x, x) - dot(b, x) + m / 2 * norm(x)^2;
end

% Helper method to evaluate the gradient of function at a given input
function grad = gradient(M, b, m, x)
    grad = M * x - b + m * x;
end

% Helper method to obtain the optimal value of the function (f(x*))
function f_optimal = minimize(M, b, m)
    n = size(M, 1);
    x_optimal = pinv(M + m * eye(n)) * b;
    f_optimal = evaluate_func(M, b, m, x_optimal);
end