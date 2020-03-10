% CS726 - Nonlinear Optimization
%   HW3
%   Author: Rohit Sharma (rohit.sharma@wisc.edu)

function q4(iter)
% OPTIMIZATION: Runs Steepest Descent variants (const step size,
%   exact line search, lagged step size) and Nesterov's method  
%   on a quadratic function defined by M of dimension nxn and b 
%   of dimension n for `iter' iterations.
    
    % Some function properties
    m = 1;
    L = 25;
    f_optimal = minimize();

    % Nesterov' Method initialization
    nesterov_x_k = 3.3;
    A_k = 1;
    v_k = nesterov_x_k - gradient(nesterov_x_k) / L;
    y_k = v_k;
    nesterov_opt_gap = [];
    
    % Heavy Ball Method initialization
    heavyball_x_k_prev = 3.3;
    heavyball_x_k = 3.3;
    heavyball_opt_gap = [];
    
    for k = 1 : iter
        [y_k, v_k, A_k] = nesterovsMethodForSmoothStronglyConvex(y_k, v_k, L, m, A_k);
        nesterov_f = evaluate_func(y_k);
        nesterov_opt_gap = [nesterov_opt_gap, nesterov_f - f_optimal];
        
        temp = heavyball_x_k;
        heavyball_x_k = heavyBallMethod(heavyball_x_k, heavyball_x_k_prev, L, m);
        heavyball_x_k_prev = temp;
        heavyball_f = evaluate_func(heavyball_x_k);
        heavyball_opt_gap = [heavyball_opt_gap, heavyball_f - f_optimal];
    end
    
    % Plot Optimality gap for Nesterov and Heavy Ball
    figure
    plot(1:1:iter, nesterov_opt_gap, '-o');
    hold on
    plot(1:1:iter, heavyball_opt_gap)
    legend('Nesterov', 'Heavy Ball')
    title('Analysis of Optimization algorithms')
    xlabel('Num iterations')
    ylabel('Optimality gap: f(x) - f(x*)');
end

% Run one step of Nesterov's acceleration for smooth and strongly convex f
function [y_k, v_k, A_k] = nesterovsMethodForSmoothStronglyConvex(y_k_prev, v_k_prev, L, m, A_k_prev)
    m0 = L - m;
    c1 = sqrt(m / L);
    a_k = c1 * A_k_prev / (1 - c1);
    A_k = A_k_prev + a_k;
    
    a_k_ = a_k * (m0 + m*A_k_prev) / (m0 + m*A_k);
    theta_k = a_k_ / (A_k_prev + a_k_);
    
    x_k = (1 - theta_k)*y_k_prev + theta_k*v_k_prev;
    v_k = (m0 + m*A_k_prev) / (m0 + m*A_k) * v_k_prev + (m*a_k) / (m0 + m*A_k) * x_k - a_k / (m0 + m*A_k) * gradient(x_k);
    y_k = x_k - 1 / L * gradient(x_k);
end

% Run one step of Heavy ball method
function x_k = heavyBallMethod(x_k_prev, x_k_prev_prev, L, m)
    alpha_1 = 4 / (sqrt(L) + sqrt(m))^2;
    alpha_2 = (sqrt(L) - sqrt(m))^2 / (sqrt(L) + sqrt(m))^2;
    
    x_k = x_k_prev - alpha_1 * gradient(x_k_prev) + alpha_2 * (x_k_prev - x_k_prev_prev);
end

% Helper method to evaluate the value of function at a given input
function f_val = evaluate_func(x)
    if x < 1
        f_val = 25/2 * x^2;
    elseif x < 2
        f_val = 1/2 * x^2 + 24*x - 12;
    else
        f_val = 25/2 * x^2 - 24*x + 36;
    end
end

% Helper method to evaluate the gradient of function at a given input
function grad = gradient(x)
    if x < 1
        grad = 25*x;
    elseif x < 2
        grad = x + 24;
    else
        grad = 25*x - 24;
    end
end

% Helper method to obtain the optimal value of the function (f(x*))
function f_optimal = minimize()
    x_optimal = 0;
    f_optimal = 0;
end
