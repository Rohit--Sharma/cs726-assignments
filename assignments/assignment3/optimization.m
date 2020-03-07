% CS726 - Nonlinear Optimization
%   HW3
%   Author: Rohit Sharma (rohit.sharma@wisc.edu)

function optimization(n, m, iter)
% OPTIMIZATION: Runs Steepest Descent variants (const step size,
%   exact line search, lagged step size) and Nesterov's method  
%   on a quadratic function defined by M of dimension nxn and b 
%   of dimension n for `iter' iterations.

    [M, b] = initializeMatrix(n);
    
    % Get L, f(x*)
%     evals = sort(eig(M));
%     L = evals(n);       % We can also prove L = 4 in this case
    L = 4 + m;
    f_optimal = minimize(M, b, m);
    
    % Nesterov's Acceleration for smooth initializations:
    nesterov_x_k = zeros(n, 1);
    nesterov_A_k = 1;
    nesterov_v_k = nesterov_x_k - gradient(M, b, m, nesterov_x_k) / L;
    nesterov_y_k = nesterov_v_k;
    nesterov_opt_gap = [];
    
    % Nesterov's Acceleration for smooth monotonically decr initializations:
    nestmono_x_k = zeros(n, 1);
    nestmono_A_k = 1;
    nestmono_v_k = nestmono_x_k - gradient(M, b, m, nestmono_x_k) / L;
    nestmono_y_k = nestmono_v_k;
    nestmono_opt_gap = [];
    
    % Nesterov's Acceleration for smooth and strongly convex initializations:
    strong_nesterov_x_k = zeros(n, 1);
    strong_nesterov_A_k = 1;
    strong_nesterov_v_k = strong_nesterov_x_k - gradient(M, b, m, strong_nesterov_x_k) / L;
    strong_nesterov_y_k = strong_nesterov_v_k;
    strong_nesterov_opt_gap = [];
    
    % Conjugate Gradients initializations:
    cgm_x_k = zeros(n, 1);
    p_k = gradient(M, b, m, cgm_x_k);
    cgm_opt_gap = [];
    
    % Heavy Ball initializations:
    hbm_x_k_prev = zeros(n, 1);
    hbm_x_k = zeros(n, 1);
    hbm_opt_gap = [];
    
    % Optimization loop
    for k = 1 : iter
        % Run Nesterov's acceleration step
        [nesterov_y_k, nesterov_v_k, nesterov_A_k] = nesterovsMethod(nesterov_y_k, nesterov_v_k, M, b, m, L, nesterov_A_k);
        nesterov_f = evaluate_func(M, b, m, nesterov_y_k);
        nesterov_opt_gap = [nesterov_opt_gap, nesterov_f - f_optimal];
        
        % Run Nesterov's acceleration monotonically decreasing step
        [nestmono_y_k, nestmono_v_k, nestmono_A_k] = monotonicallyDecreasingNesterovsMethod(nestmono_y_k, nestmono_v_k, M, b, m, L, nestmono_A_k);
        nestmono_f = evaluate_func(M, b, m, nestmono_y_k);
        nestmono_opt_gap = [nestmono_opt_gap, nestmono_f - f_optimal];
        
        % Run Nesterov's acceleration for strong cvx step
        [strong_nesterov_y_k, strong_nesterov_v_k, strong_nesterov_A_k] = nesterovForSmoothStronglyConvexRechtMethod(strong_nesterov_y_k, strong_nesterov_v_k, M, b, L, m, strong_nesterov_A_k);
        strong_nesterov_f = evaluate_func(M, b, m, strong_nesterov_y_k);
        strong_nesterov_opt_gap = [strong_nesterov_opt_gap, strong_nesterov_f - f_optimal];
        
        % Run CGM step
        [cgm_x_k, p_k] = conjugateGradientMethod(M, b, m, cgm_x_k, p_k);
        cgm_f = evaluate_func(M, b, m, cgm_x_k);
        cgm_opt_gap = [cgm_opt_gap, cgm_f - f_optimal];
        
        % Run Heavy Ball step
        temp = hbm_x_k;
        hbm_x_k = heavyBallMethod(hbm_x_k, hbm_x_k_prev, M, b, L, m);
        hbm_x_k_prev = temp;
        hbm_f = evaluate_func(M, b, m, hbm_x_k);
        hbm_opt_gap = [hbm_opt_gap, hbm_f - f_optimal];
    end
    
    disp(cgm_opt_gap)
    
    % Plot part (i): Optimality gap for SD:Const, SD:Exact and Nesterov
    figure
    plot(1:1:iter, nesterov_opt_gap)
    set(gca, 'YScale', 'log')
    hold on
    plot(1:1:iter, strong_nesterov_opt_gap)
    hold on
    plot(1:1:iter, cgm_opt_gap)
    hold on
    plot(1:1:iter, hbm_opt_gap)
    legend('Nesterov', 'Str:Nesterov', 'CGM', 'HBM')
    title(strcat('Analysis of Optimization algorithms, m=', num2str(m)))
    xlabel('Num iterations')
    ylabel('Optimality gap: f(x) - f(x*)');
    
    figure
    plot(1:1:iter, nestmono_opt_gap)
    set(gca, 'YScale', 'log')
    hold on
    plot(1:1:iter, strong_nesterov_opt_gap)
    hold on
    plot(1:1:iter, cgm_opt_gap)
    hold on
    plot(1:1:iter, hbm_opt_gap)
    legend('Mono:Nesterov', 'Str:Nesterov', 'CGM', 'HBM')
    title(strcat('Analysis of Optimization algorithms, m=', num2str(m)))
    xlabel('Num iterations')
    ylabel('Optimality gap: f(x) - f(x*)');
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

% Run one step of Nesterov's acceleration
function [y_k, v_k, A_k] = nesterovsMethod(y_k_prev, v_k_prev, M, b, m, L, A_k_prev)
    a_k = (1 + sqrt(1 + 4 * A_k_prev)) / 2;
    A_k = A_k_prev + a_k;
    
    x_k = (A_k_prev / A_k) * y_k_prev + (a_k / A_k) * v_k_prev;
    grad_x_k = gradient(M, b, m, x_k);
    v_k = v_k_prev - (a_k / L) * grad_x_k;
    y_k = x_k - grad_x_k / L;
end

function [y_k, v_k, A_k] = monotonicallyDecreasingNesterovsMethod(y_k_prev, v_k_prev, M, b, m, L, A_k_prev)
    a_k = (1 + sqrt(1 + 4 * A_k_prev)) / 2;
    A_k = A_k_prev + a_k;
    
    x_k = (A_k_prev / A_k) * y_k_prev + (a_k / A_k) * v_k_prev;
    grad_x_k = gradient(M, b, m, x_k);
    v_k = v_k_prev - (a_k / L) * grad_x_k;
    
    if evaluate_func(M, b, m, y_k_prev - gradient(M, b, m, y_k_prev) / L) < evaluate_func(M, b, m, x_k - grad_x_k / L)
        y_k = y_k_prev - gradient(M, b, m, y_k_prev) / L;
    else
        y_k = x_k - grad_x_k / L;
    end
end

function nesterovForSmoothStronglyConvex()
    % Restarting AGD for n_iter = sqrt(8L/m)
    n_iter_nesterov = ceil(sqrt(8 * L / m));
    
    for k = 1 : n_iter_nesterov
        [y_k, v_k, A_k] = nesterovsMethod(y_k, v_k, M, b, L, A_k);
    end
end

function [y_k, v_k, A_k] = nesterovForSmoothStronglyConvexRechtMethod(y_k_prev, v_k_prev, M, b, L, m, A_k_prev)
    m0 = L - m;
    c1 = m0 + 2*m*A_k_prev;
    c2 = 4*m0*A_k_prev*(m0 + m*A_k_prev);
    a_k = (c1 + sqrt(c1^2 + c2)) / (2*m0);
    A_k = A_k_prev + a_k;
    
    a_k_ = a_k * (m0 + m*A_k_prev) / (m0 + m*A_k);
    theta_k = a_k_ / (A_k_prev + a_k_);
    
    x_k = (1 - theta_k)*y_k_prev + theta_k*v_k_prev;
    v_k = (m0 + m*A_k_prev) / (m0 + m*A_k) * v_k_prev + (m*a_k) / (m0 + m*A_k) * x_k - a_k / (m0 + m*A_k) * gradient(M, b, m, x_k);
    y_k = x_k - 1 / L * gradient(M, b, m, x_k);
end

function [x_k, p_k] = conjugateGradientMethod(M, b, m, x_k_prev, p_k_prev)
    n = size(M, 1);
    gradient_x_k_prev = gradient(M, b, m, x_k_prev);
    
    % Exact line search for h_k_prev
    h_k_prev = dot(gradient_x_k_prev, p_k_prev) / dot((M + m*eye(n))*p_k_prev, p_k_prev);
    x_k = x_k_prev - h_k_prev * p_k_prev;
    
    beta_k_prev = dot(gradient_x_k_prev, gradient_x_k_prev) / dot(gradient(M, b, m, x_k) - gradient_x_k_prev, p_k_prev);
    p_k = gradient_x_k_prev - beta_k_prev * p_k_prev;
end

function x_k = heavyBallMethod(x_k_prev, x_k_prev_prev, M, b, L, m)
    alpha_1 = 4 / (sqrt(L) + sqrt(m))^2;
    alpha_2 = (sqrt(L) - sqrt(m))^2 / (sqrt(L) + sqrt(m))^2;
    
    x_k = x_k_prev - alpha_1 * gradient(M, b, m, x_k_prev) + alpha_2 * (x_k_prev - x_k_prev_prev);
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
