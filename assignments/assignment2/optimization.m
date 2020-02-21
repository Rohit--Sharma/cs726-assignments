function optimization
    n = 200;
    [M, b] = initializeMatrix(n);
    
    evals = sort(eig(M));
    L = evals(n);       % We can also prove L = 4 in this case
    
    x_optimal = pinv(M) * b;
%     disp('Global min:');
    f_optimal = (1/2) * dot(M*x_optimal, x_optimal) - dot(b, x_optimal);
%     disp(f_optimal);
    
    iter = 850;
    
    steep_desc_x_k = zeros(n, 1);
    steep_desc_opt_gap = [];
    
    line_search_x_k = zeros(n, 1);
    line_search_opt_gap = [];
    
    lagged_line_search_x_k = zeros(n, 1);
    step_k = 1 / L;     % initialization of prev step size for lagged line search
    lagged_line_search_opt_gap = [];
    lagged_line_search_lowest_f = [];
    
    % Nesterov's Acceleration initializations:
    nest_x_k = zeros(n, 1);
    a_k = 1;
    A_k = 1;
    v_k = nest_x_k - gradient(M, b, nest_x_k) / L;
    y_k = v_k;
    nest_opt_gap = [];
    nest_lowest_f = [];
    
    for k = 1 : iter
        steep_desc_x_k = steepestDescent(steep_desc_x_k, M, b, L);
        steep_desc_f = evaluate_func(M, b, steep_desc_x_k);
        steep_desc_opt_gap = [steep_desc_opt_gap, steep_desc_f - f_optimal];
        
        line_search_x_k = steepestDescentLineSearch(line_search_x_k, M, b);
        line_search_f = evaluate_func(M, b, line_search_x_k);
        line_search_opt_gap = [line_search_opt_gap, line_search_f - f_optimal];
        
        [lagged_line_search_x_k, step_k] = laggedSteepestDescent(lagged_line_search_x_k, M, b, step_k);
        lagged_line_search_f = evaluate_func(M, b, lagged_line_search_x_k);
        lagged_line_search_opt_gap = [lagged_line_search_opt_gap, lagged_line_search_f - f_optimal];
        if isempty(lagged_line_search_lowest_f)
            lagged_line_search_lowest_f = lagged_line_search_f;
        else
            lagged_line_search_lowest_f = [lagged_line_search_lowest_f, min(lagged_line_search_lowest_f(end), lagged_line_search_f)];
        end
        
        [nest_x_k, y_k, v_k, a_k, A_k] = nesterovsMethod(nest_x_k, y_k, v_k, M, b, L, a_k, A_k);
        nest_f = evaluate_func(M, b, y_k);
        nest_opt_gap = [nest_opt_gap, nest_f - f_optimal];
        if isempty(nest_lowest_f)
            nest_lowest_f = nest_f;
        else
            nest_lowest_f = [nest_lowest_f, min(nest_lowest_f(end), nest_f)];
        end
    end
    
    figure
    plot(1:1:iter, steep_desc_opt_gap)
    set(gca, 'YScale', 'log')
    hold on
    plot(1:1:iter, line_search_opt_gap)
    hold on
    plot(1:1:iter, nest_opt_gap)
    legend('SD:constant', 'SD:exact', 'Nesterov')
    title('Analysis of Optimization algorithms')
    xlabel('Num iterations')
    ylabel('Optimality gap: f(x) - f(x*)');
    
    figure
    plot(1:1:iter, nest_opt_gap)
    set(gca, 'YScale', 'log')
    hold on
    plot(1:1:iter, lagged_line_search_opt_gap)
    legend('Nesterov', 'SD:lagged')
    title('Analysis of Optimization algorithms')
    xlabel('Num iterations')
    ylabel('Optimality gap: f(x) - f(x*)');
    
    figure
    plot(1:1:iter, nest_lowest_f)
    set(gca, 'YScale', 'log')
    hold on
    plot(1:1:iter, lagged_line_search_lowest_f)
    legend('Nesterov', 'SD:lagged')
    title('Analysis of Optimization algorithms')
    xlabel('Num iterations')
    ylabel('Lowest f value attained');
end

function [M, b] = initializeMatrix(n)
    k = n;
    M = diag(2*[ones(k, 1); zeros(n-k, 1)], 0)...
        + diag([-ones(k-1, 1); zeros(n-k, 1)], -1)...
        + diag([-ones(k-1, 1); zeros(n-k, 1)], 1);
    M(n,1) = - 1;
    M(1,n) = -1;
    b = -1/n * ones(n, 1);
    b(1) = b(1) + 1;
end

function x_k1 = steepestDescent(x_k, M, b, L)
    % Converges in 50000 iters to -8.3331
    x_k1 = x_k - (1 / L) * gradient(M, b, x_k);
end

function x_kl = steepestDescentLineSearch(x_k, M, b)
    desc_dirxn = -gradient(M, b, x_k);
    step_size = dot(desc_dirxn, desc_dirxn) / dot(M*desc_dirxn, desc_dirxn);
    x_kl = x_k + step_size * desc_dirxn;
end

function [x_k, step_kl] = laggedSteepestDescent(x_k_prev, M, b, step_kl_prev)
    desc_dirxn = -gradient(M, b, x_k_prev);
    step_kl = dot(desc_dirxn, desc_dirxn) / dot(M*desc_dirxn, desc_dirxn);
    x_k = x_k_prev + step_kl_prev * desc_dirxn;
end

function [x_k, y_k, v_k, a_k, A_k] = nesterovsMethod(x_k_prev, y_k_prev, v_k_prev, M, b, L, a_k_prev, A_k_prev)
    a_k = (1 + sqrt(1 + 4 * A_k_prev)) / 2;
    A_k = A_k_prev + a_k;
    
    x_k = (A_k_prev / A_k) * y_k_prev + (a_k / A_k) * v_k_prev;
    grad_x_k = gradient(M, b, x_k);
    y_k = x_k - grad_x_k / L;
    v_k = v_k_prev - (a_k / L) * grad_x_k;
end

function f_val = evaluate_func(M, b, x)
    f_val = (1/2) * dot(M*x, x) - dot(b, x);
end

function grad = gradient(M, b, x)
    grad = M * x - b;
end
