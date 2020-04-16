function [outputArg1,outputArg2] = q2(p, n, n_iter)
%Q2 Summary of this function goes here
%   Detailed explanation goes here

    rng('default') % For reproducibility
    B = normrnd(0, 1, [p, n]);
    b = ones(p, 1);
    
    evals = sort(eig(B'*B));
    L = evals(n);
    
    x0 = zeros(n, 1);
    x0(1) = 1;
    
    x_k_fw = zeros(n, 1);
    x_k_fw(1) = 1;
    f_vals_fw = zeros(n_iter, 1);
    
    x_k_pgd = zeros(n, 1);
    x_k_pgd(1) = 1;
    f_vals_pgd = zeros(n_iter, 1);
    
    for iter = 1 : n_iter
        x_k_fw = FrankWolfeMethod(x_k_fw, B, b, iter - 1);
        f_vals_fw(iter) = evaluateFunction(B, b, x_k_fw) / evaluateFunction(B, b, x0);
        
        x_k_pgd = ProjectedGradientDescent(B, b, L, x_k_pgd);
        f_vals_pgd(iter) = evaluateFunction(B, b, x_k_pgd) / evaluateFunction(B, b, x0);
    end
    
    % Plot part (i): Optimality gap for SD:Const, SD:Exact and Nesterov
    figure
    plot(1:1:n_iter, f_vals_fw)
    set(gca, 'YScale', 'log')
    hold on
    plot(1:1:n_iter, f_vals_pgd)
    legend('Frank Wolfe Method', 'Projected Gradient Descent')
    title('Analysis of Optimization algorithms')
    xlabel('Num iterations')
    ylabel('f(x_k) / f(x_0)');
end

function x_k = FrankWolfeMethod(x_k_prev, B, b, k_prev)
    [n, ~] = size(x_k_prev);
    I = eye(n);
    
    grad_x_k_prev = computeGradient(B, b, x_k_prev);
    [~, i_k_prev] = max(abs(grad_x_k_prev));
    v_k = -sign(grad_x_k_prev(i_k_prev)) * I(:, i_k_prev);
    
    A_k_prev = (k_prev^2 + 5*k_prev + 6) / 4;
    a_k = (k_prev + 1) / 2;
    A_k = A_k_prev + a_k;
    
    % TODO: Check this step: Should we use a_k_prev instead of a_k?
    x_k = A_k_prev/A_k*x_k_prev + a_k/A_k*v_k;
end

function x_k = ProjectedGradientDescent(B, b, L, x_k_prev)
    x_k_prev = x_k_prev - 1/L * computeGradient(B, b, x_k_prev);
    x_k = ProjectOntoEllOneBall(x_k_prev', 1)';
end

function grad = computeGradient(B, b, x)
    grad = B' * (B*x - b);
end

function f_val = evaluateFunction(B, b, x)
    f_val = 1/2 * norm(B*x - b)^2;
end
