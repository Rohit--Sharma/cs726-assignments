% CS726 - Nonlinear Optimization
%   HW5
%   Author: Rohit Sharma (rohit.sharma@wisc.edu)
%
% Q1: Runs PGD and Mirror Descent Methods in Probability Simplex
%   on a function defined by max <zi, x> with x of dimension `n`
%   for `n_iter' iterations.

function q1(n, n_iter)
% Q1: Runs PGD and Mirror Descent Methods in Probability Simplex
%   on a function defined by max <zi, x> with x of dimension `n`
%   for `n_iter' iterations.
%
%   Convention: x, grad: col vector, Z: col vectors are z_is

    % Generate Rademacher vectors (columns of Z)
    rng('default') % For reproducibility
    p = 0.5;
    Z = 2*binornd(1, p * ones(n)) - 1;

    % Initialize Lipschitz constants of f in various norms
    M1 = 1;
    M2 = sqrt(n);
    
    % PGD Initializations
    x_k_pgd = 1/n * ones(n, 1);
    x_k_pgd_out = x_k_pgd;
    A_k_pgd = 1;
    f_pgd_iterate_vals = zeros(n_iter, 1);
    f_pgd_out_vals = zeros(n_iter, 1);
    
    % MD Initializations
    x_k_md = 1/n * ones(n, 1);
    x_k_md_out = x_k_md;
    A_k_md = 1;
    f_md_iterate_vals = zeros(n_iter, 1);
    f_md_out_vals = zeros(n_iter, 1);
    
    % Optimization loop
    for iter = 1 : n_iter
        % Run PGD Step
        [x_k_pgd, x_k_pgd_out, A_k_pgd] = ProjectedGradientDescent(x_k_pgd, x_k_pgd_out, Z, M2, iter, A_k_pgd);
        f_pgd_iterate_vals(iter) = max(x_k_pgd' * Z);
        f_pgd_out_vals(iter) = max(x_k_pgd_out' * Z);
        
        % Run MD Step
        [x_k_md, x_k_md_out, A_k_md] = MirrorDescent(x_k_md, x_k_md_out, Z, M1, iter, A_k_md);
        f_md_iterate_vals(iter) = max(x_k_md' * Z);
        f_md_out_vals(iter) = max(x_k_md_out' * Z);
    end
    
    % Plot f(x_k) for PGD and MD
    figure
    plot(1:1:n_iter, f_pgd_iterate_vals)
    set(gca, 'YScale', 'log')
    hold on
    plot(1:1:n_iter, f_md_iterate_vals)
    legend('PGD', 'Mirror Descent')
    title('Analysis of Optimization algorithms')
    xlabel('Num iterations')
    ylabel('f(x_k)');
    
    % Plot f(x_k^{out}) for PGD and MD
    figure
    plot(1:1:n_iter, f_pgd_out_vals)
    set(gca, 'YScale', 'log')
    hold on
    plot(1:1:n_iter, f_md_out_vals)
    legend('PGD', 'Mirror Descent')
    title('Analysis of Optimization algorithms')
    xlabel('Num iterations')
    ylabel('f(x_k^{out})');
end

% Get the next iterates of PGD
function [x_k, x_k_out, A_k] = ProjectedGradientDescent(x_k_prev, x_k_out_prev, Z, M, k, A_k_prev)
    a_k = 1 / (M * sqrt(k));
    A_k = A_k_prev + a_k;
    
    x_k_prev = x_k_prev - a_k * grad(x_k_prev, Z);
    x_k = ProjectOntoSimplex(x_k_prev', 1)';
    x_k_out = 1/A_k * (a_k*x_k + A_k_prev*x_k_out_prev);
end

% Get the next iterates of MD
function [x_k, x_k_out, A_k] = MirrorDescent(x_k_prev, x_k_out_prev, Z, M, k, A_k_prev)
    a_k = 1 / (M * sqrt(k));
    A_k = A_k_prev + a_k;
    
    v_k = a_k * grad(x_k_prev, Z) + 1 - log(x_k_prev);
    x_k = exp(-v_k) / sum(exp(-v_k));
    x_k_out = 1/A_k * (a_k*x_k + A_k_prev*x_k_out_prev);
end

% Helper method to evaluate the gradient of function at a given input
function subgrad = grad(x, Z)
    [~, amax] = max(x' * Z);
    subgrad = Z(:, amax);
end
