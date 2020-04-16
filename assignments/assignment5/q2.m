function q2(p, n, n_iter)
%Q2 Summary of this function goes here
%   Detailed explanation goes here
    
    n_trials = 30;

    f_vals_fw = zeros(n_iter, n_trials);
    iterate_sparsity_fw = zeros(n_trials, 1) - 1;
    f_vals_pgd = zeros(n_iter, n_trials);
    iterate_sparsity_pgd = zeros(n_trials, 1) - 1;
    
    trial = 1;
    while trial <= n_trials
        % rng('default') % For reproducibility
        B = normrnd(0, 1, [p, n]);
        b = ones(p, 1);

        evals = sort(eig(B'*B));
        L = evals(n);

        x0 = zeros(n, 1);
        x0(1) = 1;

        x_k_fw = zeros(n, 1);
        x_k_fw(1) = 1;
        
        x_k_pgd = zeros(n, 1);
        x_k_pgd(1) = 1;
        
        trial_f_vals_fw = zeros(n_iter, 1);
        trial_f_vals_pgd = zeros(n_iter, 1);
    
        for iter = 1 : n_iter
            x_k_fw = FrankWolfeMethod(x_k_fw, B, b, iter - 1);
            trial_f_vals_fw(iter) = evaluateFunction(B, b, x_k_fw) / evaluateFunction(B, b, x0);
            % f_vals_fw(iter, trial) = evaluateFunction(B, b, x_k_fw) / evaluateFunction(B, b, x0);
            if iterate_sparsity_fw(trial) == -1 && f_vals_fw(iter, trial) <= 0.1
                iterate_sparsity_fw(trial) = sum(x_k_fw == 0);
            end

            x_k_pgd = ProjectedGradientDescent(B, b, L, x_k_pgd);
            trial_f_vals_pgd(iter) = evaluateFunction(B, b, x_k_pgd) / evaluateFunction(B, b, x0);
            % f_vals_pgd(iter, trial) = evaluateFunction(B, b, x_k_pgd) / evaluateFunction(B, b, x0);
            if iterate_sparsity_pgd(trial) == -1 && f_vals_pgd(iter, trial) <= 0.1
                iterate_sparsity_pgd(trial) = sum(x_k_pgd == 0);
            end
        end
        
        if iterate_sparsity_fw(trial) ~= -1 && iterate_sparsity_pgd(trial) ~= -1
            f_vals_fw(:, trial) = trial_f_vals_fw;
            f_vals_pgd(:, trial) = trial_f_vals_pgd;
            
            trial = trial + 1;
        end
    end
    
    f_vals_fw = sum(f_vals_fw, 2) / n_trials;
    f_vals_pgd = sum(f_vals_pgd, 2) / n_trials;
    
    % Sanity check: if algo doesn't converge, this may happen
    if sum(iterate_sparsity_fw == -1) > 0 || sum(iterate_sparsity_pgd == -1) > 0
        disp(iterate_sparsity_fw);
        disp(iterate_sparsity_pgd);
    end
    
    disp("Frank-Wolfe sparsity: " + sum(iterate_sparsity_fw) / n_trials);
    disp("PGD sparsity: " + sum(iterate_sparsity_pgd) / n_trials);
    
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
