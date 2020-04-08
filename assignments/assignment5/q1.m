% x, grad: col vector, Z: col vectors are z_is
function q1(n, n_iter)
    rng('default') % For reproducibility
    p = 0.5;
    Z = 2*binornd(1, p * ones(n)) - 1;
    
    x_k = 1/n * ones(n, 1);
    L=1;
    
    for iter = 1 : n_iter
        x_k = ProjectedGradientDescent(x_k, Z, L)
        f_val = max(x_k' * Z)
    end
end

function x_k = ProjectedGradientDescent(x_k_prev, Z, L)
    x_k_prev = x_k_prev - 1/L * grad(x_k_prev, Z);
    x_k = ProjectOntoSimplex(x_k_prev', 1)';
end

function subgrad = grad(x, Z)
    [~, amax] = max(x' * Z);
    subgrad = Z(:, amax);
end
