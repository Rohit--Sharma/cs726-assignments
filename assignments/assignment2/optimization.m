function optimization
    n = 200;
    [M, b] = initializeMatrix(n);
    
    evals = sort(eig(M));
    L = evals(n);
    disp(L);    % L = 4 in this case
    %disp('Matrix M:')
    %disp(M)
    
    %disp('Vector b:')
    %disp(b)
    
    x_optimal = pinv(M) * b;
    disp('Global min:');
    f_optimal = (1/2) * dot(M*x_optimal, x_optimal) - dot(b, x_optimal);
    disp(f_optimal);
%     disp('Global minimizer');
%     disp(global_min);
    
    x_k = zeros(n, 1);
    f_vals = [];
    iter = 10000;
    for k = 1 : iter
        x_k = steepestDescent(x_k, M, b, L);
%         disp('f:');
        f = (1/2) * dot(M*x_k, x_k) - dot(b, x_k);
        f_vals = [f_vals, f - f_optimal];
%         disp(f);
%         disp('x:');
%         disp(x_k);
    end
    
    plot(1:1:iter, f_vals);
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

function steepestDescentLineSearch

end

function laggedSteepestDescent

end

function nesterovsMethod

end

function grad = gradient(M, b, x)
    grad = M * x - b;
end
