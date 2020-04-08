% Projections onto ell_1 ball
function w = ProjectOntoEllOneBall(v, b)
    n = length(v);
    w = zeros(n, 1);
    if (b < 0)
        error('Radius of ell_1 ball is negative: %2.3f\n', b);
    end
    r = sum(abs(v));
    if r <= b
        w(:) = v(:); % already in ell_1 ball
    else
        w = ProjectOntoSimplex(v, b);
        w = w .* sign(v);
    end
end