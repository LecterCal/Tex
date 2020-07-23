function x = pr1(v, lam)
    x = max(0, v - lam) - max(0, -v - lam);
end
