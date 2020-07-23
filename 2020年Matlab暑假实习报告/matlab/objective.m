function p = objective(A, b, gamma, x, z)
    p = 0.5*sum_square(A*x - b) + gamma*norm(z,1);
end