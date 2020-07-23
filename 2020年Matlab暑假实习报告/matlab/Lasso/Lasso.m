%need install cvx in matlab 
s = RandStream.create('seed',0);
RandStream.setGlobalStream(s);

m = 10000;       
n = 15000;      % number of features

x0 = sprandn(n,1,0.05);
A = randn(m,n);
A = A*spdiags(1./sqrt(sum(A.^2))',0,n,n); % normalize columns
v = sqrt(0.001)*randn(m,1);
b = A*x0 + v;

fprintf('solving instance with %d examples, %d variables\n', m, n);
fprintf('nnz(x0) = %d; signal-to-noise ratio: %.2f\n', nnz(x0), norm(A*x0)^2/norm(v)^2);

ga_max = norm(A'*b,'inf');
ga = 0.1*ga_max;


AtA = A'*A;
Atb = A'*b;

MI = 100;
BS   = 1e-4;
RELTOL   = 1e-2;
tic

cvx_begin quiet
    cvx_precision low
    variable x(n)
    minimize(0.5*sum_square(A*x - b) + ga*norm(x,1))
cvx_end

xcv = x;
h.p_cvx = cvx_optval;
cvxt = toc;
f = @(t) 0.5*sum_square(A*t-b);
lam = 1;
be= 0.5;

tic;

x = zeros(n,1);
xp = x;

for k = 1:MI
    while 1
        grad_x = AtA*x - Atb;
        z = pl1(x - lam*grad_x, lam*ga);
        if f(z) <= f(x) + grad_x'*(z - x) + (1/(2*lam))*sum_square(z - x)
            break;
        end
        lam = beta*lam;
    end
    xp = x;
    x = z;

    propt(k) = obj(A, b, ga, x, x);
    if k > 1 && abs(propt(k) - propt(k-1)) < BS
        break;
    end
end

xpro= x;
h.p_prox = propt(end);
h.prox_grad_toc = toc;
lam = 1;

tic;

x = zeros(n,1);
xp = x;
for k = 1:MI
    y = x + (k/(k+3))*(x - xp);
    while 1
        grad_y = AtA*y - Atb;
        z = pl1(y - lam*grad_y, lam*ga);
        if f(z) <= f(y) + grad_y'*(z - y) + (1/(2*lam))*sum_square(z - y)
            break;
        end
        lam = beta*lam;
    end
    xp = x;
    x = z;

    fasop(k) = obj(A, b, ga, x, x);
    if k > 1 && abs(fasop(k) - fasop(k-1)) < BS
        break;
    end
end

xfa = x;
pfa = fasop(end);
xfato = toc;
lam = 1;
rho = 1/lam;

tic;

x = zeros(n,1);
z = zeros(n,1);
u = zeros(n,1);

[L U] = factor(A, rho);

for k = 1:MI

    % x-update
    q = Atb + rho*(z - u);
    if m >= n
       x = U \ (L \ q);
    else
       x = lam*(q - lam*(A'*(U \ ( L \ (A*q) ))));
    end

    % z-update
    zold = z;
    z = pl1(x + u, lam*ga);

    % u-update
    u = u + x - z;

   
    h.admm_optval(k)   = obj(A, b, ga, x, z);
    hrn(k)   = norm(x - z);
    hsn(k)   = norm(-rho*(z - zold));
    hepp(k)  = sqrt(n)*BS + RELTOL*max(norm(x), norm(-z));
    hed(k) = sqrt(n)*BS + RELTOL*norm(rho*u);

    if hrn(k) < hepp(k) && hsn(k) < hed(k)
         break;
    end

end

h.pit= length(propt);
h.fait = length(fasop);
h.adit = length(h.admm_optval);
K = max([h.pit h.fait h.adit]);
cvx_op  = p_cvx*ones(K,1);
propt = padarray(propt', K-pr_iter, h.p_pr, 'post');
fasop = padarray(fasop', K-fa_iter, pfa, 'post');
adptval = padarray(h.pit', h.fait, h.adit , 'post');
fig = figure;

xlabel('迭代次数');
ylabel('目标函数最小值');
plot(1:K, cvx_op,  'k--', ...
     1:K, propt, 'r-', ...
     1:K, fasop, 'g-', ...
     1:K, adptval, 'b-');

