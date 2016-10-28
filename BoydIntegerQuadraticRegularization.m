function [xhat, X] = BoydIntegerQuadraticRegularization(X, x, P, q, n, m, s, K, window)
% n : number of variables
% m : number of Markov chains
% s : number of states
% K : number of samples for Park-Boyd method
X = (X + X')/2; % Force symmetry
mu = x; Sigma = X - x*x';
mk=m*s/window;
for i = 1:window
    for j = 1:window
        if i ~= j
            Sigma((i-1)*mk+1:i*mk,(j-1)*mk+1:j*mk) = 0;
        end
    end
end
[V, D] = eig(Sigma);
A = V*sqrt(max(D, 0));
ub = inf; xhat = zeros(n, 1); X = zeros(K,n+1);
for k = 1:K
    x = RandomAlgorithm(mu, A, m, s);
    [X(k,2:n+1), X(k,1)]=GreedyOneOpt(x, P, q, m, s);
end
X(:,1) = floor(X(:,1));
X = sortrows(X);
same = X(:,1) == X(1,1);
number = sum(same);
aux = X(same,2:end);
d = zeros(number,1);
for i=1:number
    d(i) = norm(aux(i,:)'-mu);
end
[~,Min] = min(d);
ub = X(Min(1),1);  xhat = X(Min(1),2:n+1)';
end
function [y] = RandomAlgorithm(mu, A, m, s)
% x = mulrandn_cached(mu, A);
n = m*s;
z = randn(n, 1);
x = mu + A*z ;
y = x;
y = reshape(y,s,m);
indexj = 0:s:(m-1)*s;
[~, indexi] = max(y);
idxmutual = indexi+indexj;
y = zeros(m*s,1);
y(idxmutual) = 1;

end
function [x, val]=GreedyOneOpt(x, P, q, m, s)
L1 = 10*(diag(P)>0);
g = 2*(P*x+q);
D = diag(P);
indexi = [1:m*s]';
aux = indexi(x > 0)';
aux = repmat(aux,s,1);
aux = aux(:);
% indexj = repelem(indexi(x > 0), s); % this command does not work on westgrid
indexj = aux; 
for iter = 1:30    
%     indexj = repelem(indexi(x > 0), s);
    idxmutual = indexi + (indexj-1)*(m*s);
    direction = g - g(indexj) + D + D(indexj) + L1 - L1(indexj) - 2*P(idxmutual);
    [dmin, indexd] = min(direction);
    if dmin >= 0
        break;
    end
    indexd_j = indexj(indexd);
    x(indexd_j) = 0;
    x(indexd) = 1;
    mc = floor((indexd_j+indexd)/(2*s));
%     mc = min(aux); % m_change
    indexj(mc*s+1:(mc+1)*s,1) = indexd;   
    g = g+2*(P(:,indexd)-P(:,indexd_j));      
end
val = x'*P*x + 2*q'*x + L1'*x;
end