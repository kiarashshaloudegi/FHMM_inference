function [S,R,Lag,S_total]=Proximal_BigSDP_Time_Global(itermax,M,K,T,mu_original,A,B,C,E,b,g,A_inv,BCE_inv,L_A,L_B,D,HH)
% min <C1,X1>+d'*z+<C2,X2> s.t. A*X1=b A*X2=b B*X1+C*z+E*X2=g
S = zeros(M*K+1,M*K+1,T);
S_total = zeros(M*K+1,M*K+1,T,itermax/250); 
W=zeros(M*K+1,M*K+1,T);
P=zeros(M*K+1,M*K+1,T);
R=zeros(M*K*K,T);
H=zeros(M*K*K,T);
L=zeros(L_A,T);
V=zeros(L_B,T);
% Lag=zeros(10,itermax,T);
Lag=0;
MK=M*K+1;
mu=mu_original;
for i=1:itermax
    for t=1:T
        if t>1
            v_min=V(:,t-1);
        else
            v_min=zeros(L_B,1);
        end
        
        [W(:,:,t),DQ,At_l,Et_vmin]=w_update(S(:,:,t),P(:,:,t),L(:,t),V(:,t),v_min,D(:,:,t),A,B,E,MK,mu);
        P(:,:,t)=p_update(W(:,:,t),MK,DQ);
        [L(:,t),DQ,At_l]=lambda_update(W(:,:,t),P(:,:,t),A,b,A_inv,MK,mu,DQ,At_l);
        S(:,:,t)=S_update(S(:,:,t),W(:,:,t),P(:,:,t),mu,DQ,MK);
        
        if t<T
            R(:,t)=r_update(R(:,t),H(:,t),V(:,t),HH(:,t),C,mu);
            H(:,t)=h_update(R(:,t),V(:,t),HH(:,t),C,mu);
            V(:,t)=v_update(S(:,:,t),W(:,:,t),P(:,:,t),D(:,:,t),R(:,t),H(:,t),S(:,:,t+1),W(:,:,t+1),P(:,:,t+1),L(:,t+1),V(:,t+1),D(:,:,t+1),B,E,A,C,g,HH(:,t),MK,BCE_inv,mu,At_l,Et_vmin);
        end
        if t<T
%             [Lag(:,i,t)]=Lagrangian(S(:,:,t),W(:,:,t),P(:,:,t),L(:,t),V(:,t),v_min,D(:,:,t),R(:,t),H(:,t),S(:,:,t+1),W(:,:,t+1),P(:,:,t+1),L(:,t+1),V(:,t+1),D(:,:,t+1),A,B,b,C,HH(:,t),E,g,MK,mu);
        else
            s_plus=zeros(M*K+1,M*K+1);
            w_plus=zeros(M*K+1,M*K+1);
            p_plus=zeros(M*K+1,M*K+1);
            l_plus=zeros(L_A,1);
            v_plus=zeros(L_B,1);
            D_plus=zeros(M*K+1,M*K+1);
%             [Lag(:,i,t)]=Lagrangian(S(:,:,t),W(:,:,t),P(:,:,t),L(:,t),V(:,t),v_min,D(:,:,t),R(:,t),H(:,t),s_plus,w_plus,p_plus,l_plus,v_plus,D_plus,A,B,b,C,HH(:,t),E,g,MK,mu);
        end
    end
    Lable = mod(i,250);
    if Lable == 0
       indexlabel = floor(i/250);
       S_total(:,:,:,indexlabel) = S;
    end
end

end
function [w,DQ,At_l,Et_vmin]=w_update(s,p,l,v,v_min,D,A,B,E,MK,mu)
At_l=A_tran(A,l,MK);
Bt_v=A_tran(B,v,MK);
Et_vmin=A_tran(E,v_min,MK);
DQ=At_l+Bt_v+Et_vmin-D+s/mu; % Quadratic part of the dual problem
w=max(-(DQ+p),0);
end
function [X_pls]=p_update(w,MK,DQ)
% coder.extrinsic('eig');
% V_eig=ones(MK,MK)*(1+1i);
% D_eig=ones(MK,MK)*(1+1i);
X=-(DQ+w);
[V_eig,D_eig] = eig(X);
d_real = real( diag(D_eig) );
V_eig = real (V_eig);
idxs_pls = find(d_real >= 0);
% [~,idxs_pls]=max(d_real);
n_pls = length(idxs_pls);
d_pos=d_real(idxs_pls);
V_pls=zeros(MK,n_pls);
for i=1:n_pls
    V_pls(:,i)=V_eig(:, idxs_pls(i))*sqrt(d_pos(i));
end
X_pls= (V_pls * V_pls');
end
function [l,DQ,At_l_new]=lambda_update(w,p,A,b,A_inv,MK,mu,DQ,At_l)
X=mu*(DQ+p+w-At_l);
l=(1/mu)*A_inv*(b-A*X(:)); % (1/mu) has been considered
At_l_new=A_tran(A,l,MK);
DQ=DQ-At_l+At_l_new;
end
function [s]=S_update(s,w,p,mu,DQ,MK)
s=mu*(DQ+p+w);
end
function [r]=r_update(r,h,v,d,C,mu)
r=r+mu*(h+C'*v-d);
end
function [h]=h_update(r,v,d,C,mu)
h=max(-(C'*v-d+r/mu),0);
end
function [v]=v_update(s,w,p,D,r,h,s_plus,w_plus,p_plus,l_plus,v_plus,D_plus,B,E,A,C,g,d,MK,BCE_inv,mu,At_l,Et_v_min)

X=(s+mu*(p+w+At_l+Et_v_min-D));
Bx=B*X(:);

At_l=A_tran(A,l_plus,MK);
Bt_v_plus=A_tran(B,v_plus,MK);
X=(s_plus+mu*(p_plus+w_plus+At_l+Bt_v_plus-D_plus));
Ex=E*X(:);

Cz=C*(r+mu*(h-d));

v=(1/mu)*BCE_inv*(g-Bx-Cz-Ex); % (1/mu) has been considered
end
function [At]=A_tran(A,l,MK)
at=A'*l;
At=reshape(at,MK,MK);
end
function [aux]=Lagrangian(s,w,p,l,v,v_min,D,r,h,s_plus,w_plus,p_plus,l_plus,v_plus,D_plus,A,B,b,C,d,E,g,MK,mu)
aux=zeros(10,1);
aux(1)=trace(D'*s);
aux(2)=d'*r;
aux(3)=l'*(b-A*s(:));
%%%%%%%
At_l=A_tran(A,l_plus,MK);
Bt_v=A_tran(B,v_plus,MK);
Bt_vmin=A_tran(E,v,MK);
S_plus=s_plus+mu*(p_plus+w_plus+At_l+Bt_v+Bt_vmin-D_plus);
aux(4)=v'*(g-B*s(:)-C*r-E*S_plus(:));
%%%%%%%
At_l=A_tran(A,l,MK);
Bt_v=A_tran(B,v,MK);
Et_vmin=A_tran(E,v_min,MK);
aux(5)=mu/2*norm(w+p+At_l+Bt_v+Et_vmin-D,'fro');
aux(6)=mu/2*norm(h+C'*v-d);
aux(7)=-trace(w'*s);
aux(8)=-trace(p'*s);
aux(9)=-h'*r;
aux(10)=sum(aux);
end