function []=main_admm(itermax,mu,ch,weight)
% warning off
% % if (ischar(weight)),
% %     weight = str2double(weight);
% % end
% % if (ischar(itermax)),
% %     itermax = str2double(itermax);
% % end
% % if (ischar(mu)),
% %     mu = str2double(mu);
% % end
% % if (ischar(ch)),
% %     ch = str2double(ch);
% % end
% format long
ch = 50;
mu = 0.001;
itermax = 250;
weight = 1;
% Data = load('SeveralApp_hmms.mat');
Data = load('SeveralApp_state.mat');
% test = load('Setting_EvsHMMLong.mat');
test = load('Setting_EvsStateLong.mat');
index=test.par(ch).index;
K = test.par(ch).K;
Start = test.par(ch).start;
T = test.par(ch).T;
Appliance=Data.appliance(K,index);
[Model]=Model_parameters(Appliance,Start,T);
M = Model.M;
K = Model.K;
scale = 5*M;
[A,B,C,E,b,g,A_inv,BCE_inv,L_A,L_B]=ConstraintsImproved(M,K);
% d coefficient
Mu = Model.Mu./(scale);
CC = Mu*Mu';
Y = Model.Y./(scale);
D = zeros(M*K+1,M*K+1,T);
L1 = 10*(Mu>0);
for i=1:T
    D(:,:,i)=[0,-Y(i)*Mu';-Y(i)*Mu, CC];
end
% h coefficient
HH=zeros(M*K*K,T);
for i=1:T-1
    HH(:,i)=-Model.P/1+Model.W_diff(i,:)'./(scale/M)^2;
end
A_inv_mu=A_inv;
BCE_inv_mu=BCE_inv;
tic
[S,~,~,S_total]=Proximal_BigSDP_Time_Global(itermax,M,K,T,mu,A,B,C,E,b,g,A_inv_mu,BCE_inv_mu,L_A,L_B,D,HH);
toc
YY=zeros(M*K,T);
for i=1:T
    YY(:,i)=S(2:end,1,i);
end
[error,Y_result,~]=error_analysis(Model.Mu/scale,YY,M,K,T,Model.power_ref,Model.section,scale);
error

Yhat = zeros(M*K,T);
MU=Mu*Mu';
tic
window = 3;
QQ = zeros(window*M*K);
for i = 1:window
    QQ((i-1)*M*K+1:i*M*K,(i-1)*M*K+1:i*M*K) = MU;
    if i<window
        QQ((i-1)*M*K+1:i*M*K,(i)*M*K+1:(i+1)*M*K) = -weight*Model.Ps/2;
        QQ((i)*M*K+1:(i+1)*M*K, (i-1)*M*K+1:i*M*K) = -weight*Model.Ps'/2;
    end
end
Q = QQ;
Yhat(:,1) = S(1,2:end,1)';
Yhat(:,T) = S(1,2:end,T)';
for t = 2:1:T-1
    Yhat(:,t) = S(1,2:end,t)';
    if t==2
        X = [S(2:end,2:end,t-1), zeros(M*K), zeros(M*K); zeros(M*K), S(2:end,2:end,t), zeros(M*K); zeros(M*K), zeros(M*K), S(2:end,2:end,t)];
        x = [S(1,2:end,t-1)'; S(1,2:end,t)'; S(1,2:end,t+1)'];
    else
        X = [Yhat(:,t-1)*Yhat(:,t-1)', zeros(M*K), zeros(M*K); zeros(M*K), S(2:end,2:end,t), zeros(M*K); zeros(M*K), zeros(M*K), S(2:end,2:end,t)] ;
        x = [Yhat(:,t-1); S(1,2:end,t)'; S(1,2:end,t+1)'];
    end
    q = [ -Y(t-1)*Mu; -Y(t)*Mu; -Y(t+1)*Mu;];    
    [xhat, ~] = BoydIntegerQuadraticRegularization(X, x, Q, q, window*M*K, window*M, K, 10*window*M*K, window);
    Yhat(:,t) = xhat(1*M*K+1:2*M*K);
end
toc
[loglik, loglik_actual] = Loglikelihood(Y, Yhat, Model.Mu, Model.Ps, Model.Ws, M, K, Model.srb);
Section = Model.section;
Section = Section(1:T);
[error_hat,Y_result_hat,Power_ref_hat]=error_analysis(Model.Mu/scale,Yhat,M,K,T,Model.power_ref,Section,scale);
error_hat
State = 1;
Name=['Admm_',num2str(T),'_',num2str(Start),'_',num2str(itermax),'_',num2str(1000*mu),'_',...
    num2str(ch),'_',num2str(State),'_',num2str(weight)];
save(Name,'error_hat','error','itermax','Start','T','mu','State','ch','weight','Yhat','Y_result_hat','loglik', 'loglik_actual','S_total')
end

function [ERROR,Y_result,Power_ref]=error_analysis(Mu,Y,M,K,T,power_ref,section,scale)
S_final=zeros(T,M*K);
for i=1:T
    S_final(i,:) = Y(:,i)';
end
Y_result = zeros(T,M);
Y_round = zeros(T,M);
for m=1:M
    Y_result(:,m) = (S_final(:,(m-1)*K+1:m*K))*Mu((m-1)*K+1:m*K);
    Y_round(:,m) = (round(S_final(:,(m-1)*K+1:m*K))*Mu((m-1)*K+1:m*K));
end
Power_ref = power_ref(section,:)./scale;

[RMS_error,NDE_nips,NDE_k,NDE_mit,Error]=error_cal(Y_result,Power_ref);
ERROR=[RMS_error NDE_nips NDE_k NDE_mit Error];
end
function  [RMSE,NDE_nips,NDE_k,NDE_mit,Error]=error_cal(result,reference)
[T, M]=size(result);
RMSE=1/T*sqrt(sum(sum((result-reference).^2)));
NDE_nips=sum(sum((result-reference).^2))/sum(sum((reference).^2)); % normalized disaggregation error; nips paper
NDE_mit=sqrt(NDE_nips); % normalized disaggregation error; mit paper
Error=zeros(1,M); % error for each appliance
for i=1:M
    Error(1,i)=sum((result(:,i)-reference(:,i)).^2)/max(sum((reference(:,i)).^2),T);
end
NDE_k = inf;
end

function [A,B,C,E,b,g,A_inv,BCE_inv,L_A,L_B]=ConstraintsImproved(M,K)

MK=M*K;
A=zeros(1+MK+2*MK+M,(MK+1)^2);
b=zeros(1+MK+2*MK+M,1);
Index=0;
% X(1,1)=1
Index=Index+1;
A(Index,1)=1; b(Index)=1;

% X(i,i)=(X(1,i)+X(i,1))/2;
for i=2:MK+1
    Index=Index+1;
    aux=zeros(MK+1,MK+1);
    aux(i,i)=1/2;
    aux(1,i)=-1/4;
    aux(i,1)=-1/4;
    A(Index,:)=aux(:)';
    b(Index,1)=0;
end

% X*ones(*,1)=X(:,1)
for i=2:MK+1
    Index=Index+1;
    aux=zeros(MK+1,MK+1);
    aux(1,i)=1;
    aux(2:end,i)=-1/M;
    A(Index,:)=aux(:)';
    b(Index,1)=0;
end
for i=2:MK+1
    Index=Index+1;
    aux=zeros(MK+1,MK+1);
    aux(i,1)=1;
    aux(i,2:end)=-1/M;
    A(Index,:)=aux(:)';
    b(Index,1)=0;
end

% sum(X(1,2:4))==1
for i=1:M
    Index=Index+1;
    aux=zeros(MK+1,MK+1);
    aux(1,2+(i-1)*K:i*K+1)=0.5;
    aux(2+(i-1)*K:i*K+1,1)=0.5;
    A(Index,:)=aux(:)';
    b(Index,1)=1;
end

% transition matrix constraints
B=zeros(MK+MK,(MK+1)^2);
C=zeros(MK+MK,M*K*K);
E=zeros(MK+MK,(MK+1)^2);
g=zeros(MK+MK,1);
Index=0;
for j=1:M
    for i=1:K
        Index=Index+1;
        aux=zeros(K,K);
        aux(i,:)=1/2;
        temp=zeros(1,M*K*K);
        temp((j-1)*K*K+1:j*K*K)=aux(:)';
        C(Index,:)=temp;
        E(Index,:)=zeros(1,(M*K+1)^2);
        aux=zeros(MK+1,MK+1);
        aux(1,1+(j-1)*K+i)=-1/4;
        aux(1+(j-1)*K+i,1)=-1/4;
        B(Index,:)=aux(:)';
        g(Index,1)=0;
    end
end

for j=1:M
    for i=1:K
        Index=Index+1;
        aux=zeros(K,K);
        aux(:,i)=1/2;
        temp=zeros(1,M*K*K);
        temp((j-1)*K*K+1:j*K*K)=aux(:)';
        C(Index,:)=temp;
        B(Index,:)=zeros(1,(M*K+1)^2);
        aux=zeros(MK+1,MK+1);
        aux(1,1+(j-1)*K+i)=-1/4;
        aux(1+(j-1)*K+i,1)=-1/4;
        E(Index,:)=aux(:)';
        g(Index,1)=0;
    end
end

% scaling the constraints
scale=1000;
A=A.*scale;
B=B.*scale;
C=C.*scale;
E=E.*scale;
b=b.*scale;
g=g.*scale;

% pseudo inverse
A_inv=pinv(A*A');
BCE_inv=pinv(B*B'+C*C'+E*E');

L_A=size(A,1);
L_B=size(B,1);
% A=sparse(A);
% B=sparse(B);
% E=sparse(E);
end

function [Model]=Model_parameters(appliance,start,T)
M=length(appliance);
K=appliance(1,1).Numstate;
KK=K^2;

P=zeros(M*KK,1);
Ps = zeros(M*K,M*K);
for i=1:M
    %     P((i-1)*KK+1:i*KK)=appliance(i).transition(:);
    P((i-1)*KK+1:i*KK)=appliance(i).trans(:);
    Ps((i-1)*K+1:i*K,(i-1)*K+1:i*K) =  log(appliance(i).trans);
end
P=log(P);


PI=1/K*ones(M*K,1);
PI=log(PI);

Mu=zeros(K*M,1);
for i=1:M
    Mu((i-1)*K+1:i*K)=appliance(i).mu';
end
MU=Mu*Mu';

mu_delta{M}=[];
for m=1:M
    mu=appliance(m).mu';
    delta=zeros(K);
    for i=1:K
        for j=1:K
            delta(i,j)=mu(j)-mu(i);
        end
    end
    mu_delta{m}=delta;
end

duration=appliance(1).duration;
Y=zeros(duration,1); % vector- observation
power_ref=zeros(duration,M);
state_ref=zeros(duration,M);
for i=1:M
    Y=Y+appliance(i).power';
    power_ref(:,i)=appliance(i).power';
    state_ref(:,i)=appliance(i).state';
end

srb = zeros(M*K,duration); % state reference binary
for t=1:duration
    for i=1:M
        aux=zeros(K,1);
        aux(state_ref(t,i)) = 1;
        srb((i-1)*K+1:i*K,t) = aux;
    end
end
section=start+1:start+T;
Y=Y(section);
srb=srb(:,section);
Y_delta=Y(2:end)-Y(1:end-1);
[Edge]=edge_detector(Y,50);

% in this section we define (Y_delta-Mu_delta)^2 as a wighting vector for Z(t)
W_diff = zeros(T-1,M*KK);
wdiff = zeros(M*K,M*K,T);
for t=1:T-1
    for m=1:M
        Mu_delta=mu_delta{m};
        w = zeros(K);
        ws = zeros(K);
        for i=1:K
            for j=1:K
                if i~=j
                    %                     ws(i,j)=(abs(Y_delta(t))-abs(Mu_delta(i,j)))^2;
                    ws(i,j) = (Y_delta(t)-Mu_delta(i,j))^2;
                    w(i,j)=(Y_delta(t)-Mu_delta(i,j))^2;
                end
            end
        end
        wdiff((m-1)*K+1:m*K,(m-1)*K+1:m*K,t) = ws;
        w = w(:);
        W_diff(t,(m-1)*KK+1:m*KK)=w';
    end
end

Model.M=M;
Model.K=K;
Model.KK=KK;
Model.P=P;
Model.Ps = Ps; % this is similar to P but in matrix format instead of vector format
Model.PI=PI;
Model.Mu=Mu;
Model.MU=MU;
Model.mu_delta=mu_delta;
Model.Y=Y;
Model.Y_delta=Y_delta;
Model.section=section;
Model.power_ref=power_ref;
Model.state_ref=state_ref;
Model.Edge=Edge;
Model.W_diff=W_diff;
Model.Ws=wdiff; % this is similar to W_diff but in matrix format instead of vector format
Model.srb = srb; % state reference binary
end
function [Edge]=edge_detector(Y,threshold)

edge=Y(2:end)-Y(1:end-1);
edge=abs(edge)>=threshold;

T=length(Y);
Edge=zeros(T,1);

for i=1:T-1
    if edge(i)==1
        Edge(i)=1;
        Edge(i+1)=1;
    end
end
end
function plotplot(Yh,Y,P,M)
for i=1:M
    figure
    plot(P(1:end,i), 'LineWidth', 2)
    hold on
    plot(Yh(1:end,i), 'r', 'LineWidth', 2)
    hold on
    plot(Y(1:end,i), 'g', 'LineWidth', 2)
    ax = gca; % current axes
    ax.FontSize = 30;
    xlabel('time', 'FontSize', 30)
    ylabel('power', 'FontSize', 30)
    %     legend('Real', 'ADMM' )
end

end

function [loglik, loglik_actual] = Loglikelihood(Y, S, mu, P, Ws, M, K, srb)
T = length(S);

obser = zeros(1,T); % observation probability
for i = 1:T
   obser(i) = (Y(i)-mu'*S(:,i))^2/(2*(M*5)^2);
end

trans = zeros(T-1,M);
for i = 1:T-1
    for j = 1:M
        trans(i) = S(:,i)'*(-P+Ws(:,:,i))*S(:,i+1)/5; 
    end
end
loglik = -(sum(obser)+ sum(sum(trans)));

S = srb;
obser = zeros(1,T); % observation probability
for i = 1:T
   obser(i) = (Y(i)-mu'*S(:,i))^2/(2*(M*5)^2);
end

trans = zeros(T-1,M);
for i = 1:T-1
    for j = 1:M
        trans(i) = S(:,i)'*(-P+Ws(:,:,i))*S(:,i+1)/5; 
    end
end
loglik_actual = -(sum(obser)+ sum(sum(trans)));
end