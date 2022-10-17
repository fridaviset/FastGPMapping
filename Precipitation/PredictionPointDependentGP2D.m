function [mu, var, max_set_length, I]=PredictionPointDependentGP2D(x,y,x_s,Omega,N_u1,N_u2,sigma_SE,l_SE,r_star,sigma_y)
x1_u=linspace(Omega(1,1),Omega(1,2),N_u1); %Inducing inputs along first dimension
x2_u=linspace(Omega(2,1),Omega(2,2),N_u2); %Inducing inputs along second dimension
l_u1=x1_u(2)-x1_u(1); %Distance between two inducing inputs along dim 1
l_u2=x2_u(2)-x2_u(1); %Distance between two inducing inputs along dim 2

%Now, store and refer to bigger structures
I=zeros(N_u1*N_u2,N_u1*N_u2);
iota=zeros(N_u1,N_u2);

%Learning
N=size(x,2);
for t=1:N
    %Decide upon the set to use
    [set1, set2, xu_set]=find_set(x(:,t),Omega,l_u1,l_u2,N_u1,N_u2,x1_u,x2_u,2*r_star);
    set=((set1-1)'.*N_u2+set2);
    set=set(:);
    %Perform the information matrix update
    phi=Kern2D(x(:,t),xu_set,sigma_SE,l_SE);
    I_mat=phi'*phi;
    iota_mat=phi'*y(t);
    I(set,set)=I(set,set)+I_mat;
    iota(set)=iota(set)+iota_mat;
end

%Prediction
N_s=size(x_s,2);
mu=zeros(N_s,1);
var=zeros(N_s,1);
max_set_length=0;
for t=1:N_s

%Calculate the set in a smarter way.
[set1, set2, xu_set]=find_set(x_s(:,t),Omega,l_u1,l_u2,N_u1,N_u2,x1_u,x2_u,r_star);
set=((set1-1)'.*N_u2+set2);
set=set(:);
M=length(set);

I_mat_recon=I(set,set);
iota_mat_recon=iota(set);

phi=Kern2D(x_s(:,t),xu_set,sigma_SE,l_SE);
temp=I_mat_recon+sigma_y^2.*Kern2D(xu_set,xu_set,sigma_SE,l_SE);
Kss=Kern2D(x_s(:,t),x_s(:,t),sigma_SE,l_SE);
Kuu=Kern2D(xu_set,xu_set,sigma_SE,l_SE);
Qss=phi*(Kuu\(phi'));
mu(t)=phi*(temp\(iota_mat_recon));
var(t)=diag(sigma_y^2*phi*(temp\(phi')))+Kss-Qss;
max_set_length=max(max_set_length,M);

end

I=full(I);

disp(['Max number of inducing inputs used: ',num2str(max_set_length)]);

end