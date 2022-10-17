function [mu, var, training_time, prediction_time]=FloatingDomainGP2D(x,y,x_s,Omega,N_u,sigma_SE,l_SE,r,sigma_y)
x1_u=linspace(Omega(1,1),Omega(1,2),N_u(1)); %Inducing inputs along first dimension
x2_u=linspace(Omega(2,1),Omega(2,2),N_u(2)); %Inducing inputs along second dimension
l_u1=x1_u(2)-x1_u(1); %Distance between two inducing inputs along dim 1
l_u2=x2_u(2)-x2_u(1); %Distance between two inducing inputs along dim 2

%Now, store and refer to bigger structures
M_max1=ceil(4*r(1)./l_u1);
M_max2=ceil(4*r(2)./l_u2);

%Allocate space for the information matrix and the information vector
I=zeros(N_u(1)*M_max1,N_u(2)*M_max2);
iota=zeros(N_u(1),N_u(2));

tic;
%Learning
N=size(x,2);
for t=1:N
    %Decide upon the set to use
    [set1, set2, xu_set]=find_set2D(x(:,t),Omega,l_u1,l_u2,N_u,x1_u,x2_u,2*r);
    
    %Perform the information matrix update
    phi=Kern2D(x(:,t),xu_set,sigma_SE,l_SE);
    I_mat=phi'*phi;
    iota_mat=phi'*y(t);
    
    %Find I_temp on tensor form
    I_k=reshape(I_mat,length(set1),length(set2),length(set1),length(set2));
    I_kb=permute(I_k,[1 3 2 4]);

    [set1is,set1js]=meshgrid(set1,set1);
    [set2is,set2js]=meshgrid(set2,set2);
    p1=(set1is-1)*M_max1+(set1js+1-set1is);
    p2=(set2is-1)*M_max2+(set2js+1-set2is);
    I(p1,p2)=I(p1,p2)+reshape(I_kb,length(set1)*length(set1),length(set2)*length(set2));
    
    iota(set1,set2)=iota(set1,set2)+reshape(iota_mat,length(set1),length(set2));
end
training_time=toc;

%Prediction
N_s=size(x_s,2);
mu=zeros(N_s,1);
var=zeros(N_s,1);
max_set_length=0;
tic;
for t=1:N_s
    
    %Calculate the set in a smarter way.
    [set1, set2, xu_set]=find_set2D(x_s(:,t),Omega,l_u1,l_u2,N_u,x1_u,x2_u,r);
    
    [set1is,set1js]=meshgrid(set1,set1);
    [set2is,set2js]=meshgrid(set2,set2);
    p1=(set1is-1)*M_max1+(set1js+1-set1is);
    p2=(set2is-1)*M_max2+(set2js+1-set2is);
    I_s=reshape(I(p1,p2),length(set1),length(set1),length(set2),length(set2));
    I_t_here=permute(I_s,[1 3 2 4]);
    
    M=length(set1)*length(set2);
    I_mat_recon=reshape(I_t_here,M,M);
    iota_mat_recon=reshape(iota(set1,set2),M,1);
    
    phi=Kern2D(x_s(:,t),xu_set,sigma_SE,l_SE);
    Kuu=Kern2D(xu_set,xu_set,sigma_SE,l_SE);
    Kss=Kern2D(x_s(:,t),x_s(:,t),sigma_SE,l_SE);
    Qss=phi*(Kuu\(phi'));
    temp=I_mat_recon+sigma_y^2.*Kuu;
    mu(t)=phi*(temp\iota_mat_recon);
    var(t)=diag(sigma_y^2*phi*(temp\(phi')))+Kss-Qss;
    max_set_length=max(max_set_length,M);
    
end
prediction_time=toc./N_s;

disp(['Number of inducing points in local domain: ',num2str(M_max1*M_max2)]);

end