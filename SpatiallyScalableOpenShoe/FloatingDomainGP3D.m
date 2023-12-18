function [mu, var, tts, pts, max_set_length]=FloatingDomainGP3D(x,y,x_s,Omega,N_u,sigma_SE,l_SE,r,sigma_y)
x1_u=linspace(Omega(1,1),Omega(1,2),N_u(1)); %Inducing inputs along first dimension
x2_u=linspace(Omega(2,1),Omega(2,2),N_u(2)); %Inducing inputs along second dimension
x3_u=linspace(Omega(3,1),Omega(3,2),N_u(3)); %Inducing inputs along third dimension
l_u1=x1_u(2)-x1_u(1); %Distance between two inducing inputs along first dimension
l_u2=x2_u(2)-x2_u(1); %Distance between two inducing inputs along second dimension
l_u3=x3_u(2)-x3_u(1); %Distance between two inducing inputs along third dimension

%Now, store and refer to bigger structures
M_max1=ceil(2*r(1)./l_u1);
M_max2=ceil(2*r(2)./l_u2);
M_max3=ceil(2*r(3)./l_u3);

%See if you can perform linear indexing along each dimension.
I=zeros(N_u(1)*M_max1,N_u(2)*M_max2,N_u(3)*M_max3);
iota=zeros(N_u(1),N_u(2),N_u(3));


%Learning
N=size(x,2);
training_times=zeros(N,1);
for t=1:N
    tic;
    %Update the set where basis functions are non-zero
    [set1, set2, set3, xu_set]=find_set3D(x(:,t),Omega,l_u1,l_u2,l_u3,N_u,x1_u,x2_u,x3_u,r);
    
    %Perform the information matrix update
    phi=Kern3D(x(:,t),xu_set,sigma_SE,l_SE);
    I_mat=phi'*phi;
    iota_mat=phi'*y(t);
    
    %Find I_temp on tensor form
    I_k=reshape(I_mat,length(set1),length(set2),length(set3),length(set1),length(set2),length(set3));
    I_kb=permute(I_k,[1 4 2 5 3 6]);

    [set1is,set1js]=meshgrid(set1,set1);
    [set2is,set2js]=meshgrid(set2,set2);
    [set3is,set3js]=meshgrid(set3,set3);
    p1=(set1is-1)*M_max1+(set1js+1-set1is);
    p2=(set2is-1)*M_max2+(set2js+1-set2is);
    p3=(set3is-1)*M_max3+(set3js+1-set3is);
    I(p1,p2,p3)=I(p1,p2,p3)+reshape(I_kb,length(set1)*length(set1),length(set2)*length(set2),length(set3)*length(set3));
    
    iota(set1,set2,set3)=iota(set1,set2,set3)+reshape(iota_mat,length(set1),length(set2),length(set3));
    training_times(t)=toc;
end
tts=training_times;

%Prediction
N_s=size(x_s,2);
mu=zeros(N_s,1);
var=zeros(N_s,1);
max_set_length=0;
prediction_times=zeros(N_s,1);
for t=1:N_s
    tic;
    [set1, set2, set3, xu_set]=find_set3D(x_s(:,t),Omega,l_u1,l_u2,l_u3,N_u,x1_u,x2_u,x3_u,r./2);
    
    [set1is,set1js]=meshgrid(set1,set1);
    [set2is,set2js]=meshgrid(set2,set2);
    [set3is,set3js]=meshgrid(set3,set3);
    p1=(set1is-1)*M_max1+(set1js+1-set1is);
    p2=(set2is-1)*M_max2+(set2js+1-set2is);
    p3=(set3is-1)*M_max3+(set3js+1-set3is);
    I_s=reshape(I(p1,p2,p3),length(set1),length(set1),length(set2),length(set2),length(set3),length(set3));
    I_t_here=permute(I_s,[1 3 5 2 4 6]);
    
    M=length(set1)*length(set2)*length(set3);
    I_mat_recon=reshape(I_t_here,M,M);
    iota_mat_recon=reshape(iota(set1,set2,set3),M,1);
    
    phi=Kern3D(x_s(:,t),xu_set,sigma_SE,l_SE);
    Kuu=Kern3D(xu_set,xu_set,sigma_SE,l_SE);
    Kss=Kern3D(x_s(:,t),x_s(:,t),sigma_SE,l_SE);
    Qss=phi*(Kuu\(phi'));
    temp=I_mat_recon+sigma_y^2.*Kuu;
    mu(t)=phi*(temp\iota_mat_recon);
    var(t)=diag(sigma_y^2*phi*(temp\(phi')));
    max_set_length=max(max_set_length,M);
    prediction_times(t)=toc;
end
disp(['r=',num2str(r),', M=',num2str(max_set_length)]);
pts=prediction_times;


end