function [p_hat,times,q_hat,m]=EKF_scalable(N,delta_p,delta_q,y_mag,q_0,p_0,R_p,R_q,sigma_y,xl,xu,yl,yu,traj,l_SE,sigma_SE,plot_maps,r_thresh)
%Copyright (C) 2022 by Frida Viset

%Pre-allocate position trajectory for all particles
q_hat=zeros(4,N);
p_hat=zeros(3,N);
times=zeros(N,1);

%Pre-allocate storage for pose covariance estimate
P_pose_prior=zeros(6,6,N);
P_pose_posterior=zeros(6,6,N);

%Initialise orientation and position estimates
q_hat(:,1)=q_0;
p_hat(:,1)=p_0;

%Use the limits of the magnetic field norm to have reasonable limits for
%the plot
y_mag_norm=sqrt(sum(y_mag.^2,1));
cmin=min(y_mag_norm);
cmax=max(y_mag_norm);

%Initialisation for scalable magnetic field map
margin=3;
xli=min(traj(1,:))-margin;
xui=max(traj(1,:))+margin;
yli=min(traj(2,:))-margin;
yui=max(traj(2,:))+margin;
zli=min(traj(3,:))-margin;
zui=max(traj(3,:))+margin;
Omega=[xli, xui; yli, yui; zli, zui];
density=1;
N_u(1)=ceil(density*(Omega(1,2)-Omega(1,1))./l_SE); %Number of inducing inputs along dim 1
N_u(2)=ceil(density*(Omega(2,2)-Omega(2,1))./l_SE); %Number of inducing inputs along dim 2
N_u(3)=3; %Number of inducing inputs along dim 3

x1_u=linspace(Omega(1,1),Omega(1,2),N_u(1)); %Inducing inputs along first dimension
x2_u=linspace(Omega(2,1),Omega(2,2),N_u(2)); %Inducing inputs along second dimension
x3_u=linspace(min(traj(3,:)),max(traj(3,:)),N_u(3));
l_u1=x1_u(2)-x1_u(1); %Distance between two inducing inputs along first dimension
l_u2=x2_u(2)-x2_u(1); %Distance between two inducing inputs along second dimension
l_u3=x3_u(2)-x3_u(1); %Distance between two inducing inputs along second dimension

%Inducing point locations
[xu2_set,xu1_set,xu3_set]=meshgrid(x2_u,x1_u,x3_u);
xu_set_full=[xu1_set(:) xu2_set(:) xu3_set(:)]';

%Initialise the scalable magnetic field map
N_i=N_u(1)*N_u(2)*N_u(3);
Kuu=Kern3D(xu_set_full,xu_set_full,sigma_SE,l_SE);

Kuu_phi=[eye(3), zeros(3,N_i);...
         zeros(N_i,3), Kuu];

%Pre-allocate space for the position covariance
P=zeros(N_i+9,N_i+9);
P(1:6,1:6)=0.001*eye(6);

%Alternative covariance with inducing points
P(7:end,7:end)=inv(Kuu_phi);
m=zeros(N_i+3,1);

I=inv(P);

%Find the sparsity pattern
Indices_meas=[];
Indices_dyn=[];

Sparsity_pattern_meas=zeros(9+N_i,9+N_i);
Sparsity_pattern_dyn=zeros(9+N_i,9+N_i);
Sparsity_pattern=zeros(9+N_i,9+N_i);
[js,is]=meshgrid(9+1:N_i+9,1:9);
Indices_meas=[js(:)';is(:)'];
[js,is]=meshgrid(1:9,1:9);
Indices_meas=[Indices_meas,[js(:)';is(:)']];
[js,is]=meshgrid(1:9,9+1:N_i+9);
Indices_meas=[Indices_meas,[js(:)';is(:)']];
%figure; clf; surf(Sparsity_pattern,'Edgecolor','None');
for i=1:N_i
    for j=1:N_i
        if max(abs(xu_set_full(:,i)-xu_set_full(:,j)))<=2*r_thresh
            Indices_dyn=[Indices_dyn,[i+9; j+9]];
        else
        end
    end
end
Indices=[Indices_meas, Indices_dyn];
linear_indices=sub2ind(size(I),Indices(1,:),Indices(2,:));
linear_indices_meas=sub2ind(size(I),Indices_meas(1,:),Indices_meas(2,:));
linear_indices_dyn=sub2ind(size(I),Indices_dyn(1,:),Indices_dyn(2,:));
Sparsity_pattern_meas(linear_indices_meas)=1;
Sparsity_pattern_dyn(linear_indices_dyn)=1;
Sparsity_pattern=max(Sparsity_pattern_meas,Sparsity_pattern_dyn);
figure; clf; surf(flip(-Sparsity_pattern_meas,1),'Edgecolor','None'); view(2);
colors=viridis();
map = [[30, 77, 150]./255;
       [167, 200, 252]./255];
   box off; axis equal; axis off; colormap(map);
exportgraphics(gca,'Figures/SparsityMeas.png','Resolution',300);
figure; clf; surf(flip(-Sparsity_pattern_dyn,1),'Edgecolor','None'); view(2);
box off; axis equal; axis off; colormap(map);
exportgraphics(gca,'Figures/SparsityDyn.png','Resolution',300);
figure; clf; surf(flip(-Sparsity_pattern,1),'Edgecolor','None'); view(2);
box off; axis equal; axis off; colormap(map);
exportgraphics(gca,'Figures/Sparsity.png','Resolution',300);
     
%iterate through all timesteps
for t=2:N
    tic;
    
    %KF Dyn update
    p_hat(:,t)=p_hat(:,t-1)+delta_p(:,t-1);
    q_hat(:,t)=exp_q_L(delta_q(:,t-1),q_hat(:,t-1));


    Q=[R_p,zeros(3); zeros(3), R_q];
    I_ingredient_i=I(:,1:6)*inv(inv(Q)+I(1:6,1:6));
    I_ingredient_j=(I(1:6,:)');
    I_ingredient_both=I_ingredient_i(Indices(1,:),:).*I_ingredient_j(Indices(2,:),:);
    I(linear_indices)=I(linear_indices)-sum(I_ingredient_both,2)';
    
    if (mod(t,1000)==0 && plot_maps)
        figure; clf; 
        subplot(2,1,1);
        surf(log(abs(I_new)),'Edgecolor','None');
        view(2);
        subplot(2,1,2);
        surf(log(abs(I)),'Edgecolor','None');
        view(2);
        
        figure; clf;
        plot3(p_hat(1,1:t),p_hat(2,1:t),p_hat(3,1:t),'k');
        hold on;
        scatter3(p_hat(1,t),p_hat(2,t),p_hat(3,t),'k+');
        scatter3(xu_set_full(1,:),xu_set_full(2,:),xu_set_full(3,:),[],[0.5 0.5 0.5],'o');
        scatter3(xu_set(1,:),xu_set(2,:),xu_set(3,:),'ko');
        xlim([xl xu]);
        ylim([yl yu]);
        axis equal;
    end
    
    if (isnan(y_mag(1,t)))
        %Skip meas update
    else
        r=r_thresh;
        [set,xu_set]=find_set3D(p_hat(:,t),Omega,l_u1,l_u2,l_u3,N_u,x1_u,x2_u,x3_u,r);
        set_phi=[ (1:9)'; 9+set];
        
        %KF meas update
        NablaPhi=[eye(3), NablaKern3D(p_hat(:,t),xu_set,sigma_SE,l_SE)];
        f=NablaPhi*m([ (1:3)'; 3+set]);
        JacobianPhi=JacobianKern3D(p_hat(:,t),xu_set,sigma_SE,l_SE);
        N_i_here=size(xu_set,2);
        J=reshape(JacobianPhi,9,N_i_here)*m(3+set);
        J=reshape(J,3,3);
        
        %Prepare scew-symmetric magnetic field vector
        fx=[0 -f(3) f(2);
            f(3) 0 -f(1);
            -f(2) f(1) 0];
        
        
        %Meas state update full EKF
        H_t=[J, fx, NablaPhi];
        eps_t=quat2Rot(q_hat(:,t))*y_mag(:,t)-(f);
        iota_t=1./sigma_y^2*H_t'*eps_t;
        iota_full=zeros(N_i+9,1);
        iota_full(set_phi)=iota_t;

        %Meas update
        I(set_phi,set_phi)=I(set_phi,set_phi)+1./sigma_y^2*(H_t'*H_t);
        
        r=0.5*r_thresh;
        [set]=find_set3D(p_hat(:,t),Omega,l_u1,l_u2,l_u3,N_u,x1_u,x2_u,x3_u,r);
        set_phi=[ (1:9)'; 9+set];
        iota_t=iota_full(set_phi);
        eta_t=(I(set_phi,set_phi))\iota_t;
        p_hat(:,t)=p_hat(:,t)+eta_t(1:3);
        q_hat(:,t)=exp_q_L(eta_t(4:6),q_hat(:,t));
        m([ (1:3)'; 3+set])=m([ (1:3)'; 3+set])+eta_t(7:end);
        
    end
    times(t)=toc;
end