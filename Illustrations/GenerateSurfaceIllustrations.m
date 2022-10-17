%Copyright (C) 2022 by Frida Viset

clear; close all;

seed=100;
rng(seed);

%Define a domain
Omega=[-1.5 1.5; -1.5 1.5]; %Domain
res=0.025;
x1=Omega(1,1):res:Omega(1,2);
x2=Omega(2,1):res:Omega(2,2);
[X1,X2]=meshgrid(x1, x2);
x=[X1(:)'; X2(:)'];

%Define hyperparameters
sigma_SE=0.1;
l_SE=0.17;
sigma_y=0.01;

%Set fontize and colours for plots
fontsize=19;
white=[1,1,1];
black=[0,0,0];
gray=[0.5,0.5,0.5];

%Define global inducing point grid
N_u1=ceil(1*(Omega(1,2)-Omega(1,1))./l_SE); %Number of inducing inputs along dim 1
N_u2=ceil(1*(Omega(2,2)-Omega(2,1))./l_SE); %Number of inducing inputs along dim 2
m=N_u1*N_u2;
r=3*l_SE;
x1_u=linspace(Omega(1,1),Omega(1,2),N_u1); %Inducing inputs along first dimension
x2_u=linspace(Omega(2,1),Omega(2,2),N_u2); %Inducing inputs along second dimension
l_u1=x1_u(2)-x1_u(1); %Distance between two inducing inputs along dim 1
l_u2=x2_u(2)-x2_u(1); %Distance between two inducing inputs along dim 2
[U1,U2]=meshgrid(x1_u,x2_u);

%Select measurement locations according to a uniform distribution over the
%considered domain
measured_samples=2000;
sz1=size(x1,2); sz2=size(x2,2);
indices1_measured=randi(floor(sz1),measured_samples,1);
indices2_measured=randi(floor(sz2),measured_samples,1);
x_measured=[x1(indices1_measured); x2(indices2_measured)];

%Sample measurements from the GP prior
f=mvnrnd(zeros(measured_samples,1),Kern(x_measured,x_measured,sigma_SE,l_SE));
y_measured=f+mvnrnd(zeros(measured_samples,1),sigma_y.^2*eye(measured_samples));

%Perform a full field prediction
[mu_GP, var]=GaussianProcess(x_measured,y_measured,x,sigma_y,sigma_SE,l_SE);

%Set a single prediction point
x_star=[0; 0];

%Find the inducing points that surrounds the prediction point
[set1, set2, xu_set]=find_set(x_star,Omega,l_u1,l_u2,N_u1,N_u2,x1_u,x2_u,r);

%Perform the full field prediction, but only conditioned on inducing points
%surrounding the prediction location
[mu, var, max_set_length]=LocalInducingInputsGaussianProcess2DFixedPredPoint(x_measured,y_measured,x,x_star,Omega,N_u1,N_u2,sigma_SE,l_SE,r,sigma_y);

%A flat one, with all the information content
figure; clf;
mu=reshape(mu,size(X1));
variances_RR=sqrt(var);
Opacity=-reshape(variances_RR,size(X1));
mu_GP=reshape(mu_GP,size(X1));
contour(X1,X2,mu_GP,10,'EdgeColor',gray,'ContourZLevel',-0.03);
hold on;
s2.EdgeColor = 'none';
height=0;
s1=surf(X1,X2,height-0.05+0*mu,mu,'AlphaData',Opacity,'FaceAlpha','flat');
s1.EdgeColor = 'none';
scatter3(U1(:),U2(:),0.*U1(:),20+0.*U1(:),'o','MarkerEdgeColor',gray,'LineWidth',1);
scatter3(xu_set(1,:),xu_set(2,:),0.01+0.*xu_set(1,:),20+0.*xu_set(1,:),'o','LineWidth',1.2,'MarkerEdgeColor',black);
plot([Omega(1,1) Omega(1,2)],[x_star(2) x_star(2)],'LineWidth',1.5,'Color',gray);
plot([x_star(1) x_star(1)],[Omega(2,1) Omega(2,2)],'LineWidth',1.5,'Color',gray);
width=0.15;
plot3([x_star(1)-width x_star(1)+width],[x_star(2) x_star(2)],[0.01 0.01],'LineWidth',3,'Color',black);
plot3([x_star(1) x_star(1)],[x_star(2)-width x_star(2)+width],[0.01 0.01],'LineWidth',3,'Color',black);
xlim([Omega(1,1) Omega(1,2)]);
ylim([Omega(2,1) Omega(2,2)]);
xlabel('$x_1$','Fontsize',fontsize,'Interpreter','Latex');
ylabel('$x_2$','Fontsize',fontsize,'Interpreter','Latex');
xticks([]);
yticks([]);
box off;
grid off;
view(2);
colormap(viridis());
axis equal;
exportgraphics(gca,'Flat0.png','Resolution',500);


%The flow of it
figure; clf;
contour3(X1,X2,mu_GP,10,'EdgeColor',gray);
hold on;
s2=surf(X1,X2,mu,'AlphaData',Opacity,'FaceAlpha','flat');
s2.EdgeColor = 'none';
xlim([Omega(1,1) Omega(1,2)]);
ylim([Omega(2,1) Omega(2,2)]);
grid off;
view([325 25]);
colormap(viridis());
plot3([Omega(1,1) Omega(1,2)],[x_star(2) x_star(2)],[0 0],'LineWidth',1.5,'Color',gray);
plot3([x_star(1) x_star(1)],[Omega(2,1) Omega(2,2)],[0 0],'LineWidth',1.5,'Color',gray);
width=0.15;
plot3([x_star(1)-width x_star(1)+width],[x_star(2) x_star(2)],[0.01 0.01],'LineWidth',3,'Color',black);
plot3([x_star(1) x_star(1)],[x_star(2)-width x_star(2)+width],[0.01 0.01],'LineWidth',3,'Color',black);
xlabel('$x_1$','Fontsize',fontsize,'Interpreter','Latex');
ylabel('$x_2$','Fontsize',fontsize,'Interpreter','Latex');
zlabel('$y$','Fontsize',fontsize,'Interpreter','Latex');
xticks([]);
yticks([]);
zticks([]);
box off;
axis equal;
exportgraphics(gca,'Tilted0.png','Resolution',800);

%Repeat for different inducing points
x_star=[-0.16; -0.7];

%Find the inducing points that surrounds the prediction point
[set1, set2, xu_set]=find_set(x_star,Omega,l_u1,l_u2,N_u1,N_u2,x1_u,x2_u,r);

%Perform the full field prediction, but only conditioned on inducing points
%surrounding the prediction location
[mu, var, max_set_length]=LocalInducingInputsGaussianProcess2DFixedPredPoint(x_measured,y_measured,x,x_star,Omega,N_u1,N_u2,sigma_SE,l_SE,r,sigma_y);

%A flat one, with all the information content
figure; clf;
mu=reshape(mu,size(X1));
variances_RR=sqrt(var);
Opacity=-reshape(variances_RR,size(X1));
mu_GP=reshape(mu_GP,size(X1));
contour(X1,X2,mu_GP,10,'EdgeColor',gray,'ContourZLevel',-0.03);
hold on;
s2.EdgeColor = 'none';
height=0;
s1=surf(X1,X2,height-0.05+0*mu,mu,'AlphaData',Opacity,'FaceAlpha','flat');
s1.EdgeColor = 'none';
scatter3(U1(:),U2(:),height+0.*U1(:),20+0.*U1(:),'o','MarkerEdgeColor',gray,'LineWidth',1);
scatter3(xu_set(1,:),xu_set(2,:),0.01+height+0.*xu_set(1,:),20+0.*xu_set(1,:),'o','LineWidth',1.2,'MarkerEdgeColor',black);
plot([Omega(1,1) Omega(1,2)],[x_star(2) x_star(2)],'LineWidth',1.5,'Color',gray);
plot([x_star(1) x_star(1)],[Omega(2,1) Omega(2,2)],'LineWidth',1.5,'Color',gray);
width=0.15;
plot3([x_star(1)-width x_star(1)+width],[x_star(2) x_star(2)],[0.01 0.01],'LineWidth',3,'Color',black);
plot3([x_star(1) x_star(1)],[x_star(2)-width x_star(2)+width],[0.01 0.01],'LineWidth',3,'Color',black);
xlim([Omega(1,1) Omega(1,2)]);
ylim([Omega(2,1) Omega(2,2)]);
xlabel('$x_1$','Fontsize',fontsize,'Interpreter','Latex');
ylabel('$x_2$','Fontsize',fontsize,'Interpreter','Latex');
xticks([]);
yticks([]);
box off;
grid off;
view(2);
colormap(viridis());
axis equal;
exportgraphics(gca,'Flat1.png','Resolution',500);

%Repeat for different inducing points
x_star=[0.35; 0.18];

%Find the inducing points that surrounds the prediction point
[set1, set2, xu_set]=find_set(x_star,Omega,l_u1,l_u2,N_u1,N_u2,x1_u,x2_u,r);

%Perform the full field prediction, but only conditioned on inducing points
%surrounding the prediction location
[mu, var, max_set_length]=LocalInducingInputsGaussianProcess2DFixedPredPoint(x_measured,y_measured,x,x_star,Omega,N_u1,N_u2,sigma_SE,l_SE,r,sigma_y);

%A flat one, with all the information content
figure; clf;
mu=reshape(mu,size(X1));
variances_RR=sqrt(var);
Opacity=-reshape(variances_RR,size(X1));
mu_GP=reshape(mu_GP,size(X1));
contour(X1,X2,mu_GP,10,'EdgeColor',gray,'ContourZLevel',-0.03);
hold on;
%s2=surf(X1,X2,mu,'AlphaData',Opacity,'FaceAlpha','flat');
s2.EdgeColor = 'none';
height=0;
s1=surf(X1,X2,height-0.05+0*mu,mu,'AlphaData',Opacity,'FaceAlpha','flat');
s1.EdgeColor = 'none';
scatter3(U1(:),U2(:),height+0.*U1(:),20+0.*U1(:),'o','MarkerEdgeColor',gray,'LineWidth',1);
scatter3(xu_set(1,:),xu_set(2,:),0.01+height+0.*xu_set(1,:),20+0.*xu_set(1,:),'o','LineWidth',1.2,'MarkerEdgeColor',black);
plot([Omega(1,1) Omega(1,2)],[x_star(2) x_star(2)],'LineWidth',1.5,'Color',gray);
plot([x_star(1) x_star(1)],[Omega(2,1) Omega(2,2)],'LineWidth',1.5,'Color',gray);
width=0.15;
plot3([x_star(1)-width x_star(1)+width],[x_star(2) x_star(2)],[0.01 0.01],'LineWidth',3,'Color',black);
plot3([x_star(1) x_star(1)],[x_star(2)-width x_star(2)+width],[0.01 0.01],'LineWidth',3,'Color',black);
xlim([Omega(1,1) Omega(1,2)]);
ylim([Omega(2,1) Omega(2,2)]);
grid off;
xlabel('$x_1$','Fontsize',fontsize,'Interpreter','Latex');
ylabel('$x_2$','Fontsize',fontsize,'Interpreter','Latex');
xticks([]);
yticks([]);
box off;
view(2);
colormap(viridis());
axis equal;
exportgraphics(gca,'Flat2.png','Resolution',500);