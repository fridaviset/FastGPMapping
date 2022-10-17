%Copyright (C) 2022 by Frida Viset

clear; close all;
load('precip_data_yearly.mat');

%Center the data
y_mean=mean(y_yearly);
y_yearly=y_yearly-y_mean;

%Find the domain borders
margin=4;
Omega(1,1)=min(x_yearly(:,1))-margin;
Omega(1,2)=max(x_yearly(:,1))+margin;
Omega(2,1)=min(x_yearly(:,2))-margin;
Omega(2,2)=max(x_yearly(:,2))+margin;

%Hyperparameters for Gaussian process prior
sigma_SE=3.99*sqrt(365);
l_SE=[3.094, 2.030];
sigma_y=2.789*sqrt(365);

%Set fontsize and colours for plots
fontsize=45;
white=[1 1 1];
black=[0 0 0];
gray=[0.5 0.5 0.5];

%Place the inducing points
N_u1=ceil(2*(Omega(1,2)-Omega(1,1))./l_SE(1)); %Number of inducing inputs along dim 1
N_u2=ceil(2*(Omega(2,2)-Omega(2,1))./l_SE(2)); %Number of inducing inputs along dim 2
r_star=3*l_SE; %Distance limiter for measurements used in LI inference

%Find the full grid of inducing points
x1_u=linspace(Omega(1,1),Omega(1,2),N_u1); %Inducing inputs along first dimension
x2_u=linspace(Omega(2,1),Omega(2,2),N_u2); %Inducing inputs along second dimension
[x1_um,x2_um]=meshgrid(x1_u,x2_u);
x_u=[x1_um(:);x2_um(:)];

%Make a grid for full field evaluation
res=0.25;
x1=Omega(1,1):res:Omega(1,2);
x2=Omega(2,1):res:Omega(2,2);
[X1,X2]=meshgrid(x1, x2);
x=[X1(:)'; X2(:)'];

%Plot full GP field
[muGP, varGP]=GaussianProcess(x_yearly',y_yearly',x,sigma_y,sigma_SE,l_SE);

%Add back the subtracted mean to the prediction
muGP=muGP+y_mean;

%Plot the Local approximations
[mu, var]=PredictionPointDependentGP2D(x_yearly',y_yearly',x,Omega,N_u1,N_u2,sigma_SE,l_SE,r_star,sigma_y);

%Add back the subtracted mean to the prediction
mu=mu+y_mean;

%Add back the subtracted mean to the measurements
y_yearly=y_yearly+y_mean;

OmegaMap(1,1)=min(x_yearly(:,1)-2.4);
OmegaMap(1,2)=max(x_yearly(:,1)+2.7);
OmegaMap(2,1)=min(x_yearly(:,2));
OmegaMap(2,2)=max(x_yearly(:,2)+2);

%Plot the measurements
Map=imread('MapOfTheUS.png');
Map=flip(Map,1);
figure; clf;
scatter(x_yearly(:,1),x_yearly(:,2),10+0.*y_yearly,y_yearly);
hold on;
x1map=linspace(OmegaMap(1,1),OmegaMap(1,2),size(Map,2));
x2map=linspace(OmegaMap(2,1),OmegaMap(2,2),size(Map,1));
[X1map,X2map]=meshgrid(x1map, x2map);
Borders=Map(:,:,1)-0.5*Map(:,:,2)-0.5*Map(:,:,3); %Find the part of the map that is red
contour(X1map,X2map,Borders,20:20:200,'EdgeColor',black,'Linewidth',1.5);
axis off;
caxis([0 5000]);
axis equal;
xlim([Omega(1,1) Omega(1,2)]);
ylim([Omega(2,1) Omega(2,2)]);
set(gca, 'FontName', 'Times');
set(gca,'fontsize',fontsize);
set(gca,'TickLabelInterpreter','latex');
colormap(viridis());
view(2);
exportgraphics(gca,'PrecipitationMeasurements.png','Resolution',500);

%Plot results
figure; clf;
Opacity=-reshape(varGP,size(X1));
muGP=reshape(muGP,size(X1));
surf(X1,X2,-0.1+0*X2,muGP,'EdgeColor','None','AlphaData',Opacity,'FaceAlpha','flat');
caxis([0 5000]);
axis off;
hold on;
axis equal;
colormap(viridis());
set(gca, 'FontName', 'Times');
set(gca,'fontsize',fontsize);
set(gca,'TickLabelInterpreter','latex');
grid off;
contour(X1map,X2map,Borders,20:20:200,'EdgeColor',black,'Linewidth',1.5);
view(2);
exportgraphics(gca,'PrecipitationMapFullGP.png','Resolution',500);

figure; clf;
Opacity=-reshape(varGP,size(X1));
mu=reshape(mu,size(X1));
surf(X1,X2,-0.1+0*X2,mu,'EdgeColor','None','AlphaData',Opacity,'FaceAlpha','flat');
hold on;
contour(X1map,X2map,Borders,20:20:200,'EdgeColor',black,'Linewidth',1.5);
view(2);
set(gca,'TickLabelInterpreter','latex');
grid off;
axis equal;
colormap(viridis());
axis off;
set(gca, 'FontName', 'Times');
set(gca,'fontsize',fontsize);
caxis([0 5000]);
exportgraphics(gca,'PrecipitationMapApproximation.png','Resolution',500);

%Plot the common colorbar for all maps
figure; clf;
axis off;
caxis([0 5000]);
colormap(viridis());
c=colorbar;
c.LineWidth=5;
c.Position=c.Position+[-0.3 0 0.1 0];
set(c,'YTick',[1000 2500 4000],'TickLabelInterpreter','latex','Fontsize',fontsize,'Color', 'k');
exportgraphics(gca,'ColorBar.png','Resolution',500');