clear; close all;
load('seadepth1.mat');
load('seadepth2.mat');
load('seadepth3.mat');

seadepth=[seadepth1; seadepth2; seadepth3];

%Define hyperparameters GP and RR-GP
Omega=[1 1; size(seadepth)]'; %Domain

%Hyperparameters
sigma_SE=std(seadepth(:));
l_SE=[4 4];
sigma_y=0.1*sigma_SE;

%Distance limiters
fontsize=14;
%Number of inducing inputs along dim 1
N_u(1)=ceil(0.35*(Omega(1,2)-Omega(1,1))./l_SE(1));
%Number of inducing inputs along dim 2
N_u(2)=ceil(0.35*(Omega(2,2)-Omega(2,1))./l_SE(2));
m=N_u(1)*N_u(2);
r=3*l_SE; %Radial basis function truncation

%Find x-locations
x1=Omega(1,1):Omega(1,2);
x2=Omega(2,1):Omega(2,2);
[X2,X1]=meshgrid(x2, x1);
x=[X1(:)'; X2(:)'];

%Find the coordinates of the inputs
x_meas=x;

%Make a low-res grid for prediction
res=10;
x1low=Omega(1,1):res:Omega(1,2);
x2low=Omega(2,1):res:Omega(2,2);
[X2low,X1low]=meshgrid(x2low, x1low);
xlow=[X1low(:)'; X2low(:)'];
ylow=seadepth(x1low,x2low);
ylow=ylow(:);

%Train on all measurements
y_meas=seadepth(:);

%Normalise the labels
avg_depth=mean(seadepth(:));
y_meas=y_meas-avg_depth;

[mu, variance, training_time, prediction_time]=FloatingDomainGP2D(x_meas,y_meas,xlow,Omega,N_u,sigma_SE,l_SE,r,sigma_y);
mu=mu+avg_depth;

save('WorkspaceProcessDataFewerInducingPoints.mat');

A=viridis();
%Normalise Y data
mu_normalised=reshape(mu,size(X1low))-min(min(reshape(mu,size(X1low))));
mu_normalised=mu_normalised./max(max(mu_normalised));
indices=ceil((mu_normalised).*254+1);

%Normalise Opacity
variance_normalized=reshape(variance-min(min(variance)),size(X1low));
variance_normalized=variance_normalized./max(max(variance_normalized));

%Quick calculation
ColorPicture=reshape(A(indices,:),size(mu_normalised,1),size(mu_normalised,2),3);
ColorPicture=flip(ColorPicture,1);
variance_normalized=flip(variance_normalized,1);
imagename='BathymetryPredictionsFewerInducingPoints';
imwrite(ColorPicture,[imagename,'.png'],'Alpha',variance_normalized);

%Save the original map as an image as well
seadepth_lower_res=seadepth(1:10:end,1:10:end);
seadepth_lifted=seadepth_lower_res-min(min(seadepth_lower_res));
seadepth_normalised=seadepth_lifted./max(max(seadepth_lifted));
indices=ceil((seadepth_normalised).*254+1);
ColorPictureFullMap=reshape(A(indices,:),size(seadepth_normalised,1),size(seadepth_normalised,2),3);
ColorPictureFullMap=flip(ColorPictureFullMap,1);
imagename='FullMap';
imwrite(ColorPicture,[imagename,'.png']);
