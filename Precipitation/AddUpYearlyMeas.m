%Copyright (C) 2022 by Frida Viset

clear; close all;
load('precip_data.mat');

%Concatenate all the data
X_all=[X; Xtest];
y_all=[y; ytest];


%Find the locations of the unique weather stations
x_yearly=unique(X_all(:,1:2),'rows');

%For each weather station, calculate y as the sum at each weather station
y_yearly=zeros(size(x_yearly,1),1);
for i=1:size(x_yearly,1)
    datapoints=find((X(:,1)==x_yearly(i,1))&(X(:,2)==x_yearly(i,2)));
    y_yearly(i)=sum(y_all(datapoints));
end

%% Plot it

figure; clf;
scatter(x_yearly(:,1),x_yearly(:,2),10+0.*y_yearly,y_yearly);
caxis([0 8000]);

%Find the domain borders
OmegaMap(1,1)=min(x_yearly(:,1)-2.4);
OmegaMap(1,2)=max(x_yearly(:,1)+2.7);
OmegaMap(2,1)=min(x_yearly(:,2));
OmegaMap(2,2)=max(x_yearly(:,2)+2);

black=[0 0 0];

Map=imread('MapOfTheUS.png');
Map=flip(Map,1);
figure; clf;
x1map=linspace(OmegaMap(1,1),OmegaMap(1,2),size(Map,2));
x2map=linspace(OmegaMap(2,1),OmegaMap(2,2),size(Map,1));
[X1map,X2map]=meshgrid(x1map, x2map);
scatter(x_yearly(:,1),x_yearly(:,2),10+0.*y_yearly,y_yearly);
hold on;
Borders=Map(:,:,1)-0.5*Map(:,:,2)-0.5*Map(:,:,3); %Find the part of the map that is red
contour(X1map,X2map,Borders,20:20:200,'EdgeColor',black,'Linewidth',1.5);
save('precip_data_yearly.mat','y_yearly','x_yearly');
view(2);
