clear;
close all;

%Copyright (C) 2023 by Frida Viset

disp('Running Algorithm 1 on data from foot-mounted sensor');

%Set seed
seed=42;
rng(seed);

time=clock;

%Set room dimensions
margin=10;

%Load dataset
load('OpenShoeOdometryAndMagField.mat');

%Run dead reckoning on the simplified odometry
[p_DR,q_DR]=Simple_Dead_Reckoning(delta_p,delta_q,p_0,q_0);

%Estimate the dimensions of the problem based on the first round
traj=p_DR(:,1:800);
xl=min(traj(1,:))-margin;
xu=max(traj(1,:))+margin;
yl=min(traj(2,:))-margin;
yu=max(traj(2,:))+margin;
zl=min(traj(3,:))-margin;
zu=max(traj(3,:))+margin;

%Magnetic field params
sigma_SE=1;
l_SE=2;
sigma_lin=1;
sigma_y=0.1;

%Number of basis functions used in Reduced-Rank approximation
%N_m=decide_number_of_basis_functions(xl,xu,yl,yu,zl,zu,margin,sigma_SE,l_SE,sigma_y);
N_m=1850;

%Calculate Lambda and the order of indices used in the
%analytic basis functions of the Reduced-Rank Approximation
[Indices, Lambda]=Lambda3D(N_m,xl,xu,yl,yu,zl,zu,sigma_SE,l_SE);

%Noise parameters
R_p=0.001*eye(3)*T;
R_q=0.00001*eye(3)*T;

%Init covariances
P_0=0.0001*eye(6);

%Prepare plotting
fontsize=14;

%Run the filter

plot_maps=false;
rs=[1.5*l_SE, 2*l_SE, 2.5*l_SE];
p_Ind=zeros(3,N,3);
times_Ind=zeros(N,3);
for i=1:3
[p_scalable,times]=EKF_scalable(N,delta_p,delta_q,y_mag,q_0,p_0,R_p,R_q,sigma_y,xl,xu,yl,yu,traj,l_SE,sigma_SE,plot_maps,rs(i));
p_Ind(:,:,i)=p_scalable;
times_Ind(:,i)=times;
end

[p_EKF,times_EKF,q_EKF,m]=EKF_Hilbert(N,delta_p,delta_q,y_mag,q_0,p_0,R_p,R_q,sigma_y,Lambda,Indices,sigma_lin,N_m,xl,xu,yl,yu,zl,zu);


%% Plotting section

figure; clf;
plot3(p_DR(1,:),p_DR(2,:),p_DR(3,:),'b');
view(2);
xlabel('$x($m$)$','Interpreter','Latex','Fontsize',fontsize);
ylabel('$y($m$)$','Interpreter','Latex','Fontsize',fontsize);
zlabel('$z$(m$)$','Interpreter','Latex','Fontsize',fontsize);
axis equal;
xlim([xl+2 xu-7]);
ylim([yl+4 yu-9]);
set(gca,'TickLabelInterpreter','latex');
set(gca,'XTick',[-15 -10 -5 0 5 10]);
set(gca,'YTick',[-15 -10 -5 0 5 10 15 20 25 30]);
set(gca,'xticklabel',({'$-15$','$-10$','$-5$','$0$','$5$','$10$','$15$'}));
set(gca,'yticklabel',({'$-15$','$-10$','$-5$','$0$','$5$','$10$','$15$','$20$','$25$','$30$'}));
title('Odometry','Interpreter','Latex');
grid on;
exportgraphics(gca,'Figures/Odometry.png','Resolution',300);

for i=1:3
figure; clf;
hold on;
plot3(p_Ind(1,:,i),p_Ind(2,:,i),p_Ind(3,:,i),'k');
view(2);
xlabel('$x($m$)$','Interpreter','Latex','Fontsize',fontsize);
ylabel('$y($m$)$','Interpreter','Latex','Fontsize',fontsize);
zlabel('$z$(m$)$','Interpreter','Latex','Fontsize',fontsize);
axis equal;
xlim([xl+2 xu-7]);
ylim([yl+4 yu-9]);
set(gca,'TickLabelInterpreter','latex');
set(gca,'XTick',[-15 -10 -5 0 5 10]);
set(gca,'YTick',[-15 -10 -5 0 5 10 15 20 25 30]);
set(gca,'xticklabel',({'$-15$','$-10$','$-5$','$0$','$5$','$10$','$15$'}));
set(gca,'yticklabel',({'$-15$','$-10$','$-5$','$0$','$5$','$10$','$15$','$20$','$25$','$30$'}));
title(['$r=',num2str(rs(i)./l_SE),'l_{SE}$'],'Interpreter','Latex');
grid on;
exportgraphics(gca,['Figures/r=',num2str(rs(i)./l_SE),'l_SE.png'],'Resolution',300);
end

figure; clf;
hold on;
plot3(p_EKF(1,:),p_EKF(2,:),p_EKF(3,:),'k','linewidth',1.2);
%plot3(p_DR(1,:),p_DR(2,:),p_DR(3,:),'b');
view(2);
xlabel('$x($m$)$','Interpreter','Latex','Fontsize',fontsize);
ylabel('$y($m$)$','Interpreter','Latex','Fontsize',fontsize);
zlabel('$z$(m$)$','Interpreter','Latex','Fontsize',fontsize);
axis equal;
xlim([xl+2 xu-7]);
ylim([yl+4 yu-9]);
set(gca,'TickLabelInterpreter','latex');
grid on;
set(gca,'XTick',[-15 -10 -5 0 5 10]);
set(gca,'YTick',[-15 -10 -5 0 5 10 15 20 25 30]);
set(gca,'xticklabel',({'$-15$','$-10$','$-5$','$0$','$5$','$10$','$15$'}));
set(gca,'yticklabel',({'$-15$','$-10$','$-5$','$0$','$5$','$10$','$15$','$20$','$25$','$30$'}));
title('Hilbert space basis functions','Interpreter','Latex');
exportgraphics(gca,'Figures/HilbertSpaceEKF.png','Resolution',300);

%% Arrange with floorplan

floor=imread('planritning.png');
floorvec=reshape(255-floor,[size(floor,1)*size(floor,2),3]);
floor_grayscale_vec=255-max(floorvec');
floor_grayscale=reshape(floor_grayscale_vec,[size(floor,1),size(floor,2)]);
floor_cropped=floor_grayscale(715:945,1165:1815);
figure; clf;
imshow(floor_cropped);

figure; clf;
ys=yl+7;
ye=yu-7;
xs=xl;
xe=xs+(ys-ye)*(945-715)./(1815-1165);
[Y,X]=meshgrid(linspace(ys,ye,size(floor_cropped,2)),linspace(xs,xe,size(floor_cropped,1)));
surf(X,Y,0.*X-5,floor_cropped,'EdgeColor','None');
blue=gray;
blue(:,3)=1; 
blue(:,2)=0.5+0.5.*blue(:,2);
colormap(blue);
view(2);
axis equal;
hold on;
grid off;
box off;
axis off;
shift_x=20;
R=rotz(10);
p_rot=zeros(3,N);
for t=1:N
    p_rotDR(:,t)=R*p_DR(:,t);
end
plot3(p_rotDR(1,:)-shift_x,p_rotDR(2,:),p_rotDR(3,:),'r','linewidth',1.3);
exportgraphics(gca,'Figures/Odometry.png','Resolution',300);
xlims=xlim();
ylims=ylim();

%Combine the floorplan and the trajectory
for i=1:3
figure; clf;
ys=yl+7;
ye=yu-7;
xs=xl;
xe=xs+(ys-ye)*(945-715)./(1815-1165);
[Y,X]=meshgrid(linspace(ys,ye,size(floor_cropped,2)),linspace(xs,xe,size(floor_cropped,1)));
surf(X,Y,0.*X-5,floor_cropped,'EdgeColor','None');
blue=gray;
blue(:,3)=1; 
blue(:,2)=0.5+0.5.*blue(:,2);
colormap(blue);
view(2);
axis equal;
hold on;
grid off;
box off;
axis off;
shift_x=20;
R=rotz(10);
p_rot=zeros(3,N);
for t=1:N
    p_rot(:,t)=R*p_Ind(:,t,i);
end
plot3(p_rotDR(1,:)-shift_x,p_rotDR(2,:),p_rotDR(3,:)-16,'Color',[254 254 254]./255,'linewidth',1.3);
plot3(p_rot(1,:)-shift_x,p_rot(2,:),p_rot(3,:),'k','linewidth',1.3);
xlim(xlims);
ylim(ylims);
exportgraphics(gca,['Figures/r=',num2str(rs(i)./l_SE),'l_SE.png'],'Resolution',300);
end


figure; clf;
ys=yl+7;
ye=yu-7;
xs=xl;
xe=xs+(ys-ye)*(945-715)./(1815-1165);
[Y,X]=meshgrid(linspace(ys,ye,size(floor_cropped,2)),linspace(xs,xe,size(floor_cropped,1)));
surf(X,Y,0.*X-5,floor_cropped,'EdgeColor','None');
blue=gray;
blue(:,3)=1; 
blue(:,2)=0.5+0.5.*blue(:,2);
colormap(blue);
view(2);
axis equal;
hold on;
grid off;
box off;
axis off;
shift_x=20;
R=rotz(10);
p_rot=zeros(3,N);
for t=1:N
    p_rot(:,t)=R*p_EKF(:,t);
end
plot3(p_rotDR(1,:)-shift_x,p_rotDR(2,:),p_rotDR(3,:)-16,'Color',[254 254 254]./255,'linewidth',1.3);
plot3(p_rot(1,:)-shift_x,p_rot(2,:),p_rot(3,:),'k','linewidth',1.3);
xlim(xlims);
ylim(ylims);
exportgraphics(gca,'Figures/HilbertSpaceEKF.png','Resolution',300);

for i=1:3
    avgtime=mean(times_Ind(:,i)*1000);
    stdtime=std(times_Ind(:,i)*1000);
    disp(['Average computation time for r=',num2str(rs(i)./l_SE),' is ',num2str(avgtime),'+-',num2str(stdtime),' milliseconds']); 
end
avgtime=mean(times_EKF*1000);
stdtime=std(times_EKF*1000);
disp(['Average computation time for the Hilbert Space Basis function filter is ',num2str(avgtime),'+-',num2str(stdtime),' milliseconds']);