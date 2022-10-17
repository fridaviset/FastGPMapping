%Copyright (C) 2022 by Frida Viset

clear; close all;
load('precip_data.mat');

%Center the data
y_mean=mean(y);
y=y-y_mean;
ytest=ytest-y_mean;

%Find the domain borders
Omega(1,1)=min(X(:,1))-10;
Omega(1,2)=max(X(:,1))+10;
Omega(2,1)=min(X(:,2))-10;
Omega(2,2)=max(X(:,2))+10;
Omega(3,1)=min(X(:,3));
Omega(3,2)=max(X(:,3));

%Hyperparameters for Gaussian process prior
sigma_SE=3.99;
l_SE=[3.094, 2.030, 0.189];
sigma_y=2.789;

dist_factors=[2 2.5 3];
params=size(dist_factors,2);

repetitions=20;
experiments=15;
SMSEs=zeros(params,experiments,repetitions);
PreProcessTimes=zeros(params,experiments,repetitions);
InferenceTimes=zeros(params,experiments,repetitions);

ms=round(logspace(log(10000)./log(10),log(803000)./log(10),experiments));

tStart=tic;
for repetition=1:repetitions
    
    N_u=zeros(3,1);
    
    for i=1:params
        for j=1:experiments
            
            %Place the inducing points
            m=ms(j);
            N_u(3)=365; %Number of inducing points along dim 3 - equivalent to number of days?
            L_1=Omega(1,2)-Omega(1,1);
            L_2=Omega(2,2)-Omega(2,1);
            N_u(1)=round(sqrt(m./N_u(3).*(L_1*l_SE(2))./(L_2*l_SE(1)))); %Number of inducing inputs along dim 1
            N_u(2)=floor(m./(N_u(1)*N_u(3))); %Number of inducing inputs along dim 2
            r_star=2*dist_factors(i).*l_SE; %Distance limiter for measurements used in LI inference
            r=dist_factors(i).*l_SE;
            
            %Find the local approximation
            [mu, var, preprocessingtime, inferencetime]=FloatingDomainGP3D(X',y',Xtest',Omega,N_u,sigma_SE,l_SE,r,r_star,sigma_y);
            
            %Calculate the initial SMSEs of the floating domain GP
            SMSEs(i,j,repetition)=(mean((mu-ytest).^2))./std(ytest)^2;
            PreProcessTimes(i,j,repetition)=preprocessingtime;
            InferenceTimes(i,j,repetition)=inferencetime;
            disp(['experiment: ',num2str(j),'/',num2str(experiments)]);
        end
        disp(['param: ',num2str(i),'/',num2str(params)]);
    end
    disp(['repetition: ',num2str(repetition),'/',num2str(repetitions)]);
end
toc(tStart);

%Define some colours
lavender=[100, 142, 255]./255;
pink=[220 38 127]./255;
mustard=[255 176 0]./255;
black=[0 0 0]./255;
color_palette=[lavender; pink; mustard];

%Plot settings
fontsize=19;

figure; clf;
for param=1:params
    PreProcessTimesParam(:,:)=PreProcessTimes(param,:,:);
    average=mean(PreProcessTimesParam,2);
    standard_devs=std(PreProcessTimesParam');
    errorbar(average,SMSEs(param,:,1)',standard_devs,'horizontal','linewidth',1.5,...
        'Color',color_palette(param,:));
    hold on;
end
grid on;
xlabel('Training time (s)','Interpreter','Latex','Fontsize',fontsize);
ylabel('SMSE','Interpreter','Latex','Fontsize',fontsize);
set(gca,'TickLabelInterpreter','latex');
set(gca, 'XScale', 'log');
box off;
set(gca, 'FontName', 'Times');
set(gca,'fontsize',fontsize);
set(gca,'TickLabelInterpreter','latex');
exportgraphics(gca,'PrecipSMSEvsPreProcessTime.pdf','ContentType','Vector');

figure; clf;
for param=1:params
    InferenceTimesParam(:,:)=InferenceTimes(param,:,:);
    average=mean(InferenceTimesParam,2);
    standard_devs=std(InferenceTimesParam');
    errorbar(average,SMSEs(param,:,1)',standard_devs,'horizontal','linewidth',1.5,...
        'Color',color_palette(param,:));
    hold on;
end
grid on;
xlabel('Prediction time (s)','Interpreter','Latex','Fontsize',fontsize);
ylabel('SMSE','Interpreter','Latex','Fontsize',fontsize);
set(gca,'TickLabelInterpreter','latex');
set(gca, 'XScale', 'log');
box off;
set(gca, 'FontName', 'Times');
set(gca,'fontsize',fontsize);
set(gca,'TickLabelInterpreter','latex');
exportgraphics(gca,'PrecipSMSEvsInferenceTime.pdf','ContentType','Vector');

figure; clf;
for param=1:params
    plot(ms,SMSEs(param,:,1)','-o','linewidth',1.5,...
        'Color',color_palette(param,:));
    hold on;
end
grid on;
xlabel('Number of inducing points $m$','Interpreter','Latex','Fontsize',fontsize);
ylabel('SMSE','Interpreter','Latex','Fontsize',fontsize);
set(gca,'TickLabelInterpreter','latex');
set(gca, 'XScale', 'log');
box off;
set(gca, 'FontName', 'Times');
set(gca,'fontsize',fontsize);
set(gca,'TickLabelInterpreter','latex');
exportgraphics(gca,'PrecipSMSEvsInducingPoints.pdf','ContentType','Vector');

