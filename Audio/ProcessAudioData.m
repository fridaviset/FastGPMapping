%Copyright (C) 2022 by Frida Viset

close all; clear;
load('audio_data.mat');

%Define hyperparameters GP and RR-GP
lL=min(xtrain); %lower bound
uL=max(xtrain); %upper bound

%Hyperparameters 
sigma_SE=0.009;
l_SE=10.895;
sigma_y=0.002;

%Settings for experimental results
repetitions=10;
dist_factors=[2.5,5,10,15];
params=length(dist_factors);
N_us=2000:500:9000;
experiments=length(N_us);

%Pre-allocate storage for experimental results
SMAE_vals=zeros(experiments,params,repetitions);
traintimes=zeros(experiments,params,repetitions);
inferencetimes=zeros(experiments,params,repetitions);
SMAE_valsDTC=zeros(experiments,repetitions);
traintimesDTC=zeros(experiments,repetitions);
inferencetimesDTC=zeros(experiments,repetitions);

for repetition=1:repetitions
    
    for experiment=1:experiments
        
        N_u=N_us(experiment);
        
        if N_u<4500
            [mu, var, ~, train_time, inf_time]=InducingInputsGaussianProcess(xtrain',ytrain',xtest',uL,lL,sigma_y,N_u,sigma_SE,l_SE);
            traintimesDTC(experiment,repetition)=train_time;
            inferencetimesDTC(experiment,repetition)=inf_time;
            SMAE_valsDTC(experiment,repetition)=mean(abs(ytest-mu))./mean(abs(ytest));
        else
            traintimesDTC(experiment,repetition)=NaN;
            inferencetimesDTC(experiment,repetition)=NaN;
            SMAE_valsDTC(experiment,repetition)=NaN;
        end
        
        for param=1:params
            
            N_u=N_us(experiment);
            
            r=dist_factors(param)*l_SE;
            
            %Smart Inducing-inputs GP
            [mu, var, ~,max_set_length, train_time, inf_time]=LocalInducingInputsGaussianProcess(xtrain',ytrain',xtest',uL,lL,sigma_y,N_u,sigma_SE,l_SE,r);
            traintimes(experiment,param,repetition)=train_time;
            inferencetimes(experiment,param,repetition)=inf_time;
            SMAE_vals(experiment,param,repetition)=mean(abs(ytest-mu))./mean(abs(ytest));
            
            max_set_lengths(param)=max_set_length;
            
        end
        
        disp(['Experiment ',num2str(experiment),'/',num2str(experiments),', m=',num2str(N_u)]);
        
    end
    
    disp(['Repetition ',num2str(repetition),'/',num2str(repetitions)]);
    
end

save('Workspace.mat');


%% Plot results

%Define some colours
purple=[120, 94, 240]./255;
pink=[220 38 127]./255;
orange=[254, 97, 0]./255;
mustard=[255 176 0]./255;

black=[0 0 0]./255;
color_palette=[purple;  pink; orange; mustard];

%Plot settings
fontsize=19;

figure; clf;
errorbar(mean(traintimesDTC'),mean(SMAE_valsDTC'),...
        std(traintimesDTC'),'horizontal','linewidth',1.5,...
        'Color',black);
hold on;
legends={};
legends{1}='DTC';
for param=1:params
    traintimes_all(:,:)=traintimes(:,param,:);
    avg_traintimes=mean(traintimes_all,2);
    std_traintime=std(traintimes_all');
    errorbar(avg_traintimes,mean(SMAE_vals(:,param,:),3),...
        std_traintime,'horizontal','linewidth',1.5,...
        'Color',color_palette(param,:));
    hold on;
    legends{param+1}=['$r^*=',num2str(dist_factors(param)),'$',];
end
grid on;
box off;
xlabel('Training time (s)','Interpreter','Latex','Fontsize',fontsize);
ylabel('SMAE','Interpreter','Latex','Fontsize',fontsize);
%lgd=legend(legends,'Interpreter','Latex');
set(gca, 'FontName', 'Times');
set(gca, 'XScale', 'log');
set(gca,'fontsize',fontsize);
exportgraphics(gca,'SoundSMAEvsPreProcessTime.pdf','ContentType','Vector');

figure; clf;
errorbar(mean(inferencetimesDTC'),mean(SMAE_valsDTC'),...
        std(inferencetimesDTC'),'horizontal','linewidth',1.5,...
        'Color',black);
hold on;
for param=1:params
    inference_times_all(:,:)=inferencetimes(:,param,:);
    avg_inference_times=mean(inference_times_all');
    std_inference_times=std(inference_times_all');
    errorbar(avg_inference_times,mean(SMAE_vals(:,param,:),3),...
        std_inference_times,'horizontal','linewidth',1.5,...
        'Color',color_palette(param,:));
    hold on;
end
grid on;
xlabel('Prediction time (s)','Interpreter','Latex','Fontsize',fontsize);
ylabel('SMAE','Interpreter','Latex','Fontsize',fontsize);
set(gca,'TickLabelInterpreter','latex');
set(gca, 'XScale', 'log');
view(2);
box off;
set(gca, 'FontName', 'Times');
set(gca,'fontsize',fontsize);
set(gca,'TickLabelInterpreter','latex');
exportgraphics(gca,'SoundSMAEvsInferenceTime.pdf','ContentType','Vector');

figure; clf;
for param=1:params
    plot(N_us,SMAE_vals(:,param,1)','-o','linewidth',1.5,...
        'Color',color_palette(param,:));
    hold on;
end
plot(N_us,SMAE_valsDTC,'-o','linewidth',1.5,'markersize',4,...
    'Color',black);
grid on;
xlabel('Number of inducing points $m$','Interpreter','Latex','Fontsize',fontsize);
ylabel('SMAE','Interpreter','Latex','Fontsize',fontsize);
set(gca,'TickLabelInterpreter','latex');
set(gca, 'XScale', 'log');
box off;
set(gca, 'FontName', 'Times');
set(gca,'fontsize',fontsize);
set(gca,'TickLabelInterpreter','latex');
exportgraphics(gca,'SoundSMAEvsInducingPoints.pdf','ContentType','Vector');
