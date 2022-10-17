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

%Open a results file
fileID=fopen('results.txt','w');

%Define the different amounts of training points for the GP
Ns=[10000 20000];

fprintf(fileID,'---Full GP---\n');
disp('---Full GP---');

for N=Ns
    
    %Find the full GP solution
    tic;
    [mu, variance]=GaussianProcess(X(1:N,:)',y(1:N)',Xtest',sigma_y,sigma_SE,l_SE);
    time=toc;
    
    %Compute SMSEs
    SMSE=(mean((mu-ytest).^2))./std(ytest)^2;
    
    %Write results to file
    fprintf(fileID,['N=',num2str(N),': ']);
    fprintf(fileID,['SMSE=',num2str(SMSE),', Runtime: ',num2str(time),'\n']);

    disp(['N=',num2str(N),': ']);
    disp(['SMSE=',num2str(SMSE),', Runtime: ',num2str(time)]);
end

%Define the different amounts of training points for the rest of the
%methods
Ns=[10000 20000 100000 528474];

%Define the different amounts of inducing points
ms=[10000 20000];

fprintf(fileID,'---Inducing points---\n');
disp('---Inducing points---');

for N=Ns
    for m=ms
        
    %Distribute the inducing points in an equi-spaced grid across space, 
    %and time (one layer of points per day, otherwise equispaced corrected
    %for the lengthscale of the GP
    %Place the inducing points
    N_u(3)=365; 
    L_1=Omega(1,2)-Omega(1,1); L_2=Omega(2,2)-Omega(2,1);
    N_u(1)=round(sqrt(m./N_u(3).*(L_1*l_SE(2))./(L_2*l_SE(1))));
    N_u(2)=floor(m./(N_u(1)*N_u(3))); 
    
    %Find the Inducing point solution
    [mu, variance, preprocessingtime, inferencetime]=InducingPointGP3D(X(1:N,:)',y(1:N)',Xtest',Omega,N_u,sigma_SE,l_SE,sigma_y);
    
    %Compute SMSEs
    SMSE=(mean((mu-ytest).^2))./std(ytest)^2;
    
    %Write results to file
    fprintf(fileID,['N=',num2str(N),', m=',num2str(m),': ']);
    fprintf(fileID,['SMSE=',num2str(SMSE)]);
    fprintf(fileID,['--Preprocessing time: ',num2str(preprocessingtime),', Inferencetime: ',num2str(inferencetime),'\n']);
    
    disp(['N=',num2str(N),', m=',num2str(m),': ']);
    disp(['SMSE=',num2str(SMSE)]);
    disp(['--Preprocessing time: ',num2str(preprocessingtime),', Inferencetime: ',num2str(inferencetime)]);
    
    end
end

%Define the different amounts of training points for the floating domain GP
Ns=[10000 20000 100000 528474];

%Define the different amounts of inducing points for the floating domain GP
ms=[10000 20000 200000 800000];

fprintf(fileID,'---Floating GP---\n');
disp('---Floating GP---');

for N=Ns
    for m=ms
        
    %Distribute the inducing points in an equi-spaced grid across space, 
    %and time (one layer of points per day, otherwise equispaced corrected
    %for the lengthscale of the GP
    %Place the inducing points
    N_u(3)=365; 
    L_1=Omega(1,2)-Omega(1,1); L_2=Omega(2,2)-Omega(2,1);
    N_u(1)=round(sqrt(m./N_u(3).*(L_1*l_SE(2))./(L_2*l_SE(1))));
    N_u(2)=floor(m./(N_u(1)*N_u(3))); 
    r_star=6.*l_SE; %Distance limiter for measurements used in LI inference
    r=3.*l_SE;

    %Find the local approximation
    [mu, variance, preprocessingtime, inferencetime]=FloatingDomainGP3D(X(1:N,:)',y(1:N)',Xtest',Omega,N_u,sigma_SE,l_SE,r,r_star,sigma_y);
    
    %Compute SMSEs
    SMSE=(mean((mu-ytest).^2))./std(ytest)^2;
    
    %Write results to file
    fprintf(fileID,['N=',num2str(N),', m=',num2str(m),': ']);
    fprintf(fileID,['SMSE=',num2str(SMSE)]);
    fprintf(fileID,['--Preprocessing time: ',num2str(preprocessingtime),', Inferencetime: ',num2str(inferencetime),'\n']);
    
    disp(['N=',num2str(N),', m=',num2str(m),': ']);
    disp(['SMSE=',num2str(SMSE)]);
    disp(['--Preprocessing time: ',num2str(preprocessingtime),', Inferencetime: ',num2str(inferencetime)]);
    
    end
end

%Close the results file
fclose(fileID);
