function detection(~)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% DAMAGE DETECTION BASED ON NON-PARAMETRIC METHODS%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('----- NON-PARAMETRIC STATISTICAL TIME SERIES METHODS FOR SHM -----')

load ../data/before_order_selection.mat

i=16; ii=20; % font sizes
%--------------------------------------------------------------------------
%                       Damage detection based on the PSD Method
%--------------------------------------------------------------------------
disp('   ')
%--------------------------------------------------------------------------
WINDOW = 1024; NFFT = 1024; alpha = 0.001; %                                   <----------
%--------------------------------------------------------------------------
format short g
disp('------------------------------------------------------------');
disp('       PSD BASED METHOD      ');
disp('------------------------------------------------------------');
disp('          N          Window          K           �'); 
disp([ size(Y1,1) WINDOW round(size(Y1,1)/WINDOW) alpha])
disp('------------------------------------------------------------');
disp('   ')
disp(' Damage is detected if the test statistic (blue line) is not between the critical limits (red dashed lines) ')
disp(' ')
disp('(press a key)')
pause
close all
%%%%%% Damage detection results 
disp(' baseline ')
fdi_spectral_density(Y1, Y2, WINDOW,NFFT,Fs,alpha); 
xlim([5 128])
set(gca,'fontsize',i)
title('PSD based method - Baseline phase - \alpha = 10^{-3}','fontsize', ii)
ylabel('F statistic','fontsize',ii)
xlabel(' Frequency (Hz)','fontsize',ii)
disp('(press a key)')
pause

%Baseline check with faults
disp('   ')
disp(' Test baseline with Faults')
figure;
for iter=2:2:8
fdi_spectral_density_giannis(Faults(:,iter), Y2, WINDOW,NFFT,Fs,alpha,2,2,iter/2); 
xlim([5 128])
set(gca,'fontsize',i/2)
%title('PSD based method - Baseline check Fault - \alpha = 10^{-3}','fontsize', ii/2)
ylabel('F statistic','fontsize',ii/2)
xlabel(' Frequency (Hz)','fontsize',ii/2)
%disp('(press a key)')
%pause
%disp('   ')
end
sgtitle('Baseline phase check with faulty signals','fontsize',ii);

% % Test case I : Healthy
% disp(' Test case I : Healthy Structure ')
% fdi_spectral_density(Y1, output(:,1), WINDOW,NFFT,Fs,alpha); 
% xlim([5 128])
% set(gca,'fontsize',i)
% title('PSD based method - Unknown response 1 - \alpha = 10^{-3}','fontsize', ii)
% ylabel('F statistic','fontsize',ii)
% xlabel(' Frequency (Hz)','fontsize',ii)
% disp('(press a key)')
% pause
% disp('   ')
% 
% disp(' Test case II : Fault of 8.132 gr ')
% % Test case II : Fault 8.132 gr
% fdi_spectral_density(Y1,output(:,2),WINDOW,NFFT,Fs,alpha);
% xlim([5 (Fs/2)])
% set(gca,'fontsize',i)
% title('PSD based method - Unknown response 2 - \alpha = 10^{-3}','fontsize',ii)
% ylabel('F statistic','fontsize',ii)
% xlabel(' Frequency (Hz)','fontsize',ii)
% disp('(press a key)')
% pause
% disp('   ')
% 
% disp(' Test case III : Fault of 24.396 gr ')
% % Test case III : Faulty 24.396 gr
% fdi_spectral_density(Y1,output(:,3),WINDOW,NFFT,Fs,alpha);
% xlim([5 (Fs/2)])
% set(gca,'fontsize',i)
% title('PSD based method - Unknown response 3 - \alpha = 10^{-3}','fontsize',ii)
% ylabel('F statistic','fontsize',ii)
% xlabel(' Frequency (Hz)','fontsize',ii)
% disp('(press a key)')
% pause
% disp('   ')
% 
% disp(' Test case IV : Fault of 40.66 gr ')
% % Test case IV : Faulty 40.66 gr
% fdi_spectral_density(Y1,output(:,4),WINDOW,NFFT,Fs,alpha);
% xlim([5 (Fs/2)])
% set(gca,'fontsize',i)
% title('PSD based method - Unknown response 4 - \alpha = 10^{-3}','fontsize',ii)
% ylabel('F statistic','fontsize',ii)
% xlabel(' Frequency (Hz)','fontsize',ii)
% disp('(press a key)')
% pause
% disp('   ')

% Test cases as subplots
figure
for iter=1:4
fdi_spectral_density_giannis(Y1, output(:,iter), WINDOW,NFFT,Fs,alpha,2,2,iter); 
xlim([5 128])
set(gca,'fontsize',i/2)
%title('Unknown signal i','fontsize', ii/2)
title("Unknown signal " + iter + " ",'fontsize', ii/2)
ylabel('F statistic','fontsize',ii/2)
xlabel(' Frequency (Hz)','fontsize',ii/2)
%disp('(press a key)')
%pause
%disp('   ')
end
sgtitle('PSD-based method - Unknown state signals - \alpha = 10^{-3}','fontsize',ii);
pause;

clc
close all
%--------------------------------------------------------------------------
%                       Damage detection based on the FRF Method
%--------------------------------------------------------------------------
%Method parameters
WINDOW = 512/2; NFFT = 512/2 ; alpha = 0.000000000001; % 
% WINDOW = 512; NFFT = 512 ; alpha = 0.000000000001;
%WINDOW = 512*2; NFFT = 512*2 ; alpha = 0.000000000001;
%--------------------------------------------------------------------------
disp('--------------------------------------------------------------------');
disp('       FRF BASED METHOD      ');
disp('--------------------------------------------------------------------');
disp('          N          Window          K           �'); 
disp([ size(Y1) WINDOW round(size(Y1)/WINDOW) alpha])
disp('------------------------------------------------------------');
disp('   ')
disp(' Damage is detected if the test statistic (blue line) is over the critical limit (red line) ')
disp(' ')
disp('(press a key)')
pause
disp('   ')

%%%%%% Damage detection results 
disp('Baseline phase ')
% Test case I : Healthy
fdi_frf(X1,Y1,X2,Y2,WINDOW,NFFT,Fs,alpha);
xlim([5 (Fs/2)])
set(gca,'fontsize',i)
title('FRF based method - Baseline phase - \alpha = 10^{-12}','fontsize',ii)
ylabel('Z statistic','fontsize',ii)
xlabel(' Frequency (Hz)','fontsize',ii)
disp('(press a key)')
pause
disp('   ')

%Baseline check with faults
disp('   ')
disp(' Test baseline with Faults')
figure;
for iter=2:2:8
fdi_frf_giannis(X1,Y1,Faults(:,iter-1),Faults(:,iter),WINDOW,NFFT,Fs,alpha, 2, 2, iter/2);
xlim([5 128])
set(gca,'fontsize',i/2)
ylabel('Z statistic','fontsize',ii)
xlabel(' Frequency (Hz)','fontsize',ii)
%disp('(press a key)')
%pause
%disp('   ')
end
sgtitle(' FRF-based method - Baseline phase check with faulty signals','fontsize',ii);
pause;


% disp(' Test case II : Fault of 8.132 gr ')
% % Test case II : Faulty 8.132 gr
% fdi_frf(X1,Y1,inp(:,1),output(:,1),WINDOW,NFFT,Fs,alpha);
% xlim([5 (Fs/2)])
% set(gca,'fontsize',i)
% title('FRF based method - Unknown signals 1 - \alpha = 10^{-12}','fontsize',ii)
% ylabel('Z statistic','fontsize',ii)
% xlabel(' Frequency (Hz)','fontsize',ii)
% disp('(press a key)')
% pause
% disp('   ')
% 
% disp(' Test case III: Fault of 24.396 gr ')
% % Test case III : Faulty 24.396 gr
% fdi_frf(X1,Y1,inp(:,2),output(:,2),WINDOW,NFFT,Fs,alpha);
% xlim([5 (Fs/2)])
% set(gca,'fontsize',i)
% title('FRF based method - Unknown signals 2 - \alpha = 10^{-12}','fontsize',ii)
% ylabel('Z statistic','fontsize',ii)
% xlabel(' Frequency (Hz)','fontsize',ii)
% disp('(press a key)')
% pause
% disp('   ')
% 
% disp(' Test case IV: Fault of 40.66 gr ')
% % Test case IV : Faulty 40.66 gr
% fdi_frf(X1,Y1,inp(:,3),output(:,3),WINDOW,NFFT,Fs,alpha);
% xlim([5 (Fs/2)])
% set(gca,'fontsize',i)
% title('FRF based method - Unknown signals 3 - \alpha = 10^{-12}','fontsize',ii)
% ylabel('Z statistic','fontsize',ii)
% xlabel(' Frequency (Hz)','fontsize',ii)
% disp('(press a key)')
% pause
% disp('   ')
% 
% disp(' Test case V: Fault of 56.924 gr ')
% % Test case V : Faulty 56.924 gr
% fdi_frf(X1,Y1,inp(:,4),output(:,4),WINDOW,NFFT,Fs,alpha);
% xlim([5 (Fs/2)])
% set(gca,'fontsize',i)
% title('FRF based method - Unknown signals 4 - \alpha = 10^{-12}','fontsize',ii)
% ylabel('Z statistic','fontsize',ii)
% xlabel(' Frequency (Hz)','fontsize',ii)
% disp('(press a key)')
% pause
% disp('   ')

% Test cases as subplots
figure
for iter=1:4
fdi_frf_giannis(X1,Y1,inp(:,iter),output(:,iter),WINDOW,NFFT,Fs,alpha, 2, 2, iter);
xlim([5 128])
set(gca,'fontsize',i/2)
%title('Unknown signal i','fontsize', ii/2)
title("Unknown signal " + iter + " ",'fontsize', ii/2)
ylabel('Z statistic','fontsize',ii/2)
xlabel(' Frequency (Hz)','fontsize',ii/2)
%disp('(press a key)')
%pause
%disp('   ')
end
sgtitle('FRF-based method - Unknown state signals - \alpha = 10^{-12}','fontsize',ii);
pause;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% DAMAGE DETECTION BASED ON PARAMETRIC METHODS %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear
load ../data/before_order_selection.mat
disp('  PARAMETRIC STATISTICAL TIME SERIES METHODS FOR SHM  ')
i=16; ii=20; % font sizes
%--------------------------------------------------------------------------
%                       Model Parameter Based Method
%--------------------------------------------------------------------------
%risk level
alpha = 0.01;

disp('------------------------------------------------------------');
disp('            MODEL PARAMETER BASED METHOD           ');
disp('------------------------------------------------------------');
disp('          �                Model: ARX(62,62) '); 
disp([alpha])
disp('------------------------------------------------------------');
armax_select_order = 68
DATA2=iddata(Y1, X1,1/Fs);
% %normalize
% output(:,1) = detrend(output(:,1));
% output(:,1) = output(:,1)./std(output(:,1));
% inp(:,1) = detrend(inp(:,1));
% inp(:,1) = inp(:,1)./std(inp(:,1));

DATA_unknown1 = iddata(output(:,1), inp(:,1), 1/Fs);
DATA_unknown2 = iddata(output(:,2), inp(:,2), 1/Fs);
DATA_unknown3 = iddata(output(:,3), inp(:,3), 1/Fs);
DATA_unknown4 = iddata(output(:,4), inp(:,4), 1/Fs);

opt = armaxOptions;
%opt.InitialCondition = 'zero';
%opt.Focus = 'prediction' ;
%opt.InputOffset = 0 ; 
%opt.OutputOffset = 0 ;
opt.EnforceStability = true;
armax_healthy = models_armax{armax_select_order};
armax_healthy2 = armax(DATA2,[armax_select_order armax_select_order armax_select_order 0]);
armax_unknown1 = armax(DATA_unknown1,[armax_select_order armax_select_order armax_select_order 0]);
armax_unknown2 = armax(DATA_unknown2,[armax_select_order armax_select_order armax_select_order 0]);
armax_unknown3 = armax(DATA_unknown3,[armax_select_order armax_select_order armax_select_order 0]);
armax_unknown4 = armax(DATA_unknown4,[armax_select_order armax_select_order armax_select_order 0]);


%%
clear;
load ../data/before_parametric_detection_normalized_armax68.mat
close all

%%%%% compare frf of 2 healty signals
[M1,PHASE1,wp1] = dbode(armax_healthy.B,armax_healthy.A,1/Fs,2*pi*[0:0.1:120]);
[M2,PHASE2,wp2] = dbode(armax_health ...
    y2.B,armax_healthy2.A,1/Fs,2*pi*[0:0.1:120]);

figure
plot(wp1/(2*pi),20*log10(abs(M1))),hold on
plot(wp2/(2*pi),20*log10(abs(M2)), 'r')
axis([5 120 -80 60])
set(gca,'fontsize',16)
title('Comparison of ARMAX(70,70,70) of the two healthy signals');
ylabel('Magnitude (dB)','Fontname','TimesNewRoman','Fontsize',20)
xlabel('Frequency (Hz)','Fontname','TimesNewRoman','Fontsize',20)
legend('healthy1','healthy2','Location','SouthEast','Orientation','vertical')
disp('(press a key)')
pause % ,close
disp('   ')
disp('   ')

%%%%%%%%%%%%%
%alpha = 0.000000000001;
alpha = 10e-3;
[Q_mod_par(1),limit_mod_par]=fdi_model_parameter_giannis(armax_healthy,armax_healthy2,alpha, 8e5); 
[Q_mod_par(2),limit_mod_par]=fdi_model_parameter_giannis(armax_healthy,armax_unknown1,alpha, 8e5);
[Q_mod_par(3),limit_mod_par]=fdi_model_parameter_giannis(armax_healthy,armax_unknown2,alpha, 8.8e5);
[Q_mod_par(4),limit_mod_par]=fdi_model_parameter_giannis(armax_healthy,armax_unknown3,alpha, 8.8e5); 
[Q_mod_par(5),limit_mod_par]=fdi_model_parameter_giannis(armax_healthy,armax_unknown4,alpha, 8.8e5); 
% [Q_mod_par(6),limit_mod_par]=fdi_model_parameter(arx_healthy,arx_faulty_VI,alpha); 
%NOTE: The necessary statistic and the corresponding limits for each case are computed based on the
%functions above and are included in the Q_mod_par & limit_mod_par files, respectively.

disp(' ')
disp(' Damage is detected if the test statistic (blue bar) is over the critical limit (red dashed line) ')
figure
bar(Q_mod_par,0.5)
hold on
line([0 7],[limit_mod_par limit_mod_par],'color','r','linestyle','--','linewidth',1.5)
hold off
set(gca,'fontsize',i, 'xticklabel',['Baseline ';'Unknown 1';'Unknown 2';'Unknown 3';'Unknown 4'])
title(' Model Parameter Based Method - limit assigned by hand','fontsize',ii)
ylabel('Test Statistic','fontsize',ii)
xlabel(' Test Cases','fontsize',ii)
disp('(press a key)')
pause
clc
%--------------------------------------------------------------------------
%                       Residual Based Methods
%--------------------------------------------------------------------------
disp('            MODEL RESIDUAL BASED METHODS           ');
disp(' ')
close all;
% Method A: Using the residual variance
%risk level
%alpha = 0.0000000000000001;
alpha = 10e-3;
disp('-----------------------------------------------------------------------------');
disp('     Method A: using the residual variance     ');
disp('-----------------------------------------------------------------------------');
disp('          �                Model: ARX(62,62) '); 
disp([alpha])
disp('------------------------------------------------------------');
disp(' ')
disp(' Damage is detected if the test statistic (blue bar) is over the critical limit (red dashed line) ')

[Q_res_a(1),limit_res_a]=fdi_model_residual_giannis(armax_healthy,DATA1,DATA2,'var',alpha, 1.5);
[Q_res_a(2),limit_res_a]=fdi_model_residual_giannis(armax_healthy,DATA1,DATA_unknown1,'var',alpha, 1.5);
[Q_res_a(3),limit_res_a]=fdi_model_residual_giannis(armax_healthy,DATA1,DATA_unknown2,'var',alpha, 1.5);
[Q_res_a(4),limit_res_a]=fdi_model_residual_giannis(armax_healthy,DATA1,DATA_unknown3,'var',alpha, 1.5);
[Q_res_a(5),limit_res_a]=fdi_model_residual_giannis(armax_healthy,DATA1,DATA_unknown4,'var',alpha, 1.5);
% % [Q_res_a(6),limit_res_a]=fdi_model_residual(arx_healthy,DATA_healthy,DATA_faulty_VI,'var',alpha);
% %NOTE: The necessary statistic and the corresponding limits for each case are computed based on the
% %functions above and are included in the Q_res_a & limit_res_a files, respectively.

figure
bar(Q_res_a,0.5)
hold on
line([0 7],[limit_res_a limit_res_a],'color','r','linestyle','--','linewidth',1.5)
hold off
set(gca,'fontsize',i, 'xticklabel',['Baseline ';'Unknown 1';'Unknown 2';'Unknown 3';'Unknown 4'])
title(' Redidual Based Method A: using the residual variance - limit assigned by hand','fontsize',ii)
ylabel('Test Statistic','fontsize',ii)
xlabel(' Test Cases','fontsize',ii)
disp('(press a key)')
pause

% %--------------------------------------------------------------------------
% % Method B: Using the likelihood function
% close all
% %risk level
% alpha = 10e-3;
% 
% disp('-------------------------------------------------------------------------------');
% disp('       Method B: using the likelihood function   ');
% disp('-------------------------------------------------------------------------------');
% disp('          �                Model: ARX(62,62) '); 
% disp([alpha])
% disp('------------------------------------------------------------');
% disp(' ')
% disp(' Damage is detected if the test statistic (blue bar) is over the critical limit (red dashed line) ')
% 
% [Q_res_b(1),limit_res_b]=fdi_model_residual(armax_healthy,DATA1,DATA2,'lik',alpha);
% [Q_res_b(2),limit_res_b]=fdi_model_residual(armax_healthy,DATA1,DATA_unknown1,'lik',alpha);
% [Q_res_b(3),limit_res_b]=fdi_model_residual(armax_healthy,DATA1,DATA_unknown2,'lik',alpha);
% [Q_res_b(4),limit_res_b]=fdi_model_residual(armax_healthy,DATA1,DATA_unknown3,'lik',alpha);
% [Q_res_b(5),limit_res_b]=fdi_model_residual(armax_healthy,DATA1,DATA_unknown4,'lik',alpha);
% % [Q_res_b(6),limit_res_b]=fdi_model_residual(arx_healthy,DATA_healthy,DATA_faulty_VI,'lik',alpha);
% %NOTE: The necessary statistic and the corresponding limits for each test case are computed based on the
% %functions above and are included int he Q_res_b & limit_res_b files, respectively.
% 
% figure
% bar(Q_res_b,0.5)
% hold on
% line([0 7],[limit_res_b limit_res_b],'color','r','linestyle','--','linewidth',1.5)
% hold off
% set(gca,'fontsize',i, 'xticklabel',['Baseline ';'Unknown 1';'Unknown 2';'Unknown 3';'Unknown 4'])
% title(' Redidual Based Method B: using the likelihood function','fontsize',ii)
% ylabel('Test Statistic','fontsize',ii)
% xlabel(' Test Cases','fontsize',ii)
% disp('(press a key)')
% pause

%--------------------------------------------------------------------------
% Method C: Using the residual uncorrelatedness

%risk level and mamixum lag
alpha = 10e-3;
max_lag = 25;

disp('-------------------------------------------------------------------------------------');
disp('       Method C: using the residual uncorrelatedness   ');
disp('-------------------------------------------------------------------------------------');
disp('          �          max_lag      Model: ARX(62,62) '); 
disp([alpha max_lag])
disp('-------------------------------------------------------------------------------------');
disp(' ')
disp(' Damage is detected if the test statistic (blue bar) is over the critical limit (red dashed line) ')

[Q_res_c(1),limit_res_c]=fdi_model_residual_giannis(armax_healthy,DATA1,DATA2,'unc',alpha,600,max_lag);
[Q_res_c(2),limit_res_c]=fdi_model_residual_giannis(armax_healthy,DATA1,DATA_unknown1,'unc',alpha,600,max_lag);
[Q_res_c(3),limit_res_c]=fdi_model_residual_giannis(armax_healthy,DATA1,DATA_unknown2,'unc',alpha,600,max_lag);
[Q_res_c(4),limit_res_c]=fdi_model_residual_giannis(armax_healthy,DATA1,DATA_unknown3,'unc',alpha,600,max_lag);
[Q_res_c(5),limit_res_c]=fdi_model_residual_giannis(armax_healthy,DATA1,DATA_unknown4,'unc',alpha,600,max_lag);
% [Q_res_c(6),limit_res_c]=fdi_model_residual(arx_healthy,DATA_healthy,DATA_faulty_VI,'unc',...
%     alpha,max_lag);
%The necessary statistic and the corresponding limits for each test case are computed based on the
%functions above and are included int he Q_res_c & limit_res_c files, respectively.

figure
bar(Q_res_c,0.5)
hold on
line([0 7],[limit_res_c limit_res_c],'color','r','linestyle','--','linewidth',1.5)
hold off
set(gca,'fontsize',i, 'xticklabel',['Baseline ';'Unknown 1';'Unknown 2';'Unknown 3';'Unknown 4'])
title(' Redidual Based Method C: using the residual uncorrelatedness - limit assigned by hand','fontsize',ii)
ylabel('Test Statistic','fontsize',ii)
xlabel(' Test Cases','fontsize',ii)

%--------------------------------------------------------------------------




%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%AUXILIARY FILES
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
function [F,limits]=fdi_spectral_density(y_healthy,y_faulty,window,nfft,fs,alpha)

[Pyh,w] = pwelch(y_healthy,window,0,nfft,fs);
[Pyf,w] = pwelch(y_faulty,window,0,nfft,fs);
F = Pyh./Pyf;,K = length(y_healthy)/window;,p = [alpha/2 1-alpha/2];
limits = finv(p,2*K,2*K);
figure,plot(w,F),grid on,hold on
line([0 w(end)+1],[limits(1) limits(1)],'color','r','linestyle','--','linewidth',1.5)
line([0 w(end)+1],[limits(2) limits(2)],'color','r','linestyle','--','linewidth',1.5)
hold off

%to subplot results
function [F,limits]=fdi_spectral_density_giannis(y_healthy,y_faulty,window,nfft,fs,alpha,i, j, k)

[Pyh,w] = pwelch(y_healthy,window,0,nfft,fs);
[Pyf,w] = pwelch(y_faulty,window,0,nfft,fs);
F = Pyh./Pyf;,K = length(y_healthy)/window;,p = [alpha/2 1-alpha/2];
limits = finv(p,2*K,2*K);
subplot(i,j,k),plot(w,F),grid on,hold on
line([0 w(end)+1],[limits(1) limits(1)],'color','r','linestyle','--','linewidth',1.5)
line([0 w(end)+1],[limits(2) limits(2)],'color','r','linestyle','--','linewidth',1.5)
hold off

function fdi_frf(x_healthy,y_healthy,x_faulty,y_faulty,window,nfft,fs,alpha)

[tyh,w] = tfestimate(x_healthy,y_healthy,window,0,nfft,fs);
[tyf,w] = tfestimate(x_faulty,y_faulty,window,0,nfft,fs);  
[cyh,w] = mscohere(x_healthy,y_healthy,window,0,nfft,fs);  
K = length(y_healthy)/window; 
Tyh = abs(tyh);,Tyf = abs(tyf);,var_frf = [(1-cyh)./(2*K*cyh)].*Tyh.^2;
dT = Tyh - Tyf;,p = [alpha/2 1-alpha/2];,n = norminv(p);,z = n(2); % uper bound
figure,plot(w,abs(dT)./sqrt(2*var_frf)),grid on,hold on,,plot(w,z*ones(size(w)),'r')
hold off

%to subplot results
function fdi_frf_giannis(x_healthy,y_healthy,x_faulty,y_faulty,window,nfft,fs,alpha, i, j, k)

[tyh,w] = tfestimate(x_healthy,y_healthy,window,0,nfft,fs);
[tyf,w] = tfestimate(x_faulty,y_faulty,window,0,nfft,fs);  
[cyh,w] = mscohere(x_healthy,y_healthy,window,0,nfft,fs);  
K = length(y_healthy)/window; 
Tyh = abs(tyh);,Tyf = abs(tyf);,var_frf = [(1-cyh)./(2*K*cyh)].*Tyh.^2;
dT = Tyh - Tyf;,p = [alpha/2 1-alpha/2];,n = norminv(p);,z = n(2); % uper bound
subplot(i,j,k), plot(w,abs(dT)./sqrt(2*var_frf)),grid on,hold on,,plot(w,z*ones(size(w)),'r')
hold off

function [Q,limit] = fdi_model_parameter(model_o,model_u,alpha)

theta_o = model_o.parametervector;,theta_u = model_u.parametervector;
P_o = model_o.covariancematrix;,dtheta = theta_o - theta_u;,dP = 2*P_o;
Q = dtheta.'*inv(dP)*dtheta;,p = 1-alpha;,limit = chi2inv(p,size(theta_o,1));
figure,bar([0 Q 0]),hold on
line([0 4],[limit limit],'color','r','linestyle','--','linewidth',1.5)
hold off
set(gca,'fontsize',12,'xtick',[]),
title(' Model Parameter Based Method','fontsize',16)
ylabel('Test Statistic','fontsize',16)

% to define limit by hand
function [Q,limit] = fdi_model_parameter_giannis(model_o,model_u,alpha, giannis)

theta_o = model_o.parametervector;,theta_u = model_u.parametervector;
P_o = model_o.covariancematrix;,dtheta = theta_o - theta_u;,dP = 2*P_o;
Q = dtheta.'*inv(dP)*dtheta;,p = 1-alpha;,limit = chi2inv(p,size(theta_o,1));
limit=giannis;
figure,bar([0 Q 0]),hold on
line([0 4],[limit limit],'color','r','linestyle','--','linewidth',1.5)
hold off
set(gca,'fontsize',12,'xtick',[]),
title(' Model Parameter Based Method','fontsize',16)
ylabel('Test Statistic','fontsize',16)

function [Q,limit]=fdi_model_residual(model_o,data_o,data_u,method,alpha,max_lag)
%method: 'var' uses the residual variance, 'lik' the likelihood function and 'unc' the residual uncorrelatedness.
res_o = resid(model_o,data_o) ;,res_u = resid(model_o,data_u) ; 
var_res_o = var(res_o.outputdata) ;,var_res_u = var(res_u.outputdata) ;
Nu = size(res_u.outputdata,1) ;,No = size(res_o.outputdata,1) ;,d = length(model_o.parametervector) ;
if strcmp(method,'var')
    disp('Method A: Using the residual variance')    
    Q = var_res_u/var_res_o;
    p = 1-alpha;limit = finv(p,Nu-1,No-d-1);
    figure,bar([0 Q 0])
    hold on
    line([0 4],[limit limit],'color','r','linestyle','--','linewidth',1.5)
    hold off
    set(gca,'fontsize',12,'xtick',[])
    title('Residual Based Method A: using the residual variance','fontsize',16)
    ylabel('Test Statistic','fontsize',16)
elseif strcmp(method,'lik')
    disp('Method B: Using the likelihood function')
    Q = No*var_res_u/var_res_o ;
    p = 1-alpha;,limit = chi2inv(p,No);
    figure,bar([0 Q 0])
    hold on
    line([0 4],[limit limit],'color','r','linestyle','--','linewidth',1.5)
    hold off
    set(gca,'fontsize',12,'xtick',[])
    title('Residual Based Method B: using the likelihood function','fontsize',16)
    ylabel('Test Statistic','fontsize',16)
else 
    disp('Method C: Using the residual uncorrelatedness')
    rho_u = autocorr(res_u.outputdata,max_lag);,rho = rho_u(2:end).';
    Q = No*(No+2)*sum((rho.^2)./(No-(1:max_lag)));
    p = 1-alpha;,limit = chi2inv(p,max_lag-1);
    figure,bar([0 Q 0])
    hold on
    line([0 4],[limit limit],'color','r','linestyle','--','linewidth',1.5)
    hold off
    set(gca,'fontsize',12,'xtick',[])
    title(' Residual Based Method C: using the residual uncorrelatedness','fontsize',16)
    ylabel('Test Statistic','fontsize',16)
    
end

%to assign limit by hand
function [Q,limit]=fdi_model_residual_giannis(model_o,data_o,data_u,method,alpha, giannis, max_lag)
%method: 'var' uses the residual variance, 'lik' the likelihood function and 'unc' the residual uncorrelatedness.
res_o = resid(model_o,data_o) ;,res_u = resid(model_o,data_u) ; 
var_res_o = var(res_o.outputdata) ;,var_res_u = var(res_u.outputdata) ;
Nu = size(res_u.outputdata,1) ;,No = size(res_o.outputdata,1) ;,d = length(model_o.parametervector) ;
if strcmp(method,'var')
    disp('Method A: Using the residual variance')    
    Q = var_res_u/var_res_o;
    p = 1-alpha;limit = finv(p,Nu-1,No-d-1);
    limit = giannis;
    figure,bar([0 Q 0])
    hold on
    line([0 4],[limit limit],'color','r','linestyle','--','linewidth',1.5)
    hold off
    set(gca,'fontsize',12,'xtick',[])
    title('Residual Based Method A: using the residual variance','fontsize',16)
    ylabel('Test Statistic','fontsize',16)
elseif strcmp(method,'lik')
    disp('Method B: Using the likelihood function')
    Q = No*var_res_u/var_res_o ;
    p = 1-alpha;,limit = chi2inv(p,No);
    limit = giannis;
    figure,bar([0 Q 0])
    hold on
    line([0 4],[limit limit],'color','r','linestyle','--','linewidth',1.5)
    hold off
    set(gca,'fontsize',12,'xtick',[])
    title('Residual Based Method B: using the likelihood function','fontsize',16)
    ylabel('Test Statistic','fontsize',16)
else 
    disp('Method C: Using the residual uncorrelatedness')
    rho_u = autocorr(res_u.outputdata,max_lag);,rho = rho_u(2:end).';
    Q = No*(No+2)*sum((rho.^2)./(No-(1:max_lag)));
    p = 1-alpha;,limit = chi2inv(p,max_lag-1);
    limit = giannis;
    figure,bar([0 Q 0])
    hold on
    line([0 4],[limit limit],'color','r','linestyle','--','linewidth',1.5)
    hold off
    set(gca,'fontsize',12,'xtick',[])
    title(' Residual Based Method C: using the residual uncorrelatedness','fontsize',16)
    ylabel('Test Statistic','fontsize',16)
    
end

function [ac,a,sac,sx,up,x_axis]=acf(x,lag,y)
lim=max(size(x));
if nargin==2
    a=xcov(x,'coeff');
    tmp=max(size(a));
    ac=a(((tmp-1)/2)+1:((tmp-1)/2)+1+lag,1);
    sac=size(ac);
    up=(1.96/sqrt(lim))*ones(lag,1);
    lo=-up;
    x_axis=(0:1:lag-1)';
    sx=size(x_axis);
    bar(x_axis,ac(2:length(ac)))
    hold
    plot(x_axis,up,x_axis,lo)
    hold
    axis([0 lag-1 -1 1])
else
    a=xcov(x,y,'coeff');
    tmp=max(size(a));
    ac=a(((tmp-1)/2)+1:((tmp-1)/2)+1+lag,1);
    sac=size(ac);
    up=(1.96/sqrt(lim))*ones(lag,1);
    lo=-up;
    x_axis=(0:1:lag-1)';
    sx=size(x_axis);
    bar(x_axis,ac(2:length(ac)))
    hold
    plot(x_axis,up,x_axis,lo)
    hold
    axis([0 lag-1 -1 1])
end
% Not testeeeeeeeeed
function [modal]=the_modals(num,den,Fs,s,m)
% Estimation of the modal parameters (works inside the monte_carlo.m)
for i=1:s*m;
   [Delta,fn,z] = disper_new120106(num(i,:),den,Fs);
   Delta_full(:,i) = Delta;
   clear Delta
end

Delta_na=[];
for i = 1:length(fn)
   Delta_na(i,:) = max(abs(Delta_full(i,:)));
end
modal = [fn z Delta_na];


function [Delta,fn,z,R,lambda]=disper_new120106(num,den,Fs)

% num		: The numerator of the transfer function
% den		: The denominator of the transfer function
% Fs		: The sampling frequency (Hz)
% Delta	: The precentage dispersion
% fn		: The corresponding frequencies (Hz)
% z		: The corresponding damping (%)
% R		: The residues of the discrete system
% Mag		: The magnitude of the corresponding poles
% This function computes the dispersion of each frequency of a system. The System is  
% enetred as a transfer function. In case the order of numerator polynomial is greater than 
% that of the denominator the polynomial division is apllied, and the dispersion is considered at
% the remaine tf. The analysis is done using the Residuez routine of MATLAB.
% The results are printed in the screen in asceding order of frequencies.
% This routine displays only the dispersions from the natural frequencies (Complex Poles).

% REFERENCE[1]:  MIMO LMS-ARMAX IDENTIFICATION OF VIBRATING STRUCTURES - A Critical Assessment 
% REFERENCE[2]:  PANDIT WU

%--------------------------------------------
% Created	: 08 December 1999.
% Author(s)	: A. Florakis & K.A.Petsounis
% Updated	: 17 February 2006.
%--------------------------------------------

% Sampling Period
Ts=1/Fs;

% Calculate the residues of the Transfer Function
num=num(:).';
den=den(:).';

[R,P,K]=residuez(num,den);

R=R(:);P=P(:);K=K(:);


% Distinction between Real & Image Residues  
[R,P,l_real,l_imag]=srtrp(R,P,'all');

% Construction of M(k) (Eq. 45 REF[1])
for k=1:length(P)
   ELEM=R./(ones(length(P),1)-P(k).*P);             % Construction of the terms Ri/1-pk*pi
   M(k)=R(k)*sum(ELEM);										 % Calculation of M(k)  
   clear ELEM
end

% Dispersion of Modes (Eq. 46 & 47 REF[1])
D_real=real(M(1:l_real));D_imag=M(l_real+1:l_imag+l_real);
D=[D_real';D_imag'+conj(D_imag)'];


Delta=100*D./sum(D);     % Delta (%) 

% Sorting Dispersions by asceding Frequency 
lambda=P(l_real+1:l_imag+l_real);
Wn=Fs*abs(log(lambda));          % Corresponding Frequencies 
z= -cos(angle(log(lambda)));     % Damping Ratios
[Wn sr]=sort(Wn);
fn=Wn./(2*pi);                   % fn rad/sec==>Hz 
z=100*z(sr);                     % damping ratio(%) 

Delta=Delta(l_real+1:l_real+l_imag);
Delta=Delta(sr);

% Sorting Poles by asceding Frequency
lambda=lambda(sr);
R_imag_plus=R(l_real+1:l_real+l_imag);
R=R_imag_plus(sr);
%R=R.*Fs; 		% Residues for Impulse Invariance Method
%R=R./R(1);  	% Normalized Residues
   
Mag=abs(lambda);   % Magnitude of poles
Mag=Mag(sr);

%--------------------------------------------------------
% 				Results
%--------------------------------------------------------
form1= '%1d' ;
form2 = '%7.4e';  

if nargout==0,      
   % Print results on the screen. First generate corresponding strings:
   nmode = dprint([1:l_imag]','Mode',form1);
   wnstr = dprint(fn,'Frequency (Hz)',form2);
   zstr = dprint(z,'Damping (%)',form2);
   dstr = dprint(Delta,'Dispersion (%)',form2);
   rstr = dprint(R,'Norm. Residues ',form2);
   mrstr = dprint(lambda,'Poles',form1);
disp([nmode wnstr zstr dstr rstr mrstr	]);
else
end
function [R,P,nr,ni]=srtrp(R,P,flg)

% if flg='hf' ==> Real Residues & Imag Residues (from Real poles)
% else if flg='all' ==> Real Residues & Imag Residues (from positive poles) & Imag Residues (from negative Poles)

R_real=[];P_real=[];
R_imag_plus=[];P_imag_plus=[];
R_imag=[];P_imag=[];

for i=1:length(R)
   if imag(P(i))==0
      R_real=[R_real;R(i)];P_real=[P_real;P(i)];
   elseif imag(P(i))>0
      R_imag_plus=[R_imag_plus;R(i)];P_imag_plus=[P_imag_plus;P(i)];
   else
      R_imag=[R_imag;R(i)];P_imag=[P_imag;P(i)];
   end
end
switch flg
case 'all'
   R=[R_real;R_imag_plus;R_imag];P=[P_real;P_imag_plus;P_imag];
   nr=length(P_real);ni=length(P_imag);
case 'hf'
   P=[P_real;P_imag_plus];R=[R_real;R_imag_plus];
   nr=length(P_real);ni=length(P_imag);
end
function [Delta,fn,z,R,lambda]=disper_new(num,den,Fs)

% num		: The numerator of the transfer function
% den		: The denominator of the transfer function
% Fs		: The sampling frequency (Hz)
% Delta	: The precentage dispersion
% fn		: The corresponding frequencies (Hz)
% z		: The corresponding damping (%)
% R		: The residues of the discrete system
% Mag		: The magnitude of the corresponding poles
% This function computes the dispersion of each frequency of a system. The System is  
% enetred as a transfer function. In case the order of numerator polynomial is greater than 
% that of the denominator the polynomial division is apllied, and the dispersion is considered at
% the remaine tf. The analysis is done using the Residuez routine of MATLAB.
% The results are printed in the screen in asceding order of frequencies.
% This routine displays only the dispersions from the natural frequencies (Complex Poles).

% REFERENCE[1]:  MIMO LMS-ARMAX IDENTIFICATION OF VIBRATING STRUCTURES - A Critical Assessment 
% REFERENCE[2]:  PANDIT WU

%--------------------------------------------
% Created	: 08 December 1999.
% Author(s)	: A. Florakis & K.A.Petsounis
% Updated	: 16 February 1999.
%--------------------------------------------

% Sampling Period
Ts=1/Fs;

% Calculate the residues of the Transfer Function
num=num(:).';
den=den(:).';

%---------------------------------------------------
% For Analysis with the contant term
%[UPOLOIPO,PILIKO]=deconv(fliplr(num),fliplr(den));
%UPOLOIPO=fliplr(UPOLOIPO);
%PILIKO=fliplr(PILIKO);
%---------------------------------------------------


[R,P,K]=residuez(num,den);
% keyboard
%OROS=PILIKO(1);
% Make rows columns
%R=R(:);P=P(:);K=K(:);
R=R(:);P=P(:);K=K(:);


% Distinction between Real & Image Residues  
[R,P,l_real,l_imag]=srtrp(R,P,'all');

% Construction of M(k) (Eq. 45 REF[1])
for k=1:length(P)
   ELEM=R./(ones(length(P),1)-P(k).*P);             % Construction of the terms Ri/1-pk*pi
   M(k)=R(k)*sum(ELEM);										 % Calculation of M(k)  
   clear ELEM
end

% Dispersion of Modes (Eq. 46 & 47 REF[1])
D_real=real(M(1:l_real));D_imag=M(l_real+1:l_imag+l_real);
D=[D_real';D_imag'+conj(D_imag)'];

% Precentage Dispersion (Eq. 48 REF[1])
%if ~isempty(K)
%   D=D(:).';
%   VARY=[K^2 2*K*OROS D]; 
%   Delta=100*VARY./sum(VARY);
	% tests   sum(Delta);Delta(1);Delta(2)
%   Delta=Delta(3:length(Delta))'
%else
%  disp('mhn mpeis')
	Delta=100*D./sum(D);
	%Delta=D_imag./sum(D_imag)
   sum(Delta);
   %dou=K^2/sum(D+K^2)
%end
%keyboard

% Sorting Dispersions by asceding Frequency 
lambda=P(l_real+1:l_imag+l_real);
Wn=Fs*abs(log(lambda));          % Corresponding Frequencies 
z= -cos(angle(log(lambda)));     % Damping Ratios
[Wn sr]=sort(Wn);
fn=Wn./(2*pi);                   % rad/sec==>Hz 
z=100*z(sr);

Delta=Delta(l_real+1:l_real+l_imag);
Delta=Delta(sr);

% Sorting Poles by asceding Frequency
lambda=lambda(sr);
R_imag_plus=R(l_real+1:l_real+l_imag);
R=R_imag_plus(sr);
%R=R.*Fs; 		% Residues for Impulse Invariance Method
%R=R./R(1);  	% Normalized Residues
   
Mag=abs(lambda);   % Magnitude of poles
Mag=Mag(sr);

%--------------------------------------------------------
% 				Results
%--------------------------------------------------------
form1= '%1d' ;
form2 = '%7.4e';  

if nargout==0,      
   % Print results on the screen. First generate corresponding strings:
   nmode = dprint([1:l_imag]','Mode',form1);
   wnstr = dprint(fn,'Frequency (Hz)',form2);
   zstr = dprint(z,'Damping (%)',form2);
   dstr = dprint(Delta,'Dispersion (%)',form2);
   rstr = dprint(R,'Norm. Residues ',form2);
   mrstr = dprint(lambda,'Poles',form1);
disp([nmode wnstr zstr dstr rstr mrstr	]);
else
end


