function SI_FDI_COURSE_giannis(Fs)
clear;
clc;
close all;
% This function is running a demo for: i) The non-parametric and parametric
% identification of an a/c structure. ii) The damage detection of small added masses
% on the a/c structure based on non-parametric and parametric statistical time series methods.

disp('------------- NON-PARAMETRIC & PARAMETRIC IDENTIFICATION PROCEDURE -------------  ');

%Load the necessary signals
%load si_data
load ../data/DATA_1047304.mat

%Sample frequency
Fs = 256; % (Hz)

%Healthy signals
x1=Signals(1:length(Signals),1);
y1=Signals(1:length(Signals),2);
x2=Signals(1:length(Signals),3);
y2=Signals(1:length(Signals),4);

%Discrete time
TT=0:1/Fs:(length(y1)-1)/Fs;

%Plot the input-output signals
disp('   ')
disp('  INPUT-OUTPUT SIGNALS OF THE HEALTHY STRUCTURE  ');
figure
subplot(2,1,1),plot(TT,x1)
set(gca,'xticklabel',[],'fontsize',16),box on
title('Excitation Signal 1','Fontname','TimesNewRoman','Fontsize',20)
ylabel('Force (Nt)','Fontname','TimesNewRoman','Fontsize',20)
subplot(2,1,2),plot(TT,y1)
set(gca,'fontsize',16),box on
title('Response Signal 1','Fontname','TimesNewRoman','Fontsize',20)
ylabel('Acceleration (m/s^2)','Fontname','TimesNewRoman','Fontsize',20)
xlabel('Time (s)','Fontname','TimesNewRoman','Fontsize',20)

figure
subplot(2,1,1),plot(TT,x2)
set(gca,'xticklabel',[],'fontsize',16),box on
title('Excitation Signal 2','Fontname','TimesNewRoman','Fontsize',20)
ylabel('Force (Nt)','Fontname','TimesNewRoman','Fontsize',20)
subplot(2,1,2),plot(TT,y2)
set(gca,'fontsize',16),box on
title('Response Signal 2','Fontname','TimesNewRoman','Fontsize',20)
ylabel('Acceleration (m/s^2)','Fontname','TimesNewRoman','Fontsize',20)
xlabel('Time (s)','Fontname','TimesNewRoman','Fontsize',20)
disp('(press a key)')
pause  % ,close
disp('   ')

%Plot the normalized to unity input-output signals
disp(' NORMALIZED INPUT-OUTPUT SIGNALS OF THE HEALTHY STRUCTURE  ');

%Mean removal and normalization with the max values
% Y1 = detrend(y1(:,1));
% Y1 = Y1./max(abs(Y1));
% X1 = detrend(x1);
% X1 = X1./max(abs(X1));

Y1 = detrend(y1(:,1));
Y1 = Y1./std(Y1);
X1 = detrend(x1);
X1 = X1./std(X1);

% Y2 = detrend(y2(:,1));
% Y2 = Y2./max(abs(Y2));
% X2 = detrend(x2);
% X2 = X2./max(abs(X2));

Y2 = detrend(y2(:,1));
Y2 = Y2./std(Y2);
X2 = detrend(x2);
X2 = X2./std(X2);

%Form input-output data in the "iddata" object format 
dat_io1 = iddata(Y1,X1,1/Fs);
dat_o1 = iddata(Y1,[],1/Fs);
dat_oo1 = iddata([X1 Y1],[],1/Fs);

dat_io2 = iddata(Y2,X2,1/Fs);
dat_o2 = iddata(Y2,[],1/Fs);
dat_oo2 = iddata([X2 Y2],[],1/Fs);

figure
subplot(2,1,1),plot(TT,X1)
set(gca,'fontsize',16),box on
title('Normalized Excitation Signal 1','Fontname','TimesNewRoman','Fontsize',20)
ylabel('Force (Nt)','Fontname','TimesNewRoman','Fontsize',20)
subplot(2,1,2),plot(TT,Y1)
set(gca,'fontsize',16),box on
title('Normalized Response Signal 1','Fontname','TimesNewRoman','Fontsize',20)
ylabel('Acceleration (m/s^2)','Fontname','TimesNewRoman','Fontsize',20)
xlabel('Time (s)','Fontname','TimesNewRoman','Fontsize',20)

figure
subplot(2,1,1),plot(TT,X2)
set(gca,'fontsize',16),box on
title('Normalized Excitation Signal 2','Fontname','TimesNewRoman','Fontsize',20)
ylabel('Force (Nt)','Fontname','TimesNewRoman','Fontsize',20)
subplot(2,1,2),plot(TT,Y2)
set(gca,'fontsize',16),box on
title('Normalized Response Signal 2','Fontname','TimesNewRoman','Fontsize',20)
ylabel('Acceleration (m/s^2)','Fontname','TimesNewRoman','Fontsize',20)
xlabel('Time (s)','Fontname','TimesNewRoman','Fontsize',20)
disp('(press a key)')
pause  % ,close
disp('   ')
% Separate signal into verification part and estimation part
lll=9000
X1ver=X1((lll+1):10240);
Y1ver=Y1((lll+1):10240);
X2ver=X2((lll+1):10240);
Y2ver=Y2((lll+1):10240);

X1=X1(1:lll);
Y1=Y1(1:lll);
X2=X2(1:lll);
Y2=Y2(1:lll);

TT=0:1/Fs:(length(Y1)-1)/Fs;
TTver=0:1/Fs:(length(Y1ver)-1)/Fs;
%Reform input-output data in the "iddata" object format for estimation part
dat_io1 = iddata(Y1,X1,1/Fs);
dat_o1 = iddata(Y1,[],1/Fs);
dat_oo1 = iddata([X1 Y1],[],1/Fs);

dat_io2 = iddata(Y2,X2,1/Fs);
dat_o2 = iddata(Y2,[],1/Fs);
dat_oo2 = iddata([X2 Y2],[],1/Fs);

figure
subplot(2,1,1),plot(TT,X1)
set(gca,'fontsize',16),box on
title('Normalized Excitation Signal 1-Estimation part','Fontname','TimesNewRoman','Fontsize',20)
ylabel('Force (Nt)','Fontname','TimesNewRoman','Fontsize',20)
subplot(2,1,2),plot(TT,Y1)
set(gca,'fontsize',16),box on
title('Normalized Response Signal 1-Estimation part','Fontname','TimesNewRoman','Fontsize',20)
ylabel('Acceleration (m/s^2)','Fontname','TimesNewRoman','Fontsize',20)
xlabel('Time (s)','Fontname','TimesNewRoman','Fontsize',20)

figure
subplot(2,1,1),plot(TT,X2)
set(gca,'fontsize',16),box on
title('Normalized Excitation Signal 2-Estimation part','Fontname','TimesNewRoman','Fontsize',20)
ylabel('Force (Nt)','Fontname','TimesNewRoman','Fontsize',20)
subplot(2,1,2),plot(TT,Y2)
set(gca,'fontsize',16),box on
title('Normalized Response Signal 2-Estimation part','Fontname','TimesNewRoman','Fontsize',20)
ylabel('Acceleration (m/s^2)','Fontname','TimesNewRoman','Fontsize',20)
xlabel('Time (s)','Fontname','TimesNewRoman','Fontsize',20)
disp('(press a key)')

figure
subplot(2,1,1),plot(TTver,X1ver)
set(gca,'fontsize',16),box on
title('Normalized Excitation Signal 1-Verificaiton part','Fontname','TimesNewRoman','Fontsize',20)
ylabel('Force (Nt)','Fontname','TimesNewRoman','Fontsize',20)
subplot(2,1,2),plot(TTver,Y1ver)
set(gca,'fontsize',16),box on
title('Normalized Response Signal 1-Verificaiton part','Fontname','TimesNewRoman','Fontsize',20)
ylabel('Acceleration (m/s^2)','Fontname','TimesNewRoman','Fontsize',20)
xlabel('Time (s)','Fontname','TimesNewRoman','Fontsize',20)

figure
subplot(2,1,1),plot(TTver,X2ver)
set(gca,'fontsize',16),box on
title('Normalized Excitation Signal 2-Verificaiton part','Fontname','TimesNewRoman','Fontsize',20)
ylabel('Force (Nt)','Fontname','TimesNewRoman','Fontsize',20)
subplot(2,1,2),plot(TTver,Y2ver)
set(gca,'fontsize',16),box on
title('Normalized Response Signal 2-Verificaiton part','Fontname','TimesNewRoman','Fontsize',20)
ylabel('Acceleration (m/s^2)','Fontname','TimesNewRoman','Fontsize',20)
xlabel('Time (s)','Fontname','TimesNewRoman','Fontsize',20)
disp('(press a key)')
pause  % ,close
disp('   ')

% Spectrogram

figure;
spectrogram(X1ver,1000,950,1024,Fs,'yaxis')
title('Excitation Spectrogram');
figure;
spectrogram(Y1ver,1000,950,1024,Fs,'yaxis')
title('Response Spectrogram');
view(-45,65)
%%

clear Pyy* w

%--------------------------------------------------------------------------
WINDOW1=256; WINDOW2=512; WINDOW3=1024; WINDOW4=2048; OVERLAP=0.8;
%--------------------------------------------------------------------------

[Pyy1,w1] = pwelch(Y1,WINDOW1,round(OVERLAP*WINDOW1),WINDOW1,fs);
[Pyy2,w2] = pwelch(Y1,WINDOW2,round(OVERLAP*WINDOW2),WINDOW2,fs);
[Pyy3,w3] = pwelch(Y1,WINDOW3,round(OVERLAP*WINDOW3),WINDOW3,fs);
[Pyy4,w4] = pwelch(Y1,WINDOW4,round(OVERLAP*WINDOW4),WINDOW4,fs);

i=10; ii=14;

figure
plot(w1,20*log10(abs(Pyy1)),'k'),hold on
plot(w2,20*log10(abs(Pyy2)),'b')
plot(w3,20*log10(abs(Pyy3)),'r')
plot(w4,20*log10(abs(Pyy4)),':g')
set(gca,'fontsize',i)
xlim([0 fs/2])
title('Welch based output spectrum (window effect)','Fontname','TimesNewRoman','Fontsize',ii)
ylabel('Magnitude (dB)','Fontname','TimesNewRoman','Fontsize',ii)
xlabel('Frequency (Hz)','Fontname','TimesNewRoman','Fontsize',ii)
legend('256','512','1024','2048','Location','SouthEast','Orientation','horizontal')

pause % ,close

%Welch based spectra
disp(' THE WELCH BASED EXCITATION-RESPONSE SPECTRA ');

%PSD estimation parameters
%--------------------------------------------------------------------------
WINDOW = 1024; NFFT = 1024; OVERLAP = 0.8; %Fs = 256; 
%--------------------------------------------------------------------------
[Pxx1,w1] = pwelch(X1,WINDOW,round(OVERLAP*WINDOW),NFFT,Fs); % excitation spectrum 1
[Pyy1,w1] = pwelch(Y1,WINDOW,round(OVERLAP*WINDOW),NFFT,Fs); % response spectrum 1
[Pxy1,wxy1] = cpsd(X1,Y1,WINDOW,round(OVERLAP*WINDOW),NFFT,Fs);

[Pxx2,w2] = pwelch(X2,WINDOW,round(OVERLAP*WINDOW),NFFT,Fs); % excitation spectrum 2
[Pyy2,w2] = pwelch(Y2,WINDOW,round(OVERLAP*WINDOW),NFFT,Fs); % response spectrum 2
[Pxy2,wxy2] = cpsd(X2,Y2,WINDOW,round(OVERLAP*WINDOW),NFFT,Fs);

disp(' ');
disp(' Welch based estimation       ');
disp('-----------------------------------');
disp('Window        NFFT      Overlap (%)');
disp('-----------------------------------');
disp([ WINDOW; NFFT; OVERLAP]');

%PSDs for excitation,response 1
figure
subplot(2,1,1),plot(w1,20*log10(abs(Pxx1)))
set(gca,'fontsize',16),box on
axis([5,128,-140,-40])
title('Welch based excitation 1 spectrum','Fontname','TimesNewRoman','Fontsize',20)
ylabel('PSD (dB)','Fontname','TimesNewRoman','Fontsize',20)
subplot(2,1,2),plot(w1,20*log10(abs(Pyy1)))
set(gca,'fontsize',16),box on
axis([5,128,-140,-40])
title('Welch based response 1 spectrum','Fontname','TimesNewRoman','Fontsize',20)
ylabel('PSD (dB)','Fontname','TimesNewRoman','Fontsize',20)
xlabel('Frequency (Hz)','Fontname','TimesNewRoman','Fontsize',20)

%------ CSD for exc-response 1-----
figure
subplot(2,1,1);plot(wxy1,20*log10(abs(Pxy1)))
set(gca,'fontsize',16),box on
axis([5,128,-140,-40])
title('Welch based CSD 1','Fontname','TimesNewRoman','Fontsize',20)
ylabel('CSD magn(dB)','Fontname','TimesNewRoman','Fontsize',20)
subplot(2,1,2);plot(wxy1,rad2deg(angle(Pxy1)))
set(gca,'fontsize',16),box on
axis([5,128,-200,200])
%title('Welch based CSD 1','Fontname','TimesNewRoman','Fontsize',20)
ylabel('CSD phase(deg)','Fontname','TimesNewRoman','Fontsize',20)

%PSDs for excitation,response 2
figure
subplot(2,1,1),plot(w2,20*log10(abs(Pxx2)))
set(gca,'fontsize',16),box on
axis([5,128,-140,-40])
title('Welch based excitation 2 spectrum','Fontname','TimesNewRoman','Fontsize',20)
ylabel('PSD (dB)','Fontname','TimesNewRoman','Fontsize',20)
subplot(2,1,2),plot(w2,20*log10(abs(Pyy2)))
set(gca,'fontsize',16),box on
axis([5,128,-140,-40])
title('Welch based response 2 spectrum','Fontname','TimesNewRoman','Fontsize',20)
ylabel('PSD (dB)','Fontname','TimesNewRoman','Fontsize',20)
xlabel('Frequency (Hz)','Fontname','TimesNewRoman','Fontsize',20)

%------ CSD for exc-response 1-----
figure
subplot(2,1,1);plot(wxy2,20*log10(abs(Pxy2)))
set(gca,'fontsize',16),box on
axis([5,128,-140,-40])
title('Welch based CSD 2','Fontname','TimesNewRoman','Fontsize',20)
ylabel('CSD magn(dB)','Fontname','TimesNewRoman','Fontsize',20)
subplot(2,1,2);plot(wxy2,rad2deg(angle(Pxy2)))
set(gca,'fontsize',16),box on
axis([5,128,-200,200])
%title('Welch based CSD 1','Fontname','TimesNewRoman','Fontsize',20)
ylabel('CSD phase(deg)','Fontname','TimesNewRoman','Fontsize',20)
disp('(press a key)')
pause % ,close
disp('   ')

%interval frf by jason
%interv_new(X1,Y1,1024*0.8,1024,256);

pause
%%
%Welch based FRF 
disp(' WELCH BASED FREQUENCY RESPONSE FUNCTION ')

%FRF estimation parameters
%--------------------------------------------------------------------------
WINDOW = 1024; NFFT = 1024; OVERLAP = 0.8; %%Fs = 256; %estimation parameters <----------
%--------------------------------------------------------------------------
[Txy1,w1] = tfestimate(X1,Y1,WINDOW,round(OVERLAP*1024),NFFT,Fs); % FRF estimation
ph1=angle(Txy1); Ph1=rad2deg(ph1);

[Txy2,w2] = tfestimate(X2,Y2,WINDOW,round(OVERLAP*1024),NFFT,Fs); % FRF estimation
ph2=unwrap(angle(Txy2)); Ph2=(ph2.*180)./pi;

disp(' ');
disp('      Welch based FRF estimation       ');
disp('-----------------------------------');
disp('Window        NFFT      Overlap (%)');
disp('-----------------------------------');
disp([ WINDOW; NFFT; OVERLAP]');
figure
subplot(2,1,1),plot(w1,20*log10(abs(Txy1)))
set(gca,'fontsize',16),box on
axis([5,128,-60,40])
title('Welch based FRF magnitude 1','Fontname','TimesNewRoman','Fontsize',20)
ylabel('Magnitude (dB)','Fontname','TimesNewRoman','Fontsize',20)
subplot(2,1,2),plot(w1,Ph1)
set(gca,'fontsize',16),box on
xlim([5,128])
title('Welch based FRF phase 1','Fontname','TimesNewRoman','Fontsize',20)
ylabel('Phase (deg)','Fontname','TimesNewRoman','Fontsize',20)
xlabel('Frequency (Hz)','Fontname','TimesNewRoman','Fontsize',20)

figure
subplot(2,1,1),plot(w2,20*log10(abs(Txy2)))
set(gca,'fontsize',16),box on
axis([5,128,-60,40])
title('Welch based FRF magnitude 2','Fontname','TimesNewRoman','Fontsize',20)
ylabel('Magnitude (dB)','Fontname','TimesNewRoman','Fontsize',20)
subplot(2,1,2),plot(w2,Ph2)
set(gca,'fontsize',16),box on
xlim([5,128])
title('Welch based FRF phase 2','Fontname','TimesNewRoman','Fontsize',20)
ylabel('Phase (deg)','Fontname','TimesNewRoman','Fontsize',20)
xlabel('Frequency (Hz)','Fontname','TimesNewRoman','Fontsize',20)
%%
%----------------------------------------------------------------------
%------------------------Coherence function----------------------------
%----------------------------------------------------------------------
[coherence1,wcoh1] = mscohere(X1,Y1,WINDOW,round(OVERLAP*1024),NFFT,Fs); 
[coherence2,wcoh2] = mscohere(X2,Y2,WINDOW,round(OVERLAP*1024),NFFT,Fs); 

figure
subplot(2,1,1);plot(wcoh1,coherence1);
set(gca,'fontsize',16),box on
axis([5,128,0,1.05])
title('Welch based coherence function 1','Fontname','TimesNewRoman','Fontsize',20)
ylabel('Coherence','Fontname','TimesNewRoman','Fontsize',20)
%xlabel('Frequency (Hz)','Fontname','TimesNewRoman','Fontsize',20)
subplot(2,1,2);plot(wcoh2,coherence2);
set(gca,'fontsize',16),box on
axis([5,128,0,1.05])
title('Welch based coherence function 2','Fontname','TimesNewRoman','Fontsize',20)
ylabel('Coherence','Fontname','TimesNewRoman','Fontsize',20)
xlabel('Frequency (Hz)','Fontname','TimesNewRoman','Fontsize',20)
disp('(press a key)')
pause % ,close
disp('   ')

figure
plot(wcoh1,coherence1);
set(gca,'fontsize',16),box on
axis([5,128,0,1.05])
title('Welch based coherence function 1','Fontname','TimesNewRoman','Fontsize',20)
ylabel('Coherence','Fontname','TimesNewRoman','Fontsize',20)
xlabel('Frequency (Hz)','Fontname','TimesNewRoman','Fontsize',20)
%%
%--------------------------------------------------------------------------
%                    Histogram and Normal Probability Plot
%--------------------------------------------------------------------------

i=10; ii=14; 
N=length(Y1);

figure
subplot(2,1,1),histfit(X1,round(sqrt(N)))
set(gca,'fontsize',i),box on
feval('title',sprintf('Histogram'),'Fontname','TimesNewRoman','fontsize',ii)
subplot(2,1,2), normplot(X1)
set(gca,'fontsize',i),box on
sgtitle('Normal Distribution Check for Excitation 1')

figure
subplot(2,1,1),histfit(Y1,round(sqrt(N)))
set(gca,'fontsize',i),box on
feval('title',sprintf('Histogram'),'Fontname','TimesNewRoman','fontsize',ii)
subplot(2,1,2), normplot(Y1)
set(gca,'fontsize',i),box on
sgtitle('Normal Distribution Check for Response 1')

figure
subplot(2,1,1),histfit(X2,round(sqrt(N)))
set(gca,'fontsize',i),box on
feval('title',sprintf('Histogram'),'Fontname','TimesNewRoman','fontsize',ii)
subplot(2,1,2), normplot(X2)
set(gca,'fontsize',i),box on
sgtitle('Normal Distribution Check for Excitation 2')

figure
subplot(2,1,1),histfit(Y2,round(sqrt(N)))
set(gca,'fontsize',i),box on
feval('title',sprintf('Histogram'),'Fontname','TimesNewRoman','fontsize',ii)
subplot(2,1,2), normplot(Y2)
set(gca,'fontsize',i),box on
sgtitle('Normal Distribution Check for Response 2')
pause
%%

%--------------------------------------------------------------------------
%                                  ACFs 
%--------------------------------------------------------------------------

%if nx==1; 
    
%    figure
 %   acf_wn(X1,100,0.8);
   % ylim([-1 1])
  %  title('Excitation (input) AutoCorrelation Function (ACF)','fontsize',ii,'Fontname','TimesNewRoman')
 
%     [acf_x,lags,bounds_acf]=autocorr(X,100); 
%     figure
%     bar(acf_x,0.5)
%     line([0 size(acf_x,1)],[bounds_acf(1) bounds_acf(1)],'color','r')
%     line([0 size(acf_x,1)],[bounds_acf(2) bounds_acf(2)],'color','r')
%     axis([0 100 -1 1])
%     set(gca,'fontsize',i,'Fontname','TimesNewRoman')
%     title('Excitation (input) AutoCorrelation Function (ACF)','fontsize',ii,'Fontname','TimesNewRoman')
%     ylabel('Excitation ACF','fontsize',ii,'Fontname','TimesNewRoman','interpreter','tex')
%     xlabel('Lag','fontsize',ii,'Fontname','TimesNewRoman','interpreter','tex')
%end

%[ccf_xy,lags,bounds_ccf]=crosscorr(X,Y,100);

i=10; ii=14;
lagselection=100;
figure
acf_wn(X1,lagselection,0.8);
ylim([-1 1])
set(gca,'fontsize',i,'Fontname','TimesNewRoman')
feval('title',sprintf('Excitation 1 ACF'),'Fontname','TimesNewRoman','fontsize',ii)
figure
acf_wn(Y1,lagselection,0.8);
ylim([-1 1])
set(gca,'fontsize',i,'Fontname','TimesNewRoman')
feval('title',sprintf('Response 1 ACF'),'Fontname','TimesNewRoman','fontsize',ii)
figure
ccf_wn(X1,Y1,lagselection,0.8);
ylim([-1 1])
set(gca,'fontsize',i,'Fontname','TimesNewRoman')
feval('title',sprintf('Excitation-Response 1 CCF'),'Fontname','TimesNewRoman','fontsize',ii)

figure
acf_wn(X2,lagselection,0.8);
ylim([-1 1])
set(gca,'fontsize',i,'Fontname','TimesNewRoman')
feval('title',sprintf('Excitation 2 ACF'),'Fontname','TimesNewRoman','fontsize',ii)
figure
acf_wn(Y2,lagselection,0.8);
ylim([-1 1])
set(gca,'fontsize',i,'Fontname','TimesNewRoman')
feval('title',sprintf('Response 2 ACF'),'Fontname','TimesNewRoman','fontsize',ii)
figure
ccf_wn(X2,Y2,lagselection,0.8);
ylim([-1 1])
set(gca,'fontsize',i,'Fontname','TimesNewRoman')
feval('title',sprintf('Excitation-Response 2 CCF'),'Fontname','TimesNewRoman','fontsize',ii)

%     [acf_y(:,output),lags,bounds_acf]=autocorr(Y(:,output),100);
%     subplot(outputs,1,output),bar(acf_y(:,output),0.5)
%     line([0 size(acf_y,1)],[bounds_acf(1) bounds_acf(1)],'color','r')
%     line([0 size(acf_y,1)],[bounds_acf(2) bounds_acf(2)],'color','r')
%     axis([0 100 -1 1])
%     set(gca,'fontsize',i,'Fontname','TimesNewRoman')
%     feval('title',sprintf('Response ACF (output %d)',output),'Fontname','TimesNewRoman','fontsize',ii)
%     ylabel('Response ACF','fontsize',ii,'Fontname','TimesNewRoman','interpreter','tex')
%     xlabel('Lag','fontsize',ii,'Fontname','TimesNewRoman','interpreter','tex')


pause

%%

%------------------------not tested--------------------------------
%--------------------------------------------------------------------------
%                           ARX Identification
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

clc

models_armax=[];
%models_arx=[];

DATA1=iddata(Y2, X2,1/Fs);
DATA2=iddata(Y1, X1,1/Fs);

        disp('-----------------------------------');
        disp('        ARMA Identification        ')
        disp('-----------------------------------');

        
    
        minar=input('Give minimum AutoRegressive (AR) order: \n');
        maxar=input('Give maximum AutoRegressive (AR) order: \n');
        opt = armaxOptions;
        opt = armaxOptions('EnforceStability',true);
        tic
        for order=minar:maxar
            disp('Order is: \n');
            disp(order);
            
%             opt = armaxOptions;
%             opt.Focus = 'simulation';
%             opt.SearchMethod = 'lm';
%             opt.SearchOptions.MaxIterations = 10;
%             opt.Display = 'on';
            models_armax{order}=armax(DATA1,[order order order 0]);
             %models_armax{order}=n4sid(DATA1,order); %see freq.stab.plot
%             modal_armax{order}=the_modals(models_armax{order}.c,models_armax{order}.a,Fs,1,1);
        end
        toc
         
        pause

Yp_armax=cell(1,maxar); rss_armax=zeros(1,maxar); BIC_armax=zeros(1,maxar);

for order=minar:maxar
    BIC_armax(order)=log(models_armax{order}.noisevariance)+...
         (size(models_armax{order}.parametervector,1)*log(N))/N;
end

for order=minar:maxar
    Yp_armax{order}=predict(models_armax{order},DATA1,1);
    rss_armax(order)=100*(norm(DATA1.outputdata-Yp_armax{order}.outputdata)^2)/(norm(DATA1.outputdata)^2);
end


   %% 
%--------------------------------------------------------------------------
%-----------------------------  BIC-RSS plot ------------------------------
%--------------------------------------------------------------------------

i=10; ii=14;

% figure
% subplot(2,1,1),plot(minar:maxar,BIC_arx(minar:maxar),'-o')
% xlim([minar maxar])
% title('BIC criterion for ARX','Fontname','TimesNewRoman','Fontsize',ii)
% ylabel('BIC','Fontname','TimesNewRoman','Fontsize',ii)
% set(gca,'fontsize',i)
% subplot(2,1,2),plot(minar:maxar,rss_arx(minar:maxar),'-o')
% xlim([minar maxar])
% title('RSS/SSS criterion for ARX','Fontname','TimesNewRoman','Fontsize',ii)
% ylabel('RSS/SSS (%)','Fontname','TimesNewRoman','Fontsize',ii)
% xlabel('ARX(n,n)','Fontname','TimesNewRoman','Fontsize',ii)
% set(gca,'fontsize',i)

% subplot(3,1,3),plot(minar:maxar,aic(models{minar:maxar}),'-o')
% xlim([minar maxar])
% title('AIC criterion','Fontname','TimesNewRoman','Fontsize',ii)
% ylabel('AIC','Fontname','TimesNewRoman','Fontsize',ii)
% set(gca,'fontsize',i)

figure
subplot(2,1,1),plot(minar:maxar,BIC_armax(minar:maxar),'-o')
xlim([minar maxar])
title('BIC criterion for ARMAX','Fontname','TimesNewRoman','Fontsize',ii)
ylabel('BIC','Fontname','TimesNewRoman','Fontsize',ii)
set(gca,'fontsize',i)
subplot(2,1,2),plot(minar:maxar,rss_armax(minar:maxar),'-o')
xlim([minar maxar])
title('RSS/SSS criterion for ARMAX','Fontname','TimesNewRoman','Fontsize',ii)
ylabel('RSS/SSS (%)','Fontname','TimesNewRoman','Fontsize',ii)
xlabel('ARMAX(n,n,n)','Fontname','TimesNewRoman','Fontsize',ii)
set(gca,'fontsize',i)
pause 
%%
%--------------------------------------------------------------------------
%----------------- ARX/ARMA frequency stabilization plot ------------------
%--------------------------------------------------------------------------
%clear;
clc;
%load subspace_50.mat
    [D_armax,fn_armax,z_armax] = deal(zeros(maxar,round(maxar/2+1)));

%ARMAX stab. plot
    for order=minar:maxar
        clear num den
%         %subspace
%             statesp = ss(models_armax{order}.A,[models_armax{order}.B models_armax{order}.K], models_armax{order}.C, [models_armax{order}.D 1], 1/Fs);
%             [num_armax, den_armax] = ss2tf(statesp.A, statesp.B, statesp.C, statesp.D, 1);
%             
%         %
        num_armax = models_armax{order}.B;
        den_armax = models_armax{order}.A;
        [DELTA_armax,Wn_armax,ZETA_armax,R_armax,lambda_armax]=disper_new(num_armax,den_armax,Fs);
        qq_armax = length(DELTA_armax);
        D_armax(order,1:qq_armax) = DELTA_armax';
        fn_armax(order,1:qq_armax) = Wn_armax';
        z_armax(order,1:qq_armax) = ZETA_armax';
    end

i=10; ii=14;

figure, hold on
for order=minar:maxar
    for jj=1:qq_armax
        imagesc(3,3,[z_armax(order,jj)])
        disp(jj);
    end
end
axis([0,5*fs/2,minar,maxar])
for order=minar:maxar
    for jj=1:maxar/2
        imagesc([5*fn_armax(order,jj)],[order],[z_armax(order,jj)])
    end
end

colorbar,box on,
h = get(gca,'xtick');
set(gca,'xticklabel',h/5,'fontsize',i);
title('Frequency stabilization plot (colormap indicates damping ratio)','Fontname','TimesNewRoman','Fontsize',ii)
ylabel('ARMAX(n,n,n)','Fontname','TimesNewRoman','Fontsize',ii)
xlabel('Frequency (Hz)','Fontname','TimesNewRoman','Fontsize',ii)


pause % ,close


%----------------order selection-----------------------

%arx_select_order=input('select order for arx model \n');
armax_select_order=input('select order for armax\n');

% disp('Natural Frequencies (Hz)for ARX');
% disp(nonzeros(fn_arx(arx_select_order,:)))
%  
% disp('Damping Ratios for ARX(%)');
% disp(nonzeros(z_arx(arx_select_order,:)))

disp('Natural Frequencies (Hz)for ARMAX');
disp(nonzeros(fn_armax(armax_select_order,:)))
 
disp('Damping Ratios for ARMAX(%)');
disp(nonzeros(z_armax(armax_select_order,:)))
pause
%--------------------------------------------------------------------------
%----------------------------- Residual ACF -------------------------------
%--------------------------------------------------------------------------
%ARX
% res_arx=DATA1.outputdata-Yp_arx{arx_select_order}.outputdata;
% 
% figure
% acf_wn(res_arx(arx_select_order+1:end),100,0.8);
% title('Residuals ACF for ARX','Fontname','TimesNewRoman','fontsize',12)
% ylim([-0.5 0.5])
%ARMAX
res_armax=DATA1.outputdata-Yp_armax{armax_select_order}.outputdata;

figure
acf_wn(res_armax(armax_select_order+1:end),100,0.8);
%title('Residuals ACF for ARMAX','Fontname','TimesNewRoman','fontsize',12)
feval('title',sprintf('Residuals ACF for ARMAX(%d,%d,%d)',armax_select_order,armax_select_order, armax_select_order),...
        'Fontname','TimesNewRoman','fontsize',ii)
ylim([-0.5 0.5])

% [acf_res,lags,bounds_acf]=autocorr(res(order+1:end),100); 
% figure
% bar(acf_res,0.8)
% line([0 size(acf_res,1)],[bounds_acf(1) bounds_acf(1)],'color','r')
% line([0 size(acf_res,1)],[bounds_acf(2) bounds_acf(2)],'color','r')
% axis([0 100 -0.5 0.5])
% set(gca,'fontsize',i,'Fontname','TimesNewRoman')
% title('Model residuals AutoCorrelation Function (ACF)','fontsize',ii,'Fontname','TimesNewRoman')
% ylabel('Excitation ACF','fontsize',ii,'Fontname','TimesNewRoman','interpreter','tex')
% xlabel('Lag','fontsize',ii,'Fontname','TimesNewRoman','interpreter','tex')
    
pause


%-------------------------------- ARX FRF ---------------------------------
%--------------------------------------------------------------------------
disp(' ARX(,) FREQUENCY RESPONSE FUNCTION ')
%--------------------------------------------------------------------------
%order = armax_select_order; % select ARX model orders                                       <----------
%--------------------------------------------------------------------------
[MAG_armax,PHASE_armax,wp] = dbode(models_armax{armax_select_order}.B,models_armax{armax_select_order}.A,1/Fs,2*pi*[0:0.1:80]);
figure
plot(wp/(2*pi),20*log10(abs(MAG_armax)))
%plot(w,20*log10(abs(Txy)),'r')
axis([5 80 -80 60])
set(gca,'fontsize',16)
feval('title',sprintf('ARMAX(%d,%d,%d) FRF',armax_select_order,armax_select_order, armax_select_order),...
        'Fontname','TimesNewRoman','fontsize',ii)
ylabel('Magnitude (dB)','Fontname','TimesNewRoman','Fontsize',20)
xlabel('Frequency (Hz)','Fontname','TimesNewRoman','Fontsize',20)
disp('(press a key)')
pause % ,close
disp('   ')

%---------------- Parametric vs non-parametric FRF comparison -------------
disp(' NON-PARAMETRIC AND PARAMETRIC FRFs ')
%--------------------------------------------------------------------------
%order = armax_select_order % select ARX orders                                             <----------
WINDOW = 1024; NFFT = 1024; OVERLAP = 0.8; % Welch based FRF parameters     <----------
%--------------------------------------------------------------------------
[MAG_armax,PHASE_armax,wp] = dbode(models_armax{armax_select_order}.B,models_armax{armax_select_order}.A,1/Fs,2*pi*[0:0.1:80]);
[Txy,w] = tfestimate(X1,Y1,WINDOW,round(OVERLAP*1024),NFFT,Fs);
figure
plot(wp/(2*pi),20*log10(abs(MAG_armax))),hold on
plot(w,20*log10(abs(Txy)),'r')
axis([5 80 -80 60])
set(gca,'fontsize',16)
feval('title',sprintf('ARMAX(%d,%d,%d) vs Welch based FRF',armax_select_order,armax_select_order, armax_select_order),...
        'Fontname','TimesNewRoman','fontsize',ii)
ylabel('Magnitude (dB)','Fontname','TimesNewRoman','Fontsize',20)
xlabel('Frequency (Hz)','Fontname','TimesNewRoman','Fontsize',20)
legend('ARMAX based','Welch based','Location','SouthEast','Orientation','vertical')
disp('(press a key)')
pause % ,close
disp('   ')
disp('   ')

% reselect order for armax and check residuals

while 1

answer=input('do you want to select again order?\nyes-->1   no-->0\n');

if answer~=1
    break;
else
    
armax_select_order=input('select order for armax\n');

disp('Natural Frequencies (Hz)for ARMAX');
disp(nonzeros(fn_armax(armax_select_order,:)))
 
disp('Damping Ratios for ARMAX(%)');
disp(nonzeros(z_armax(armax_select_order,:)))
pause

res_armax=DATA1.outputdata-Yp_armax{armax_select_order}.outputdata;

figure
acf_wn(res_armax(armax_select_order+1:end),100,0.8);
%title('Residuals ACF for ARMAX','Fontname','TimesNewRoman','fontsize',12)
feval('title',sprintf('Residuals ACF for ARMAX(%d,%d,%d)',armax_select_order,armax_select_order, armax_select_order),...
        'Fontname','TimesNewRoman','fontsize',ii)
ylim([-0.5 0.5])

%order = armax_select_order; % select ARX model orders                                       <----------
%--------------------------------------------------------------------------
[MAG_armax,PHASE_armax,wp] = dbode(models_armax{armax_select_order}.B,models_armax{armax_select_order}.A,1/Fs,2*pi*[0:0.1:80]);
figure
plot(wp/(2*pi),20*log10(abs(MAG_armax)))
%plot(w,20*log10(abs(Txy)),'r')
axis([5 80 -80 60])
set(gca,'fontsize',16)
feval('title',sprintf('ARMAX(%d,%d,%d) FRF',armax_select_order,armax_select_order, armax_select_order),...
        'Fontname','TimesNewRoman','fontsize',ii)
ylabel('Magnitude (dB)','Fontname','TimesNewRoman','Fontsize',20)
xlabel('Frequency (Hz)','Fontname','TimesNewRoman','Fontsize',20)
disp('(press a key)')
pause % ,close
disp('   ')

%---------------- Parametric vs non-parametric FRF comparison -------------
disp(' NON-PARAMETRIC AND PARAMETRIC FRFs ')
%--------------------------------------------------------------------------
%order = armax_select_order; % select ARMAX orders                                             <----------
WINDOW = 1024; NFFT = 1024; OVERLAP = 0.8; % Welch based FRF parameters     <----------
%--------------------------------------------------------------------------
[MAG_armax,PHASE_armax,wp] = dbode(models_armax{armax_select_order}.B,models_armax{armax_select_order}.A,1/Fs,2*pi*[0:0.1:80]);
[MAG_20,PHASE_20,wp_20] = dbode(models_armax{20}.B,models_armax{20}.A,1/Fs,2*pi*[0:0.1:80]);
[MAG_33,PHASE_33,wp_33] = dbode(models_armax{33}.B,models_armax{33}.A,1/Fs,2*pi*[0:0.1:80]);
[MAG_40,PHASE_40,wp_40] = dbode(models_armax{40}.B,models_armax{40}.A,1/Fs,2*pi*[0:0.1:80]);

figure
plot(wp_20/(2*pi),20*log10(abs(MAG_20)),'g'),hold on
plot(wp/(2*pi),20*log10(abs(MAG_armax)),'b'),hold on
plot(wp_33/(2*pi),20*log10(abs(MAG_33)),'k'),hold on
plot(wp_40/(2*pi),20*log10(abs(MAG_40)),'r'),hold on
% plot(w,20*log10(abs(Txy)),'r')
axis([5 80 -80 60])
set(gca,'fontsize',16)
title('ARMAX FRF COMPARISON','Fontname','TimesNewRoman','Fontsize',20)
ylabel('Magnitude (dB)','Fontname','TimesNewRoman','Fontsize',20)
xlabel('Frequency (Hz)','Fontname','TimesNewRoman','Fontsize',20)
legend('ARMAX(20,20,20)','ARMAX(25,25,25)','ARMAX(33,33,33)','ARMAX(40,40,40)','Location','SouthEast','Orientation','vertical')
disp('(press a key)')
pause % ,close
disp('   ')
disp('   ')
pause
end
end
pause
final_order=input('select final order');

[MAG_armax,PHASE_armax,wp] = dbode(models_armax{final_order}.B,models_armax{final_order}.A,1/Fs,2*pi*[0:0.1:80]);
[Txy,w] = tfestimate(X1,Y1,WINDOW,round(OVERLAP*1024),NFFT,Fs);

figure
plot(wp/(2*pi),20*log10(abs(MAG_armax)),'b'),hold on
plot(w,20*log10(abs(Txy)),'r')
axis([5 80 -80 60])
set(gca,'fontsize',16)
feval('title',sprintf('ARMAX(%d,%d,%d) vs Welch based FRF',final_order,final_order, final_order),...
        'Fontname','TimesNewRoman','fontsize',ii)
ylabel('Magnitude (dB)','Fontname','TimesNewRoman','Fontsize',20)
xlabel('Frequency (Hz)','Fontname','TimesNewRoman','Fontsize',20)
legend('ARMAX based','Welch based','Location','SouthEast','Orientation','vertical')
disp('(press a key)')
pause % ,close
disp('   ')
disp('   ')
pause

%--------------------------------------------------------------------------
%                              Poles - Zeros
%--------------------------------------------------------------------------

figure
pzmap(models_armax{final_order})

pause
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% DAMAGE DETECTION BASED ON NON-PARAMETRIC METHODS%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('----- NON-PARAMETRIC STATISTICAL TIME SERIES METHODS FOR SHM -----')
load ../data/course_fdi_data.mat
load ../data/DATA_1047304.mat
load ../data/before_order_selection.mat
%load ../data/SHM_giannis.mat

i=16; ii=20; % font sizes
%--------------------------------------------------------------------------
%                       Damage detection based on the PSD Method
%--------------------------------------------------------------------------
disp('   ')
%--------------------------------------------------------------------------
WINDOW = 256; NFFT = 256; alpha = 0.01; %                                   <----------
%--------------------------------------------------------------------------
format short g
disp('------------------------------------------------------------');
disp('       PSD BASED METHOD      ');
disp('------------------------------------------------------------');
disp('          N          Window          K           á'); 
disp([ size(Y1,1) WINDOW round(size(Y1,1)/WINDOW) alpha])
disp('------------------------------------------------------------');
disp('   ')
disp(' Damage is detected if the test statistic (blue line) is not between the critical limits (red dashed lines) ')
disp(' ')
disp('(press a key)')
pause

%%%%%% Damage detection results 

% Test case I : Healthy
disp(' Test case I : Healthy Structure ')
fdi_spectral_density(Y1,output(:,1),WINDOW,NFFT,Fs,alpha); 
xlim([5 128])
set(gca,'fontsize',i)
title(' Test Case I: Healthy Structure','fontsize',ii)
ylabel('F statistic','fontsize',ii)
xlabel(' Frequency (Hz)','fontsize',ii)
disp('(press a key)')
pause
disp('   ')

disp(' Test case II : Fault of 8.132 gr ')
% Test case II : Fault 8.132 gr
fdi_spectral_density(Y1,output(:,2),WINDOW,NFFT,Fs,alpha);
xlim([5 80])
set(gca,'fontsize',i)
title(' Test Case II: 8.132 gr','fontsize',ii)
ylabel('F statistic','fontsize',ii)
xlabel(' Frequency (Hz)','fontsize',ii)
disp('(press a key)')
pause
disp('   ')

disp(' Test case III : Fault of 24.396 gr ')
% Test case III : Faulty 24.396 gr
fdi_spectral_density(Y1,output(:,3),WINDOW,NFFT,Fs,alpha);
xlim([5 80])
set(gca,'fontsize',i)
title(' Test Case III: 24.396 gr','fontsize',ii)
ylabel('F statistic','fontsize',ii)
xlabel(' Frequency (Hz)','fontsize',ii)
disp('(press a key)')
pause
disp('   ')

disp(' Test case IV : Fault of 40.66 gr ')
% Test case IV : Faulty 40.66 gr
fdi_spectral_density(Y1,output(:,4),WINDOW,NFFT,Fs,alpha);
xlim([5 80])
set(gca,'fontsize',i)
title(' Test Case IV: 40.66 gr','fontsize',ii)
ylabel('F statistic','fontsize',ii)
xlabel(' Frequency (Hz)','fontsize',ii)
disp('(press a key)')
pause
disp('   ')

% disp(' Test case V : Fault of 56.924 gr ')
% % Test case V : Faulty 56.924 gr
% fdi_spectral_density(y_healthy,y_faulty_V,WINDOW,NFFT,Fs,alpha);
% xlim([5 80])
% set(gca,'fontsize',i)
% title(' Test Case V: 65.056 gr','fontsize',ii)
% ylabel('F statistic','fontsize',ii)
% xlabel(' Frequency (Hz)','fontsize',ii)
% disp('(press a key)')
% pause
% disp('   ')
% 
% disp(' Test case VI : Fault of 81.32 gr ')
% % Test case VI : Faulty 81.32 gr
% fdi_spectral_density(y_healthy,y_faulty_VI,WINDOW,NFFT,Fs,alpha);
% xlim([5 80])
% set(gca,'fontsize',i)
% title(' Test Case VI: 81.32 gr','fontsize',ii)
% ylabel('F statistic','fontsize',ii)
% xlabel(' Frequency (Hz)','fontsize',ii)
% disp('(press a key)')
% pause
clc
%--------------------------------------------------------------------------
%                       Damage detection based on the FRF Method
%--------------------------------------------------------------------------
%Method parameters
WINDOW = 512; NFFT = 512; alpha = 0.0001; %                                   <----------
%--------------------------------------------------------------------------
disp('--------------------------------------------------------------------');
disp('       FRF BASED METHOD      ');
disp('--------------------------------------------------------------------');
disp('          N          Window          K           á'); 
disp([ size(y_healthy,1) WINDOW round(size(y_healthy,1)/WINDOW) alpha])
disp('------------------------------------------------------------');
disp('   ')
disp(' Damage is detected if the test statistic (blue line) is over the critical limit (red line) ')
disp(' ')
disp('(press a key)')
pause
disp('   ')

%%%%%% Damage detection results 
disp(' Test case I: Healthy Structure ')
% Test case I : Healthy
fdi_frf(x_healthy,y_healthy,x_healthy_I,y_healthy_I,WINDOW,NFFT,Fs,alpha);
xlim([5 80])
set(gca,'fontsize',i)
title(' Test Case I: Healthy Structure','fontsize',ii)
ylabel('Z statistic','fontsize',ii)
xlabel(' Frequency (Hz)','fontsize',ii)
disp('(press a key)')
pause
disp('   ')

disp(' Test case II : Fault of 8.132 gr ')
% Test case II : Faulty 8.132 gr
fdi_frf(x_healthy,y_healthy,x_faulty_II,y_faulty_II,WINDOW,NFFT,Fs,alpha);
xlim([5 80])
set(gca,'fontsize',i)
title(' Test Case II: 8.132 gr','fontsize',ii)
ylabel('Z statistic','fontsize',ii)
xlabel(' Frequency (Hz)','fontsize',ii)
disp('(press a key)')
pause
disp('   ')

disp(' Test case III: Fault of 24.396 gr ')
% Test case III : Faulty 24.396 gr
fdi_frf(x_healthy,y_healthy,x_faulty_III,y_faulty_III,WINDOW,NFFT,Fs,alpha);
xlim([5 80])
set(gca,'fontsize',i)
title(' Test Case III: 24.396 gr','fontsize',ii)
ylabel('Z statistic','fontsize',ii)
xlabel(' Frequency (Hz)','fontsize',ii)
disp('(press a key)')
pause
disp('   ')

disp(' Test case IV: Fault of 40.66 gr ')
% Test case IV : Faulty 40.66 gr
fdi_frf(x_healthy,y_healthy,x_faulty_IV,y_faulty_IV,WINDOW,NFFT,Fs,alpha);
xlim([5 80])
set(gca,'fontsize',i)
title(' Test Case IV: 40.66 gr','fontsize',ii)
ylabel('Z statistic','fontsize',ii)
xlabel(' Frequency (Hz)','fontsize',ii)
disp('(press a key)')
pause
disp('   ')

disp(' Test case V: Fault of 56.924 gr ')
% Test case V : Faulty 56.924 gr
fdi_frf(x_healthy,y_healthy,x_faulty_V,y_faulty_V,WINDOW,NFFT,Fs,alpha);
xlim([5 80])
set(gca,'fontsize',i)
title(' Test Case V: 65.056 gr','fontsize',ii)
ylabel('Z statistic','fontsize',ii)
xlabel(' Frequency (Hz)','fontsize',ii)
disp('(press a key)')
pause
disp('   ')

disp(' Test case VI: Fault of 81.32 gr ')
% Test case VI : Faulty 81.32 gr
fdi_frf(x_healthy,y_healthy,x_faulty_VI,y_faulty_VI,WINDOW,NFFT,Fs,alpha);
xlim([5 80])
set(gca,'fontsize',i)
title(' Test Case VI: 81.32 gr','fontsize',ii)
ylabel('Z statistic','fontsize',ii)
xlabel(' Frequency (Hz)','fontsize',ii)
disp('(press a key)')
pause
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% DAMAGE DETECTION BASED ON PARAMETRIC METHODS %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
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
disp('          á                Model: ARX(62,62) '); 
disp([alpha])
disp('------------------------------------------------------------');

% [Q_mod_par(1),limit_mod_par]=fdi_model_parameter(arx_healthy,arx_healthy_I,alpha); 
% [Q_mod_par(2),limit_mod_par]=fdi_model_parameter(arx_healthy,arx_faulty_II,alpha); 
% [Q_mod_par(3),limit_mod_par]=fdi_model_parameter(arx_healthy,arx_faulty_III,alpha);
% [Q_mod_par(4),limit_mod_par]=fdi_model_parameter(arx_healthy,arx_faulty_IV,alpha); 
% [Q_mod_par(5),limit_mod_par]=fdi_model_parameter(arx_healthy,arx_faulty_V,alpha); 
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
set(gca,'fontsize',i, 'xticklabel',...
    ['I: Healthy   ';'II: 8.132 gr ';'III: 24.39 gr';'IV: 40.66 gr ';'V: 56.92 gr  ';'VI: 81.32 gr '])
title(' Model Parameter Based Method','fontsize',ii)
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

% Method A: Using the residual variance
%risk level
alpha = 0.01;
disp('-----------------------------------------------------------------------------');
disp('     Method A: using the residual variance     ');
disp('-----------------------------------------------------------------------------');
disp('          á                Model: ARX(62,62) '); 
disp([alpha])
disp('------------------------------------------------------------');
disp(' ')
disp(' Damage is detected if the test statistic (blue bar) is over the critical limit (red dashed line) ')

% [Q_res_a(1),limit_res_a]=fdi_model_residual(arx_healthy,DATA_healthy,DATA_healthy_I,'var',alpha);
% [Q_res_a(2),limit_res_a]=fdi_model_residual(arx_healthy,DATA_healthy,DATA_faulty_II,'var',alpha);
% [Q_res_a(3),limit_res_a]=fdi_model_residual(arx_healthy,DATA_healthy,DATA_faulty_III,'var',alpha);
% [Q_res_a(4),limit_res_a]=fdi_model_residual(arx_healthy,DATA_healthy,DATA_faulty_IV,'var',alpha);
% [Q_res_a(5),limit_res_a]=fdi_model_residual(arx_healthy,DATA_healthy,DATA_faulty_V,'var',alpha);
% [Q_res_a(6),limit_res_a]=fdi_model_residual(arx_healthy,DATA_healthy,DATA_faulty_VI,'var',alpha);
%NOTE: The necessary statistic and the corresponding limits for each case are computed based on the
%functions above and are included in the Q_res_a & limit_res_a files, respectively.

figure
bar(Q_res_a,0.5)
hold on
line([0 7],[limit_res_a limit_res_a],'color','r','linestyle','--','linewidth',1.5)
hold off
set(gca,'fontsize',i, 'xticklabel',...
    ['I: Healthy   ';'II: 8.132 gr ';'III: 24.39 gr';'IV: 40.66 gr ';'V: 56.92 gr  ';'VI: 81.32 gr '])
title(' Redidual Based Method A: using the residual variance','fontsize',ii)
ylabel('Test Statistic','fontsize',ii)
xlabel(' Test Cases','fontsize',ii)
disp('(press a key)')
pause

%--------------------------------------------------------------------------
% Method B: Using the likelihood function

%risk level
alpha = 0.01;

disp('-------------------------------------------------------------------------------');
disp('       Method B: using the likelihood function   ');
disp('-------------------------------------------------------------------------------');
disp('          á                Model: ARX(62,62) '); 
disp([alpha])
disp('------------------------------------------------------------');
disp(' ')
disp(' Damage is detected if the test statistic (blue bar) is over the critical limit (red dashed line) ')

% [Q_res_b(1),limit_res_b]=fdi_model_residual(arx_healthy,DATA_healthy,DATA_healthy_I,'lik',alpha);
% [Q_res_b(2),limit_res_b]=fdi_model_residual(arx_healthy,DATA_healthy,DATA_faulty_II,'lik',alpha);
% [Q_res_b(3),limit_res_b]=fdi_model_residual(arx_healthy,DATA_healthy,DATA_faulty_III,'lik',alpha);
% [Q_res_b(4),limit_res_b]=fdi_model_residual(arx_healthy,DATA_healthy,DATA_faulty_IV,'lik',alpha);
% [Q_res_b(5),limit_res_b]=fdi_model_residual(arx_healthy,DATA_healthy,DATA_faulty_V,'lik',alpha);
% [Q_res_b(6),limit_res_b]=fdi_model_residual(arx_healthy,DATA_healthy,DATA_faulty_VI,'lik',alpha);
%NOTE: The necessary statistic and the corresponding limits for each test case are computed based on the
%functions above and are included int he Q_res_b & limit_res_b files, respectively.

figure
bar(Q_res_b,0.5)
hold on
line([0 7],[limit_res_b limit_res_b],'color','r','linestyle','--','linewidth',1.5)
hold off
set(gca,'fontsize',i, 'xticklabel',...
    ['I: Healthy   ';'II: 8.132 gr ';'III: 24.39 gr';'IV: 40.66 gr ';'V: 56.92 gr  ';'VI: 81.32 gr '])
title(' Redidual Based Method B: using the likelihood function','fontsize',ii)
ylabel('Test Statistic','fontsize',ii)
xlabel(' Test Cases','fontsize',ii)
disp('(press a key)')
pause

%--------------------------------------------------------------------------
% Method C: Using the residual uncorrelatedness

%risk level and mamixum lag
alpha = 0.01;
max_lag = 25;

disp('-------------------------------------------------------------------------------------');
disp('       Method C: using the residual uncorrelatedness   ');
disp('-------------------------------------------------------------------------------------');
disp('          á          max_lag      Model: ARX(62,62) '); 
disp([alpha max_lag])
disp('-------------------------------------------------------------------------------------');
disp(' ')
disp(' Damage is detected if the test statistic (blue bar) is over the critical limit (red dashed line) ')

% [Q_res_c(1),limit_res_c]=fdi_model_residual(arx_healthy,DATA_healthy,DATA_healthy_I,'unc',...
%     alpha,max_lag);
% [Q_res_c(2),limit_res_c]=fdi_model_residual(arx_healthy,DATA_healthy,DATA_faulty_II,'unc',...
%     alpha,max_lag);
% [Q_res_c(3),limit_res_c]=fdi_model_residual(arx_healthy,DATA_healthy,DATA_faulty_III,'unc',...
%     alpha,max_lag);
% [Q_res_c(4),limit_res_c]=fdi_model_residual(arx_healthy,DATA_healthy,DATA_faulty_IV,'unc',...
%     alpha,max_lag);
% [Q_res_c(5),limit_res_c]=fdi_model_residual(arx_healthy,DATA_healthy,DATA_faulty_V,'unc',...
%     alpha,max_lag);
% [Q_res_c(6),limit_res_c]=fdi_model_residual(arx_healthy,DATA_healthy,DATA_faulty_VI,'unc',...
%     alpha,max_lag);
%The necessary statistic and the corresponding limits for each test case are computed based on the
%functions above and are included int he Q_res_c & limit_res_c files, respectively.

figure
bar(Q_res_c,0.5)
hold on
line([0 7],[limit_res_c limit_res_c],'color','r','linestyle','--','linewidth',1.5)
hold off
set(gca,'fontsize',i, 'xticklabel',...
    ['I: Healthy   ';'II: 8.132 gr ';'III: 24.39 gr';'IV: 40.66 gr ';'V: 56.92 gr  ';'VI: 81.32 gr '])
title(' Redidual Based Method C: using the residual uncorrelatedness','fontsize',ii)
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

function fdi_frf(x_healthy,y_healthy,x_faulty,y_faulty,window,nfft,fs,alpha)

[tyh,w] = tfestimate(x_healthy,y_healthy,window,0,nfft,fs);
[tyf,w] = tfestimate(x_faulty,y_faulty,window,0,nfft,fs);  
[cyh,w] = mscohere(x_healthy,y_healthy,window,0,nfft,fs);  
K = length(y_healthy)/window; 
Tyh = abs(tyh);,Tyf = abs(tyf);,var_frf = [(1-cyh)./(2*K*cyh)].*Tyh.^2;
dT = Tyh - Tyf;,p = [alpha/2 1-alpha/2];,n = norminv(p);,z = n(2); % uper bound
figure,plot(w,abs(dT)./sqrt(2*var_frf)),grid on,hold on,,plot(w,z*ones(size(w)),'r')
hold off

function [Q,limit] = fdi_model_parameter(model_o,model_u,alpha)

theta_o = model_o.parametervector;,theta_u = model_u.parametervector;
P_o = model_o.covarianvematrix;,dtheta = theta_o - theta_u;,dP = 2*P_o;
Q = dtheta.'*inv(dP)*dtheta;,p = 1-alpha;,limit = chi2inv(p,size(theta_o,1));
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


