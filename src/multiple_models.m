function multiple_models(~)
clear;
clc;
close all;
load PART_III_DATA_3
Fs = 980;
fs=980;
N=length(Baseline_Signals(:,2));
TT=0:1/Fs:(length(Baseline_Signals(:,2))-1)/Fs;

baseline = [];
inspection = [];
for i=1:30
    Baseline_Signals(:,i)=detrend(Baseline_Signals(:,i));
    baseline(:,i)=Baseline_Signals(:,i)./std(Baseline_Signals(:,i));
end

for i=1:10
    Inspection_Unknown_Signals(:,i) = detrend(Inspection_Unknown_Signals(:,i));
    inspection(:,i) =  Inspection_Unknown_Signals(:,i)./std( Inspection_Unknown_Signals(:,i));
end

giannis=input('Which output is to be identified? \n');

%--------------------------------------------------------------------------
%                    Histogram and Normal Probability Plot
%--------------------------------------------------------------------------

i=10; ii=14; 

for output=1:10:21
    figure
    subplot(2,1,1),histfit(baseline(:,output),round(sqrt(N)))
    set(gca,'fontsize',i),box on
    feval('title',sprintf('Output %d',output),'Fontname','TimesNewRoman','fontsize',ii)
    subplot(2,1,2), normplot(baseline(:,output))
    set(gca,'fontsize',i),box on
end

pause

%--------------------------------------------------------------------------
%---------------- Welch based excitation-response spectra -----------------
%--------------------------------------------------------------------------

clc

disp('Welch based excitation-response spectra')
WINDOW=input('Give window length (samples): \n');
OVERLAP=0.8

NFFT=WINDOW;

%----------------------- estimation parameters ----------------------------
%--------------------------------------------------------------------------
if isempty(WINDOW); WINDOW = 1024; NFFT = WINDOW; end                     
%--------------------------------------------------------------------------

i=10; ii=14;

figure
[Pyy,w] = pwelch(baseline(:,giannis),WINDOW,round(OVERLAP*WINDOW),NFFT,fs); % response spectrum
plot(w,20*log10(abs(Pyy)))
set(gca,'fontsize',i),box on
xlim([0 fs/2])
feval('title',sprintf('Welch based response spectrum for output'),'Fontname','TimesNewRoman','fontsize',ii)
ylabel('PSD (dB)','Fontname','TimesNewRoman','Fontsize',ii)
xlabel('Frequency (Hz)','Fontname','TimesNewRoman','Fontsize',ii)


%--------------------------------------------------------------------------
%---------------- Welch based output spectra (window effect) --------------
%--------------------------------------------------------------------------

clear Pyy* w

%--------------------------------------------------------------------------
WINDOW1=512; WINDOW2=1024; WINDOW3=2048; WINDOW4=4096; OVERLAP=0.8;
%--------------------------------------------------------------------------

[Pyy1,w1] = pwelch(baseline(:,giannis),WINDOW1,round(OVERLAP*WINDOW1),WINDOW1,fs);
[Pyy2,w2] = pwelch(baseline(:,giannis),WINDOW2,round(OVERLAP*WINDOW2),WINDOW2,fs);
[Pyy3,w3] = pwelch(baseline(:,giannis),WINDOW3,round(OVERLAP*WINDOW3),WINDOW3,fs);
[Pyy4,w4] = pwelch(baseline(:,giannis),WINDOW4,round(OVERLAP*WINDOW4),WINDOW4,fs);

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
legend('512','1024','2048','4096','Location','SouthEast','Orientation','horizontal')

pause % ,close

%--------------------------------------------------------------------------
%---------------------------- AR identification -----------------------------
%--------------------------------------------------------------------------

disp('-----------------------------------');
disp('         AR Identification        ')
disp('-----------------------------------');
clc;
close all;
output=input('Select output for which to run ARX models:');

DATA=iddata(baseline(:,output),[],1/Fs);

minar=input('Give minimum AutoRegressive (AR) order: \n');
maxar=input('Give maximum AutoRegressive (AR) order: \n');

tic
for order=minar:maxar
%     models{order}=armax(DATA,[order order]);
    models{order}=ar(DATA,order);
    modal{order}=the_modals(models{order}.c,models{order}.a,Fs,1,1);
end
toc

Yp=cell(1,maxar); rss=zeros(1,maxar); BIC=zeros(1,maxar);

for order=minar:maxar
    BIC(order)=log(models{order}.noisevariance)+...
         (size(models{order}.parametervector,1)*log(N))/N;
end

for order=minar:maxar
    Yp{order}=predict(models{order},DATA,1);
    rss(order)=100*(norm(DATA.outputdata-Yp{order}.outputdata)^2)/(norm(DATA.outputdata)^2);
end
%% 
clear;
clc;
close all;
load MM_condC
%--------------------------------------------------------------------------
%-----------------------------  BIC-RSS plot ------------------------------
%--------------------------------------------------------------------------

i=10; ii=14;

figure
subplot(3,1,1),plot(minar:maxar,BIC(minar:maxar),'-o')
xlim([minar maxar])
title('BIC criterion','Fontname','TimesNewRoman','Fontsize',ii)
ylabel('BIC','Fontname','TimesNewRoman','Fontsize',ii)
set(gca,'fontsize',i)
subplot(3,1,2),plot(minar:maxar,rss(minar:maxar),'-o')
xlim([minar maxar])
title('RSS/SSS criterion','Fontname','TimesNewRoman','Fontsize',ii)
ylabel('RSS/SSS (%)','Fontname','TimesNewRoman','Fontsize',ii)
xlabel('AR(n)','Fontname','TimesNewRoman','Fontsize',ii)
set(gca,'fontsize',i)

subplot(3,1,3),plot(minar:maxar,aic(models{minar:maxar}),'-o')
xlim([minar maxar])
title('AIC criterion','Fontname','TimesNewRoman','Fontsize',ii)
ylabel('AIC','Fontname','TimesNewRoman','Fontsize',ii)
set(gca,'fontsize',i)

pause 

%--------------------------------------------------------------------------
%----------------- ARX/ARMA frequency stabilization plot ------------------
%--------------------------------------------------------------------------

    
[D,fn,z] = deal(zeros(maxar,round(maxar/2+1)));

for order=minar:maxar
    qq=size(modal{order},1);
    D(order,1:qq) = modal{order}(:,3).';
    fn(order,1:qq) = modal{order}(:,1).';
    z(order,1:qq) = modal{order}(:,2).';
end

i=10; ii=14;

figure, hold on
for order=minar:maxar
    for jj=1:maxar/2
        imagesc([5*fn(order,jj)],[order],[z(order,jj)])
    end
end
axis([0,5*Fs/2,minar,maxar])
colorbar,box on,
h = get(gca,'xtick');
set(gca,'xticklabel',h/5,'fontsize',i);
title('Frequency stabilization plot (colormap indicates damping ratio)','Fontname','TimesNewRoman','Fontsize',ii)

ylabel('AR(n)','Fontname','TimesNewRoman','Fontsize',ii)

xlabel('Frequency (Hz)','Fontname','TimesNewRoman','Fontsize',ii)
%text(5*Fs/2,45,'Damping Ratio','Fontname','TimesNewRoman','Fontsize',ii,...
%    'Rotation',-90,'VerticalAlignment','Middle','HorizontalAlignment','center')

pause % ,close

order=input('Select final model order: \n'); % select ARX/ARMA model orders <----------

disp('Natural Frequencies (Hz)');
disp(nonzeros(fn(order,:)))
 
disp('Damping Ratios (%)');
disp(nonzeros(z(order,:)))

pause

%--------------------------------------------------------------------------
%----------------------------- ARX/ARMA FRF -------------------------------
%--------------------------------------------------------------------------

[MAG,PHASE,wp] = dbode(models{order}.c,models{order}.a,1/Fs,2*pi*[0:0.01:Fs/2]);

i=10; ii=14;

figure
plot(wp/(2*pi),20*log10(abs(MAG)))
%plot(w,20*log10(abs(Txy)),'r')
xlim([0 Fs/2])
set(gca,'fontsize',i)

feval('title',sprintf('Parametric FRF for selected orders - AR(%d)',order),...
    'Fontname','TimesNewRoman','fontsize',ii)
ylabel('Magnitude (dB)','Fontname','TimesNewRoman','Fontsize',ii)
xlabel('Frequency (Hz)','Fontname','TimesNewRoman','Fontsize',ii)

pause % ,close

%--------------------------------------------------------------------------
%---------------- Parametric vs non-parametric spectrum comparison -------------
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
%order = 30; % select ARX orders                                           <----------
WINDOW = 512; NFFT = WINDOW; OVERLAP = 0.8; % Welch based parameters       <----------
%--------------------------------------------------------------------------

[Pyy,w] = pwelch(baseline(:,output),WINDOW,round(OVERLAP*WINDOW),NFFT,Fs);
[MAG,PHASE,wp] = ffplot(models{order},0:0.01:Fs/2);
MAG=reshape(MAG,[1 size(MAG,3)]);

figure
plot(wp,20*log10(abs(MAG))),hold on
plot(w,20*log10(abs(Pyy)),'r')
xlim([0 Fs/2])
set(gca,'fontsize',i)
title('Parametric (AR based) vs non-parametric (Welch based) spectrum comparison',...
    'Fontname','TimesNewRoman','Fontsize',ii)
ylabel('Magnitude (dB)','Fontname','TimesNewRoman','Fontsize',ii)
xlabel('Frequency (Hz)','Fontname','TimesNewRoman','Fontsize',ii)
legend('Parametric','Welch based','Location','SouthEast','Orientation','vertical')
pause 

%--------------------------------------------------------------------------
%---------------- Parametric vs non-parametric FRF comparison -------------
%--------------------------------------------------------------------------
clear Pyy w MAG PHASE wp
%--------------------------------------------------------------------------
%order = 30; % select ARX orders                                           <----------
WINDOW = 512; NFFT = WINDOW; OVERLAP = 0.8; % Welch based parameters       <----------
%--------------------------------------------------------------------------

[Pyy,w] = pwelch(baseline(:,output),WINDOW,round(OVERLAP*WINDOW),NFFT,Fs);
% [MAG,PHASE,wp] = dbode(models{order}.A,models{order}.C,1/Fs,2*pi*[0:0.01:fs/2]);
[MAG1,PHASE1,wp1] = ffplot(models{(50)},0:0.01:Fs/2);
MAG1=reshape(MAG1,[1 size(MAG1,3)]);
[MAG2,PHASE2,wp2] = ffplot(models{(60)},0:0.01:Fs/2);
MAG2=reshape(MAG2,[1 size(MAG2,3)]);
[MAG3,PHASE3,wp3] = ffplot(models{(70)},0:0.01:Fs/2);
MAG3=reshape(MAG3,[1 size(MAG3,3)]);
[MAG4,PHASE4,wp4] = ffplot(models{(80)},0:0.01:Fs/2);
MAG4=reshape(MAG4,[1 size(MAG4,3)]);
[MAG5,PHASE5,wp5] = ffplot(models{(90)},0:0.01:Fs/2);
MAG5=reshape(MAG5,[1 size(MAG5,3)]);

% MAGN=[]; PHAS=[]; wpp=[];
% for i =1:5
%     [MAGN(:,i),PHAS,wpp(:,i)] = ffplot(models{(40+i*10)},0:0.01:Fs/2);
% end
% for i =1:5
%     MAGN(:,i)=reshape(MAGN(:,i),[1 size(MAGN(:,i),3)]);
% end
i=10; ii=14;

figure
% for iii=1:5
%     plot(wpp(:,iii)/(2*pi),20*log10(abs(MAGN(:,iii)))),hold on
% end
plot(wp1,20*log10(abs(MAG1)),'g'),hold on
plot(wp2,20*log10(abs(MAG2)),'y'),hold on
plot(wp3,20*log10(abs(MAG3)),'b'),hold on
plot(wp4,20*log10(abs(MAG4)),'m'),hold on
plot(wp5,20*log10(abs(MAG5)),'r'),hold on
plot(w,20*log10(abs(Pyy)),'k')
xlim([0 Fs/2])
set(gca,'fontsize',i)
title('Parametric (AR based) vs non-parametric (Welch based) spectrum comparison',...
    'Fontname','TimesNewRoman','Fontsize',ii)
ylabel('Magnitude (dB)','Fontname','TimesNewRoman','Fontsize',ii)
xlabel('Frequency (Hz)','Fontname','TimesNewRoman','Fontsize',ii)
legend('AR(50)','AR(60)','AR(70)','AR(80)','AR(90)','Welch based','Location','SouthEast','Orientation','vertical')
pause

%--------------------------------------------------------------------------
%----------------------------- Residual ACF -------------------------------
%--------------------------------------------------------------------------

res=DATA.outputdata-Yp{order}.outputdata;

figure
acf_wn(res(order+1:end),100,0.8);
title('Residuals ACF','Fontname','TimesNewRoman','fontsize',12)
ylim([-0.5 0.5])
    
pause

%--------------------------------------------------------------------------
%                    Histogram and Normal Probability Plot
%--------------------------------------------------------------------------

N=length(res);

i=10; ii=14; 

figure
subplot(2,1,1),histfit(res,round(sqrt(N)))
set(gca,'fontsize',i),box on
title('Residuals','Fontname','TimesNewRoman','fontsize',ii)
subplot(2,1,2), normplot(res)
set(gca,'fontsize',i),box on

pause


%--------------------------------------------------------------------------
%                              Poles - Zeros
%--------------------------------------------------------------------------

figure
pzmap(models{order})

pause

%--------------------------------------------------------------------------
%                           Signal - predictions
%--------------------------------------------------------------------------

figure
plot(TT,DATA.outputdata,'-o'), hold on
plot(TT,Yp{order}.outputdata,'*')
title('Model one-step-ahead prediction (*) vs actual signal (o)','Fontname','TimesNewRoman','fontsize',12)
ylabel('Signal','Fontname','TimesNewRoman','Fontsize',ii)
xlabel('Time (s)','Fontname','TimesNewRoman','Fontsize',ii)
%xlim([11 11.3])

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
