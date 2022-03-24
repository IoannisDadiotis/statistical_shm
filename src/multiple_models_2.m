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

%------------------------------------------------------------------------------
%MM models
DATA1=iddata(baseline(:,1),[],1/Fs);
DATA2=iddata(baseline(:,2),[],1/Fs);
DATA3=iddata(baseline(:,3),[],1/Fs);
DATA4=iddata(baseline(:,4),[],1/Fs);
DATA5=iddata(baseline(:,5),[],1/Fs);
DATA6=iddata(baseline(:,11),[],1/Fs);
DATA7=iddata(baseline(:,12),[],1/Fs);
DATA8=iddata(baseline(:,13),[],1/Fs);
DATA9=iddata(baseline(:,14),[],1/Fs);
DATA10=iddata(baseline(:,15),[],1/Fs);
DATA11=iddata(baseline(:,21),[],1/Fs);
DATA12=iddata(baseline(:,22),[],1/Fs);
DATA13=iddata(baseline(:,23),[],1/Fs);
DATA14=iddata(baseline(:,24),[],1/Fs);
DATA15=iddata(baseline(:,25),[],1/Fs);

order = input('Select MM order: \n');

MM1=ar(DATA1,order);
MM2=ar(DATA2,order);
MM3=ar(DATA3,order);
MM4=ar(DATA4,order);
MM5=ar(DATA5,order);
MM6=ar(DATA6,order);
MM7=ar(DATA7,order);
MM8=ar(DATA8,order);
MM9=ar(DATA9,order);
MM10=ar(DATA10,order);
MM11=ar(DATA11,order);
MM12=ar(DATA12,order);
MM13=ar(DATA13,order);
MM14=ar(DATA14,order);
MM15=ar(DATA15,order);

%Threshold models
DATAth1=iddata(baseline(:,6),[],1/Fs);
DATAth2=iddata(baseline(:,7),[],1/Fs);
DATAth3=iddata(baseline(:,8),[],1/Fs);
DATAth4=iddata(baseline(:,9),[],1/Fs);
DATAth5=iddata(baseline(:,10),[],1/Fs);
DATAth6=iddata(baseline(:,16),[],1/Fs);
DATAth7=iddata(baseline(:,17),[],1/Fs);
DATAth8=iddata(baseline(:,18),[],1/Fs);
DATAth9=iddata(baseline(:,19),[],1/Fs);
DATAth10=iddata(baseline(:,20),[],1/Fs);
DATAth11=iddata(baseline(:,26),[],1/Fs);
DATAth12=iddata(baseline(:,27),[],1/Fs);
DATAth13=iddata(baseline(:,28),[],1/Fs);
DATAth14=iddata(baseline(:,29),[],1/Fs);
DATAth15=iddata(baseline(:,30),[],1/Fs);

TH1=ar(DATAth1,order);
TH2=ar(DATAth2,order);
TH3=ar(DATAth3,order);
TH4=ar(DATAth4,order);
TH5=ar(DATAth5,order);
TH6=ar(DATAth6,order);
TH7=ar(DATAth7,order);
TH8=ar(DATAth8,order);
TH9=ar(DATAth9,order);
TH10=ar(DATAth10,order);
TH11=ar(DATAth11,order);
TH12=ar(DATAth12,order);
TH13=ar(DATAth13,order);
TH14=ar(DATAth14,order);
TH15=ar(DATAth15,order);

%Inspection
DATAin1=iddata(inspection(:,1),[],1/Fs);
DATAin2=iddata(inspection(:,2),[],1/Fs);
DATAin3=iddata(inspection(:,3),[],1/Fs);
DATAin4=iddata(inspection(:,4),[],1/Fs);
DATAin5=iddata(inspection(:,5),[],1/Fs);
DATAin6=iddata(inspection(:,6),[],1/Fs);
DATAin7=iddata(inspection(:,7),[],1/Fs);
DATAin8=iddata(inspection(:,8),[],1/Fs);
DATAin9=iddata(inspection(:,9),[],1/Fs);
DATAin10=iddata(inspection(:,10),[],1/Fs);

IN1=ar(DATAin1,order);
IN2=ar(DATAin2,order);
IN3=ar(DATAin3,order);
IN4=ar(DATAin4,order);
IN5=ar(DATAin5,order);
IN6=ar(DATAin6,order);
IN7=ar(DATAin7,order);
IN8=ar(DATAin8,order);
IN9=ar(DATAin9,order);
IN10=ar(DATAin10,order);

%Parameters of MM
Param = [MM1.parametervector MM2.parametervector MM3.parametervector MM4.parametervector MM5.parametervector MM6.parametervector MM7.parametervector MM8.parametervector MM9.parametervector MM10.parametervector MM11.parametervector MM12.parametervector MM13.parametervector MM14.parametervector MM15.parametervector;];

%Threshold
dist_th(:,1)=distance(Param, TH1, order);
dist_th(:,2)=distance(Param, TH2, order);
dist_th(:,3)=distance(Param, TH3, order);
dist_th(:,4)=distance(Param, TH4, order);
dist_th(:,5)=distance(Param, TH5, order);
dist_th(:,6)=distance(Param, TH6, order);
dist_th(:,7)=distance(Param, TH7, order);
dist_th(:,8)=distance(Param, TH8, order);
dist_th(:,9)=distance(Param, TH9, order);
dist_th(:,10)=distance(Param, TH10, order);
dist_th(:,11)=distance(Param, TH11, order);
dist_th(:,12)=distance(Param, TH12, order);
dist_th(:,13)=distance(Param, TH13, order);
dist_th(:,14)=distance(Param, TH14, order);
dist_th(:,15)=distance(Param, TH15, order);

figure;
for i=1:15
    plot(i,dist_th(:,i),'o','MarkerEdgeColor','blue'),hold on
end
xlim([0 16]);
xlabel('Number of baseline cases','fontsize',16);
ylabel('D (sum of distances)','fontsize',16);
title('MM-based Fault detection - Baseline phase','fontsize',20);

%inspection
dist_IN(:,1)=distance(Param, IN1, order);
dist_IN(:,2)=distance(Param, IN2, order);
dist_IN(:,3)=distance(Param, IN3, order);
dist_IN(:,4)=distance(Param, IN4, order);
dist_IN(:,5)=distance(Param, IN5, order);
dist_IN(:,6)=distance(Param, IN6, order);
dist_IN(:,7)=distance(Param, IN7, order);
dist_IN(:,8)=distance(Param, IN8, order);
dist_IN(:,9)=distance(Param, IN9, order);
dist_IN(:,10)=distance(Param, IN10, order);

figure;
for i=1:10
    plot(i,dist_IN(:,i),'x','MarkerEdgeColor','black'),hold on
end
plot([0 11], [max(dist_th) max(dist_th)], 'r');
xlim([0 11]);
xlabel('Number of Inspection Experiment','fontsize',16);
ylabel('D (sum of distances)','fontsize',16);
title('MM-based Fault Detection for the 10 Inspection cases','fontsize',20);

%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
%AUXILIARY FILES
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

function D=distance(Parameters, model, order)

for count=1:15
    
    for i=1:order
        dd(i)=(Parameters(i,count)-model.parametervector(i))^2;
    end
    d(count)=sqrt(sum(dd));
end
D=sum(d);

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








