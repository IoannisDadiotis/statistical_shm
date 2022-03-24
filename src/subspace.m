function subspace (~)

clear;
clc;
close all;

%load subspace_10_50
 %load subspace_51_60
 %load subspace_61_65
%load subspace_66_70
%load subspace_71_75
load subspace_76_80
% load subspace_85_86
% load subspace_90_91

Yp_armax=cell(minar,maxar); rss_armax=zeros(minar,maxar); BIC_armax=zeros(minar,maxar);

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

figure
subplot(2,1,1),plot(minar:maxar,BIC_armax(minar:maxar),'-o')
xlim([minar maxar])
title('BIC criterion for Subspace Identification','Fontname','TimesNewRoman','Fontsize',ii)
ylabel('BIC','Fontname','TimesNewRoman','Fontsize',ii)
set(gca,'fontsize',i)
subplot(2,1,2),plot(minar:maxar,rss_armax(minar:maxar),'-o')
xlim([minar maxar])
title('RSS/SSS criterion for Subspace Identification','Fontname','TimesNewRoman','Fontsize',ii)
ylabel('RSS/SSS (%)','Fontname','TimesNewRoman','Fontsize',ii)
xlabel('State space(n)','Fontname','TimesNewRoman','Fontsize',ii)
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
%         statesp = ss(models_armax{order}.A,[models_armax{order}.B models_armax{order}.K], models_armax{order}.C, [models_armax{order}.D 1], 1/Fs);
%         [num_armax, den_armax] = ss2tf(statesp.A, statesp.B, statesp.C, statesp.D, 1);
%             
%         %
        [num_armax, den_armax]=subspace_poly(models_armax{order});
        
%         num_armax = models_armax{order}.B;
%         den_armax = models_armax{order}.A;
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
ylabel('State space(n)','Fontname','TimesNewRoman','Fontsize',ii)
xlabel('Frequency (Hz)','Fontname','TimesNewRoman','Fontsize',ii)


pause % ,close


%----------------order selection-----------------------

%arx_select_order=input('select order for arx model \n');
armax_select_order=input('select order for armax\n');

%define num, den f selected model
[num_armax, den_armax] = subspace_poly(models_armax{armax_select_order});
            
disp('Natural Frequencies (Hz)for ARMAX');
disp(nonzeros(fn_armax(armax_select_order,:)))
 
disp('Damping Ratios for ARMAX(%)');
disp(nonzeros(z_armax(armax_select_order,:)))
pause
%--------------------------------------------------------------------------
%----------------------------- Residual ACF -------------------------------
%--------------------------------------------------------------------------

%ARMAX
res_armax=DATA1.outputdata-Yp_armax{armax_select_order}.outputdata;

figure
acf_wn(res_armax(armax_select_order+1:end),100,0.8);
%title('Residuals ACF for ARMAX','Fontname','TimesNewRoman','fontsize',12)
feval('title',sprintf('Residuals ACF for SS(%d)',armax_select_order),...
        'Fontname','TimesNewRoman','fontsize',ii)
ylim([-0.5 0.5])

pause


%-------------------------------- ARX FRF ---------------------------------
%--------------------------------------------------------------------------
disp(' ARX(,) FREQUENCY RESPONSE FUNCTION ')
%--------------------------------------------------------------------------
%order = armax_select_order; % select ARX model orders                                       <----------
%--------------------------------------------------------------------------
[MAG_armax,PHASE_armax,wp] = dbode(num_armax,den_armax,1/Fs,2*pi*[0:0.1:123]);
figure
plot(wp/(2*pi),20*log10(abs(MAG_armax)))
%plot(w,20*log10(abs(Txy)),'r')
axis([5 123 -80 60])
set(gca,'fontsize',16)
feval('title',sprintf('SS(%d) FRF',armax_select_order),...
        'Fontname','TimesNewRoman','fontsize',ii)
ylabel('Magnitude (dB)','Fontname','TimesNewRoman','Fontsize',20)
xlabel('Frequency (Hz)','Fontname','TimesNewRoman','Fontsize',20)
disp('(press a key)')
pause % ,close
disp('   ')
load armax68plot
%---------------- Parametric vs non-parametric FRF comparison -------------
disp(' NON-PARAMETRIC AND PARAMETRIC FRFs ')
%--------------------------------------------------------------------------
%order = armax_select_order % select ARX orders                                             <----------
WINDOW = 1024; NFFT = 1024; OVERLAP = 0.8; % Welch based FRF parameters     <----------
%--------------------------------------------------------------------------
[MAG_armax,PHASE_armax,wp] = dbode(num_armax,den_armax,1/Fs,2*pi*[0:0.1:123]);
[Txy,w] = tfestimate(X1,Y1,WINDOW,round(OVERLAP*1024),NFFT,Fs);
figure
plot(wp/(2*pi),20*log10(abs(MAG_armax))),hold on
plot(wp68/(2*pi),20*log10(abs(MAG68)),'g'),hold on
plot(w,20*log10(abs(Txy)),'r')
axis([5 123 -80 60])
set(gca,'fontsize',16)
feval('title',sprintf('SS(%d) vs ARMAX(68,68,68) vs Welch based FRF',armax_select_order),...
        'Fontname','TimesNewRoman','fontsize',ii)
ylabel('Magnitude (dB)','Fontname','TimesNewRoman','Fontsize',20)
xlabel('Frequency (Hz)','Fontname','TimesNewRoman','Fontsize',20)
legend('State space based','ARMAX based','Welch based','Location','SouthEast','Orientation','vertical')
disp('(press a key)')
pause % ,close
disp('   ')
disp('   ')


%Check ccf between excitation and residuals
ccf_wn(X2,res_armax,100,0.8);
title('Excitation-Residuals CCF','fontsize',ii);
%--------------------------------------------------------------------------
%                           Signal - predictions
%--------------------------------------------------------------------------
TTtot=0:1/Fs:(length([Y2; Y2ver])-1)/Fs;
DAT_tot=iddata([Y2; Y2ver], [X2; X2ver], 1/Fs)
Yp_armax_tot=cell(1,armax_select_order);
Yp_armax_tot=predict(models_armax{armax_select_order},DAT_tot,1);
figure
plot(TTtot,DAT_tot.outputdata,'-o'), hold on
plot(TTtot,Yp_armax_tot.outputdata,'*')
title('Model one-step-ahead prediction (*) vs actual signal (o)','Fontname','TimesNewRoman','fontsize',12)
ylabel('Signal','Fontname','TimesNewRoman','Fontsize',ii)
xlabel('Time (s)','Fontname','TimesNewRoman','Fontsize',ii)
%xlim([11 11.3])

% reselect order for armax and check residuals

while 1

answer=input('do you want to select again order?\nyes-->1   no-->0\n');

if answer~=1
    break;
else
    
armax_select_order=input('select order for armax\n');

%define num, den f selected model
[num_armax, den_armax] = subspace_poly(models_armax{armax_select_order});
            
disp('Natural Frequencies (Hz)for ARMAX');
disp(nonzeros(fn_armax(armax_select_order,:)))
 
disp('Damping Ratios for ARMAX(%)');
disp(nonzeros(z_armax(armax_select_order,:)))
pause

res_armax=DATA1.outputdata-Yp_armax{armax_select_order}.outputdata;

figure
acf_wn(res_armax(armax_select_order+1:end),100,0.8);
%title('Residuals ACF for ARMAX','Fontname','TimesNewRoman','fontsize',12)
feval('title',sprintf('Residuals ACF for SS(%d)',armax_select_order),...
        'Fontname','TimesNewRoman','fontsize',ii)
ylim([-0.5 0.5])

%order = armax_select_order; % select ARX model orders                                       <----------
%--------------------------------------------------------------------------
[MAG_armax,PHASE_armax,wp] = dbode(num_armax,den_armax,1/Fs,2*pi*[0:0.1:123]);
figure
plot(wp/(2*pi),20*log10(abs(MAG_armax)))
%plot(w,20*log10(abs(Txy)),'r')
axis([5 123 -80 60])
set(gca,'fontsize',16)
feval('title',sprintf('SS(%d) FRF',armax_select_order),...
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
if maxar == 60
[num_52, den_52] = subspace_poly(models_armax{52});
[num_58, den_58] = subspace_poly(models_armax{58});
[num_60, den_60] = subspace_poly(models_armax{60});

[MAG_armax,PHASE_armax,wp] = dbode(num_armax,den_armax,1/Fs,2*pi*[0:0.1:123]);
[MAG_52,PHASE_52,wp_52] = dbode(num_52, den_52,1/Fs,2*pi*[0:0.1:123]);
[MAG_58,PHASE_58,wp_58] = dbode(num_58, den_58,1/Fs,2*pi*[0:0.1:123]);
[MAG_60,PHASE_60,wp_60] = dbode(num_60, den_60,1/Fs,2*pi*[0:0.1:123]);

figure
plot(wp_52/(2*pi),20*log10(abs(MAG_52)),'g'),hold on
plot(wp/(2*pi),20*log10(abs(MAG_armax)),'b'),hold on
plot(wp_58/(2*pi),20*log10(abs(MAG_58)),'k'),hold on
plot(wp_60/(2*pi),20*log10(abs(MAG_60)),'r'),hold on
% plot(w,20*log10(abs(Txy)),'r')
axis([5 123 -80 60])
set(gca,'fontsize',16)
title('State space FRF COMPARISON','Fontname','TimesNewRoman','Fontsize',20)
ylabel('Magnitude (dB)','Fontname','TimesNewRoman','Fontsize',20)
xlabel('Frequency (Hz)','Fontname','TimesNewRoman','Fontsize',20)
legend('SS(52)','SS(55)','SS(58)','SS(60)','Location','SouthEast','Orientation','vertical')
disp('(press a key)')
pause % ,close
disp('   ')
disp('   ')
pause

elseif maxar==65
        [num_61, den_61] = subspace_poly(models_armax{61});
        [num_62, den_62] = subspace_poly(models_armax{62});
        [num_64, den_64] = subspace_poly(models_armax{64});

        [MAG_armax,PHASE_armax,wp] = dbode(num_armax,den_armax, 1/Fs, 2*pi*[0:0.1:123]);
        [MAG_61,PHASE_61,wp_61] = dbode(num_61, den_61, 1/Fs, 2*pi*[0:0.1:123]);
        [MAG_62,PHASE_62,wp_62] = dbode(num_62, den_62, 1/Fs, 2*pi*[0:0.1:123]);
        [MAG_64,PHASE_64,wp_64] = dbode(num_64, den_64, 1/Fs, 2*pi*[0:0.1:123]);

        figure
        plot(wp_61/(2*pi),20*log10(abs(MAG_61)),'g'),hold on
        plot(wp/(2*pi),20*log10(abs(MAG_armax)),'b'),hold on
        plot(wp_62/(2*pi),20*log10(abs(MAG_62)),'k'),hold on
        plot(wp_64/(2*pi),20*log10(abs(MAG_64)),'r'),hold on
        % plot(w,20*log10(abs(Txy)),'r')
        axis([5 123 -80 60])
        set(gca,'fontsize',16)
        title('State space FRF COMPARISON','Fontname','TimesNewRoman','Fontsize',20)
        ylabel('Magnitude (dB)','Fontname','TimesNewRoman','Fontsize',20)
        xlabel('Frequency (Hz)','Fontname','TimesNewRoman','Fontsize',20)
        legend('SS(61)','SS(65)','SS(62)','SS(64)','Location','SouthEast','Orientation','vertical')
        disp('(press a key)')
        pause % ,close
        disp('   ')
        disp('   ')
        pause
        
        elseif maxar==70
        [num_66, den_66] = subspace_poly(models_armax{66});
        [num_68, den_68] = subspace_poly(models_armax{68});
        [num_69, den_69] = subspace_poly(models_armax{69});

        [MAG_armax,PHASE_armax,wp] = dbode(num_armax,den_armax, 1/Fs, 2*pi*[0:0.1:123]);
        [MAG_66,PHASE_66,wp_66] = dbode(num_66, den_66, 1/Fs, 2*pi*[0:0.1:123]);
        [MAG_68,PHASE_68,wp_68] = dbode(num_68, den_68, 1/Fs, 2*pi*[0:0.1:123]);
        [MAG_69,PHASE_69,wp_69] = dbode(num_69, den_69, 1/Fs, 2*pi*[0:0.1:123]);

        figure
        plot(wp_66/(2*pi),20*log10(abs(MAG_66)),'g'),hold on
        plot(wp/(2*pi),20*log10(abs(MAG_armax)),'b'),hold on
        plot(wp_68/(2*pi),20*log10(abs(MAG_68)),'k'),hold on
        plot(wp_69/(2*pi),20*log10(abs(MAG_69)),'r'),hold on
        % plot(w,20*log10(abs(Txy)),'r')
        axis([5 123 -80 60])
        set(gca,'fontsize',16)
        title('State space FRF COMPARISON','Fontname','TimesNewRoman','Fontsize',20)
        ylabel('Magnitude (dB)','Fontname','TimesNewRoman','Fontsize',20)
        xlabel('Frequency (Hz)','Fontname','TimesNewRoman','Fontsize',20)
        legend('SS(66)','SS(70)','SS(68)','SS(69)','Location','SouthEast','Orientation','vertical')
        disp('(press a key)')
        pause % ,close
        disp('   ')
        disp('   ')
        pause
        
        elseif maxar==75
        [num_71, den_71] = subspace_poly(models_armax{71});
        [num_73, den_73] = subspace_poly(models_armax{73});
        [num_74, den_74] = subspace_poly(models_armax{74});

        [MAG_armax,PHASE_armax,wp] = dbode(num_armax,den_armax, 1/Fs, 2*pi*[0:0.1:123]);
        [MAG_71,PHASE_71,wp_71] = dbode(num_71, den_71, 1/Fs, 2*pi*[0:0.1:123]);
        [MAG_73,PHASE_73,wp_73] = dbode(num_73, den_73, 1/Fs, 2*pi*[0:0.1:123]);
        [MAG_74,PHASE_74,wp_74] = dbode(num_74, den_74, 1/Fs, 2*pi*[0:0.1:123]);

        figure
        plot(wp_71/(2*pi),20*log10(abs(MAG_71)),'g'),hold on
        plot(wp/(2*pi),20*log10(abs(MAG_armax)),'b'),hold on
        plot(wp_73/(2*pi),20*log10(abs(MAG_73)),'k'),hold on
        plot(wp_74/(2*pi),20*log10(abs(MAG_74)),'r'),hold on
        % plot(w,20*log10(abs(Txy)),'r')
        axis([5 123 -80 60])
        set(gca,'fontsize',16)
        title('State space FRF COMPARISON','Fontname','TimesNewRoman','Fontsize',20)
        ylabel('Magnitude (dB)','Fontname','TimesNewRoman','Fontsize',20)
        xlabel('Frequency (Hz)','Fontname','TimesNewRoman','Fontsize',20)
        legend('SS(71)','SS(75)','SS(73)','SS(74)','Location','SouthEast','Orientation','vertical')
        disp('(press a key)')
        pause % ,close
        disp('   ')
        disp('   ')
        pause
        
        elseif maxar==80
        [num_76, den_76] = subspace_poly(models_armax{76});
        [num_78, den_78] = subspace_poly(models_armax{78});
        [num_79, den_79] = subspace_poly(models_armax{79});

        [MAG_armax,PHASE_armax,wp] = dbode(num_armax,den_armax, 1/Fs, 2*pi*[0:0.1:123]);
        [MAG_76,PHASE_76,wp_76] = dbode(num_76, den_76, 1/Fs, 2*pi*[0:0.1:123]);
        [MAG_78,PHASE_78,wp_78] = dbode(num_78, den_78, 1/Fs, 2*pi*[0:0.1:123]);
        [MAG_79,PHASE_79,wp_79] = dbode(num_79, den_79, 1/Fs, 2*pi*[0:0.1:123]);

        figure
        plot(wp_76/(2*pi),20*log10(abs(MAG_76)),'g'),hold on
        plot(wp/(2*pi),20*log10(abs(MAG_armax)),'b'),hold on
        plot(wp_78/(2*pi),20*log10(abs(MAG_78)),'k'),hold on
        plot(wp_79/(2*pi),20*log10(abs(MAG_79)),'r'),hold on
        % plot(w,20*log10(abs(Txy)),'r')
        axis([5 123 -80 60])
        set(gca,'fontsize',16)
        title('State space FRF COMPARISON','Fontname','TimesNewRoman','Fontsize',20)
        ylabel('Magnitude (dB)','Fontname','TimesNewRoman','Fontsize',20)
        xlabel('Frequency (Hz)','Fontname','TimesNewRoman','Fontsize',20)
        legend('SS(76)','SS(80)','SS(78)','SS(79)','Location','SouthEast','Orientation','vertical')
        disp('(press a key)')
        pause % ,close
        disp('   ')
        disp('   ')
        pause
end
end
end
pause
final_order=input('select final order');

[num_armax, den_armax] = subspace_poly(models_armax{final_order});
            
[MAG_armax,PHASE_armax,wp] = dbode(num_armax,den_armax,1/Fs,2*pi*[0:0.1:123]);
[Txy,w] = tfestimate(X1,Y1,WINDOW,round(OVERLAP*1024),NFFT,Fs);

figure
plot(wp/(2*pi),20*log10(abs(MAG_armax)),'b'),hold on
plot(w,20*log10(abs(Txy)),'r')
axis([5 123 -80 60])
set(gca,'fontsize',16)
feval('title',sprintf('SS(%d) vs Welch based FRF',final_order),...
        'Fontname','TimesNewRoman','fontsize',ii)
ylabel('Magnitude (dB)','Fontname','TimesNewRoman','Fontsize',20)
xlabel('Frequency (Hz)','Fontname','TimesNewRoman','Fontsize',20)
legend('SS based','Welch based','Location','SouthEast','Orientation','vertical')
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

function [num,den] = subspace_poly(x)

statesp = ss(x.A,[x.B x.K], x.C, [x.D 1], 1/256);
[num, den] = ss2tf(statesp.A, statesp.B, statesp.C, statesp.D, 1);
