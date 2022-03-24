
function [frff,frffcu,frffcd]=interv(X,Y,noverlap,L,fs);

j=14 ;jj=16;
colC=[0.8 0.8 0.8];

% [pxx,f1,pxxc]=pwelch(X,hamming(L),noverlap,L,fs,'ConfidenceLevel',0.95);
[pxx,f1]=pwelch(X,hamming(L),noverlap,L,fs);
plot(f1,20*log10(pxx));
hold on
plot(f1,20*log10(pxx),'-');
hold off

figure
[pxy,f2,pxyc]=cpsd(X,Y,hamming(L),noverlap,L,fs,'ConfidenceLevel',0.95);
plot(f2,20*log10(pxy));
hold on
plot(f2,20*log10(pxyc),'-');
hold off

% [pyy,f2,pyyc]=pwelch(Y,hamming(L),noverlap,L,fs,'ConfidenceLevel',0.95);
[pyy,f2]=pwelch(Y,hamming(L),noverlap,L,fs);

figure 
frff=pxy./pxx;
% frffcu=pxyc(:,1)./pxxc(:,2);
% frffcd=pxyc(:,2)./pxxc(:,1);
frffcu=pxyc(:,1)./pxx;
frffcd=pxyc(:,2)./pxx;

coh=(pxy.^2)./(pxx.*pyy);
% cohu=(pxyc(:,2).^2)./(pxxc(:,2).*pyyc(:,2));
% cohd=(pxyc(:,1).^2)./(pxxc(:,1).*pyyc(:,1));
cohu=(pxyc(:,2).^2)./(pxx.*pyy);
cohd=(pxyc(:,1).^2)./(pxx.*pyy);

plot(f2,20*log10(frff));
hold on
plot(f2,20*log10(frffcu));
plot(f2,20*log10(frffcd));
hold off



figure
set(gcf,'paperorientation','landscape','paperposition',[0.63 0.63 28.41 19.72]);
%subplot(3,2,1) 
box on, hold on
plot(f2,20*log10(abs(frff)),'color',colC); 
plot(f2,20*log10(frffcu),'color',colC);
plot(f2,20*log10(frffcd),'color',colC);
%plot(w,20*log10(abs(Txy)),'--b')
xlim([3 fs/2])
set(gca,'Fontname','timesnewroman','fontsize',j)
title('Non-Parametric FRF with 95% confidence intervals','Fontname','timesnewroman','fontsize',jj)
ylabel('Magnitude (dB)','Fontname','TimesNewRoman','Fontsize',jj)
xlabel('Frequency (Hz)','Fontname','TimesNewRoman','Fontsize',jj)
%legend('Healhty','Damage A','Damage B','Damage C','Damage D','Damage E','Location','SouthEast',...
%    'Orientation','horizontal')
%legend([ph,pA,pB],'healthy','damage I','damage II','Location','SouthWest','Orientation','horizontal')
set(patch,'EdgeAlpha',0,'FaceAlpha',0)
patch([f2;flipud(f2)],[20*log10(abs(frffcu));flipud(20*log10(abs(frffcd)))],colC);


figure
set(gcf,'paperorientation','landscape','paperposition',[0.63 0.63 28.41 19.72]);
%subplot(3,2,1) 
box on, hold on
plot(f2,(abs(coh)),'color',colC); 
plot(f2,(cohu),'color',colC);
plot(f2,(cohd),'color',colC);
%plot(w,20*log10(abs(Txy)),'--b')
xlim([3 fs/2])
set(gca,'Fontname','timesnewroman','fontsize',j)
title('Non-Parametric Coherence with 95% confidence intervals','Fontname','timesnewroman','fontsize',jj)
ylabel('Magnitude (dB)','Fontname','TimesNewRoman','Fontsize',jj)
xlabel('Frequency (Hz)','Fontname','TimesNewRoman','Fontsize',jj)
%legend('Healhty','Damage A','Damage B','Damage C','Damage D','Damage E','Location','SouthEast',...
%    'Orientation','horizontal')
%legend([ph,pA,pB],'healthy','damage I','damage II','Location','SouthWest','Orientation','horizontal')
set(patch,'EdgeAlpha',0,'FaceAlpha',0);
patch([f2;flipud(f2)],[abs(cohu);flipud(cohd)],colC);