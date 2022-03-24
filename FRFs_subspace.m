clear;
clc;
close all;

load FRFs_subspace

figure
%plot(wp60/(2*pi),20*log10(abs(MAG_armax60)),'y'),hold on
plot(wp65/(2*pi),20*log10(abs(MAG_armax65)),'y'),hold on
plot(wp70/(2*pi),20*log10(abs(MAG_armax70)),'c'),hold on
plot(wp75/(2*pi),20*log10(abs(MAG_armax75)),'k'),hold on
plot(wp80/(2*pi),20*log10(abs(MAG_armax80)),'r'),hold on
plot(wp85/(2*pi),20*log10(abs(MAG_armax85)),'b'),hold on
plot(wp90/(2*pi),20*log10(abs(MAG_armax90)),'g'),hold on
plot(w,20*log10(abs(Txy)),'m')
axis([5 123 -80 60])
set(gca,'fontsize',16)
title('State space FRF COMPARISON','Fontname','TimesNewRoman','Fontsize',20)
ylabel('Magnitude (dB)','Fontname','TimesNewRoman','Fontsize',20)
xlabel('Frequency (Hz)','Fontname','TimesNewRoman','Fontsize',20)
legend('SS(65)','SS(70)','SS(75)','SS(80)','SS(85)','SS(90)','Welch based','Location','SouthEast','Orientation','vertical')
disp('(press a key)')
pause 