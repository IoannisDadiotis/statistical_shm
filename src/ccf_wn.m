function rk=ccf_wn(x,y,maxlag,barsize);
% rk=acf_wn(x,maxlag);

R=xcorr(x,y,maxlag,'coeff');
%disp(length(R))
rk=R(:);
bar([-maxlag:maxlag],rk,barsize,'b'),hold
plot([-maxlag:maxlag],(1.96/sqrt(length(x))).*ones(2*maxlag+1,1),'r',[-maxlag:maxlag],(-1.96/sqrt(length(y))).*ones(2*maxlag+1,1),'r')
axis([-maxlag-1 maxlag+1 -1 1]),xlabel('Lag'),ylabel('C.C.F. ( \rho_\kappa )')
zoom on,hold
end

