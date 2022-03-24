function rk=acf_wn(x,maxlag,barsize,i);
% rk=acf_wn(x,maxlag);

R=xcorr(x,maxlag,'coeff');
%disp(length(R))
rk=R(maxlag+2:end);
bar([1:maxlag],rk,barsize,'b'),hold
plot([1:maxlag],(1.96/sqrt(length(x))).*ones(maxlag,1),'r',[1:maxlag],(-1.96/sqrt(length(x))).*ones(maxlag,1),'r')

axis([0 maxlag+1 -1 1]),xlabel('Lag'),ylabel('A.C.F. ( \rho_\kappa )')

zoom on,hold

end

