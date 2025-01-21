close all
clear all

study_site='Rasnicka_2024';

%To estimate the uncertainties associated with the distribution
% we use a bootstrap approach with n_iter repetitions.
n_iter = 100;

load_data_folder='Data/';
save_results_folder='Results/';
save_figure_folder='Figures/';

ellipsoid_file=['ellipse_fit_' study_site '.txt'];
pc_file=['pc_labelled_' study_site '.txt'];

f1=load([load_data_folder ellipsoid_file]);
baxis=2*f1(:,9); % FILES ARE IN RADIUS, WE NEED DIAMETERS
dx=1.1*max(baxis);labels_ellipsoid=f1(:,7);
f2=load([load_data_folder  pc_file]);
x=f2(:,1);y=f2(:,2);z=f2(:,3);labels_grains=f2(:,8);

d=cell(n_iter);d_sample=[];
for i=1:n_iter
    clear iwol dist
    r=rand(2,1);
    xgrid=min(x)-r(1)*dx:dx:max(x);ygrid=min(y)-r(2)*dx:dx:max(y);
    [Xgrid,Ygrid]=meshgrid(xgrid,ygrid);Xgrid=reshape(Xgrid,numel(Xgrid),1);Ygrid=reshape(Ygrid,numel(Ygrid),1);

    dist=zeros(1,numel(Xgrid));iwol=zeros(1,numel(Xgrid));
    for k=1:numel(Xgrid)
        [dist(k),iwol(k)]=min(sqrt((x-Xgrid(k)).^2+(y-Ygrid(k)).^2));
    end
    iwol=iwol(dist<dx/10);
    [~,~,ib] = intersect(labels_grains(iwol),labels_ellipsoid);

    d(i)={baxis(ib)*1000}; % conversion to millimeter
end

for i=1:n_iter
    d_sample=[d_sample;d{i}];
    dq(i,:)=quantile(d{i},[0.1 0.5 0.9]);
end

edq=std(dq);
dq_final=quantile(d_sample,[0.1 0.5 0.9]);

figure
semilogx(sort(d_sample),(1:numel(d_sample))/numel(d_sample),'g','linewidth',2);
hold on
errorbar(dq_final,[0.1 0.5 0.9],0,0,edq,edq,'og')
plot(dq_final,[0.1 0.5 0.9],'og','markerfacecolor','w')
xlabel('Diameter (mm)');ylabel('CDF');
xwidth = 10; ywidth = 8;
set(gcf, 'PaperPosition', [0 0 xwidth ywidth]);
set(gcf, 'PaperSize', [xwidth,ywidth])
set(gca,'box','on')
set(gca, 'LooseInset', [1,1,1,1]*0.03);
saveas(gcf,[save_figure_folder study_site '_GSD_G3Point.pdf'])

figure
semilogx(xx,yy,'r','linewidth',2)
hold on
semilogx(sort(d_sample),(1:numel(d_sample))/numel(d_sample),'g','linewidth',2);
errorbar(dd,[0.1 0.5 0.9],0,0,e,e,'or')
plot(dd,[0.1 0.5 0.9],'or','markerfacecolor','w')
hold on
errorbar(dq_final,[0.1 0.5 0.9],0,0,edq,edq,'og')
plot(dq_final,[0.1 0.5 0.9],'og','markerfacecolor','w')
legend('Manual','G3Point','location','northwest')
xlabel('Diameter (mm)')
ylabel('CDF')
xwidth=10; ywidth=8;
set(gcf, 'PaperPosition', [0 0 xwidth ywidth]);
set(gcf, 'PaperSize', [xwidth,ywidth])
set(gca,'box','on')
hold off
saveas(gcf,[save_figure_folder study_site '_Wolman_G3POINT.pdf'])

filename=([save_results_folder study_site '_GSD_G3POINT.txt']);
fid=fopen(filename,'w');
fprintf(fid,'%%D10 D50 D90 (mm)\n');
fprintf(fid,'%5.0f\t %5.0f\t %5.0f\t \n',dq_final(1),dq_final(2),dq_final(3));
fprintf(fid,'%%std(D10) std(D50) std(D90) (mm)\n');
fprintf(fid,'%4.0f\t %4.0f\t %4.0f\t\n',edq(1),edq(2),edq(3));
fclose(fid);

%% Cette partie est reprise de ellipsoidorientation3d.m
%%J'ai un doute sur les lignes 101, 102 et 103 : est-ce que ce sont les
%%bons paramètres ?

delta=1e32;

data=load([load_data_folder ellipsoid_file]);
k=0; 

for j=1:size(data,1)
    sensorCenter = [data(j,1),data(j,2)+delta,data(j,3)]; %x-y plot - mapview (angle with y axis)
    k=k+1;
    u(k)=data(j,11); % 
    v(k)=data(j,12);
    w(k)=data(j,13);

    p1 = sensorCenter ;  p2 = [u(k),v(k),w(k)];
    angle = atan2(norm(cross(p1,p2)),p1*p2');
    if angle > pi/2 || angle < -pi/2
        u(k) = -u(k); v(k) = -v(k); w(k) = -w(k);
    end
    alpha(k)=atan(v(k)/u(k))+pi/2;
end
granulo.angle_Mview=alpha;

k=0; 
for j=1:size(data,1)
    sensorCenter = [data(j,1),data(j,2),data(j,3)+delta ]; %x-z plot
    k=k+1;
    u(k)=data(j,11);
    v(k)=data(j,12);
    w(k)=data(j,13);
    p1 = sensorCenter ;  p2 = [u(k),v(k),w(k)];
    angle = atan2(norm(cross(p1,p2)),p1*p2');
    if angle > pi/2 || angle < -pi/2
        u(k) = -u(k); v(k) = -v(k); w(k) = -w(k);
    end
   alpha2(k)=atan(v(k)/w(k))+pi/2;          
end

granulo.angle_Xview=alpha2;

figure
subplot(211);hist(granulo.angle_Mview*180/pi);xlabel('Azimut (°)');axis tight;
subplot(212);hist(granulo.angle_Xview*180/pi);xlabel('Dip (°)');axis tight;
xwidth=10; ywidth=10;
set(gcf, 'PaperPosition', [0 0 xwidth ywidth]);
set(gcf, 'PaperSize', [xwidth,ywidth])
set(gca,'box','on')
hold off
set(gcf, 'PaperPosition', [0 0 xwidth ywidth]);
set(gcf, 'PaperSize', [xwidth,ywidth])
set(gca,'box','on')
hold off
saveas(gcf,[save_figure_folder study_site '_Orientation.pdf'])




