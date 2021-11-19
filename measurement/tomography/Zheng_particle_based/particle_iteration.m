
clear all
it=10;
init=1;
for iter=init:it
Beam_I= 100; %mA
Mass=939.294; %Mass(MeV)
startstep=0.1;
step=0.4;
%��ɢ�������沽�� step of measured profile
stepx=startstep+step*(1-(iter-init)/it);%mm
stepy=startstep+step*(1-(iter-init)/it); %mm

%�ؽ����沽�� step for reconstruction
ystep=startstep+step*(1-(iter-init)/it);
xstep=startstep+step*(1-(iter-init)/it);

xrange=ceil(36/stepx)*stepx;%���ݰ����С��ѡ����ɢ����ķ�Χ range of the measured profiles
yrange=ceil(36/stepy)*stepy;
pnum=6;%�����ؽ��Ľ����� no. of profiles used to reconstruct
% 
% % %��һ����Ϊ����������ɢ���õ�prodis����ɢ��һ�ξͿ��� load the measured profile
% pro1=dlmread('projection1gauss100mA.txt','',3,0);
% % % pro1=pro1.data;
% pro2=dlmread('projection6gauss100mA.txt','',3,0);
% % % pro2=pro2.data;
% pro3=dlmread('projection8gauss100mA.txt','',3,0);
% pro4=dlmread('projection10gauss100mA.txt','',3,0);
% pro5=dlmread('projection13gauss100mA.txt','',3,0);
% pro6=dlmread('projection16gauss100mA.txt','',3,0);
% % % pro7=dlmread('projection7.txt','',3,0);
% % % pro3=pro3.data;
% pro={pro1,pro2,pro3,pro4,pro5,pro6};%,pro7
% % for i=1:pnum
% %     pro{i}(:,1)=pro{i}(:,1)*(-1);%����tracewin�����
% % end
% save 'pro.mat' pro
load pro.mat

prodis=zeros(round(2*xrange/stepx),round(2*yrange/stepy),pnum);%
% discretize measured profile with setted step
for j=1:pnum
    for i= 1: length(pro{j}(:,1))
     if abs(pro{j}(i,1))<xrange&&abs(pro{j}(i,3))<yrange
        prodis(ceil((pro{j}(i,1)+xrange)/stepx),ceil((pro{j}(i,3)+yrange)/stepy),j)=prodis(ceil((pro{j}(i,1)+xrange)/stepx),ceil((pro{j}(i,3)+yrange)/stepy),j)+1;
        if isnan(prodis(ceil((pro{j}(i,1)+xrange)/stepx),ceil((pro{j}(i,3)+yrange)/stepy),j))
            sprintf('kk');
        end
     end
    end
end
prodis=permute(prodis,[2 1 3]);%������Ϊ�������ķֲ���ת�õģ���������
%normalization
for i=1:pnum
    prodis(:,:,i)=prodis(:,:,i)/sum(sum(prodis(:,:,i)));
end

% 
% 
% save prodis.mat prodis
% load prodis.mat
% %��ʾ��ɢ��prodis
figure(1)
for i=1:pnum
    subplot(2,3,i);
    imagesc(-xrange:stepx:xrange-stepx,-yrange:stepy:yrange-stepy,prodis(:,:,i));
    xlabel('x [mm]');
    ylabel('y [mm]');
end
% if iter==1
%     N=1000000;
% else
load parnum.mat
% end
Q2=[6.46 	4.97 	3.99 	3.47 	4.32 	5.18   ];   %�ı�Q��kֵ������1,6,8,10,13,16
Q3=[-5.63 	-4.53 	-2.59 	-4.01 	-4.42 	-4.84  ];      %�ı����һQ��kֵ
iter
elenum=2;%�ܹ�Ԫ����
for i=1:pnum
    
    cmd=['TraceWin.exe XiPAF-IHDTL-beamline_setting1_3_new.ini  current1=',num2str(Beam_I),' dst_file1=beam',num2str(iter-1),'.dst ele[2][2]=',num2str(Q2(i)),' ele[4][2]=',num2str(Q3(i)),' nbr_part1=',num2str(N)];

    system(cmd);
pdata=tracewinplt2txt(elenum);
temdata=pdata{1,elenum}';
%��λת��
temdata(:,[1,3])=temdata(:,[1,3])*10;
temdata(:,[2,4])=temdata(:,[2,4])*1000;
pror{i}=temdata;
end
% load prorsimu.mat
%������ڷֲ�
pdo= pdata{1,1}';
pdo(:,[1,3])=pdo(:,[1,3])*10;
pdo(:,[2,4])=pdo(:,[2,4])*1000;


%���ӱ��
for i=1:pnum
    pror{i}(:,8)=1:length(pror{1}(:,1));
end
    pdo(:,8)=1:length(pror{1}(:,1));
 %�ҳ������н�����������ӣ���˺��洦����ֻ�ڴ������ӽ���,��֮ǰtracewinģ��ʱ���������ӽ��в�ͬ
 indexloss=[];
for i=1:pnum
 inde{i}=find(pror{i}(:,7)~=0);
 indexloss=[indexloss;inde{i}];
end
indexloss=unique(indexloss);
pror{pnum}(indexloss,:)=[];
for i=1:pnum-1
  interp=ismember(pror{i}(:,8),pror{pnum}(:,8));
  in=find(interp==0);
  pror{i}(in,:)=[];
end


%ÿ�������ڲ�ͬ����ı���
ind=zeros(pnum,length(pror{1}(:,1)));
for j=1:pnum
    for k=1:length(pror{j}(:,1))
        if abs(pror{j}(k,1))<xrange&&abs(pror{j}(k,3))<yrange
         ind(j,k)=prodis(ceil((pror{j}(k,3)+yrange)/stepy),ceil((pror{j}(k,1)+xrange)/stepx),j);
        end
        
    end
end

%�������ڶ����������ı�����˵õ��ܱ���
indtemp=ind(1,:);
for i=2:pnum
indtemp=indtemp.*ind(i,:);
end

%��Ч������ţ��ڴ������������
cx=find(indtemp~=0);
%����Ч�����ڸ�������ֲ�
prodispart=zeros(round(2*xrange/stepx),round(2*yrange/stepy),pnum);
for j=1:pnum
    for i= 1: length(cx)
    
        prodispart(ceil((pror{j}(cx(i),1)+xrange)/stepx),ceil((pror{j}(cx(i),3)+yrange)/stepy),j)=prodispart(ceil((pror{j}(cx(i),1)+xrange)/stepx),ceil((pror{j}(cx(i),3)+yrange)/stepy),j)+1;
    
   end
end
prodispart=permute(prodispart,[2 1 3]);%������Ϊ�������ķֲ���ת�õģ���������
prorec=prodispart;
%��һ��
for i=1:pnum
prorec(:,:,i)=prorec(:,:,i)/sum(sum(prorec(:,:,i)));
end
%�����ؽ������������������
for i=1:pnum
  stde(i)=sqrt(sum(sum((prorec(:,:,i)-prodis(:,:,i)).^2))/numel(prorec(:,:,i)));
end

 %������ϵ�������������й�һ�����õ�ÿ��������ռ����
 for i=1:pnum
    prodis(:,:,i)=prodis(:,:,i)./prodispart(:,:,i);
 end
 prodis(isnan(prodis))=0;
 prodis(isinf(prodis))=0;
  %����Ч���������ϵ��
 ind=zeros(pnum,length(pror{1}(:,1)));
for j=1: pnum
 for k=1:length(cx)
    if abs(pror{j}(cx(k),1))<xrange&&abs(pror{j}(cx(k),3))<yrange
     ind(j,cx(k))=prodis(ceil((pror{j}(cx(k),3)+yrange)/stepy),ceil((pror{j}(cx(k),1)+xrange)/stepx),j);
      else 
      ind(j,cx(k))=0;
    end
end
end
%�������ڶ����������ı�����˵õ��ܱ���
index=ind(1,:);
for i=2:pnum
index=index+ind(i,:);
end
%���ع�һ��
index =index/sum(index);
%��ʾ�ؽ������������������ֲ�
figure(2);
for i=1:pnum
    subplot(2,3,i);
    imagesc(-xrange:stepx:xrange-stepx,-yrange:stepy:yrange-stepy,prorec(:,:,i));
    xlabel('x [mm]');
    ylabel('y [mm]');
end

%���ؽ�������
num=max(500000,length(cx));
%ѡ����ʼ�ֲ��еĴ������
pdi=pdo;
interp=ismember(pdo(:,8),pror{pnum}(:,8));
in=find(interp==0);
pdo(in,:)=[];
%����ÿ������ϵ���������ɷֲ�
pdit=zeros(num,6);
 numtep1=find(num*index<1.4);
 numtep2=find(num*index>=0.5);
 numinter1=intersect(numtep1,numtep2);
 numtep3=find(num*index>1.4);
 pdit(1:length(numinter1),:)=pdo(numinter1,1:6);
% numtep=intersect(numtep1,numtep2);
% num=num-length(numtep);
istep=0;
for i=1:length(numtep3)
   if i==1
      start=length(numinter1)+1;
   else
    start=start+istep;
   end
      %��ԭ�����������񴦣���x,x',y,y'���������������(0~stepx��
    istep=round(index(numtep3(i))*num);
    x=floor((pdo(numtep3(i),1))/xstep);
  xp=floor((pdo(numtep3(i),2))/xstep);
   y=floor((pdo(numtep3(i),3))/ystep);
  yp=floor((pdo(numtep3(i),4))/ystep);
  u=unifrnd(0,xstep,istep,4);
  pri=repmat([x,xp,y,yp],istep,1);
   pdit(start:start+istep-1,1:4)=pri.*repmat([xstep,xstep,ystep,ystep],istep,1)+u;

  
end
 %����ÿ�ε����ؽ�����������������
std=[iter stde];
save('std.txt','std','-ascii','-append');
 
azero= pdit(:,1)==0;
pdit(azero,:)=[];
N=length(pdit(:,1));
if length(pdi(:,1))<N
     pdi=repmat(pdi,ceil(N/length(pdi(:,1))),1);
end

%����λת����dstҪ��
 pdit(:,[1,3])= pdit(:,[1,3])/10;
 pdit(:,[2,4])= pdit(:,[2,4])/1000;
 pdit(1:N,5:6)=pdi(1:N,5:6);

% %����ÿ�ε����ؽ��ķ����
ex=dlmread('partran1.out','',14,0); 
emit=[iter ex(1,[16,17,26,27])];
save('emit.txt','emit','-ascii','-append');

%�����µ�dst�ļ� generate new input particle distribution for the next iteration
 txt2tracewindst(pdit,iter,Beam_I,Mass);
if iter ~= it
    save parnum.mat N 
    clear 
    it=10;
    init=1;
end
end

