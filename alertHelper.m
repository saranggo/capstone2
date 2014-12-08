function varargout = alertHelper( action, varargin )
varargout = cell(1,max(1,nargout));
[varargout{:}] = feval(action,varargin{:});
end

function [gt,dt] = loadAlerts( gtFile, dtFile )
gt=load(gtFile,'-ascii');
gt=double(gt);
dt=load(dtFile,'-ascii');
dt=dt(:,[1,end]);
%dt1=zeros(max(length(gt),max(dt(:,1))),2) - 100;
%dt1(:,1)=dt1(:,1) + (1:length(dt1))' + 100;
%dt1(dt(:,1),2)=dt(:,2);
dt=dt(dt(:,1)<length(gt),:);
dt=double(dt);
end

function [prec, tpr, fpr, thresh] = compPlots(gt,dt)
target=gt(:,end);
n=length(target);
ids=dt(:,1); assert(max(ids)<=n);
scores=[(1:n)',(zeros(n,1) + min(dt(:,2)))]; 
for i=1:n
    tmp=max(dt(ids==i,2));
    if ~isempty(tmp)
        scores(i,2)=max(scores(i,2),tmp);
    end
end
gt0=gt(:,end)==0;
gt1=gt(:,end)==1;
n=scores(gt0,:);
p=scores(gt1,:);
n1=n(n(:,2)>-0.749,:);
p1=p(p(:,2)<-0.749,:);
p2=p(p(:,2)>-0.749,:);
[~,ord]=sort(p2(:,2),'ascend');
p2=p2(ord,:);
[prec, tpr, fpr, thresh]=prec_rec(scores(:,2),target,'plotBaseline',0);
end