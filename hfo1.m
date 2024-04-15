%Copyright (c) 2024, Zeki Yetgin & Uğur Ercan
%All rights reserved.
%Redistribution and use in source and binary forms, with or without modification, are permitted provided that following papers are cited:
%Please cite the following papers
%Z. Yetgin, U. Ercan, "Honey formation optimization with single component for numerical function optimization: HFO-1", Neural Computing and Applications, 2023,https://doi.org/10.1007/s00521-023-08984-1
%Z. Yetgin, H. Abaci,"Honey formation optimization framework for design problems", Applied Mathematics and Computation, 394(1), 2021.

%Important Notes: 
%1- Define ObjectiveFun(x) function in this file (or in the folder) before calling Hfo1 Algorithm. 
%2- By default Algorithm terminates when the error is less then 1e-18 (comment the break line at the end of main loop if not suitable )
%3-Change external and internal variables in model for further adaptation. 
%4-Hfo1 uses explicit for loops that slows the Algorithm. Check Hfo1 vectoral implementation if available 


%Example usage of Hfo1
%As a demo, 10 dimensional Trid Function (ObjectiveFun is defined in this file) can be optimized by calling Hfo1 as: 
%[GBest, tcurve] = hfo1(15,10,-100*ones(1,10),100*ones(1,10),70000,-210); 

function [GBest, tcurve]=hfo1(nSource,dim,lb,ub,maxIter,goptimum)
%inputs: nSource:number of Sources; dim: dimension; lb,ub: vector of lower and upper bounds; maxIter: maximum iterations
%        goptimum(optional):global optimum value,used to terminate when the error is less than 1e-18. To eliminate this feature, dont use the last param  
%outputs: GBest.Position is the solution with GBest.cost as objective value; tcurve: the evolution curve
optExit=(nargin==6); %check whether goptimum is given

model=CreateModel(nSource,dim,lb,ub); % Algorithm settings and problem setup

% Bee Memory Structure
empty_bee.Position=[]; %represent the honey source : position and honey are associated
empty_bee.Cost=[];     %represent the honey fitness
empty_bee.CCost=[];    %represent the component fitness
gbestCost=0;  newSite=false; premature=false;

% create and initialize Food Sources  
Sources=repmat(empty_bee,nSource,1);    
Sources=InitializePop(model,Sources,gbestCost); 

%% internal parameters and initial settings
MF=model.MF; 			   %mixing freq. factor in terms of nSource
randPool=model.randPool;   %rand pool for mixer sizes
mixPeriod=16*MF;    	   %min waiting period for mixing sources, used by two mixers:mixer1, mixer2 
maturePeriod=4*mixPeriod;  %min waiting period to check for honey maturation 
sitePeriod=4*maturePeriod; %min period to stay in a site 
finalizationPeriod=4*sitePeriod; %min waiting period to go into finalization phase for the site 
precision=1e-3;  %to eliminate division by zero and also affect the selection prob P
siteLim=2+abs(log10(precision)); %maximum number of sites to replace

% C is the update counter for each source, counters c1 and c2 keep track of mixer1 & mixer2 states 
%c3 is the update counter for GBest,counters c4 and c5 keep track of maturation and saturation state
C=zeros(nSource,1); c1=0; c2=0; c3=0; c4=0; c5=0;  %counters 

tcurve=zeros(1,maxIter); % to visualize the evaluation curve 

%save frequently used values to variables
ix=1:dim; 
fox=1:nSource; 
halfPeriod=2*mixPeriod;

mx=1; sn=4; %sn=ceil(nSource/4); %other alternative
PBest=Sources(mx); GBest=PBest; pGBest=GBest; 

%% HFO Main Loop
for it=1:maxIter 
 %Worker Phase
  	for i=fox  
      xi= Sources(i).Position;     		%xi is the source associated with worker i
	  ccostxi=  Sources(i).CCost;  		%ccostxi is the component cost of xi
			
      vi= LocalSearch(Sources,i,model); % vi is the alternative solution
	  
	  %analyse the alternative solution vi
	  costvi=ObjectiveFun(vi);
	  ccostvi=costvi-gbestCost;
    	
	  if ccostvi<ccostxi 		%keep the better one in memory according to component cost
           Sources(i).Position = vi;
           Sources(i).CCost=ccostvi;
		   Sources(i).Cost=costvi; 
            C(i)=0;		
        else
            C(i)=C(i)+1;
        end   
    end

  %Onlooker Phase
    hcosts = [Sources.Cost];
	hcosts = hcosts-min(hcosts)+precision;
	fits = 1./hcosts; %convert costs to fitnesses
	P=fits/sum(fits);
	%P=0.1+0.9*fits/sum(fits); %alternative for P
    CP=cumsum(P);
	
     for m=fox
        i=find(rand<=CP,1);	%select a source by RouletteWheelSelection		 

		xi= Sources(i).Position;       	  %xi is the source associated with onlooker m
        costxi = Sources(i).Cost;      	  %costxi is the cost of xi
		
        vi= LocalSearch(Sources,i,model); % vi is the alternative solution
        costvi=ObjectiveFun(vi);

        if costvi<costxi 		%keep the better one in memory according to cost
            Sources(i).Position = vi;
			Sources(i).Cost=costvi;
			Sources(i).CCost=costvi-gbestCost;
			C(i)= 0;
        else
            C(i)=C(i)+1;
        end
        
    end
	
%mix the sources if they are matured enough
	mixed1=false; mixed2=false;
	session1=c1>halfPeriod; session2=c2>maturePeriod;
	if session1 || session2
	for i=fox
	      m=0;
		  if session2  &&  C(i)>maturePeriod 
			mixed2=true; 
			m=randi(model.mixSize(2)); %choose mixer2 size (update factor)
		  elseif session1 &&  C(i)>halfPeriod 
			mixed1=true; 
			m=randi(model.mixSize(1)); %choose mixer1 size (update factor)
          end	
          if m>0	 %mix the source i	  
			x=Sources(i).Position; 
			K=[1:i-1 i+1:nSource];
            k=K(randi([1 numel(K)]));
			xk=Sources(k).Position; 
			xb=PBest.Position ;

			j=randsample(ix,m);
			jk=randsample(ix,m);
			if rand>rand
			    x(j)=xk(jk);             
            else
				x(j)=xb(jk);			 
			end	
			%%following two if lines are added according to the modified Hfo1 that corrects any out of border assignment if dimensions have different search ranges
         	tx=j(x(j)>ub(j));
            if ~isempty(tx), x(tx)=unifrnd(lb(tx),ub(tx),[1 length(tx)]); end
		 
            tx=j(x(j)<lb(j));
            if ~isempty(tx), x(tx)=unifrnd(lb(tx),ub(tx),[1 length(tx)]);  end
			
			Sources(i).Position=x;
			% Analyse the quality of the food source 
            Sources(i).Cost = ObjectiveFun(x); 
			Sources(i).CCost=Sources(i).Cost- gbestCost;
           C(i)=0; 
        end
    end %for
    end %if
	
	if mixed1  
	  c1=0;	  
	else
      c1=c1+1;   
    end;

	if mixed2 
	   c2=0; 
	 else
       c2=c2+1;
       if c2>maturePeriod %force mix if not mixed for along time
  	      C=4*C;  	   
	   end;
	 end

 %dont accept if previous pbest is distrupted 
    if Sources(mx).Cost>PBest.Cost
	   Sources(mx)=PBest;
    end	 

 %Update Best Solution Ever Found
	hcosts = [Sources.Cost];
	[minv,mx]=min(hcosts); 
	
	if minv<GBest.Cost  
	   GBest=Sources(mx);
	   model.mixSize(:)=dim;
	   c3=0; 
	 else  
       c3=c3+1; 	   
	end
    
	gbestCost=GBest.Cost;
    PBest=Sources(mx);
 
    %check periodically for convergence, whether honey is homegenous 
    if mod(it,maturePeriod)==0
			kx=randsample(fox,sn); 
			same1=EpsilonSame(hcosts(kx));
			same2=EpsilonSame([GBest.Cost,pGBest.Cost]);
            mature=same1&&same2;			
			if c4<=siteLim&&premature&&mature&&c5<=siteLim   %if maturation state, change the current site
				Sources=InitializePop(model,Sources,gbestCost);  PBest=Sources(mx); 		   
			    newSite=true;  premature=false; 				
			    c4=c4+1; C(:)=0;
            else 
				if newSite || c3>finalizationPeriod 		%if saturation state or newSite, force homegenization
					mx=kx(1); Sources(mx)=GBest;
				    newSite=false; 
                end
				if not(same2), c5=c5+1; end			
			    if abs(pGBest.Cost-GBest.Cost)>0.1, c4=0; c5=0;  end	%if imrovement exceeds the 0.1 neighbourhood, reset counters c4 & c5
			    pGBest=GBest; premature=mature;

			end 
	 end

	 %if honey is not improved within the maturePeriod, auto tune the mixer sizes
	if c3>maturePeriod 		
		  model.mixSize(1)=randsample(randPool,1);
		  model.mixSize(3)=randsample(randPool,1);	  
    end  

  % Store Best Cost Ever Found
    tcurve(it)=gbestCost;
    if mod(it,mixPeriod)==0
	 fprintf('Iter:%6d --> Best:%.32f \n',it,gbestCost); 
	end
	
   if  optExit && (abs(goptimum-gbestCost) <1e-18), break;, end %comment this line if there is no exit criteria on error
   
 end %end of main loop
 fprintf('Iter:%6d --> Best:%.32f \n',it,gbestCost); 
end 

%this function emitates honey formation inside the bee where the components are mixed using bee's enzyms and evolves to honey. 
function vi=LocalSearch(pop,i,model)    
   ub=model.ub; 
   lb=model.lb; 
   n=model.n;
   q=model.qo;
   
   nSource = model.nSource;
   xi=pop(i).Position;
   
   %Choose mixer3 size (the update factor)
   m=randi(model.mixSize(3));
	 
    %Choose a random food source xk (k != i) 
    K=[1:i-1 i+1:nSource];
    k=K(randi([1 numel(K)]));
	xk=pop(k).Position;
	
    if model.randWalk && rand<0.5 , q=ceil(0.25/rand); end %activate random walk
	
    %Choose dimensions to update
	j=randsample(1:n,m);

	%Choose the step length
    phi=q*unifrnd(-1,1);
	vi=xi;
  % solution update equation : ABC	
	vi(j)=xi(j)+phi*(xi(j)-xk(j));
  
  % border control
  ubx=j(vi(j)> ub(j));
  lbx=j(vi(j)< lb(j));
  lu=length(ubx);
  ll=length(lbx);
 
  vi(ubx)=ub(ubx);
  vi(lbx)=lb(lbx);
  
  if lu>1
   nru=randi(lu);
   rux=randsample(ubx,nru);
   vi(rux)=unifrnd(lb(rux),ub(rux),[1 nru]);
  end
  if ll>1
   nrl=randi(ll); 
   rlx=randsample(lbx,nrl); 
   vi(rlx)=unifrnd(lb(rlx),ub(rlx),[1 nrl]);
  end
 
end

function model=CreateModel(nSource,n,lb,ub)
	%% external parameters
    model.nSource=nSource;	
    model.n = n;
    model.ub=ub;
    model.lb=lb;
	model.qo=2; %some recommended values={1/8,1/4,2}
    model.MF=8; %some recommended values={2,8,16}
    model.randWalk=true;  %optional property uses random walk in local search
	
	%% internal parameters and initial settings
	if n==2
	  model.randPool=[n];
	else
	  model.randPool=[1,ceil(n/4), n];
    end
	model.mixSize=[n,n,n];
 end

function Sources=InitializePop(model,Sources,gbestCost)
nSource=model.nSource;
lb=model.lb;         % Variables Lower Bound   
ub= model.ub;         % Variables Upper Bound  

for i=1:nSource 
    Sources(i).Position=unifrnd(lb,ub,[1 model.n]);    
    Sources(i).Cost = ObjectiveFun(Sources(i).Position); 
	Sources(i).CCost=Sources(i).Cost-gbestCost;
end
end

function y=EpsilonSame(cost)
   y=true;
   cost2=cost(2:end);
   cost1=cost(1:end-1);
   ix=cost1~=cost2;
   if isempty(ix), return; end
   cost1=cost1(ix); cost2=cost2(ix);
   d=abs(cost1-cost2);
   if all(d>1)
      n=ceil(log10(d))+2;
      cost2=cost2./(10.^n);
      cost1=cost1./(10.^n);
    end
	fx1=round(cost1);
	fx2=round(cost2);
	ix=cost1==fx1; cost1(ix)=cost1(ix)+0.001;
	ix=cost2==fx2; cost2(ix)=cost2(ix)+0.001;
    x1=abs(cost1-fx1);
	x2=abs(cost2-fx2);
	exp1=floor(log10(x1));
	exp2=floor(log10(x2));
	coef1=x1./(10.^exp1);
	coef2=x2./(10.^exp2);
	y= all(fx1==fx2) && all(exp1==exp2) && all(abs(coef1-coef2)<0.1);
end

function x=unifrnd(lb,ub,shape)
  if nargin==2, n=1; else n=shape(2); end 
  x=rand(1,n).*(ub-lb)+lb;
end

function samples=randsample(arr,n)
   len = length(arr);
   indices = randperm(len, n);
   samples = arr(indices);
end

%trid function as demo, alternatively you can define ObjectiveFun in the folder as well. 
function [y] = ObjectiveFun(x)
    d = length(x);
sum1 = (x(1)-1)^2;
sum2 = 0;

for ii = 2:d
	xi = x(ii);
	xold = x(ii-1);
	sum1 = sum1 + (xi-1)^2;
	sum2 = sum2 + xi*xold;
end

y = sum1 - sum2;
end