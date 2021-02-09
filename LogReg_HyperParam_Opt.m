%MACHINE LEARNING COURSEWORK
%PREDICTING ONLINE SHOPPING PURCHASING INTENTION 
%AUSTIN WRIGHT

%SCRIPT: BAYESIAN OPTIMIZATION OF LOGISITIC REGRESSION HYPERPARAMETERS 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear
close all
clf

X = readmatrix('DataX.csv');
Y = readmatrix('DataY.csv');
rng(123)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%SELECTION OUR 80:20 TRAIN.TEST SPLIT FOR TRAINING AND VALIDATION IN THE
%HYPERPARAMER OPTIMIZATION
D           = [X Y];
numD        = length(D);
numtrain    = (numD *.80);
p           = randperm(numD); %Generating random permuatations of the row numbrs
D_rand      = D(p,:); %Creating array D_rand with random row permutations of D
Dtest       = D_rand(end-(numD-numtrain-1):end,:); %Test set 
Dtrain      = D_rand(1:numtrain,:);


trainx      = Dtrain(:,1:end-1);
trainy      = Dtrain(:,end);
testx       = Dtest(:,1:end-1);
testy       = Dtest(:,end);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%SETTING UP HYPERPARAMETERS TO OPTIMIZE

%ADAM B1
beta1 = optimizableVariable('b1',[0,1],'Type','real');
%ADAM B2
beta2 = optimizableVariable('b2',[0,1],'Type','real');
%ALPHA, LEARNING RATE
alpha = optimizableVariable('a',[0.0001,0.1],'Type','real');
%ITERS, NUMBER OF ITERATIONS
iters = optimizableVariable('it',[1,10000],'Type','integer');

%FUNCTION WE ARE GOING TO OPTIMIZE
fun = @(x)LOGREG_FORBAYESOPT(trainx,trainy,testx,testy,[x.b1 x.b2],x.a,x.it);

% set(gca, 'XScale', 'log')
tic
%OPTIMIZING WITH BAYESOPT
results_logreg = bayesopt(fun,[beta1,beta2,alpha,iters],'Verbose',1,...
    'AcquisitionFunctionName','expected-improvement-plus');
toc

% params = table2array(results_logreg.XAtMinObjective);
% save('bayes_opt_logregPARAMS.mat','params');



function RTM = LOGREG_FORBAYESOPT(trainx,trainy,testx,testy,ADAM_COF,alpha,num_iters)

% RTM result to maximize 
%LOGISITC REGRESSION FUNCTION:
[wout,bout,cost_out2] = LogRegression_ADAM(trainx,trainy,ADAM_COF,alpha,num_iters);
%METRICS
acc = sum(predict_AW(testx,wout,bout) == testy)/length(testy);
[TP,FP,TN,FN] = FPR(testy,predict_AW(testx,wout,bout));
f1_score_test = TP / ( TP + (FP + FN))/2 ; 
%balanced_acc = 0.5 * ((TP/(TP+FN))+(TN/(TN+FP)));

%SETTING OBJECTIVE FUNCTION FOR BAYESOPT = 1-F1SCIRE
RTM = (1-f1_score_test);
end



% % if mod(alpha_scan,10)== 
% disp(strcat('Test set acc:-> ',num2str(acc)))
% disp(strcat('Test set balanced accuracy score:-> ',num2str(balanced_acc)))
% disp(strcat('Test set F1 :-> ',num2str(f1_score_test)))
% % end









function [TP,FP,TN,FN] = FPR(ytrue,ypred)
TP=0;FP=0;TN=0;FN=0;
          for i=1:length(ytrue)
              if(ytrue(i)==1 & ypred(i)==1)
                  TP=TP+1;
              elseif(ytrue(i)==0 & ypred(i)==1)
                  FP=FP+1;
              elseif(ytrue(i)==0 & ypred(i)==0)
                  TN=TN+1;
              else
                  FN=FN+1;
              end
          end
end
      

function [yhat] = predict_AW(X,wf,bf)

flin = sum(X.*repmat(wf',length(X),1),2)+bf;
A    =  (1 ./ (1 + exp(-1.*flin))); % [length(x),1]



yhat = A>0.5;


end