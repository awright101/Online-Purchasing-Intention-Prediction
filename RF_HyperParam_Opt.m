%MACHINE LEARNING COURSEWORK
%PREDICTING ONLINE SHOPPING PURCHASING INTENTION 
%AUSTIN WRIGHT

%SCRIPT: BAYESIAN OPTIMIZATION OF RANDOM FOREST HYPERPARAMETERS 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear
clf
close all

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
NumTrees = optimizableVariable('t',[1,500],'Type','integer');
NumPredictors = optimizableVariable('p',[1,16],'Type','integer');


%FUNCTION WE ARE GOING TO OPTIMIZE
fun = @(x)RF_FORBAYESOPT(trainx,trainy,testx,testy,x.t,x.p);





%OPTIMIZING WITH BAYESOPT
results_RF_opto = bayesopt(fun,[NumTrees,NumPredictors],'Verbose',1,...
    'AcquisitionFunctionName','expected-improvement-plus');

%%params = table2array(results_RF_opto.XAtMinObjective);
%params = [88 3];
%save('bayes_opt_RF-PARAMS.mat','params');




function RTM = RF_FORBAYESOPT(trainx,trainy,testx,testy,NumTrees,NumPredictors)

% RTM result to maximize 
%RANDOM FOREST CODE FOR OPTIMIZATION:
B = TreeBagger(NumTrees,trainx,trainy,'OOBPredictorImportance','On',...
'Method','classification','NumPredictorsToSample',NumPredictors,'Reproducible',true);
%acc = sum(predict_AW(testx,wout,bout) == testy)/length(testy);

%METRICS
yhat = predict(B,testx);
yhat = cellfun(@str2num,yhat);
acc = sum(yhat == testy)/length(testy);

[TP,FP,TN,FN] = FPR(testy,yhat);
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