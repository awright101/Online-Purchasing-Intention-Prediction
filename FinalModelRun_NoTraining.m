%MACHINE LEARNING COURSEWORK
%PREDICTING ONLINE SHOPPING PURCHASING INTENTION 
%AUSTIN WRIGHT

%SCRIPT: FOR CALCULATING MODEL PREFORMANCE ON TRAINING DATA 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all 
clear 

rng(123)


X = readmatrix('DataX.csv');
Y = readmatrix('DataY.csv');

%load in the final models
%They have been output from ML_Coursework_Master_AW.m
load('FinalLogisitcRegressionModel.mat')
load('FinalRandomForestModel.mat')



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
D           = [X Y];
numD        = length(D);
numtrain    = (numD *.80);
p           = randperm(numD); %Generating random permuatations of the row numbrs
D_rand      = D(p,:);         %Creating array D_rand with random row permutations of D
Dtest       = D_rand(end-(numD-numtrain-1):end,:); %Test set 
Dtrain      = D_rand(1:numtrain,:);                %Training Set


trainx      = Dtrain(:,1:end-1); %Predictors (training)
trainy      = Dtrain(:,end);     %Targets    (training)
testx       = Dtest(:,1:end-1);  %Predictors (validation)
testy       = Dtest(:,end);      %Targets    (validation)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





%PREDICTIONS USING RANDOM FOREST
yhat = predict(B,testx);
yhat = cellfun(@str2num,yhat);
acc = sum(yhat == testy)/length(testy);

%METRICS
[TP,FP,TN,FN] = FPR(testy,yhat);
f1_score_test = TP / ( TP + (FP + FN))/2;
balanced_acc = 0.5 * ((TP/(TP+FN))+(TN/(TN+FP)));

disp('RANDOM FOREST')
disp(strcat('Test set accuracy score:-> ',num2str(acc)))
disp(strcat('Test set balanced accuracy score:-> ',num2str(balanced_acc)))
disp(strcat('Test set F1 :-> ',num2str(f1_score_test)))



%PREDICTIONS USING LOGISTIC REGRESSION
acc = sum(predict_AW(testx,wout,bout) == testy)/length(testy);
[TP,FP,TN,FN] = FPR(testy,predict_AW(testx,wout,bout));
f1_score_test = TP / ( TP + (FP + FN))/2 ; 
balanced_acc = 0.5 * ((TP/(TP+FN))+(TN/(TN+FP)));
disp(' ')
disp('LOGISTIC REGRESSION')
disp(strcat('Test set accuracy score:-> ',num2str(acc)))
disp(strcat('Test set balanced accuracy score:-> ',num2str(balanced_acc)))
disp(strcat('Test set F1 :-> ',num2str(f1_score_test)))






%CALCULATING LINEAR RESPONSE
flin = sum(testx.*repmat(wout',length(testx),1),2)+bout;

%SIGMOID 
A    =  (1 ./ (1 + exp(-1.*flin))); %

%Built in function from matlab perfcurve for ROC analysis:

%Logisitic Regression 
[Xlog,Ylog,T,AUC] = perfcurve(testy,A,1);

%For the Random Forest
[Yfit,scores] = predict(B,testx);
[Xrf,Yrf,Trf,AUCrf] = perfcurve(testy,scores(:,2),1);


%Plotting ROC Curve of models and a random classifier 
figure(1)
plot(Xlog,Ylog)
hold on
plot(Xrf,Yrf)
hold on
plot([0 1],[0 1])
hold on
legend('Logistic Regression','Random Forest','Random Classification','Location','southeast')
xlabel('False positive rate'); ylabel('True positive rate');
%title('ROC Curves for Logistic Regression and Random Forest Classification')
set(gca, 'FontSize', 18, 'LineWidth', 0.5)
grid on
hold off

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
%linear response of model 
flin = sum(X.*repmat(wf',length(X),1),2)+bf;

%SIGMOID ACTIVATION
A    =  (1 ./ (1 + exp(-1.*flin))); % [length(x),1]


%ARGMAX
yhat = A>0.5;


end
