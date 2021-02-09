
%MACHINE LEARNING COURSEWORK
%PREDICTING ONLINE SHOPPING PURCHASING INTENTION 
%AUSTIN WRIGHT

%SCRIPT: MASTER SCRIPT FOR TRAINING THE MODEL AND 3 SEPERATE MODES

%           1) Oversample the training set -> Equal Positive/Negative class
%           2) K Fold cross validation
%           3) Final Model Training/Validation 
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




clear 
close all 
clf


X = readmatrix('DataX.csv');
Y = readmatrix('DataY.csv');



disp(' ') 
disp(' Multiple options:')
disp(' 1. Oversample, Equal Positives + Negatives') 
disp(' 2. K-fold Cross Validation')
disp(' 3. Final Model Run')
disp('~~~~~~~~~~~~~~~~~~~ ') 
rng(123)

option  = input('Option 1,2,3..?');
verbose = 1;

switch option 

    
    case 1 %Option 0 initates the oversampling workflow
disp('Balancing Dataset with oversampling')
%%% Generating data for oversampling
positives   = [X(Y==1,:) Y(Y==1)]; 
negatives   = [X(Y==0,:) Y(Y==0)];

%Equal Num of positive and negative instances in training 80/20
trainequal  = [positives(1:1525,:);negatives(1:1525,:)]; 
testequal   =  [positives(1526:end,:);negatives(1526:length(positives),:)]; 


%Shuffling
trainequal  = trainequal(randperm(length(trainequal)),:); 
testequal   = testequal(randperm(length(testequal)),:);

%Selecting the predictors and the targets
trainequal_x = trainequal(:,1:16);
trainequal_y = trainequal(:,17);

%Selecting the predictors and the targets
testequal_x = testequal(:,1:16);
testequal_y = testequal(:,17);


trainx      = trainequal_x;
trainy      = trainequal_y;
testx       = testequal_x;
testy       = testequal_y;

if verbose == 1
% logistic regression code + ADAM
disp('Logisitic Regression with ADAM Optimization')
disp('')
disp('')
end

if verbose == 1
tic
end

%Loading params in from bayesian optimized hyperparameters that have been
%saved 
params_logreg = cell2mat(struct2cell(load('bayes_opt_logregPARAMS.mat')));
ADAM_COF  = [params_logreg(1) params_logreg(2)];
alpha     = params_logreg(3);
%Fixing num iters due to lack of computational challenge
num_iters = 10000;%params_logreg(4);


%disp(params_logreg)

%Running the Logistic Regression with ADAM
[wout,bout,cost_out2] = LogRegression_ADAM(trainx,trainy,ADAM_COF,alpha,num_iters);

if verbose == 1
toc
end

%Model Metrics
acc = sum(predict_AW(testx,wout,bout) == testy)/length(testy);

%Calculating the number of TP, FP, TN and FN's
[TP,FP,TN,FN] = FPR(testy,predict_AW(testx,wout,bout));

%F1 Score
f1_score_test = TP / ( TP + (FP + FN))/2 ; 

%Balanced Accuracy
balanced_acc = 0.5 * ((TP/(TP+FN))+(TN/(TN+FP)));

if verbose == 1
disp(strcat('Test set acc:-> ',num2str(acc)))
disp(strcat('Test set balanced accuracy score:-> ',num2str(balanced_acc)))
disp(strcat('Test set F1 :-> ',num2str(f1_score_test)))
end




if verbose == 1
% Random Forrest
disp(' ')
disp(' ')
disp('RANDOM FOREST')
disp('')
disp('')
end

%Loading in the model parameters as ouput from Bayesopt run for Random
%Forest
params_RF    = cell2mat(struct2cell(load('bayes_opt_RF-PARAMS.mat')));
NumTrees     = params_RF(1) ;
NumPredictors = params_RF(2);

if verbose == 1
tic
end



%Running Random Forest code using the treebagger implementation
B = TreeBagger(NumTrees,trainx,trainy,'OOBPredictorImportance','On',...
'Method','classification','NumPredictorsToSample',NumPredictors,'Reproducible',true);



if verbose == 1
toc
end

%Extracting from the cell array the estiamtes of class on test data (yhat)
yhat = predict(B,testx);
yhat = cellfun(@str2num,yhat);
acc = sum(yhat == testy)/length(testy);


%Metrics (see above)
[TP,FP,TN,FN] = FPR(testy,yhat);
f1_score_test = TP / ( TP + (FP + FN))/2;
balanced_acc = 0.5 * ((TP/(TP+FN))+(TN/(TN+FP)));


if verbose == 1
disp(strcat('Test set accuracy score:-> ',num2str(acc)))
disp(strcat('Test set balanced accuracy score:-> ',num2str(balanced_acc)))
disp(strcat('Test set F1 :-> ',num2str(f1_score_test)))
end




%%% Plotting 

%Plot num iters vs cost
figure(1)
plot(1:num_iters,cost_out2,'-r','LineWidth',1)
set(gca, 'FontSize', 18, 'LineWidth', 0.5);
grid on%


%Plot OOB error vs num of trees in the forest
figure(2);
oobErrorBaggedEnsemble = oobError(B);
plot(oobErrorBaggedEnsemble,'-r','LineWidth',1)
set(gca, 'FontSize', 18, 'LineWidth', 0.5);
grid on%
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';


%Plot predictor feature importance
figure(3)
bar(B.OOBPermutedPredictorDeltaError)
xlabel('Feature Index')
ylabel('Out-of-Bag Feature Importance')



    case 2 %Cross validation case

      

       

    
        
disp('K-FOLD CROSS VAL')        
D           = [X Y];
numD        = length(D);
        
%Defining K, the cross validation fold        
k           = 5;

disp(strcat(num2str(k),'-','Fold Cross Val'))


%Size of groups to split the data into from crossvalidation
groupz = numD/k;

%number of bins
inds      = zeros(k+1,1);




%2:K+1 = K
for ii = 2:k+1
    
    
    inds(ii) = groupz*(ii-1);

end

 inds(1,1) = 1;
% disp(inds)
        

%Shuffling the data
p           = randperm(numD);   
D_rand      = D(p,:);


%Pre generating arrays to store the accuracy of the model runs
RF_ACC = zeros(k,1);
LR_ACC = zeros(k,1);

for idx = 2:length(inds)
    
disp(strcat(num2str(idx-1),'- group as validation set'))

%For each CV run we have a different subset of the data as test (thus why
%its done in a loop!
Dtest       = D_rand(inds(idx-1):inds(idx),:); 
Dtrain      = D_rand(setdiff(inds(1):inds(end),inds(idx-1):inds(idx)),:);

trainx      = Dtrain(:,1:end-1);
trainy      = Dtrain(:,end);
testx       = Dtest(:,1:end-1);
testy       = Dtest(:,end);  
    


disp('    ')    
if verbose == 1
disp(strcat('Number of Positives in test set: ',num2str(sum(testy))))   
end
disp('    ')    
   
    
    
    
    
% logistic regression code
% disp('')
% disp('')
% disp('Logisitic Regression')
% disp('')
% disp('')
% 
% tic
 %[wout,bout,cost_out] = logistic_Regression_AW(trainx,trainy,0.001,6000);
% toc
% 
% acc = sum(predict_AW(testx,wout,bout) == testy)/length(testy)
% 
% 
%  
% [TP,FP,TN,FN] = FPR(testy,predict_AW(testx,wout,bout));
% f1_score_test = TP / ( TP + (FP + FN))/2  
% disp('')
% disp('~~~~~~~~~~~~~~~~~~')



if verbose == 1
% logistic regression code + ADAM
disp('Logisitic Regression with ADAM Optimization')
disp('')
disp('')
end

%for i = 0:0.1:1


%disp(ADAM_COF(1))
if verbose == 1
tic
end


%Loading the params for RF from Bayesopt stored data!
params_logreg = cell2mat(struct2cell(load('bayes_opt_logregPARAMS.mat')));
ADAM_COF  = [params_logreg(1) params_logreg(2)];
alpha     = params_logreg(3);
num_iters = 10000;%params_logreg(4);

%Running Logreg
[wout,bout,cost_out2] = LogRegression_ADAM(trainx,trainy,ADAM_COF,alpha,num_iters);

if verbose == 1
toc
end

acc = sum(predict_AW(testx,wout,bout) == testy)/length(testy);


 
[TP,FP,TN,FN] = FPR(testy,predict_AW(testx,wout,bout));
f1_score_test = TP / ( TP + (FP + FN))/2 ; 
balanced_acc = 0.5 * ((TP/(TP+FN))+(TN/(TN+FP)));

%Storing the accuracy from logistic regression for this run of CV
LR_ACC(idx-1) = acc;

if verbose == 1
disp(strcat('Test set acc:-> ',num2str(acc)))
disp(strcat('Test set balanced accuracy score:-> ',num2str(balanced_acc)))
disp(strcat('Test set F1 :-> ',num2str(f1_score_test)))
end

%Loading the params for RF from stored params from Bayesopt
params_RF    = cell2mat(struct2cell(load('bayes_opt_RF-PARAMS.mat')));
NumTrees     =params_RF(1) ;
NumPredictors = params_RF(2);


if verbose == 1
tic
end

%Running random forest using Treebagger
B = TreeBagger(NumTrees,trainx,trainy,'OOBPredictorImportance','On',...
'Method','classification','NumPredictorsToSample',NumPredictors,'Reproducible',true);


if verbose == 1
toc
end

%Extracting the prediction of RF on the test set
yhat = predict(B,testx);
yhat = cellfun(@str2num,yhat);
acc = sum(yhat == testy)/length(testy);

[TP,FP,TN,FN] = FPR(testy,yhat);
f1_score_test = TP / ( TP + (FP + FN))/2;
balanced_acc = 0.5 * ((TP/(TP+FN))+(TN/(TN+FP)));

%Storing the accuracy of the RF Classifier 
RF_ACC(idx-1) = acc;

if verbose == 1
disp(strcat('Test set accuracy score:-> ',num2str(acc)))
disp(strcat('Test set balanced accuracy score:-> ',num2str(balanced_acc)))
disp(strcat('Test set F1 :-> ',num2str(f1_score_test)))
end




%%% Plotting 

%Plot Cost vs. number of iterations 
figure(1)
plot(cost_out2,'LineWidth',3)
set(gca, 'FontSize', 18, 'LineWidth', 0.5);
grid on%
xlabel('Number of Iterations')
ylabel('MSE')
title(strcat('Cost vs Iters',{' '},num2str(k),'-Fold CV'))
hold on


%Plotting error vs number of trees in the forest
figure(2);
oobErrorBaggedEnsemble = oobError(B);
plot(oobErrorBaggedEnsemble,'-r','LineWidth',1)
set(gca, 'FontSize', 18, 'LineWidth', 0.5);
grid on%
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';
hold on


end

%Displaying the results of averaged accuracy after K fold CV for each
%classifier 
disp('  ');
disp('Averaged accuracy for LR after CV: ')
disp(mean(LR_ACC));
disp('  ');disp('  ');
disp('Averaged accuracy for RF after CV: ')
disp(mean(RF_ACC));




    case 3 %Final Model Run case
        
        
D           = [X Y];
numD        = length(D);

%Selecting 80/20 train test split
numtrain    = (numD *.80); 

%Generating random permuatations of the row numbrs
p           = randperm(numD); 
D_rand      = D(p,:); %Creating array D_rand with random row permutations of D

%Selecting test and train
Dtest       = D_rand(end-(numD-numtrain-1):end,:); 
Dtrain      = D_rand(1:numtrain,:);


trainx      = Dtrain(:,1:end-1);
trainy      = Dtrain(:,end);
testx       = Dtest(:,1:end-1);
testy       = Dtest(:,end);

%Droppig Feature 9 which is page value to see how it effects model
%preformance 

% trainx(:,9) = [] ;
% testx(:,9)  = [];






if verbose == 1
% logistic regression code + ADAM
disp('Logisitic Regression with ADAM Optimization')
disp('')
disp('')
end

%for i = 0:0.1:1


%Running LR without adam (Much lower preforming implementation - AVOID)


% tic
%[wout,bout,cost_out] = logistic_Regression_AW(trainx,trainy,0.01,6000);
% toc
% 
% acc = sum(predict_AW(testx,wout,bout) == testy)/length(testy)
% 
% 
%  
% [TP,FP,TN,FN] = FPR(testy,predict_AW(testx,wout,bout));
% f1_score_test = TP / ( TP + (FP + FN))/2 ;
% balanced_acc = 0.5 * ((TP/(TP+FN))+(TN/(TN+FP)));

% disp('')
% disp('~~~~~~~~~~~~~~~~~~')



%disp(ADAM_COF(1))
if verbose == 1
tic
end

%Loading model params
params_logreg = cell2mat(struct2cell(load('bayes_opt_logregPARAMS.mat')));

%Adam specific variables
ADAM_COF  = [params_logreg(1) params_logreg(2)];
alpha     = params_logreg(3);

%Fixing num iters due to low computation demand
num_iters = 10000;%params_logreg(4);


%Running LR
[wout,bout,cost_out2] = LogRegression_ADAM(trainx,trainy,ADAM_COF,alpha,num_iters);

if verbose == 1
toc
end

%Metrics 
acc = sum(predict_AW(testx,wout,bout) == testy)/length(testy);
[TP,FP,TN,FN] = FPR(testy,predict_AW(testx,wout,bout));
f1_score_test = TP / ( TP + (FP + FN))/2 ; 
balanced_acc = 0.5 * ((TP/(TP+FN))+(TN/(TN+FP)));


%confusion_matplt(TP,TN,FP,FN)

if verbose == 1
disp(strcat('Test set acc:-> ',num2str(acc)))
disp(strcat('Test set balanced accuracy score:-> ',num2str(balanced_acc)))
disp(strcat('Test set F1 :-> ',num2str(f1_score_test)))

%Preformance on training set
disp('Final Training Acc.')
disp(sum(Predict_AW(trainx,wout,bout)==trainy)/length(trainy))

end

if verbose == 1
% Random Forrest
disp(' ')
disp(' ')
disp('RANDOM FOREST')
disp('')
disp('')
end

%Loading data from Bayesopt 
params_RF    = cell2mat(struct2cell(load('bayes_opt_RF-PARAMS.mat')));
NumTrees     = params_RF(1) ;
NumPredictors = params_RF(2);


if verbose == 1
tic
end

%Treebagger Final Model Training run
B = TreeBagger(NumTrees,trainx,trainy,'OOBPredictorImportance','On',...
'Method','classification','NumPredictorsToSample',NumPredictors,'Reproducible',true);

%confusion_matplt(TP,TN,FP,FN)
% 

%Viewing the first tree from the treeplot
%view(B.Trees{1},'Mode','graph')

if verbose == 1
toc
end

%Calculating validation classification 
yhat = predict(B,testx);
yhat = cellfun(@str2num,yhat);
acc = sum(yhat == testy)/length(testy);

[TP,FP,TN,FN] = FPR(testy,yhat);
f1_score_test = TP / ( TP + (FP + FN))/2;
balanced_acc = 0.5 * ((TP/(TP+FN))+(TN/(TN+FP)));


if verbose == 1
disp(strcat('Test set accuracy score:-> ',num2str(acc)))
disp(strcat('Test set balanced accuracy score:-> ',num2str(balanced_acc)))
disp(strcat('Test set F1 :-> ',num2str(f1_score_test)))

%training accuracy of 
yhat_train = predict(B,trainx);
yhat_train = cellfun(@str2num,yhat_train);
acc_train = sum(yhat_train == trainy)/length(trainy);

disp('Final Training Acc.')
disp(acc_train);

end




%%% Plotting 

%Plotting COST VS NUM ITERS
figure(get(groot,'CurrentFigure').Number +1)
plot(1:num_iters,cost_out2,'-r','LineWidth',1)
% semilogx(1:num_iters,cost_out2,'-r','LineWidth',1)
% hold on
% semilogx(1:num_iters,cost_out,'-b','LineWidth',1)
set(gca, 'FontSize', 18, 'LineWidth', 0.5); %<- Set properties
xlabel('Number of Iterations')
ylabel('MSE')
grid on
% legend('LogReg w Adam','LogReg')

%PLOTTING OOB ERROR VS NUMBER 
figure(get(groot,'CurrentFigure').Number +1);
oobErrorBaggedEnsemble = oobError(B);
plot(oobErrorBaggedEnsemble,'-r','LineWidth',1)
set(gca, 'FontSize', 18, 'LineWidth', 0.5);
grid on%<- Set properties
xlabel('Number of grown trees');
ylabel('Out-of-bag classification error');

%
figure(get(groot,'CurrentFigure').Number +1)
bar(B.OOBPermutedPredictorDeltaError,'LineWidth',3)
xlabel('Feature Index')
ylabel('Out-of-Bag Feature Importance')
set(gca, 'FontSize', 18, 'LineWidth', 0.5)
grid on

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
figure(get(groot,'CurrentFigure').Number +1)
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


    otherwise 
        disp('                                ')
        disp('WRONG OPTION NUMBER RERUN SCRIPT')
        disp('                                ')
end 




%FUNCTIONS


function [TP,FP,TN,FN] = FPR(ytrue,ypred)

%CALCULATING METRICS TP, FP, TN, FN
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

%SIGMOID ACTIVATION FUNCTION
A    =  (1 ./ (1 + exp(-1.*flin))); % [length(x),1]


%ARGMAX
yhat = A>0.5;


end

function pltout = confusion_matplt(TP,TN,FP,FN);
%UNUSED 
%FUNCTION TO PLOT THE CONFUSION MATRIX GIVEN NUMBER OF TP, TN FP AND FN'S
mat = [TP TN; FP FN];

fig = get(groot,'CurrentFigure').Number +1;
figure(fig)
imagesc(mat)
colormap jet;
text(1,1,strcat('TP =',{' '},num2str(TP)))
text(1,2,strcat('TN =',{' '},num2str(TN)))
text(2,1,strcat('FP =',{' '},num2str(FP)))
text(2,2,strcat('FN =',{' '},num2str(FN)))
hold on

end 
