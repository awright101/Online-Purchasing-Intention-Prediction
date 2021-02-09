%MACHINE LEARNING COURSEWORK
%PREDICTING ONLINE SHOPPING PURCHASING INTENTION 
%AUSTIN WRIGHT

%FUNCTION: LOGISTIC REGRESSION

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [wout,bout,cost_out] = logistic_Regression_AW(trainx,trainy,alpha,num_iters)
%LOGISTIC_REGRESSION_AW Summary of this function goes here
%   Detailed explanation goes here


%Num input predictors
num_features = size(trainx,2);

%Initating the weights for logistic regression params
w1 = randn(num_features,1);
b1 = 0;

%Initating array for storing cost vals
cost_store = zeros(num_iters,1);


%LOOPING THROUGH NUMBER OF ITERATIONS
 for iters = 1:num_iters

     
    %CALCULATING COST AND GRADIENTS FOR THAT ITERATION
    [dw,db,cost] = for_back_prop(w1,b1,trainx,trainy);
    
    
    
   %UPDATING WEIGHTS WITH ADAM, NOT BIAS TERMS
    w1 = w1 - alpha*dw; 
    b1 = b1 - alpha*db;
    cost_store(iters) = cost; 
    
    
    
%     if mod(iters,10)==0
%         disp(cost)
%     end
    
    
%     if cost_store(iters)/cost_store(1) < 0.3 && entry ==  0
%         alpha = alpha * 0.1;
%         entry = 1;
%     end


 end

%OUTPUTS
wout     = w1;
bout     = b1;
cost_out = cost_store;







%FUNCTION FOR GRADIENT DESCENT AND FORWARD CALCULATION OF SIGMOID RESPONSE
function [dw,db,cost] = for_back_prop(w,b,X,Y)

m    = length(X); %[1,1]


%LINEAR RESPONSE
flin = sum(X.*repmat(w',length(X),1),2)+b;

%SIGMOID RESPONSE
A    =  (1 ./ (1 + exp(-1.*flin))); % [length(x),1]


%cost = (-1 / m) * sum(Y .* log(A) + (1 - Y) .* (log(1 - A))) ;

%DUE TO NANS IN CROSS BINARY ENTROPY, OUTPUT COST AS MSE
cost =  (1 / m) * sum(sqrt((Y-A).^2));



%UPDATING GRADIENTS, D(COST)/DSIGMOID(DW)
dw   = ((1/m) * sum(X.* repmat((A-Y),1,size(X,2)),1))';
%UPDATING GRADIENTS, D(COST)/DSIGMOID(DB)
db   = (1/m) * sum(A-Y);

end 



end

