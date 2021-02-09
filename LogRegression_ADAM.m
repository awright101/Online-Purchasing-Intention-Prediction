
%MACHINE LEARNING COURSEWORK
%PREDICTING ONLINE SHOPPING PURCHASING INTENTION 
%AUSTIN WRIGHT

%FUNCTION: LOGISTIC REGRESSION WITH ADAM OPT. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [wout,bout,cost_out] = logistic_Regression_ADAM(trainx,trainy,ADAM_COF,alpha,num_iters)
%LOGISTIC_REGRESSION using ADAM optimization


%Num input predictors
num_features = size(trainx,2);

%Initating the weights for logistic regression params
w1         = randn(num_features,1);
b1         = 0;


%Initating array for storing cost vals
cost_store = zeros(num_iters,1);

%Initation of params for ADAM
M0_w       =zeros(num_features,1);
V0_w       =zeros(num_features,1);

% M0_b       =zeros(1,1);
% V0_b       =zeros(1,1);

%INPUT PARAMETERS FOR ADAM
B1         =ADAM_COF(1);
B2         =ADAM_COF(2);
E          = 10^-8;

%LOOPING THROUGH NUMBER OF ITERATIONS
 for iters = 1:num_iters

     %CALCULATING COST AND GRADIENTS FOR THAT ITERATION
    [dw,db,cost] = for_back_prop(w1,b1,trainx,trainy);
    
    
 
    %ADAM PARAM UPDATING BASED ON MOMENT OF GRADIENTS
    Mw_curr = (B1 * M0_w) + (1 - B1) * dw;
    Vw_curr = (B2 * V0_w) + (1 - B2) * dw.^2;
    Mw_hat  = Mw_curr ./ (1-B1^iters);
    Vw_hat   = Vw_curr ./ (1-B2^iters);
    
    
%     Mb_curr = (B1 * M0_b) + (1 - B1) * db;
%     Vb_curr = (B2 * V0_b) + (1 - B2) * db^2;
%     Mb_hat  = Mb_curr / (1-B1^iters);
%     Vb_hat   = Vb_curr / (1-B2^iters);
    
    
    %UPDATING WEIGHTS WITH ADAM, NOT BIAS TERMS
    w1 = w1 - alpha*Mw_hat./(sqrt(Vw_hat) + E); 
    %b1 = b1 - alpha*Mb_hat/(sqrt(Vb_hat) + E);
    b1 = b1 - alpha*db;
    cost_store(iters) = cost; 
    
    
    
    
    %STORING CURRENT ITERATIONS PARAMS FROM ADAM FOR USE IN NEXT ITERATION
    M0_w  = Mw_curr;
    V0_w  = Vw_curr;
%     M0_b  = Mb_curr;
%     M0_b  = Vb_curr;
%     
    
    
    
    

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
cost =  (1 / m) * sum((Y-A).^2);



%UPDATING GRADIENTS, D(COST)/DSIGMOID(DW)
dw   = ((1/m) * sum(X.* repmat((A-Y),1,size(X,2)),1))';

%UPDATING GRADIENTS, D(COST)/DSIGMOID(DB)
db   = (1/m) * sum(A-Y);

end 



end


