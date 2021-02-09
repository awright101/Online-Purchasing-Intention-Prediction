%MACHINE LEARNING COURSEWORK
%PREDICTING ONLINE SHOPPING PURCHASING INTENTION 
%AUSTIN WRIGHT

%FUNCTION: FORWARD CALCULATION OF LOGISITIC REGRESSION RESPONSE WITH
%CURRENT MODEL WEIGHTS

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [yhat,probz] = predict_AW(X,wf,bf)

flin = sum(X.*repmat(wf',length(X),1),2)+bf;
A    =  (1 ./ (1 + exp(-1.*flin))); % [length(x),1]

probz = A;


yhat = zeros(length(X),1);

%ARGMAX 
for J = 1:length(A)
    
    if A(J) >= 0.5
        yhat(J) = 1;
    elseif A(J) < 0.5
        yhat(J) = 0;
    end
end
        

end