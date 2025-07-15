function badness_of_fit=fit_all_possible_models(params,data_to_fit)
% adapted from code for "Activation and disruption of a neural mechanism for novel choice in monkeys" (Bongioanni et al, 2021)
% original code and data available at https://osf.io/kzhaq/
% INPUT
%   params = model parameters:
%       1 - eta (integration coefficient)
%       2 - beta (magnitude vs probability weight)
%       3 - alpha (magnitude distortion)
%       4 - gamma (probability distortion)
%       5 - theta (inverse temperature)
%       6 - delta (random error)
%       7 - zeta1 (constant side bias)
%       8 - zeta2 (repetition side bias)
%       9 - zeta3 (win-stay-lose-shift side bias)
%
%   data_to_fit:
%       1 - left/option 1 magnitude,   2 - left/option 1 probability
%       3 - right/option 2 magnitude,  4 - right/option 2 probability
%       5 - chosen option,             6 - reward/no reward
%       7 - session
% OUPUT
%   badness_of_fit = negative log likelihood
% EG 24


% ensure magnitude and probability parameters are normalised between 0 and 1
data_to_fit(:,1)=normalize_bound(data_to_fit(:,1), 0.1, 1); 
data_to_fit(:,2)=normalize_bound(data_to_fit(:,2), 0.1, 1); 
data_to_fit(:,3)=normalize_bound(data_to_fit(:,3), 0.1, 1);
data_to_fit(:,4)=normalize_bound(data_to_fit(:,4), 0.1, 1);

% magnitude and probability distortions
magnitude=data_to_fit(:,[1,3]).^params(3);
probability=exp(-(-log(data_to_fit(:,[2,4]))).^params(4));

% 3 types of side bias
if length(params) > 6
    prev_choice=2*([0.5;data_to_fit(1:end-1,5)])-1; % code as +1/-1
    prev_win=2*([0.5;data_to_fit(1:end-1,6)])-1;    % code as +1/-1
    wsls=prev_choice.*prev_win;                     % win-stay-lose-shift
    prev_choice(logical([0;data_to_fit(2:end,7)~=data_to_fit(1:end-1,7)]))=0;   % remove effects from first trial of each session
    wsls(logical([0;data_to_fit(2:end,7)~=data_to_fit(1:end-1,7)]))=0;          % remove effects from first trial of each session
    sidebias=params(7)+params(8)*prev_choice+params(9)*wsls;
else
    sidebias = 0;
end

% determine subjective values
sub_val_left= (params(1) * magnitude(:,1).*probability(:,1)) + ((1-params(1)) * ((params(2).*magnitude(:,1))+((1-params(2)).*probability(:,1))));
sub_val_right= (params(1) * magnitude(:,2).*probability(:,2)) + ((1-params(1)) * ((params(2).*magnitude(:,2))+((1-params(2)).*probability(:,2)))); 

% predict choice
softmax_right=1./(1+exp(-params(5).*(sub_val_right-sub_val_left+sidebias)));
p_right=(softmax_right*(1-params(6)))+(params(6)/2); % estimated probability of choosing right    
p_right(p_right==1)=0.999999;                        % avoid log(0) 
p_right(p_right==0)=0.000001; 

% Measure badness of fit 
badness_of_fit=-(sum(log(double(p_right(data_to_fit(:,5)==0))))+sum(log(1-p_right(data_to_fit(:,5)==1))));
end

