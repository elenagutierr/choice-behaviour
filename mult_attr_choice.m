%% analyse multiple-attribute choice behaviour in a value-based decision-making task 
% analyse choice behaviour in a two-attribute binary choice task with multiple stimulus sets. analyses include:
% 1. modelling session-by-session behaviour according to prospect theory
% 2. compare choice according to a strategy of maximising EV or of minimising distance to a goal 
% 3. standard behavioural analyses: psychometric curves, brainer/no brainer trials
%
% data (s_attr) is a struct with fields [1 x trials]. minimum data requirements are: 
% ID for subject, session, stimulus set, and trial (if multiple trial types), 
% option values (EV and attributes, eg. magnitude and probability), option chosen (e.g. left/right, first/second), chose correct 

% EG 25

%% set paths and load data
my_dir   = 'C:\Users\elena\OneDrive\Documents\PhD\Projects\Discrete Value Space\Analysis\chapter3\';
addpath(genpath(my_dir));
m_attr = readstruct([my_dir '\data\m_attr.xml'], "FileType", "xml");

%% set constants
% dataset specifics
choice_attributes = {'magnitude', 'probability'};   
subjects   = {'A', 'T'};
stim_sets  = unique(m_attr.stimulus_set);
session_id = unique([m_attr.session' m_attr.subject'], 'rows', 'stable');

% subset the trials of interest here: 
choice_trials = 'choice_made';  

% colours
stim_colours = [255, 69, 0;255, 140, 0]./255;
attr_colours = [0, 100, 0;153, 204 47]./255;

%% prospect theory modelling
% here we test session-by-session behaviour under the lense of prospect theory in two ways:
% 1. we'd like to know what the best fitting model is on a session-by-session basis
% 2. if we fit a full model (all parameters), how do these evolve across sessions?
%
% to do this, we adapted code from "Activation and disruption of a neural mechanism for novel choice in monkeys" (Bongioanni et al, 2021)

% set constants
min_trial_n = 200;  % minimum number of trials/session to be included in the analysis
save_best_n = 5;    % save best n models 

% parameters
% 1 - eta (integration coefficient)
% 2 - beta (magnitude vs probability weight)
% 3 - alpha (magnitude distortion)
% 4 - gamma (probability distortion)
% 5 - theta (inverse temperature)
% 6 - delta (random error)
% 7 - zeta1 (constant side bias)
% 8 - zeta2 (repetition side bias)
% 9 - zeta3 (win stay lose shift side bias)
param_names = {'\eta ','\beta ','\alpha ','\gamma ','\theta ','\delta ','\zeta_1 ','\zeta_2 ','\zeta_3 '};
param_out   = {'integration coefficient', 'magnitude/probability weight', 'magnitude distortion', ...
                'probability distortion', 'choice temperature', 'random error',...
                'constant side bias', 'repetition side bias', 'WSLS side bias'};

% create all possible model-defining combinations 
% (9 columns, one per parameter, 0 if it's fixed to default value, 1 if it's a free parameter)
tot_params = length(param_names);
models     = zeros((2^tot_params),tot_params);
for x = 1:tot_params
    models(:,x) = repmat([zeros(2^(x-1),1);ones(2^(x-1),1)],2^(tot_params-x),1);
end
num_models = length(models);
num_params = sum(models(1:2^9,1:tot_params)>0,2);

% set priors - these are in line with Bongioanni et al (2019)
prior     = [1, 0.5, 1, 1, 10, 0, 0, 0, 0];
min_bound = [0, 0, 0, 0, 0, 0, -Inf, -Inf, -Inf];
max_bound = [1, 1, 1, 1, Inf, 1, Inf, Inf, Inf];
options   = optimoptions(@fmincon, 'Display', 'off');

% format data necessary (see fit_all_possible_models.m for details)
all_data_to_fit = [m_attr.left_mag', m_attr.left_prob', m_attr.right_mag', m_attr.right_prob',...
                    m_attr.chose_left', m_attr.rewarded_trial', m_attr.session', m_attr.subject'];
all_data_to_fit = all_data_to_fit(m_attr.(choice_trials)==1);   % subset trials of interest

% 1. test best-fitting model session-by-session
best_models_i      = nan(length(session_id), save_best_n);
best_models_params = nan(save_best_n, tot_params, length(session_id));
best_models_bic    = nan(length(session_id), save_best_n);

for i = 1:length(session_id)
    data_to_fit = all_data_to_fit(all_data_to_fit(:,7)==session_id(i,1) & all_data_to_fit(:,8)==session_id(i,2), :);
    n_trials = length(data_to_fit);

    if n_trials < min_trial_n    % exclude the session if minimum trials aren't reached
        continue,
    end
    
    all_param_fits = nan(num_models, tot_params);
    all_neg_NLL    = nan(num_models,1);
    
    for ind = 1:num_models      
        min_param = prior;
        min_param(logical(models(ind, 1:tot_params))) = min_bound(logical(models(ind, 1:tot_params))); 
        max_param = prior;
        max_param(logical(models(ind, 1:tot_params))) = max_bound(logical(models(ind, 1:tot_params)));

        [all_param_fits(ind,:), all_neg_NLL(ind)]=fmincon(@(params)fit_all_possible_models(params,data_to_fit), prior, [], [], [], [], min_param, max_param, [], options); 
    end
      
    % save results from best fitting models
    BIC = log(n_trials)*(num_params)+2*neg_NLL;
    BIC(:,2) = (1:512)';
    sort_BIC = sortrows([BIC (1:num_models)'],1);
    best_ind = sort_BIC(1:save_best_n,2);
    best_models_i(i,:)   = best_ind;
    best_models_bic(i,:) = BIC(best_ind,1);
    best_models_params(:,:,i) = param_fits(best_ind,:);
    disp([i ' of ' length(session_id)])
end

% 2. run full model session-by-session
param_fits = nan(length(sessionIDs), tot_params); % parameter fits
neg_nll    = nan(length(sessionIDs),1);           % negative log-likelihood (badness of fit)
parfor i = 1:length(sessionIDs)
    
    data_to_fit = all_data_to_fit(all_data_to_fit(:,7)==session_id(i,1) & all_data_to_fit(:,8)==session_id(i,2), :);

    if length(data_to_fit) < min_trial_n    % exclude the session if minimum trials aren't reached
        continue,
    end

    min_param = prior;
    max_param = prior;
    min_param(logical(models(end,1:tot_params))) = min_bound(logical(models(end,1:tot_params))); % set min and max to be equal to the prior if that specific parameter is not tested
    max_param(logical(models(end,1:tot_params))) = max_bound(logical(models(end,1:tot_params)));


    [param_fits(i,:), neg_nll(i)]=fmincon(@(params)fit_all_possible_models(params, data_to_fit), prior, [], [], [], [], min_param, max_param, [], options); 
    disp([i ' of ' length(session_id)])
end
