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
session_id = unique([m_attr.session' m_attr.subject' m_attr.stimulus_set'], 'rows', 'stable');

% subset the trials of interest here: 
choice_trials = 'choice_made';  

% colours
stim_colours = [51, 102, 0;153, 76, 0]./255;
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

% format data necessary (see fit_all_possible_models.m for details) and
% exclude sessions that do not meet minimum trial number requirement
all_data_to_fit = [m_attr.left_mag', m_attr.left_prob', m_attr.right_mag', m_attr.right_prob',...
                    m_attr.chose_left', m_attr.rewarded_trial', m_attr.session', m_attr.subject'];
all_data_to_fit = all_data_to_fit(m_attr.(choice_trials)==1, :);   % subset trials of interest
sessions_to_fit = [];
for i = 1:length(session_id)
    n_trials = sum(all_data_to_fit(:,7)==session_id(i,1) & all_data_to_fit(:,8)==session_id(i,2));
    rew = sum(isnan(all_data_to_fit(all_data_to_fit(:,7)==session_id(i,1) & all_data_to_fit(:,8)==session_id(i,2), 6)));
    if n_trials >= min_trial_n && rew==0
        sessions_to_fit = [sessions_to_fit; session_id(i, :)];
    end
end

% 1. test best-fitting model session-by-session
best_models_i      = nan(length(sessions_to_fit), save_best_n);
best_models_params = nan(save_best_n, tot_params, length(sessions_to_fit));
best_models_bic    = nan(length(sessions_to_fit), save_best_n);

for i = 1:length(sessions_to_fit)
    data_to_fit = all_data_to_fit(all_data_to_fit(:,7)==sessions_to_fit(i,1) & all_data_to_fit(:,8)==sessions_to_fit(i,2), :);
    
    all_param_fits = nan(num_models, tot_params);
    all_neg_NLL    = nan(num_models,1);
    
    for ind = 1:num_models      
        % set min and max to be equal to the prior if that specific parameter is not tested
        min_param = prior;
        max_param = prior;
        min_param(logical(models(ind, 1:tot_params))) = min_bound(logical(models(ind, 1:tot_params))); 
        max_param(logical(models(ind, 1:tot_params))) = max_bound(logical(models(ind, 1:tot_params)));

        [all_param_fits(ind,:), all_neg_NLL(ind)]=fmincon(@(params)fit_all_possible_models(params,data_to_fit), prior, [], [], [], [], min_param, max_param, [], options); 
    end
      
    % save results from best fitting models
    BIC = log(n_trials)*(num_params)+2*neg_NLL;     % compute BIC
    sort_BIC = sortrows([BIC (1:num_models)'],1);
    best_ind = sort_BIC(1:save_best_n,2);
    best_models_i(i,:)   = best_ind;
    best_models_bic(i,:) = BIC(best_ind,1);
    best_models_params(:,:,i) = param_fits(best_ind,:);
    disp([i ' of ' num2str(length(sessions_to_fit)) ' sessions done'])
end

% 2. run full model session-by-session
param_fits = nan(length(sessions_to_fit), tot_params); % parameter fits
neg_nll    = nan(length(sessions_to_fit),1);           % negative log-likelihood (badness of fit)
parfor i = 1:length(sessions_to_fit)
    
    data_to_fit = all_data_to_fit(all_data_to_fit(:,7)==sessions_to_fit(i,1) & all_data_to_fit(:,8)==sessions_to_fit(i,2), :);

    % set min and max to be equal to the prior if that specific parameter is not tested
    min_param = prior;
    max_param = prior;
    min_param(logical(models(end,1:tot_params))) = min_bound(logical(models(end,1:tot_params))); 
    max_param(logical(models(end,1:tot_params))) = max_bound(logical(models(end,1:tot_params)));


    [param_fits(i,:), neg_nll(i)]=fmincon(@(params)fit_all_possible_models(params, data_to_fit), prior, [], [], [], [], min_param, max_param, [], options); 
    disp(['Session ' num2str(i) ' of ' num2str(length(sessions_to_fit)) ' done'])
end

% plot parameters across sessions
% we'd like to format this in chronological manner while also highlighting the different stimulus sets

% set plotting constants
params_to_plot = 1:5;
plot_params.Color = stim_colours;
plot_params.MarkerSize = 12;
plot_params.Legend = {'Stimulus set 1', 'Stimulus set 2'};

figure; set(gcf,'color','w');
plots = [1:length(params_to_plot); length(params_to_plot)+1:length(params_to_plot)*2];
m = length(subjects); n = length(params_to_plot);

min_p=[0,0,0,0,0,0,-1,-1,-1];   % set limits
max_p=[1,1,1,1,20,1,1,1,1];

for i = 1:length(params_to_plot)
    for subj = 1:length(subjects)
        subj_data = param_fits(sessions_to_fit(:, 2)==subj, params_to_plot(i));
        subj_sessions = sessions_to_fit(sessions_to_fit(:, 2)==subj, :);
        x = 1:length(subj_data);

        subplot(m, n, plots(subj, i)) 
        scatter(x, subj_data, plot_params.MarkerSize, 'k', 'filled'); 
        
        max_sessions = length(x);
        xlabel('sessions');
        ylabel(['\bf' param_names(params_to_plot(i))]); 
        title(['\rm' param_out(params_to_plot(i))])
        xlim([1 max_sessions]);
        xticks([1 ceil(max_sessions/2) max_sessions]); 
        ylim([min_p(params_to_plot(i)) max_p(params_to_plot(i))])

        % change the background colour to reflect the stimulus set
        stims = unique(subj_sessions(:, 3));
        background = gobjects(length(stim_sets), 1); % preallocate
        for stim = 1:length(stims)
            stim_col = subj_sessions(:, 3)==stim;

            % find transition points (start and end of 1's)
            d = diff([0 stim_col' 0]); % pad to catch edges
            starts = find(d == 1);
            ends   = find(d == -1) - 1;
            
            % shade the background
            yl = ylim; % get y-axis limits
            for col = 1:length(starts)
                x_start = x(starts(col));
                x_end = x(ends(col));
                background(stim) = patch([x_start x_end x_end x_start], ...
                                          [yl(1) yl(1) yl(2) yl(2)], ...
                                          plot_params.Color(stims(stim), :), ...
                                          'FaceAlpha', 0.4, 'EdgeColor', 'none');
            end
        end
       
    end
end
ha = axes('Position',[0 0 1 1],'Visible','off');
legend(ha, background, plot_params.Legend, 'Location','northoutside', 'Orientation', 'horizontal');


% 3. plot value distortions
plot_params.Color = [repmat(96, 1, 3); repmat(224, 1, 3); 153, 0, 76]./255;
plot_params.MarkerSize = 40;
figure; set(gcf,'color','w');

% compute subjective values
obj = normalize_bound(1:10, 0.1, 0.999);
subj_mag  = nan(length(subjects), length(obj));
subj_prob = nan(length(subjects), length(obj));
subj_ev   = nan(length(subjects), length(obj));
for s = 1:length(subjects)         % subject loop
    i = sessions_to_fit(:,2)==s;

    int_coeff = mean(param_fits(i, 1));
    w_ratio   = mean(param_fits(i, 2));
    subj_mag(s, :)  = obj.^mean(param_fits(i, 3));
    subj_prob(s, :) = exp(-(-log(obj)).^mean(param_fits(i, 4)));
    subj_ev(s, :)   = (int_coeff.*mag.*prob) + (1-int_coeff).*(w_ratio.*mag + (1 - w_ratio).*prob);
end

subj_val = cat(3, subj_mag, subj_prob, subj_ev);
obj_val  = [obj; obj; obj.*obj];
for j = 1:size(subj_val, 3)
    handle(j) = scatter(obj_val(j, :), mean(subj_val(:, :, j)), plot_params.MarkerSize, 'filled', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', plot_params.Color(j,:));
    hold on  
end
axis square;
plot([0, 1], [0, 1], '--k')
xlabel('objective'); ylabel('subjective');
xticks(0:0.2:1); yticks(0:0.2:1)
legend(handle, {choice_attributes{1}, choice_attributes{2}, 'expected value'}, 'location','southeast')
