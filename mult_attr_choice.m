%% analyse multiple-attribute choice behaviour in a value-based decision-making task 
% analyse choice behaviour in a two-attribute binary choice task with multiple stimulus sets. analyses include:
% 1. accuracy on congruent and incongruent trials: are choice outcome and probability correctly integrated?
% 2. modelling session-by-session behaviour according to prospect theory
%
% data (m_attr) is a struct with fields [1 x trials]. minimum data requirements are: 
% ID for subject, session, stimulus set, and trial (if multiple trial types), 
% option values (EV and attributes, eg. magnitude and probability), option chosen (e.g. left/right, first/second), chose correct 

% EG 25

%% set paths and load data
my_dir   = 'C:\Users\elena\OneDrive\Documents\PhD\Projects\Discrete Value Space\Analysis\chapter3\';
addpath(genpath(my_dir));
m_attr = readstruct([my_dir '\data\m_attr.xml'], "FileType", "xml");

fig_dir  = 'C:\Users\elena\OneDrive\Documents\PhD\Writing\Thesis\Chapters\Chapter3_behaviour\figures\raw_mat'; % where to save figures

%% set constants
% dataset specifics
choice_attributes = {'magnitude', 'probability'};   
subjects   = {'A', 'T'};
stim_sets  = unique(m_attr.stimulus_set);
session_id = unique([m_attr.session' m_attr.subject' m_attr.stimulus_set'], 'rows', 'stable');

% subset the trials of interest here: 
choice_trials = 'choice_made';  

% plotting parameters
stim_colours  = [0, 78, 78; 153, 76, 0]./255;        % stimulus sets
acc_colours   = [229, 165, 197; 153, 0, 76]./255;    % ev + accuracy on brainer/no-brainer
ev_dg_colours = [229, 204, 255; 255, 153, 153]./255; % ev / dg choices (if also add: 245, 245, 94)
subj_colours  = [repmat(96, 1, 3)./255; repmat(192, 1, 3)./255];

font_size = 12;
set(0, 'DefaultAxesFontSize', font_size);
set(0, 'DefaultTextFontSize', font_size);
set(0, 'DefaultLegendFontSize', font_size);
set(0, 'DefaultColorbarFontSize', font_size);

%% 1. choice behaviour: accuracy measures and psychometric curve
% accuracy on congruent and incongruent trials over time
brain_i = m_attr.option1_mag > m_attr.option2_mag & m_attr.option1_prob < m_attr.option2_prob |...
          m_attr.option1_mag < m_attr.option2_mag & m_attr.option1_prob > m_attr.option2_prob;
br_i = unique(brain_i);

accuracy_brain = nan(length(session_id), length(br_i));
for s = 1:length(session_id)
    for b = 1:length(br_i)
        i = m_attr.session==session_id(s,1) & m_attr.subject==session_id(s,2) & brain_i==br_i(b) & m_attr.(choice_trials)==1;
    
        accuracy_brain(s, b) = sum(m_attr.correct(i)) / length(m_attr.correct(i));
    end
end

% confirm that accuracy on brainer (incongruent) trials is significantly above chance
for subj = 1:length(subjects)
    [~, p_chance(subj), ~, stats_chance(subj)] = ttest(accuracy_brain(session_id(:, 2)==subj, 2), 0.5, 'Tail', 'right');
    [~, p_brain(subj), ~, stats_brain(subj)] = ttest(accuracy_brain(session_id(:, 2)==subj, 1), accuracy_brain(session_id(:, 2)==subj, 2), 'Tail', 'right');

end 

% set plotting parameters
% line and bar plot for accuracy
plot_line = 0;
plot_bar  = 1;

% psychometric curves
plot_params.LineWidth = 2.5;
plot_params.MarkerSize = 10;
plot_params.Color = [subj_colours(1, :); subj_colours(1, :)];
plot_params.Legend = {'A','T'};

figure(); set(gcf,'color','w');
t = tiledlayout(1, length(subjects)+sum([plot_line*2 plot_bar]), 'Padding', 'loose', 'TileSpacing', 'compact');
t.OuterPosition = [0.05 0.05 1 1];  % Add margin around the entire layout

for subj = 1:length(subjects)
    nexttile
    % index 
    i = m_attr.subject==subj & m_attr.(choice_trials)==1;
    left_value  = m_attr.left_value(i);
    right_value = m_attr.right_value(i);
    chose_left  = m_attr.chose_left(i);

    value_diff_ranks = -100:5:100;

    [stats_out(subj), p(:, subj)] = plot_psych_curve(left_value, right_value, chose_left, plot_params, subj, value_diff_ranks);

    % label plot
    xl = xlabel('Left-Right Value Difference'); yl = ylabel('P(Chose left)');
    tidy_labels(gca, yl, 0.05, xl, 0.09)
    xticks(value_diff_ranks(1):value_diff_ranks(end)/4:value_diff_ranks(end)); 
    yticks(0:0.25:1); xticks(-100:50:100);
    box off
end

% bar/line plot for accuracy
plot_params.LineWidth = 2;
plot_params.LineStyle = "-";
plot_params.Color = acc_colours;
plot_params.Legend = {'Congruent trials', 'Incongruent trials'};
plot_params.SmoothWindow = 3;

if plot_bar % bar plot

    mean_acc = nan(length(subjects), length(br_i));
    sem_acc  = nan(length(subjects), length(br_i));
    for subj = 1:length(subjects)
        for b = 1:size(accuracy_brain, 2)
            mean_acc(subj, b) = mean(accuracy_brain(session_id(:, 2)==subj, b));
            sem_acc(subj, b)  = std(accuracy_brain(session_id(:, 2)==subj, b)) / sqrt(sum(session_id(:, 2)==subj));
        end
    end
    
    % set plot parameters
    err_params.Color  = [0.2 0.2 0.2];   % plotting parameters for error bars
    err_params.LineWidth = 1;
    
    nexttile
    b = bar(mean_acc*100, 'FaceColor', 'flat', 'EdgeColor', 'k');
    for k = 1:size(mean_acc,2), b(k).CData = squeeze(plot_params.Color(k, :)); end
    hold on
    plot_grouped_errorbars(b, mean_acc*100, sem_acc*100, err_params);
    yline(50,'k--')
    ylim([0 100]); yticks(0:25:100)
    xticklabels(subjects)
    yl = ylabel('Accuracy (%)'); xl = xlabel('Subjects');
    tidy_labels(gca, yl, 0.05, xl, 0.05)
    box off
    legend(plot_params.Legend, 'Location','northoutside');
end

if plot_line % line plot across sessions
   for subj = 1:length(subjects)
        nexttile 
        for b = 1:size(accuracy_brain, 2)
            acc = q_smooth((accuracy_brain(session_id(:, 2)==subj, b)*100)', plot_params.SmoothWindow, 1);
            p(subj, b) = plot(acc,'Color', plot_params.Color(b, :),'LineWidth', plot_params.LineWidth);
            hold on
        end
    
        max_sessions = sum(session_id(:, 2)==subj);
        hold off
        box off
        yline(50,'--k'); 
        title(subjects(subj))
        xlim([1 max_sessions]); ylim([25 100]);
        xticks([1 ceil(max_sessions/2) max_sessions]); yticks(25:25:100);
        xlabel('Session'); ylabel('Accuracy (%)');
        legend(p(subj, :), plot_params.Legend, 'Location','southeast');
        clear p
    end
end

% save figure
f = gcf; f.Units = 'inches'; % or 'centimeters'
exportgraphics(f, [fig_dir '\fig3_psych_brain.svg'], 'Resolution', 300, 'ContentType', 'vector', 'BackgroundColor', 'white');

%% 2. prospect theory modelling
% here we test session-by-session behaviour under the lense of prospect theory in two ways:
% 1. we'd like to know what the best fitting model is on a session-by-session basis
% 2. if we fit a full model (all parameters), how do these evolve across sessions?
%
% to do this, we adapted code from "Activation and disruption of a neural mechanism for novel choice in monkeys" (Bongioanni et al, 2021)

% set constants
min_trial_n = 200;  % minimum number of trials/session to be included in the analysis

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
                'probability distortion', 'inverse choice temperature', 'random error',...
                'constant side bias', 'repetition side bias', 'WSLS side bias'};

% create all possible model-defining combinations 
% (one column per parameter, 0 if it's fixed to default value, 1 if it's a free parameter)
tot_params = length(param_names)-3;     % in this instance we are not interested in spatial biases
models     = zeros((2^tot_params),tot_params);
for x = 1:tot_params
    models(:,x) = repmat([zeros(2^(x-1),1);ones(2^(x-1),1)],2^(tot_params-x),1);
end
num_models = length(models);
num_params = sum(models(1:2^tot_params,1:tot_params)>0,2);

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
n_trials_to_fit = [];
for i = 1:length(session_id)
    n_trials = sum(all_data_to_fit(:,7)==session_id(i,1) & all_data_to_fit(:,8)==session_id(i,2));
    if n_trials >= min_trial_n 
        sessions_to_fit = [sessions_to_fit; session_id(i, :)];
        n_trials_to_fit = [n_trials_to_fit; n_trials];
    end
end

% 1. test best-fitting models (i) session-by-session OR (ii) pooling all sessions
best_models_i      = nan(length(sessions_to_fit), save_best_n);
best_models_params = nan(length(sessions_to_fit), tot_params, save_best_n);
best_models_bic    = nan(length(sessions_to_fit), save_best_n);

% (i) session-by-session
for i = 1:length(sessions_to_fit)
    data_to_fit = all_data_to_fit(all_data_to_fit(:, 7)==sessions_to_fit(i, 1) & all_data_to_fit(:, 8)==sessions_to_fit(i, 2), :);

    all_param_fits = nan(num_models, tot_params);
    all_neg_NLL    = nan(num_models,1);

    for ind = 1:num_models      
        % set min and max to be equal to the prior if that specific parameter is not tested
        min_param = prior;
        max_param = prior;
        min_param(logical(models(ind, 1:tot_params))) = min_bound(logical(models(ind, 1:tot_params))); 
        max_param(logical(models(ind, 1:tot_params))) = max_bound(logical(models(ind, 1:tot_params)));

        [all_param_fits(ind,:), all_neg_NLL(ind)]=fmincon(@(params)fit_all_possible_models(params,data_to_fit), prior, [], [], [], [], min_param, max_param, [], options); 

        if ind==256, disp('50% of the models done'), end 
    end

    % save results from best fitting models
    BIC = log(n_trials_to_fit(i))*(num_params)+2*all_neg_NLL;     % compute BIC
    sort_BIC = sortrows([BIC (1:num_models)'], 1);
    best_ind = sort_BIC(1:save_best_n, 2);
    best_models_i(i, :)   = best_ind;
    best_models_bic(i, :) = BIC(best_ind,1);
    best_models_params(i, :, :) = all_param_fits(best_ind,:);
    disp([num2str(i) ' of ' num2str(length(sessions_to_fit)) ' sessions done'])
end
out.best_model = best_models_i;
out.best_model_BIC = best_models_bic;
out.best_model_par = best_models_params;
save([my_dir 'stats\best_models.mat'], "out")

% (ii) pool all sessions
save_best_n = 5;
best_models_i      = nan(length(subjects), save_best_n);
best_models_params = nan(length(subjects), tot_params, save_best_n);
best_models_bic    = nan(length(subjects), save_best_n);
best_models_sw     = nan(length(subjects), save_best_n);
for i = 1:length(subjects)
    data_to_fit = all_data_to_fit(all_data_to_fit(:, 8)==i, :);

    all_param_fits = nan(num_models, tot_params);
    all_neg_NLL    = nan(num_models,1);

    for ind = 1:num_models      
        % set min and max to be equal to the prior if that specific parameter is not tested
        pr = prior(1:tot_params);
        min_param = pr;
        max_param = pr;
        min_param(logical(models(ind, 1:tot_params))) = min_bound(logical(models(ind, 1:tot_params))); 
        max_param(logical(models(ind, 1:tot_params))) = max_bound(logical(models(ind, 1:tot_params)));

        [all_param_fits(ind,:), all_neg_NLL(ind)]=fmincon(@(params)fit_all_possible_models(params,data_to_fit), pr, [], [], [], [], min_param, max_param, [], options); 

        disp([num2str(ind) ' of the ' num2str(num_models) ' models done']) 
    end

    % save results from best fitting models
    BIC = log(n_trials_to_fit(i))*(num_params)+2*all_neg_NLL;     % compute BIC
    sort_BIC = sortrows([BIC (1:num_models)'], 1);
    best_ind = sort_BIC(1:save_best_n, 2);
    best_models_i(i, :)   = best_ind';
    best_models_bic(i, :) = BIC(best_ind,1)';
    best_models_params(i, :, :) = all_param_fits(best_ind,:)';

    % compute weights for best models
    delta = sort_BIC(:, 1) - min(sort_BIC(:, 1));   % delta BICs relative to the minimum BIC
    likelihoods = exp(-0.5 * delta);    % convert BIC differences to likelihood scale
    w = likelihoods / sum(likelihoods); % normalize to get weights (they sum to 1)

    best_models_sw(i, :) = w(1:save_best_n)';
end
out.best_model = best_models_i;
out.best_model_BIC = best_models_bic;
out.best_model_par = best_models_params;
out.best_model_sw  = best_models_sw;
save([my_dir 'stats\best_model_all_sessions_inf.mat'], "out")

% 2. run winning model overall session-by-session
out = load([my_dir 'stats\best_model_all_sessions_no_side.mat']); out = out.out;
param_fits = nan(length(sessions_to_fit), tot_params); % parameter fits
neg_nll    = nan(length(sessions_to_fit),1);           % negative log-likelihood (badness of fit)
parfor i = 1:length(sessions_to_fit)
    
    data_to_fit = all_data_to_fit(all_data_to_fit(:,7)==sessions_to_fit(i,1) & all_data_to_fit(:,8)==sessions_to_fit(i,2), :);
    subj    = sessions_to_fit(i, 2);
    model_i = out.best_model(subj, 1);

    % set min and max to be equal to the prior if that specific parameter is not tested
    pr = prior(1:tot_params);
    min_param = pr;
    max_param = pr;
    min_param(logical(models(model_i, 1:tot_params))) = min_bound(logical(models(model_i, 1:tot_params))); 
    max_param(logical(models(model_i, 1:tot_params))) = max_bound(logical(models(model_i, 1:tot_params)));


    [param_fits(i,:), neg_nll(i)]=fmincon(@(params)fit_all_possible_models(params, data_to_fit), pr, [], [], [], [], min_param, max_param, [], options); 
    disp(['Session ' num2str(i) ' of ' num2str(length(sessions_to_fit)) ' done'])
end

% test increase in theta - do choices get more deterministic over time?
for subj = 1:length(subjects)
    subj_data = param_fits(sessions_to_fit(:, 2)==subj, 5);
    subj_data(subj_data > 50) = [];
    [~, ~, stats_theta(subj)] = glmfit(1:length(subj_data), subj_data);
end

% 3. is there learning set for converging towards multiplicative strategy?
% number of sessions for eta > 0.5 for 2 consecutive sessions for each stimulus set switch
trials_eta = zeros(length(session_id), 2);
for s = 1:length(session_id)
        i = m_attr.session==session_id(s,1) & m_attr.subject==session_id(s,2) & m_attr.(choice_trials)==1;    
        trials_eta(s, 1) = sum(i); 

        i = sessions_to_fit(:, 1)==session_id(s, 1) & sessions_to_fit(:, 2)==session_id(s, 2);
        if sum(i) > 0
            trials_eta(s, 2) = param_fits(i, 1);
        end
end

sessions_to_crit = nan(length(subjects), 5);
trials_to_crit   = nan(length(subjects), 5);
criterion = 0.5;
for subj = 1:length(subjects)
    sessions   = session_id(session_id(:, 2)==subj, :);
    subj_data  = trials_eta(session_id(:, 2)==subj, 2);
    subj_trials = trials_eta(session_id(:, 2)==subj, 1);

    i = 1;
    for j = 1:length(sessions)-1
        if sessions(j, 3) ~= sessions(j+1, 3), i = [i, i(end)+1]; else,  i = [i, i(end)]; end
    end

    for j = 1:length(unique(i))
        data = subj_data(i==j);
        trials = subj_trials(i==j);
        session = sessions(i==j);

        for k = 1:length(data)
            % sessions to criterion
            if data(k) > criterion 
                sessions_to_crit(subj, j) = k; 
                trials_to_crit(subj, j) = sum(trials(1:k));
                break; 
            end 
        end        
    end   
end

%% plot parameters for winning model across sessions
% we'd like to format this in chronological manner while also highlighting the different stimulus sets
out = load([my_dir 'stats\best_model_all_sessions_no_side.mat']); out = out.out;

% plot the winning model and value distortions
plot_params.Color = [repmat(96, 1, 3)./255; repmat(224, 1, 3)./255; acc_colours(2, :)];
plot_params.MarkerSize = 30;
plot_params.LineStyle = "none";
plot_params.LineColor = [0 0 0];

figure(); set(gcf,'color','w');
t = tiledlayout(2, 6, 'Padding', 'loose', 'TileSpacing', 'loose');
t.OuterPosition = [0.05 0.05 1 1];  % Add margin around the entire layout
y_lim = [15 30];
for subj = 1:length(subjects)
    % plot winning model
    bic = out.best_model_BIC(subj, 1:end-1)-out.best_model_BIC(subj, 1);
    par_i = logical(models(out.best_model(subj, 1:end-1), :));
    par_n = param_names(1:tot_params);
    for i = 1: size(par_i, 1), par_lab{i} = strjoin(par_n(par_i(i, :)), ' '); end

    nexttile([2, 1])
    b_plot = bar(bic, 'FaceColor', [0.87 0.87 0.87], 'EdgeColor', 'k');
    xl = xlabel('Model'); yl = ylabel('\Delta BIC');
    tidy_labels(gca, yl, 0.07, xl, 0.22)
    xticklabels(par_lab)
    ylim([0 y_lim(subj)]); yticks([0 y_lim(subj)/2 y_lim(subj)])
    box off

end

obj = normalize_bound(1:10, 0.1, 1);
for subj = 1:length(subjects)
    % plot value distortions
    int_coeff = out.best_model_par(subj, 1);
    w_ratio   = out.best_model_par(subj, 2);
    subj_mag  = obj.^out.best_model_par(subj, 3);
    subj_prob = exp(-(-log(obj)).^out.best_model_par(subj, 4));
    subj_ev   = (int_coeff.*subj_mag.*subj_prob) + (1-int_coeff).*(w_ratio.*subj_mag + (1 - w_ratio).*subj_prob);

    nexttile([2, 2])
    subj_val = cat(3, subj_mag, subj_prob, subj_ev);
    obj_val  = [obj; obj; obj.*obj];

    for j = 1:size(subj_val, 3)
        handle(j) = scatter(obj_val(j, :), subj_val(:, :, j), plot_params.MarkerSize, 'filled', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', plot_params.Color(j,:));
        hold on
    end
    axis square;
    plot([0, 1], [0, 1], '--k')
    xl = xlabel('objective'); yl = ylabel('subjective');
    tidy_labels(gca, yl, 0.005, xl, 0.05)
    xticks(0:0.5:1); yticks(0:0.5:1)
    legend(handle, {choice_attributes{1}, choice_attributes{2}, 'value'}, 'location','southeast')
    box off
    
end

% save figure
f = gcf; f.Units = 'inches'; % or 'centimeters'
exportgraphics(f, [fig_dir '\fig5_best_model.svg'], 'Resolution', 300, 'ContentType', 'vector', 'BackgroundColor', 'white');

%% plot winning model session-by-session
% set plotting constants
% model_to_plot = out.best_model(:, 1);
model_to_plot = [64; 64];
params_to_plot = [find(models(model_to_plot(1), :)); find(models(model_to_plot(2), :))];
% params_to_plot = params_to_plot(:, 1:end-1);
params_to_plot = params_to_plot(:, [1 2 3 5]);

plot_params.Color = stim_colours;
plot_params.MarkerSize = 12;
plot_params.Legend = {'Stimulus set 1', 'Stimulus set 2'};

min_p=[0,0,0,0,0,0];   % set limits
max_p=[1,1,1,1,30,1];

figure(); set(gcf,'color','w');
t = tiledlayout(length(subjects), length(params_to_plot), 'Padding', 'loose', 'TileSpacing', 'loose');
t.OuterPosition = [0.05 0.05 1 1];  % Add margin around the entire layout

for subj = 1:length(subjects)
    for i = 1:size(params_to_plot, 2)
    % plot parameters
    subj_data = param_fits(sessions_to_fit(:, 2)==subj, params_to_plot(subj, i));
    subj_sessions = sessions_to_fit(sessions_to_fit(:, 2)==subj, :);
    x = 1:length(subj_data);

    nexttile
    scatter(x, subj_data, plot_params.MarkerSize, 'k', 'filled'); 
    
    max_sessions = length(x);
    xl = xlabel('sessions');
    yl = ylabel(['\bf' param_names(params_to_plot(subj, i))]);
    tidy_labels(gca, yl, 0, xl, 0.05)
    title(['\rm' param_out(params_to_plot(subj, i))], 'Units', 'normalized', 'Position', [0.5, 1.005, 0])
    xlim([1 max_sessions]);
    xticks([1 ceil(max_sessions/2) max_sessions]); 
    ylim([min_p(params_to_plot(i)) max_p(params_to_plot(subj, i))])
    yticks([min_p(params_to_plot(i)) max_p(params_to_plot(subj, i))/2 max_p(params_to_plot(subj, i))])

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

% save figure
f = gcf; f.Units = 'inches'; % or 'centimeters'
exportgraphics(f, [fig_dir '\app_fig2_model_over_time.svg'], 'Resolution', 300, 'ContentType', 'vector', 'BackgroundColor', 'white');

%% plot probability distortion
params_to_plot = [4; 4];
plot_params.Color = stim_colours;
plot_params.MarkerSize = 12;
plot_params.Legend = {'Stimulus set 1', 'Stimulus set 2'};

min_p=[0,0,0,0,0,0];   % set limits
max_p=[1,1,1,1,30,1];

figure(); set(gcf,'color','w');
t = tiledlayout(length(subjects), length(params_to_plot), 'Padding', 'loose', 'TileSpacing', 'loose');
t.OuterPosition = [0.05 0.05 1 1];  % Add margin around the entire layout

for subj = 1:length(subjects)
    for i = 1:size(params_to_plot, 2)
    % plot parameters
    subj_data = param_fits(sessions_to_fit(:, 2)==subj, params_to_plot(subj, i));
    subj_sessions = sessions_to_fit(sessions_to_fit(:, 2)==subj, :);
    x = 1:length(subj_data);

    nexttile
    scatter(x, subj_data, plot_params.MarkerSize, 'k', 'filled'); 
    
    max_sessions = length(x);
    xl = xlabel('sessions');
    yl = ylabel(['\bf' param_names(params_to_plot(subj, i))]);
    tidy_labels(gca, yl, 0, xl, 0.05)
    title(['\rm' param_out(params_to_plot(subj, i))], 'Units', 'normalized', 'Position', [0.5, 1.005, 0])
    xlim([1 max_sessions]);
    xticks([1 ceil(max_sessions/2) max_sessions]); 
    ylim([min_p(params_to_plot(i)) max_p(params_to_plot(subj, i))])
    yticks([min_p(params_to_plot(i)) max_p(params_to_plot(subj, i))/2 max_p(params_to_plot(subj, i))])

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

plot_params.Color = repmat(224, 1, 3)./255;
plot_params.MarkerSize = 30;

obj = normalize_bound(1:10, 0.1, 1);
for subj = 1:length(subjects)
    subj_data = param_fits(sessions_to_fit(:, 2)==subj, 4);
    subj_prob = exp(-(-log(obj')).^mean(subj_data(1:10)));

    nexttile
    scatter(obj, subj_prob, plot_params.MarkerSize, 'filled', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', plot_params.Color);
    hold on
    axis square;
    plot([0, 1], [0, 1], '--k')
    xl = xlabel('objective probability'); yl = ylabel('subjective probability');
    tidy_labels(gca, yl, 0.05, xl, 0.05)
    xticks(0:0.5:1); yticks(0:0.5:1)
    box off
    
end

f = gcf; f.Units = 'inches'; % or 'centimeters'
exportgraphics(f, [fig_dir '\fig6_prob.svg'], 'Resolution', 300, 'ContentType', 'vector', 'BackgroundColor', 'white');

%% plot learning set for decision strategy
figure(); set(gcf,'color','w');

plot_params.LineWidth = 2;
plot_params.Color = subj_colours;
plot_params.MarkerSize = 5;
plot_params.Marker = {'o', 'd'};

for subj = 1:length(subjects)
    
    p = plot(trials_to_crit(subj, :), "LineWidth", plot_params.LineWidth, "Color", plot_params.Color(subj, :),...
        "Marker", plot_params.Marker(subj), "MarkerSize", plot_params.MarkerSize, ...
        "MarkerFaceColor", plot_params.Color(subj, :));
    hold on

end
xl = xlabel('Stimulus set switch'); 
yl = ylabel('Trials to criterion'); 
tidy_labels(gca, yl, 0, xl, 0.05)
xlim([0.5 length(trials_to_crit)+0.5]); ylim([0 13000]);
xticks(1:length(trials_to_crit)); yticks(0:3250:13000)
legend(subjects, 'Location', 'northeast', 'fontsize', 12)
box off

f = gcf; f.Units = 'inches'; % or 'centimeters'
exportgraphics(f, [fig_dir '\fig7_learn_set.svg'], 'Resolution', 300, 'ContentType', 'vector', 'BackgroundColor', 'white');


%% final general stats for each subject
% number of trials per session
n_trials = nan(length(session_id), 1);
for s = 1:length(session_id)
        i = m_attr.session==session_id(s,1) & m_attr.subject==session_id(s,2) & m_attr.(choice_trials)==1;    
        n_trials(s) = sum(i);    
end

% number of sessions per subject
n_sessions = nan(length(subjects), length(stim_sets));
mean_sem_t = nan(length(subjects), 2);
for subj = 1:length(subjects)
    for stim = 1:length(stim_sets)
        i = session_id(:, 2)==subj & session_id(:, 3)==stim;
    
        n_sessions(subj, stim) = sum(i);
        mean_sem_t(subj, :) = [mean(n_trials(i)), std(n_trials(i))/sqrt(sum(i))];
    end
end
