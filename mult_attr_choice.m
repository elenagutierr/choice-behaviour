%% analyse multiple-attribute choice behaviour in a value-based decision-making task 
% analyse choice behaviour in a two-attribute binary choice task with multiple stimulus sets. analyses include:
% 1. modelling session-by-session behaviour according to prospect theory
% 2. accuracy on brainer/no brainer trials
% 3. compare choice according to a strategy of maximising EV or of minimising distance to a goal 
%
% data (m_attr) is a struct with fields [1 x trials]. minimum data requirements are: 
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
    if n_trials >= min_trial_n 
        sessions_to_fit = [sessions_to_fit; session_id(i, :)];
    end
end

% % 1. test best-fitting model session-by-session
% best_models_i      = nan(length(sessions_to_fit), save_best_n);
% best_models_params = nan(save_best_n, tot_params, length(sessions_to_fit));
% best_models_bic    = nan(length(sessions_to_fit), save_best_n);
% 
% for i = 1:length(sessions_to_fit)
%     data_to_fit = all_data_to_fit(all_data_to_fit(:,7)==sessions_to_fit(i,1) & all_data_to_fit(:,8)==sessions_to_fit(i,2), :);
% 
%     all_param_fits = nan(num_models, tot_params);
%     all_neg_NLL    = nan(num_models,1);
% 
%     for ind = 1:num_models      
%         % set min and max to be equal to the prior if that specific parameter is not tested
%         min_param = prior;
%         max_param = prior;
%         min_param(logical(models(ind, 1:tot_params))) = min_bound(logical(models(ind, 1:tot_params))); 
%         max_param(logical(models(ind, 1:tot_params))) = max_bound(logical(models(ind, 1:tot_params)));
% 
%         [all_param_fits(ind,:), all_neg_NLL(ind)]=fmincon(@(params)fit_all_possible_models(params,data_to_fit), prior, [], [], [], [], min_param, max_param, [], options); 
%     end
% 
%     % save results from best fitting models
%     BIC = log(n_trials)*(num_params)+2*neg_NLL;     % compute BIC
%     sort_BIC = sortrows([BIC (1:num_models)'],1);
%     best_ind = sort_BIC(1:save_best_n,2);
%     best_models_i(i,:)   = best_ind;
%     best_models_bic(i,:) = BIC(best_ind,1);
%     best_models_params(:,:,i) = param_fits(best_ind,:);
%     disp([i ' of ' num2str(length(sessions_to_fit)) ' sessions done'])
% end

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
        xl = xlabel('sessions');
        yl = ylabel(['\bf' param_names(params_to_plot(i))]);
        tidy_labels(gca, yl, xl, 0.05)
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
plot_params.MarkerSize = 20;
plot_params.LineStyle = "none";
plot_params.LineColor = [0 0 0];


figure; set(gcf,'color','w');
m = 1; n = length(subjects);

% compute subjective values
obj = normalize_bound(1:10, 0.1, 0.999);

for s = 1:length(subjects)         % subject loop
    i = sessions_to_fit(:,2)==s;

    int_coeff = mean(param_fits(i, 1));
    w_ratio   = mean(param_fits(i, 2));
    subj_mag  = obj.^mean(param_fits(i, 3));
    subj_prob = exp(-(-log(obj)).^mean(param_fits(i, 4)));
    subj_ev   = (int_coeff.*subj_mag.*subj_prob) + (1-int_coeff).*(w_ratio.*subj_mag + (1 - w_ratio).*subj_prob);

    err_mag = std(obj.^param_fits(i, 3)) ./ sqrt(size(param_fits(i, 3), 1));
    err_prob = std(obj.^param_fits(i, 4)) ./ sqrt(size(param_fits(i, 4), 1));
    all_ev = (int_coeff.*(obj.^param_fits(i, 3)).*(obj.^param_fits(i, 4))) + (1-int_coeff).*(w_ratio.*(obj.^param_fits(i, 3)) + (1 - w_ratio).*(obj.^param_fits(i, 4)));
    err_ev = std(all_ev) ./ sqrt(size(all_ev, 1));

    subplot(m, n, s)
    subj_val = cat(3, subj_mag, subj_prob, subj_ev);
    subj_err = [err_mag; err_prob; err_ev];
    obj_val  = [obj; obj; obj.*obj];
    for j = 1:size(subj_val, 3)
        errorbar(obj_val(j, :), subj_val(:, :, j), subj_err(j, :), "Color", plot_params.LineColor, "LineStyle", plot_params.LineStyle)
        hold on  
        handle(j) = scatter(obj_val(j, :), subj_val(:, :, j), plot_params.MarkerSize, 'filled', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', plot_params.Color(j,:));
    end
    axis square;
    plot([0, 1], [0, 1], '--k')
    xl = xlabel('objective'); yl = ylabel('subjective');
    tidy_labels(gca, yl, xl, 0.05)
    title(subjects{s})
    xticks(0:0.2:1); yticks(0:0.2:1)
    legend(handle, {choice_attributes{1}, choice_attributes{2}, 'value'}, 'location','southeast')
    box off
    
end

%% 2. accuracy measures
% accuracy on brainer and no-brainer trials over time
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

% line and bar plot for accuracy
% set plotting parameters
plot_params.LineWidth = 2;
plot_params.LineStyle = "-";
plot_params.Color = [229, 165, 197; 153, 0, 76]./255;
plot_params.Legend = {'congruent trials', 'incongruent trials'};
plot_params.SmoothWindow = 3;

% % line plot across sessions
% figure, set(gcf,'color','w');
% m = 1; n = length(subjects);
% 
% for subj = 1:length(subjects)
%     subplot(m, n, subj)
%     for b = 1:size(accuracy_brain, 2)
%         acc = q_smooth((accuracy_brain(session_id(:, 2)==subj, b)*100)', plot_params.SmoothWindow, 1);
%         p(subj, b) = plot(acc,'Color', plot_params.Color(b, :),'LineWidth', plot_params.LineWidth);
%         hold on
%     end
% 
%     max_sessions = sum(session_id(:, 2)==subj);
%     hold off
%     box off
%     yline(50,'--k'); 
%     title(subjects(subj))
%     xlim([1 max_sessions]); ylim([25 100]);
%     xticks([1 ceil(max_sessions/2) max_sessions]); yticks(25:25:100);
%     xlabel('session'); ylabel('accuracy (%)');
%     legend(p(subj, :), plot_params.Legend, 'Location','southeast');
%     clear p
% end

% bar plot
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

figure(); set(gcf,'color','w');
b = bar(mean_acc*100, 'FaceColor', 'flat', 'EdgeColor', 'k');
for k = 1:size(mean_acc,2), b(k).CData = squeeze(plot_params.Color(k, :)); end
hold on
plot_grouped_errorbars(b, mean_acc*100, sem_acc*100, err_params);
yline(50,'k--')
ylim([0 100]); yticks(0:25:100)
xticklabels(subjects)
yl = ylabel('accuracy (%)'); xl = xlabel('subjects');
tidy_labels(gca, yl, xl, 0.05)
box off
legend(plot_params.Legend, 'Location','northoutside');

% confirm that accuracy on brainer (incongruent) trials is significantly above chance
for subj = 1:length(subjects)
    [~, p_chance(subj), ~, stats_chance(subj)] = ttest(accuracy_brain(session_id(:, 2)==subj, 2), 0.5, 'Tail', 'right');
    [~, p_brain(subj), ~, stats_brain(subj)] = ttest(accuracy_brain(session_id(:, 2)==subj, 2), accuracy_brain(session_id(:, 2)==subj, 1), 'Tail', 'left');
end 

%% 3. expected value vs distance to goal
% find trials that make different predictions
chosen_dg   = round(sqrt((10 - m_attr.chosen_mag).^2 + (10 - m_attr.chosen_prob).^2), 1);  
unchosen_dg = round(sqrt((10 - m_attr.unchosen_mag).^2 + (10 - m_attr.unchosen_prob).^2), 1);
chosen_dg   = normalize_bound(max(chosen_dg) - chosen_dg, 0.1, 1);      % invert dg for comparability with ev (better options would otherwise have lower dg)
unchosen_dg = normalize_bound(max(unchosen_dg) - unchosen_dg, 0.1, 1);

chosen_ev   = normalize_bound(m_attr.chosen_value, 0.1, 1);
unchosen_ev = normalize_bound(m_attr.unchosen_value, 0.1, 1);

% session-by-session
prop_dg = nan(length(session_id), 2);
prop_session_dg = nan(length(session_id), 1);
prop_session_ev = nan(length(session_id), 1);
for s = 1:length(session_id)
    i = m_attr.session==session_id(s, 1) & m_attr.subject==session_id(s, 2) & m_attr.(choice_trials)==1;
    diff_pred_i = chosen_ev(i) > unchosen_ev(i) & chosen_dg(i) < unchosen_dg(i) | chosen_ev(i) < unchosen_ev(i) & chosen_dg(i) > unchosen_dg(i);

    prop_dg(s, :) = [(sum(chosen_ev(i) < unchosen_ev(i) & chosen_dg(i) > unchosen_dg(i)) / sum(diff_pred_i)), sum(diff_pred_i)];
    prop_session_dg(s) = sum(chosen_dg(i) > unchosen_dg(i)) / sum(i);
    prop_session_ev(s) = sum(chosen_ev(i) > unchosen_ev(i)) / sum(i);
end

% group trials across sessions
prop_all_dg = nan(length(subjects), 33);
prop_all_ev = nan(length(subjects), 33);
n_trials = 20;
for subj = 1:length(subjects)
    i = m_attr.subject==subj & m_attr.(choice_trials)==1;
    diff_pred_i = chosen_ev > unchosen_ev & chosen_dg < unchosen_dg & i | chosen_ev < unchosen_ev & chosen_dg > unchosen_dg & i;

    ch_ev = chosen_ev(diff_pred_i);
    un_ev = unchosen_ev(diff_pred_i);
    ch_dg = chosen_dg(diff_pred_i);
    un_dg = unchosen_dg(diff_pred_i);

    n_block = floor(sum(diff_pred_i) / n_trials);
    for block = 1:n_block
        block_trials = (block-1)*n_trials+1:block*n_trials; % trials in the current block
        prop_all_dg(subj, block) = sum(ch_dg(block_trials) > un_dg(block_trials)) / n_trials;
        prop_all_ev(subj, block) = sum(ch_ev(block_trials) > un_ev(block_trials)) / n_trials;
    end 

end

% bar plot across all trials
% bar plot
for subj = 1:2
    mean_prop(:, subj) = [mean(prop_session_ev(session_id(:, 2)==subj)) mean(prop_session_dg(session_id(:, 2)==subj))];
    sem_prop(subj, :)  = [std(prop_session_ev(session_id(:, 2)==subj)) / sqrt(nnz(session_id(:, 2)==subj)),  std(prop_session_dg(session_id(:, 2)==subj)) / sqrt(nnz(session_id(:, 2)==subj))];
end

% set plot parameters
plot_params.Color = [229, 165, 197; 153, 0, 76]./255;
plot_params.Legend = {'chose higher EV', 'chose higher DG'};
err_params.Color  = [0.2 0.2 0.2];   % plotting parameters for error bars
err_params.LineWidth = 1;

figure(); set(gcf,'color','w');
b = bar(mean_prop*100, 'FaceColor', 'flat', 'EdgeColor', 'k');
for k = 1:size(mean_prop,2), b(k).CData = plot_params.Color(k, :); end
hold on
plot_grouped_errorbars(b, mean_prop*100, sem_prop*100, err_params);
yline(50,'k--')
ylim([0 100]); yticks(0:25:100)
xticklabels(subjects)
yl = ylabel('% choices'); 
xl = xlabel('subjects');
tidy_labels(gca, yl, xl, 0.05)
box off
legend(plot_params.Legend, 'Location','northoutside');

% zoom into difference between the two
plot_params.LineWidth_mean = 0.5;
plot_params.LineWidth_data = 2;
plot_params.LineStyle_mean = "-";
plot_params.LineStyle_ci   = "--";
plot_params.LineStyle_data = "-";
plot_params.Color_mean = [0, 0, 0]./255;
plot_params.Color_ci   = [224, 224, 224]./255;
plot_params.Color_data = [153, 0, 76]./255;
plot_params.Marker = ".";
plot_params.MarkerSize = 12;

figure, set(gcf,'color','w');
t = tiledlayout(length(subjects), 1, 'Padding', 'loose', 'TileSpacing', 'loose');

for subj = 1:length(subjects)

    data = [prop_all_dg(subj, :)' prop_all_ev(subj, :)'].*100;
    data(isnan(data(:, 1)), :) = [];
    diff_prop = data(:,1) - data(:, 2);
    ci95_prop = bootci(1000, @mean, diff_prop);

    nexttile
    b = patch([0 length(diff_prop) length(diff_prop) 0], ...
                   [ci95_prop(1) ci95_prop(1) ci95_prop(2) ci95_prop(2)], ...
                    plot_params.Color_ci, 'FaceAlpha', 0.4, 'EdgeColor', 'none');
    hold on
    scatter(1:length(diff_prop), diff_prop, 'w')
    ls = lsline;
    ls.LineStyle = "--";
    p(1) = plot(diff_prop, "LineStyle", plot_params.LineStyle_data, "LineWidth", plot_params.LineWidth_data,...
        "Color", plot_params.Color_data, "Marker", plot_params.Marker, "MarkerSize", plot_params.MarkerSize);
    p(2) = yline(0, "LineStyle", plot_params.LineStyle_mean, "LineWidth", plot_params.LineWidth_mean,...
        "Color", plot_params.Color_mean);

    xl = xlabel('trial block'); 
    yl = ylabel('% chose higher DG - % chose higher EV'); 
    tidy_labels(gca, yl, xl, 0.05)
    title(subjects{subj})
    hold off
end
sgtitle({'\rm proportion of all trials where'; '\rm\itchosen(\bfDG\rm\it)>unchosen(\bfDG\rm\it)\rm vs \itchosen(\bfEV\rm\it)>unchosen(\bfEV\rm\it)'}, ...
    "Color", "k")
