%% analyse single-attribute choice behaviour in a value-based decision-making task 
% run a standard battery of tests to analyse choice behaviour in a single attribute binary choice task with multiple stimulus sets. analyses include:
% 1. plotting a psychometric curve 
% 2. testing learning set across stimulus sets
%
% data (s_attr) is a struct with fields [1 x trials]. minimum data requirements are: 
% ID for subject, session, choice attribute, stimulus set, and trial (if multiple trial types), 
% option values, option chosen (e.g. left/right, first/second), chose correct 

% EG 25

%% set paths and load data
my_dir   = 'C:\Users\elena\OneDrive\Documents\PhD\Projects\Discrete Value Space\Analysis\chapter3\';
addpath(genpath(my_dir));
s_attr = readstruct([my_dir '\data\s_attr.xml'], "FileType", "xml");

%% set constants
% task specifics
choice_attributes = {'magnitude', 'probability'};   
subjects = {'A', 'V'};
stim_sets = unique(s_attr.stimulus_set);

% in my task, only a subset of trials involve binary choices. 
% subset the trials of interest here: 
choice_trials = 'probe_test';   % logical for trials of interest  
s_attr.(choice_trials) = logical(s_attr.(choice_trials));

% colours
stim_colours = [255, 69, 0;255, 140, 0]./255;
attr_colours = [0, 100, 0;153, 204 47]./255;

%% psychometric curves 
% plot psychometric curves and compare across subjects, attributes and stimulus sets

% 1. plot across all sessions

% set plotting parameters
plot_params.LineWidth = 2;
plot_params.MarkerSize = 10;
plot_params.Color = stim_colours;

figure(); set(gcf,'color','w');
m = length(choice_attributes); n = length(subjects); ps = 1;

for subj = 1:length(subjects)
    for attr = 1:length(choice_attributes)
        sets = unique(s_attr.stimulus_set(s_attr.subject==subj & s_attr.attribute==attr));
        
        for s = 1:length(sets)

            % index 
            i = s_attr.subject==subj & s_attr.attribute==attr & s_attr.stimulus_set==s & s_attr.(choice_trials);
            left_value  = s_attr.left_value(i);
            right_value = s_attr.right_value(i);
            chose_left  = s_attr.chose_left(i);

            subplot(m, n, ps)
            [~, p(:, s)] = plot_psych_curve(left_value, right_value, chose_left, plot_params, s);
            hold on
        end

        % label plot
        xlabel('Left-Right Value Difference'); ylabel('P(Chose left)');
        title(subjects{subj})
        xticks(-9:3:9); yticks(0:0.25:1);
        box off
        hold off 
        ps = ps+1;
    end
end

% 2. compare curves between choice attributes and stimulus sets
session_id = unique([s_attr.subject' s_attr.session' s_attr.attribute' s_attr.stimulus_set'], "rows");
slopes     = nan(length(session_id), 1); % measures discrimination - how quickly do the subjects switch from guessing to correct response?
x_theshold = nan(length(session_id), 1); % measures bias - at what stimulus value does response probability reach 50%?

for j = 1:length(session_id)
    i = s_attr.subject==session_id(j,1) & s_attr.session==session_id(j,2) & s_attr.(choice_trials);

    left_value  = s_attr.left_value(i);
    right_value = s_attr.right_value(i);
    chose_left  = s_attr.chose_left(i);
    
    [stats_out] = plot_psych_curve(left_value, right_value, chose_left);
    slopes(j)     = 1 / stats_out.beta(2);
    x_theshold(j) = -stats_out.beta(1) / stats_out.beta(2);
end

% now we test with a 2-way anova with two factors: choice attribute and stimulus set
for s = 1:length(subjects)
    attr = categorical(session_id(session_id(:,1)==s,3));
    set  = categorical(session_id(session_id(:,1)==s, 4));
    
    [p_slope(:, s), tbl_slope(:, :, s), stats_slope(s)] = anovan(slopes(session_id(:,1)==s), {attr, set}, ...
                                'model', 'interaction', 'varnames', {'Choice attribute', 'Stimulus set'});
    [p_thres(:, s), tbl_thres(:, :, s), stats_thres(s)] = anovan(x_theshold(session_id(:,1)==s), {attr, set}, ...
                                'model', 'interaction', 'varnames', {'Choice attribute', 'Stimulus set'});
end

%% learning sets
% show a learning set once the second stimulus set is introduced
% normally measured as trials to criterion, we measure:
% (i) number of sessions to reach a given accuracy
% (ii) number of trials to reach a given accuracy over a block of trials

% set thresholds
criterion = 0.85;   % accuracy threshold
n_trials  = 20;     % number of trials within block

% 1. compute performance
sessions_to_crit = nan(length(subjects), length(choice_attributes), length(stim_sets));   
trials_to_crit   = nan(length(subjects), length(choice_attributes), length(stim_sets));

for subj = 1:length(subjects)
    for attr = 1:length(choice_attributes)
        sets = unique(s_attr.stimulus_set(s_attr.subject==subj & s_attr.attribute==attr));      
        for stim = 1:length(sets)
            i = s_attr.subject==subj & s_attr.attribute==attr & s_attr.stimulus_set==sets(stim) & s_attr.(choice_trials);             
    
            % sessions to criterion
            sessions = unique(s_attr.session(i));
            for s = 1:length(sessions)
                session_i = i & s_attr.session == sessions(s);
                accuracy = sum(s_attr.correct(session_i)) / length(s_attr.correct(session_i));
                if accuracy >= criterion
                    sessions_to_crit(subj, attr, stim) = s; % store the session number if criterion is reached, then exit loop
                    break; 
                end
            end
    
            % trials to criterion
            correct = s_attr.correct(i);

            trial_blocks = floor(length(correct) / n_trials); 
            for block = 1:trial_blocks

                block_trials = correct((block-1)*n_trials+1:min(block*n_trials, end)); % trials in the current block
                accuracy = sum(block_trials) / length(block_trials);
                if accuracy >= criterion
                    trials_to_crit(subj, attr, stim) = block*n_trials; % store the number of trials if criterion is reached, then exit loop
                    break; 
                end
            end 
        end
    end
end


% 2. plot the average trials to criterion between stimulus sets
mean_vals = squeeze(mean(trials_to_crit, [1, 2]));
sem_vals  = squeeze(std(trials_to_crit, 0, [1 2]) ./ sqrt(4));     

% set plot parameters
plot_params.Markers = {'o', 'd'};       % subjects
plot_params.Color = attr_colours;     % choice attributes
figure(); set(gcf,'color','w');

% bar plot
b_plot = bar(mean_vals, 'FaceColor', [0.87 0.87 0.87], 'EdgeColor', 'k');
xlabel('stimulus set'); ylabel('trials to criterion');
ylim([0 180]); yticks(0:60:180)
box off
hold on

% add error bars
errorbar(1:2, mean_vals, sem_vals,"Color", [0.2 0.2 0.2], "LineWidth", 1, 'LineStyle', 'none');

% add individual points
for subj = 1:length(subjects)
    for attr = 1:length(choice_attributes)
        % extract the two points for each subject+attribute across stimulus sets
        y_vals = squeeze(trials_to_crit(subj, attr, :));  
        
        % x values are jittered around bar centers 
        x_vals = [1, 2] + 0.15*(rand(1,2) - 0.5);  

        % connect points with a line
        plot(x_vals, y_vals', '-', 'Color', [0.2 0.2 0.2], "LineWidth", 1);  % semi-transparent gray

        % plot individual points
        scatter(x_vals, y_vals', 60, 'filled', ...
            'MarkerFaceColor', plot_params.Color(attr, :), ...
            'MarkerEdgeColor', 'k', ...
            'Marker', plot_params.Markers(subj));        
    end
end

%% basic behavioural measures
% finally, we'd like to plot overall accuracy, trial numbers and accuracy as a function of value difference

% 1. compute overall accuracy, number of trials per session and accuracy as a function of value difference
session_id = unique([s_attr.subject' s_attr.session' s_attr.attribute' s_attr.stimulus_set'], "rows");
accuracy = nan(length(sessions),1);
n_trials = nan(length(sessions),1);

value_diff_ranks = unique(abs(s_attr.left_value(s_attr.(choice_trials)) - s_attr.right_value(s_attr.(choice_trials))));
acc_value_diff   = nan(length(session_id), max(value_diff_ranks));

for s = 1:length(session_id)
    i = s_attr.subject==session_id(s,1) & s_attr.session==session_id(s,2) & s_attr.(choice_trials);
    accuracy(s) = sum(s_attr.correct(i)) / length(s_attr.correct(i));
    n_trials(s) = sum(i);
    
    value_diff = abs(s_attr.left_value(i) - s_attr.right_value(i));  
    correct = s_attr.correct(i);

    for j = 1:length(value_diff_ranks)
        ii = value_diff==value_diff_ranks(j);
        acc_value_diff(s, j) = sum(correct(ii)) / length(correct(ii));
    end
end


% we want to find the max amount of trials to for line plots
for j = 1: length(subjects), for k = 1:length(choice_attributes), for s = 1:length(stim_sets), ...
    tmp(j, k, s) = max(length(n_trials(session_id(:,1)==j & session_id(:,3)==k & session_id(:,4)==s))); end, end, end
max_sessions = max(tmp, [], "all");

% set plotting parameters
plot_params.LineWidth = 2;
plot_params.LineStyle = "-";
plot_params.Color = cat(3, attr_colours, stim_colours);
plot_params.Legend = {'Magnitude Set 1','Probability Set 1','Magnitude Set 2','Probability Set 2'};

figure, set(gcf,'color','w');
m = length(subjects); n = 2;

p_acc = 1;
p_tr  = 2;
for subj = 1:length(subjects)

    % plot accuracy by subject
    subplot(m, n, p_acc)
    for stim = 1:length(stim_sets)
        for attr = 1:length(choice_attributes)
            i = session_id(:,1)==subj & session_id(:, 3)==attr & session_id(:, 4)==stim;
            plot(accuracy(i)*100,'Color', plot_params.Color(attr, :, stim),'LineWidth', plot_params.LineWidth)
            hold on
        end
    end
    hold off
    box off
    yline(50,'--k'); 
    xlim([1 max_sessions]); ylim([25 100]);
    xticks(1:max_sessions/2:max_sessions); yticks(25:25:100);
    xlabel('session'); ylabel('accuracy (%)');
    legend(plot_params.Legend,'location','southeast')

    p_acc = p_acc+n;

    % plot number of trials by subject
    subplot(m, n, p_tr)
    for stim = 1:length(stim_sets)
        for attr = 1:length(choice_attributes)
            i = session_id(:,1)==subj & session_id(:, 3)==attr & session_id(:, 4)==stim;
            plot(n_trials(i),'Color', plot_params.Color(attr, :, stim),'LineWidth', plot_params.LineWidth)
            hold on
        end
    end
    hold off
    box off
    yline(50,'--k'); 
    y = ceil(max(n_trials(session_id(:, 1)==subj))./10).*10;
    xlim([1 max_sessions]); ylim([0 y]);
    xticks(1:max_sessions/2:max_sessions); yticks(0:y/2:y);
    xlabel('session'); ylabel('number of trials');
    legend(plot_params.Legend,'location','southeast')

    p_tr = p_tr+n;
end


% 2. plot accuracy as a function of value difference
figure, set(gcf,'color','w');
m = length(subjects); n = length(choice_attributes);

err_params.Color = [0.2 0.2 0.2];   % plotting parameters for error bars
err_params.LineWidth = 1;

pl = 1;
for subj = 1:length(subjects)
    for attr = 1:length(choice_attributes)

        i = session_id(:,1)==subj & session_id(:, 3)==attr;
        mean_acc = [mean(acc_value_diff(i & session_id(:,4)==1,:))' mean(acc_value_diff(i & session_id(:,4)==2,:))'];
        sem_acc = [squeeze(std(acc_value_diff(i & session_id(:,4)==1,:), 0, 1) ./ sqrt(size(acc_value_diff(i & session_id(:,4)==1,:), 1)))'...
            squeeze(std(acc_value_diff(i & session_id(:,4)==2,:), 0, 1) ./ sqrt(size(acc_value_diff(i & session_id(:,4)==2,:), 1)))'];

        subplot(m, n, pl)
        b = bar(mean_acc*100, 'FaceColor', 'flat');
        for k = 1:size(mean_acc,2), b(k).CData = squeeze(plot_params.Color(attr, :, k)); end
        hold on
        plot_grouped_errorbars(b, mean_acc*100, sem_acc*100, err_params);
        yline(50,'k--')
        ylim([0 100]); yticks(0:25:100)
        xticklabels(value_diff_ranks)
        ylabel('accuracy (%)')
        xlabel('ranked value difference')
        title(subjects{subj})
        box off

        pl = pl+1;
    end
end