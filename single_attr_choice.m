%% analyse single-attribute choice behaviour in a value-based decision-making task 
% run a standard battery of tests to analyse choice behaviour in a single attribute 
% binary choice task with multiple stimulus sets. analyses include:
% 1. plotting a psychometric curve 
% 2. testing learning set across stimulus sets

% EG 25

%% set paths and load data
my_dir   = 'C:\Users\elena\OneDrive\Documents\PhD\Projects\Discrete Value Space\Analysis\chapter3\';
addpath(genpath(my_dir));
s_attr = readstruct([my_dir '\data\s_attr.xml'], "FileType", "xml");

%% set constants
choice_attributes = {'magnitude', 'probability'};
subjects = {'A', 'V'};
stim_sets = unique(s_attr.stimulus_set);

% colours
stim_colours = [255, 69, 0;255, 140, 0]./255;
attr_colours = [0, 100, 0;153, 204 47]./255;

%% psychometric curves 
% plot psychometric curves and compare across subjects, attributes and stimulus sets

% 1. plot across all sessions

% set plotting parameters
plot_params.LineWidth = 2;
plot_params.MarkerSize = 10;
plot_params.Colours = stim_colours;

figure(); set(gcf,'color','w');
m = length(choice_attributes); n = length(subjects); ps = 1;

for subj = 1:length(subjects)
    for attr = 1:length(choice_attributes)
        sets = unique(s_attr.stimulus_set(s_attr.subject==subj & s_attr.attribute==attr));
        
        for s = 1:length(sets)

            % index 
            i = s_attr.subject==subj & s_attr.attribute==attr & s_attr.stimulus_set==s & s_attr.probe_test;
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
        
        hold off 
        ps = ps+1;
    end
end

% 2. compare slopes: is it steeper for one attribute, or for different stimulus sets?
% no, so no significant differences in learning between choice attributes or stimuli
session_id = unique([s_attr.subject' s_attr.session' s_attr.attribute' s_attr.stimulus_set'], "rows");
slopes = nan(length(session_id), 1);

for j = 1:length(session_id)
    i = s_attr.subject==session_id(j,1) & s_attr.session==session_id(j,2);

    left_value  = s_attr.left_value(i);
    right_value = s_attr.right_value(i);
    chose_left  = s_attr.chose_left(i);
    
    [stats_out] = plot_psych_curve(left_value, right_value, chose_left);
    slopes(j) = 1/stats_out.beta(2);
end

[~, ~, ~, ttest_stats] = ttest2(slopes(session_id(:,1)==2 & session_id(:,3)==1), slopes(session_id(:,1)==2 & session_id(:,3)==2));

%% learning sets
% show a learning set once the second stimulus set is introduced
% normally measured as trials to criterion, we measure:
% (i) number of sessions to reach 85% accuracy in probe choices
% (ii) number of trials to reach 85% accuracy over a block of 20 trials

% set thresholds
criterion = 0.85;
n_trials  = 20;

% 1. compute performance
sessions_to_crit = nan(length(subjects), length(choice_attributes), length(stim_sets));   
trials_to_crit   = nan(length(subjects), length(choice_attributes), length(stim_sets));

for subj = 1:length(subjects)
    for attr = 1:length(choice_attributes)
        sets = unique(s_attr.stimulus_set(s_attr.subject==subj & s_attr.attribute==attr));      
        for stim = 1:length(sets)
            i = s_attr.subject==subj & s_attr.attribute==attr & s_attr.stimulus_set==sets(stim) & s_attr.probe_test==1;             
    
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
plot_params.Markers = {'o', 'd'};
plot_params.Colours = attr_colours;
figure(); set(gcf,'color','w');

% bar plot
b_plot = bar(mean_vals, 'FaceColor', [0.87 0.87 0.87], 'EdgeColor', 'k');
xlabel('stimulus set'); ylabel('trials to criterion');
ylim([0 180]); yticks(0:60:180)
box off
hold on

% add error bars
errorbar(1:2, mean_vals, sem_vals,'Color', [0.2 0.2 0.2], "LineWidth", 1, 'LineStyle', 'none');

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
        s = scatter(x_vals, y_vals', 60, 'filled', ...
                'MarkerFaceColor', plot_params.Colours(attr, :), ...
                'MarkerEdgeColor', 'k', ...
                'Marker', plot_params.Markers(subj));
        
    end
end
