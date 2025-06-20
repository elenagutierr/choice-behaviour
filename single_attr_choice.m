%% analyse single-attribute choice behaviour in a value-based decision-making task 
% run a standard battery of tests to analyse choice behaviour in a single attribute binary choice task with multiple stimulus sets. analyses include:
% 1. plotting a psychometric curve 
% 2. testing learning set across stimulus sets
% 3. test performance as a function of choice value difference
%
% data (m_attr) is a struct with fields [1 x trials]. minimum data requirements are: 
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
subjects = {'A', 'T', 'V'};
stim_sets = 1:2;

% in my task, only a subset of trials involve binary choices. 
% subset the trials of interest here: 
choice_trials = 'probe_test';   % logical for trials of interest  
s_attr.(choice_trials) = logical(s_attr.(choice_trials));

% colours
stim_colours = [255, 69, 0; 255, 140, 0; 102, 51, 0]./255;
attr_colours = [0, 100, 0; 153, 204 47; 25, 51, 0]./255;
early_late_colours = [128, 128, 128; 192, 192, 192]./255;

%% psychometric curves 
% plot psychometric curves and compare across subjects, attributes and stimulus sets
min_trial_n = 40; % for session-by-session analyses, we want at least a whole block to have been completed

% 1. plot across all sessions

% set plotting parameters
plot_params.LineWidth = 2;
plot_params.MarkerSize = 10;
plot_params.Color = cat(3, attr_colours, stim_colours);
plot_params.Legend = {'Magnitude Set 1','Probability Set 1','Magnitude Set 2','Probability Set 2'};

plot_params.inset.LineWidth  = 2;
plot_params.inset.MarkerSize = 10;
plot_params.inset.Color = cat(3, early_late_colours, attr_colours(1:2,:));

% for one of the subjects, we also plot an inset to compare early vs late training sessions (n = inset_sessions)
inset_sessions = 5; 
inset_subj = 2;

figure(); set(gcf,'color','w');
m = 1; n = length(subjects); 

for subj = 1:length(subjects)
    subplot(m, n, subj)
    for attr = 1:length(choice_attributes)        
        for s = 1:length(stim_sets)

            % index 
            i = s_attr.subject==subj & s_attr.attribute==attr & s_attr.stimulus_set==s & s_attr.(choice_trials);
            left_value  = s_attr.left_value(i);
            right_value = s_attr.right_value(i);
            chose_left  = s_attr.chose_left(i);

            [~, p(:, attr, s)] = plot_psych_curve(left_value, right_value, chose_left, plot_params, [attr s]);
            hold on
        end   
    end
    % label plot
    xl = xlabel('Left-Right Value Difference'); yl = ylabel('P(Chose left)');
    tidy_labels(gca, yl, xl, 0.05)
    title(subjects{subj})
    xticks(-9:3:9); yticks(0:0.25:1);
    box off
    hold off

    % add inset plot for toto
    if subj == inset_subj    
        % set inset axes position and adjust it based on current subplot position
        parent_pos = get(gca, 'Position');  % get subplot position
        inset_pos = [0.70 0.1 0.3 0.6];     % [left bottom width height] in normalized units
        inset_abs = [ parent_pos(1) + inset_pos(1) * parent_pos(3), ...
            parent_pos(2) + inset_pos(2) * parent_pos(4), ...
            inset_pos(3) * parent_pos(3), ...
            inset_pos(4) * parent_pos(4)];
        
        % create the inset axes
        inset_axes = axes('Position', inset_abs);
        
        % plot the inset
        for attr_in = 1:length(choice_attributes)
            i = s_attr.subject==subj & s_attr.attribute==attr_in & s_attr.stimulus_set==1;
            these_sessions = unique(s_attr.session(i));
            these_session_i = [these_sessions(1:inset_sessions); these_sessions(end-inset_sessions+1:end)];
            for sess = 1:size(these_session_i,1)
                inset_i = s_attr.subject==subj & s_attr.attribute==attr_in & s_attr.stimulus_set==1 & ismember(s_attr.session, these_session_i(sess,:)) & s_attr.(choice_trials);
                
                left_value  = s_attr.left_value(inset_i);
                right_value = s_attr.right_value(inset_i);
                chose_left  = s_attr.chose_left(inset_i);
    
                [~, p_ins(:, attr_in, s)] = plot_psych_curve(left_value, right_value, chose_left, plot_params.inset, [attr_in, sess]);
                hold on
            end
        end
        xticks([-9 0 9]); yticks(0:0.5:1);
        title('early vs late')
    end

end
ha = axes('Position',[0 0 1 1],'Visible','off');
p_leg = p(2,:,:);
legend(ha, p_leg(:), plot_params.Legend, 'Location','northoutside', 'Orientation', 'horizontal');

% 2. compare curves between choice attributes and stimulus sets
session_id = unique([s_attr.subject' s_attr.session' s_attr.attribute' s_attr.stimulus_set'], "rows");
session_id = session_id(session_id(:,4)<=length(stim_sets), :);
slopes      = nan(length(session_id), 1); % measures discrimination - how quickly do the subjects switch from guessing to correct response?
x_threshold = nan(length(session_id), 1); % measures bias - at what stimulus value does response probability reach 50%?

for j = 1:length(session_id)
    i = s_attr.subject==session_id(j,1) & s_attr.session==session_id(j,2) & s_attr.(choice_trials);

    if sum(i)<min_trial_n, continue, end % this means less than one full task block was completed, so we exclude the session

    left_value  = s_attr.left_value(i);
    right_value = s_attr.right_value(i);
    chose_left  = s_attr.chose_left(i);
    
    [stats_out] = plot_psych_curve(left_value, right_value, chose_left);
    slopes(j)     = 1 / stats_out.beta(2);
    x_threshold(j) = -stats_out.beta(1) / stats_out.beta(2);
end

% now we test the effect of choice attribute and stimulus set with a linear mixed-effect model 
for s = 1:length(subjects)
    i = session_id(:,1)==s & ~isnan(slopes) & ~isnan(x_threshold);

    lmm_slopes   = table(slopes(i), categorical(session_id(i, 3)), categorical(session_id(i, 4)), categorical(session_id(i, 2)),...
        'VariableNames', {'slope', 'choice_attr', 'stim_set', 'session'});
    lmm_x_thresh = table(x_threshold(i), categorical(session_id(i, 3)), categorical(session_id(i, 4)), categorical(session_id(i, 2)),...
        'VariableNames', {'x_threshold', 'choice_attr', 'stim_set', 'session'});

    slope_model     = fitlme(lmm_slopes,'slope ~ choice_attr*stim_set + (1|session)');
    x_thresh_model  = fitlme(lmm_x_thresh,'x_threshold ~ choice_attr*stim_set + (1|session)');
    slope_aov(s)    = anova(slope_model); 
    x_thresh_aov(s) = anova(x_thresh_model); 

    if s == inset_subj       
        for j = 1:2
            % index sessions
            these_sessions_i = [];
            for stim = 1:length(stim_sets)
                for att = 1:(length(choice_attributes))
                    ind = find(session_id(:,1)==s & session_id(:,3)==att & session_id(:,4)==stim);            
                    if j==1, these_sessions_i = [these_sessions_i; ind(1:inset_sessions)];
                    else, these_sessions_i = [these_sessions_i; ind(end-inset_sessions+1:end)];
                    end

                end
            end
            i = session_id(:,1)==s & ~isnan(slopes) & ~isnan(x_threshold) & ismember(session_id(:,2), session_id(these_sessions_i,2));

            lmm_slopes   = table(slopes(i), categorical(session_id(i, 3)), categorical(session_id(i, 4)), categorical(session_id(i, 2)),...
                'VariableNames', {'slope', 'choice_attr', 'stim_set', 'session'});
            lmm_x_thresh = table(x_threshold(i), categorical(session_id(i, 3)), categorical(session_id(i, 4)), categorical(session_id(i, 2)),...
                'VariableNames', {'x_threshold', 'choice_attr', 'stim_set', 'session'});
        
            slope_model     = fitlme(lmm_slopes,'slope ~ choice_attr*stim_set + (1|session)');
            x_thresh_model  = fitlme(lmm_x_thresh,'x_threshold ~ choice_attr*stim_set + (1|session)');
            slope_aov_in(j)    = anova(slope_model); 
            x_thresh_aov_in(j) = anova(x_thresh_model); 
        end
    end
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
blocks_to_crit   = nan(length(subjects), length(choice_attributes), length(stim_sets));

for subj = 1:length(subjects)
    for attr = 1:length(choice_attributes)
        sets = unique(s_attr.stimulus_set(s_attr.subject==subj & s_attr.attribute==attr));      
        for stim = 1:length(stim_sets)
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

            % forced choice blocks to criterion    
            n_blocks_tmp = 1;
            crit_reached = 0;
            for s = 1:length(sessions)
                session_i = i & s_attr.session == sessions(s);
                train_blocks = unique(s_attr.probe_block(session_i));

                for block = 1:length(train_blocks)
                    block_i = s_attr.probe_block==train_blocks(block) & session_i;
                    accuracy = sum(s_attr.correct(block_i)) / length(s_attr.correct(block_i));
                    if accuracy >= criterion
                        blocks_to_crit(subj, attr, stim) = n_blocks_tmp; % store the session number if criterion is reached, then exit loop
                        crit_reached = 1;
                        break;
                    else
                        n_blocks_tmp = n_blocks_tmp+1;       
                    end    
                end
                if crit_reached
                    break
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
sem_vals  = squeeze(std(trials_to_crit, 0, [1 2]) ./ sqrt(6));     

% set plot parameters
plot_params.Markers = {'o', 'd', 's'};  % subjects
plot_params.Color = attr_colours;       % choice attributes
figure(); set(gcf,'color','w');

% bar plot
b_plot = bar(mean_vals, 'FaceColor', [0.87 0.87 0.87], 'EdgeColor', 'k');
xl = xlabel('stimulus set'); yl = ylabel('trials to criterion');
tidy_labels(gca, yl, xl, 0.05)
% ylim([0 180]); yticks(0:60:180)
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
accuracy = nan(length(session_id),1);
n_trials = nan(length(session_id),1);

value_diff_ranks = unique(abs(s_attr.left_value(s_attr.(choice_trials)) - s_attr.right_value(s_attr.(choice_trials))));
acc_value_diff   = nan(length(session_id), max(value_diff_ranks));

for s = 1:length(session_id)
    i = s_attr.subject==session_id(s,1) & s_attr.session==session_id(s,2) & s_attr.(choice_trials);
    all_i = s_attr.subject==session_id(s,1) & s_attr.session==session_id(s,2);
    accuracy(s) = sum(s_attr.correct(i)) / length(s_attr.correct(i));
    n_trials(s) = sum(all_i);
    
    value_diff = abs(s_attr.left_value(i) - s_attr.right_value(i));  
    correct = s_attr.correct(i);

    for j = 1:length(value_diff_ranks)
        ii = value_diff==value_diff_ranks(j);
        acc_value_diff(s, j) = sum(correct(ii)) / length(correct(ii));
    end
end


% set plotting parameters
plot_params.LineWidth = 2;
plot_params.LineStyle = "-";
plot_params.Color = cat(3, attr_colours, stim_colours);
plot_params.Legend = {'Magnitude Set 1','Probability Set 1','Magnitude Set 2','Probability Set 2'};
plot_params.plot_trials = 0;    % set to 1 if you'd also like to plot trial numbers

figure, set(gcf,'color','w');
m = 1+plot_params.plot_trials; n = length(subjects);

p_acc = 1;
p_tr  = 2;
for subj = 1:length(subjects)

    n_sessions = [];

    % plot accuracy by subject
    subplot(m, n, p_acc)
    for stim = 1:length(stim_sets)
        for attr = 1:length(choice_attributes)
            i = session_id(:,1)==subj & session_id(:, 3)==attr & session_id(:, 4)==stim;
            this_accuracy = accuracy(i)*100;
            plot(this_accuracy(~isnan(this_accuracy)),'Color', plot_params.Color(attr, :, stim),'LineWidth', plot_params.LineWidth)
            hold on

            n_sessions = [n_sessions sum(i)];
        end
    end
    max_sessions = max(n_sessions);
    hold off
    box off
    yline(50,'--k'); 
    title(subjects(subj))
    xlim([1 max_sessions]); ylim([25 100]);
    xticks([1 ceil(max_sessions/2) max_sessions]); yticks(25:25:100);
    xl = xlabel('session'); yl = ylabel('accuracy (%)');
    tidy_labels(gca, yl, xl, 0.05)

    % creat an inset plot for toto
    if subj==2
        inset_sessions = 5;

        % set inset axes position and adjust it based on current subplot position
        parent_pos = get(gca, 'Position');  % get subplot position
        inset_pos = [0.75 0.1 0.2 0.65];  % [left bottom width height] in normalized units
        inset_abs = [ parent_pos(1) + inset_pos(1) * parent_pos(3), ...
            parent_pos(2) + inset_pos(2) * parent_pos(4), ...
            inset_pos(3) * parent_pos(3), ...
            inset_pos(4) * parent_pos(4)];
        
        % create the inset axes
        inset_axes = axes('Position', inset_abs);
        
        % Plot something in the inset
        for stim = 1:length(stim_sets)
        for attr = 1:length(choice_attributes)
            i = session_id(:,1)==subj & session_id(:, 3)==attr & session_id(:, 4)==stim;
            inset_acc = accuracy(i)*100;
            p(attr, stim) = plot(inset_acc(1:inset_sessions),'Color', plot_params.Color(attr, :, stim),'LineWidth', plot_params.LineWidth);
            hold on
        end
        end
        yline(50,'--k'); 
        xlim([1 inset_sessions]); ylim([25 100]);
        xticks([1 inset_sessions]); yticks([25 100]);
    end

    p_acc = p_acc+n;

    if plot_params.plot_trials
        % plot number of trials by subject
        subplot(m, n, p_tr)
        for stim = 1:length(stim_sets)
            for attr = 1:length(choice_attributes)
                i = session_id(:,1)==subj & session_id(:, 3)==attr & session_id(:, 4)==stim;
                these_trials = n_trials(i);
                plot(these_trials(~isnan(these_trials)),'Color', plot_params.Color(attr, :, stim),'LineWidth', plot_params.LineWidth)
                hold on
            end
        end
        hold off
        box off
        y = ceil(max(n_trials(session_id(:, 1)==subj))./10).*10;
        xlim([1 max_sessions]); ylim([0 y]);
        xticks([1 ceil(max_sessions/2) max_sessions]); yticks([0 y/2 y]);
        xl = xlabel('session'); yl = ylabel('number of trials');
        tidy_labels(gca, yl, xl, 0.05)
        legend(plot_params.Legend,'location','southeast')
    
        p_tr = p_tr+n;
    end
end
ha = axes('Position',[0 0 1 1],'Visible','off');
legend(ha, p(:), plot_params.Legend, 'Location','northoutside', 'Orientation', 'horizontal');


% 2. plot accuracy as a function of value difference
figure, set(gcf,'color','w');
m = length(subjects); n = 1;

plot_params.Color = [attr_colours(1:2,:); stim_colours(1:2,:)];
err_params.Color = [0.2 0.2 0.2];   % plotting parameters for error bars
err_params.LineWidth = 1;

pl = 1;
for subj = 1:length(subjects)
        mean_acc = [];
        sem_acc  = [];
        for stim = 1:length(stim_sets)
            for attr = 1:length(choice_attributes)
                i = session_id(:,1)==subj & session_id(:,3)==attr & session_id(:,4)==stim;
                mean_acc = [mean_acc mean(acc_value_diff(i,:), 'omitnan')'];
                sem_acc  = [sem_acc (std(acc_value_diff(i,:), 0, 1, 'omitnan') ./ sqrt(size(acc_value_diff(i,:), 1)))'];
            end
        end

        subplot(m, n, pl)
        b = bar(mean_acc*100, 'FaceColor', 'flat');
        for k = 1:size(mean_acc,2), b(k).CData = squeeze(plot_params.Color(k, :)); end
        hold on
        plot_grouped_errorbars(b, mean_acc*100, sem_acc*100, err_params);
        yline(50,'k--')
        ylim([0 100]); yticks(0:25:100)
        xticklabels(value_diff_ranks)
        yl = ylabel('accuracy (%)');
        xl = xlabel('ranked value difference');
        tidy_labels(gca, yl, xl, 0.05)
        title(subjects{subj})
        box off

        pl = pl+1;
    
end

% now we test the effect of stimulus set, choice attribute and value difference on accuracy using a linear mixed-effect model
for subj = 1:length(subjects)
    i = session_id(:,1)==subj;
    acc = acc_value_diff(i,:);
    attr = repmat(session_id(i,3), 9, 1);
    stim = repmat(session_id(i,4), 9, 1);
    sess = repmat((1:sum(i))', 9, 1);
    v_diff = repmat(1:9, sum(i), 1);

    lmm_accuracy  = acc(:);
    lmm_attribute = categorical(attr(:));
    lmm_stimulus_set = categorical(stim(:));
    lmm_value_diff = categorical(v_diff(:));
    lmm_session = sess(:);
    lmm_data = table(lmm_accuracy, lmm_attribute, lmm_stimulus_set, lmm_value_diff, lmm_session);

    model = fitlme(lmm_data,'lmm_accuracy ~ lmm_attribute*lmm_stimulus_set*lmm_value_diff + (1|lmm_session)');
    disp(anova(model)); 
end