%% plot a psychometric curve 
% INPUT
%   x = value of option X 
%   y = value of option Y     
%   chose_x = logical for choosing x
%   plot_params = struct with plotting parameters
%   plot_index = index for plotting parameters 
% OUTPUT
%   psych_out = plot
%   stats_out = stats for logistic regression

function [stats_out, plot_out] = plot_psych_curve(x, y, chose_x, plot_params, plot_index, ranks)


    value_difference = x - y;

    if nargin > 5   % were predefined ranks provided?
        value_diff_ranks = ranks;
        value_diff_step  = value_diff_ranks(2)-value_diff_ranks(1);
        value_difference = round(value_difference/value_diff_step)*value_diff_step;
    
    else            % if not, compute range of value differences empirically
        
        value_diff_ranks = unique(value_difference);
    end

    % compute choices for each value rank
    n_chose_x = zeros(length(value_diff_ranks),1);                        
    n_rank    = zeros(length(value_diff_ranks),1); 
    for j = 1:length(value_diff_ranks)
        n_chose_x(j) = sum(chose_x(value_difference==value_diff_ranks(j)));
        n_rank(j)    = nnz(value_difference==value_diff_ranks(j));
    end
    
    % logistic regression
    [b_value, ~, stats_out] = glmfit(value_diff_ranks, [n_chose_x n_rank], 'binomial', 'Link', 'logit');   
    
    if nargout > 1
        if length(plot_index) < 2
            plot_index(2) = 1;
        end
        % plot
        yfit_value = glmval(b_value, value_diff_ranks, 'logit', 'size', n_rank);
        plot_out = plot(value_diff_ranks, n_chose_x./n_rank, '.', ...
                        value_diff_ranks, yfit_value./n_rank, '-', ...
                        'LineWidth', plot_params.LineWidth, ...
                        'MarkerSize', plot_params.MarkerSize, ...
                        'Color', plot_params.Color(plot_index(1), :, plot_index(2)));
    end
end
