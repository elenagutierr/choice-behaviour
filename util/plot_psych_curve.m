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

function [stats_out, plot_out] = plot_psych_curve(x, y, chose_x, plot_params, plot_index)

    % compute range of value differences
    value_difference = x - y;                               
    value_diff_ranks = unique(value_difference);

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
        % plot
        yfit_value = glmval(b_value, value_diff_ranks, 'logit', 'size', n_rank);
        plot_out = plot(value_diff_ranks, n_chose_x./n_rank, '.', ...
                        value_diff_ranks, yfit_value./n_rank, '-', ...
                        'LineWidth', plot_params.LineWidth, ...
                        'MarkerSize', plot_params.MarkerSize, ...
                        'Color', plot_params.Colours(plot_index, :));
    end
end
