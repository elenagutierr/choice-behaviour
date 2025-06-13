%% plot error bars on grouped bar plot
% INPUT
%   b = bar plot handle
%   data = bar plot values
%   err = error bar values
%   plot_params = struct with desired plotting parameters

function plot_grouped_errorbars(bar_plot, data, err, plot_params)

    % get number of groups and bars
    [ngroups, nbars] = size(data);
    
    % get the x coordinates of the groups
    x = nan(nbars, ngroups);
    for i = 1:nbars
        x(i,:) = bar_plot(i).XEndPoints;  
    end
    
    % loop through each bar group and add error bars
    for i = 1:nbars
        errorbar(x(i,:), data(:,i), err(:,i), ...
            'Color', plot_params.Color,...
            'LineStyle', 'none',...
            'LineWidth', plot_params.LineWidth);
    end

end