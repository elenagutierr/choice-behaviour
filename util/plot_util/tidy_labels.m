function tidy_labels(ax, yl, xl, offset)

    ax.Units = 'normalized';  % Use relative positioning
    
    % Offset Y label (move further left)
    yl.Units = 'normalized';
    yl.Position(1) = yl.Position(1) - offset;
    
    % Offset X label (move lower)
    xl.Units = 'normalized';
    xl.Position(2) = xl.Position(2) - offset;
end