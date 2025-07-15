function tidy_labels(ax, yl, offset_y, xl, offset_x)

    ax.Units = 'normalized';  % Use relative positioning
    
    if offset_y ~= 0
        % Offset Y label (move further left)
        yl.Units = 'normalized';
        yl.Position(1) = yl.Position(1) - offset_y;
    end
    
    if offset_x ~= 0
        % Offset X label (move lower)
        xl.Units = 'normalized';
        xl.Position(2) = xl.Position(2) - offset_x;
    end
end