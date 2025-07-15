function out = q_smooth(inp, windowSize, slideWidth)
    % inp is matrix (trials x time), aux has windowSize and slideWidth information

    auxStruct.windowSize = windowSize;
    auxStruct.slideWidth = slideWidth;

    totBins = 1;

    nr = auxStruct.windowSize;

    while nr  < size(inp, 2)
        totBins = totBins + 1;
        nr = nr + auxStruct.slideWidth;
    end

    slice = 1:auxStruct.windowSize;
    for i = 1:totBins
        out(:, i) = mean(inp(:, slice), 2, 'omitnan');

        slice = (slice(1) + auxStruct.slideWidth):(slice(end) + auxStruct.slideWidth);
    end

end