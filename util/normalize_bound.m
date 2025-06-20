function out = normalize_bound(inp, lb, ub, varargin)
% normalise input between two margins
% INPUT
%   inp = input, vector [trials x 1] or matrix [trials x time]; if it's a matrix, we normalize within trial
%   lb = lower bound
%   ub = upper bound
% OUPUT
%   out  = normalisd data
% SV

mat_size = size(inp);

if mat_size(1) > 1 && mat_size(2) > 1

    ma = max(inp, [], 2);
    mi = min(inp, [], 2);

    out = (ub - lb) * ((inp - mi) ./ (ma - mi)) + lb;

else
    if isempty(varargin)
        ma = max(inp(:));
        mi = min(inp(:));
    else
        ma = varargin{1};
        mi = varargin{2};
    end
    
    out = (ub - lb) *((inp - mi) ./ (ma - mi)) + lb;
end

end