function path_str = docPath(theBlock) %#ok<INUSD>
%drivingcycleutils.docPath
%   path_str = drivingcycleutils.docPath(theBlock)

narginchk(1, 1);
shortName_str = 'drivingcycleblock.html';
path_str = fullfile(drivingcycleutils.docFolder(), shortName_str);

end
