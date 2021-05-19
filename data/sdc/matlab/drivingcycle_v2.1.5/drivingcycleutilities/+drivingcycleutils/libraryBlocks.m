function blocks_cell = libraryBlocks()
%drivingcycleutils.libraryBlocks
%   blocks_cell = drivingcycleutils.libraryBlocks()

modelName = drivingcycleutils.libraryModelName();
if ~isloaded(modelName)
    load_system(modelName);
    close_onCleanup = onCleanup(@()close_system(modelName, 0));
end

blocksFull_cell = find_system(modelName, 'SearchDepth', 1, ...
    'LookUnderMasks', 'off');

nonRootIdx = ~strcmp(modelName, blocksFull_cell);

blocks_cell = blocksFull_cell(nonRootIdx);

end


function success = isloaded(modelName)
success = false;
try %#ok<TRYNC>
    get_param(modelName, 'Solver');
    success = true;
end
end 
