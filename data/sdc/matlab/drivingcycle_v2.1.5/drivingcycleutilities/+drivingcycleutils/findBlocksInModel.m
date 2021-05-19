function blocks_cell = findBlocksInModel(modelName)
%drivingcycleutils.findBlocksInModel
%   str = drivingcycleutils.findBlocksInModel()

if isequal(nargin, 0)
    modelName = bdroot();
end
    
libBlocks_cell = drivingcycleutils.libraryBlocks();
nLibBlocks = numel(libBlocks_cell);
matchingBlocks = cell(nLibBlocks, 1);
for iLibBlock = 1:nLibBlocks
    thisLibBlock = libBlocks_cell{iLibBlock};
    theseMatches = find_system(modelName, 'ReferenceBlock', thisLibBlock);
    matchingBlocks{iLibBlock} = theseMatches;
end
blocks_cell = vertcat(matchingBlocks{:});

end
