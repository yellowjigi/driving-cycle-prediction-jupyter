function turnBlockNamesOn(modelName)

if isequal(nargin, 0)
    modelName = bdroot();
end

blocks_cell = drivingcycleutils.findBlocksInModel(modelName);
nBlocks = numel(blocks_cell);
for iBlock = 1:nBlocks
    thisBlock = blocks_cell{iBlock};
    try
        nameSetting = get_param(thisBlock, 'HideAutomaticName');
        if isequal(nameSetting, 'on')
            set_param(thisBlock, 'HideAutomaticName', 'off');
        end
    catch ME
        warning(ME.identifier, '%s', ME.message);
    end
end

end
