function blkStruct = slblocks
%SLBLOCKS Defines the block library for Driving Cycle Blockset.
%   blkStruct = slblocks()

blkStruct.Name = sprintf('Driving Cycles Blockset');
blkStruct.OpenFcn = 'drivingcycle_lib';
blkStruct.MaskDisplay = '';

Browser(1).Library = 'drivingcycle_lib';
Browser(1).Name    = 'Driving Cycle Blockset';
Browser(1).IsFlat  = 1;% Is this library "flat" (i.e. no subsystems)?

blkStruct.Browser = Browser;
