function install(mode)
%INSTALL Installation script for Drive Cycle Blockset.
%   INSTALL installs the Drive Cycle Blockset for the current session and
%   all future sessions.
%
%   See also: UNINSTALL.

%   INSTALL has some undocumented modes:
%
%   INSTALL('INSTALL') is the same as INSTALL.
%   INSTALL('UNINSTALL') reverses the installation.
%
%   We've done this so that all the installation/uninstallation code is in
%   the same place. The user is expected to call the functions with INSTALL
%   and UNINSTALL.

%   Copyright 2013-2020 Cranfield University


% Check input arguments.
narginchk(0, 1);
if nargin < 1
    mode = 'install';
end
allowedModes = {'install', 'uninstall'};
if ~ischar(mode) || ~ismember(lower(mode), allowedModes)
    id = 'drivecyclelib:install:unknownModeArg';
    msg = ['Unknown mode argument: allowed values are ''install'' and ' ...
        '''uninstall''.'];
    ME = MException(id, msg);
    throwAsCaller(ME);
end
mode = lower(mode);

% Store the original directory.
originalDir = pwd;
restoreDirFcn = onCleanup(@()cd(originalDir));

% Do the installation/uninstallation.
switch mode
    case 'install'
        doInstall();
    otherwise
        assert(strcmp(mode, 'uninstall'));
        doUninstall();
end

end % install()



function p = projectPath()
%projectPath Return the project path.
%   p = projectPath() returns the project path.

projectRoot = fileparts(mfilename('fullpath'));
mainFolder = fullfile(projectRoot, 'drivingcycle');
dataFolder = fullfile(projectRoot, 'drivingcycledata');
demoFolder = fullfile(projectRoot, 'drivingcycledemos');
utilFolder = fullfile(projectRoot, 'drivingcycleutilities');

PATH_SEP = pathsep();

p = [mainFolder, PATH_SEP, dataFolder, PATH_SEP, demoFolder, PATH_SEP, ...
    utilFolder];

end % projectPath()



function doInstall()
%doInstall Do the installation.
%   doInstall() does the installation, both for the current session and all
%   future ones.

% Add the installation for the current session.
theProjectPath = projectPath();
addpath(theProjectPath, '-end');

% Add the installation for all future sessions.
theCurrentPath = path();
restorePathFcn = onCleanup(@()path(theCurrentPath));
path(pathdef());
addpath(theProjectPath, '-end');
savepath();

% Now we want to do a bit of fixing files.
restorePwd_onCleanup = onCleanup(@()cd(pwd()));
folderList = {libFolder(), demoFolder()};
nFolders = numel(folderList);
for iFolder = 1:nFolders
    thisFolder = folderList{iFolder};
    cd(thisFolder);
    modelList = dir('*.slx');
    nModelFiles = numel(modelList);
    for iModelFile = 1:nModelFiles
        thisModelFile = modelList(iModelFile).name;
        [~, thisModelName] = fileparts(thisModelFile);
        load_system(thisModelName);
        fileattrib(thisModelFile, '+w');
        drivingcycleutils.turnBlockNamesOn(thisModelName);
        save_system(thisModelName);
        force_close(thisModelName);
        fileattrib(thisModelFile, '-w');
    end
    oldVersionFiles = dir('*.slx.r2*');
    nOldVersionFiles = numel(oldVersionFiles);
    for iOldVersionFile = 1:nOldVersionFiles
        thisOldVersionFile = oldVersionFiles(iOldVersionFile).name;
        thisOldVersionFile = fullfile(thisFolder, thisOldVersionFile);
        fileattrib(thisOldVersionFile, '+w');
        fprintf('Deleting %s.\n', thisOldVersionFile);
        delete(thisOldVersionFile);
    end
end

ensureModelOnSimulinkLibraryBrowserPath();

end % doInstall()



function doUninstall()
%doUninstall Do the uninstallation.
%   doUninstall() does the uninstallation, both for the current session and
%   all future ones.

% Turn off warning messages.
S = warning('query', 'all');
[msg, id] = lastwarn();
warnRestoreFcn1 = onCleanup(@()warning(S));
warnRestoreFcn2 = onCleanup(@()lastwarn(msg, id));
warning('off', 'MATLAB:rmpath:DirNotFound');

% Remove the installation for the current session.
theProjectPath = projectPath();
rmpath(theProjectPath);

% Remove the installation for all future sessions.
theCurrentPath = path();
restorePathFcn = onCleanup(@()path(theCurrentPath));
path(pathdef());
rmpath(theProjectPath);
savepath();

end % doUninstall()


function str = rootFolder()
str = fileparts(mfilename('fullpath'));
end


function str = libFolder()
str = fullfile(rootFolder(), 'drivingcycle');
end


function str = demoFolder()
str = fullfile(rootFolder(), 'drivingcycledemos');
end


function ensureModelOnSimulinkLibraryBrowserPath()
modelName = libraryModelName();

fileName = which(modelName);
[~, lastAttribs] = fileattrib(fileName);
if ~lastAttribs.UserWrite
    restoreAttribs_onCleanup = onCleanup(@()fileattrib(fileName, '-w'));
    fileattrib(fileName, '+w');
end

load_system(modelName);
close_onCleanup = onCleanup(@()force_close(modelName));

try
    set_param(modelName, 'Lock', 'off');
    set_param(modelName, 'EnableLBRepository', 'on');
    save_system(modelName);
    lb = LibraryBrowser.LibraryBrowser2;
    refresh(lb);
catch ME
    warning(ME.identifier, '%s', ME.message);
end

end


function str = libraryModelName()
str = 'drivingcycle_lib';
end


function force_close(sys)
try %#ok<TRYNC>
    close_system(sys, 0);
end
end
