clear;
close all;
visualize_results = false;

method = 'fitting';
repopath = '~/Data/InternetRecon3/%s';
person = 'Andy_Lau';
%person = 'Benedict_Cumberbatch';
%person = 'Donald_Trump';
%person = 'George_W_Bush';

files = dir(fullfile(method, [person, '*']));
for i=1:length(files)
    fprintf('deleting %s\n', files(i).name);
    delete(fullfile(method, files(i).name));
end

AAMfilter_script;