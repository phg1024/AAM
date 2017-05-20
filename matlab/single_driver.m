clear;
close all;
visualize_results = false;

method = 'fitting';
repopath = '~/Storage/Data/InternetRecon3/%s';
person = 'Andy_Lau';
%person = 'George_W_Bush';
%person = 'Hillary_Clinton';
%person = 'Zhang_Ziyi';
%person = 'Benedict_Cumberbatch';
%person = 'Donald_Trump';


files = dir(fullfile(method, [person, '*']));
for i=1:length(files)
    fprintf('deleting %s\n', files(i).name);
    delete(fullfile(method, files(i).name));
end

method = 'robustpca';
AAMfilter_script;

%AAM_script;
