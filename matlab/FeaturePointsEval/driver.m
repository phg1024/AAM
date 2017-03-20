params.feat_window_size = 32;
params.nbins = 8;
params.cell_size = 8;
params.nblocks = 4;

img_path = '/Users/peihongguo/Codes/FaceSeg/InternetRecon3/';
person = 'George_W_Bush';

image_files = dir(fullfile(img_path, person, '*.jpg'));
N = length(image_files);

%% Load images
clear I pts;
I = {};
pts = {};
for i=1:N
    I{i} = rgb2gray(im2double(imread(fullfile(img_path, person, image_files(i).name))));
    [~, name_i, ~] = fileparts(image_files(i).name);
    pts{i} = read_points(fullfile(img_path, person, [name_i, '.pts']));
end

%% Show the images
if false
for i=1:length(I)
    figure(1);imshow(I{i});hold on;draw_shape(pts{i}, 'g.');pause;
end
end

%% Extract features
addpath ./features;
Nfp = size(pts{1},2);
fvectors = zeros(N, Nfp*128);
for i=1:N
    for j=1:Nfp
        x = pts{i}(1,j);
        y = pts{i}(2,j);
        p = extract_patch(I{i}, x, y, params.feat_window_size);
        
        %figure(2);
        %imshow(p);pause;
        
        fvectors(i, (j-1)*128+1:j*128) = HoG(p, [params.nbins, params.cell_size, params.nblocks, 1, 0.2])';
    end
end

%% Visualize features
if false
for j=1:Nfp
    figure(1);plot(fvectors(:,(j-1)*128+1:j*128)');pause;
end
end

%% Build per-feature point model
models = {};
for j=1:Nfp
    fvectors_j = fvectors(:,(j-1)*128+1:j*128);
    models{j}.mean = mean(fvectors_j);
    [coeff, score, latent, tsquared, explained] = pca(fvectors_j - repmat(models{j}.mean, N, 1));
    sum_explained = cumsum(explained);
    idx = find(sum_explained>=25, 1);
    models{j}.pcs = coeff(:,1:idx);
end

%% Visualize recon
total_score = [];
for i=1:N    
    %figure(1);imshow(I{i});hold on;
    total_score(i) = 0;
    for j=1:Nfp
        fvectors_ij = fvectors(i,(j-1)*128+1:j*128);
        proj_ij = models{j}.pcs' * (fvectors_ij - models{j}.mean)';
        recon_ij = models{j}.pcs * reshape(proj_ij, [], 1) + models{j}.mean';
        score = norm(recon_ij - fvectors_ij');
        total_score(i) = total_score(i) + score;
        %plot(pts{i}(1,j), pts{i}(2,j), '.', 'color', [score, 0.5, 1-score], 'markersize', 15);
        %title(num2str(total_score));
    end
    %pause;
end

%% Show the ordering
[sorted_score, order] = sort(total_score, 2, 'descend');
for i=1:N
    figure(1);imshow(I{order(i)});title(num2str(sorted_score(i)));draw_shape(pts{order(i)}, 'g.');pause;
end