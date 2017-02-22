datapath = sprintf(repopath, person);
settings_filename = fullfile(datapath, 'settings.txt');

[all_images, all_points] = read_settings(settings_filename);
num_images = length(all_images);

mean_shape_tri = importdata('~/Data/Multilinear/landmarks_triangulation.dat');

h = 250;
w = 250;
nverts = 73;
shapes = zeros(nverts, 2, num_images);
images = zeros(h, w, 3, num_images);
for i=1:num_images
    images(:,:,:,i) = im2double(imread(fullfile(datapath, all_images{i})));
    pts = read_points(fullfile(datapath, all_points{i}));
    shapes(:,:,i) = fliplr(pts);
end

shape_data = shapes;
app_data = images;
triangulation = mean_shape_tri;
build_model_2d_script;

%% fit input images
for i=1:num_images
    %init_shape = reshape(shapes(:,:,i), [], 2) + rand(73,2)*2.5;
    init_shape = (reshape(AAM.s0, [], 2) - repmat(mean(reshape(AAM.s0, [], 2)), 73, 1)) * 0.75 + repmat(mean(reshape(shapes(:,:,i), [], 2)), 73, 1);
    init_shape = 0.25 * init_shape + 0.75 * reshape(shapes(:,:,i), [], 2);
    [fitted_shape fitted_app] = fit_2d(AAM, init_shape, images(:,:,:,i), 20);
    figure(1)
    imshow(images(:,:,:,i));
    hold on;
    triplot(double(AAM.shape_mesh), shapes(:,2,i), shapes(:,1,i), 'b');
    triplot(double(AAM.shape_mesh), init_shape(:,2), init_shape(:,1), '');
    triplot(double(AAM.shape_mesh), fitted_shape(:,2), fitted_shape(:,1), 'r');
    hold off;
    
    figure(2)
    imshow(fitted_app);
    
    pause(2);
end