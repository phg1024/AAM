datapath = sprintf(repopath, person);
settings_filename = fullfile(datapath, 'settings.txt');

[all_images, all_points] = read_settings(settings_filename);
num_images = length(all_images);

for i=1:length(all_images)
    I{i} = imread(fullfile(datapath, all_images{i}));
    pts{i} = read_points(fullfile(datapath, all_points{i}));

    % scale up
    scale_factor = 1.0;
    I{i} = imresize(I{i}, scale_factor);
    pts{i} = pts{i} * scale_factor;

    shape{i} = reshape(pts{i}', [], 1);
end

all_shapes = cell2mat(shape);

all_aligned_shapes = all_shapes;
mean_shape = all_aligned_shapes(:,1);

for iter=1:5
    new_mean_shape = mean(all_aligned_shapes, 2);

    new_mean_shape = scale_shape(new_mean_shape, 250*0.95);

    if norm(new_mean_shape - mean_shape) < 1e-4
        break;
    end
    fprintf('%d: %.6f\n', iter, norm(new_mean_shape - mean_shape));

    mean_shape = new_mean_shape;
    if visualize_results
        figure(1); draw_shape(mean_shape); axis equal; view([0 -90]);
        pause;
    end

    % align every shape to the mean shape
    for i=1:length(all_images)
        aligned_shape{i} = align_shape(shape{i}, mean_shape);
        if visualize_results
            figure(1); clf; hold on;
            draw_shape(mean_shape);
            draw_shape(aligned_shape{i}, 'r.');
            pause;
        end
    end
    all_aligned_shapes = cell2mat(aligned_shape);
end

% build shape model
all_aligned_shapes = cell2mat(aligned_shape);
all_aligned_shapes = all_aligned_shapes - repmat(mean_shape, 1, size(all_aligned_shapes,2));
[coeff, score, latent, tsquared, explained] = pca(all_aligned_shapes');
total_explained = cumsum(explained);
num_modes = find(total_explained>98, 1, 'first');
model.shape.num_modes = num_modes;
model.shape.x = mean_shape;
model.shape.P = coeff(:, 1:num_modes);

% compute global transformation basis
model.shape.s0 = mean_shape;
model.shape.ss_1 = model.shape.s0;
model.shape.ss_2 = reshape(mean_shape, 2, []);
model.shape.ss_2 = [-model.shape.ss_2(2,:);model.shape.ss_2(1,:)];
model.shape.ss_2 = reshape(model.shape.ss_2, [], 1);
model.shape.ss_3 = zeros(length(mean_shape), 1);
model.shape.ss_3(1:2:end) = 1;
model.shape.ss_4 = zeros(length(mean_shape), 1);
model.shape.ss_4(2:2:end) = 1;
model.shape.s_star = [model.shape.ss_1, model.shape.ss_2, model.shape.ss_3, model.shape.ss_4];
model.shape.s = model.shape.P;
% TODO we may need to orthogonalize s and s_star

% create warping
tic;
mean_shape_verts = reshape(mean_shape, 2, [])';
mean_shape_tri0 = delaunay(mean_shape_verts(:,1), mean_shape_verts(:,2));
save('triagulation.mat', 'mean_shape_tri0');

mean_shape_tri = importdata('~/Data/Multilinear/landmarks_triangulation.dat');
%pause;

for i=1:length(all_images)
    xi = pts{i}(:,1);
    yi = pts{i}(:,2);
    tri_i = mean_shape_tri;

    if visualize_results
        figure(1); clf;
        imshow(I{i}); hold on;
        triplot(tri_i, xi, yi);
        plot(xi, yi, 'g.');
        pause;
    end
end

[h, w, d] = size(I{1});
for j=1:length(mean_shape_tri)
    tri_indices{j} = mean_shape_tri(j,:);
    tri_verts_j = mean_shape_verts(tri_indices{j},:);
    mean_mask{j} = repmat(poly2mask(tri_verts_j(:,1), tri_verts_j(:,2), h, w), [1, 1, d]);
end

mean_shape_poly_k = convhull(mean_shape_verts(:,1), mean_shape_verts(:,2));
mean_shape_poly = mean_shape_verts(mean_shape_poly_k,:);
mean_shape_region = poly2mask(mean_shape_poly(:,1), mean_shape_poly(:,2), h, w);
toc;

mask_sum = mean_mask{1};
for j=2:length(mean_mask)
    mask_sum = mask_sum + mean_mask{j};
end
if true%visualize_results
    figure(3);
    subplot(1,2,1);imshow(mask_sum);hold on;trimesh(mean_shape_tri0, mean_shape_verts(:,1), mean_shape_verts(:,2), ones(size(mean_shape_verts, 1), 1));
    subplot(1,2,2);imshow(mask_sum);hold on;trimesh(mean_shape_tri, mean_shape_verts(:,1), mean_shape_verts(:,2), ones(size(mean_shape_verts, 1), 1));
%     for j=1:size(mean_shape_tri,1)
%         cj = mean(mean_shape_verts(mean_shape_tri(j,:), :));
%         text(cj(1), cj(2), 1.1, num2str(j));
%     end
%    pause;
end

% compute jabocian of warp function \frac{\mathbf W}{\mathbf p}
figure; imshow(mask_sum);
mask_pixels = find(mask_sum(:,:,1) > 0);
num_triangles = size(mean_shape_tri);
pixel_triangle_indices = zeros(h*w,1);
alphas = zeros(h*w, 1);
betas = zeros(h*w, 1);
num_vertices = size(mean_shape_verts, 1);
pw_px = cell(num_vertices, 1);
for i=1:num_vertices
    pw_px{i} = zeros(h*w, 1);
end
for j=1:num_triangles
    triangles{j} = poly2mask(mean_shape_verts(mean_shape_tri(j,:),1), mean_shape_verts(mean_shape_tri(j,:),2), h, w);
    pixels_in_tri_j = find(triangles{j} > 0);
    pixel_triangle_indices(pixels_in_tri_j) = j;

    v1 = mean_shape_verts(mean_shape_tri(j,1), :);
    v2 = mean_shape_verts(mean_shape_tri(j,2), :);
    v3 = mean_shape_verts(mean_shape_tri(j,3), :);

    % for the first vertex
    v2_v1 = v2 - v1;
    v3_v1 = v3 - v1;
    for pi=1:length(pixels_in_tri_j)
        pixel_idx = pixels_in_tri_j(pi);
        pi_x = min(floor(pixel_idx / h) + 1, w);
        pi_y = mod(pixel_idx, h) + 1;
        vi = [pi_x, pi_y];
        vi_v1 = vi - v1;
        denom_i = v2_v1(1)*v3_v1(2) - v3_v1(1)*v2_v1(2);
        alphas(pixel_idx) = (vi_v1(1) * v3_v1(2) - vi_v1(2) * v3_v1(1)) / denom_i;
        betas(pixel_idx) = (vi_v1(2) * v2_v1(1) - vi_v1(1) * v2_v1(2)) / denom_i;

        pw_px{mean_shape_tri(j,1)}(pixel_idx) = 1-alphas(pixel_idx)-betas(pixel_idx);
    end

    % for the second vertex
    v1_v2 = v1 - v2;
    v3_v2 = v3 - v2;
    for pi=1:length(pixels_in_tri_j)
        pixel_idx = pixels_in_tri_j(pi);
        pi_x = min(floor(pixel_idx / h) + 1, w);
        pi_y = mod(pixel_idx, h) + 1;
        vi = [pi_x, pi_y];
        vi_v2 = vi - v2;
        denom_i = v1_v2(1)*v3_v2(2) - v3_v2(1)*v1_v2(2);
        alphas(pixel_idx) = (vi_v2(1) * v3_v2(2) - vi_v2(2) * v3_v2(1)) / denom_i;
        betas(pixel_idx) = (vi_v2(2) * v1_v2(1) - vi_v2(1) * v1_v2(2)) / denom_i;

        pw_px{mean_shape_tri(j,2)}(pixel_idx) = 1-alphas(pixel_idx)-betas(pixel_idx);
    end

    % for the third vertex
    v1_v3 = v1 - v3;
    v2_v3 = v2 - v3;
    for pi=1:length(pixels_in_tri_j)
        pixel_idx = pixels_in_tri_j(pi);
        pi_x = min(floor(pixel_idx / h) + 1, w);
        pi_y = mod(pixel_idx, h) + 1;
        vi = [pi_x, pi_y];
        vi_v3 = vi - v3;
        denom_i = v1_v3(1)*v2_v3(2) - v2_v3(1)*v1_v3(2);
        alphas(pixel_idx) = (vi_v3(1) * v2_v3(2) - vi_v3(2) * v2_v3(1)) / denom_i;
        betas(pixel_idx) = (vi_v3(2) * v1_v3(1) - vi_v3(1) * v1_v3(2)) / denom_i;

        pw_px{mean_shape_tri(j,3)}(pixel_idx) = 1-alphas(pixel_idx)-betas(pixel_idx);
    end
end
pw_py = pw_px;  % the only difference is that pw_px is valid for x channel and pw_py is valid for y channel
figure;imshow(reshape(pixel_triangle_indices, h, w)/255.0+0.5);
for i=1:num_vertices
    figure(3);clf;
    imshow(reshape(pw_px{i}, h, w));title(['pw\_px ', num2str(i)]);
    hold on;
    trimesh(mean_shape_tri, mean_shape_verts(:,1), mean_shape_verts(:,2), ones(1,num_vertices), 'facecolor', 'none', 'edgecolor', [.5 .5 .75]);
    plot(mean_shape_verts(i, 1), mean_shape_verts(i, 2), 'g.');
    pause;
end

% compute the jacobian of each shape component
num_shape_components = size(model.shape.P, 2);
px_pp = zeros(num_vertices, num_shape_components);
py_pp = zeros(num_vertices, num_shape_components);
px_pq = zeros(num_vertices, 4);
py_pq = zeros(num_vertices, 4);
for i=1:num_vertices
    for j=1:num_shape_components
        px_pp(i,j) = model.shape.P(2*(i-1)+1, j);
        py_pp(i,j) = model.shape.P(2*(i-1)+2, j);
    end
    for j=1:4
        px_pq(i,j) = model.shape.s_star(2*(i-1)+1, j);
        py_pq(i,j) = model.shape.s_star(2*(i-1)+2, j);
    end
end

% visualize the composite jacobian for each shape mode
for j=1:num_shape_components
    pw_pp_x = zeros(h*w, 1);
    pw_pp_y = zeros(h*w, 1);
    for i=1:num_vertices
        pw_pp_x = pw_pp_x + pw_px{i} * px_pp(i,j);
        pw_pp_y = pw_pp_y + pw_py{i} * py_pp(i,j);
    end
    figure(3);clf;
    subplot(1,3,1);imshow(reshape(pw_pp_x, h, w));hold on;
    trimesh(mean_shape_tri, mean_shape_verts(:,1), mean_shape_verts(:,2), ones(1,num_vertices), 'facecolor', 'none', 'edgecolor', [.5 .5 .75]);
    for i=1:num_vertices
        vsi_x = model.shape.P(2*(i-1)+1, j) * 100;
        vsi_y = model.shape.P(2*(i-1)+2, j) * 100;

        line([mean_shape_verts(i,1), mean_shape_verts(i,1) + vsi_x], ...
             [mean_shape_verts(i,2), mean_shape_verts(i,2) + vsi_y]);
        line([mean_shape_verts(i,1), mean_shape_verts(i,1) - vsi_x], ...
             [mean_shape_verts(i,2), mean_shape_verts(i,2) - vsi_y], 'Color', [1, 0, 0]);
    end

    subplot(1,3,2);imagesc(reshape(pw_pp_x, h, w));axis equal;
    subplot(1,3,3);imagesc(reshape(pw_pp_y, h, w));axis equal;
    pause;
end

tic;
mean_texture = zeros(size(I{1}));

parfor i=1:length(all_images)
    t_i = tic;
    I_i = im2double(I{i});

    [h, w, d] = size(I_i);
    mask{i} = zeros(size(I_i));
    masked_i = zeros(size(I_i));

    N = length(mean_shape_tri);
    tri_verts = cell(1,N);
    masks  = cell(1, N);
    masked = cell(1, N);
    for j=1:length(mean_shape_tri)
        tri_ij = tri_indices{j};
        tri_verts{j} = pts{i}(tri_ij, :);
        masks{j} = poly2mask(tri_verts{j}(:,1), tri_verts{j}(:,2), h, w);
        masks{j} = repmat(masks{j}, [1, 1, d]);
        masked{j} = I_i .* masks{j};
        mask{i} = mask{i} + masks{j};
        masked_i = masked_i + masked{j};
    end

    % warped masks
    warped{i} = zeros(size(I_i));
    warped_masked = cell(1,N);
    for j=1:length(mean_shape_tri)
        r_j = imref2d(size(I_i));

        if false
            % warp mean_shape_verts to tri_verts{j}
        else
            % warp tri_verts{j} to mean_shape_verts
            tform_j = estimateGeometricTransform(tri_verts{j}, mean_shape_verts(tri_indices{j}, :), 'affine');
            warped_masked{j} = imwarp(I_i, tform_j, 'OutputView', r_j);
            warped{i} = warped{i} + warped_masked{j} .* mean_mask{j};
        end
    end

    if visualize_results
        figure(2); N = 3;
        subplot(1, N, 1); imshow(I{i}); axis on;
        subplot(1, N, 2); imshow(masked_i); axis on;
        subplot(1, N, 3); imshow(warped{i}); axis on;
        pause;
    end
    toc(t_i);
    mean_texture = mean_texture + warped{i};
end

mean_texture = mean_texture / length(I);
if true%visualize_results
    figure(4);imshow(mean_texture);title('mean texture');pause;
end
toc;

% compute A0 and dA0
tic;
[mean_r, mean_g, mean_b] = split_channels(mean_texture);
mean_grad_r = gradient(mean_r); mean_grad_r(~mean_shape_region) = 0;
mean_grad_g = gradient(mean_g); mean_grad_g(~mean_shape_region) = 0;
mean_grad_b = gradient(mean_b); mean_grad_b(~mean_shape_region) = 0;
mean_grad = cat(3, mean_grad_r, mean_grad_g, mean_grad_b);

model.A0 = mean_texture;
model.dA0 = mean_grad;

figure;subplot(1,2,1);imshow(model.A0);
subplot(1,2,2);imshow(model.dA0);pause;
toc;

% compute appearance model
tic;
app_matrix = zeros(h*w*3, num_images);
for i=1:length(I)
    app_matrix(:,i) = reshape(warped{i}, [], 1);
end
app_matrix = app_matrix - repmat(reshape(model.A0, [], 1), 1, num_images);

addpath inexact_alm_rpca/
addpath inexact_alm_rpca/PROPACK/

tic;
app_matrix_low_rank = inexact_alm_rpca(app_matrix);
toc;

[coeff, score, latent, tsquared, explained] = pca(app_matrix_low_rank');
total_explained = cumsum(explained);
num_modes = find(total_explained>98, 1, 'first');
model.A_nmodes = num_modes;
model.A = coeff(:,1:num_modes);
model.A_eiv = latent(1:num_modes);
toc;

% compute steepest descent images
tic;
app_modes = reshape(model.A, [h w 3 num_modes]);
for i=1:num_modes
    figure(5);imshow(app_modes(:,:,:,i)*255);pause;
end

SD = zeros(h, w, 3, 4 + num_vertices);
for i=1:4
    prj_diff = zeros(3, num_modes);
    for j=1:num_modes
        for c=1:3
            %prj_diff(c,j) = sum(sum(app_modes(:,:,c,j) .* ()));
        end
    end
end

toc;


tic;
[mean_r, mean_g, mean_b] = split_channels(mean_texture);
mean_texture_vec = [mean_r(mean_shape_region); mean_g(mean_shape_region); mean_b(mean_shape_region)];
mean_texture_vec0 = mean_texture_vec;

for iter=1:100
    new_mean_texture_vec = zeros(size(mean_texture_vec));
    N = length(mean_texture_vec) / 3;
    for i=1:length(I)
        [Ii_r, Ii_g, Ii_b] = split_channels(warped{i});
        wr = Ii_r(mean_shape_region);
        wg = Ii_g(mean_shape_region);
        wb = Ii_b(mean_shape_region);

        warped_vec_i = [wr; wg; wb];
        alpha_i = dot(warped_vec_i, mean_texture_vec);
        beta_i = [mean(wr) * ones(N,1); mean(wg) * ones(N, 1); mean(wb) * ones(N,1)];
        scaled_warped_vec{i} = (warped_vec_i - beta_i) / alpha_i;
        new_mean_texture_vec = new_mean_texture_vec + scaled_warped_vec{i};
    end
    new_mean_texture_vec = new_mean_texture_vec / length(I);
    fprintf('%d: %.6f, %.6f\n', iter, norm(mean_texture_vec - new_mean_texture_vec), norm(new_mean_texture_vec));

    if norm(mean_texture_vec - new_mean_texture_vec) < 1e-6
        break
    end
    step_alpha = 0.5;
    mean_texture_vec = mean_texture_vec * step_alpha + (1.0 - step_alpha) * new_mean_texture_vec;

    [final_r, final_g, final_b] = split_channels(mean_texture);

    final_r(mean_shape_region) = mean_texture_vec(1:N);
    final_g(mean_shape_region) = mean_texture_vec(N+1:N*2);
    final_b(mean_shape_region) = mean_texture_vec(N*2+1:end);
    final_mean_texture = cat(3, final_r, final_g, final_b);

    if visualize_results
        figure(1);
        subplot(1, 2, 1); imshow(mean_texture); axis equal; colorbar;
        subplot(1, 2, 2); imshow(final_mean_texture); axis equal; colorbar;
        pause;
    end
end
toc;

% build texture model
all_texture_vec = cell2mat(scaled_warped_vec);
all_texture_vec = all_texture_vec - repmat(mean_texture_vec, 1, size(all_texture_vec, 2));

addpath inexact_alm_rpca/
addpath inexact_alm_rpca/PROPACK/

%% Try different methods for detecting failure
%method = 'tournament';
%method = 'geometry';
mkdir(method);

switch method
    case 'geometry'
        fprintf('using tournament method\n');
        exclude_count = 0;
        excluded_set = [];
        current_set = 1:length(I);

        fit_error_thres = 10.0;
        max_fit_error_step = 100.0;
        while max_fit_error_step > fit_error_thres
            t_step=tic;
            fit_error=[];
            for j=1:length(current_set)
                i = current_set(j);

                % build the shape model
                samples_k = setdiff(current_set, i);
                [coeff, score, latent, tsquared, explained] = pca(all_aligned_shapes(:, samples_k)');
                total_explained = cumsum(explained);
                num_modes = find(total_explained>98, 1, 'first');
                model.shape.num_modes = num_modes;
                model.shape.x = mean_shape;
                model.shape.P = coeff(:, 1:num_modes);

                [normalized_shape_i, tform_i] = align_shape(shape{i}, mean_shape);
                svec = model.shape.P' * (normalized_shape_i - model.shape.x);

                % build only 1 models for texture
                num_models = 1;
                model.num_texture_models = num_models;
                model.texture = cell(num_models, 1);
                for k=1:num_models
                    samples_k = setdiff(current_set, i);
                    [coeff, score, latent, tsquared, explained] = pca(all_texture_vec(:, samples_k)');
                    total_explained = cumsum(explained);
                    num_modes = find(total_explained>98, 1, 'first');
                    model.texture{k}.num_modes = num_modes;
                    model.texture{k}.x = mean_texture_vec;
                    model.texture{k}.P = coeff(:, 1:num_modes);
                end

                [Ii_r, Ii_g, Ii_b] = split_channels(warped{i});
                Ir = Ii_r(mean_shape_region);
                Ig = Ii_g(mean_shape_region);
                Ib = Ii_b(mean_shape_region);

                diff_i = zeros(size(I{i}));
                syn_i = zeros(size(I{i}));
                norm_i = 0;
                max_norm_i = 0;
                for k=1:model.num_texture_models
                    % normalize the input vector first
                    normalized_texture_i = [Ir; Ig; Ib];
                    alpha_i = dot(normalized_texture_i, model.texture{k}.x);
                    beta_i = [mean(Ir) * ones(N,1); mean(Ig) * ones(N,1); mean(Ib) * ones(N,1)];
                    normalized_texture_i = (normalized_texture_i - beta_i) / alpha_i;
                    size(normalized_texture_i)
                    size(model.texture{k}.x)
                    size(model.texture{k}.P)
                    tvec = model.texture{k}.P' * (normalized_texture_i - model.texture{k}.x);
                    [s, g] = synthesize(model, svec, tvec, k);

                    max_norm_i = max(max_norm_i, norm(g - normalized_texture_i));
                    norm_i = norm_i + norm(g - normalized_texture_i);

                    % unnormalize the fitted vector
                    g = g * alpha_i + beta_i;
                    [s_r, s_g, s_b] = split_channels(mean_texture);
                    N = length(g) / 3;
                    s_r(mean_shape_region) = g(1:N);
                    s_g(mean_shape_region) = g(N+1:N*2);
                    s_b(mean_shape_region) = g(N*2+1:end);
                    norm(s_r-s_g)
                    norm(s_g-s_b)
                    syn_texture = cat(3, s_r, s_g, s_b);

                    syn_i = syn_i + syn_texture;
                    diff_i = diff_i + abs(syn_texture - im2double(warped{i}));
                end

                syn_i = syn_i / model.num_texture_models;
                diff_i = diff_i / model.num_texture_models;
                norm_i = norm_i / model.num_texture_models;
                diff2_i = warped{i} - syn_i;

                syn{j} = syn_i;
                norm_shape_i = norm(normalized_shape_i - s);
                s = transformPointsInverse(tform_i, reshape(s, 2, [])');
                syn_shape{j} = s;

                if visualize_results
                    figure(2); M = 6;
                    subplot(1, M, 1); imshow(I{i}); hold on; plot(pts{i}(:,1), pts{i}(:,2), 'g.');
                    subplot(1, M, 2); imshow(I{i}); hold on; plot(syn_shape{j}(:,1), syn_shape{j}(:,2), 'r.');
                    subplot(1, M, 3); imshow(I{i}); hold on; plot(pts{i}(:,1), pts{i}(:,2), 'g.');plot(syn_shape{j}(:,1), syn_shape{j}(:,2), 'r.');
                    subplot(1, M, 4); imshow(warped{i});
                    subplot(1, M, 5); imshow(syn_i);
                    subplot(1, M, 6); imagesc(diff_i);axis equal;title(sprintf('norm = %.6f\nnorm shape = %.6f\ndiff = %.6f', norm_i, norm_shape_i, norm(diff2_i(:))));
                    pause;
                end

                fit_error(j) = norm_i;
                fit_error(j) = norm_shape_i;
            end
            [max_fit_error_step, max_j] = max(fit_error);

            if max_fit_error_step < fit_error_thres
                break
            end

            figure(2);set(gcf, 'Position', get(0,'Screensize'));
            to_exclude = current_set(max_j);
            M = 6;
            subplot(1, M, 1); imshow(I{to_exclude}); hold on; plot(pts{to_exclude}(:,1), pts{to_exclude}(:,2), 'g.');
            subplot(1, M, 2); imshow(I{to_exclude}); hold on; draw_shape(reshape(syn_shape{max_j}', [], 1), 'r.');
            subplot(1, M, 3); imshow(I{to_exclude}); hold on; plot(pts{to_exclude}(:,1), pts{to_exclude}(:,2), 'g.');draw_shape(reshape(syn_shape{max_j}', [], 1), 'r.');
            subplot(1, M, 4); imshow(warped{to_exclude});
            subplot(1, M, 5); imshow(syn{max_j});
            diff2_i = warped{to_exclude} - syn{max_j};
            subplot(1, M, 6); imagesc(diff2_i);axis equal;title(sprintf('norm = %.6f\ndiff = %.6f', max_fit_error_step, norm(diff2_i(:))));

            exclude_count = exclude_count + 1;
            max_fit_error(exclude_count) = max_fit_error_step
            excluded_set(exclude_count) = to_exclude
            current_set = setdiff(current_set, [to_exclude])
            toc(t_step);
            %pause;
            pause(1);
        end

        visualize_set(I, pts, excluded_set, struct('saveit', true, 'filename', fullfile(method, [person, '_excluded'])));
        visualize_set(I, pts, current_set, struct('saveit', true, 'filename', fullfile(method, [person, '_filtered'])));
    case 'robustpca'
        fprintf('using robust-pca method\n');

        all_texture_vec = cell2mat(scaled_warped_vec);

        % extracing low-rank matrix approximation
        tic;
        size(all_texture_vec)
        all_texture_vec = inexact_alm_rpca(all_texture_vec);
        toc;

        exclude_count = 0;
        excluded_set = [];
        current_set = 1:length(I);

        use_single_model = true;
        measure_error_in_image_space = true;

        max_fit_error_step = 100.0;
        fit_error_step_thres = 0.15;
        model_explained_factor = 98;
        %while max_fit_error_step > fit_error_step_thres
            t_step=tic;
            fit_error=[];

            if use_single_model
                num_models = 1;
                model.num_texture_models = num_models;
                model.texture = cell(num_models, 1);

                [coeff, score, latent, tsquared, explained] = pca((all_texture_vec(:, current_set) - repmat(mean_texture_vec, 1, length(current_set)))');
                total_explained = cumsum(explained);
                num_modes = find(total_explained>model_explained_factor, 1, 'first');
                model.texture{1}.num_modes = num_modes;
                model.texture{1}.x = mean_texture_vec;
                model.texture{1}.P = coeff(:, 1:num_modes);
            end

            for j=1:length(current_set)
                i = current_set(j);
                % build only 1 models
                model_j = model;
                if ~use_single_model
                    tic;
                    num_models = 1;
                    model_j.num_texture_models = num_models;
                    model_j.texture = cell(num_models, 1);

                    samples_k = setdiff(current_set, i);
                    [coeff, score, latent, tsquared, explained] = pca((all_texture_vec(:, samples_k) - repmat(mean_texture_vec, 1, length(samples_k)))');
                    total_explained = cumsum(explained);
                    num_modes = find(total_explained>98, 1, 'first');
                    model_j.texture{1}.num_modes = num_modes;
                    model_j.texture{1}.x = mean_texture_vec;
                    model_j.texture{1}.P = coeff(:, 1:num_modes);
                    toc;
                end

                [Ii_r, Ii_g, Ii_b] = split_channels(warped{i});
                Ir = Ii_r(mean_shape_region);
                Ig = Ii_g(mean_shape_region);
                Ib = Ii_b(mean_shape_region);

                N = length(Ir(:));

                diff_i = zeros(size(I{i}));
                syn_i = zeros(size(I{i}));

                norm_i = 0;
                max_norm_i = 0;

                % normalize the input vector first
                normalized_texture_i = [Ir; Ig; Ib];
                alpha_i = dot(normalized_texture_i, model_j.texture{1}.x);
                beta_i = [mean(Ir) * ones(N,1); mean(Ig) * ones(N,1); mean(Ib) * ones(N,1)];
                normalized_texture_i = (normalized_texture_i - beta_i) / alpha_i;

                tvec = model_j.texture{1}.P' * (normalized_texture_i - model_j.texture{1}.x);
                [s, g] = synthesize(model_j, zeros(model_j.shape.num_modes, 1), tvec, 1);

                max_norm_i = max(max_norm_i, norm(g - normalized_texture_i));
                norm_i = norm_i + norm(g - normalized_texture_i);

                % unnormalize the fitted vector
                g = g * alpha_i + beta_i;
                [s_r, s_g, s_b] = split_channels(mean_texture);
                %N = length(g) / 3;
                s_r(mean_shape_region) = g(1:N);
                s_g(mean_shape_region) = g(N+1:N*2);
                s_b(mean_shape_region) = g(N*2+1:end);

                syn_texture = cat(3, s_r, s_g, s_b);

                syn_i = syn_i + syn_texture;
                diff_i = diff_i + abs(syn_texture - im2double(warped{i}));

                % warp syn_i back to image space
                syn_image_i = zeros(size(I{i}));
                M = length(mean_shape_tri);
                tri_verts = cell(1,M);
                for k=1:length(mean_shape_tri)
                    tri_ij = tri_indices{k};
                    tri_verts{k} = pts{i}(tri_ij, :);
                    r_j = imref2d(size(I{i}));

                    mask_k = poly2mask(tri_verts{k}(:,1), tri_verts{k}(:,2), h, w);
                    % warp tri_verts{j} to mean_shape_verts
                    tform_j = estimateGeometricTransform(mean_shape_verts(tri_indices{k}, :), tri_verts{k}, 'affine');
                    warped_syn_i = imwarp(syn_i, tform_j, 'OutputView', r_j);
                    syn_image_i = syn_image_i + warped_syn_i .* mask_k;
                end

                diff2_i = warped{i} - syn_i;
                diff3_i = im2double(I{i}) .* mask{i} - syn_image_i;
                syn{j} = syn_i;

                npixels_i = length(find(mask{i}(:,:,1)>0));
                rmse_i = norm(diff3_i(:)) / sqrt(npixels_i);
                if false%visualize_results
                    h_vis_ = figure(2);set(gcf, 'Position', get(0,'Screensize'));
                    subplot(2, 3, 1); imshow(I{i}); hold on; plot(pts{i}(:,1), pts{i}(:,2), 'g.');
                    subplot(2, 3, 2); imshow(syn_image_i);

                    subplot(2, 3, 3); imshow(diff3_i);title(sprintf('diff = %.6f\n RMSE = %.6f\n', norm(diff3_i(:)), rmse_i));
                    subplot(2, 3, 4); imshow(warped{i});
                    subplot(2, 3, 5); imshow(syn_i);
                    subplot(2, 3, 6); imshow(diff_i);title(sprintf('diff = %.6f\n RMSE = %.6f\n', norm(diff2_i(:)), norm(diff2_i(:)) / sqrt(npixels_i)));
                    %saveas(h_vis_, sprintf('result_%03d.jpg', i));
                    pause;
                end
                if true
                  mkdir(sprintf('./robustpca/%s', person));
                  imwrite(I{i}, sprintf('./robustpca/%s/%03d_input.jpg', person, i));
                  imwrite(syn_image_i, sprintf('./robustpca/%s/%03d_synimage.jpg', person, i));
                  %imwrite(diff3_i, sprintf('./robustpca/%s/%03d_syndiff.jpg', person, i));
                  imwrite(warped{i}, sprintf('./robustpca/%s/%03d_warped.jpg', person, i));
                  imwrite(syn_i, sprintf('./robustpca/%s/%03d_syn.jpg', person, i));
                  %imwrite(diff_i, sprintf('./robustpca/%s/%03d_warpeddiff.jpg', person, i));
                  imwrite(mask{i}, sprintf('./robustpca/%s/%03d_mask.jpg', person, i));
                end

                fit_error(j) = rmse_i;
            end
            [max_fit_error_step, max_j] = max(fit_error);

            %if max_fit_error_step < fit_error_step_thres
            %    break
            %end

            figure(2);set(gcf, 'Position', get(0,'Screensize'));
            to_exclude = current_set(max_j);
            subplot(1, 4, 1); imshow(I{to_exclude}); hold on; plot(pts{to_exclude}(:,1), pts{to_exclude}(:,2), 'g.');
            subplot(1, 4, 2); imshow(warped{to_exclude});
            subplot(1, 4, 3); imshow(syn{max_j});
            diff2_i = warped{to_exclude} - syn{max_j};
            subplot(1, 4, 4); imagesc(diff2_i);axis equal;title(sprintf('norm = %.6f\ndiff = %.6f', max_fit_error_step, norm(diff2_i(:))));

            exclude_count = exclude_count + 1;
            max_fit_error(exclude_count) = max_fit_error_step
            excluded_set(exclude_count) = to_exclude
            if use_single_model
                current_set = setdiff(current_set, [to_exclude])
            end
            toc(t_step);
            pause(1);
        %end

        visualize_set(I, pts, excluded_set, struct('saveit', true, 'filename', fullfile(method, [person, '_excluded'])));
        visualize_set(I, pts, current_set, struct('saveit', true, 'filename', fullfile(method, [person, '_filtered'])));
    case 'fitting'
        fprintf('using robust-pca method\n');

        all_texture_vec = cell2mat(scaled_warped_vec);

        % extracing low-rank matrix approximation
        all_texture_vec = inexact_alm_rpca(all_texture_vec);


        %

        exclude_count = 0;
        excluded_set = [];
        current_set = 1:length(I);

        use_single_model = true;
        measure_error_in_image_space = true;

        max_fit_error_step = 100.0;
        fit_error_step_thres = 0.15;
        model_explained_factor = 98;
        while max_fit_error_step > fit_error_step_thres
            t_step=tic;
            fit_error=[];

            if use_single_model
                num_models = 1;
                model.num_texture_models = num_models;
                model.texture = cell(num_models, 1);

                [coeff, score, latent, tsquared, explained] = pca((all_texture_vec(:, current_set) - repmat(mean_texture_vec, 1, length(current_set)))');
                total_explained = cumsum(explained);
                num_modes = find(total_explained>model_explained_factor, 1, 'first');
                model.texture{1}.num_modes = num_modes;
                model.texture{1}.x = mean_texture_vec;
                model.texture{1}.P = coeff(:, 1:num_modes);
            end

            for j=1:length(current_set)
                i = current_set(j);
                % build only 1 models
                model_j = model;
                if ~use_single_model
                    tic;
                    num_models = 1;
                    model_j.num_texture_models = num_models;
                    model_j.texture = cell(num_models, 1);

                    samples_k = setdiff(current_set, i);
                    [coeff, score, latent, tsquared, explained] = pca((all_texture_vec(:, samples_k) - repmat(mean_texture_vec, 1, length(samples_k)))');
                    total_explained = cumsum(explained);
                    num_modes = find(total_explained>98, 1, 'first');
                    model_j.texture{1}.num_modes = num_modes;
                    model_j.texture{1}.x = mean_texture_vec;
                    model_j.texture{1}.P = coeff(:, 1:num_modes);
                    toc;
                end

                [Ii_r, Ii_g, Ii_b] = split_channels(warped{i});
                Ir = Ii_r(mean_shape_region);
                Ig = Ii_g(mean_shape_region);
                Ib = Ii_b(mean_shape_region);

                N = length(Ir(:));

                diff_i = zeros(size(I{i}));
                syn_i = zeros(size(I{i}));

                norm_i = 0;
                max_norm_i = 0;

                % normalize the input vector first
                normalized_texture_i = [Ir; Ig; Ib];
                alpha_i = dot(normalized_texture_i, model_j.texture{1}.x);
                beta_i = [mean(Ir) * ones(N,1); mean(Ig) * ones(N,1); mean(Ib) * ones(N,1)];
                normalized_texture_i = (normalized_texture_i - beta_i) / alpha_i;

                tvec = model_j.texture{1}.P' * (normalized_texture_i - model_j.texture{1}.x);
                [s, g] = synthesize(model_j, zeros(model_j.shape.num_modes, 1), tvec, 1);

                max_norm_i = max(max_norm_i, norm(g - normalized_texture_i));
                norm_i = norm_i + norm(g - normalized_texture_i);

                % unnormalize the fitted vector
                g = g * alpha_i + beta_i;
                [s_r, s_g, s_b] = split_channels(mean_texture);
                %N = length(g) / 3;
                s_r(mean_shape_region) = g(1:N);
                s_g(mean_shape_region) = g(N+1:N*2);
                s_b(mean_shape_region) = g(N*2+1:end);

                syn_texture = cat(3, s_r, s_g, s_b);

                syn_i = syn_i + syn_texture;
                diff_i = diff_i + abs(syn_texture - im2double(warped{i}));

                % warp syn_i back to image space
                syn_image_i = zeros(size(I{i}));
                M = length(mean_shape_tri);
                tri_verts = cell(1,M);
                for k=1:length(mean_shape_tri)
                    tri_ij = tri_indices{k};
                    tri_verts{k} = pts{i}(tri_ij, :);
                    r_j = imref2d(size(I{i}));

                    mask_k = poly2mask(tri_verts{k}(:,1), tri_verts{k}(:,2), h, w);
                    % warp tri_verts{j} to mean_shape_verts
                    tform_j = estimateGeometricTransform(mean_shape_verts(tri_indices{k}, :), tri_verts{k}, 'affine');
                    warped_syn_i = imwarp(syn_i, tform_j, 'OutputView', r_j);
                    syn_image_i = syn_image_i + warped_syn_i .* mask_k;
                end

                diff2_i = warped{i} - syn_i;
                diff3_i = im2double(I{i}) .* mask{i} - syn_image_i;
                syn{j} = syn_i;

                npixels_i = length(find(mask{i}(:,:,1)>0));
                rmse_i = norm(diff3_i(:)) / sqrt(npixels_i);
                if true%visualize_results
                    figure(2);
                    subplot(1, 6, 1); imshow(I{i}); hold on; plot(pts{i}(:,1), pts{i}(:,2), 'g.');
                    subplot(1, 6, 2); imshow(syn_image_i);

                    subplot(1, 6, 3); imshow(diff3_i);title(sprintf('diff = %.6f\n RMSE = %.6f\n', norm(diff3_i(:)), rmse_i));
                    subplot(1, 6, 4); imshow(warped{i});
                    subplot(1, 6, 5); imshow(syn_i);
                    subplot(1, 6, 6); imagesc(diff_i);axis equal;title(sprintf('norm = %.6f\nmax norm = %.6f\ndiff = %.6f', norm_i, max_norm_i, norm(diff2_i(:))));
                    pause;
                end

                fit_error(j) = rmse_i;
            end
            [max_fit_error_step, max_j] = max(fit_error);

            if max_fit_error_step < fit_error_step_thres
                break
            end

            figure(2);set(gcf, 'Position', get(0,'Screensize'));
            to_exclude = current_set(max_j);
            subplot(1, 4, 1); imshow(I{to_exclude}); hold on; plot(pts{to_exclude}(:,1), pts{to_exclude}(:,2), 'g.');
            subplot(1, 4, 2); imshow(warped{to_exclude});
            subplot(1, 4, 3); imshow(syn{max_j});
            diff2_i = warped{to_exclude} - syn{max_j};
            subplot(1, 4, 4); imagesc(diff2_i);axis equal;title(sprintf('norm = %.6f\ndiff = %.6f', max_fit_error_step, norm(diff2_i(:))));

            exclude_count = exclude_count + 1;
            max_fit_error(exclude_count) = max_fit_error_step
            excluded_set(exclude_count) = to_exclude
            if use_single_model
                current_set = setdiff(current_set, [to_exclude])
            end
            toc(t_step);
            pause(1);
        end

        visualize_set(I, pts, excluded_set, struct('saveit', true, 'filename', fullfile(method, [person, '_excluded'])));
        visualize_set(I, pts, current_set, struct('saveit', true, 'filename', fullfile(method, [person, '_filtered'])));
    case 'tournament'
        fprintf('using tournament method\n');

        all_texture_vec = cell2mat(scaled_warped_vec);

        exclude_count = 0;
        excluded_set = [];
        current_set = 1:length(I);

        use_single_model = false;
        measure_error_in_image_space = true;

        max_fit_error_step = 100.0;
        fit_error_step_thres = 0.15;
        model_explained_factor = 80;
        while max_fit_error_step > fit_error_step_thres
            t_step=tic;
            fit_error=[];

            if use_single_model
                num_models = 1;
                model.num_texture_models = num_models;
                model.texture = cell(num_models, 1);

                [coeff, score, latent, tsquared, explained] = pca((all_texture_vec(:, current_set) - repmat(mean_texture_vec, 1, length(current_set)))');
                total_explained = cumsum(explained);
                num_modes = find(total_explained>model_explained_factor, 1, 'first');
                model.texture{1}.num_modes = num_modes;
                model.texture{1}.x = mean_texture_vec;
                model.texture{1}.P = coeff(:, 1:num_modes);
            end

            for j=1:length(current_set)
                i = current_set(j);
                % build only 1 models
                model_j = model;
                if ~use_single_model
                    tic;
                    num_models = 1;
                    model_j.num_texture_models = num_models;
                    model_j.texture = cell(num_models, 1);

                    samples_k = setdiff(current_set, i);
                    [coeff, score, latent, tsquared, explained] = pca((all_texture_vec(:, samples_k) - repmat(mean_texture_vec, 1, length(samples_k)))');
                    total_explained = cumsum(explained);
                    num_modes = find(total_explained>98, 1, 'first');
                    model_j.texture{1}.num_modes = num_modes;
                    model_j.texture{1}.x = mean_texture_vec;
                    model_j.texture{1}.P = coeff(:, 1:num_modes);
                    toc;
                end

                [Ii_r, Ii_g, Ii_b] = split_channels(warped{i});
                Ir = Ii_r(mean_shape_region);
                Ig = Ii_g(mean_shape_region);
                Ib = Ii_b(mean_shape_region);

                N = length(Ir(:));

                diff_i = zeros(size(I{i}));
                syn_i = zeros(size(I{i}));

                norm_i = 0;
                max_norm_i = 0;

                % normalize the input vector first
                normalized_texture_i = [Ir; Ig; Ib];
                alpha_i = dot(normalized_texture_i, model_j.texture{1}.x);
                beta_i = [mean(Ir) * ones(N,1); mean(Ig) * ones(N,1); mean(Ib) * ones(N,1)];
                normalized_texture_i = (normalized_texture_i - beta_i) / alpha_i;

                tvec = model_j.texture{1}.P' * (normalized_texture_i - model_j.texture{1}.x);
                [s, g] = synthesize(model_j, zeros(model_j.shape.num_modes, 1), tvec, 1);

                max_norm_i = max(max_norm_i, norm(g - normalized_texture_i));
                norm_i = norm_i + norm(g - normalized_texture_i);

                % unnormalize the fitted vector
                g = g * alpha_i + beta_i;
                [s_r, s_g, s_b] = split_channels(mean_texture);
                %N = length(g) / 3;
                s_r(mean_shape_region) = g(1:N);
                s_g(mean_shape_region) = g(N+1:N*2);
                s_b(mean_shape_region) = g(N*2+1:end);

                syn_texture = cat(3, s_r, s_g, s_b);

                syn_i = syn_i + syn_texture;
                diff_i = diff_i + abs(syn_texture - im2double(warped{i}));

                % warp syn_i back to image space
                syn_image_i = zeros(size(I{i}));
                M = length(mean_shape_tri);
                tri_verts = cell(1,M);
                for k=1:length(mean_shape_tri)
                    tri_ij = tri_indices{k};
                    tri_verts{k} = pts{i}(tri_ij, :);
                    r_j = imref2d(size(I{i}));

                    mask_k = poly2mask(tri_verts{k}(:,1), tri_verts{k}(:,2), h, w);
                    % warp tri_verts{j} to mean_shape_verts
                    tform_j = estimateGeometricTransform(mean_shape_verts(tri_indices{k}, :), tri_verts{k}, 'affine');
                    warped_syn_i = imwarp(syn_i, tform_j, 'OutputView', r_j);
                    syn_image_i = syn_image_i + warped_syn_i .* mask_k;
                end

                diff2_i = warped{i} - syn_i;
                diff3_i = im2double(I{i}) .* mask{i} - syn_image_i;
                syn{j} = syn_i;

                npixels_i = length(find(mask{i}(:,:,1)>0));
                rmse_i = norm(diff3_i(:)) / sqrt(npixels_i);
                if true%visualize_results
                    figure(2);
                    subplot(1, 6, 1); imshow(I{i}); hold on; plot(pts{i}(:,1), pts{i}(:,2), 'g.');
                    subplot(1, 6, 2); imshow(syn_image_i);

                    subplot(1, 6, 3); imshow(diff3_i);title(sprintf('diff = %.6f\n RMSE = %.6f\n', norm(diff3_i(:)), rmse_i));
                    subplot(1, 6, 4); imshow(warped{i});
                    subplot(1, 6, 5); imshow(syn_i);
                    subplot(1, 6, 6); imagesc(diff_i);axis equal;title(sprintf('norm = %.6f\nmax norm = %.6f\ndiff = %.6f', norm_i, max_norm_i, norm(diff2_i(:))));
                    pause;
                end

                fit_error(j) = rmse_i;
            end
            [max_fit_error_step, max_j] = max(fit_error);

            if max_fit_error_step < fit_error_step_thres
                break
            end

            figure(2);set(gcf, 'Position', get(0,'Screensize'));
            to_exclude = current_set(max_j);
            subplot(1, 4, 1); imshow(I{to_exclude}); hold on; plot(pts{to_exclude}(:,1), pts{to_exclude}(:,2), 'g.');
            subplot(1, 4, 2); imshow(warped{to_exclude});
            subplot(1, 4, 3); imshow(syn{max_j});
            diff2_i = warped{to_exclude} - syn{max_j};
            subplot(1, 4, 4); imagesc(diff2_i);axis equal;title(sprintf('norm = %.6f\ndiff = %.6f', max_fit_error_step, norm(diff2_i(:))));

            exclude_count = exclude_count + 1;
            max_fit_error(exclude_count) = max_fit_error_step
            excluded_set(exclude_count) = to_exclude
            current_set = setdiff(current_set, [to_exclude])
            toc(t_step);
            pause(1);
        end

        visualize_set(I, pts, excluded_set, struct('saveit', true, 'filename', fullfile(method, [person, '_excluded'])));
        visualize_set(I, pts, current_set, struct('saveit', true, 'filename', fullfile(method, [person, '_filtered'])));
    case 'leaveoneout'
        fprintf('using leaveoneout method\n');
        for i=1:length(I)
            % build k models
            num_models = 1;
            model.num_texture_models = num_models;
            model.texture = cell(num_models, 1);
            for k=1:num_models
                samples_k = setdiff([1:size(all_texture_vec,2)], i);
                [coeff, score, latent, tsquared, explained] = pca(all_texture_vec(:, samples_k)');
                total_explained = cumsum(explained);
                num_modes = find(total_explained>75, 1, 'first');
                model.texture{k}.num_modes = num_modes;
                model.texture{k}.x = mean_texture_vec;
                model.texture{k}.P = coeff(:, 1:num_modes);
            end

            [Ii_r, Ii_g, Ii_b] = split_channels(warped{i});
            Ir = Ii_r(mean_shape_region);
            Ig = Ii_g(mean_shape_region);
            Ib = Ii_b(mean_shape_region);

            diff_i = zeros(size(I{i}));
            syn_i = zeros(size(I{i}));
            norm_i = 0;
            max_norm_i = 0;
            for k=1:model.num_texture_models
                % normalize the input vector first
                normalized_texture_i = [Ir; Ig; Ib];
                alpha_i = dot(normalized_texture_i, model.texture{k}.x);
                beta_i = [mean(Ir) * ones(N,1); mean(Ig) * ones(N,1); mean(Ib) * ones(N,1)];
                normalized_texture_i = (normalized_texture_i - beta_i) / alpha_i;
                size(normalized_texture_i)
                size(model.texture{k}.x)
                size(model.texture{k}.P)
                tvec = model.texture{k}.P' * (normalized_texture_i - model.texture{k}.x);
                [s, g] = synthesize(model, zeros(model.shape.num_modes, 1), tvec, k);

                max_norm_i = max(max_norm_i, norm(g - normalized_texture_i));
                norm_i = norm_i + norm(g - normalized_texture_i);

                % unnormalize the fitted vector
                g = g * alpha_i + beta_i;
                [s_r, s_g, s_b] = split_channels(mean_texture);
                N = length(g) / 3;
                s_r(mean_shape_region) = g(1:N);
                s_g(mean_shape_region) = g(N+1:N*2);
                s_b(mean_shape_region) = g(N*2+1:end);
                norm(s_r-s_g)
                norm(s_g-s_b)
                syn_texture = cat(3, s_r, s_g, s_b);

                syn_i = syn_i + syn_texture;
                diff_i = diff_i + abs(syn_texture - im2double(warped{i}));
            end

            syn_i = syn_i / model.num_texture_models;
            diff_i = diff_i / model.num_texture_models;
            norm_i = norm_i / model.num_texture_models;
            diff2_i = warped{i} - syn_i;

            figure(2);
            subplot(1, 4, 1); imshow(I{i}); hold on; plot(pts{i}(:,1), pts{i}(:,2), 'g.');
            subplot(1, 4, 2); imshow(warped{i});
            subplot(1, 4, 3); imshow(syn_i);
            subplot(1, 4, 4); imagesc(diff_i);axis equal;title(sprintf('norm = %.6f\nmax norm = %.6f\ndiff = %.6f', norm_i, max_norm_i, norm(diff2_i(:))));
            pause;
        end
    case 'kmodels'
        fprintf('using kmodels method\n');
        % build k models
        num_models = 64;
        model.num_texture_models = num_models;
        model.texture = cell(num_models, 1);
        for k=1:num_models
            samples_k = randperm(size(all_texture_vec, 2), ceil(length(I) / 10));
            [coeff, score, latent, tsquared, explained] = pca(all_texture_vec(:, samples_k)');
            total_explained = cumsum(explained);
            num_modes = find(total_explained>98, 1, 'first');
            model.texture{k}.num_modes = num_modes;
            model.texture{k}.x = mean_texture_vec;
            model.texture{k}.P = coeff(:, 1:num_modes);
        end

        % build the joint model
        % for i=1:83
        %     tvec = zeros(83,1);
        %     tvec(i) = 50;
        %     [s, g] = synthesize(model, zeros(16, 1), tvec);
        %     syn_texture = mean_texture;
        %     syn_texture(mean_shape_region) = g;
        %     figure(1);imshow(syn_texture); axis equal; colorbar;
        %     pause;
        % end

        for i=1:length(I)
            [Ii_r, Ii_g, Ii_b] = split_channels(warped{i});
            Ir = Ii_r(mean_shape_region);
            Ig = Ii_g(mean_shape_region);
            Ib = Ii_b(mean_shape_region);

            diff_i = zeros(size(I{i}));
            syn_i = zeros(size(I{i}));
            norm_i = 0;
            max_norm_i = 0;
            for k=1:model.num_texture_models
                % normalize the input vector first
                normalized_texture_i = [Ir; Ig; Ib];
                alpha_i = dot(normalized_texture_i, model.texture{k}.x);
                beta_i = [mean(Ir) * ones(N,1); mean(Ig) * ones(N,1); mean(Ib) * ones(N,1)];
                normalized_texture_i = (normalized_texture_i - beta_i) / alpha_i;
                size(normalized_texture_i)
                size(model.texture{k}.x)
                size(model.texture{k}.P)
                tvec = model.texture{k}.P' * (normalized_texture_i - model.texture{k}.x);
                [s, g] = synthesize(model, zeros(16, 1), tvec, k);

                max_norm_i = max(max_norm_i, norm(g - normalized_texture_i));
                norm_i = norm_i + norm(g - normalized_texture_i);

                % unnormalize the fitted vector
                g = g * alpha_i + beta_i;
                [s_r, s_g, s_b] = split_channels(mean_texture);
                N = length(g) / 3;
                s_r(mean_shape_region) = g(1:N);
                s_g(mean_shape_region) = g(N+1:N*2);
                s_b(mean_shape_region) = g(N*2+1:end);
                norm(s_r-s_g)
                norm(s_g-s_b)
                syn_texture = cat(3, s_r, s_g, s_b);

                syn_i = syn_i + syn_texture;
                diff_i = diff_i + abs(syn_texture - im2double(warped{i}));
            end

            syn_i = syn_i / model.num_texture_models;
            diff_i = diff_i / model.num_texture_models;
            norm_i = norm_i / model.num_texture_models;
            diff2_i = warped{i} - syn_i;

            figure(2);
            subplot(1, 4, 1); imshow(I{i}); hold on; plot(pts{i}(:,1), pts{i}(:,2), 'g.');
            subplot(1, 4, 2); imshow(warped{i});
            subplot(1, 4, 3); imshow(syn_i);
            subplot(1, 4, 4); imagesc(diff_i);axis equal;title(sprintf('norm = %.6f\nmax norm = %.6f\ndiff = %.6f', norm_i, max_norm_i, norm(diff2_i(:))));
            pause;
        end
    otherwise
        fprintf('unsupported method\n');
end
