function warped = pa_warp_m(AAM, s, I)
    t_i = tic;
    I_i = im2double(I);
    pts = fliplr(reshape(s, [], 2));
    mean_shape_verts = fliplr(reshape(AAM.s0, [], 2));
    mean_mask = AAM.mean_mask;
    
    [h, w, d] = size(I_i);
    mask = zeros(size(I_i));
    masked_i = zeros(size(I_i));
    
    N = size(AAM.shape_mesh, 1);
    tri_verts = cell(1,N);
    masks  = cell(1, N);
    masked = cell(1, N);    
    for j=1:N
        tri_ij = AAM.shape_mesh(j,:);
        tri_verts{j} = pts(tri_ij, :);
        masks{j} = poly2mask(tri_verts{j}(:,1), tri_verts{j}(:,2), h, w);
        masks{j} = repmat(masks{j}, [1, 1, d]);
        masked{j} = I_i .* masks{j};
        mask = mask + masks{j};
        masked_i = masked_i + masked{j};
    end
    
    % warped masks
    warped = zeros(size(mean_mask{1}));
    warped_masked = cell(1,N);
    for j=1:N
        r_j = imref2d(size(mean_mask{1}));
        
        if false
            % warp mean_shape_verts to tri_verts{j}        
        else
            % simplify this: just warp the ROI, the copy the warped ROI to
            % a template image and mask it
            
            % warp tri_verts{j} to mean_shape_verts
            tform_j = estimateGeometricTransform(pts(AAM.shape_mesh(j,:), :), mean_shape_verts(AAM.shape_mesh(j,:), :), 'affine');
            warped_masked{j} = imwarp(I_i, tform_j, 'OutputView', r_j);      
            warped = warped + warped_masked{j} .* mean_mask{j};
            %figure(3);imshow(warped);pause;
        end
    end
    
    if 0
        figure(2); N = 3;
        subplot(1, N, 1); imshow(I); axis on;
        subplot(1, N, 2); imshow(masked_i); axis on;hold on;
        triplot(double(AAM.shape_mesh), s(73+1:2*73)', s(1:73)');
        triplot(double(AAM.shape_mesh(1,:)), s(73+1:2*73)', s(1:73)', '-r');

        subplot(1, N, 3); imshow(warped); axis on; hold on;
        triplot(double(AAM.shape_mesh), AAM.s0(73+1:2*73)', AAM.s0(1:73)');
        plot(mean_shape_verts(:,1), mean_shape_verts(:,2), 'g.');
        triplot(double(AAM.shape_mesh(1,:)), AAM.s0(73+1:2*73)', AAM.s0(1:73)', '-r');
        pause;
    end
    toc(t_i);
end