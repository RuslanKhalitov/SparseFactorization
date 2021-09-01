function A = load_square_matrix(category, filename)
switch category
    case {'square_image', 'gradient_image'}
        img_dir = sprintf('square_matrices/%s/', category);
        img = imread([img_dir, filename]);
        if ndims(img)==3
            img = rgb2gray(img);
        end
        A = double(img);
    case {'dense_graph', 'network', 'surface_mesh', 'covariance_matrix'}
        mat_dir = sprintf('square_matrices/%s/', category);
        load([mat_dir, filename], 'A');
        A = full(A);
    otherwise
        error('unknown category');
end