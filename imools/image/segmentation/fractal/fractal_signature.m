function [D_local] = fractal_signature(image_input, scales_iterations, visu)
    if (~exist('scales_iterations', 'var'))
        scales_iterations = 20;
    end
    if (~exist('visu', 'var'))
        visu = false;
    end
    if (size(image_input,3)==3)
        image_input = rgb2gray(image_input);
    end
    image_input = im2double(image_input);
    
    image_upper = image_input;
    image_lower = image_input;
    area_total = zeros(scales_iterations,1);
    area_pixel = zeros(size(image_input, 1), size(image_input, 2), scales_iterations);
    level_step = 1;
    K = 1;
    
    if (visu)
        figure;
        subplot(2, 2, [1 2]);
        imshow(image_input);
    end
    for k = 1:scales_iterations
        image_upper = calculate_next_upper_level(image_upper, 3, level_step);
        image_lower = calculate_next_lower_level(image_lower, 3, level_step);        
        if (visu)
            subplot(2, 2, 3);
            imshow(image_upper,[])
            title(num2str(k));
            subplot(2, 2, 4);
            imshow(image_lower,[])
            title(num2str(k));
            drawnow;
        end
        area_total(k) = calculate_area_total(image_upper, image_lower, k, level_step);
        area_pixel(:,:,k) = calculate_area_pixel(image_upper, image_lower, k, level_step);
    end
    
    npartitions = 1;
    D_global = zeros(floor(scales_iterations/npartitions), 1);
    D_local = zeros(size(area_pixel));
    M = zeros(floor(scales_iterations/npartitions), 1);
    if (visu)
        figure;
    end
    for m = 2:(floor(scales_iterations/npartitions))
        D_numerator_local = (log(area_pixel(:,:,1))-log(area_pixel(:,:,m)))/log(2);
        D_denominator_local = (log(m)-log(1))/log(2);
        D_local(:,:,m) = D_numerator_local./D_denominator_local;
        if (visu)
            imshow(D_local(:,:,m),[]);
            title(num2str(m));
            drawnow;
        end
        D_numerator_global = (log(area_total(1))-log(area_total(m)))/log(2);
        D_denominator_global = (log(m)-log(1))/log(2);
        D_global(m) = D_numerator_global/D_denominator_global;
        M(m) = K*(m*npartitions)^(3-D_global(m));
    end
%     area_total
%     prod(size(image_input))
%     D_global
%     M
    if (visu)
        figure;
        colordef black;
        axes();hold on;
        plot(2:npartitions:(scales_iterations), D_global(2:end), 'co-');
%         figure;
%         axes();hold on;
%         plot(2:npartitions:(scales_iterations), M(2:end), 'r.-');
        colordef white;
    end
end

function area_value = calculate_area_total(image_upper, image_lower, scale_factor, level_step)
    if (~exist('level_step','var'))
        level_step = 1;
    end
    image_diff = (image_upper-image_lower);
    area_value = sum(sum(image_diff))/(2*scale_factor*level_step);
end

function area_pixel_values = calculate_area_pixel(image_upper, image_lower, scale_factor, level_step)
    if (~exist('level_step','var'))
        level_step = 1;
    end
    window_size = 20;
    image_diff = (image_upper-image_lower);
    area_pixel_values = zeros(size(image_diff));
%     for m = 1:size(image_upper,1)
%         for n = 1:size(image_lower,2)
%             window_neighborhood_i = (m-window_size):(m+window_size);
%             window_neighborhood_j = (n-window_size):(n+window_size);
%             [valid_neiborhood_i_idx,valid_neiborhood_j_idx] = valid_neiborhood(image_diff, window_neighborhood_i, window_neighborhood_j);
%             image_diff_windowed =
%             image_diff(valid_neiborhood_i_idx,valid_neiborhood_j_idx);
%         end
    %     end
    image_diff_windowed = conv2(image_diff,ones(window_size),'same');
    area_pixel_values = image_diff_windowed./(2*scale_factor*level_step);
end


function image_output = calculate_next_upper_level(image_input, neiborhood, level_step)
    if (~exist('level_step','var'))
        level_step = 1;
    end
    if (~exist('neiborhood','var'))
        neiborhood = 3;
    end    
    struct_elem = strel([1 1 1; 1 1 1; 1 1 1],...
                        level_step*[1 0 1; 0 1 0; 1 0 1]);
%     struct_elem = strel([1 1 1; 1 1 1; 1 1 1],...
%                         level_step*[0 1 0; 1 1 1; 0 1 0]);
    image_output = imdilate(image_input, struct_elem);
end

function image_output = calculate_next_lower_level(image_input, neiborhood, level_step)
    if (~exist('level_step','var'))
        level_step = 1;
    end
    if (~exist('neiborhood','var'))
        neiborhood = 3;
    end    
    struct_elem = strel([1 1 1; 1 1 1; 1 1 1],...
                        level_step*[1 0 1; 0 1 0; 1 0 1]);    
%     struct_elem = strel([1 1 1; 1 1 1; 1 1 1],...
%                         level_step*[0 1 0; 1 1 1; 0 1 0]);
    image_output = imerode(image_input, struct_elem);
end

function [valid_neiborhood_i_idx,valid_neiborhood_j_idx] = valid_neiborhood(image_input, neiborhood_i_idx, neiborhood_j_idx)
    neiborhood_lt_bound_i_mask = floor(neiborhood_i_idx)<1;
    neiborhood_lt_bound_j_mask = floor(neiborhood_j_idx)<1;
    
    valid_neiborhood_i_idx = (neiborhood_i_idx.*~neiborhood_lt_bound_i_mask);
    valid_neiborhood_j_idx = (neiborhood_j_idx.*~neiborhood_lt_bound_j_mask);
    
    neiborhood_gt_bound_i_mask = ceil(neiborhood_i_idx)>size(image_input,1);
    neiborhood_gt_bound_j_mask = ceil(neiborhood_j_idx)>size(image_input,2);
    
    valid_neiborhood_i_idx = (valid_neiborhood_i_idx.*~neiborhood_gt_bound_i_mask);
    valid_neiborhood_j_idx = (valid_neiborhood_j_idx.*~neiborhood_gt_bound_j_mask);
    
    neiborhood_concat = cat(2,valid_neiborhood_i_idx',valid_neiborhood_j_idx');
    elim_idx = [];
    for k = 1:size(neiborhood_concat,1)
        if (prod(neiborhood_concat(k,:))==0)
            elim_idx(end+1) = k;
        end
    end
    
    valid_neiborhood_i_idx = valid_neiborhood_i_idx(setxor(elim_idx,1:length(valid_neiborhood_i_idx)));
    valid_neiborhood_j_idx = valid_neiborhood_j_idx(setxor(elim_idx,1:length(valid_neiborhood_j_idx)));
end