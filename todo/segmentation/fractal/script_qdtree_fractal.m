image_input = im2double(image_input01);
image_input = imnoise(image_input, 'speckle');

power2size_image_input = 2^(floor(log2(length(image_input))));
image_input = imresize(image_input, [power2size_image_input power2size_image_input]);

S = qtdecomp(image_input, .2, 8);

blocks = im2double(repmat(0,size(S)));

dims = [512 256 128 64 32 16 8 4 2 1];

for dim = dims    
  numblocks = length(find(S==dim));    
  if (numblocks > 0)        
    [values, r, c] = qtgetblk(image_input, S, dim);
    new_values = im2double(zeros(size(values)));
    dim
    for k = 1:size(values,3)
        fractal_dimensions = fractal_signature(values(:,:,k), 2);
        fractal_dimensions = fractal_dimensions - mean(fractal_dimensions(:));        
        new_values(:,:,k) = fractal_dimensions(2);
    end
    blocks = qtsetblk(blocks,S,dim,new_values);
  end
end

% blocks(end,1:end) = 1;
% blocks(1:end,end) = 1;

figure;
imshow(image_input);
figure;
imshow(blocks,[]);
