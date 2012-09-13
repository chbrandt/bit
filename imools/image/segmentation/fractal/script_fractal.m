% image_input = imnoise(image_input01,'speckle');
image_input = im2double(a);

colordef black;
figure_mainimage = figure();
imshow(image_input);hold on;
figure_fractal_dimensions = figure();

colormap_count = 1;
colormap_slice_size = 20;
colormap_slice = lines(colormap_slice_size);
while(true)
    figure(figure_mainimage);hold on;
%     [image_slice, slice_rect] = imcrop(image_input);hold on;
    [x,y] = ginput(2);
    if (x(2)<x(1))
        x = x([2 1]);
    end
    if (y(2)<y(1))
        y = y([2 1]);
    end    
    slice_rect(1:2) = [x(1) y(1)];
    slice_rect(3:4) = [x(2)-x(1) y(2)-y(1)];
    plot([x(1) x(2)], [y(1) y(1)], 'Color', colormap_slice(mod(colormap_count,colormap_slice_size)+1,:), 'LineWidth', 2)
    plot([x(2) x(2)], [y(1) y(2)], 'Color', colormap_slice(mod(colormap_count,colormap_slice_size)+1,:), 'LineWidth', 2)
    plot([x(2) x(1)], [y(2) y(2)], 'Color', colormap_slice(mod(colormap_count,colormap_slice_size)+1,:), 'LineWidth', 2)
    plot([x(1) x(1)], [y(2) y(1)], 'Color', colormap_slice(mod(colormap_count,colormap_slice_size)+1,:), 'LineWidth', 2)
    image_slice = imcrop(image_input, slice_rect);
    fractal_dimensions = fractal_signature(image_slice, 5)
	figure(figure_fractal_dimensions);
%     fractal_dimensions = fractal_dimensions - mean(fractal_dimensions);
    plot(2:length(fractal_dimensions),fractal_dimensions(2:end,:), 'Color', colormap_slice(mod(colormap_count,colormap_slice_size)+1, :));hold on;
    xlabel('scale');
    ylabel('fractal dimension');
    colormap_count = colormap_count + 1;
end

colordef white;