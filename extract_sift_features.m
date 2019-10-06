function [frames, descriptors] = extract_sift_features(image)
if size(image, 3) == 3
    I = single(rgb2gray(image));
else
    I = single(image);
end
    [frames,descriptors] = vl_sift(I) ;
end