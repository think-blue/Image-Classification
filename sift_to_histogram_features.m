function [histogram_values] = sift_to_histogram_features(image_descriptors, centers)
    num_clusters = size(centers, 2);
    distances = pdist2(image_descriptors', centers');
    [~, label] = min(distances,[], 2);
    % uncomment normalization probability to normalize histograms (decreases accuracy)
    h = histogram(label, (num_clusters) );%, 'Normalization', 'probability');
    histogram_values = h.Values;
end