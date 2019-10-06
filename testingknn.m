clear variables;
load sift_features.mat
load histograms.mat

%% distance calculation
chi_distance = chisq_pdist2(histogram_matrix(:,1:num_clusters), histogram_matrix_test(:,1:num_clusters), 'chisq');
[sorted_chi_distance, index_matrix] = sort(chi_distance);
num_neighbours = 200;
trained_labels = histogram_matrix(:, num_clusters+1);
actual_labels = histogram_matrix_test(:,num_clusters+1);
matched_labels = trained_labels(index_matrix(1:num_neighbours,:));
predictions = mode(matched_labels);


%% Accuracy
difference = predictions' - actual_labels;
accuracy_knn = length(find(difference == 0))/length(difference);
a1(count) = accuracy_knn;
%% weighed
%% nearest neighbour
num_labels = 4;
num_test_images = 200;
 for i = 1:num_test_images
     for j = 1:num_labels
         distance_column = sorted_chi_distance(1:num_neighbours, i);
         matched_label = matched_labels(1:num_neighbours, i);
         sum_weights(j, i) = sum( 1./distance_column(matched_label == j-1) );
     end
 end
 
[~, predictions_weighed] = max(sum_weights, [], 1);
difference = (predictions_weighed - 1)' - actual_labels;
accuracy_knn_weighed = length(find(difference == 0))/length(difference);
a2(count) = accuracy_knn_weighed;
%% SVM features
% train_data = histogram_matrix(:, 1:num_clusters);
% train_label = histogram_matrix(:, num_clusters + 1);
% 
% test_data = histogram_matrix_test(:, 1:num_clusters);
% test_label = histogram_matrix_test(:, num_clusters + 1);
% 
% %% Linear Kernel
% model_linear = svmtrain(train_label, train_data, '-t 0');
% [predict_label_L, accuracy_L, dec_values_L] = svmpredict(test_label, test_data, model_linear);
% a3(count) = accuracy_L(1);
 count = count +1;
%% precomputed kernel
% model_precomputed = svmtrain(train_label, [(1:400)', train_data*train_data'], '-t 4');
% [predict_label_P, accuracy_P, dec_values_P] = svmpredict(test_label, [(1:200)', test_data*train_data'], model_precomputed);

