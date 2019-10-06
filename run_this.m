
%% Initialization
clear variables; close all;

%% Creating structures for directories of training images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Change Directory before running. MATLAB only accepts   %
% absolute file names                                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
src_files_cars = dir('C:\Users\chinm\Documents\MATLAB\ComputerVision\Assignment2-image_classification\cars_train\*.jpg');
src_files_planes = dir('C:\Users\chinm\Documents\MATLAB\ComputerVision\Assignment2-image_classification\airplanes_train\*.jpg');
src_files_bikes = dir('C:\Users\chinm\Documents\MATLAB\ComputerVision\Assignment2-image_classification\motorbikes_train\*.jpg');
src_files_faces = dir('C:\Users\chinm\Documents\MATLAB\ComputerVision\Assignment2-image_classification\faces_train\*.jpg');

%% Reading individual training images and extracting features (only done once). No need to run again as the descriptors are saved as matrix in the working folder for efficiency

% descriptor_matrix = [];
% descriptor_matrix_info = [];
% 
% for i = 1 : length(src_files_cars)
%     image_label = 0;
%     file_name = strcat('C:\Users\chinm\Documents\MATLAB\ComputerVision\Assignment2-image_classification\cars_train\', src_files_cars(i).name);
%     images_cars{i} = imread(file_name);
%     [~, d] = extract_sift_features(images_cars{i});
%     descriptor_matrix = [descriptor_matrix, d];
%     descriptor_matrix_info = [descriptor_matrix_info, [size(d, 2); image_label] ];
% end
% 
% for i = 1 : length(src_files_planes)
%     image_label = 1;
%     file_name = strcat('C:\Users\chinm\Documents\MATLAB\ComputerVision\Assignment2-image_classification\airplanes_train\', src_files_planes(i).name);
%     images_planes{i} = imread(file_name);
%     [~, d] = extract_sift_features(images_planes{i});
%     descriptor_matrix = [descriptor_matrix, d];
%     descriptor_matrix_info = [descriptor_matrix_info, [size(d, 2); image_label] ];
% 
% end
% 
% for i = 1 : length(src_files_bikes)
%     image_label = 2;
%     file_name = strcat('C:\Users\chinm\Documents\MATLAB\ComputerVision\Assignment2-image_classification\motorbikes_train\', src_files_bikes(i).name);
%     images_bikes{i} = imread(file_name);
%     [~, d] = extract_sift_features(images_bikes{i});
%     descriptor_matrix = [descriptor_matrix, d];
%     descriptor_matrix_info = [descriptor_matrix_info, [size(d, 2); image_label] ];
% end
% 
% for i = 1 : length(src_files_faces)
%     image_label = 3;
%     file_name = strcat('C:\Users\chinm\Documents\MATLAB\ComputerVision\Assignment2-image_classification\faces_train\', src_files_faces(i).name);
%     images_faces{i} = imread(file_name);
%     [f, d] = extract_sift_features(images_faces{i});
%     descriptor_matrix = [descriptor_matrix, d];
%     descriptor_matrix_info = [descriptor_matrix_info, [size(d, 2); image_label] ];
% end
% 
% descriptor_matrix = double(descriptor_matrix);
% save('sift_features', 'descriptor_matrix', 'descriptor_matrix_info');

%% loading descriptor matrix
clear variables; close all;
load sift_features.mat;

%% K-means clustering

% Number of times to repeat K-means clustering to choose the one with
% minimum energy
repetitions = 4; % reduce repetitons to 1  to improve computation time or increase to 7 to improve accuracy

% Number of clusters
num_clusters = 200;

% initialized with maximum distance between centers (K-means++)
[centres, assignments, energy] = vl_kmeans( double(descriptor_matrix),....
                                            num_clusters, 'verbose', ...
                                            'Initialization', 'plusplus', ...
                                            'NumRepetitions', repetitions );

%% calculating histograms from training data

% initializing matrix to hold the histograms for training images
histogram_matrix = [];

% creating histogram by extracting information from descriptor matrix
start_index = 1;
for i = 1:length(descriptor_matrix_info)
    d = descriptor_matrix( :, start_index:(start_index + descriptor_matrix_info(1, i) - 1) );
    start_index = start_index + descriptor_matrix_info(1, i);
    hist_values = sift_to_histogram_features(d, centres);
    histogram_matrix = [ histogram_matrix ; [hist_values, descriptor_matrix_info(2,i)] ];
end


%% Reading test images and extracting descriptor matrix for them

% Creating structures for test directories
src_files_cars_test = dir('C:\Users\chinm\Documents\MATLAB\ComputerVision\Assignment2-image_classification\cars_test\*.jpg');
src_files_planes_test = dir('C:\Users\chinm\Documents\MATLAB\ComputerVision\Assignment2-image_classification\airplanes_test\*.jpg');
src_files_bikes_test = dir('C:\Users\chinm\Documents\MATLAB\ComputerVision\Assignment2-image_classification\motorbikes_test\*.jpg');
src_files_faces_test = dir('C:\Users\chinm\Documents\MATLAB\ComputerVision\Assignment2-image_classification\faces_test\*.jpg');

% initializing descriptor matrix for test images
descriptor_matrix_test = [];
descriptor_matrix_info_test = [];

for i = 1 : length(src_files_cars_test)
    image_label = 0;
    file_name = strcat('C:\Users\chinm\Documents\MATLAB\ComputerVision\Assignment2-image_classification\cars_test\', src_files_cars_test(i).name);
    images_cars{i} = imread(file_name);
    [~, d] = extract_sift_features(images_cars{i});
    descriptor_matrix_test = [descriptor_matrix_test, d];
    descriptor_matrix_info_test = [descriptor_matrix_info_test, [size(d, 2); image_label] ];
end

for i = 1 : length(src_files_planes_test)
    image_label = 1;
    file_name = strcat('C:\Users\chinm\Documents\MATLAB\ComputerVision\Assignment2-image_classification\airplanes_test\', src_files_planes_test(i).name);
    images_planes_test{i} = imread(file_name);
    [~, d] = extract_sift_features(images_planes_test{i});
    descriptor_matrix_test = [descriptor_matrix_test, d];
    descriptor_matrix_info_test = [descriptor_matrix_info_test, [size(d, 2); image_label] ];

end

for i = 1 : length(src_files_bikes_test)
    image_label = 2;
    file_name = strcat('C:\Users\chinm\Documents\MATLAB\ComputerVision\Assignment2-image_classification\motorbikes_test\', src_files_bikes_test(i).name);
    images_bikes{i} = imread(file_name);
    [~, d] = extract_sift_features(images_bikes{i});
    descriptor_matrix_test = [descriptor_matrix_test, d];
    descriptor_matrix_info_test = [descriptor_matrix_info_test, [size(d, 2); image_label] ];
end

for i = 1 : length(src_files_faces_test)
    image_label = 3;
    file_name = strcat('C:\Users\chinm\Documents\MATLAB\ComputerVision\Assignment2-image_classification\faces_test\', src_files_faces_test(i).name);
    images_faces{i} = imread(file_name);
    [f, d] = extract_sift_features(images_faces{i});
    descriptor_matrix_test = [descriptor_matrix_test, d];
    descriptor_matrix_info_test = [descriptor_matrix_info_test, [size(d, 2); image_label] ];
end

descriptor_matrix_test = double(descriptor_matrix_test);

%% calculating histograms for test images(from test descriptor matrix)

histogram_matrix_test = [];
start_index = 1;
for i = 1:length(descriptor_matrix_info_test)
    d = descriptor_matrix_test( :, start_index:(start_index + descriptor_matrix_info_test(1, i) - 1) );
    start_index = start_index + descriptor_matrix_info_test(1, i);
    hist_values = sift_to_histogram_features(d, centres);
    histogram_matrix_test = [ histogram_matrix_test ; [hist_values, descriptor_matrix_info_test(2,i)] ];
end

% saving test and training variables so, the process can be fastened, if
% required
save('histograms', 'histogram_matrix', 'histogram_matrix_test');

%% K Nearest Neighbours

%calculating chi sq. distances between the histograms
chi_distance = chisq_pdist2(histogram_matrix(:,1:num_clusters), histogram_matrix_test(:,1:num_clusters), 'chisq');

% sorthing the distances to get best(least distances) k results
[sorted_chi_distance, index_matrix] = sort(chi_distance);

% setting k for knn
num_neighbours = 5;

% extracting labels from training data
training_labels = histogram_matrix(:, num_clusters+1);

% extracting labels from test data
actual_labels = histogram_matrix_test(:,num_clusters+1);

% matching least distances to corrosponding labels of training images
matched_labels = training_labels(index_matrix(1:num_neighbours,:));

% predictions of labels by majority voting
predictions = mode(matched_labels);

% calculating accuracy
difference = predictions' - actual_labels;
accuracy_knn = length(find(difference == 0))/length(difference);

%% weighted KNN

num_labels = 4;
num_test_images = 200;

% adding weights(1/d) for corosponding labels for each test image
for i = 1:num_test_images
     for j = 1:num_labels
         distance_column = sorted_chi_distance(1:num_neighbours, i);
         matched_label = matched_labels(1:num_neighbours, i);
         sum_weights(j, i) = sum( 1./distance_column(matched_label == j-1) );
     end
end

% calculating accuracy
[~, predictions_weighed] = max(sum_weights, [], 1);
difference = (predictions_weighed - 1)' - actual_labels;
accuracy_knn_weighed = length(find(difference == 0))/length(difference);

%% SVM

% seprating training data and testing data from train and test labels
train_data = histogram_matrix(:, 1:num_clusters);
train_label = histogram_matrix(:, num_clusters + 1);

test_data = histogram_matrix_test(:, 1:num_clusters);
test_label = histogram_matrix_test(:, num_clusters + 1);

% Linear Kernel implementation
model_linear = svmtrain(train_label, train_data, '-t 0');
[predict_label_L, accuracy_svm_L, dec_values_L] = svmpredict(test_label, test_data, model_linear);


% precomputed kernel implementation (linear)
% model_precomputed = svmtrain(train_label, [(1:400)', train_data*train_data'], '-t 4');
% [predict_label_P, accuracy_P, dec_values_P] = svmpredict(test_label, [(1:200)', test_data*train_data'], model_precomputed);
