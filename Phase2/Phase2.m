
%% Loading Datasets

EEG_02 = pop_biosig('chb01_02.edf');
EEG_03 = pop_biosig('chb01_03.edf');
EEG_04 = pop_biosig('chb01_04.edf');
EEG_15 = pop_biosig('chb01_15.edf');
EEG_16 = pop_biosig('chb01_16.edf');
EEG_18 = pop_biosig('chb01_18.edf');
EEG_26 = pop_biosig('chb01_26.edf');


EEG_02 = eeg_checkset(EEG_02);
EEG_03 = eeg_checkset(EEG_03);
EEG_04 = eeg_checkset(EEG_04);
EEG_15 = eeg_checkset(EEG_15);
EEG_16 = eeg_checkset(EEG_16);
EEG_18 = eeg_checkset(EEG_18);
EEG_26 = eeg_checkset(EEG_26);

%% Calculating PSD & Shannon Entropy

data_to_evaluate = EEG_16;

epoch_length_samples = 10 * 60 * 256; 

num_channels = size(data_to_evaluate.data, 1);
num_samples = size(data_to_evaluate.data, 2);
num_epochs = floor(num_samples / 153600);

window_length = 256;
noverlap = 0;
nfft = 512;
fs = 256;

psd_all_epochs = cell(num_channels, num_epochs);
shannon_entropy_all_epochs = zeros(num_channels, num_epochs);

for ep = 1:num_epochs
    epoch_start = (ep-1) * epoch_length_samples + 1;
    epoch_end = ep * epoch_length_samples;
    
    for ch = 1:num_channels
        data = data_to_evaluate.data(ch, epoch_start:epoch_end);
        
        [pxx, f] = pwelch(data, window_length, noverlap, nfft, fs);
        
        pxx_norm = pxx / sum(pxx);
        
        shannon_entropy = -sum(pxx_norm .* log2(pxx_norm + eps));
        
        psd_all_epochs{ch, ep} = struct('pxx', pxx, 'f', f);
        shannon_entropy_all_epochs(ch, ep) = shannon_entropy;
    end
end

%% Plot PSD & Shannon Entropy
channel_to_plot = 8;
epoch_to_plot = 6;

pxx = psd_all_epochs{channel_to_plot, epoch_to_plot}.pxx;
f = psd_all_epochs{channel_to_plot, epoch_to_plot}.f;

figure;
plot(f, 10*log10(pxx),"LineWidth",1.5);
title(['PSD of Channel ' num2str(channel_to_plot) ' Epoch ' num2str(epoch_to_plot)]);
grid minor
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');

figure;
stem(1:num_epochs,shannon_entropy_all_epochs(5, :),'LineWidth',1.5);
grid minor
title(['Shannon Entropy of Channel ' num2str(channel_to_plot) ' Across Epochs']);
xlabel('Epoch');
ylabel('Shannon Entropy (bits)');

%% Feature Extraction

seizure_start_time = 1015; 
interval_length_minutes = 10; 
epoch_length_seconds = 16; 

data_before_seizure = getDataBeforeTime(data_to_evaluate, seizure_start_time, interval_length_minutes);

data_matrix = getMatrix(data_before_seizure, epoch_length_seconds, data_to_evaluate.srate);

features = getFeature(data_matrix, data_to_evaluate.srate);

figure;
stem(1:size(features.shannon_entropy, 2), features.shannon_entropy(8, :),'LineWidth',1.5);
grid minor
title('Shannon Entropy of Channel 8 Across Epochs');
xlabel('Epoch');
ylabel('Shannon Entropy (bits)');

%% Train and Test Data Selection

fs = 256; 
ten_min_samples = 10 * 60 * fs; 
nine_half_min_samples = 9.5 * 60 * fs; 
thirty_sec_samples = 30 * fs; 

category_A = EEG_02.data(:, 1:6 * ten_min_samples);
category_A = reshape(category_A, size(EEG_02.data, 1), ten_min_samples, []);

B1_start = size(EEG_03.data, 2) - ten_min_samples + 1;
category_B1 = EEG_03.data(:, B1_start:end);
category_B2 = EEG_03.data(:, 1:ten_min_samples);
category_B = cat(3, category_B1, category_B2);

seizure_files = {
    'chb01_03.edf', 2996;
    'chb01_04.edf', 1467;
    'chb01_15.edf', 1732;
    'chb01_16.edf', 1015;
    'chb01_18.edf', 1720;
    'chb01_26.edf', 1862
};
category_C = [];

for i = 1:size(seizure_files, 1)
    EEG_C = pop_biosig(seizure_files{i, 1});
    seizure_start_sample = seizure_files{i, 2} * fs;
    C_start = seizure_start_sample - nine_half_min_samples + 1;
    C_end = seizure_start_sample + thirty_sec_samples;
    category_C = cat(3, category_C, EEG_C.data(:, C_start:C_end));
end

split_matricesA = cell(1, 6);
split_matricesC = cell(1, 6);

for i = 1:6
    split_matricesA{i} = category_A(:, :, i);
end

for i = 1:6
    split_matricesC{i} = category_C(:, :, i);
end

dataA1_matrix = getMatrix(split_matricesA{1}, 16, 256);
dataA2_matrix = getMatrix(split_matricesA{2}, 16, 256);
dataA3_matrix = getMatrix(split_matricesA{3}, 16, 256);
dataA4_matrix = getMatrix(split_matricesA{4}, 16, 256);
dataA5_matrix = getMatrix(split_matricesA{5}, 16, 256);
dataA6_matrix = getMatrix(split_matricesA{6}, 16, 256);

featuresA1 = getFeature(dataA1_matrix, 256);
featuresA2 = getFeature(dataA2_matrix, 256);
featuresA3 = getFeature(dataA3_matrix, 256);
featuresA4 = getFeature(dataA4_matrix, 256);
featuresA5 = getFeature(dataA5_matrix, 256);
featuresA6 = getFeature(dataA6_matrix, 256);

dataC1_matrix = getMatrix(split_matricesC{1}, 16, 256);
dataC2_matrix = getMatrix(split_matricesC{2}, 16, 256);
dataC3_matrix = getMatrix(split_matricesC{3}, 16, 256);
dataC4_matrix = getMatrix(split_matricesC{4}, 16, 256);
dataC5_matrix = getMatrix(split_matricesC{5}, 16, 256);
dataC6_matrix = getMatrix(split_matricesC{6}, 16, 256);

featuresC1 = getFeature(dataC1_matrix, 256);
featuresC2 = getFeature(dataC2_matrix, 256);
featuresC3 = getFeature(dataC3_matrix, 256);
featuresC4 = getFeature(dataC4_matrix, 256);
featuresC5 = getFeature(dataC5_matrix, 256);
featuresC6 = getFeature(dataC6_matrix, 256);

dataB1_matrix = getMatrix(category_B1, 16, 256);
featuresB1 = getFeature(dataB1_matrix, 256);

dataB2_matrix = getMatrix(category_B2, 16, 256);
featuresB2 = getFeature(dataB2_matrix, 256);

resultsA1 = getResult(featuresA1);
resultsA2 = getResult(featuresA2);
resultsA3 = getResult(featuresA3);
resultsA4 = getResult(featuresA4);
resultsA5 = getResult(featuresA5);
resultsA6 = getResult(featuresA6);
resultsB1 = getResult(featuresB1);
resultsB2 = getResult(featuresB2);
resultsC1 = getResult(featuresC1);
resultsC2 = getResult(featuresC2);
resultsC3 = getResult(featuresC3);
resultsC4 = getResult(featuresC4);
resultsC5 = getResult(featuresC5);
resultsC6 = getResult(featuresC6);

%% Feature Selection / t-test & and p-value Calculation

p_valueA1 = getPs(resultsA1);
p_valueA2 = getPs(resultsA2);
p_valueA3 = getPs(resultsA3);
p_valueA4 = getPs(resultsA4);
p_valueA5 = getPs(resultsA5);
p_valueA6 = getPs(resultsA6);
p_valueB1 = getPs(resultsB1);
p_valueB2 = getPs(resultsB2);
p_valueC1 = getPs(resultsC1);
p_valueC2 = getPs(resultsC2);
p_valueC3 = getPs(resultsC3);
p_valueC4 = getPs(resultsC4);
p_valueC5 = getPs(resultsC5);
p_valueC6 = getPs(resultsC6);

train = struct();
test = struct();

train.shannon_entropy = [p_valueC1.shannon_entropy, p_valueC2.shannon_entropy, ...
    p_valueC3.shannon_entropy, p_valueC4.shannon_entropy, p_valueA1.shannon_entropy, ...
    p_valueA2.shannon_entropy, p_valueA3.shannon_entropy, p_valueA4.shannon_entropy, ...
    p_valueA5.shannon_entropy, p_valueB1.shannon_entropy]';

train.mean = [p_valueC1.mean, p_valueC2.mean, p_valueC3.mean, p_valueC4.mean, ...
    p_valueA1.mean, p_valueA2.mean, p_valueA3.mean, p_valueA4.mean, p_valueA5.mean, p_valueB1.mean]';

train.std = [p_valueC1.std, p_valueC2.std, p_valueC3.std, p_valueC4.std, p_valueA1.std, ...
    p_valueA2.std, p_valueA3.std, p_valueA4.std, p_valueA5.std, p_valueB1.std]';

train.max = [p_valueC1.max, p_valueC2.max, p_valueC3.max, p_valueC4.max, p_valueA1.max, ...
    p_valueA2.max, p_valueA3.max, p_valueA4.max, p_valueA5.max, p_valueB1.max]';

train.min = [p_valueC1.min, p_valueC2.min, p_valueC3.min, p_valueC4.min, p_valueA1.min, ...
    p_valueA2.min, p_valueA3.min, p_valueA4.min, p_valueA5.min, p_valueB1.min]';

test.shannon_entropy = [p_valueC5.shannon_entropy, p_valueC6.shannon_entropy, ...
    p_valueA6.shannon_entropy, p_valueB2.shannon_entropy]';

test.mean = [p_valueC5.mean, p_valueC6.mean, p_valueA6.mean, p_valueB2.mean]';

test.std = [p_valueC5.std, p_valueC6.std, p_valueA6.std, p_valueB2.std]';

test.max = [p_valueC5.max, p_valueC6.max, p_valueA6.max, p_valueB2.max]';

test.min = [p_valueC5.min, p_valueC6.min, p_valueA6.min, p_valueB2.min]';

train = [train.shannon_entropy, train.mean, train.std, train.max, train.min];
test = [test.shannon_entropy, test.mean, test.std, test.max, test.min];

train = double(train);
test = double(test);

%% classifiers svm

train_labels = [ones(1, 4), zeros(1, 6)];
test_labels = [ones(1, 2), zeros(1, 2)];

train_labels = double(train_labels);
test_labels = double(test_labels);

total_data = [train; test];
total_labels = [train_labels, test_labels]';

k = 5;
indices = crossvalind('Kfold', total_labels, k);

sensitivity = zeros(k, 1);
specificity = zeros(k, 1);

for i = 1:k
    test1 = (indices == i);
    train1 = ~test1;

    svmModel = fitcsvm(total_data(train1,:), total_labels(train1), 'KernelFunction', 'linear', ...
        'BoxConstraint', 1);
    
    predictions = predict(svmModel, total_data(test1,:));

    sensitivity(i) = calculate_tpr(predictions, total_labels(test1));
    specificity(i) = calculate_tnr(predictions, total_labels(test1));
end

average_sensitivity = mean(sensitivity);
average_specificity = mean(specificity);
fprintf('Average sensitivity across %d folds is: %.2f%%\n', k, average_sensitivity * 100);
fprintf('Average specificity across %d folds is: %.2f%%\n', k, average_specificity * 100);

%% knn classifier

numNeighbors = 5; 

k = 5;
indices = crossvalind('Kfold', total_labels, k);

sensitivity = zeros(k, 1);
specificity = zeros(k, 1);

for i = 1:k
    test2 = (indices == i);
    train2 = ~test2;
    
    Mdl = fitcknn(total_data(train2,:), total_labels(train2), ...
                  'NumNeighbors', numNeighbors, ...
                  'Distance', 'minkowski');
    
    predictions = predict(Mdl, total_data(test2,:));
    
    sensitivity(i) = calculate_tpr(predictions, total_labels(test2));
    specificity(i) = calculate_tnr(predictions, total_labels(test2));
end

average_sensitivity = mean(sensitivity);
average_specificity = mean(specificity);
fprintf('Average sensitivity across %d folds is: %.2f%%\n', k, average_sensitivity * 100);
fprintf('Average specificity across %d folds is: %.2f%%\n', k, average_specificity * 100);

%% functions
function data_before_time = getDataBeforeTime(EEG, event_time, interval_length)
    fs = EEG.srate;
    interval_length_samples = interval_length * 60 * fs; 
    end_sample = event_time * fs;
    start_sample = end_sample - interval_length_samples + 1;
    data_before_time = EEG.data(:, start_sample:end_sample);
end

function data_matrix = getMatrix(data, epoch_length, fs)
    epoch_length_samples = epoch_length * fs; 
    num_epochs = floor(size(data, 2) / epoch_length_samples);
    data_matrix = reshape(data(:, 1:num_epochs * epoch_length_samples), size(data, 1), ...
        epoch_length_samples, num_epochs);
end

function features = getFeature(data_matrix, fs)
    [num_channels, ~, num_epochs] = size(data_matrix);
    features = struct('shannon_entropy', [], 'mean', [], 'std', [], 'min', [], 'max', []);

    window_length = 256;
    noverlap = 128;
    nfft = 512;

    for ch = 1:num_channels
        for ep = 1:num_epochs
            data = data_matrix(ch, :, ep);
            [pxx, ~] = pwelch(data, window_length, noverlap, nfft, fs);
            pxx_norm = pxx / sum(pxx);
            shannon_entropy = -sum(pxx_norm .* log2(pxx_norm + eps));
            
            features.shannon_entropy(ch, ep) = shannon_entropy;
            features.mean(ch, ep) = mean(data);
            features.std(ch, ep) = std(data);
            features.min(ch, ep) = min(data);
            features.max(ch, ep) = max(data);
        end
    end
end

function results = getResult(features)
    num_channels = size(features.shannon_entropy, 1);
    results = struct(); 
    
    feature_names = fieldnames(features);
    
    for f = 1:numel(feature_names)
        feature = feature_names{f};
        results.(feature).p_values = zeros(num_channels, 1);
        results.(feature).is_significant = false(num_channels, 1);
    end
    
    for f = 1:numel(feature_names)
        feature = feature_names{f};
        for ch = 1:num_channels
            feature_data = features.(feature)(ch, :);
            
            [h, p] = ttest(feature_data, 0, 'Alpha', 0.005);
            
            results.(feature).p_values(ch) = p;
            results.(feature).is_significant(ch) = h; 
        end
    end
end

function p_values = getPs(results)
    p_values = struct();
    feature_names = fieldnames(results);

    for f = 1:numel(feature_names)
        feature = feature_names{f};
        p_values.(feature) = results.(feature).p_values;
    end
end

function TPR = calculate_tpr(predictions, labels)
    predictions = predictions(:);
    labels = labels(:);
    
    TP = sum((predictions == 1) & (labels == 1));
    
    FN = sum((predictions == 0) & (labels == 1));

    if TP + FN > 0
        TPR = TP / (TP + FN);
    else
        TPR = 0;
    end
end

function TNR = calculate_tnr(predictions, labels)
    predictions = predictions(:);
    labels = labels(:);
    
    TN = sum((predictions == 0) & (labels == 0));
    
    FP = sum((predictions == 1) & (labels == 0));
    
    if TN + FP > 0
        TNR = TN / (TN + FP);
    else
        TNR = 0; 
    end
end

