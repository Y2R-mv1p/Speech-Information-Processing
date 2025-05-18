clear all;
load('hmm_single_speaker.mat'); % 加载训练好的模型

% 实验配置
train_speaker = 'p2';    % 训练时使用的说话人
test_speakers = {'p1', 'p2', 'p3'}; % 包含训练和其他说话人

% 初始化统计变量
results = struct();
for s = 1:length(test_speakers)
    speaker = test_speakers{s};
    results.(speaker).correct = 0;
    results.(speaker).total = 0;
end

% 执行测试
for s = 1:length(test_speakers)
    current_speaker = test_speakers{s};
    
    for digit = 0:9
        % 读取测试文件
        file_path = sprintf('Data/%s/%d.wav', current_speaker, digit);
        [sph, fs] = audioread(file_path);
        
        % 特征提取
        rec_fea = mfcc(sph);
        
        % 计算概率
        pxsm = zeros(1,10);
        for i = 1:10
            pxsm(i) = viterbi(hmm{i}, rec_fea);
        end
        
        % 获取识别结果
        [~, pred] = max(pxsm);
        true_label = digit + 1;
        
        % 更新统计
        results.(current_speaker).total = results.(current_speaker).total + 1;
        if pred == true_label
            results.(current_speaker).correct = results.(current_speaker).correct + 1;
        end
    end
end

% 显示结果
fprintf('\n=== 实验结果 ===\n');
fprintf('训练说话人: %s\n', train_speaker);
for s = 1:length(test_speakers)
    speaker = test_speakers{s};
    acc = results.(speaker).correct / results.(speaker).total * 100;
    if strcmp(speaker, train_speaker)
        fprintf('同说话人识别率: %.2f%% (%d/%d)\n', acc, ...
                results.(speaker).correct, results.(speaker).total);
    else
        fprintf('说话人%s识别率: %.2f%% (%d/%d)\n', speaker, acc, ...
                results.(speaker).correct, results.(speaker).total);
    end
end