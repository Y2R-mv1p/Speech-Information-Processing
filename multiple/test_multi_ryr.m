clear all;
load('hmm_models.mat', 'hmm'); % 加载训练好的HMM模型

% 设置测试数据（训练中的说话人p1/p2/p3和新说话人p4）
test_speakers = {'p1', 'p4'}; % 包含训练和未训练说话人
num_digits = 10;
results = struct('TrueDigit',[], 'PredDigit',[], 'SpeakerType',[]);

% 遍历所有测试数据
for s = 1:length(test_speakers)
    speaker = test_speakers{s};
    for d = 0:9
        file_path = fullfile('SpeechData', speaker, [num2str(d) '.wav']);
        [y, fs] = audioread(file_path);
        fea = mfcc(y); % 提取特征
        
        % 计算每个HMM的输出概率
        scores = zeros(1,10);
        for i = 1:10
            scores(i) = viterbi(hmm{i}, fea);
        end
        [~, pred] = max(scores);
        
        % 记录结果
        results(end+1).TrueDigit = d;
        results(end).PredDigit = pred-1; % 转换为0-9
        results(end).SpeakerType = ismember(speaker, {'p1','p2','p3'});
    end
end

% 分析结果
train_acc = sum([results.SpeakerType] & ([results.TrueDigit] == [results.PredDigit])) / sum([results.SpeakerType]);
untrain_acc = sum(~[results.SpeakerType] & ([results.TrueDigit] == [results.PredDigit])) / sum(~[results.SpeakerType]);

fprintf('训练说话人准确率: %.2f%%\n', train_acc*100);
fprintf('未训练说话人准确率: %.2f%%\n', untrain_acc*100);