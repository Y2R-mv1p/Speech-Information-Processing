clear all;

% ===== 配置区域 =====
data_root = '';        % 语音数据根目录
train_speakers = {               % 需要参与训练的所有说话人（性别+ID）
    {'female', 'ryr'}, {'female', 'ljq'}, {'female', 'lby'},...
    {'female', 'bd'},  {'female', 'jzy'}, {'female', 'alx'},...
    {'male', 'fwx'},   {'male', 'rj'},    {'male', 'rzy'},...
    {'male', 'p1'},    {'male', 'p2'},    {'male', 'p3'}
};
num_digits = 10;                 % 0-9共10个数字
hmm_params.N = 5;                % HMM状态数（建议5-8）
hmm_params.M = [3,3,3,3,3];      % 每个状态的高斯混合数
% ===================

% 初始化数据结构（按数字分组）
tdata = cell(1, num_digits); 

% 加载所有说话人数据
for s = 1:length(train_speakers)
    % 解析说话人信息
    gender = train_speakers{s}{1};
    speaker_id = train_speakers{s}{2};
    
    fprintf('正在加载 %s/%s 的数据...\n', gender, speaker_id);
    
    % 加载该说话人的所有数字语音
    for d = 0:num_digits-1
        file_path = fullfile(data_root, gender, speaker_id, [num2str(d) '.wav']);
        
        % 异常处理
        try
            [y, fs] = audioread(file_path);
            tdata{d+1}{end+1} = y; % 按数字分组存储语音
        catch
            warning('文件加载失败: %s', file_path);
        end
    end
end

% 保存预处理数据
save('tra_data.mat', 'tdata');
fprintf('\n数据预处理完成，共加载%d个说话人\n', length(train_speakers));

% 训练HMM模型
hmm = cell(1, num_digits);
for digit = 1:num_digits
    fprintf('\n=== 训练数字%d的HMM模型 ===\n', digit-1);
    
    % 提取该数字的所有语音特征
    obs_data = struct('sph', {}, 'fea', {});
    for k = 1:length(tdata{digit})
        obs_data(k).sph = tdata{digit}{k};
        obs_data(k).fea = mfcc(obs_data(k).sph); % MFCC特征提取
    end
    
    % HMM训练流程
    initial_hmm = inithmm(obs_data, hmm_params.N, hmm_params.M); % 初始化
    hmm{digit} = baum_welch(initial_hmm, obs_data);              % Baum-Welch训练
end

% 保存模型
save('hmm_multi_speaker.mat', 'hmm');
fprintf('\n训练完成！模型已保存为hmm_multi_speaker.mat\n');