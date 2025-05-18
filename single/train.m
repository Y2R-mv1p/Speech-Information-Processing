clear all;

% 设置训练说话人
train_speaker = 'alx'; % 使用p1的数据进行训练

% 初始化训练数据结构
tdata = cell(10,1); % 存储0-9共10个数字的语音

% 读取训练数据（单说话人）
for digit = 0:9
    file_path = sprintf('female/%s/%d.wav', train_speaker, digit);
    [sph, fs] = audioread(file_path);
    tdata{digit+1} = {sph}; % 将语音存入对应数字的位置
end

N = 4;   % HMM状态数
M = [3,3,3,3]; % 每个状态的混合高斯成分数

% 训练每个数字的HMM模型
hmm = cell(10,1); % 存储10个HMM模型
for i = 1:10
    fprintf('\n处理数字%d...', i-1);
    
    % 特征提取
    num_samples = length(tdata{i});
    obs = struct('sph', {}, 'fea', {});
    for k = 1:num_samples
        obs(k).sph = tdata{i}{k};
        obs(k).fea = mfcc(obs(k).sph);
    end
    
    % HMM训练
    hmm_temp = inithmm(obs, N, M);    % 初始化HMM
    hmm{i} = baum_welch(hmm_temp, obs); % Baum-Welch训练
end

% 保存模型
save('hmm_single.mat', 'hmm');
fprintf('\n训练完成！模型已保存为hmm_single.mat\n');