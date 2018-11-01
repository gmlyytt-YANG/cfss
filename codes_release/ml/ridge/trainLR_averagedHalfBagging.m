% Codes for CVPR-15 work `Face Alignment by Coarse-to-Fine Shape Searching'
% Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
% Released on July 25, 2015

function reg_model = trainLR_averagedHalfBagging(X,y,iter,regressorInfo, level)

% train regressor using all the samples only once

assert(size(X,1) == size(y,1));
m = size(X,1);
REG = zeros(size(X,2)+1,size(y,2));

model_file = sprintf('./model/trainLR_averageHalfBagging_%d_%d.mat', iter, level);
if exist(model_file) 
    str = cell2mat(['load' char(32) {['./model/trainLR_averageHalfBagging_' num2str(iter) '_' num2str(level) '.mat']} char(32) ' reg_model;']);
    eval(str);
else 
    for i = 1:regressorInfo.times
        train_ind = randperm(m,ceil(m/2));
        REG = REG + (1/regressorInfo.times) * ...
            regressorInfo.trainMethod(X(train_ind,:),y(train_ind,:),regressorInfo.lambda(iter));
        if mod(i, 5) == 0
            disp(['train ' num2str(i) 'times']);
        end
    end;
    reg_model = REG;
    str = cell2mat(['save' char(32) {['./model/trainLR_averageHalfBagging_' num2str(iter) '_' num2str(level) '.mat']} char(32) ' reg_model;']);
    eval(str);
end


end
