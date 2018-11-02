% Codes for CVPR-15 work `Face Alignment by Coarse-to-Fine Shape Searching'
% Any question please contact Shizhan Zhu: zhshzhutah2@gmail.com
% Released on July 25, 2015

function feat = extractSIFTs_toosimple( images , currentPose, iter, featInfo, level, ii, user, mode)

if nargin < 4 || isempty( featInfo )
    error('you didn''t set SIFT param!!!');
end
scale = featInfo.scale(iter);
n = length( images );
n_pts = size( currentPose,2)/2;
feat = zeros(n,128*n_pts);

if isempty(currentPose), return; end;

% if (length(images)<100) || (parpool('size')==0)
%     for i = 1:length(images)
%         pts = reshape(currentPose(i,:),n_pts,2);
%         descriptor = extractSIFT_toosimple( single( images{i} ), pts ,scale);
%         feat(i,:) = descriptor(:)';
%     end
%     if length(images)>=500
%         warning('Please launch parpool to speed up your program!');
%     end
% else
model_file = sprintf('./model/extractSIFS_toosimple_%d_%d_%d_%s_%s.mat', iter, level, ii, user, mode);
if exist(model_file) 
    str = cell2mat(['load' char(32) {['./model/extractSIFS_toosimple_' num2str(iter) '_' num2str(level) '_' num2str(ii) '_' user '_' mode '.mat']} char(32) ' feat;']);
    eval(str);
else 
    parfor i = 1:length(images)
        feat(i,:) = reshape(extractSIFT_toosimple( single( images{i} ), reshape(currentPose(i,:),n_pts,2) ,scale),...
            1,128*n_pts)';
        if mod(i, 1000) == 0
            disp(['extracted ' num2str(i) 'features']);
        end
    end
    str = sprintf('./model/extractSIFS_toosimple_%d_%d_%d_%s_%s.mat', iter, level, ii, user, mode);
    save(str, 'feat', '-v7.3');
end


