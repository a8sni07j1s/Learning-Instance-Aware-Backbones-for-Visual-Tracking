function eval_tracker(seqs, trackers, eval_type, name_tracker_all, tmp_mat_path, path_anno, rp_all, norm_dst)
% evaluate each tracker
num_tracker = numel(trackers);

threshold_set_overlap = 0:0.05:1;
threshold_set_error   = 0:50;
if norm_dst
    threshold_set_error = threshold_set_error / 100;
end

for i = 1:numel(seqs) % for each sequence
    s    = seqs{i};      % name of sequence
    
    % load GT and the absent flags
    anno        = dlmread([path_anno s '.txt']);
    absent_anno = dlmread([path_anno 'absent/' s '.txt']);
    
    
%     sub=dir(txt_folder) ; 
%     txtname = []; j = 0;    
%     for i = 1:length(sub)
%         if strcmp(sub(i).name, '.tank') || strcmp(sub(i).name, '..') 
%             continue
%     end
    
    
    for k = 1:num_tracker  % evaluate each tracker
        t = trackers{k};   % name of tracker
        
        % load tracking result
        tracker_seq_path = [rp_all t.name '_tracking_result/' s '.txt'];
%         crest_root = ['./tracking_results/CREST_Var_tracking_result/' s '.txt']; 
%         crest_no_root = ['./tracking_results/CREST_with_TB_tracking_result/' s '.txt']; 
%         split_root = strsplit(crest_root,'-') ;
%         first_name = split_root{1,1};
%         split_no_root = strsplit(crest_no_root,'-') ;
%         first_no_name = split_no_root{1,1};
        
        if strcmp(tracker_seq_path,'./tracking_results/DSiam_with_TB_tracking_result/drone-13.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/DSiam_with_TB_tracking_result/tank-14.txt')
            continue
%         elseif strcmp(first_name,'./tracking_results/CREST_Var_tracking_result/tiger') 
%             continue
%         elseif strcmp(first_name,'./tracking_results/CREST_Var_tracking_result/tank') 
%             continue
%         elseif strcmp(first_name,'./tracking_results/CREST_Var_tracking_result/train') 
%             continue
%         elseif strcmp(first_name,'./tracking_results/CREST_Var_tracking_result/truck') 
%             continue
%         elseif strcmp(first_name,'./tracking_results/CREST_Var_tracking_result/turtle') 
%             continue
%         elseif strcmp(first_name,'./tracking_results/CREST_Var_tracking_result/umbrella') 
%             continue
%         elseif strcmp(first_name,'./tracking_results/CREST_Var_tracking_result/volleyball') 
%             continue
%         elseif strcmp(first_name,'./tracking_results/CREST_Var_tracking_result/yoyo') 
%             continue
%         elseif strcmp(first_name,'./tracking_results/CREST_Var_tracking_result/zebra') 
%             continue
%         elseif strcmp(first_name,'./tracking_results/CREST_Var_tracking_result/swing') 
%             continue
%         elseif strcmp(first_no_name,'./tracking_results/CREST_with_TB_tracking_result/tiger') 
%             continue
%         elseif strcmp(first_no_name,'./tracking_results/CREST_with_TB_tracking_result/tank') 
%             continue
%         elseif strcmp(first_no_name,'./tracking_results/CREST_with_TB_tracking_result/train') 
%             continue
%         elseif strcmp(first_no_name,'./tracking_results/CREST_with_TB_tracking_result/truck') 
%             continue
%         elseif strcmp(first_no_name,'./tracking_results/CREST_with_TB_tracking_result/turtle') 
%             continue
%         elseif strcmp(first_no_name,'./tracking_results/CREST_with_TB_tracking_result/umbrella') 
%             continue
%         elseif strcmp(first_no_name,'./tracking_results/CREST_with_TB_tracking_result/volleyball') 
%             continue
%         elseif strcmp(first_no_name,'./tracking_results/CREST_with_TB_tracking_result/yoyo') 
%             continue
%         elseif strcmp(first_no_name,'./tracking_results/CREST_with_TB_tracking_result/zebra') 
%             continue
%         elseif strcmp(first_no_name,'./tracking_results/CREST_with_TB_tracking_result/swing') 
%             continue
        elseif strcmp(tracker_seq_path,'./tracking_results/DSiam_with_TB_tracking_result/truck-6.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/DSiam_with_TB_tracking_result/truck-7.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/DSiam_with_TB_tracking_result/turtle-9.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/DSiam_with_TB_tracking_result/hippo-1.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/DSiam_with_TB_tracking_result/lizard-13.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/DSiam_with_IB_tracking_result/drone-13.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/DSiam_with_IB_tracking_result/truck-7.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/DSiam_with_IB_tracking_result/turtle-9.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/DSiam_with_IB_tracking_result/turtle-8.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/DSiam_with_IB_tracking_result/lizard-13.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/DSiam_with_IB_tracking_result/lizard-3.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/HCFT_with_TB_tracking_result/drone-2.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/HCFT_with_TB_tracking_result/drone-7.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/HCFT_with_TB_tracking_result/electricfan-18.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/HCFT_with_TB_tracking_result/gametarget-2.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/HCFT_with_TB_tracking_result/gecko-1.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/HCFT_with_TB_tracking_result/kangaroo-14.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/HCFT_with_TB_tracking_result/microphone-16.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/HCFT_with_TB_tracking_result/skateboard-3.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/HCFT_with_TB_tracking_result/kite-15.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/HCFT_with_TB_tracking_result/licenseplate-12.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/HCFT_with_IB_tracking_result/drone-13.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/HCFT_with_IB_tracking_result/drone-2.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/HCFT_with_IB_tracking_result/drone-7.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/HCFT_with_IB_tracking_result/gecko-1.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/HCFT_with_IB_tracking_result/guitar-8.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/HCFT_with_IB_tracking_result/kangaroo-14.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/HCFT_with_IB_tracking_result/racing-20.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/HCFT_with_IB_tracking_result/kite-15.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/HCFT_with_IB_tracking_result/licenseplate-12.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/HCFT_with_IB_tracking_result/monkey-9.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/HCFT_with_IB_tracking_result/motorcycle-18.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/HCFT_with_IB_tracking_result/mouse-8.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/DSiam_with_IB_tracking_result/airplane-9.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/HCFT_with_TB_tracking_result/racing-20.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/HCFT_with_IB_tracking_result/volleyball-19.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/swing-14.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/swing-14.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/swing-17.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/swing-17.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/swing-20.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/swing-20.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/swing-11.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/swing-11.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/swing-12.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/swing-12.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/swing-13.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/swing-13.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/swing-15.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/swing-15.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/swing-18.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/swing-18.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/swing-19.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/swing-19.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/swing-15.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/swing-15.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/tank-14.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/tank-14.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/tank-16.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/tank-16.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/tank-6.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/tank-6.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/tank-7.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/tank-7.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/tank-9.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/tank-9.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/tank-10.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/tank-10.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/tank-11.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/tank-11.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/tiger-12.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/tiger-12.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/tiger-18.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/tiger-18.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/tiger-4.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/tiger-4.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/tiger-12.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/tiger-12.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/tiger-6.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/tiger-6.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/train-1.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/train-1.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/train-11.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/train-11.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/train-20.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/train-20.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/train-7.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/train-7.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/truck-16.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/truck-16.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/truck-3.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/truck-3.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/truck-6.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/truck-6.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/truck-7.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/truck-7.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/turtle-16.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/turtle-16.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/turtle-5.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/turtle-5.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/turtle-8.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/turtle-8.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/turtle-9.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/turtle-9.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/umbrella-17.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/umbrella-17.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/umbrella-19.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/umbrella-19.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/umbrella-2.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/umbrella-2.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/umbrella-9.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/umbrella-9.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/yoyo-15.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/yoyo-15.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/yoyo-17.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/yoyo-17.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/yoyo-19.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/yoyo-19.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/yoyo-7.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/yoyo-7.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/zebra-10.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/zebra-10.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/zebra-14.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/zebra-14.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/zebra-16.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/zebra-16.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/zebra-17.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/zebra-17.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/volleyball-1.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/volleyball-1.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/volleyball-13.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/volleyball-13.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/volleyball-18.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/volleyball-18.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_TB_tracking_result/volleyball-19.txt')
            continue
        elseif strcmp(tracker_seq_path,'./tracking_results/CREST_with_IB_tracking_result/volleyball-19.txt')
            continue
        else
            res = dlmread(tracker_seq_path);
        end
%         res = dlmread([rp_all t.name '_tracking_result/' s '.txt']);
%         res = load([rp_all t.name '_tracking_result/' s '.txt']);
%         results = load([rp_all t.name '_tracking_result/' s '.txt']);
%         results = load([rp_all s '_' t.name '.mat']);
%         results = results.results;
        fprintf(['evaluating ' t.name ' on ' s ' ...\n']);
        
%         anno     = rect_anno;
        
        success_num_overlap = zeros(1, numel(threshold_set_overlap));
        success_num_err     = zeros(1, numel(threshold_set_error));
        
%         res = results{1};
        
        if isempty(res)
            break;
        end
        
        [err_coverage, err_center] = calc_seq_err_robust(res, anno, absent_anno, norm_dst);
        
        for t_idx = 1:numel(threshold_set_overlap)
            success_num_overlap(1, t_idx) = sum(err_coverage > threshold_set_overlap(t_idx));
        end
        
        for t_idx = 1:length(threshold_set_error)
            success_num_err(1, t_idx) = sum(err_center <= threshold_set_error(t_idx));
        end
        
        len_all = size(anno, 1);  % number of frames in the sequence
        
        ave_success_rate_plot(k, i, :)     = success_num_overlap/(len_all + eps);
%         fprintf('ave_success_rate_plot %d', ave_success_rate_plot(k, i, :)) ;
        ave_success_rate_plot_err(k, i, :) = success_num_err/(len_all + eps);
    end
end

% save results
if ~exist(tmp_mat_path, 'dir')
    mkdir(tmp_mat_path);
end

dataName1 = [tmp_mat_path 'aveSuccessRatePlot_' num2str(num_tracker) 'alg_overlap_' eval_type '.mat'];
save(dataName1, 'ave_success_rate_plot', 'name_tracker_all');

dataName2 = [tmp_mat_path 'aveSuccessRatePlot_' num2str(num_tracker) 'alg_error_' eval_type '.mat'];
ave_success_rate_plot = ave_success_rate_plot_err;
save(dataName2, 'ave_success_rate_plot', 'name_tracker_all');

end