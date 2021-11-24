function genPerfMat(seqs, trackers, evalType, nameTrkAll, perfMatPath)

pathAnno = './anno/';
numTrk = length(trackers);

thresholdSetOverlap = 0:0.05:1;
thresholdSetError = 0:50;

switch evalType
    case 'SRE'
        rpAll=['./results/results_SRE/'];
    case {'TRE'}
        rpAll=['./results/results_TRE/'];
    case {'OPE'}
        rpAll=['./results/results_OPE/'];
end

for idxSeq=1:length(seqs)
    losseq={};
    decineseq = seqs;
    loss = 0;
    failedseq = {2,3,14,18,42,43,87,89,97,102,104,106,116};
    for dexfail = 1:length(failedseq)
        failedseq{dexfail} = failedseq{dexfail} - loss;
        losseq{dexfail}=decineseq{failedseq{dexfail}}.name;
        decineseq(failedseq{dexfail}) = [];   
        
        loss=loss+1;
    end  
    
    s = seqs{idxSeq};
    
%     if idxSeq <= length(decineseq)
%         h = decineseq{idxSeq};
%         h.len = h.endFrame - h.startFrame + 1;
%         h.s_frames = cell(h.len,1);
%         nz	= strcat('%0',num2str(h.nz),'d'); %number of zeros in the name of image
%         for i=1:h.len
%             image_no = h.startFrame + (i-1);
%             id = sprintf(nz,image_no);
%             h.s_frames{i} = strcat(h.path,id,'.',h.ext);
%         end
% 
%         rect_anno = dlmread([pathAnno s.name '.txt']);
%         numSeg = 20;
%         [subSeqs, subAnno]=splitSeqTRE(s,numSeg,rect_anno);
% 
%         nameAll=[];   
%     end
    s.len = s.endFrame - s.startFrame + 1;
    s.s_frames = cell(s.len,1);
    nz	= strcat('%0',num2str(s.nz),'d'); %number of zeros in the name of image
    for i=1:s.len
        image_no = s.startFrame + (i-1);
        id = sprintf(nz,image_no);
        s.s_frames{i} = strcat(s.path,id,'.',s.ext);
    end
    
    rect_anno = dlmread([pathAnno s.name '.txt']);
    numSeg = 20;
    [subSeqs, subAnno]=splitSeqTRE(s,numSeg,rect_anno);
    
    nameAll=[];
    for idxTrk=1:numTrk
        judge = 0;
        judgestr='HCFT_no_var,HCFT_var';
        t = trackers{idxTrk};
        %         load([rpAll s.name '_' t.name '.mat'], 'results','coverage','errCenter');
        for judex = 1:length(losseq)
            sname = s.name;
            lossname = losseq{judex};
            tname = t.name;
            if contains(sname,lossname) && contains(judgestr,tname)               
                judge = 1;
                break;              
            end
        end
        
        if contains(s.name,'car1_s') && contains('DSiam_with_TB',t.name)
            judge = 1;
        end
        
        if contains(s.name,'car3_s') && contains('DSiam_with_TB',t.name)
            judge = 1;
        end
        
        if contains(s.name,'car3_s') && contains('DSiam_with_IB ',t.name)
            judge = 1;
        end
        if contains(s.name,'uav2') && contains('DAT_with_IB',t.name)
            judge = 1;
        end
        
        if contains(s.name,'uav2') && contains('DAT_with_TB',t.name)
            judge = 1;
        end
        
        if contains(s.name,'person21') && contains('DAT_with_IB',t.name)
            judge = 1;
        end
        
        if contains(s.name,'person21') && contains('DAT_with_TB',t.name)
            judge = 1;
        end
        
        if contains(s.name,'car15') && contains('DAT_with_IB',t.name)
            judge = 1;
        end
        
        if contains(s.name,'car15') && contains('DAT_with_TB',t.name)
            judge = 1;
        end
        if judge == 1
            continue;
        end
        load([rpAll s.name '_' t.name '.mat'])
        disp([s.name ' ' t.name]);
        
        aveCoverageAll=[];
        aveErrCenterAll=[];
        errCvgAccAvgAll = 0;
        errCntAccAvgAll = 0;
        errCoverageAll = 0;
        errCenterAll = 0;
        
        lenALL = 0;
        
        switch evalType
            case 'SRE'
                idxNum = length(results);
                anno=subAnno{1};
            case 'TRE'
                idxNum = length(results);
            case 'OPE'
                idxNum = 1;
                anno=subAnno{1};
        end
        
        successNumOverlap = zeros(idxNum,length(thresholdSetOverlap));
        successNumErr = zeros(idxNum,length(thresholdSetError));
        
        for idx = 1:idxNum
            res = results(idx);
            
            if strcmp(evalType, 'TRE')
                anno=subAnno{idx};
            end
            
            len = size(anno,1);
            
            if isempty(res)
                break;
            elseif isempty(res.result)
                break;
            end
            
            if ~isfield(res,'type')&&isfield(res,'transformType')
                res.type = res.transformType;
                res.result = res.result';
            end
            
            [aveCoverage, aveErrCenter, errCoverage, errCenter] = calcSeqErrRobust(res, anno);
            
            for tIdx=1:length(thresholdSetOverlap)
                successNumOverlap(idx,tIdx) = sum(errCoverage >thresholdSetOverlap(tIdx));
            end
            
            for tIdx=1:length(thresholdSetError)
                successNumErr(idx,tIdx) = sum(errCenter <= thresholdSetError(tIdx));
            end
            
            lenALL = lenALL + len;
            
        end
        
        
        if strcmp(evalType, 'OPE')
            aveSuccessRatePlot(idxTrk, idxSeq,:) = successNumOverlap/(lenALL+eps);
            aveSuccessRatePlotErr(idxTrk, idxSeq,:) = successNumErr/(lenALL+eps);
        else
            aveSuccessRatePlot(idxTrk, idxSeq,:) = sum(successNumOverlap)/(lenALL+eps);
            aveSuccessRatePlotErr(idxTrk, idxSeq,:) = sum(successNumErr)/(lenALL+eps);
        end
        
    end
end
%
dataName1=[perfMatPath 'aveSuccessRatePlot_' num2str(numTrk) 'alg_overlap_' evalType '.mat'];
save(dataName1,'aveSuccessRatePlot','nameTrkAll');

dataName2=[perfMatPath 'aveSuccessRatePlot_' num2str(numTrk) 'alg_error_' evalType '.mat'];
aveSuccessRatePlot = aveSuccessRatePlotErr;
save(dataName2,'aveSuccessRatePlot','nameTrkAll');
