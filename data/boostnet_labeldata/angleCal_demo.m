
ap_num = 32;
lat_num = 32;

%get labels
folder_l = 'APL_Labels/';
% folder_l = 'C:\Users\zhenxt\Documents\Python Scripts\Spine\labels\training\';
files_l = [folder_l  '*.mat'];

dirOutput_l = dir(files_l);
fileNames_l = {dirOutput_l.name}';

N = size(fileNames_l,1);

% get image
folder_im = 'APL_Crops/';
% folder_im = 'C:\Users\zhenxt\Documents\Python Scripts\Spine\data\training\';
% files_im = [folder_im  '*.jpg'];
% 
% dirOutput_im = dir(files_im);
% fileNames_im = {dirOutput_im.name}';
fileNames_im{N}=[];
for k=1:N
    fileNames_im{k}=fileNames_l{k}(1:end-4);
end

CobbAn_ap = [];
CobbAn_lat = [];
landmarks_ap = [];
landmarks_lat = [];
%landmarks = csvread('C:\Users\zhenxt\Documents\Python Scripts\Spine\output\BoostNet\landmarks.csv');

for k=1:N
    %get images
    l = [folder_im fileNames_im{k}];
    im = imread(l);
    [H,W] = size(im);
    
    %get labels
    l_n = [folder_l fileNames_l{k}];
    p = load(l_n);
    
    coord = p.p2;
    
%     p2 = [landmarks(k,1:68) ; landmarks(k,69:136)]';
    if isempty(strfind(lower(l_n),'lateral'))
        p2 = [coord(1:ap_num) ; coord(ap_num+1:ap_num*2)]';
        vnum = ap_num / 4;
        landmarks_ap = [landmarks_ap ; coord(1:ap_num)/W, coord(ap_num+1:ap_num*2)/H]; %scale landmark coordinates
    else
        p2 = [coord(1:lat_num) ; coord(lat_num+1:lat_num*2)]';
        vnum = lat_num / 4;
        landmarks_lat = [landmarks_lat ; coord(1:lat_num)/W, coord(lat_num+1:lat_num*2)/H]; %scale landmark coordinates
    end
    cob_angles = zeros(1,3);
        
    figure,imshow(im)
    title('GroundTruth');
    hold on
    
    mid_p_v = zeros(size(p2,1)/2,2);
    for n=1:size(p2,1)/2
        mid_p_v(n,:) = (p2(n*2,:) + p2((n-1)*2+1,:))/2;
    end
    
    
    %calculate the middle vectors & plot the labeling lines
    mid_p = zeros(size(p2,1)/2,2);
    for n=1:size(p2,1)/4
        mid_p((n-1)*2+1,:) = (p2(n*4-1,:) + p2((n-1)*4+1,:))/2;
        mid_p(n*2,:) = (p2(n*4,:) + p2((n-1)*4+2,:))/2;
    end
    
    
    %pause(1)
    %plot the midpoints
    plot(mid_p(:,1),mid_p(:,2),'y.','MarkerSize',20);
    %pause(1)
    
    
    vec_m = zeros(size(mid_p,1)/2,2);
    for n=1:size(mid_p,1)/2
        vec_m(n,:) = mid_p(n*2,:) - mid_p((n-1)*2+1,:);
        %plot the midlines
        plot([mid_p(n*2,1),mid_p((n-1)*2+1,1)],...
            [mid_p(n*2,2),mid_p((n-1)*2+1,2)],'Color','r','LineWidth',2);
    end
    
    mod_v = power(sum(vec_m .* vec_m, 2),0.5);
    dot_v = vec_m * vec_m';
    
    %calculate the Cobb angle
    angles = acos(roundn(dot_v./(mod_v * mod_v'),-8));
    [maxt, pos1] = max(angles);
    [pt, pos2] = max(maxt);
    pt = pt/pi*180;
    cob_angles(1) = pt;
    
    %plot the selected lines
    %pause(1)
    plot([mid_p(pos2*2,1),mid_p((pos2-1)*2+1,1)],...
        [mid_p(pos2*2,2),mid_p((pos2-1)*2+1,2)],'Color','g','LineWidth',2);
    plot([mid_p(pos1(pos2)*2,1),mid_p((pos1(pos2)-1)*2+1,1)],...
        [mid_p(pos1(pos2)*2,2),mid_p((pos1(pos2)-1)*2+1,2)],'Color','g','LineWidth',2);
    
    if ~isS(mid_p_v) % 'S'
        
        
        
        mod_v1 = power(sum(vec_m(1,:) .* vec_m(1,:), 2),0.5);
        mod_vs1 = power(sum(vec_m(pos2,:) .* vec_m(pos2,:), 2),0.5);
        mod_v2 = power(sum(vec_m(vnum,:) .* vec_m(vnum,:), 2),0.5);
        mod_vs2 = power(sum(vec_m(pos1(pos2),:) .* vec_m(pos1(pos2),:), 2),0.5);
        
        dot_v1 = vec_m(1,:) * vec_m(pos2,:)';
        dot_v2 = vec_m(vnum,:) * vec_m(pos1(pos2),:)';
        
        mt = acos(roundn(dot_v1./(mod_v1 * mod_vs1'),-8));
        tl = acos(roundn(dot_v2./(mod_v2 * mod_vs2'),-8));
        
        mt = mt/pi*180;
        cob_angles(2) = mt;
        tl = tl/pi*180;
        cob_angles(3) = tl;
        
    else
        
    % max angle in the upper part
        if (mid_p_v(pos2*2,2) + mid_p_v(pos1(pos2)*2,2)) < size(im,1)
            
            %calculate the Cobb angle (upside)
            mod_v_p = power(sum(vec_m(pos2,:) .* vec_m(pos2,:), 2),0.5);
            mod_v1 = power(sum(vec_m(1:pos2,:) .* vec_m(1:pos2,:), 2),0.5);
            dot_v1 = vec_m(pos2,:) * vec_m(1:pos2,:)';
            
            
            angles1 = acos(roundn(dot_v1./(mod_v_p * mod_v1'),-8));
            [CobbAn1, pos1_1] = max(angles1);
            mt = CobbAn1/pi*180;
            cob_angles(2) = mt;
            
            plot([mid_p(pos1_1*2,1),mid_p((pos1_1-1)*2+1,1)],...
                [mid_p(pos1_1*2,2),mid_p((pos1_1-1)*2+1,2)],'Color','g','LineWidth',2);
            
            
            %calculate the Cobb angle?downside?
            mod_v_p2 = power(sum(vec_m(pos1(pos2),:) .* vec_m(pos1(pos2),:), 2),0.5);
            mod_v2 = power(sum(vec_m(pos1(pos2):vnum,:) .* vec_m(pos1(pos2):vnum,:), 2),0.5);
            dot_v2 = vec_m(pos1(pos2),:) * vec_m(pos1(pos2):vnum,:)';
            
            angles2 = acos(roundn(dot_v2./(mod_v_p2 * mod_v2'),-8));
            [CobbAn2, pos1_2] = max(angles2);
            tl = CobbAn2/pi*180;
            cob_angles(3) = tl;
            
            pos1_2 = pos1_2 + pos1(pos2) - 1;
            plot([mid_p(pos1_2*2,1),mid_p((pos1_2-1)*2+1,1)],...
                [mid_p(pos1_2*2,2),mid_p((pos1_2-1)*2+1,2)],'Color','g','LineWidth',2);
            
        else
            %calculate the Cobb angle (upside)
            mod_v_p = power(sum(vec_m(pos2,:) .* vec_m(pos2,:), 2),0.5);
            mod_v1 = power(sum(vec_m(1:pos2,:) .* vec_m(1:pos2,:), 2),0.5);
            dot_v1 = vec_m(pos2,:) * vec_m(1:pos2,:)';
            
            
            angles1 = acos(roundn(dot_v1./(mod_v_p * mod_v1'),-8));
            [CobbAn1, pos1_1] = max(angles1);
            mt = CobbAn1/pi*180;
            cob_angles(2) = mt;
            
            plot([mid_p(pos1_1*2,1),mid_p((pos1_1-1)*2+1,1)],...
                [mid_p(pos1_1*2,2),mid_p((pos1_1-1)*2+1,2)],'Color','g','LineWidth',2);
            
            
            %calculate the Cobb angle (upper upside)
            mod_v_p2 = power(sum(vec_m(pos1_1,:) .* vec_m(pos1_1,:), 2),0.5);
            mod_v2 = power(sum(vec_m(1:pos1_1,:) .* vec_m(1:pos1_1,:), 2),0.5);
            dot_v2 = vec_m(pos1_1,:) * vec_m(1:pos1_1,:)';
            
            angles2 = acos(roundn(dot_v2./(mod_v_p2 * mod_v2'),-8));
            [CobbAn2, pos1_2] = max(angles2);
            tl = CobbAn2/pi*180;
            cob_angles(3) = tl;
            
            %pos1_2 = pos1_2 + pos1(pos2) - 1;
            plot([mid_p(pos1_2*2,1),mid_p((pos1_2-1)*2+1,1)],...
                [mid_p(pos1_2*2,2),mid_p((pos1_2-1)*2+1,2)],'Color','g','LineWidth',2);
        end
    end
    
    %pop up a text window
%     pause(1)
    output = [ num2str(k) ': the Cobb Angles(PT, MT, TL/L) are '  num2str(pt) ', ' num2str(mt) ' and '  num2str(tl) ...
        ', and the two most tilted vertebrae are ' num2str(pos2) ' and ' num2str(pos1(pos2)) '.\n'];
    %h = msgbox(output);
    
    fprintf(output);
    %         fprintf('No. %d :The Cobb Angles(PT, MT, TL/L) are %3.1f, and the two most tilted vertebrae are %d and %d. ',...
    %             k,CobbAn,pos2,pos1(pos2));
    
    %pause(2)
    close all
    if isempty(strfind(lower(fileNames{k}),'lateral'))
        CobbAn_ap = [CobbAn_ap ; cob_angles]; %cobb angles
    else
        CobbAn_lat = [CobbAn_lat ; cob_angles]; %cobb angles
    end

end

% write to csv file
csvwrite('angles_ap.csv',CobbAn_ap);
csvwrite('angles_lat.csv',CobbAn_lat);
csvwrite('landmarks_ap.csv',landmarks_ap);
csvwrite('landmarks_lat.csv',landmarks_lat);
fid = fopen('filenames_aplat.csv','wt');
if fid>0
    for k=1:N
        fprintf(fid,'%s\n',fileNames_im{k});
    end
        fclose(fid);
end




