clear all
close all

observer_name = 'sunhl-1th-';

%------default setting
ap_num = 30;
lat_num = 30;
folder = 'images/';
labelFolder = 'APL_Labels/';
cropFolder = 'APL_Crops/';
files = [folder  '*.jpg'];

dirOutput = dir(files);
fileNames = {dirOutput.name}';


for k=1:size(fileNames,1)
    fname = fileNames{k};
    
    % get points
    if isempty(strfind(lower(fileNames{k}),'lat'))
        name1 = strsplit(fileNames{k})
        name2 = strsplit(fileNames{k+1})
        if strcmp(name1{1},name2{1}) & strfind(lower(fileNames{k+1}),'lat')
            point_num = ap_num
        else
            movefile([folder fname], ['bad data/' fname], 'f');
            continue;
        end
    else
        if k>1
            name1 = strsplit(fileNames{k})
            name2 = strsplit(fileNames{k-1})
            if strcmp(name1{1},name2{1}) & strfind(lower(fileNames{k-1}),'ap')
                point_num = lat_num
            else
                movefile([folder fname], ['bad data/' fname], 'f');
                continue;
            end
        else
            point_num = lat_num
        end
    end
    
    %read image
    crop_name = [cropFolder fileNames{k}];
    if exist(crop_name, 'file')
        crop = imread(crop_name);
    else
        image = imread([folder fname]);
        imsize = size(image);
        if imsize(end)==3
            image = rgb2gray(image);
        end

        %center image
        scrsz = get(0,'ScreenSize');
        h=figure(1);
        imshow(imadjust(image),[]);
        title(['crop',fname]);
        set(h,'OuterPosition',[1 1 scrsz(3) scrsz(4)]);

        %crop images
        h = imrect;
        wait(h);
        pos = getPosition(h);
        %save the cropped images
        crop_name = [cropFolder fileNames{k}];
        crop = image(pos(1,2):pos(1,2)+pos(1,4),pos(1,1):pos(1,1)+pos(1,3));
        imwrite(crop,crop_name);
    end
    
    %center cropped
    scrsz = get(0,'ScreenSize');
    f=figure(1);
    imshow(histeq(crop),[]);
    title(['mark',fname]);
    set(f,'OuterPosition',[1 1 scrsz(3) scrsz(4)]);
    
    %mark image
    left = zeros(point_num,2);
    right = zeros(point_num,2);
    mid = zeros(point_num,2);
    lines = {point_num};
    for i = 1 :point_num
        %h = impoint;
        h = imline;
        lines{i} = h;
        %wait(h);
        %p(i,:) = wait(h);
    end
    
    %wait for enter to continue
    while true
        w = waitforbuttonpress; 
        if w == 1 % (keyboard press) 
            key = get(gcf,'currentcharacter'); 
            switch key
                case 27 % 27 is the escape key
                    disp('User pressed the escape key. Skipping this bad image.')
                    movefile([folder fname], ['bad data/' fname], 'f');
                    break % break out of the while loop
                case 13 % 13 is the return key 
                    disp('User pressed the return key. Saving markers.')
                    % getting marker positions
                    for i = 1 :point_num
                        %h = impoint;
                        position = lines{i}.getPosition();
                        left(i,:) = position(1,:);
                        right(i,:) = position(2,:);
                        mid(i,:) = (position(2,:)+position(1,:))/2;
                        %p(i,:) = wait(h);
                    end

                    %makes sure markers are left->right
                    flipped = left(:,1)<right(:,1);
                    I = find(flipped==0);
                    if ~isempty(I)
                        for i = I'
                            [right(i,:),left(i,:)]=deal(left(i,:),right(i,:));
                            disp(['index ',int2str(i),' is flipped'])
                        end
                    end
                    
                    %sort points
                    [~,I] = sort(left(:,2));
                    left = left(I,:);
                    [~,I] = sort(right(:,2));
                    right = right(I,:);

                    %combine left and right markers
                    p1 = [left(:,1),right(:,1)].';
                    p2 = [left(:,2),right(:,2)].';
                    p1 = p1(:);
                    p2 = p2(:);
                    p = [p1;p2]';

                    %save the coordinates of labeled points
                    mat_name = [labelFolder fileNames{k}];
                    save([mat_name '.mat'],'p');
                    csvwrite([mat_name '.csv'],p)

                    %move image to finished
                    movefile([folder fname], ['finished/' fname], 'f');
                    break
                otherwise 
                    % Wait for a different command. 
            end
        end
    end
end