clear all
close all

observer_name = 'sunhl-1th-';

%------default setting
folder = 'finished/';
labelFolder = 'APL_Labels/';
cropFolder = 'APL_Crops/';
files = [folder  '*.jpg'];

dirOutput = dir(files);
fileNames = {dirOutput.name}';

orig_num = 17*2;


for k=1:size(fileNames,1)
    fname = fileNames{k};
    
    %load the coordinates of labeled points
    mat_name = [labelFolder fileNames{k}];
    load([mat_name '.mat'],'p');
    
    % get points
    point_num = length(p)/4;
    maxnum = max(point_num,orig_num);
    minnum = min(point_num,orig_num);
    
    %read image
    crop_name = [cropFolder fileNames{k}];
    crop = imread(crop_name);
    
    %center cropped
    scrsz = get(0,'ScreenSize');
    f=figure(1);
    imshow(histeq(crop),[]);
    title(['mark',fname]);
    set(f,'OuterPosition',[1 1 scrsz(3) scrsz(4)]);
    
    %mark image
    if point_num>orig_num
        lines = {point_num};
    else
        lines = {orig_num};
    end
    
    for i = 0 : orig_num-1
        %h = impoint;
        if i<minnum
            x1=i*2+1;
            y1=point_num*2+i*2+1;
            x2=i*2+2;
            y2=point_num*2+i*2+2;
            position = [p(x1),p(y1); p(x2),p(y2)];
            h = imline(gca,position);
        else
            h = imline(gca);
        end
        lines{i+1} = h;
        %p(i,:) = wait(h);
    end
    
    %wait for enter to continue
    while true
        w = waitforbuttonpress; 
        if w == 1 % (keyboard press) 
            key = get(gcf,'currentcharacter'); 
            switch key
                case 27 % 27 is the escape key
                    disp('No changes made')
                    break % break out of the while loop
                case 13 % 13 is the return key 
                    disp('User pressed the return key. Fixed markers.')
                    % read points
                    
                    left = zeros(orig_num,2);
                    right = zeros(orig_num,2);
                    mid = zeros(orig_num,2);
                    for i = 1 :orig_num
                        position = lines{i}.getPosition();
                        left(i,:) = position(1,:);
                        right(i,:) = position(2,:);
                        mid(i,:) = (position(2,:)+position(1,:))/2;
                        %p(i,:) = wait(h);
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

                    %save points
                    save([mat_name '.mat'],'p');
                    csvwrite([mat_name '.csv'],p)
                    
                    break
                otherwise 
                    % Wait for a different command. 
            end
        end
    end
end