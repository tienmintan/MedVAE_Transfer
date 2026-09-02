clear all
close all

observer_name = 'sunhl-1th-';

%------default setting
point_num = 0;
folder = 'images/';
labelFolder = 'SupLabels/';
cropFolder = 'SupCrops/';
files = [folder  '*.jpg'];

dirOutput = dir(files);
fileNames = {dirOutput.name}';


for k=1:size(fileNames,1)
 
    l = [folder fileNames{k}];
    im = imread(l);
    image = rgb2gray(im);
    scrsz = get(0,'ScreenSize');
    h=figure(1);
    imshow(imadjust(image),[]);
    title(fileNames(k));
    set(h,'OuterPosition',[1 1 scrsz(3) scrsz(4)]);
 
    % get points
    p = zeros(point_num,2);
    for i = 1 :point_num
        h = impoint;
        p(i,:) = wait(h);
    end
    
%     pause(10)

    %crop images
    h = imrect;
    wait(h);
    pos = getPosition(h);
        
    %save the coordinates of labeled points
%     p2 = round([p(:,1) - pos(1,1) ,p(:,2) - pos(1,2)]);
%     mat_name = [labelFolder observer_name date '-' fileNames{k}];
%     save([mat_name '.mat'],'p2');
    
    %save the cropped images
    crop_name = [cropFolder observer_name date '-' fileNames{k}];
    crop = im(pos(1,2):pos(1,2)+pos(1,4),pos(1,1):pos(1,1)+pos(1,3));
    imwrite(crop,crop_name);
    movefile(l, ['finished/' l(8:end)]);
       
end