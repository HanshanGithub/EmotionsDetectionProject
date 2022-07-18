img1 = imread('cutA.png');
img2 = imread('cutB.png');
subplot(1,2,1);
imshow(img1)
title('原始数据','FontSize',20)
subplot(1,2,2)
imshow(img2)
title('人脸对齐处理后','FontSize',20)


% 获取figure界面
frame = getframe(gcf);
% 转为图像
im = frame2im(frame);
% 保存
imwrite(im,'data2.png');
