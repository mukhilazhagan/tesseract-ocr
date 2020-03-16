clc
close all;

%Acquisition
pcb_red = imread('orig_red.jpg');
pcb_red_gray = rgb2gray(pcb_red);
im = pcb_red_gray;

%Histogram equalization of pcb_red_gray
hEq_gray = histeq(im);
figure(1), imshow(im), title('Gray scale Image')
figure(2), imshow(hEq_gray), title('Histogram Equalized Image')

% Gamma modification
gamma = 2.2;
gammaCorrected(gamma) = GammaCorrection(im,gamma); %Using function GammaCorrection
figure(3), imshow(gammaCorrected), title('GammaCorrected Image')

% Thresholding Gamma modified image at threshold value of 127
[row, col] = size(gammaCorrected);
t_image = zeros(row,col);
thresh = 127;
for i = 1:row
    for j = 1:col
        if gammaCorrected(i,j)>thresh
            t_image(i,j) = 1;
        else
            t_image(i,j) = 0;
        end
    end
end
figure, imshow(t_image), title('Thresholded image')
imwrite(t_image, 'pcb_red_gammaStrech.jpg')



