function [imCorrected] = GammaCorrection(im, gamma)

im = im2uint8(im);

c = 1;
r = 0:255; %input gray level

s = c*r.^gamma; % power law 
s = (s/max(s))*255; %output gray level
%figure, plot(r,s); title('Gamma Correction Function')

imCorrected = s(im+1);
imCorrected = uint8(imCorrected);

end