close all;

I = imread('house_in.png');
%I = rgb2gray(I) ; 
for i = 0.05:0.05:0.8
i_noised = imnoise(I, 'salt & pepper', i);
imwrite(i_noised,strcat(strcat('house_noise_',num2str(i)),'.png'));  

end