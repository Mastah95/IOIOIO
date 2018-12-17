close all;

I = imread('cameraman_in.png');
I = rgb2gray(I) ;
edges_can = edge(I, 'Canny');
figure(1)
imshow(edges_can)
imwrite(edges_can, 'cameraman_canny.png')

edges_prew = edge(I, 'Prewitt');
figure(2)
imshow(edges_prew)
imwrite(edges_prew, 'cameraman_prewitt.png')

edges_sob = edge(I, 'Sobel');
figure(3)
imshow(edges_sob)
imwrite(edges_sob, 'cameraman_sobel.png')

edges_log = edge(I, 'log');
figure(4)
imshow(edges_log)
imwrite(edges_log, 'cameraman_log.png')
edges_rob = edge(I, 'Roberts');
figure(5)
imshow(edges_rob)
imwrite(edges_rob, 'cameraman_rob.png')
