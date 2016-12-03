# maptomic

From a simple image : 

![image](https://cloud.githubusercontent.com/assets/18567118/20143822/c50a4422-a69a-11e6-9eef-1a023102e804.jpg)

Height map computation is done,although it needs some customizable parameters...will be added later since.
We compute the edges of the image, using sobel or prewitt,then invert the colors:

![edge detection](https://cloud.githubusercontent.com/assets/18567118/20143397/2660f704-a699-11e6-9165-8a8569a420d6.jpg)

Normal map computation is done. It needs to compute the height map first,and use it to generate the normal map.


![normal map](https://cloud.githubusercontent.com/assets/18567118/20143711/5bcad760-a69a-11e6-8239-112496ff6c19.jpg)



DUDV map computation is done.Use a normal map first,to generate dudvs.
![dudv map](https://cloud.githubusercontent.com/assets/18567118/20150815/0d44dbd6-a6b8-11e6-9974-9553c61324ff.jpg)


There will soon be an OpenGL window viewer to display objects mapped with textures in real time and a usable executable. 
