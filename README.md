# Axomae

Axomae is a texture generation tool, to create normal maps , height maps , and distortion maps.
-We can "convert" diffuse textures to a normal map , using an edge detection:
![axomae](https://user-images.githubusercontent.com/18567118/32781705-4d34f8aa-c946-11e7-9b6a-851b5d6e4cea.png)
-Or we can project a 3D mesh normals on it's UVs , hence baking the normals : 
![axomae](https://user-images.githubusercontent.com/18567118/47607072-add21a80-da1b-11e8-8a4d-5e14f9c9133a.png)
![axomae](https://user-images.githubusercontent.com/18567118/47607074-b32f6500-da1b-11e8-9ca0-90e7b64464eb.png)
![axomae](https://user-images.githubusercontent.com/18567118/47607073-b0cd0b00-da1b-11e8-823a-9839e13ac352.png)



The software is still in developpement. I plan integrating : 

  -Full GPGPU support : the software can already use CUDA (not tested for GPUs other than compute capability 61), but needs some               work.
  
  -Mesh normals projection into texture : using a high poly mesh to create a normal map texture.
  
  -Batch processing : either processing textures or creating normal maps from numerous meshes.
  
  -An OpenGL renderer : A 3D renderer using PBR materials and reading GLTF , FBX , OBJ formats.
  
  -UV editor .
  
  -real time texture modifications and material modifications.
  
