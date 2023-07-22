#include "../includes/RenderPipeline.h"
#include "../includes/FrameBufferInterface.h"
#include "../includes/RenderCubeMap.h"
#include <QOffscreenSurface>


namespace camera_angles {
    const auto up = glm::lookAt(glm::vec3(0.f) , glm::vec3(0.f , 1.f , 0.f) , glm::vec3(0.f , 0.f , -1.f)); 
    const auto down = glm::lookAt(glm::vec3(0.f) , glm::vec3(0.f , -1.f , 0.f) , glm::vec3(0.f , 0.f , 1.f)); 
    const auto left = glm::lookAt(glm::vec3(0.f) , glm::vec3(-1.f , 0.f , 0.f) , glm::vec3(0.f , 1.f , 0.f)); 
    const auto right = glm::lookAt(glm::vec3(0.f) , glm::vec3(1.f , 0.f , 0.f) , glm::vec3(0.f , 1.f , 0.f)); 
    const auto back  = glm::lookAt(glm::vec3(0.f) , glm::vec3(0.f , 0.f , 1.f) , glm::vec3(0.f , 1.f , 0.f)); 
    const auto front  = glm::lookAt(glm::vec3(0.f) , glm::vec3(0.f , 0.f , -1.f) , glm::vec3(0.f , 1.f , 0.f));
}





RenderPipeline::RenderPipeline(Renderer* _renderer , ResourceDatabaseManager* _resource_database) {
    renderer = _renderer ; 
    resource_database = _resource_database; 
}

RenderPipeline::~RenderPipeline(){

}

void RenderPipeline::clean(){

}


/**
 * The function "bakeEnvmapToCubemap" takes an environment map HDR texture and converts it into a floating point cubemap.
 * 
 * @param hdri_map The `hdri_map` parameter is an `EnvironmentMapTexture` object, which represents the
 * high dynamic range image used for the environment map. It contains the texture data and other
 * properties of the image.
 * @param width The width of the cubemap texture to be rendered.
 * @param height The height parameter in the code represents the height of the cubemap texture that
 * will be generated. It determines the resolution of the cubemap texture.
 * @param gl_widget GLViewer object used to extract informations like default FBO id , current width/height of the rendering surface etc.
 * 
 * @return a pointer to a CubeMapMesh object.
 */
// "You have to clean this shit before things go boom"
CubeMapMesh* RenderPipeline::bakeEnvmapToCubemap( EnvironmentMapTexture *hdri_map , unsigned width , unsigned height , GLViewer* gl_widget){
    assert(resource_database != nullptr); 
    assert(resource_database->getTextureDatabase() != nullptr); 
    assert(renderer != nullptr);
    assert(hdri_map != nullptr); 
    TextureDatabase *texture_database = resource_database->getTextureDatabase();
    ShaderDatabase *shader_database = resource_database->getShaderDatabase();
    Shader* bake_shader = shader_database->get(Shader::ENVMAP_CUBEMAP_CONVERTER); 
    ScreenSize tex_dim , cam_dim ;
    /*Set up camera ratio + cubemap texture resolution*/
    cam_dim.width = width ; 
    cam_dim.height = height ; 
    tex_dim.width = width ; 
    tex_dim.height = height ;  
    FreePerspectiveCamera camera(90.f ,&cam_dim , 0.1f , 2000.f ) ; //Generic camera object 
    CubeMapMesh *cubemap = new CubeMapMesh();
    CubeMesh *cube = new CubeMesh(); 
    cube->setShader(bake_shader);
    cube->setSceneCameraPointer(&camera);
    std::pair<int , Texture*> query_envmap_result = texture_database->contains(hdri_map); 
    int database_id_envmap = query_envmap_result.first; 
    if(query_envmap_result.second == nullptr)
        database_id_envmap = texture_database->add(hdri_map , false);  
    cube->material.addTexture(database_id_envmap , Texture::ENVMAP);    
    Drawable cube_drawable(cube);  
    
    /*Set up view angles , each rotated 90° to get all faces */
    std::vector<glm::mat4> views = { camera_angles::left , 
                                    camera_angles::right , 
                                    camera_angles::down , 
                                    camera_angles::up , 
                                    camera_angles::back , 
                                    camera_angles::front};
   
    /* Generate a framebuffer that will render to a cubemap*/
    RenderCubeMap cubemap_renderer_framebuffer(texture_database , &cam_dim , renderer->getDefaultFrameBufferIdPointer()); 
    cubemap_renderer_framebuffer.initializeFrameBufferTexture();
    bake_shader->bind(); 
    cubemap_renderer_framebuffer.getFrameBufferTexturePointer()->setGlData(bake_shader);
    bake_shader->release(); 
    cubemap_renderer_framebuffer.initializeFrameBuffer(); 
    cube_drawable.startDraw();
    cubemap_renderer_framebuffer.bindFrameBuffer();
    glViewport(0 , 0 , width , height); 
    for(unsigned i = 0 ; i < 6 ; i++){    
        camera.setView(views[i]); 
        cube_drawable.bind(); 
        cubemap_renderer_framebuffer.renderToFace(i , GLFrameBuffer::COLOR0);  
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);   
        glDrawElements(GL_TRIANGLES , cube->geometry.indices.size() , GL_UNSIGNED_INT , 0 );
        cube_drawable.unbind();     
    }
   
   /*Irradiance convolution here*/
   
   
   
    glViewport(0 , 0 , gl_widget->width() , gl_widget->height());
    cubemap_renderer_framebuffer.unbindFrameBuffer();  
    cube_drawable.clean();
    
    //TODO: [AX-43] Fix memory leak in RenderPipeline
    
    std::pair<int , Texture*> query_baked_cubemap_texture = texture_database->contains(cubemap_renderer_framebuffer.getFrameBufferTexturePointer());
    assert(query_baked_cubemap_texture.second); 
    cubemap->material.addTexture(query_baked_cubemap_texture.first , Texture::CUBEMAP);
    cubemap->setShader(shader_database->get(Shader::CUBEMAP));
    return cubemap;  
}