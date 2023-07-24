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


const std::vector<glm::mat4> views = { camera_angles::left , 
                                    camera_angles::right , 
                                    camera_angles::down , 
                                    camera_angles::up , 
                                    camera_angles::back , 
                                    camera_angles::front};



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
//TODO: [AX-43] Fix memory leak in RenderPipeline
CubeMapMesh* RenderPipeline::bakeEnvmapToCubemap( EnvironmentMapTexture *hdri_map , unsigned width , unsigned height , GLViewer* gl_widget){
    assert(resource_database != nullptr); 
    assert(resource_database->getTextureDatabase() != nullptr); 
    assert(renderer != nullptr);
    assert(hdri_map != nullptr); 
    TextureDatabase *texture_database = resource_database->getTextureDatabase();
    ShaderDatabase *shader_database = resource_database->getShaderDatabase();
    Shader* bake_shader = shader_database->get(Shader::ENVMAP_CUBEMAP_CONVERTER); 
    ScreenSize tex_dim , cam_dim;
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
    /* Generate a framebuffer that will render to a cubemap*/
    RenderCubeMap cubemap_renderer_framebuffer(texture_database , &cam_dim , renderer->getDefaultFrameBufferIdPointer());
    cubemap_renderer_framebuffer.initializeFrameBufferTexture(GLFrameBuffer::COLOR0 , false , Texture::RGB16F , Texture::RGB , Texture::FLOAT , width , height , Texture::CUBEMAP) ; 
    errorCheck(__FILE__ , __LINE__); 
    bake_shader->bind(); 
    cubemap_renderer_framebuffer.getFrameBufferTexturePointer(GLFrameBuffer::COLOR0)->setGlData(bake_shader); 
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
    cubemap_renderer_framebuffer.unbindFrameBuffer(); 
    glViewport(0 , 0 , gl_widget->width() , gl_widget->height()); 
    cubemap_renderer_framebuffer.clean() ;  
    cube_drawable.clean();
    std::pair<int , Texture*> query_baked_cubemap_texture = texture_database->contains(cubemap_renderer_framebuffer.getFrameBufferTexturePointer(GLFrameBuffer::COLOR0)); 
    cubemap->material.addTexture(query_baked_cubemap_texture.first , Texture::CUBEMAP); 
    cubemap->setShader(shader_database->get(Shader::CUBEMAP));
    errorCheck(__FILE__ , __LINE__);
    bakeIrradianceCubemap(query_baked_cubemap_texture.first , 64 , 64 , gl_widget); 
    return cubemap;  
}

int RenderPipeline::bakeIrradianceCubemap(int cube_envmap , unsigned width , unsigned height , GLViewer* gl_widget){
    ScreenSize irrad_dim , cam_dim ;
    TextureDatabase *texture_database = resource_database->getTextureDatabase();
    ShaderDatabase *shader_database = resource_database->getShaderDatabase();
    Shader* irradiance_shader = shader_database->get(Shader::IRRADIANCE_CUBEMAP_COMPUTE); 
    RenderCubeMap cubemap_irradiance_framebuffer(texture_database , &irrad_dim , renderer->getDefaultFrameBufferIdPointer());   
    FreePerspectiveCamera camera(90.f ,&cam_dim , 0.1f , 2000.f ) ;
    CubeMesh *cube = new CubeMesh(); 
    
    cam_dim.width = width ; 
    cam_dim.height = height ;  
    irrad_dim.width = width ; 
    irrad_dim.height = height ; 
    cubemap_irradiance_framebuffer.initializeFrameBufferTexture(GLFrameBuffer::COLOR0 , false , Texture::RGB16F , Texture::RGB , Texture::FLOAT , irrad_dim.width , irrad_dim.height , Texture::IRRADIANCE);
    cube->setShader(irradiance_shader); 
    cube->material.addTexture(cube_envmap , Texture::CUBEMAP);
    cube->setSceneCameraPointer(&camera); 
    Drawable cube_drawable(cube); 
    
    irradiance_shader->bind(); 
    cubemap_irradiance_framebuffer.getFrameBufferTexturePointer(GLFrameBuffer::COLOR0)->setGlData(irradiance_shader);
    irradiance_shader->release();
    cubemap_irradiance_framebuffer.initializeFrameBuffer();
    cubemap_irradiance_framebuffer.bindFrameBuffer();
    cube_drawable.startDraw();  
    glViewport(0 , 0 , irrad_dim.width , irrad_dim.height);  
    for(unsigned i = 0 ; i < 6 ; i++){    
        camera.setView(views[i]); 
        cube_drawable.bind();  
        cubemap_irradiance_framebuffer.renderToFace(i , GLFrameBuffer::COLOR0);         
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);    
        glDrawElements(GL_TRIANGLES , cube->geometry.indices.size() , GL_UNSIGNED_INT , 0 );    
    }
    glFinish(); 
    cube_drawable.unbind();     
    cubemap_irradiance_framebuffer.unbindFrameBuffer(); 
    glViewport(0 , 0 , gl_widget->width() , gl_widget->height()); 
    cubemap_irradiance_framebuffer.clean();
    cube_drawable.clean(); 
    std::pair<int , Texture*> query_baked_irradiance_texture = texture_database->contains(cubemap_irradiance_framebuffer.getFrameBufferTexturePointer(GLFrameBuffer::COLOR0));  
    assert(query_baked_irradiance_texture.second);
    return query_baked_irradiance_texture.first ;
}