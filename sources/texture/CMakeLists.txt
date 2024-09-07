add_library(ax_texture STATIC
        Texture.cpp
        TextureGroup.cpp
        texture_utils.cpp
        GenericTextureProcessing.cpp
        GenericTexture.cpp
        DiffuseTexture.cpp
        NormalTexture.cpp
        MetallicTexture.cpp
        RoughnessTexture.cpp
        AmbiantOcclusionTexture.cpp
        SpecularTexture.cpp
        EmissiveTexture.cpp
        OpacityTexture.cpp
        Generic2DTexture.cpp
        CubemapTexture.cpp
        GenericCubemapTexture.cpp
        IrradianceTexture.cpp
        EnvironmentMap2DTexture.cpp
        FrameBufferTexture.cpp
        BRDFLookupTexture.cpp
)

target_link_libraries(ax_texture
PRIVATE

    ax_core
    ax_shader
    ax_opengl
    ax_database
    ax_controller
)
target_include_directories(ax_texture
PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)