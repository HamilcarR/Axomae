#ifndef OFFLINECUBEMAPPROCESSING_H
#define OFFLINECUBEMAPPROCESSING_H
#include "Texture.h"
#include "../kernels/CubemapProcessing.cuh"
#include <sstream>

template <class T> 
class EnvmapProcessing{
public:

    /**
     * @brief Construct a texture from an envmap double HDR with 3 channels 
     * 
     * @param _data 
     * @param _width 
     * @param _height 
     */
    EnvmapProcessing(const T *_data , const unsigned _width , const unsigned _height , const unsigned int num_channels = 3){
        width = _width ; 
        height = _height ;
        channels = num_channels ; 
        for(unsigned i = 0 ; i < width * height * num_channels ; i+=num_channels){
            data.push_back(glm::vec3(_data[i] , _data[i+1] , _data[i+2])); 
        } 
    }
    /**
     * @brief Destroy the Envmap Processing object
     * 
     */
    virtual ~EnvmapProcessing(){}

    const std::vector<glm::vec3>& getData(){return data;}

    
    /**
     * @brief 
     * 
     * @param roughness 
     * @return TextureData 
     */
    TextureData computeSpecularIrradiance(double roughness);

/* Implementation of templated methods*/
public:

    /**
     * @brief Get the normalized UV in [0 , 1] range , from [0 ,width/height] coordinates.
     * @tparam D Type of coordinates .
     * @param x Width coordinates in the range [0 , width[.
     * @param y Height coordinates in the range [0 , height[.
     * @return const glm::dvec2 UVs in the range [0 , 1].
     */
    template<class D>
    inline const glm::dvec2 getUvFromPixelCoords(const D x , const D y) const {
        return glm::dvec2(static_cast<double>(x) / static_cast<double>(width - 1) , static_cast<double>(y) / static_cast<double>(height - 1)) ; 
    } 

    /**
     * @brief Get the Pixel Coords From Uv object
     * 
     * @tparam D 
     * @param u 
     * @param v 
     * @return const glm::dvec2 
     */
    template<class D>
    inline const glm::dvec2 getPixelCoordsFromUv(const D u , const D v) const {
            return glm::dvec2(u * (static_cast<double>(width) - 1) , v * (static_cast<double>(height) - 1)); 
    }


    /**
     * @brief This method wrap around if the texture coordinates provided land beyond the texture dimensions, repeating the texture values on both axes. 
     * 
     * @tparam D Data type of the coordinates.
     * @param u Horizontal UV coordinates.
     * @param v Vertical UV coordinates.
     * @return const glm::dvec2 Normalized coordinates in the [0 , 1] range.
     */
    template<class D>
    inline const glm::dvec2 wrapAroundTexCoords(const D u , const D v) const {
        int u_integer = 0 , v_integer = 0 ; 
        D u_double_p = 0. , v_double_p = 0. ; 
        u_integer = std::floor(u); 
        v_integer = std::floor(v); 
        if(u > 1.)
            u_double_p = u - u_integer ; 
        else if(u < 0.)
            u_double_p = u - u_integer ; 
        else
            u_double_p = u ; 
        if (v > 1.)
            v_double_p = v - v_integer ; 
        else if (v < 0.)
            v_double_p = v - v_integer ; 
        else
            v_double_p = v ;
        return glm::dvec2(u_double_p , v_double_p); 
}       

    /**
     * @brief Normalizes a set of pixel coordinates into texture bounds. 
     *       
     * @param x Horizontal coordinates. 
     * @param y Vertical coordinates.
     * @return const glm::dvec2 Normalized coordinates 
     */
    inline const glm::dvec2 wrapAroundPixelCoords(const int x , const int y) const {
        unsigned int x_coord = 0 , y_coord = 0 ;
        int _width = static_cast<int>(width); 
        int _height = static_cast<int>(height);  
        if(x >= _width)
            x_coord = x % _width ; 
        else if(x < 0)
            x_coord = _width + (x % _width) ; 
        else
            x_coord = x ; 
        if(y >= _height)
            y_coord = y % _height ; 
        else if(y < 0)
            y_coord = _height + (y % _height) ; 
        else
            y_coord = y ;
        return glm::dvec2(x_coord , y_coord) ; 
    }

   /**
    * @brief 
    * 
    * @param top_left 
    * @param top_right 
    * @param bottom_left 
    * @param bottom_right 
    * @param point 
    * @return * const T 
    */
    const glm::dvec3 bilinearInterpolate(const glm::dvec2 top_left , const glm::dvec2 top_right , const glm::dvec2 bottom_left , const glm::dvec2 bottom_right , const glm::dvec2 point) const {
        const double u = (point.x - top_left.x) / (top_right.x - top_left.x); 
        const double v = (point.y - top_left.y) / (bottom_left.y - top_left.y);
        const glm::dvec3 top_interp = (1 - u) * discreteSample(top_left.x , top_left.y) + u * discreteSample(top_right.x , top_right.y);
        const glm::dvec3 bot_interp = (1 - u) * discreteSample(bottom_left.x , bottom_left.y) + u * discreteSample(bottom_right.x , bottom_right.y) ;  
        return (1 - v) * top_interp + v * bot_interp ; 
    }


    /**
     * @brief 
     * 
     * @param x 
     * @param y 
     * @return const T 
     */
    inline const glm::dvec3 discreteSample(int x , int y) const{
        const glm::dvec2 normalized = wrapAroundPixelCoords(static_cast<int>(x) , static_cast<int>(y));
        const glm::dvec3 texel_value = data[normalized.x * height + normalized.y] ;
        return texel_value ; 
    }

    template<class D>
    inline bool isPixel(const D x , const D y) const {
        return std::floor(x) == x && std::floor(y) == y ;  
    }

    /**
     * @brief This method samples a value from the equirectangular envmap 
     *! Note : In case the coordinates go beyond the bounds of the texture , we wrap around .
     *! In addition , sampling texels may return a bilinear interpolated value when u,v are converted to a (x ,y) non integer texture coordinate.  
     * @tparam D Type of the coordinates 
     * @param u Horizontal uv coordinates
     * @param v Vertical uv coordinates
     * @return T Returns a texel value of type T 
     */
    template<class D> 
    inline const glm::dvec3 uvSample(const D u , const D v) const {
        const glm::dvec2 wrap_uv = wrapAroundTexCoords(u , v); 
        const glm::dvec2 pixel_coords = getPixelCoordsFromUv(wrap_uv.x , wrap_uv.y);
        if(isPixel(pixel_coords.x , pixel_coords.y)) 
            return discreteSample(pixel_coords.x , pixel_coords.y); 
        const glm::dvec2 top_left(std::floor(pixel_coords.x) , std::floor(pixel_coords.y));
        const glm::dvec2 top_right(std::floor(pixel_coords.x) + 1 , std::floor(pixel_coords.y));
        const glm::dvec2 bottom_left(std::floor(pixel_coords.x)  , std::floor(pixel_coords.y) + 1);
        const glm::dvec2 bottom_right(std::floor(pixel_coords.x) + 1 , std::floor(pixel_coords.y) + 1);
        const glm::dvec3 texel_value = bilinearInterpolate(top_left , top_right , bottom_left , bottom_right , pixel_coords); 
        return texel_value ; 
    } 

    /**
     * @brief Bake an equirect envmap to an irradiance map
     * @param delta Size of the step
     * @return std::unique_ptr<TextureData> Texture data containing width , height , and double f_data about the newly created map.
     */
    std::unique_ptr<TextureData> computeDiffuseIrradiance(const T delta) const {
        TextureData envmap_tex_data ; 
        envmap_tex_data.data_format = Texture::RGB ; 
        envmap_tex_data.internal_format = Texture::RGB32F ; 
        envmap_tex_data.data_type = Texture::FLOAT;
        envmap_tex_data.width = width ;
        envmap_tex_data.height = height ;  
        envmap_tex_data.mipmaps = 0;
        envmap_tex_data.f_data = new float[width * height * channels];
        unsigned index = 0 ;  
        for(unsigned i = 0 ; i < width ; i++){
                for(unsigned j = 0 ; j < height ; j++){
                        const glm::dvec2 uv = getUvFromPixelCoords(i , j); 
                        const glm::dvec2 sph = gpgpu_math::uvToSpherical(uv.x , uv.y); 
                        const glm::dvec3 cart = gpgpu_math::sphericalToCartesian(sph.x , sph.y);
                        //const glm::dvec3 irrad = computeHemisphereIrradiance(cart.x , cart.y , cart.z  , delta);
                   //     glm::dvec2 uvt = gpgpu_math::sphericalToUv(gpgpu_math::cartesianToSpherical(cart)); 
                        glm::dvec2 uvt = gpgpu_math::sphericalToUv(gpgpu_math::uvToSpherical(uv)); 
                        const glm::dvec3 irrad = uvSample(uvt.x , uvt.y); 
                        std::stringstream test ;
                        test << "i : " << i << " : j : " << j << "  UV:(" << uv.x << "," << uv.y <<")" << "  UVT:(" << uvt.x << "," << uvt.y << ")\n" ; 
                        std::string test_str = test.str() ;  
                        assert(irrad != glm::dvec3(0.) && test_str.c_str()); 
                        envmap_tex_data.f_data[index++] = irrad.x ; 
                        envmap_tex_data.f_data[index++] = irrad.y ; 
                        envmap_tex_data.f_data[index++] = irrad.z ; 
                }
        }

        return std::make_unique<TextureData>(envmap_tex_data);
    } 

    /**
     * @brief Returns the irradiance value at a specific position of the texture , in accordance to it's computed hemisphere. 
     * 
     * @tparam D Data type of the coordinates 
     * @param x Cartesian X
     * @param y Cartesian Y 
     * @param z Cartesian Z
     * @param step Value of the increment
     * @return const glm::dvec3 Irradiance value texel 
     *
    template<class D>
    inline const glm::dvec3 computeHemisphereIrradiance(const D x , const D y , const D z , const T step) const { //! use importance sampling ?
        unsigned int samples = 0 ; 
        glm::dvec3 irradiance = glm::dvec3(0.f); 
       // const glm::dvec2 sph = gpgpu_math::cartesianToSpherical(x , y , z);
      //  const glm::dvec2 uv = gpgpu_math::sphericalToUv(sph.x , sph.y); 
        glm::dvec3 normal = glm::dvec3(x , y , z);
        glm::dvec3 tangent = glm::normalize(glm::cross(UP_dvecTOR , normal));
        glm::dvec3 bitangent = glm::normalize(glm::cross(normal , tangent));
        if(1 - glm::abs(dot(UP_dvecTOR , normal)) <= 1e-6)
            tangent = glm::dvec3(0.f , 1.f , 0.f); 
        for(double phi = 0.f ; phi < 2*PI ; phi += step)
                for(double theta = 0.f ; theta < PI/2.f ; theta += step){
                        glm::dvec3 uv_cart = gpgpu_math::sphericalToCartesian(phi , theta);
                        uv_cart = uv_cart.x * tangent + uv_cart.y * bitangent + uv_cart.z * normal ; 
                        auto spherical = gpgpu_math::cartesianToSpherical(uv_cart.x , uv_cart.y , uv_cart.z);
                        glm::dvec2 uv = gpgpu_math::sphericalToUv(spherical.x , spherical.y); 
                        irradiance += uvSample(uv.x , uv.y)  ; 
                        samples ++ ; 
                }
        return irradiance  / static_cast<double>(samples) ; 
    } 
    */

    static inline glm::dvec3 pgc3d(unsigned x , unsigned y , unsigned z) {
        x = x * 1664525u + 1013904223u; 
        y = y * 1664525u + 1013904223u;
        z = z * 1664525u + 1013904223u;

        x += y*z; 
        y += z*x; 
        z += x*y;
        
        x ^= x >> 16u;
        y ^= y >> 16u;
        z ^= z >> 16u;
        
        x += y*z; 
        y += z*x; 
        z += x*y;

        return glm::dvec3(x , y , z);
    }
    static inline double radicalInverse(unsigned bits) {
        bits = (bits << 16u) | (bits >> 16u);
        bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
        bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
        bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
        bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
        return double(bits) * 2.3283064365386963e-10; 
    }

    static inline glm::dvec3 hammersley3D(unsigned  i , unsigned N){
        return glm::dvec3(double(i) / double(N) , radicalInverse(i) , radicalInverse(i ^ 0xAAAAAAAAu)); 
    }

    template<class D>
    inline const glm::dvec3 computeHemisphereIrradiance(const D x , const D y , const D z , const T step) const { //! use importance sampling ?
       /* unsigned int samples = 0 ; 
        glm::dvec2 uv =gpgpu_math::sphericalToUv(gpgpu_math::cartesianToSpherical(x , y , z));
        glm::dvec2 pix = getPixelCoordsFromUv(uv.x , uv.y); 
        glm::dvec3 irradiance = glm::dvec3(0.f); 
        glm::dvec3 normal = glm::dvec3(x , y , z);
        glm::dvec3 tangent = glm::normalize(glm::cross(UP_dvecTOR , normal));
        glm::dvec3 bitangent = glm::normalize(glm::cross(normal , tangent));
        if(1 - glm::abs(dot(UP_dvecTOR , normal)) <= 1e-6)
            tangent = glm::dvec3(0.f , 0.f , 1.f); 
        unsigned N = 100; 
        for(samples = 0.f ; samples <= N ; samples ++){ 
            glm::dvec3 random = hammersley3D(samples , N);
            double phi = 2 * PI * random.x ; 
            double theta = asin(sqrt(random.y));
            glm::dvec3 uv_cart = gpgpu_math::sphericalToCartesian(phi , theta) ;
            uv_cart = uv_cart.x * tangent + uv_cart.y * bitangent + uv_cart.z * normal ; 
            auto spherical = gpgpu_math::cartesianToSpherical(uv_cart.x , uv_cart.y , uv_cart.z);
            glm::dvec2 uv = gpgpu_math::sphericalToUv(spherical.x , spherical.y); 
            irradiance += uvSample(uv.x , uv.y)  ; 
        }
        return irradiance  / static_cast<double>(N) ; */

        glm::dvec2 uv =gpgpu_math::sphericalToUv(gpgpu_math::cartesianToSpherical(x , y , z));
        return uvSample(uv.x , uv.y);  
    } 



protected:
    std::vector<glm::vec3> data ;
    unsigned width ; 
    unsigned height ;
    unsigned channels ; 
};



#endif