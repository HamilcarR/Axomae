#ifndef NOVA_TEXTURING_H
#define NOVA_TEXTURING_H
#include "NovaTextureInterface.h"
#include "internal/common/math/math_utils.h"
#include "internal/common/type_list.h"
#include "internal/macro/project_macros.h"
#include "internal/memory/MemoryArena.h"
#include <memory>
namespace nova::texturing {

  struct TextureResourcesHolder {
    std::vector<NovaTextureInterface> textures;

    template<class T, class... Args>
    NovaTextureInterface add_texture(T *allocation_buffer, std::size_t offset, Args &&...args) {
      static_assert(core::has<T, TYPELIST>::has_type, "Provided type is not a Texture type.");
      T *allocated_ptr = core::memory::MemoryArena<>::construct<T>(&allocation_buffer[offset], std::forward<Args>(args)...);
      textures.push_back(allocated_ptr);
      return textures.back();
    }

    std::vector<NovaTextureInterface> &get_textures() { return textures; }
    ax_no_discard const std::vector<NovaTextureInterface> &get_textures() const { return textures; }
    void clear() { textures.clear(); }
  };

  glm::vec3 sample_cubemap(const glm::vec3 &sample_vector, const TextureRawData *res_holder);
  glm::vec3 sample_cubemap_plane(const glm::vec3 &sample_vector, const glm::vec3 &up_vector, const TextureRawData *res_holder);

}  // namespace nova::texturing
#endif  // NOVA_TEXTURING_H
