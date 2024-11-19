#ifndef TEXTURE_H
#define TEXTURE_H
#include "AmbiantOcclusionTexture.h"
#include "BRDFLookupTexture.h"
#include "CubemapTexture.h"
#include "DiffuseTexture.h"
#include "EmissiveTexture.h"
#include "EnvironmentMap2DTexture.h"
#include "FrameBufferTexture.h"
#include "Generic2DTexture.h"
#include "GenericCubemapTexture.h"
#include "GenericTexture.h"
#include "IrradianceTexture.h"
#include "MetallicTexture.h"
#include "NormalTexture.h"
#include "OpacityTexture.h"
#include "RoughnessTexture.h"
#include "SpecularTexture.h"

/* Number of minimal textures needed for a render pass : albedo, normal map, metallic ,roughness, AO , and emissive */
constexpr unsigned PBR_PIPELINE_TEX_NUM = 6 ;

#endif
