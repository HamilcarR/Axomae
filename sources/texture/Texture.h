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

/* Total number of textures used for mesh rendering.*/
constexpr unsigned PBR_PIPELINE_TEX_NUM = 8;

#endif
