
#include "nova_engine.h"
using namespace nova::engine;

void EngineResourcesHolder::setTilesWidth(int width) { tiles_w = width; }

void EngineResourcesHolder::setTilesHeight(int height) { tiles_h = height; }

void EngineResourcesHolder::setSampleIncrement(int increment) { sample_increment = increment; }

void EngineResourcesHolder::setAliasingSamples(int samples) { aliasing_samples = samples; }

void EngineResourcesHolder::setMaxSamples(int samples) { renderer_max_samples = samples; }

void EngineResourcesHolder::setMaxDepth(int depth) { max_depth = depth; }

void EngineResourcesHolder::setCancelPtr(bool *cancel_ptr) { cancel_render = cancel_ptr; }

int EngineResourcesHolder::getTilesWidth() const { return tiles_w; }

int EngineResourcesHolder::getTilesHeight() const { return tiles_h; }

int EngineResourcesHolder::getSampleIncrement() const { return sample_increment; }

int EngineResourcesHolder::getAliasingSamples() const { return aliasing_samples; }

int EngineResourcesHolder::getMaxSamples() const { return renderer_max_samples; }

int EngineResourcesHolder::getMaxDepth() const { return max_depth; }

bool *EngineResourcesHolder::getCancelPtr() const { return cancel_render; }
