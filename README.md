# Axomae
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg) 
![Build](https://github.com/HamilcarR/Axomae/actions/workflows/cmake-single-platform.yml/badge.svg)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/HamilcarR/Axomae)

# Table Of Contents:
* [Introduction](#Introduction)
* [Features](#Features)
* [Screenshots](#Screenshots)
* [Requirements](#Requirements)
* [Installation](#Installation)
* [Documentation](#Documentation)


## Introduction

Axomae is a 3D rendering engine and raytracer, designed as a foundation to explore advanced rendering techniques .    
The goal of the software is to facilitate the implementation of rendering algorithms and display of photorealistic images. 
At the same time, the application is intended to be interactive , and customizable. 

## Features

* Nova: A multi-thread/GPU parallelized path tracing engine. Nova can be used as an offline renderer, or a pseudo-realtime , more interactive viewer. 
* PBR rasterizer 3D viewer. 
* UV viewer.
* (Old feature) A normal map generation tool : Will probably be scrapped for an entirely new material generation pipeline. 
* .hdr texture viewer.
* Irradiance baking.

## Screenshots

Here are some examples of what Axomae can currently render:

### Nova Engine

<p align="center">
  <img src="Documentation/Screenshots/chessboard3.png" width="45%" alt="4K Chess Scene" />
  <img src="Documentation/Screenshots/bathroom.png" width="45%" alt="Bathroom Scene" />
</p>

*High-quality path-traced scenes: Chess scene at 4K resolution and realistic bathroom interior*

### PBR viewers

<p align="center">
  <img src="Documentation/Screenshots/viewer_boombox.png" width="30%" alt="Viewer Boombox" />
  <img src="Documentation/Screenshots/viewer_helmet.png" width="30%" alt="Viewer Helmet" />
  <img src="Documentation/Screenshots/viewer_mic_gxl.png" width="30%" alt="Viewer Microphone" />
</p>

<p align="center">
  <img src="Documentation/Screenshots/viewer_ship.png" width="30%" alt="Viewer Ship" />
  <img src="Documentation/Screenshots/viewer_spheres.png" width="30%" alt="Viewer Spheres" />
  <img src="Documentation/Screenshots/viewer_steampunk.png" width="30%" alt="Viewer Steampunk" />
</p>

*Real-time PBR renderer showcasing various 3D models with different materials and lighting*

## Requirements

* GCC 
* Qt6(Axomae can also build Qt from source , see [Installation](#Installation) for details)
* OpenGL
* Glew
* Cuda 12.x + Optix > 8.0.x (Optional)
* Any Linux distribution

## Installation

### First clone the repository : 

```
$ mkdir Axomae-git
$ cd Axomae-git
$ git clone https://github.com/HamilcarR/Axomae

```

### Download the dependencies : 

```
$ cd Axomae
$ ./scripts/update_deps.sh

```

### Build the project : 

Axomae doesn't support in-source builds, so I suggest building in the parent folder: 
Note that: 

- Building Unit tests is enabled by default , set ```-DAXOMAE_BUILD_TESTS=OFF``` if they are not needed. 
- Building QT from source is enabled by default. It is more reliable to build Axomae and it's dependencies through a unique toolchain, but it will take longer(~15 min on an 8 cores ... Build time optimizations are on the backlog. )
- If nonetheless you can use your own QT system library , use ```-DAXOMAE_FROMSOURCE_QT_BUILD=OFF``` .
- For CUDA , use ```-DAXOMAE_USE_CUDA=ON``` and ```-DAXOMAE_USE_OPTIX``` for ray-tracing gpu acceleration (Currently uses RTX hardware for acceleration, a native solution using only CUDA is on the backlog for non RTX cards).
```
$ cd Axomae
$ cmake -S . -B ../build -DAXOMAE_BUILD_TESTS=OFF -DAXOMAE_FROMSOURCE_QT_BUILD=OFF
$ cd ../build
$ make
```

## Documentation
### Architecture and modules : 
![Modules](Documentation/Axomae_modules.png)

### Bibliography : 
- Multiple-Scattering Microfacet Model for Real-Time Image-based Lighting - Carmelo J.Fdez-Aguera
- Physically Based Rendering: From Theory To Implementation - Matt Pharr, Wenzel Jakob, and Greg Humphreys
- Fast Minimum Storage Ray/Triangle Intersection - Thomas Müller , Ben Trumbore
- Unbiased physically based rendering on the GPU - Dietger van Antwerpen
- Wide BVH Traversal with a Short Stack - K.Vaidyanathan, S.Woop, C.Benthin
- A Generalized Ray Formulation For Wave-Optics Rendering - Shlomi Steinberg, Ravi Ramamoorthi, Benedikt Bitterli, Eugene D'Eon, Ling-Qi Yan, Matt Pharr
- Realtime Ray Tracing on current CPU Architectures - Carsten Benthin
- ALGORITHM 659 Implementing Sobol’s Quasirandom Sequence Generator - Paul Bratley, Bennett L.Fox
- Stochastic Generation of (t, s) Sample Sequences - Andrew Helmer, Per Christensen, Andrew Kensler
- Megakernels Considered Harmful: Wavefront Path Tracing on GPUs - Samuli Laine, Tero Karras, Timo Aila
- Practical Hash-Based Owen Scrambling - Brent Burley
- Average Irregularity Representation Of A Rough Surface For Ray Reflection - T.S. Trowbridge , K.P. Reitz
- Sampling the GGX Distribution of Visible Normals - Eric Heitz
- Microfacet Models for Refraction through Rough Surfaces - B. Walter, S.R. Marschner, H. Li, K.E. Torrance
- Extending the Disney BRDF to a BSDF with integrated subsurface scattering - Brent Burley 
