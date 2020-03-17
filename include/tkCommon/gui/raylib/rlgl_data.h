#pragma once
#define RAYMATH_STANDALONE 
#define RAYMATH_HEADER_ONLY
#include "raymath.h"

namespace rl {
 #ifndef __cplusplus
    // Boolean type
    typedef enum { false, true } bool;
#endif

    // Color type, RGBA (32bit)
    typedef struct Color {
        unsigned char r;
        unsigned char g;
        unsigned char b;
        unsigned char a;
    } Color;

    // Rectangle type
    typedef struct Rectangle {
        float x;
        float y;
        float width;
        float height;
    } Rectangle;

    // Texture2D type
    // NOTE: Data stored in GPU memory
    typedef struct Texture2D {
        unsigned int id;        // OpenGL texture id
        int width;              // Texture base width
        int height;             // Texture base height
        int mipmaps;            // Mipmap levels, 1 by default
        int format;             // Data format (PixelFormat)
    } Texture2D;

    // Texture type, same as Texture2D
    typedef Texture2D Texture;

    // TextureCubemap type, actually, same as Texture2D
    typedef Texture2D TextureCubemap;

    // RenderTexture2D type, for texture rendering
    typedef struct RenderTexture2D {
        unsigned int id;        // OpenGL framebuffer (fbo) id
        Texture2D texture;      // Color buffer attachment texture
        Texture2D depth;        // Depth buffer attachment texture
        bool depthTexture;      // Track if depth attachment is a texture or renderbuffer
    } RenderTexture2D;

    // RenderTexture type, same as RenderTexture2D
    typedef RenderTexture2D RenderTexture;

    // Vertex data definning a mesh
    typedef struct Mesh {
        int vertexCount;        // number of vertices stored in arrays
        int triangleCount;      // number of triangles stored (indexed or not)
        float *vertices;        // vertex position (XYZ - 3 components per vertex) (shader-location = 0)
        float *texcoords;       // vertex texture coordinates (UV - 2 components per vertex) (shader-location = 1)
        float *texcoords2;      // vertex second texture coordinates (useful for lightmaps) (shader-location = 5)
        float *normals;         // vertex normals (XYZ - 3 components per vertex) (shader-location = 2)
        float *tangents;        // vertex tangents (XYZW - 4 components per vertex) (shader-location = 4)
        unsigned char *colors;  // vertex colors (RGBA - 4 components per vertex) (shader-location = 3)
        unsigned short *indices;// vertex indices (in case vertex data comes indexed)

        // Animation vertex data
        float *animVertices;    // Animated vertex positions (after bones transformations)
        float *animNormals;     // Animated normals (after bones transformations)
        int *boneIds;           // Vertex bone ids, up to 4 bones influence by vertex (skinning)
        float *boneWeights;     // Vertex bone weight, up to 4 bones influence by vertex (skinning)

        // OpenGL identifiers
        unsigned int vaoId;     // OpenGL Vertex Array Object id
        unsigned int *vboId;    // OpenGL Vertex Buffer Objects id (7 types of vertex data)
    } Mesh;

    // Shader and material limits
    #define MAX_SHADER_LOCATIONS    32
    #define MAX_MATERIAL_MAPS       12

    // Shader type (generic)
    typedef struct Shader {
        unsigned int id;        // Shader program id
        int *locs;              // Shader locations array (MAX_SHADER_LOCATIONS)
    } Shader;

    // Material texture map
    typedef struct MaterialMap {
        Texture2D texture;      // Material map texture
        Color color;            // Material map color
        float value;            // Material map value
    } MaterialMap;

    // Material type (generic)
    typedef struct Material {
        Shader shader;          // Material shader
        MaterialMap *maps;      // Material maps (MAX_MATERIAL_MAPS)
        float *params;          // Material generic parameters (if required)
    } Material;

    // Camera type, defines a camera position/orientation in 3d space
    typedef struct Camera {
        Vector3 position;       // Camera position
        Vector3 target;         // Camera target it looks-at
        Vector3 up;             // Camera up vector (rotation over its axis)
        float fovy;             // Camera field-of-view apperture in Y (degrees)
    } Camera;

    // Head-Mounted-Display device parameters
    typedef struct VrDeviceInfo {
        int hResolution;                // HMD horizontal resolution in pixels
        int vResolution;                // HMD vertical resolution in pixels
        float hScreenSize;              // HMD horizontal size in meters
        float vScreenSize;              // HMD vertical size in meters
        float vScreenCenter;            // HMD screen center in meters
        float eyeToScreenDistance;      // HMD distance between eye and display in meters
        float lensSeparationDistance;   // HMD lens separation distance in meters
        float interpupillaryDistance;   // HMD IPD (distance between pupils) in meters
        float lensDistortionValues[4];  // HMD lens distortion constant parameters
        float chromaAbCorrection[4];    // HMD chromatic aberration correction parameters
    } VrDeviceInfo;

    // VR Stereo rendering configuration for simulator
    typedef struct VrStereoConfig {
        Shader distortionShader;        // VR stereo rendering distortion shader
        Matrix eyesProjection[2];       // VR stereo rendering eyes projection matrices
        Matrix eyesViewOffset[2];       // VR stereo rendering eyes view offset matrices
        int eyeViewportRight[4];        // VR stereo rendering right eye viewport [x, y, w, h]
        int eyeViewportLeft[4];         // VR stereo rendering left eye viewport [x, y, w, h]
    } VrStereoConfig;


    // TraceLog message types
    typedef enum {
        LOG_ALL,
        LOG_TRACE,
        LOG_DEBUG,
        LOG_INFO,
        LOG_WARNING,
        LOG_ERROR,
        LOG_FATAL,
        LOG_NONE
    } TraceLogType;

    // Texture formats (support depends on OpenGL version)
    typedef enum {
        UNCOMPRESSED_GRAYSCALE = 1,     // 8 bit per pixel (no alpha)
        UNCOMPRESSED_GRAY_ALPHA,
        UNCOMPRESSED_R5G6B5,            // 16 bpp
        UNCOMPRESSED_R8G8B8,            // 24 bpp
        UNCOMPRESSED_R5G5B5A1,          // 16 bpp (1 bit alpha)
        UNCOMPRESSED_R4G4B4A4,          // 16 bpp (4 bit alpha)
        UNCOMPRESSED_R8G8B8A8,          // 32 bpp
        UNCOMPRESSED_R32,               // 32 bpp (1 channel - float)
        UNCOMPRESSED_R32G32B32,         // 32*3 bpp (3 channels - float)
        UNCOMPRESSED_R32G32B32A32,      // 32*4 bpp (4 channels - float)
        COMPRESSED_DXT1_RGB,            // 4 bpp (no alpha)
        COMPRESSED_DXT1_RGBA,           // 4 bpp (1 bit alpha)
        COMPRESSED_DXT3_RGBA,           // 8 bpp
        COMPRESSED_DXT5_RGBA,           // 8 bpp
        COMPRESSED_ETC1_RGB,            // 4 bpp
        COMPRESSED_ETC2_RGB,            // 4 bpp
        COMPRESSED_ETC2_EAC_RGBA,       // 8 bpp
        COMPRESSED_PVRT_RGB,            // 4 bpp
        COMPRESSED_PVRT_RGBA,           // 4 bpp
        COMPRESSED_ASTC_4x4_RGBA,       // 8 bpp
        COMPRESSED_ASTC_8x8_RGBA        // 2 bpp
    } PixelFormat;

    // Texture parameters: filter mode
    // NOTE 1: Filtering considers mipmaps if available in the texture
    // NOTE 2: Filter is accordingly set for minification and magnification
    typedef enum {
        FILTER_POINT = 0,               // No filter, just pixel aproximation
        FILTER_BILINEAR,                // Linear filtering
        FILTER_TRILINEAR,               // Trilinear filtering (linear with mipmaps)
        FILTER_ANISOTROPIC_4X,          // Anisotropic filtering 4x
        FILTER_ANISOTROPIC_8X,          // Anisotropic filtering 8x
        FILTER_ANISOTROPIC_16X,         // Anisotropic filtering 16x
    } TextureFilterMode;

    // Color blending modes (pre-defined)
    typedef enum {
        BLEND_ALPHA = 0,
        BLEND_ADDITIVE,
        BLEND_MULTIPLIED
    } BlendMode;

    // Shader location point type
    typedef enum {
        LOC_VERTEX_POSITION = 0,
        LOC_VERTEX_TEXCOORD01,
        LOC_VERTEX_TEXCOORD02,
        LOC_VERTEX_NORMAL,
        LOC_VERTEX_TANGENT,
        LOC_VERTEX_COLOR,
        LOC_MATRIX_MVP,
        LOC_MATRIX_MODEL,
        LOC_MATRIX_VIEW,
        LOC_MATRIX_PROJECTION,
        LOC_VECTOR_VIEW,
        LOC_COLOR_DIFFUSE,
        LOC_COLOR_SPECULAR,
        LOC_COLOR_AMBIENT,
        LOC_MAP_ALBEDO,          // LOC_MAP_DIFFUSE
        LOC_MAP_METALNESS,       // LOC_MAP_SPECULAR
        LOC_MAP_NORMAL,
        LOC_MAP_ROUGHNESS,
        LOC_MAP_OCCLUSION,
        LOC_MAP_EMISSION,
        LOC_MAP_HEIGHT,
        LOC_MAP_CUBEMAP,
        LOC_MAP_IRRADIANCE,
        LOC_MAP_PREFILTER,
        LOC_MAP_BRDF
    } ShaderLocationIndex;

    // Shader uniform data types
    typedef enum {
        UNIFORM_FLOAT = 0,
        UNIFORM_VEC2,
        UNIFORM_VEC3,
        UNIFORM_VEC4,
        UNIFORM_INT,
        UNIFORM_IVEC2,
        UNIFORM_IVEC3,
        UNIFORM_IVEC4,
        UNIFORM_SAMPLER2D
    } ShaderUniformDataType;

    #define LOC_MAP_DIFFUSE      LOC_MAP_ALBEDO
    #define LOC_MAP_SPECULAR     LOC_MAP_METALNESS

    // Material map type
    typedef enum {
        MAP_ALBEDO    = 0,       // MAP_DIFFUSE
        MAP_METALNESS = 1,       // MAP_SPECULAR
        MAP_NORMAL    = 2,
        MAP_ROUGHNESS = 3,
        MAP_OCCLUSION,
        MAP_EMISSION,
        MAP_HEIGHT,
        MAP_CUBEMAP,             // NOTE: Uses GL_TEXTURE_CUBE_MAP
        MAP_IRRADIANCE,          // NOTE: Uses GL_TEXTURE_CUBE_MAP
        MAP_PREFILTER,           // NOTE: Uses GL_TEXTURE_CUBE_MAP
        MAP_BRDF
    } MaterialMapType;

    #define MAP_DIFFUSE      MAP_ALBEDO
    #define MAP_SPECULAR     MAP_METALNESS

}