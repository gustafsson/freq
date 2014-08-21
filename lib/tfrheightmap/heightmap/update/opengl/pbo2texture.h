#ifndef HEIGHTMAP_UPDATE_OPENGL_PBO2TEXTURE_H
#define HEIGHTMAP_UPDATE_OPENGL_PBO2TEXTURE_H

#include "GlTexture.h"
#include "tfr/chunk.h"
#include "glprojection.h"

namespace Heightmap {
namespace Update {
namespace OpenGL {

class Shaders;

class Shader {
public:
    Shader(unsigned program);
    Shader(const Shader&)=delete;
    Shader& operator=(const Shader&)=delete;
    ~Shader();

    void setParams(int data_width, int data_height, int tex_width, int tex_height,
                   float normalization_factor, int amplitude_axis, const glProjection& M );

    const unsigned program;

private:
    int normalization_location_;
    int amplitude_axis_location_;
    int modelViewProjectionMatrix_location_;
    int data_size_loc_;
    int tex_size_loc_;
};


class Shaders {
public:
    Shaders ();

    int load_shaders_;
    Shader chunktoblock_shader_;
    Shader chunktoblock_maxwidth_shader_;
    Shader chunktoblock_maxheight_shader_;
};


class ShaderTexture {
public:
    ShaderTexture(Shaders& shaders_);

    void prepareShader (int data_width, int data_height, unsigned chunk_pbo, bool f32);
    void prepareShader (int data_width, int data_height, void* data, bool f32);

    GlTexture& getTexture ();
    unsigned getProgram (float normalization_factor, int amplitude_axis, const glProjection& M);

private:
    void prepareShader (int data_width, int data_height, unsigned chunk_pbo, void* data, bool f32);

    int data_width, data_height, tex_width, tex_height;
    GlTexture::ptr chunk_texture_;
    Shaders& shaders_;
    Shader* shader_;
};


//struct Parameters {
//    AmplitudeAxis amplitude_axis;
////    Tfr::FreqAxis display_scale;
//    BlockLayout block_layout;
//    float normalization_factor;

//    bool operator==(const Parameters& b) const {
//        return amplitude_axis == b.amplitude_axis && display_scale == b.display_scale &&
//                block_layout == b.block_layout && normalization_factor == b.normalization_factor;
//    }
//};


class Pbo2Texture {
public:
    class ScopeMap {
    public:
        ScopeMap();
        ScopeMap(ScopeMap&&) = default;
        ScopeMap(const ScopeMap&) = delete;
        ScopeMap operator=(const ScopeMap&) = delete;
        ~ScopeMap();
    };

    Pbo2Texture(Shaders& shaders, Tfr::pChunk chunk, int pbo, bool f32);
    Pbo2Texture(Shaders& shaders, Tfr::pChunk chunk, void *p, bool f32);

    ScopeMap map (float normalization_factor, int amplitude_axis, const glProjection& M, int &vertex_attrib, int &tex_attrib);

private:
    ShaderTexture shader_;
};

} // namespace OpenGL
} // namespace Update
} // namespace Heightmap

#endif // HEIGHTMAP_UPDATE_OPENGL_PBO2TEXTURE_H
