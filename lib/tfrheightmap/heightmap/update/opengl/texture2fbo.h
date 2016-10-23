#ifndef HEIGHTMAP_UPDATE_OPENGL_TEXTURE2FBO_H
#define HEIGHTMAP_UPDATE_OPENGL_TEXTURE2FBO_H

#include "heightmap/amplitudeaxis.h"
#include "heightmap/blocklayout.h"
#include "heightmap/freqaxis.h"
#include "heightmap/position.h"
#include "tfr/freqaxis.h"
#include "tfr/chunk.h"

#include "zero_on_move.h"

#include <functional>
#include <QOpenGLFunctions>

namespace Heightmap {
namespace Update {
namespace OpenGL {

/**
 * @brief The Texture2Fbo class just draws a VBO. It has nothing to do with any FBO nor any
 * texture.
 */
class Texture2Fbo: QOpenGLFunctions
{
public:
    /**
     * @brief The Params class should be fast and cheap to create
     */
    class Params: QOpenGLFunctions {
    public:
        Params(Tfr::pChunk chunk,
               Heightmap::Region region,
               Heightmap::FreqAxis display_scale,
               Heightmap::BlockLayout block_layout);

        int createVbo(int& out_num_elements) const;

    private:
        const BlockLayout block_layout;
        const Heightmap::Region region;
        const Heightmap::FreqAxis display_scale;
        const Tfr::FreqAxis chunk_scale;

        float a_t, b_t, u0, u1;
        unsigned nScales, nSamples, nValidSamples;
        bool transpose;
        int data_width, data_height;
    };

    Texture2Fbo(const Params& p, float normalization_factor);
    Texture2Fbo(Texture2Fbo&&)=default;
    Texture2Fbo(const Texture2Fbo&)=delete;
    Texture2Fbo& operator=(const Texture2Fbo&)=delete;
    ~Texture2Fbo();

    void draw(int vertex_attrib, int tex_attrib) const;
    float normalization_factor() const { return normalization_factor_; }

private:
    float                               normalization_factor_;
    JustMisc::zero_on_move<unsigned>    vbo_;
    int                                 num_elements_;
};

} // namespace OpenGL
} // namespace Update
} // namespace Heightmap


namespace std {
    template<>
    struct hash<Heightmap::Update::OpenGL::Texture2Fbo::Params>
    {
        std::size_t operator()(Heightmap::Update::OpenGL::Texture2Fbo::Params const& p) const
        {
            // good enough
            std::size_t s = 0;
            char const* a = (char const*)&p;
            for (unsigned i=0; i<sizeof(p); i++)
                s += a[i];
            return s;
        }
    };
}

#endif // HEIGHTMAP_UPDATE_OPENGL_TEXTURE2FBO_H
