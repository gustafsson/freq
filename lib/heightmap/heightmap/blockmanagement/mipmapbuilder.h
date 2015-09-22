#ifndef HEIGHTMAP_BLOCKMANAGEMENT_MIPMAPBUILDER_H
#define HEIGHTMAP_BLOCKMANAGEMENT_MIPMAPBUILDER_H

#include "GlTexture.h"
#include "heightmap/render/shaderresource.h"

namespace Heightmap {
namespace BlockManagement {

/**
 * @brief The MipmapBuilder class should build custom mipmaps fast.
 *
 * It uses a framebuffer object and custom shader to build mipmaps on the GPU,
 * i.e without transferring the texture over to the CPU and back.
 */
class MipmapBuilder final
{
public:
    MipmapBuilder();
    ~MipmapBuilder();

    enum MipmapOperator {
        MipmapOperator_ArithmeticMean, // just a regular "mean", equivalent to glGenerateMipmap (GL_TEXTURE_2D)
        MipmapOperator_GeometricMean, // pow(x1*x2*...*xN,1/N), aka log-average (but this is different from the logarithmic mean)
        MipmapOperator_HarmonicMean, // 1/(1/x1 + 1/x2 + ... + 1/xN)
        MipmapOperator_SqrMean, // aka gamma corrected mean, for Gamma=2.0
        MipmapOperator_Max,
        MipmapOperator_Min,

        // OTA: On the Performance of the Order-Truncate-Average-Ratio Spectral Filter
        // This is not really an implementation of OTA, but rather inspired from OTA.
        MipmapOperator_OTA,

        MipmapOperator_Last
    };

    // mipmaps must already be allocated in t
    void buildMipmaps(const GlTexture& t, MipmapOperator op=MipmapOperator_Max, int maxlevels = -1);

private:
    struct ShaderInfo {
        ShaderPtr p;
        int qt_Vertex;
        int qt_MultiTexCoord0;
        int qt_Texture0;
        int subtexeloffset;
        int level;

    };

    unsigned fbo_;
    unsigned vbo_;

    ShaderInfo shaders_[MipmapOperator_Last];

    void init();

public:
    static void test();
};

} // namespace BlockManagement
} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKMANAGEMENT_MIPMAPBUILDER_H
