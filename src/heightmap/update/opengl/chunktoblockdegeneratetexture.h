#ifndef HEIGHTMAP_UPDATE_OPENGL_CHUNKTOBLOCKDEGENERATETEXTURE_H
#define HEIGHTMAP_UPDATE_OPENGL_CHUNKTOBLOCKDEGENERATETEXTURE_H

#include "block.h"
#include "tfr/chunkdata.h"
#include "tfr/chunkfilter.h"
#include "tfr/freqaxis.h"
#include "heightmap/amplitudeaxis.h"
#include "glframebuffer.h"

#include <future>

class GlTexture;

namespace Heightmap {
namespace Update {
namespace OpenGL {

class BlockFbo {
public:
    BlockFbo (pBlock block);
    ~BlockFbo();

    GlFrameBuffer::ScopeBinding begin ();

private:
    pBlock block;
    boost::shared_ptr<Render::GlBlock> glblock;
    std::unique_ptr<GlFrameBuffer> fbo;
};

/**
 * @brief The ChunkToBlockDegenerateTexture class should merge the contents of
 * a chunk directly onto the texture of a block.
 *
 * TODO: Optimizing OpenGL Texture Transfers
 * http://on-demand.gputechconf.com/gtc/2012/presentations/S0356-GTC2012-Texture-Transfers.pdf
 */
class ChunkToBlockDegenerateTexture
{
public:
    typedef std::map<pBlock, std::shared_ptr<BlockFbo>> BlockFbos;
    class Shaders;

    class Shader {
    public:
        Shader(unsigned program);
        Shader(const Shader&)=delete;
        Shader& operator=(const Shader&)=delete;
        ~Shader();

        void setParams(int data_width, int data_height, int tex_width, int tex_height,
                       float normalization_factor, int amplitude_axis);

        const unsigned program;

    private:        
        int normalization_location_;
        int amplitude_axis_location_;
        int data_size_loc_;
        int tex_size_loc_;
    };


    class Shaders {
    public:
        Shaders ();

        Shader chunktoblock_shader_;
        Shader chunktoblock_maxwidth_shader_;
        Shader chunktoblock_maxheight_shader_;
    };


    class ShaderTexture {
    public:
        ShaderTexture(Shaders& shaders_);

        void prepareShader (int data_width, int data_height, unsigned chunk_pbo);
        void prepareShader (int data_width, int data_height, float* data);

        const GlTexture& getTexture ();
        unsigned getProgram (float normalization_factor, int amplitude_axis);

    private:
        void prepareShader (int data_width, int data_height, unsigned chunk_pbo, float* data);

        int data_width, data_height, tex_width, tex_height;
        std::shared_ptr<GlTexture> chunk_texture_;
        Shaders& shaders_;
        Shader* shader_;
    };


    struct Parameters {
        AmplitudeAxis amplitude_axis;
        Tfr::FreqAxis display_scale;
        BlockLayout block_layout;
        float normalization_factor;

        bool operator==(const Parameters& b) const {
            return amplitude_axis == b.amplitude_axis && display_scale == b.display_scale &&
                    block_layout == b.block_layout && normalization_factor == b.normalization_factor;
        }
    };

    class DrawableChunk {
    public:
        DrawableChunk(Tfr::pChunk chunk,
                      const Parameters& params,
                      BlockFbos& block_fbos,
                      Shaders& shaders);
        DrawableChunk(DrawableChunk&& b) = default;
        DrawableChunk(DrawableChunk&) = delete;
        DrawableChunk& operator=(DrawableChunk&) = delete;
        ~DrawableChunk();

        bool has_chunk() const;
        bool ready() const;
        std::packaged_task<void()> transferData(float *p);
        void prepareShader();
        void draw();

    private:
        void setupPbo();
        void setupVbo();

        template<class T>
        class ZeroOnMove {
        public:
            ZeroOnMove(T t) : t(t) {}
            ZeroOnMove(ZeroOnMove&&b) : t(b.t) {b.t=0;}
            ZeroOnMove(ZeroOnMove&b)=delete;
            ZeroOnMove& operator=(ZeroOnMove&b)=delete;
            ZeroOnMove& operator=(T v) { t = v; return *this; }
            operator bool() const { return (bool)t; }
            operator T() const { return t; }
            T* operator &() { return &t; } // Behave like a 'T'
            const T* operator &() const { return &t; } // Behave like a 'T'
        private:
            T t;
        };

        Tfr::ChunkData::ptr chunk_;
        const Parameters params_;
        BlockFbos& block_fbos_;
        ShaderTexture shader_;

        ZeroOnMove<float*> mapped_chunk_data_;
        std::future<void> data_transfer;

        Tfr::FreqAxis chunk_scale;
        ZeroOnMove<unsigned> vbo_;
        ZeroOnMove<unsigned> chunk_pbo_;
//        ZeroOnMove<void*> sync_;

        float a_t, b_t, u0, u1;
        unsigned nScales, nSamples, nValidSamples;
        int data_width, data_height;
        bool transpose;
    };


    ChunkToBlockDegenerateTexture();
    ~ChunkToBlockDegenerateTexture();

    void            setParams (AmplitudeAxis amplitude_axis, Tfr::FreqAxis display_scale, BlockLayout block_layout, float normalization_factor);
    void            prepareBlock (pBlock block);
    DrawableChunk   prepareChunk (Tfr::pChunk chunk);
    void            finishBlocks ();
    BlockFbos&      block_fbos() { return block_fbos_; }

private:
    Shaders shaders_;
    BlockFbos block_fbos_;
    Parameters params_;

public:
    static void test();
};

} // namespace OpenGL
} // namespace Update
} // namespace Heightmap

#endif // HEIGHTMAP_UPDATE_OPENGL_CHUNKTOBLOCKDEGENERATETEXTURE_H
