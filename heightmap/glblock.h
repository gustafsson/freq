#ifndef HEIGHTMAPVBO_H
#define HEIGHTMAPVBO_H

// gpumisc
#include <mappedvbo.h>

// std
#include <stdio.h>
#include <complex>


namespace Heightmap {
    class Collection;

GLuint loadGLSLProgram(const char *vertFileName, const char *fragFileName);

class GlBlock
{
public:
    GlBlock( Collection* collection, float width, float height );
    ~GlBlock();

    typedef boost::shared_ptr< MappedVbo<float> > pHeight;
    typedef DataStorage<float>::Ptr pHeightReadOnlyCpu;
    //typedef cudaArray* HeightReadOnlyArray;

    void                reset( float width, float height );

    pHeight             height();
    pHeightReadOnlyCpu  heightReadOnlyCpu();
    //HeightReadOnlyArray heightReadOnlyArray();
    DataStorageSize     heightSize() const;

    /**
        'unmap' releases copies of pHeight and pSlope held by GlBlock and
        updates the OpenGL textures that are used for rendering.

        It is an error for a client to call unmap while keeping an instance
        of pHeight or pSlope. Because if there is an instance of pHeight left.
        The Vbo is not unmapped from cuda, glBindBuffer doesn't do anything and
        glTexSubImage2D fails. unmap ensures this condition with an assert.
      */
    void unmap();

    bool has_texture();
    void delete_texture();

    void draw( unsigned vbo_size );
    void draw_flat( );
    //void draw_directMode( );

    unsigned allocated_bytes_per_element();

private:
    typedef boost::shared_ptr< MappedVbo<std::complex<float> > > pSlope;
    pSlope slope();

    void createHeightVbo();

    void create_texture( bool create_slope );
    void update_texture( bool create_slope );
    /**
      Update the slope texture used by the vertex shader. Called when height
      data has been updated.
      */
    void computeSlope( unsigned /*cuda_stream */);

    Collection* _collection;

    pHeightReadOnlyCpu _read_only_cpu;
    //cudaGraphicsResource* _read_only_array_resource;
    //HeightReadOnlyArray _read_only_array;

    pVbo _height;
    pVbo _slope;

    pHeight _mapped_height;
    pSlope _mapped_slope;

    unsigned _tex_height;
    unsigned _tex_slope;

    float _world_width, _world_height;
    bool _got_new_height_data;
};

typedef boost::shared_ptr<GlBlock> pGlBlock;

} // namespace Heightmap

#endif // HEIGHTMAPVBO_H

