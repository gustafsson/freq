#ifndef HEIGHTMAPVBO_H
#define HEIGHTMAPVBO_H

// gpumisc
#include <mappedvbo.h>


//#define BLOCK_INDEX_TYPE GL_UNSIGNED_SHORT
//#define BLOCKindexType GLushort
#define BLOCK_INDEX_TYPE GL_UNSIGNED_INT
#define BLOCKindexType GLuint


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

    enum HeightMode
    {
        HeightMode_Flat=0,
        HeightMode_VertexTexture=1,
        HeightMode_VertexBuffer=2
    };

    void draw( unsigned vbo_size, HeightMode heightMode=HeightMode_VertexTexture );
    //void draw_directMode( );

    unsigned allocated_bytes_per_element();

private:
    void createHeightVbo();

    bool create_texture( HeightMode heightMode );
    void update_texture( HeightMode heightMode );

    Collection* _collection;

    pHeightReadOnlyCpu _read_only_cpu;
    //cudaGraphicsResource* _read_only_array_resource;
    //HeightReadOnlyArray _read_only_array;

    pVbo _height;
    pVbo _mesh;

    pHeight _mapped_height;

    unsigned _tex_height;
    unsigned _tex_height_nearest;

    float _world_width, _world_height;
    bool _got_new_height_data;
};

typedef boost::shared_ptr<GlBlock> pGlBlock;

} // namespace Heightmap

#endif // HEIGHTMAPVBO_H

