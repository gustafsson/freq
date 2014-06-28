#include "renderblock.h"
#include "glblock.h"
#include "shaderresource.h"

// gpumisc
#include "computationkernel.h"
#include "GlException.h"
#include "gl.h"
#include "tasktimer.h"
#include "glPushContext.h"
#include "unused.h"


//#define TIME_RENDERER_BLOCKS
#define TIME_RENDERER_BLOCKS if(0)


using namespace std;


void initShaders() {
    Q_INIT_RESOURCE(shaders);
}

namespace Heightmap {
namespace Render {


RenderBlock::Renderer::Renderer(RenderBlock* render_block, BlockLayout block_size)
    :
      vbo_size(render_block->_vbo_size),
      render_settings(*render_block->render_settings)
{
    render_block->beginVboRendering(block_size);
}


RenderBlock::Renderer::~Renderer()
{
    RenderBlock::endVboRendering();
}


void RenderBlock::Renderer::
        renderBlock( pBlock block )
{
    TIME_RENDERER_BLOCKS ComputationCheckError();
    TIME_RENDERER_BLOCKS GlException_CHECK_ERROR();

    Region r = block->getRegion ();
    glPushMatrixContext mc( GL_MODELVIEW );

    TIME_RENDERER_BLOCKS TaskTimer tt(boost::format("renderBlock %s") % r);

    glTranslatef(r.a.time, 0, r.a.scale);
    glScalef(r.time(), 1, r.scale());

    if (0 /* direct rendering */ )
        ;//block->glblock->draw_directMode();
    else if (1 /* vbo */ ) {
        block->glblock->draw( vbo_size, render_settings.draw_flat ? GlBlock::HeightMode_Flat : render_settings.vertex_texture ? GlBlock::HeightMode_VertexTexture : GlBlock::HeightMode_VertexBuffer);
    }

    TIME_RENDERER_BLOCKS ComputationCheckError();
    TIME_RENDERER_BLOCKS GlException_CHECK_ERROR();
}


RenderBlock::
        RenderBlock(RenderSettings* render_settings)
    :
        render_settings( render_settings ),
        _color_texture_colors( (RenderSettings::ColorMode)-1 ),
        _shader_prog(0),
      _mesh_index_buffer(0),
      _mesh_width(0),
      _mesh_height(0)
{
}


void RenderBlock::
        init()
{
    initShaders();

    // load shader
    if (render_settings->vertex_texture)
        _shader_prog = ShaderResource::loadGLSLProgram(":/shaders/heightmap.vert", ":/shaders/heightmap.frag");
    else
        _shader_prog = ShaderResource::loadGLSLProgram(":/shaders/heightmap_noshadow.vert", ":/shaders/heightmap.frag");

    if (0 == _shader_prog)
        return;
}


void RenderBlock::
        clearCaches()
{
    _mesh_width = 0;
    _mesh_height = 0;
    _mesh_position.reset();
    glDeleteProgram(_shader_prog);
    _shader_prog = 0;
    _colorTexture.reset();
    _color_texture_colors = (RenderSettings::ColorMode)-1;
}


void RenderBlock::
        beginVboRendering(BlockLayout block_size)
{
    GlException_CHECK_ERROR();
    //unsigned meshW = collection->samples_per_block();
    //unsigned meshH = collection->scales_per_block();

    createColorTexture(24); // These will be linearly interpolated when rendering, so a high resolution texture is not needed
    glActiveTexture(GL_TEXTURE2);
    _colorTexture->bindTexture2D();
    glActiveTexture(GL_TEXTURE0);

    glUseProgram(_shader_prog);

    // TODO check if this takes any time
    {   // Set default uniform variables parameters for the vertex and pixel shader
        TIME_RENDERER_BLOCKS TaskTimer tt("Setting shader parameters");
        GLuint uniVertText0,
                uniVertText1,
                uniVertText2,
                uniColorTextureFactor,
                uniFixedColor,
                uniClearColor,
                uniContourPlot,
                uniYScale,
                uniYOffset,
                uniLogScale,
                uniScaleTex,
                uniOffsTex;

        uniVertText0 = glGetUniformLocation(_shader_prog, "tex");
        glUniform1i(uniVertText0, 0); // GL_TEXTURE0

        uniVertText1 = glGetUniformLocation(_shader_prog, "tex_nearest");
        glUniform1i(uniVertText1, 1); // GL_TEXTURE1

        uniVertText2 = glGetUniformLocation(_shader_prog, "tex_color");
        glUniform1i(uniVertText2, 2); // GL_TEXTURE2

        uniFixedColor = glGetUniformLocation(_shader_prog, "fixedColor");
        switch (render_settings->color_mode)
        {
        case RenderSettings::ColorMode_Grayscale:
            glUniform4f(uniFixedColor, 0.f, 0.f, 0.f, 0.f);
            break;
        case RenderSettings::ColorMode_BlackGrayscale:
            glUniform4f(uniFixedColor, 1.f, 1.f, 1.f, 0.f);
            break;
        default:
        {
            tvector<4, float> fixed_color = render_settings->fixed_color;
            glUniform4f(uniFixedColor, fixed_color[0], fixed_color[1], fixed_color[2], fixed_color[3]);
            break;
        }
        }

        uniClearColor = glGetUniformLocation(_shader_prog, "clearColor");
        tvector<4, float> clear_color = render_settings->clear_color;
        glUniform4f(uniClearColor, clear_color[0], clear_color[1], clear_color[2], clear_color[3]);

        uniColorTextureFactor = glGetUniformLocation(_shader_prog, "colorTextureFactor");
        switch(render_settings->color_mode)
        {
        case RenderSettings::ColorMode_Rainbow:
        case RenderSettings::ColorMode_GreenRed:
        case RenderSettings::ColorMode_GreenWhite:
        case RenderSettings::ColorMode_Green:
            glUniform1f(uniColorTextureFactor, 1.f);
            break;
        default:
            glUniform1f(uniColorTextureFactor, 0.f);
            break;
        }

        uniContourPlot = glGetUniformLocation(_shader_prog, "contourPlot");
        glUniform1f(uniContourPlot, render_settings->draw_contour_plot ? 1.f : 0.f );

        uniYScale = glGetUniformLocation(_shader_prog, "yScale");
        glUniform1f(uniYScale, render_settings->y_scale);

        uniYOffset = glGetUniformLocation(_shader_prog, "yOffset");
        glUniform1f(uniYOffset, render_settings->y_offset);

        // yOffset specifies 'b' which says which 'v' that should render as 0
        // yOffset=-1 => v>1 => fragColor>0
        // yOffset=0  => v>L => fragColor>0
        // yOffset=1  => v>0 => fragColor>0
        float L = 0.00001;
        float tb = 1.0/L - 1.0;
        float tc = L/(1.0 - tb);
        float ta = L - tc;
        float b = ta * exp(-render_settings->y_offset * log(tb)) + tc;

        // yScale specifies which intensity 'v=1' should have
        // v<1 => fragColor < yScale
        // v=1 => fragColor = yScale
        // v>1 => fragColor > yScale
        float x1 = render_settings->y_scale / (log(1.0) - log(b));
        float x2 = - log(b) * x1;

        uniLogScale = glGetUniformLocation(_shader_prog, "logScale");
        glUniform3f(uniLogScale, render_settings->log_scale, x1, x2);

        float
                w = block_size.texels_per_row (),
                h = block_size.texels_per_column ();

        uniScaleTex = glGetUniformLocation(_shader_prog, "scale_tex");
        glUniform2f(uniScaleTex, (w-1.f)/w, (h-1.f)/h);

        uniOffsTex = glGetUniformLocation(_shader_prog, "offset_tex");
        glUniform2f(uniOffsTex, .5f/w, .5f/h);
    }

    glBindBuffer(GL_ARRAY_BUFFER, *_mesh_position);
    glVertexPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _mesh_index_buffer);

    GlException_CHECK_ERROR();
}


void RenderBlock::
        endVboRendering()
{
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glActiveTexture(GL_TEXTURE2);
    GlTexture::unbindTexture2D();
    glActiveTexture(GL_TEXTURE0);

    glDisableClientState(GL_VERTEX_ARRAY);
    glUseProgram(0);
}


void RenderBlock::
        setSize( unsigned w, unsigned h)
{
    if (w == _mesh_width && h ==_mesh_height)
        return;

    createMeshIndexBuffer(w, h);
    createMeshPositionVBO(w, h);
}


unsigned RenderBlock::
        trianglesPerBlock()
{
    return (_mesh_width-1) * (_mesh_height-1) * 2;
}


// create index buffer for rendering quad mesh
void RenderBlock::
        createMeshIndexBuffer(int w, int h)
{
    GlException_CHECK_ERROR();

    // create index buffer
    if (_mesh_index_buffer)
        glDeleteBuffersARB(1, &_mesh_index_buffer);

    // edge dropout to eliminate visible glitches
    if (w>2) w+=2;
    if (h>2) h+=2;

    _mesh_width = w;
    _mesh_height = h;

    _vbo_size = ((w*2)+4)*(h-1);

    std::vector<BLOCKindexType> indicesdata(_vbo_size);
    BLOCKindexType *indices = &indicesdata[0];
    if (indices) for(int y=0; y<h-1; y++) {
        *indices++ = y*w;
        for(int x=0; x<w; x++) {
            *indices++ = y*w+x;
            *indices++ = (y+1)*w+x;
        }
        // start new strip with degenerate triangle
        *indices++ = (y+1)*w+(w-1);
        *indices++ = (y+1)*w;
        *indices++ = (y+1)*w;
    }

    glGenBuffers(1, &_mesh_index_buffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _mesh_index_buffer);

    // fill with indices for rendering mesh as triangle strips
    GlException_SAFE_CALL( glBufferData(GL_ELEMENT_ARRAY_BUFFER, _vbo_size*sizeof(BLOCKindexType), &indicesdata[0], GL_STATIC_DRAW) );

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    GlException_CHECK_ERROR();
}


// create fixed vertex buffer to store mesh vertices
void RenderBlock::
        createMeshPositionVBO(int w, int h)
{
    int y1 = 0, x1 = 0, y2 = h, x2 = w;

    // edge dropout to eliminate visible glitches
    if (w>2) x1--, x2++;
    if (h>2) y1--, y2++;

    std::vector<float> posdata( (x2-x1)*(y2-y1)*4 );
    float *pos = &posdata[0];

    for(int y=y1; y<y2; y++) {
        for(int x=x1; x<x2; x++) {
            float u = x / (float) (w-1);
            float v = y / (float) (h-1);
            *pos++ = u;
            *pos++ = 0.0f;
            *pos++ = v;
            *pos++ = 1.0f;
        }
    }

    _mesh_position.reset( new Vbo( (w+2)*(h+2)*4*sizeof(float), GL_ARRAY_BUFFER, GL_STATIC_DRAW, &posdata[0] ));
}


tvector<4,float> mix(tvector<4,float> a, tvector<4,float> b, float f)
{
    return a*(1-f) + b*f;
}

tvector<4,float> getWavelengthColorCompute( float wavelengthScalar, RenderSettings::ColorMode scheme ) {
    tvector<4,float> spectrum[12];
    int count = 0;
    spectrum[0] = tvector<4,float>( 0, 0, 0, 0 );

    switch (scheme)
    {
    case RenderSettings::ColorMode_GreenRed:
        spectrum[0] = tvector<4,float>( 0, 1, 0, 0 ),
        spectrum[1] = tvector<4,float>( 0, 1, 0, 0 ),
        spectrum[2] = tvector<4,float>( 0, 1, 0, 0 ),
        spectrum[3] = tvector<4,float>( 0, 1, 0, 0 ),
        spectrum[4] = tvector<4,float>( 0, 1, 0, 0 ),
        spectrum[5] = tvector<4,float>( 1, 1, 0, 0 ),
        spectrum[6] = tvector<4,float>( 1, 1, 0, 0 ),
        spectrum[7] = tvector<4,float>( 1, 0, 0, 0 );
        spectrum[8] = tvector<4,float>( 1, 0, 0, 0 );
        spectrum[9] = tvector<4,float>( 1, 0, 0, 0 );
        spectrum[10] = tvector<4,float>( -0.5, 0, 0, 0 ); // dark line, almost black
        spectrum[11] = tvector<4,float>( 0.75, 0, 0, 0 ); // dark red when over the top
        count = 11;
        break;
    case RenderSettings::ColorMode_GreenWhite:
        if (wavelengthScalar<0)
            return tvector<4,float>( 0, 0, 0, 0 );
        spectrum[0] = tvector<4,float>( 0, 1, 0, 0 ),
        spectrum[1] = tvector<4,float>( 0, 1, 0, 0 ),
        spectrum[2] = tvector<4,float>( 0, 1, 0, 0 ),
        spectrum[3] = tvector<4,float>( 1, 1, 1, 0 );
        spectrum[4] = tvector<4,float>( 1, 1, 1, 0 );
        spectrum[5] = tvector<4,float>( 1, 1, 1, 0 );
        spectrum[6] = tvector<4,float>( 1, 1, 1, 0 );
        //spectrum[7] = tvector<4,float>( -0.5, -0.5, -0.5, 0 ); // dark line, almost black
        spectrum[7] = tvector<4,float>( 1, 1, 1, 0 ); // darker when over the top
        count = 7;
        break;
    case RenderSettings::ColorMode_Green:
        if (wavelengthScalar<0)
            return tvector<4,float>( 0, 0, 0, 0 );
        spectrum[0] = tvector<4,float>( 0, 0, 0, 0 );
        spectrum[1] = tvector<4,float>( 0, 0, 0, 0 );
        spectrum[2] = tvector<4,float>( 0, 1, 0, 0 );
        spectrum[3] = tvector<4,float>( 0, 1, 0, 0 );
        spectrum[4] = tvector<4,float>( 0, 1, 0, 0 );
        spectrum[5] = tvector<4,float>( 0, 1, 0, 0 );
        spectrum[6] = tvector<4,float>( 0, 1, 0, 0 );
        count = 6;
        break;
    case RenderSettings::ColorMode_Grayscale:
        break;
    case RenderSettings::ColorMode_BlackGrayscale:
        if (wavelengthScalar<0)
            return tvector<4,float>( 0, 0, 0, 0 );
        break;
    default:
        /* for white background */
        float a = 1/255.f;
        // rainbow http://en.wikipedia.org/wiki/Rainbow#Spectrum
        spectrum[0] = tvector<4,float>( 1, 0, 0, 0 ), // red
        spectrum[1] = tvector<4,float>( 148*a, 0, 211*a, 0 ), // violet
        spectrum[2] = tvector<4,float>( 148*a, 0, 211*a, 0 ), // violet
        spectrum[3] = tvector<4,float>( 75*a, 0, 130*a, 0 ), // indigo
        spectrum[4] = tvector<4,float>( 0, 0, 1, 0 ), // blue
        spectrum[5] = tvector<4,float>( 0, 0.5, 0, 0 ), // green
        spectrum[6] = tvector<4,float>( 1, 1, 0, 0 ), // yellow
        spectrum[7] = tvector<4,float>( 1, 0.5, 0, 0 ), // orange
        spectrum[8] = tvector<4,float>( 1, 0, 0, 0 ), // red
        spectrum[9] = tvector<4,float>( 1, 0, 0, 0 );
        spectrum[10] = tvector<4,float>( -0.5, 0, 0, 0 ); // dark line, almost black
        spectrum[11] = tvector<4,float>( 0.75, 0, 0, 0 ); // dark red when over the top
        count = 11;
//        spectrum[0] = tvector<4,float>( 1, 0, 1, 0 ),
//        spectrum[1] = tvector<4,float>( 0, 0, 1, 0 ),
//        spectrum[2] = tvector<4,float>( 0, 1, 1, 0 ),
//        spectrum[3] = tvector<4,float>( 0, 1, 0, 0 ),
//        spectrum[4] = tvector<4,float>( 1, 1, 0, 0 ),
//        spectrum[5] = tvector<4,float>( 1, 0, 1, 0 ),
//        spectrum[6] = tvector<4,float>( 1, 0, 0, 0 );
//        spectrum[7] = tvector<4,float>( 1, 0, 0, 0 );
//        spectrum[8] = tvector<4,float>( 0, 0, 0, 0 );
//        spectrum[9] = tvector<4,float>( 0.5, 0, 0, 0 );
//        count = 9;//sizeof(spectrum)/sizeof(spectrum[0])-1;

        /* for black background
            { 0, 0, 0 },
            { 1, 0, 1 },
            { 0, 0, 1 },
            { 0, 1, 1 },
            { 0, 1, 0 },
            { 1, 1, 0 },
            { 1, 0, 0 }}; */
        break;
    }

    if (wavelengthScalar<0)
        return tvector<4,float>( 1, 1, 1, 1 );

    float f = float(count)*wavelengthScalar;
    int i1 = int(floor(max(0.f, min(f-1.f, float(count)))));
    int i2 = int(floor(max(0.f, min(f, float(count)))));
    int i3 = int(floor(max(0.f, min(f+1.f, float(count)))));
    int i4 = int(floor(max(0.f, min(f+2.f, float(count)))));
    float t = (f-float(i2))*0.5;
    float s = 0.5 + t;

    tvector<4,float> rgb = mix(spectrum[i1], spectrum[i3], s) + mix(spectrum[i2], spectrum[i4], t);
    rgb = rgb * 0.5;
    //TaskInfo("%g %g %g: %g %g %g %g", f, t, s, rgb[0], rgb[1], rgb[2], rgb[3]);
    return rgb;
}


void RenderBlock::
        createColorTexture(unsigned N)
{
    if (_color_texture_colors == render_settings->color_mode && _colorTexture && _colorTexture->getWidth()==N)
        return;

    _color_texture_colors = render_settings->color_mode;

    std::vector<tvector<4,float> > texture(N);
    for (unsigned i=0; i<N; ++i) {
        texture[i] = getWavelengthColorCompute( i/(float)(N-1), _color_texture_colors );
    }
    _colorTexture.reset( new GlTexture(N,1, GL_RGBA, GL_RGBA, GL_FLOAT, &texture[0]));

    render_settings->clear_color = getWavelengthColorCompute( -1.f, _color_texture_colors );
}

} // namespace Render
} // namespace Heightmap
