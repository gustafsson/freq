#include "renderblock.h"
#include "shaderresource.h"
#include "heightmap/uncaughtexception.h"
#include "heightmap/render/blocktextures.h"

// gpumisc
#include "computationkernel.h"
#include "GlException.h"
#include "glstate.h"
#include "tasktimer.h"
#include "glPushContext.h"
#include "unused.h"
#include "gluinvertmatrix.h"
#include "float16.h"
#include "log.h"
#include "neat_math.h"

#include <QSettings>

//#define BLOCK_INDEX_TYPE GL_UNSIGNED_SHORT
//#define BLOCKindexType GLushort
#define BLOCK_INDEX_TYPE GL_UNSIGNED_INT
#define BLOCKindexType GLuint

//#define TIME_RENDERER_BLOCKS
#define TIME_RENDERER_BLOCKS if(0)

//#define LOG_DIVS
#define LOG_DIVS if(0)

#define DRAW_POINTS false
#define DRAW_WIREFRAME false

using namespace std;


void initShaders() {
    Q_INIT_RESOURCE(shaders);
}

namespace Heightmap {
namespace Render {


RenderBlock::Renderer::Renderer(RenderBlock* render_block, BlockLayout block_size, glProjecter gl_projecter)
    :
      render_block(render_block),
      gl_projecter(gl_projecter)
{
    render_block->beginVboRendering(block_size);
}


RenderBlock::Renderer::~Renderer()
{
    try {
        render_block->endVboRendering();
    } catch (...) {
        TaskInfo(boost::format("!!! ~RenderBlock::Renderer: endVboRendering failed\n%s") % boost::current_exception_diagnostic_information());
    }
}


void RenderBlock::Renderer::
        renderBlock( pBlock block, CornerResolution cr )
{
    TIME_RENDERER_BLOCKS GlException_CHECK_ERROR();

    Region r = block->getVisibleRegion ();

    TIME_RENDERER_BLOCKS TaskTimer tt(boost::format("renderBlock %s") % r);

    glProjecter blockProj = gl_projecter;
    blockProj.translate (vectord(r.a.time, 0, r.a.scale));
    blockProj.scale (vectord(r.time(), 1, r.scale()));
    glUniformMatrix4fv (render_block->uniModelviewprojection, 1, false, GLmatrixf(blockProj.mvp ()).v ());
    glUniformMatrix4fv (render_block->uniModelview, 1, false, GLmatrixf(blockProj.modelview ()).v ());
    glUniformMatrix4fv (render_block->uniNormalMatrix, 1, false, GLmatrixf(blockProj.modelview_inverse ()).transpose ().v ());

    float mintp =  min(min(min(cr.x00, cr.x01), min(cr.x10, cr.x11)),
                       min(min(cr.y00, cr.y01), min(cr.y10, cr.y11)));
    int subdiv = max(0, min(subdivs-1, (int)log2(mintp)));
    glUniform4f (render_block->uniVertexTextureBiasX, cr.x00, cr.x01, cr.x10, cr.x11);
    glUniform4f (render_block->uniVertexTextureBiasY, cr.y00, cr.y01, cr.y10, cr.y11);

    LOG_DIVS Log("%s x(%g, %g, %g, %g) y(%g, %g, %g, %g) -> %d")
            % block->getVisibleRegion ()
            % cr.x00 % cr.x01 % cr.x10 % cr.x11
            % cr.y00 % cr.y01 % cr.y10 % cr.y11
            % subdiv;

    const auto& pVbo = *render_block->_mesh_index_buffer[subdiv];
    unsigned vbo = pVbo;
    GLsizei n = pVbo.size () / sizeof(BLOCKindexType);
    if (prev_vbo != vbo)
    {
        GlState::glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo);
        prev_vbo = vbo;
    }
    glBindTexture (GL_TEXTURE_2D, block->texture ()->getOpenGlTextureId ());
    glActiveTexture(GL_TEXTURE2);
    glBindTexture (GL_TEXTURE_2D, block->texture_ota ());
    glActiveTexture(GL_TEXTURE0);
    draw(n);

    TIME_RENDERER_BLOCKS GlException_CHECK_ERROR();
}


void RenderBlock::Renderer::
        draw(GLsizei n)
{
    GlException_CHECK_ERROR();

    if (DRAW_POINTS) {
        GlState::glDrawElements(GL_POINTS, n, BLOCK_INDEX_TYPE, 0);
    } else if (DRAW_WIREFRAME) {
#ifdef GL_ES_VERSION_2_0
        GlState::glDrawElements(GL_LINE_STRIP, n, BLOCK_INDEX_TYPE, 0);
#else
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE );
            GlState::glDrawElements(GL_TRIANGLE_STRIP, n, BLOCK_INDEX_TYPE, 0);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
#endif
    } else {
        GlState::glDrawElements(GL_TRIANGLE_STRIP, n, BLOCK_INDEX_TYPE, 0);
    }

    GlException_CHECK_ERROR();
}


RenderBlock::
        RenderBlock(RenderSettings* render_settings)
    :
      _initialized( NotInitialized ),
        render_settings( render_settings ),
        _color_texture_colors( (RenderSettings::ColorMode)-1 ),
      _mesh_width(0),
      _mesh_height(0)
{
}


RenderBlock::
        ~RenderBlock()
{
}


RenderBlock::ShaderData::ShaderData()
{
}


RenderBlock::ShaderData::ShaderData(const char* defines)
{
    _shader_progp = ShaderResource::loadGLSLProgram(":/shaders/heightmap.vert", ":/shaders/heightmap.frag", defines, defines);
    _shader_prog = _shader_progp->programId();
    GlException_CHECK_ERROR_MSG(defines);
}


void RenderBlock::
        init()
{
    if (NotInitialized != _initialized)
        return;

    // assume failure unless we reach the end of this method
    _initialized = InitializationFailed;

    checkExtensions();

    initShaders();

    // default, expected to be overwritten
    setSize (2, 2);

    GlException_CHECK_ERROR();

    _initialized = Initialized;
}


bool RenderBlock::
        isInitialized()
{
    return Initialized == _initialized;
}


void RenderBlock::
        clearCaches()
{
    _initialized = NotInitialized;

    _mesh_width = 0;
    _mesh_height = 0;
    _mesh_position.reset();
    _colorTexture.reset();
    _color_texture_colors = (RenderSettings::ColorMode)-1;
}

#ifdef DARWIN_NO_CARBON
void RenderBlock::
        checkExtensions ()
{
    return;
}
#else
void RenderBlock::
        checkExtensions ()
{
    TaskTimer tt("RenderBlock: checkExtensions");

    GlException_CHECK_ERROR();

#ifdef USE_CUDA
    int cudaDevices=0;
    CudaException_SAFE_CALL( cudaGetDeviceCount( &cudaDevices) );
    if (0 == cudaDevices ) {
        Sawe::NonblockingMessageBox::show(
                QMessageBox::Critical,
                "Couldn't find any \"cuda capable\" device",
                "This version of Sonic AWE requires a graphics card that supports CUDA and no such graphics card was found.\n\n"
                "If you think this messge is an error, please file this as a bug report at bugs.muchdifferent.com to help us fix this." );

        // fail
        return;
    }
#endif

#ifndef __APPLE__ // glewInit is not needed on Mac
    if (0 != glewInit() ) {
        Sawe::NonblockingMessageBox::show(
                QMessageBox::Critical,
                "Couldn't properly setup graphics",
                "Sonic AWE failed to setup required graphics hardware.\n\n"
                "If you think this messge is an error, please file this as a bug report at bugs.muchdifferent.com to help us fix this.",

                "Couldn't initialize \"glew\"");

        // fail
        return;
    }
#endif

    // verify necessary OpenGL extensions
    const char* glversion = (const char*)glGetString(GL_VERSION);
    string glversionstring(glversion);
    stringstream versionreader(glversionstring);
    int gl_major=0, gl_minor=0;
    char dummy;
    versionreader >> gl_major >> dummy >> gl_minor;

    //TaskInfo("OpenGL version %d.%d (%s)", gl_major, gl_minor, glversion);

    if ((1 > gl_major )
        || ( 1 == gl_major && 4 > gl_minor ))
    {
        try {
            BOOST_THROW_EXCEPTION(logic_error(
                    "Couldn't properly setup graphics\n"
                    "Sonic AWE requires a graphics driver that supports OpenGL 2.0 and no such graphics driver was found.\n\n"
                    "If you think this messge is an error, please file this as a bug report at muchdifferent.com to help us fix this."
            ));
        } catch(...) {
            Heightmap::UncaughtException::handle_exception(boost::current_exception ());
        }

        // fail
        return;
    }

    // Don't need extensions on OpenGL > 3
    if (gl_major>=3)
        return;

    const char* exstensions[] = {
        "GL_ARB_vertex_buffer_object",
        "GL_ARB_pixel_buffer_object",
        "",
        "GL_ARB_texture_float"
    };

    bool required_extension = true;
    const char* all_extensions = (const char*)glGetString(GL_EXTENSIONS);
    if (0==all_extensions) {
        Log("glGetString(GL_EXTENSIONS) failed. Assuimg all necessary extensions are in place");
        return;
    }

    //TaskInfo("Checking extensions %s", all_extensions);
    for (unsigned i=0; i < sizeof(exstensions)/sizeof(exstensions[0]); ++i)
    {
        if (0 == strlen(exstensions[i]))
        {
            required_extension = false;
            continue;
        }


        bool hasExtension = 0 != strstr(all_extensions, exstensions[i]);
        if (!hasExtension)
            TaskInfo("%s %s extension %s",
                     hasExtension?"Found":"Couldn't find",
                     required_extension?"required":"preferred",
                     exstensions[i]);

        if (hasExtension)
            continue;

        stringstream err;
        stringstream details;

        err << "Sonic AWE can't properly setup graphics. ";
        if (required_extension)
        {
            err << "Sonic AWE requires features that couldn't be found on your graphics card.";
            details << "Sonic AWE requires a graphics card that supports '" << exstensions[i] << "'";
        }
        else
        {
            bool warn_expected_opengl = QSettings().value("warn_expected_opengl", true).toBool();
            if (!warn_expected_opengl)
                continue;
            QSettings().setValue("warn_expected_opengl", false);

            err << "Sonic AWE works better with features that couldn't be found on your graphics card. "
                << "However, Sonic AWE might still run. Click OK to try.";
            details << "Sonic AWE works better with a graphics card that supports '" << exstensions[i] << "'";
        }

        err << endl << endl << "If you think this messge is an error, please file this as a bug report at bugs.muchdifferent.com to help us fix this.";

        try {
            BOOST_THROW_EXCEPTION(logic_error(
                                      str(boost::format(
                      "Couldn't properly setup graphics\n"
                      "required_extension = %s\n"
                      "err = %s\n"
                      "details = %s\n")
                          % required_extension
                          % err.str()
                          % details.str()
            )));
        } catch(...) {
            try {
            Heightmap::UncaughtException::handle_exception(boost::current_exception ());
            } catch (...) {}
        }

        if (required_extension)
            return;
    }
}
#endif


void RenderBlock::ShaderData::prepShader(BlockLayout block_size, RenderSettings* render_settings)
{
    GlState::glUseProgram(_shader_prog);

    // Set default uniform variables parameters for the vertex and pixel shader
    TIME_RENDERER_BLOCKS TaskTimer tt("Setting shader parameters");

    if (uniModelviewprojection<-1) uniModelviewprojection = glGetUniformLocation (_shader_prog, "ModelViewProjectionMatrix");
    if (uniModelview<-1) uniModelview = glGetUniformLocation (_shader_prog, "ModelViewMatrix");
    if (uniNormalMatrix<-1) uniNormalMatrix = glGetUniformLocation (_shader_prog, "NormalMatrix");

    if (uniVertText0<-1) uniVertText0 = glGetUniformLocation(_shader_prog, "tex");
    if (uniVertText0>=0) if (u_tex != 0) glUniform1i(uniVertText0, u_tex=0); // GL_TEXTURE0 + i

    if (uniVertText2<-1) uniVertText2 = glGetUniformLocation(_shader_prog, "tex_color");
    if (uniVertText2>=0) if (u_tex_color != 1) glUniform1i(uniVertText2, u_tex_color=1);

    if (uniVertTextOta<-1) uniVertTextOta = glGetUniformLocation(_shader_prog, "tex_ota");
    if (uniVertTextOta>=0) if (u_tex_ota != 2) glUniform1i(uniVertTextOta, u_tex_ota=2); // GL_TEXTURE0 + i

    if (uniFixedColor<-1) uniFixedColor = glGetUniformLocation(_shader_prog, "fixedColor");
    tvector<4, float> fixed_color;
    switch (render_settings->color_mode)
    {
    case RenderSettings::ColorMode_Grayscale:
        fixed_color = tvector<4, float>(0,0,0,0);
        break;
    case RenderSettings::ColorMode_BlackGrayscale:
        fixed_color = tvector<4, float>(1,1,1,0);
        break;
    default:
        fixed_color = render_settings->fixed_color;
        break;
    }
    if (fixed_color != u_fixed_color)
    {
        u_fixed_color = fixed_color;
        if (uniFixedColor>=0) glUniform4f(uniFixedColor, fixed_color[0], fixed_color[1], fixed_color[2], fixed_color[3]);
    }

    if (uniClearColor<-1) uniClearColor = glGetUniformLocation(_shader_prog, "clearColor");
    if (u_clearColor != render_settings->clear_color)
    {
        u_clearColor = render_settings->clear_color;
        if (uniClearColor>=0) glUniform4f(uniClearColor, u_clearColor[0], u_clearColor[1], u_clearColor[2], u_clearColor[3]);
    }

    if (uniColorTextureFactor<-1) uniColorTextureFactor = glGetUniformLocation(_shader_prog, "colorTextureFactor");
    float colorTextureFactor;
    switch(render_settings->color_mode)
    {
    case RenderSettings::ColorMode_Rainbow:
    case RenderSettings::ColorMode_GreenRed:
    case RenderSettings::ColorMode_GreenWhite:
    case RenderSettings::ColorMode_Green:
    case RenderSettings::ColorMode_WhiteBlackGray:
        colorTextureFactor = 1.f;
        break;
    default:
        colorTextureFactor = 0.f;
        break;
    }
    if (u_colorTextureFactor != colorTextureFactor)
        if (uniColorTextureFactor>=0) glUniform1f(uniColorTextureFactor, u_colorTextureFactor = colorTextureFactor);

    if (uniContourPlot<-1) uniContourPlot = glGetUniformLocation(_shader_prog, "contourPlot");
    if (u_draw_contour_plot != render_settings->draw_contour_plot)
        if (uniContourPlot>=0) glUniform1f(uniContourPlot, (u_draw_contour_plot=render_settings->draw_contour_plot) ? 1.f : 0.f );

    if (uniFlatness<-1) uniFlatness = glGetUniformLocation(_shader_prog, "flatness");
    float v = render_settings->draw_flat ? 0 : render_settings->last_ysize;
    if (u_flatness != v)
        if (uniFlatness>=0) glUniform1f(uniFlatness, u_flatness=v);

    if (uniYScale<-1) uniYScale = glGetUniformLocation(_shader_prog, "yScale");
    if (u_yScale != render_settings->y_scale)
        if (uniYScale>=0) glUniform1f(uniYScale, u_yScale = render_settings->y_scale);

    if (uniYOffset<-1) uniYOffset = glGetUniformLocation(_shader_prog, "yOffset");
    if (u_yOffset != render_settings->y_offset)
        if (uniYOffset>=0) glUniform1f(uniYOffset, u_yOffset = render_settings->y_offset);

    if (uniYNormalize<-1) uniYNormalize = glGetUniformLocation(_shader_prog, "yNormalize");
    if (u_yNormalize != render_settings->y_normalize)
        if (uniYNormalize>=0) glUniform1f(uniYNormalize, u_yNormalize = render_settings->y_normalize);

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

    if (uniLogScale<-1) uniLogScale = glGetUniformLocation(_shader_prog, "logScale");
    if (u_logScale != render_settings->log_scale || u_logScale_x1 != x1 || u_logScale_x2 != x2)
        if (uniLogScale>=0) glUniform3f(uniLogScale, u_logScale=render_settings->log_scale, u_logScale_x1 = x1, u_logScale_x2 = x2);

    float
            vw = block_size.visible_texels_per_row (),
            vh = block_size.visible_texels_per_column (),
            w = block_size.texels_per_row (),
            h = block_size.texels_per_column (),
            m = block_size.margin ();

    if (uniScaleTex<-1) uniScaleTex = glGetUniformLocation(_shader_prog, "scale_tex");
    if (u_scale_tex1 != vw/w || u_scale_tex2 != vh/h)
        if (uniScaleTex>=0) glUniform2f(uniScaleTex, u_scale_tex1=vw/w, u_scale_tex2=vh/h);

    if (uniOffsTex<-1) uniOffsTex = glGetUniformLocation(_shader_prog, "offset_tex");
    if (u_offset_tex1 != m/w || u_offset_tex2 != m/h)
        if (uniOffsTex>=0) glUniform2f(uniOffsTex, u_offset_tex1=m/w, u_offset_tex2=m/h);

    if (uniTexDelta<-1) uniTexDelta = glGetUniformLocation(_shader_prog, "tex_delta");
    if (u_tex_delta1 != 1.f/w || u_tex_delta2 != 1.f/h)
        if (uniTexDelta>=0) glUniform2f(uniTexDelta, u_tex_delta1=1.f/w, u_tex_delta2=1.f/h);

    if (uniTexSize<-1) uniTexSize = glGetUniformLocation(_shader_prog, "texSize");
    if (u_texSize1 != w || u_texSize2 != h)
        if (uniTexSize>=0) glUniform2f(uniTexSize, u_texSize1=w, u_texSize2=h);

    if (uniVertexTextureBiasX<-1) uniVertexTextureBiasX = glGetUniformLocation (_shader_prog, "vertexTextureBiasX");
    if (uniVertexTextureBiasY<-1) uniVertexTextureBiasY = glGetUniformLocation (_shader_prog, "vertexTextureBiasY");

    if (attribVertex<-1) attribVertex = glGetAttribLocation (_shader_prog, "qt_Vertex");
}


void RenderBlock::
        beginVboRendering(BlockLayout block_size)
{
    GlException_CHECK_ERROR();
    //unsigned meshW = collection->samples_per_block();
    //unsigned meshH = collection->scales_per_block();

    createColorTexture(24); // These will be linearly interpolated when rendering, so a high resolution texture is not needed
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, _colorTexture->getOpenGlTextureId ());
    glActiveTexture(GL_TEXTURE0);


    ShaderSettings shadersettings;
    shadersettings.use_mipmap = render_settings->y_normalize > 0 && Render::BlockTextures::max_level > 0;
    shadersettings.draw3d = !render_settings->draw_flat;
    shadersettings.drawIsarithm = render_settings->draw_contour_plot;
    ShaderData* d=getShader(shadersettings);

    d->prepShader(block_size, render_settings);
    uniModelviewprojection=d->uniModelviewprojection;
    uniModelview=d->uniModelview;
    uniNormalMatrix=d->uniNormalMatrix;
    attribVertex=d->attribVertex;
    uniVertexTextureBiasX=d->uniVertexTextureBiasX;
    uniVertexTextureBiasY=d->uniVertexTextureBiasY;

    GlState::glBindBuffer(GL_ARRAY_BUFFER, *_mesh_position);
    GlState::glEnableVertexAttribArray (attribVertex);
    glVertexAttribPointer (attribVertex, 2, GL_FLOAT, GL_FALSE, 0, 0);

    GlException_CHECK_ERROR();
}


bool RenderBlock::ShaderSettings::operator<(const ShaderSettings& b) const
{
    return memcmp (this,&b,sizeof(ShaderSettings)) < 0;
}


RenderBlock::ShaderData* RenderBlock::
        getShader(ShaderSettings s)
{
    auto i = shaders.find (s);
    if (i==shaders.end ())
    {
        std::string defines;
        if (s.draw3d) defines += "#define DRAW3D\n";
        if (s.drawIsarithm) defines += "#define DRAWISARITHM\n";
        if (s.use_mipmap) defines += "#define USE_MIPMAP\n";
        auto pi = shaders.insert (pair<ShaderSettings,ShaderData>(s, ShaderData(defines.c_str ())));
        i = pi.first;
    }
    return &i->second;
}


void RenderBlock::
        endVboRendering()
{
    GlException_CHECK_ERROR();

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);

    GlState::glBindBuffer (GL_ARRAY_BUFFER, 0);
    GlState::glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, 0);

    GlState::glDisableVertexAttribArray (attribVertex);

    GlException_CHECK_ERROR();
}


void RenderBlock::
        setSize( unsigned w, unsigned h)
{
    if (w == _mesh_width && h == _mesh_height)
        return;

    _mesh_width = w;
    _mesh_height = h;

    // edge dropout to eliminate visible glitches
    if (w > 2 && h > 2)
    {
        for (int i=0;i<subdivs;i++)
            createMeshIndexBuffer(w, h, _mesh_index_buffer[i], 1<<i, 1<<i);
    }
    else
    {
        createMeshIndexBuffer(w, h, _mesh_index_buffer[0], 1, 1);
        for (int i=1;i<subdivs;i++)
            _mesh_index_buffer[i] = _mesh_index_buffer[0];
    }

    createMeshPositionVBO(w, h);
}


unsigned RenderBlock::
        trianglesPerBlock()
{
    return (_mesh_width-1) * (_mesh_height-1) * 2;
}


// create index buffer for rendering quad mesh
void RenderBlock::
        createMeshIndexBuffer(int w, int h, pVbo& vbo, int stepx, int stepy)
{
    GlException_CHECK_ERROR();


    int n_vertices;
    if (h>2 || w>2)
    {
        h+=2;
        w+=2;

        n_vertices = (int_div_ceil (w-2, stepx)*2 + 8)*
                     (int_div_ceil (h-2, stepy)+1);
    }
    else
    {
        EXCEPTION_ASSERT_EQUALS(h,2);
        EXCEPTION_ASSERT_EQUALS(w,2);
        n_vertices = 4;
    }

    vector<BLOCKindexType> indicesdata(n_vertices);
    BLOCKindexType *indices = &indicesdata[0];

    if (h==2 && w==2) {
        *indices++ = 0;
        *indices++ = 1;
        *indices++ = 2;
        *indices++ = 3;
    }
    else
    {
        int y, y2;
        for(int iy=1-stepy; iy<h-1; iy+=stepy) {
            if (iy<1) {
                y = 0;
                y2 = 1;
            } else {
                y = max(0,iy);
                y2 = min(y + stepy,h-1);
            }

            *indices++ = y*w;
            *indices++ = y*w;
            *indices++ = y2*w;

            for(int x=1; x<w+stepx-1; x+=stepx) {
                if (x>=w) x=w-1;
                *indices++ = y*w+x;
                *indices++ = y2*w+x;
            }

            // start new strip with degenerate triangle
            *indices++ = y2*w+(w-1);
            *indices++ = y2*w;
            *indices++ = y2*w;
        }
    }

    EXCEPTION_ASSERT_EQUALS(indices-&indicesdata[0], n_vertices);

    // fill with indices for rendering mesh as triangle strips
    vbo.reset (new Vbo(n_vertices*sizeof(BLOCKindexType), GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW, &indicesdata[0]));

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

    vector<float> posdata( (x2-x1)*(y2-y1)*2 );
    float *pos = &posdata[0];

    for(int y=y1; y<y2; y++) {
        for(int x=x1; x<x2; x++) {
            float u = x / (float) (w-1);
            float v = y / (float) (h-1);
            *pos++ = u;
            *pos++ = v;
        }
    }

    _mesh_position.reset( new Vbo( (w+2)*(h+2)*2*sizeof(float), GL_ARRAY_BUFFER, GL_STATIC_DRAW, &posdata[0] ));
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
    case RenderSettings::ColorMode_WhiteBlackGray:
        spectrum[0] = tvector<4,float>( 1, 1, 1, 0 );
        spectrum[1] = tvector<4,float>( 1, 1, 1, 0 );
        spectrum[2] = tvector<4,float>( 0, 0, 0, 0 );
        spectrum[3] = tvector<4,float>( 0, 0, 0, 0 );
        spectrum[4] = tvector<4,float>( 0, 0, 0, 0 );
        spectrum[5] = tvector<4,float>( 0.5, 0.5, 0.5, 0 );
        spectrum[6] = tvector<4,float>( 0.5, 0.5, 0.5, 0 );
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
    rgb[3] = 1.0;
    //TaskInfo("%g %g %g: %g %g %g %g", f, t, s, rgb[0], rgb[1], rgb[2], rgb[3]);
    return rgb;
}


void RenderBlock::
        createColorTexture(unsigned N)
{
    if (_color_texture_colors == render_settings->color_mode && _colorTexture && _colorTexture->getWidth()==N)
        return;

    _color_texture_colors = render_settings->color_mode;

    vector<tvector<4,unsigned char> > texture(N);
    for (unsigned i=0; i<N; ++i) {
        tvector<4,float> wc = getWavelengthColorCompute( i/(float)(N-1), _color_texture_colors );
        // Make UNSIGNED_BYTE. gles can't interpolate GL_FLOAT
        tvector<4,unsigned char> wb;
        wb[0] = 255*wc[0];
        wb[1] = 255*wc[1];
        wb[2] = 255*wc[2];
        wb[3] = 255*wc[3];
        texture[i] = wb;
//        Log("%d/%d = %g %g %g %g") % i % N % wc[0] % wc[1] % wc[2] % wc[3];
    }

    _colorTexture.reset( new GlTexture(N,1, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, &texture[0][0]));
    render_settings->clear_color = getWavelengthColorCompute( -1.f, _color_texture_colors );
}

} // namespace Render
} // namespace Heightmap
