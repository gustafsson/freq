#include "GL/glew.h"
#include "spectrogram-vbo.h"
#include <stdio.h>
#include "spectrogram.h"
#include "spectrogram-renderer.h"

// Helpers from Cuda SDK sample, ocean FFT

// Attach shader to a program
int attachShader(GLuint prg, GLenum type, const char *name)
{
    GLuint shader;
    FILE * fp;
    int size, compiled;
    char * src;

    fp = fopen(name, "rb");
    if (!fp) return 0;

    fseek(fp, 0, SEEK_END);
    size = ftell(fp);
    src = (char*)malloc(size);

    fseek(fp, 0, SEEK_SET);
    fread(src, sizeof(char), size, fp);
    fclose(fp);

    shader = glCreateShader(type);
    glShaderSource(shader, 1, (const char**)&src, (const GLint*)&size);
    glCompileShader(shader);
    glGetShaderiv(shader, GL_COMPILE_STATUS, (GLint*)&compiled);
    if (!compiled) {
        char log[2048];
        int len;

        glGetShaderInfoLog(shader, 2048, (GLsizei*)&len, log);
        printf("Info log: %s\n", log);
        glDeleteShader(shader);
        return 0;
    }
    free(src);

    glAttachShader(prg, shader);
    glDeleteShader(shader);

    return 1;
}

// Create shader program from vertex shader and fragment shader files
GLuint loadGLSLProgram(const char *vertFileName, const char *fragFileName)
{
    GLint linked;
    GLuint program;

    program = glCreateProgram();
    if (!attachShader(program, GL_VERTEX_SHADER, vertFileName)) {
        glDeleteProgram(program);
        fprintf(stderr, "Couldn't attach vertex shader from file %s\n", vertFileName);
        return 0;
    }

    if (!attachShader(program, GL_FRAGMENT_SHADER, fragFileName)) {
        glDeleteProgram(program);
        fprintf(stderr, "Couldn't attach fragment shader from file %s\n", fragFileName);
        return 0;
    }

    glLinkProgram(program);
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) {
        glDeleteProgram(program);
        char temp[256];
        glGetProgramInfoLog(program, 256, 0, temp);
        fprintf(stderr, "Failed to link program: %s\n", temp);
        return 0;
    }

    return program;
}


Vbo::Vbo(size_t size)
:   _sz(size),
    _vbo(0)
{
    // create buffer object
    glGenBuffers(1, &_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

Vbo::~Vbo()
{
    if (_vbo)
        glDeleteBuffers(1, &_vbo);
}


SpectrogramVbo::SpectrogramVbo( Spectrogram* spectrogram )
:   _spectrogram( spectrogram ),
    _height( new Vbo(spectrogram->samples_per_block()*spectrogram->scales_per_block()*sizeof(float)) ),
    _slope( new Vbo(spectrogram->samples_per_block()*spectrogram->scales_per_block()*sizeof(float2)) )
{
    //_renderer->setSize(renderer->spectrogram()->samples_per_block(), renderer->spectrogram()->scales_per_block());
    cudaGLRegisterBufferObject(*_height);
    cudaGLRegisterBufferObject(*_slope);
}

SpectrogramVbo::~SpectrogramVbo() {
    cudaGLUnregisterBufferObject(*_height);
    cudaGLUnregisterBufferObject(*_slope);
}

SpectrogramVbo::pHeight SpectrogramVbo::height() {
    return pHeight(new MappedVbo<float>(_height, make_cudaExtent(
            _spectrogram->samples_per_block(),
            _spectrogram->scales_per_block(), 1)));
}

SpectrogramVbo::pSlope SpectrogramVbo::slope() {
    return pSlope(new MappedVbo<float2>(_slope, make_cudaExtent(
            _spectrogram->samples_per_block(),
            _spectrogram->scales_per_block(), 1)));
}

void SpectrogramVbo::draw() {
    unsigned meshW = _spectrogram->samples_per_block();
    unsigned meshH = _spectrogram->scales_per_block();

    glBindBuffer(GL_ARRAY_BUFFER, *_height);
    glClientActiveTexture(GL_TEXTURE0);
    glTexCoordPointer(1, GL_FLOAT, 0, 0);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, *_slope);
    glClientActiveTexture(GL_TEXTURE1);
    glTexCoordPointer(2, GL_FLOAT, 0, 0);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);




    // bool wireFrame = false;
    bool drawPoints = false;

    glColor3f(1.0, 1.0, 1.0);
    if (drawPoints) {
        glDrawArrays(GL_POINTS, 0, meshW * meshH);
    } else {
//        glPolygonMode(GL_FRONT_AND_BACK, wireFrame ? GL_LINE : GL_FILL);
            glDrawElements(GL_TRIANGLE_STRIP, ((meshW*2)+2)*(meshH-1), GL_UNSIGNED_INT, 0);
//        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    glClientActiveTexture(GL_TEXTURE0);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glClientActiveTexture(GL_TEXTURE1);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
}

int clamp(int val, int max) {
    if (val<0) return 0;
    if (val>max) return max;
    return val;
}

static void setWavelengthColor( float wavelengthScalar ) {
    const float spectrum[][3] = {
        { 1, 0, 1 },
        { 0, 0, 1 },
        { 0, 1, 1 },
        { 0, 1, 0 },
        { 1, 1, 0 },
        { 1, 0, 0 }};

    unsigned count = sizeof(spectrum)/sizeof(spectrum[0]);
    float f = count*wavelengthScalar;
    unsigned i = clamp(f, count-1);
    unsigned j = clamp(f+1, count-1);
    float t = f-i;

    GLfloat rgb[] = {  spectrum[i][0]*(1-t) + spectrum[j][0]*t,
                       spectrum[i][1]*(1-t) + spectrum[j][1]*t,
                       spectrum[i][2]*(1-t) + spectrum[j][2]*t
                   };
    glColor3fv( rgb );
}

void SpectrogramVbo::draw_directMode( )
{
    pHeight block = height();
    cudaExtent n = block->data->getNumberOfElements();
    const float* data = block->data->getCpuMemory();

    float ifs = 1./(n.width-3);
    float depthScale = 1.f/(n.height-3);

    glEnable(GL_NORMALIZE);
    for (unsigned fi=1; fi<n.height-2; fi++)
    {
        glBegin(GL_TRIANGLE_STRIP);
            float v[3][4] = {{0}}; // vertex buffer needed to compute normals

            for (unsigned t=0; t<n.width; t++)
            {
                for (unsigned dt=0; dt<2; dt++)
                    for (unsigned df=0; df<4; df++)
                        v[dt][df] = v[dt+1][df];

                for (int df=-1; df<3; df++)
                    v[2][df] = data[ t + (fi + df)*n.width ];

                if (2>t) // building v
                    continue;

                // Add two vertices to the triangle strip.
                for (int j=0; j<2; j++)
                {
                    setWavelengthColor( v[1][j+1] );
                    float dt=(v[2][j+1]-v[0][j+1]);
                    float df=(v[1][j+2]-v[1][j+0]);
                    glNormal3f( -dt, 2, -df ); // normalized by OpenGL
                    glVertex3f( ifs*(t-2), v[1][j+1], (fi-1+j)*depthScale);
                }
            }
        glEnd(); // GL_TRIANGLE_STRIP
    }
    glDisable(GL_NORMALIZE);
}
