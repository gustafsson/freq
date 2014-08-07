// GLSL fragment shader
varying lowp vec2 qt_TexCoord0;
uniform highp sampler2D mytex;
uniform highp float normalization;
uniform lowp int amplitude_axis;
uniform lowp vec2 data_size;
uniform lowp vec2 tex_size;

void main()
{
    mediump float a = 0.0;
    mediump float stepx = fwidth(qt_TexCoord0.s)*data_size.x;
    mediump vec2 uvd = qt_TexCoord0.st * data_size;

    if (stepx < 1.0)
        stepx = 1.0;

    // fetch an integer number of samples centered around uv.x
    // multiples of 0.5 are represented exactly for small floats
    stepx = 0.5*floor(stepx-0.5);
    for (mediump float x=-stepx; x<=stepx; ++x)
    {
        mediump vec2 uv = vec2(uvd.x + x, uvd.y);

        // Compute degenerate texel index (float)
        uv.x = uv.x + floor( uv.y/(tex_size.y-1.0) )*data_size.x;
        uv.y = mod( uv.y, tex_size.y-1.0 );

        // Compute texture position that will make a linear lookup
        // interpolate the right texels
        uv = (uv + 0.5) / tex_size;

        mediump float r = texture2D(mytex, vec2(uv.x, uv.y)).r;
        a = max(a, r);
    }

    if (0==amplitude_axis)
        a = normalization*25.0*sqrt(a);
    if (1==amplitude_axis) {
        a = 0.5 * 0.019 * log2(a*normalization*normalization) + 0.3333;
        a = max(0.0, a);
    }

    gl_FragColor = vec4(a, 0, 0, 1);
}
