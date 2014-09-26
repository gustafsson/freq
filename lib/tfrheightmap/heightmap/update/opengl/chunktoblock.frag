// GLSL fragment shader
varying highp vec2 qt_TexCoord0;
uniform highp sampler2D mytex;
uniform highp float normalization;
uniform lowp int amplitude_axis;
uniform highp vec2 data_size;
uniform highp vec2 tex_size;

void main()
{
    mediump float a = 0.0;
    highp float stepx = fwidth(qt_TexCoord0.s)*data_size.x;
    highp vec2 uvd = qt_TexCoord0.xy * data_size;

    if (stepx < 1.0)
        stepx = 1.0;

    // fetch an integer number of samples centered around uv.x
    // multiples of 0.5 are represented exactly for small floats
    stepx = 0.5*floor(stepx-0.5);
    highp float x;
    for (x=-stepx; x<=stepx; ++x)
    {
        highp vec2 uv = vec2(uvd.x + x, uvd.y);

        // Compute texture position that will make a linear lookup
        // interpolate the right texels
        uv = (uv + 0.5) / tex_size;

        mediump float r = texture2D(mytex, uv).x;
        a = max(a, r);
    }

    if (0==amplitude_axis)
        a = 25.0*sqrt(a*normalization);
    if (1==amplitude_axis) {
        a = 0.5 * 0.019 * log2(a*normalization) + 0.3333;
        a = max(0.0, a);
    }

    //a = mod(gl_FragCoord.y,2.0); // this shows that more than necessary is replaced by some updates
    //a = fwidth(qt_TexCoord0.s);

    gl_FragColor = vec4(a, 0, 0, 1);
}
