// GLSL fragment shader

uniform sampler2D mytex;
uniform float normalization;
uniform int amplitude_axis;
uniform vec2 data_size;
uniform vec2 tex_size;

void main()
{
    float a = 0.0;
    // Only step.x is used. Assuming dFdx(t)==0 and dFdy(s)==0.
    vec2 step = 0.5*abs(vec2(dFdx(gl_TexCoord[0].s), dFdy(gl_TexCoord[0].t)));
    vec2 uvd = gl_TexCoord[0].st * data_size;

    if (step.x < 0.5)
        step.x = 0.0;

    // fetch an integer number of samples centered around uv.x
    // multiples of 0.5 are represented exactly for small floats
    step.x = 0.5*floor(2.0*step.x+0.5);
    for (float x=-step.x; x<=step.x; ++x)
    {
        vec2 uv = vec2(uvd.x + x, uvd.y);

        // Compute degenerate texel index (float)
        uv.y = uv.y + floor( uv.x/(tex_size.x-1.0) )*data_size.y;
        uv.x = mod( uv.x, tex_size.x-1.0 );

        // Compute texture position that will make a linear lookup
        // interpolate the right texels
        uv = (uv + 0.5) / tex_size;

        float r = texture2D(mytex, vec2(uv.x, uv.y)).r;
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
