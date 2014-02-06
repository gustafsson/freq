// GLSL fragment shader

uniform sampler2D mytex;
uniform float normalization;
uniform int amplitude_axis;
uniform vec2 data_size;
uniform vec2 tex_size;

void main()
{
    vec2 uv = gl_TexCoord[0].st;
    // Translate to data texel corner
    uv = floor(uv * data_size);

    // Compute index of data texel
    // With IEEE-754 single floats 'i' is an exact integer up to 16 million.
    // But that's only guaranteed in GLSL 1.30 and above.
    float i = uv.x + uv.y*data_size.x;

    // Compute degenerate texel coorner
    i /= tex_size.x;
    uv.y = floor(i);
    uv.x = (i - uv.y)*tex_size.x;

    // Compute normalized texture coordinates in degenerate texture
    uv.x /= (tex_size.x - 1.0);
    uv.y /= (tex_size.y - 1.0);

    float v = texture2D(mytex, uv).r;

    if (0==amplitude_axis)
        v = normalization*25.0*sqrt(v);
    if (1==amplitude_axis) {
        v = 0.5*0.019 * log2(v*normalization*normalization) + 0.3333;
        v = v < 0.0 ? 0.0 : v;
    }

    gl_FragColor = vec4(v, 0, 0, 1);
}
