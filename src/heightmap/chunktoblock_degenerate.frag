// GLSL fragment shader

uniform sampler2D mytex;
uniform float normalization;
uniform int amplitude_axis;
uniform vec2 data_size;
uniform vec2 tex_size;

void main()
{
    // Translate to data texel corner
    vec2 uv = floor(gl_TexCoord[0].st * data_size);
    vec2 f = gl_TexCoord[0].st * data_size - uv;
    vec4 u = vec4(uv.x, uv.x+1.0, uv.x, uv.x+1.0);
    vec4 v = vec4(uv.y, uv.y, uv.y+1.0, uv.y+1.0);

    // Compute index of data texel
    // With IEEE-754 single floats 'i' is an exact integer up to 16 million.
    // But that's only guaranteed in GLSL 1.30 and above.
    vec4 i = u + v*data_size.x;

    // Compute degenerate texel coorner
    v = floor(i/tex_size.x);
    u = mod(i,tex_size.x);

    // Compute normalized texture coordinates in degenerate texture
    u /= (tex_size.x - 1.0);
    v /= (tex_size.y - 1.0);

    vec4 r = vec4(
                texture2D(mytex, vec2(u.s, v.s)).r,
                texture2D(mytex, vec2(u.t, v.t)).r,
                texture2D(mytex, vec2(u.p, v.p)).r,
                texture2D(mytex, vec2(u.q, v.q)).r );

    r.xy = mix(r.xz, r.yw, f.x);
    float a = mix(r.x, r.y, f.y);

    if (0==amplitude_axis)
        a = normalization*25.0*sqrt(a);
    if (1==amplitude_axis) {
        a = 0.5*0.019 * log2(a*normalization*normalization) + 0.3333;
        a = a < 0.0 ? 0.0 : a;
    }

    gl_FragColor = vec4(a, 0, 0, 1);
}
