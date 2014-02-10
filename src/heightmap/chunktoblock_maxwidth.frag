// GLSL fragment shader

uniform sampler2D mytex;
uniform float normalization;
uniform int amplitude_axis;
uniform vec2 data_size;
uniform vec2 tex_size;

void main()
{
    // Translate normalized index to data index (integers)
    vec2 uv = floor(gl_TexCoord[0].st * data_size);
    // Compute neighbouring indices as well, and their distance
    vec2 f = gl_TexCoord[0].st * data_size - uv;
    vec4 u = vec4(uv.x, uv.x+1.0, uv.x, uv.x+1.0);
    vec4 v = vec4(uv.y, uv.y, uv.y+1.0, uv.y+1.0);

    // Compute degenerate texel index (integers)
    v = v + floor( u/tex_size.x )*data_size.y;
    u = mod( u, tex_size.x );

    // Compute texture position that will make a nearest lookup
    // find the right texel in degenerate texture
    u = (u + 0.5) / tex_size.x;
    v = (v + 0.5) / tex_size.y;

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
        a = max(0.0, a);
    }

    gl_FragColor = vec4(a, 0.0, 0.0, 1.0);
}
