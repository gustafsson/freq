// GLSL fragment shader

uniform sampler2D mytex;
uniform float normalization;
uniform int amplitude_axis;
uniform vec2 data_size;
uniform vec2 tex_size;

void main()
{
    // Translate normalized index to data index (integers)
    vec2 uv1 = floor((gl_TexCoord[0].st - gl_TexCoord[0].pq) * data_size);
    vec2 uv2 = floor((gl_TexCoord[0].st + gl_TexCoord[0].pq) * data_size);
    // Compute neighbouring indices as well, and their distance
    vec2 f = (gl_TexCoord[0].st - gl_TexCoord[0].pq) * data_size - uv1;
    float a = 0.0;
    vec2 uv;
    for (uv.x=floor(uv1.x); uv.x<=ceil(uv2.x); ++uv.x) {
        for (uv.y=floor(uv1.y); uv.y<=ceil(uv2.y); ++uv.y)
        {
            // Compute linear index common for data and degenerate texture
            // With IEEE-754 single floats 'i' is an exact integer up to 16 million.
            // But that's only guaranteed in GLSL 1.30 and above.
            float i = uv.x + uv.y*data_size.x;

            // Compute degenerate texel index (integers)
            float v = floor(i / tex_size.x);
            float u = mod(i, tex_size.x);

            // Compute texture position that will make a nearest lookup
            // find the right texel in degenerate texture
            u = (u + 0.5) / tex_size.x;
            v = (v + 0.5) / tex_size.y;

            // Linear interpolation on edge
//            float f1 = max(0, uv1 - uv);
//            float f2 = max(0, uv - uv2);
            float f = texture2D(mytex, vec2(u, v)).r;
//            a = max(a, (1.0-f1)*(1.0-f2)*f);
            a = max(a, f);
        }
    }

    if (0==amplitude_axis)
        a = normalization*25.0*sqrt(a);
    if (1==amplitude_axis) {
        a = 0.5*0.019 * log2(a*normalization*normalization) + 0.3333;
        a = a < 0.0 ? 0.0 : a;
    }

    gl_FragColor = vec4(a, 0.0, 0.0, 1.0);
}
