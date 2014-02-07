// GLSL fragment shader

uniform sampler2D mytex;
uniform float normalization;
uniform int amplitude_axis;
uniform vec2 data_size;
uniform vec2 tex_size;

void main()
{
    float a = 0.0;

    float xstep = gl_TexCoord[0].p;
    if (xstep < 0.5 / data_size.x)
    {
        // Translate normalized index to data index (integers)
        vec2 uv = floor(gl_TexCoord[0].st * data_size);
        // Compute neighbouring indices as well, and their distance
        vec2 f = gl_TexCoord[0].st * data_size - uv;
        vec4 u = min(vec4(uv.x, uv.x+1.0, uv.x, uv.x+1.0), data_size.x);
        vec4 v = min(vec4(uv.y, uv.y, uv.y+1.0, uv.y+1.0), data_size.y);

        // Compute linear index common for data and degenerate texture
        // With IEEE-754 single floats 'i' is an exact integer up to 16 million.
        // But that's only guaranteed in GLSL 1.30 and above.
        vec4 i = u + v*data_size.x;

        // Compute degenerate texel index (integers)
        v = floor(i / tex_size.x);
        u = mod(i, tex_size.x);

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
        a = mix(r.x, r.y, f.y);
    }
    else
    {
        // Translate normalized index to data index (integers)
        vec2 xrange = (gl_TexCoord[0].xx + gl_TexCoord[0].p*vec2(-1.0,1.0)) * data_size.x;
        xrange = clamp(floor(xrange), 0.0, data_size.x);
        float iy = floor(gl_TexCoord[0].y * data_size.y);
        float fy = gl_TexCoord[0].y * data_size.y - iy;
        vec2 y = clamp(vec2(iy, iy + 1.0), 0.0, data_size.y-1.0);

        // Assume the primary resolution has the highest resolution and only implement max along that.
        // Interpolate the other.

        vec2 av = vec2(0.0, 0.0);
        for (float x=xrange.x; x<xrange.y; ++x)
        {
            // Compute linear index common for data and degenerate texture
            // With IEEE-754 single floats 'i' is an exact integer up to 16 million.
            // But that's only guaranteed in GLSL 1.30 and above.
            vec2 i = x + y*data_size.x;

            // Compute degenerate texel index (integers)
            vec2 v = floor(i / tex_size.x);
            vec2 u = mod(i, tex_size.x);

            // Compute texture position that will make a nearest lookup
            // find the right texel in degenerate texture
            u = (u + 0.5) / tex_size.x;
            v = (v + 0.5) / tex_size.y;

            vec2 t = vec2(texture2D(mytex, vec2(u.x, v.x)).r,
                          texture2D(mytex, vec2(u.y, v.y)).r);
            av = max(av, t);
        }

        a = mix(av.x, av.y, fy);
    }

    if (0==amplitude_axis)
        a = normalization*25.0*sqrt(a);
    if (1==amplitude_axis) {
        a = 0.5*0.019 * log2(a*normalization*normalization) + 0.3333;
        a = a < 0.0 ? 0.0 : a;
    }

    gl_FragColor = vec4(a, 0.0, 0.0, 1.0);
}


