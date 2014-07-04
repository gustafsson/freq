uniform sampler2D tex;

void main(void)
{
    vec2 dt = 0.25*fwidth(gl_TexCoord[0].xy);
    vec2 x1 = round(gl_TexCoord[0].xy - dt);
    vec2 x2 = round(gl_TexCoord[0].xy + dt);

    float v = 0;
    vec2 t;
    for (t.x = x1.x; t.x<=x2.x; t.x++)
        for (t.y = x1.y; t.y<=x2.y; t.y++)
            v = max(v, texture2D(tex, t).x);

    gl_FragColor = vec4(v,0,0,0);
}
