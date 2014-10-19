uniform mediump sampler2D qt_Texture0;
varying mediump vec2 qt_TexCoord0;
uniform mediump vec2 invtexsize;

void main(void)
{
    mediump vec2 dt = vec2(dFdx(qt_TexCoord0.x), dFdy(qt_TexCoord0.y));
    dt = 0.5*abs(dt);
    mediump vec2 t1 = qt_TexCoord0 - dt;
    mediump vec2 t2 = qt_TexCoord0 + dt;

    mediump float v = 0.0;
    mediump vec2 t;
    for (t.x = t1.x; t.x<t2.x; t.x+=invtexsize.x)
        for (t.y = t1.y; t.y<t2.y; t.y+=invtexsize.y)
            v = max(v, texture2D(qt_Texture0, t).x);

    gl_FragColor = vec4(v, 0.0, 0.0, 1.0);
}
