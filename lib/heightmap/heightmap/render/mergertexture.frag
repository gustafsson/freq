uniform mediump sampler2D qt_Texture0;
uniform mediump vec2 invtexsize;

varying mediump vec2 qt_TexCoord0;

void main(void)
{
    mediump vec2 dt = vec2(dFdx(qt_TexCoord0.x), dFdy(qt_TexCoord0.y));
    dt = abs(dt);
    mediump vec2 t1 = qt_TexCoord0 - 0.5*dt;
    mediump vec2 t2 = qt_TexCoord0 + 0.5*dt;

    // texels are found at: 0.5*invtexsize + i*invtexsize
    // round to nearest texel
    t1 = (floor((t1-0.5*invtexsize)/invtexsize+0.5)+0.5)*invtexsize;
    t2 = (floor((t2-0.5*invtexsize)/invtexsize+0.5)+0.5)*invtexsize;
    //t1 = t1 + 0.5*invtexsize;
    //t2 = t2 + 0.5*invtexsize;

    mediump float v = 0.0;
    mediump vec2 t;
    if (dt.x > invtexsize.x && dt.y > invtexsize.y)
    {
        // minifying both x and y
        for (t.x = t1.x; t.x<=t2.x; t.x+=invtexsize.x)
            for (t.y = t1.y; t.y<=t2.y; t.y+=invtexsize.y)
                v = max(v, texture2DLod(qt_Texture0, t, 0.0).x);
    }
    else if (dt.x > invtexsize.x)
    {
        // minifying x but not y
        t.y = qt_TexCoord0.y;
        for (t.x = t1.x; t.x<=t2.x; t.x+=invtexsize.x)
            v = max(v, texture2DLod(qt_Texture0, t, 0.0).x);
    }
    else if (dt.y > invtexsize.y)
    {
        // minifying y but not x
        t.x = qt_TexCoord0.x;
        for (t.y = t1.y; t.y<=t2.y; t.y+=invtexsize.y)
            v = max(v, texture2DLod(qt_Texture0, t, 0.0).x);
    }
    else
    {
        // magnifying or equal
        v = texture2DLod(qt_Texture0, qt_TexCoord0, 0.0).x;
    }


    gl_FragColor = vec4(v, 0.0, 0.0, 1.0);
}
