uniform mediump sampler2D qt_Texture0;
varying mediump vec2 qt_TexCoord0;

void main(void)
{
    mediump float dx = dFdx(qt_TexCoord0.x)*0.25;
    mediump float dy = dFdx(qt_TexCoord0.y)*0.25;
    mediump vec2 t = qt_TexCoord0;
    mediump vec4 v = vec4(
                    texture2D(qt_Texture0, vec2(t.x-dx, t.y-dy), -1.0).x,
                    texture2D(qt_Texture0, vec2(t.x-dx, t.y+dy), -1.0).x,
                    texture2D(qt_Texture0, vec2(t.x+dx, t.y-dy), -1.0).x,
                    texture2D(qt_Texture0, vec2(t.x+dx, t.y+dy), -1.0).x);

    mediump float r = max(max(v.x,v.y),max(v.z,v.w));
    gl_FragColor = vec4(r, 0.0, 0.0, 1.0);
}
