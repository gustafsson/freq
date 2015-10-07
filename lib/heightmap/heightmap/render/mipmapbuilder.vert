attribute highp vec4 qt_Vertex;
attribute mediump vec2 qt_MultiTexCoord0;

uniform mediump vec2 subtexeloffset;

varying mediump vec2 qt_TexCoord0;
varying mediump vec2 qt_TexCoord1;
varying mediump vec2 qt_TexCoord2;
varying mediump vec2 qt_TexCoord3;

void main(void)
{
    gl_Position = qt_Vertex;

    vec2 t = qt_MultiTexCoord0;
    qt_TexCoord0 = t - subtexeloffset;
    qt_TexCoord1 = t - vec2(subtexeloffset.x,-subtexeloffset.y);
    qt_TexCoord2 = t + vec2(subtexeloffset.x,-subtexeloffset.y);
    qt_TexCoord3 = t + subtexeloffset;
}
