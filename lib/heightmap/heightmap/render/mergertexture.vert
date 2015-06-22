attribute highp vec4 qt_Vertex;
attribute mediump vec2 qt_MultiTexCoord0;

uniform highp mat4 qt_ModelViewMatrix;
uniform highp mat4 qt_ProjectionMatrix;

varying mediump vec2 qt_TexCoord0;

void main(void)
{
    gl_Position = qt_ProjectionMatrix * qt_ModelViewMatrix * qt_Vertex;
    qt_TexCoord0 = qt_MultiTexCoord0;
}
