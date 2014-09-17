// GLSL vertex shader
attribute highp vec4 qt_Vertex;
varying highp float vertex_height;
varying mediump float shadow;
varying highp vec2 texCoord;

uniform highp sampler2D tex;
uniform mediump float flatness;
uniform mediump float yScale;
uniform mediump float yOffset;
uniform mediump vec3 logScale;
uniform mediump vec2 scale_tex;
uniform mediump vec2 offset_tex;
uniform highp mat4 ModelViewProjectionMatrix;
uniform mediump mat4 ModelViewMatrix;
uniform mediump mat4 NormalMatrix;

mediump float heightValue(mediump float v) {
    // the linear case is straightforward
    mediump float h = mix(v*yScale + yOffset,
                  log(v) * logScale.y + logScale.z,
                  logScale.x);

    h *= flatness;

    return v == 0.0 ? 0.0 : max(0.01, h);
}

void main()
{
    // We want linear interpolation all the way out to the edge
    mediump vec2 vertex = clamp(qt_Vertex.xz, 0.0, 1.0);
    mediump vec2 tex0 = vertex*scale_tex + offset_tex;

    texCoord = tex0;

    mediump float height       = texture2D(tex, tex0).x;

    height = heightValue(height);

    mediump vec4 pos         = vec4(vertex.x, height, vertex.y, 1.0);

    // transform to homogeneous clip space
    gl_Position      = ModelViewProjectionMatrix * pos;
    vertex_height = height;

    shadow = 1.0;
}
