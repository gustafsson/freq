// GLSL vertex shader
varying float vertex_height;
varying float shadow;

uniform vec2 scale_tex;
uniform float yScale;
uniform vec2 offset_tex;

void main()
{
    // We want linear interpolation all the way out to the edge
    vec2 vertex = clamp(gl_Vertex.xz, 0.0, 1.0);
    vec2 tex = vertex*scale_tex + offset_tex;
    gl_TexCoord[0].xy = tex;
    vec4 vert_new = gl_Vertex;
    vert_new.y*=yScale;

    // calculate position and transform to homogeneous clip space
    gl_Position      = gl_ModelViewProjectionMatrix * vert_new;

    shadow = .5;
    vertex_height = .0;
}
