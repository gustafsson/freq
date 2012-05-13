
// GLSL vertex shader
//varying vec3 eyeSpacePos;
//varying vec3 worldSpaceNormal;
//varying vec3 eyeSpaceNormal;
/*varying float vertex_height;
varying float shadow;

uniform sampler2D tex_nearest;
uniform float yScale;
*/uniform vec2 scale_tex;
uniform vec2 offset_tex;

void main()
{
    // We want linear interpolation all the way out to the edge
    gl_TexCoord[0].xy = gl_Vertex.xz*scale_tex + offset_tex;
/*
    vertex_height       = texture2D(tex_nearest, gl_TexCoord[0].xy).x * yScale;
*/
    // calculate position and transform to homogeneous clip space
    //gl_Position      = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.x, vertex_height, gl_Vertex.z, 1.0);
    gl_Position      = gl_ModelViewProjectionMatrix * gl_Vertex;
/*
    shadow = .5;*/
}
