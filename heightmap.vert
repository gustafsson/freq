// GLSL vertex shader
varying vec3 eyeSpacePos;
varying vec3 worldSpaceNormal;
varying vec3 eyeSpaceNormal;
/*uniform float heightScale; // = 0.5;
uniform vec2  size;        // = vec2(256.0, 256.0);*/

const float heightScale = 1.0; // 0.0125;
const vec2  size = vec2(128.0, 256.0);

uniform sampler2D tex;
uniform sampler2D tex_slope;

void main()
{
    gl_TexCoord[0].xy= gl_Vertex.xz*vec2(1,1);
    float height     = texture2D(tex, gl_TexCoord[0].xy).x;
    vec2 slope       = texture2D(tex_slope, gl_TexCoord[0].xy).xy;

    // calculate surface normal from slope for shading
    worldSpaceNormal = cross( vec3(0.0, slope.y*heightScale, 2.0 / size.x), vec3(2.0 / size.y, slope.x*heightScale, 0.0));

    // calculate position and transform to homogeneous clip space
    vec4 pos         = vec4(gl_Vertex.x, height, gl_Vertex.z, 1.0);
    gl_Position      = gl_ModelViewProjectionMatrix * pos;
    
    eyeSpacePos      = (gl_ModelViewMatrix * pos).xyz;
    eyeSpaceNormal   = (gl_NormalMatrix * worldSpaceNormal).xyz;

    eyeSpacePos      = normalize(eyeSpacePos);
    eyeSpaceNormal   = normalize(eyeSpaceNormal);
    worldSpaceNormal = normalize(worldSpaceNormal);
}
