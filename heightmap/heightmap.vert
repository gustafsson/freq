// GLSL vertex shader
varying vec3 eyeSpacePos;
varying vec3 worldSpaceNormal;
varying vec3 eyeSpaceNormal;
varying float vertex_height;
/*uniform float heightScale; // = 0.5;
//uniform vec2  size;        // = vec2(256.0, 256.0);*/

//const float heightScale = 1.0; // 0.0125;
//const vec2  size = vec2(512.0, 128.0);
const vec2  sizeinv = vec2(0.0019531, 0.0078125);

uniform sampler2D tex;
//uniform sampler2D tex_slope;
uniform float yScale;
uniform vec2 scale_tex;
uniform vec2 offset_tex;

void main()
{
    // We want linear interpolation all the way out to the edge
    gl_TexCoord[0].xy = gl_Vertex.xz*scale_tex+offset_tex;

    float height     = texture2D(tex, gl_TexCoord[0].xy).x;
    float heightx     = texture2D(tex, gl_TexCoord[0].xy + vec2(offset_tex.x,0)).x;
    float heighty     = texture2D(tex, gl_TexCoord[0].xy + vec2(0,offset_tex.y)).x;
    vec2 slope       = vec2(heightx-height, heighty-height)*(yScale*10000.0);
    //vec2 slope       = texture2D(tex_slope, gl_TexCoord[0].xy).xw;
    //vec2 slope = vec2(0,0);
    height *= yScale;

    // calculate surface normal from slope for shading
    worldSpaceNormal = cross( vec3(0.0,             slope.y, 2.0 * sizeinv.x),
                              vec3(2.0 * sizeinv.y, slope.x, 0.0));

    // calculate position and transform to homogeneous clip space
    vec4 pos         = vec4(gl_Vertex.x, height, gl_Vertex.z, 1.0);
    gl_Position      = gl_ModelViewProjectionMatrix * pos;
    
    eyeSpacePos      = (gl_ModelViewMatrix * pos).xyz;
    eyeSpaceNormal   = (gl_NormalMatrix * worldSpaceNormal).xyz;

    //eyeSpacePos      = normalize(eyeSpacePos);
    //eyeSpaceNormal   = normalize(eyeSpaceNormal);
    //worldSpaceNormal = normalize(worldSpaceNormal);
    vertex_height = height;
}
