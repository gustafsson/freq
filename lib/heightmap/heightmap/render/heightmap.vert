// GLSL vertex shader
//varying vec3 eyeSpacePos;
//varying vec3 worldSpaceNormal;
//varying vec3 eyeSpaceNormal;
attribute vec4 qt_Vertex;
varying float vertex_height;
varying float shadow;
varying vec2 texCoord;

uniform sampler2D tex;
uniform float flatness;
uniform float yScale;
uniform float yOffset;
uniform vec3 logScale;
uniform vec2 scale_tex;
uniform vec2 offset_tex;
uniform mat4 ModelViewProjectionMatrix;
uniform mat4 ModelViewMatrix;
uniform mat4 NormalMatrix;

float heightValue(float v) {
    // the linear case is straightforward
    float h = mix(v*yScale + yOffset,
                  log(v) * logScale.y + logScale.z,
                  logScale.x);

    h *= flatness;

    return v == 0.0 ? 0.0 : max(0.01, h);
}


void main()
{
    // We want linear interpolation all the way out to the edge
    vec2 vertex = clamp(qt_Vertex.xz, 0.0, 1.0);
    vec2 tex0 = vertex*scale_tex + offset_tex;

    texCoord = tex0;

    vec2 tex1 = max(tex0 - offset_tex*2.0, offset_tex);
    vec2 tex2 = min(tex0 + offset_tex*2.0, 1.0-offset_tex);

    float height       = texture2D(tex, tex0).x;
    float heightx1     = texture2D(tex, vec2(tex1.x, tex0.y)).x;
    float heightx2     = texture2D(tex, vec2(tex2.x, tex0.y)).x;
    float heighty1     = texture2D(tex, vec2(tex0.x, tex1.y)).x;
    float heighty2     = texture2D(tex, vec2(tex0.x, tex2.y)).x;

    height = heightValue(height);
    heightx1 = heightValue(heightx1);
    heightx2 = heightValue(heightx2);
    heighty1 = heightValue(heighty1);
    heighty2 = heightValue(heighty2);

    vec2 slope       = vec2(heightx2-heightx1, heighty2-heighty1);

    // calculate surface normal from slope for shading
    vec4 worldSpaceNormal;
    worldSpaceNormal.xyz = cross( vec3(0.0,            slope.y, tex2.y-tex1.y),
                                   vec3(tex2.x-tex1.x,  slope.x, 0.0));
    worldSpaceNormal.w = 1.0;

    vec4 pos         = vec4(vertex.x, height, vertex.y, 1.0);

    // transform to homogeneous clip space
    gl_Position      = ModelViewProjectionMatrix * pos;

    vec3 eyeSpacePos      = (ModelViewMatrix * pos).xyz;
    vec3 eyeSpaceNormal   = (NormalMatrix * worldSpaceNormal).xyz;

    eyeSpacePos      = normalize(eyeSpacePos);
    eyeSpaceNormal   = normalize(eyeSpaceNormal);
    worldSpaceNormal = normalize(worldSpaceNormal);

    float facing    = max(0.0, dot(eyeSpaceNormal, -eyeSpacePos));
    float diffuse   = max(0.0, worldSpaceNormal.y); // max(0.0, dot(worldSpaceNormalVector, lightDir));

    //shadow = clamp( 0.5 + diffuse+facing + fresnel, 0.5, 1.0);
    shadow = min( 0.5 + (diffuse+facing)*0.5, 1.0);

    vertex_height = height;
}
