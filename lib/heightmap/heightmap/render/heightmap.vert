// GLSL vertex shader
//varying vec3 eyeSpacePos;
//varying vec3 worldSpaceNormal;
//varying vec3 eyeSpaceNormal;
varying float vertex_height;
varying float shadow;

uniform sampler2D tex_nearest;
uniform float yScale;
uniform float yOffset;
uniform float logScale;
uniform vec2 scale_tex;
uniform vec2 offset_tex;

float heightValue(float v) {
    float a = v == 0.0 ? 0.0 : 1.0;
    float b = 0.99/98.0 * exp(-yOffset * log(98.0)) - 0.01/98.0;
    float x1 = yScale / (log(1.0) - log(b));
    float x2 = - log(b) * x1;
    float logvalue = log(v) * x1 + x2;
    float h = mix(v*yScale + yOffset, logvalue, logScale);
    return v == 0.0 ? 0.0 : h;
}

void main()
{
    // We want linear interpolation all the way out to the edge
    vec2 vertex = clamp(gl_Vertex.xz, 0.0, 1.0);
    vec2 tex = vertex*scale_tex + offset_tex;

    gl_TexCoord[0].xy = tex;

    vec2 tex1 = max(tex - offset_tex*2.0, offset_tex);
    vec2 tex2 = min(tex + offset_tex*2.0, 1.0-offset_tex);

    float height       = texture2D(tex_nearest, tex).x;
    float heightx1     = texture2D(tex_nearest, vec2(tex1.x, tex.y)).x;
    float heightx2     = texture2D(tex_nearest, vec2(tex2.x, tex.y)).x;
    float heighty1     = texture2D(tex_nearest, vec2(tex.x, tex1.y)).x;
    float heighty2     = texture2D(tex_nearest, vec2(tex.x, tex2.y)).x;

    height = heightValue(height);
    heightx1 = heightValue(heightx1);
    heightx2 = heightValue(heightx2);
    heighty1 = heightValue(heighty1);
    heighty2 = heightValue(heighty2);

    vec2 slope       = vec2(heightx2-heightx1, heighty2-heighty1);

    // calculate surface normal from slope for shading
    vec3 worldSpaceNormal = cross( vec3(0.0,            slope.y, tex2.y-tex1.y),
                                   vec3(tex2.x-tex1.x,  slope.x, 0.0));

    vec4 pos         = vec4(vertex.x, height, vertex.y, 1.0);

    // transform to homogeneous clip space
    gl_Position      = gl_ModelViewProjectionMatrix * pos;

    vec3 eyeSpacePos      = (gl_ModelViewMatrix * pos).xyz;
    vec3 eyeSpaceNormal   = (gl_NormalMatrix * worldSpaceNormal).xyz;

    eyeSpacePos      = normalize(eyeSpacePos);
    eyeSpaceNormal   = normalize(eyeSpaceNormal);
    worldSpaceNormal = normalize(worldSpaceNormal);

    float facing    = max(0.0, dot(eyeSpaceNormal, -eyeSpacePos));
    float diffuse   = max(0.0, worldSpaceNormal.y); // max(0.0, dot(worldSpaceNormalVector, lightDir));

    //shadow = clamp( 0.5 + diffuse+facing + fresnel, 0.5, 1.0);
    shadow = min( 0.5 + (diffuse+facing)*0.5, 1.0);

    vertex_height = height;
}
