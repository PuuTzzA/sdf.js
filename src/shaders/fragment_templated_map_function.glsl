#version 300 es
precision highp float;

#define MAX_NUM_COMMANDS 1024
#define MAX_SIZE_ELEMENT_BUFFER 512
#define EPSILON 1e-4
#define MAX_FLOAT 3.402823466e+38f
#define ZERO (min(uNumCommands,0)) // non-constant zero to avoid inlining of functions

// ╔══════════════════════════════════════════════════════════╗
// ║                       UNIFORMS                           ║
// ╚══════════════════════════════════════════════════════════╝
layout (std140) uniform CommandBlock {
    ivec4 commandData[MAX_SIZE_ELEMENT_BUFFER / 4];
};
layout (std140) uniform GeometryBlock {
    vec4 geometryData[MAX_SIZE_ELEMENT_BUFFER];
};
layout (std140) uniform ShadingBlock {
    vec4 shadingData[MAX_SIZE_ELEMENT_BUFFER];
};

uniform int uNumCommands;

uniform vec2 uResolution;
uniform float uTopOffset;
uniform float uLeftOffset;
uniform float uWindowWidth;
uniform float uWindowHeight;

uniform float uCameraZ;
uniform bool uTwoDMode;

// Uniforms for the Glyph Texture
uniform highp sampler2DArray uSdfArray;
uniform vec2 uBoxMin;
uniform vec2 uBoxMax;

// ╔══════════════════════════════════════════════════════════╗
// ║              SHADER INPUT, OUTPUT, STRUCTS               ║
// ╚══════════════════════════════════════════════════════════╝
in vec2 vUv;
out vec4 fragColor;

struct Surface {
    vec3 colorDiffuse;
    vec3 colorSpecular;
    vec3 colorAmbient;
    float kd; // diffuse material property
    float ks; // specular material property
    float p; // specular exponent (specular fall off)
    float ka; // ambient material property
    float mix; // mix factor
    float distance;
};

struct HitInfo {
    int id;
    vec3 pos;
    vec3 normal;
    Surface surface;
};

// ╔══════════════════════════════════════════════════════════╗
// ║                         SDFs                             ║
// ╚══════════════════════════════════════════════════════════╝
float sdSphere(vec3 p, float s) {
    return length(p) - s;
}

float sdBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0f)) + min(max(q.x, max(q.y, q.z)), 0.0f);
}

float sdRoundBox(vec3 p, vec3 b, float r) {
    vec3 q = abs(p) - b + r;
    return length(max(q, 0.0f)) + min(max(q.x, max(q.y, q.z)), 0.0f) - r;
}

float sdCornerCircle(in vec2 uv) {
    return length(uv - vec2(0.0f, -1.0f)) - sqrt(2.0f);
}

float sdCornerParabola(in vec2 uv) {
    // https://www.shadertoy.com/view/ws3GD7
    float y = (0.5f + uv.y) * (2.0f / 3.0f);
    float h = uv.x * uv.x + y * y * y;
    float w = pow(uv.x + sqrt(abs(h)), 1.0f / 3.0f);
    float x = w - y / w;
    vec2 q = vec2(x, 0.5f * (1.0f - x * x));
    return length(uv - q) * sign(uv.y - q.y);
}

const float kT = 6.28318531f;

float sdCornerCosine(in vec2 uv) {
    // https://www.shadertoy.com/view/3t23WG
    uv *= (kT / 4.0f);

    float ta = 0.0f, tb = kT / 4.0f;
    for (int i = 0; i < 8; i++) {
        float t = 0.5f * (ta + tb);
        float y = t - uv.x + sin(t) * (uv.y - cos(t));
        if (y < 0.0f)
            ta = t;
        else
            tb = t;
    }
    vec2 qa = vec2(ta, cos(ta)), qb = vec2(tb, cos(tb));
    vec2 pa = uv - qa, di = qb - qa;
    float h = clamp(dot(pa, di) / dot(di, di), 0.0f, 1.0f);
    return length(pa - di * h) * sign(pa.y * di.x - pa.x * di.y) * (4.0f / kT);
}

float sdCornerCubic(in vec2 uv) {
    float ta = 0.0f, tb = 1.0f;
    for (int i = 0; i < 12; i++) {
        float t = 0.5f * (ta + tb);
        float c = (t * t * (t - 3.0f) + 2.0f) / 3.0f;
        float dc = t * (t - 2.0f);
        float y = (uv.x - t) + (uv.y - c) * dc;
        if (y > 0.0f)
            ta = t;
        else
            tb = t;
    }
    vec2 qa = vec2(ta, (ta * ta * (ta - 3.0f) + 2.0f) / 3.0f);
    vec2 qb = vec2(tb, (tb * tb * (tb - 3.0f) + 2.0f) / 3.0f);
    vec2 pa = uv - qa, di = qb - qa;
    float h = clamp(dot(pa, di) / dot(di, di), 0.0f, 1.0f);
    return length(pa - di * h) * sign(pa.y * di.x - pa.x * di.y);
}

float sdRoundBox2d(in vec2 p, in vec2 b, in vec4 r, int type) {
    // select corner radius
    r.xy = (p.x > 0.0f) ? r.xy : r.zw;
    r.x = (p.y > 0.0f) ? r.x : r.y;

    // box coordinates
    vec2 q = abs(p) - b + r.x;

    // distance to sides
    if (min(q.x, q.y) < 0.0f)
        return max(q.x, q.y) - r.x;

    // rotate 45 degrees, offset by r and scale by r*sqrt(0.5)
    // to canonical corner coordinates
    r.x = max(EPSILON, r.x);
    vec2 uv = vec2(abs(q.x - q.y), q.x + q.y - r.x) / r.x;

    // compute distance to corner shape
    float d;
    if (type == 0)
        d = sdCornerCircle(uv);
    else if (type == 1)
        d = sdCornerParabola(uv);
    else if (type == 2)
        d = sdCornerCosine(uv);
    else if (type == 3)
        d = sdCornerCubic(uv);
    // undo scale
    return d * r.x * sqrt(0.5f);
}

// ╔══════════════════════════════════════════════════════════╗
// ║                    SDF OPERATIONS                        ║
// ╚══════════════════════════════════════════════════════════╝
float opExtrusion(in vec3 p, in float sdf, in float h) {
    // https://iquilezles.org/articles/distfunctions
    vec2 w = vec2(sdf, abs(p.z) - h);
    return min(max(w.x, w.y), 0.0f) + length(max(w, 0.0f));
}

float opRound(in float primitive, in float rad) {
    return primitive - rad;
}

// ╔══════════════════════════════════════════════════════════╗
// ║                 SDF COMBINING OPERATIONS                 ║
// ╚══════════════════════════════════════════════════════════╝
vec2 smin(float a, float b, float k) { // ret.a = distance, ret.b = blendfactor //return vec2(min(a, b), a);
    k *= 6.0f;
    float h = max(k - abs(a - b), 0.0f) / k;
    float m = h * h * h * 0.5f;
    float s = m * k * (1.0f / 3.0f);
    return (a < b) ? vec2(a - s, m) : vec2(b - s, 1.0f - m);
}

float opUnion(float a, float b) {
    return min(a, b);
}

Surface opUnion(Surface a, Surface b) {
    float t = a.distance < b.distance ? 0.f : 1.f;

    return Surface(
        mix(a.colorDiffuse, b.colorDiffuse, t),
        mix(a.colorSpecular, b.colorSpecular, t),
        mix(a.colorAmbient, b.colorAmbient, t),
        mix(a.kd, b.kd, t),
        mix(a.ks, b.ks, t),
        mix(a.p, b.p, t),
        mix(a.ka, b.ka, t),
        t,
        min(a.distance, b.distance)
    );
}

float opSubtraction(float a, float b) {
    return max(a, -b);
}

Surface opSubtraction(Surface a, Surface b) {
    float t = a.distance > -b.distance ? 0.f : 1.f;

    return Surface(
        mix(a.colorDiffuse, b.colorDiffuse, t),
        mix(a.colorSpecular, b.colorSpecular, t), 
        mix(a.colorAmbient, b.colorAmbient, t), 
        mix(a.kd, b.kd, t), 
        mix(a.ks, b.ks, t), 
        mix(a.p, b.p, t), 
        mix(a.ka, b.ka, t), 
        t, 
        max(a.distance, -b.distance)
    ); 
}

float opIntersection(float a, float b) {
    return max(a, b);
}

Surface opIntersection(Surface a, Surface b) {
    float t = a.distance > b.distance ? 0.f : 1.f;

    return Surface(
        mix(a.colorDiffuse, b.colorDiffuse, t),
        mix(a.colorSpecular, b.colorSpecular, t), 
        mix(a.colorAmbient, b.colorAmbient, t), 
        mix(a.kd, b.kd, t), 
        mix(a.ks, b.ks, t), 
        mix(a.p, b.p, t), 
        mix(a.ka, b.ka, t), 
        t, 
        max(a.distance, b.distance)
    ); 
}

float opXor(float a, float b) {
    return max(min(a, b), -max(a, b));
}

Surface opXor(Surface a, Surface b) {
    float dist = max(min(a.distance, b.distance), -max(a.distance, b.distance));
    float t = dist == a.distance ? 0.f : 1.f;

    return Surface(
        mix(a.colorDiffuse, b.colorDiffuse, t),
        mix(a.colorSpecular, b.colorSpecular, t), 
        mix(a.colorAmbient, b.colorAmbient, t), 
        mix(a.kd, b.kd, t), 
        mix(a.ks, b.ks, t), 
        mix(a.p, b.p, t), 
        mix(a.ka, b.ka, t), 
        t, 
        dist
    ); 
}

float opSmoothUnion(float a, float b, float smoothness) {
    return smin(a, b, smoothness).x;
}

Surface opSmoothUnion(Surface a, Surface b, float smoothness) {
    vec2 blend = smin(a.distance, b.distance, smoothness);

    return Surface(
        mix(a.colorDiffuse, b.colorDiffuse, blend.y),
        mix(a.colorSpecular, b.colorSpecular, blend.y),
        mix(a.colorAmbient, b.colorAmbient, blend.y),
        mix(a.kd, b.kd, blend.y),
        mix(a.ks, b.ks, blend.y),
        mix(a.p, b.p, blend.y),
        mix(a.ka, b.ka, blend.y),
        blend.y,
        blend.x
    );
}

float opSmoothSubtraction(float a, float b, float smoothness) {
    return -smin(-a, b, smoothness).x;
}

Surface opSmoothSubtraction(Surface a, Surface b, float smoothness) {
    vec2 blend = smin(-a.distance, b.distance, smoothness);
    blend.x *= -1.0f;

    return Surface(
        mix(a.colorDiffuse, b.colorDiffuse, blend.y),
        mix(a.colorSpecular, b.colorSpecular, blend.y),
        mix(a.colorAmbient, b.colorAmbient, blend.y),
        mix(a.kd, b.kd, blend.y),
        mix(a.ks, b.ks, blend.y),
        mix(a.p, b.p, blend.y),
        mix(a.ka, b.ka, blend.y),
        blend.y,
        blend.x
    );
}

float opSmoothIntersection(float a, float b, float smoothness) {
    return -smin(-a, -b, smoothness).x;
}

Surface opSmoothIntersection(Surface a, Surface b, float smoothness) {
    vec2 blend = smin(-a.distance, -b.distance, smoothness);
    blend.x *= -1.0f;

    return Surface(
        mix(a.colorDiffuse, b.colorDiffuse, blend.y),
        mix(a.colorSpecular, b.colorSpecular, blend.y),
        mix(a.colorAmbient, b.colorAmbient, blend.y),
        mix(a.kd, b.kd, blend.y),
        mix(a.ks, b.ks, blend.y),
        mix(a.p, b.p, blend.y),
        mix(a.ka, b.ka, blend.y),
        blend.y,
        blend.x
    );
}

// ╔══════════════════════════════════════════════════════════╗
// ║                      RAYMARCHING                         ║
// ╚══════════════════════════════════════════════════════════╝
vec3 unpackColor(float f) {
    uint u = floatBitsToUint(f);
    return vec3(
        float((u >> 24u) & 255u), 
        float((u >> 16u) & 255u), 
        float((u >> 8u) & 255u)
    ) / 255.0f;
}

void initializeData(inout float data) {
    data = MAX_FLOAT;
}

void initializeData(inout Surface data) {
    data.colorDiffuse = vec3(0.0f);
    data.colorSpecular = vec3(0.0f);
    data.colorAmbient = vec3(0.0f);
    data.kd = 0.0f; // diffuse material property
    data.ks = 0.0f; // specular material property
    data.p = 0.0f; // specular exponent, fall of of specular light
    data.ka = 0.0f; // ambient material property
    data.distance = MAX_FLOAT;
}

void populateData(inout float data, int elementIdx) {
}

void populateData(inout Surface data, int elementIdx) {
    data.colorDiffuse = unpackColor(shadingData[elementIdx].x);
    data.colorSpecular = unpackColor(shadingData[elementIdx].y);
    data.colorAmbient = unpackColor(shadingData[elementIdx].z);
    data.kd = shadingData[elementIdx].w; // diffuse material property 
    data.ks = shadingData[elementIdx + 1].x; // specular material property 
    data.p = shadingData[elementIdx + 1].y; // specular exponent, fall of of specular light
    data.ka = shadingData[elementIdx + 1].z; // ambient material property
}

void setDistance(inout float destination, float distance) {
    destination = distance;
}

void setDistance(inout Surface destination, float distance) {
    destination.distance = distance;
}

float getBakedSDF(int charIndex, vec3 pos, float scale, float depth) {
    float rangeX = uBoxMax.x - uBoxMin.x;
    float rangeY = uBoxMax.y - uBoxMin.y;

    float bakeToWorldRatio = rangeX * scale; // scale is the reciprocal of the size of the texture in world space
    vec2 pBake = vec2(pos.x, -pos.y) * bakeToWorldRatio; // invert y because in the texture the origin is bot-left and in the world top-left
    if (charIndex == 37) { // The '.' is an actual 3d sphere not just an extruded 2d one
        float radius = 22.5f / bakeToWorldRatio;
        return length(pos + vec3(-radius, radius, 0.0f)) - radius; // (length(pBakeMetric - uBoxMin) - 22.5f) / bakeToWorldRatio;
    }

    vec2 pTex = (pBake - uBoxMin) / vec2(rangeX, rangeY); // apply the offset so that the texture's origin lines up and convert to [0..1] range
    vec2 pTexClamped = clamp(pTex, vec2(0.0f), vec2(1.0f));
    float baseDist = textureLod(uSdfArray, vec3(pTexClamped, float(charIndex)), 0.0f).r;

    // Extrapolation (for points outside of the texture)
    vec2 pBakeMetric = pTex * vec2(rangeX, rangeY);
    vec2 pBakeMetricClamped = pTexClamped * vec2(rangeX, rangeY);
    float exteriorDist = length(pBakeMetric - pBakeMetricClamped);

    return opExtrusion(pos, (baseDist + exteriorDist) / bakeToWorldRatio, depth); // convert the distance form glyph-space to world space
}

#define SETUP_M populateData(current, elementIdx);                                                                                      \
                M = mat4(                                                                                                               \
                    vec4(geometryData[elementIdx].xyz, 0.f),                                                                            \
                    vec4(geometryData[elementIdx].w, geometryData[elementIdx + 1].x, geometryData[elementIdx + 1].y, 0.f),              \
                    vec4(geometryData[elementIdx + 1].z, geometryData[elementIdx + 1].w, geometryData[elementIdx + 2].x, 0.f),          \
                    vec4(geometryData[elementIdx + 2].yzw, 1.f)                                                                         \
                );                                                                                                                      \
                pos = (M * vec4(p, 1.0f)).xyz;\

#define GENERATE_MAP_FUNCTION(FUNCTION_NAME, RETURN_TYPE)                                                                               \
RETURN_TYPE FUNCTION_NAME(vec3 p) {                                                                                                     \
    RETURN_TYPE accumulatedResult; /* fixed size stack */                                                                               \
    initializeData(accumulatedResult);                                                                                                  \
                                                                                                                                        \
    int layerOperation = 101; /* persistent layer operation */                                                                          \
    float smoothness = 0.001f; /* persistent smoothness parameter for layerOperations */                                                \
                                                                                                                                        \
    for (int i = 0; i < uNumCommands; i++) {                                                                                            \
        int command = commandData[i].x;                                                                                                 \
        int elementIdx = commandData[i].y;                                                                                              \
        mat4 M;                                                                                                                         \
        vec3 pos;                                                                                                                       \
        float sdValue;                                                                                                                  \
        RETURN_TYPE current;                                                                                                            \
                                                                                                                                        \
        /* Sphere */ /* If-else chain due to shorter compile time and ability to use continue */                                        \
        if (command == 0) {                                                                                                             \
            SETUP_M sdValue = sdSphere(pos, geometryData[elementIdx + 3].x);                                                            \
            setDistance(current, sdValue);                                                                                              \
        }                                                                                                                               \
        /* Box Simple */                                                                                                                \
        else if (command == 1) {                                                                                                        \
            SETUP_M sdValue = sdBox(pos, vec3(geometryData[elementIdx + 3].xyz));                                                       \
            setDistance(current, sdValue);                                                                                              \
        }                                                                                                                               \
        /* Box (with rounded corners) */                                                                                                \
        else if (command == 2) {                                                                                                        \
            SETUP_M float w = geometryData[elementIdx + 3].x;                                                                           \
            float h = geometryData[elementIdx + 3].y;                                                                                   \
            float d = geometryData[elementIdx + 3].z;                                                                                   \
                                                                                                                                        \
            int initialRotation = floatBitsToInt(geometryData[elementIdx + 5].y);                                                       \
                                                                                                                                        \
            /* Adiddional Rotation (rounded edge selection) */                                                                          \
            if (initialRotation == 1) {                                                                                                 \
                mat3 Rot = mat3(0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 0.0f);                                                 \
                pos = Rot * pos;                                                                                                        \
                float temp = w;                                                                                                         \
                w = d;                                                                                                                  \
                d = temp;                                                                                                               \
            } else if (initialRotation == 2) {                                                                                          \
                mat3 Rot = mat3(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f);                                                 \
                pos = Rot * pos;                                                                                                        \
                float temp = h;                                                                                                         \
                h = d;                                                                                                                  \
                d = temp;                                                                                                               \
            }                                                                                                                           \
                                                                                                                                        \
            float val = sdRoundBox2d(pos.xy, vec2(w, h), geometryData[elementIdx + 4], floatBitsToInt(geometryData[elementIdx + 5].x)); \
            sdValue = opExtrusion(pos, val, d);                                                                                         \
                                                                                                                                        \
            /* sdValue = opRound(sdValue, geometryData[elementIdx + 5].z); */                                                           \
            setDistance(current, sdValue);                                                                                              \
        }                                                                                                                               \
        /* Round Box (all rounded edges) */                                                                                             \
        else if (command == 3) {                                                                                                        \
            SETUP_M sdValue = sdRoundBox(pos, geometryData[elementIdx + 3].xyz, geometryData[elementIdx + 3].w);                        \
            setDistance(current, sdValue);                                                                                              \
        }                                                                                                                               \
        /* Text */                                                                                                                      \
        else if (command == 4) {                                                                                                        \
            SETUP_M                                                                                                                     \
            /* The letters are stored in a TextureArray according to their index */                                                     \
            int numLetters = floatBitsToInt(geometryData[elementIdx + 3].x);                                                            \
            float scale = geometryData[elementIdx + 3].y; /* inverse scale */                                                           \
            float depth = geometryData[elementIdx + 3].z;                                                                               \
            float letterSmoothness = geometryData[elementIdx + 3].w;                                                                    \
                                                                                                                                        \
            sdValue = MAX_FLOAT;                                                                                                        \
            for (int letterIdx = ZERO; letterIdx < numLetters; letterIdx++) {                                                           \
                M[3][0] = geometryData[elementIdx + 4 + letterIdx].x; /* matrix[column][row] */                                         \
                M[3][1] = geometryData[elementIdx + 4 + letterIdx].y;                                                                   \
                M[3][2] = geometryData[elementIdx + 4 + letterIdx].z;                                                                   \
                int letterCode = floatBitsToInt(geometryData[elementIdx + 4 + letterIdx].w);                                            \
                                                                                                                                        \
                pos = (M * vec4(p, 1.f)).xyz;                                                                                           \
                sdValue = opSmoothUnion(getBakedSDF(letterCode, pos, scale, depth), sdValue, letterSmoothness);                         \
            }                                                                                                                           \
            setDistance(current, sdValue);                                                                                              \
        }                                                                                                                               \
        /* Set Layer Data */                                                                                                            \
        else if (command == 100) {                                                                                                      \
            layerOperation = floatBitsToInt(geometryData[elementIdx].x);                                                                \
            smoothness = geometryData[elementIdx].y;                                                                                    \
            continue;                                                                                                                   \
        }                                                                                                                               \
                                                                                                                                        \
        switch (layerOperation) {                                                                                                       \
            case 101: /* Union */                                                                                                       \
                accumulatedResult = opUnion(current, accumulatedResult);                                                                \
                break;                                                                                                                  \
            case 102: /* Subtraction */                                                                                                 \
                accumulatedResult = opSubtraction(current, accumulatedResult);                                                          \
                break;                                                                                                                  \
            case 103: /* Intersection */                                                                                                \
                accumulatedResult = opIntersection(current, accumulatedResult);                                                         \
                break;                                                                                                                  \
            case 104: /* Xor */                                                                                                         \
                accumulatedResult = opXor(current, accumulatedResult);                                                                  \
                break;                                                                                                                  \
            case 105: /* Smooth union */                                                                                                \
                accumulatedResult = opSmoothUnion(current, accumulatedResult, smoothness);                                              \
                break;                                                                                                                  \
            case 106: /* Smooth subtraction */                                                                                          \
                accumulatedResult = opSmoothSubtraction(current, accumulatedResult, smoothness);                                        \
                break;                                                                                                                  \
            case 107: /* Smooth intersection */                                                                                         \
                accumulatedResult = opSmoothIntersection(current, accumulatedResult, smoothness);                                       \
                break;                                                                                                                  \
        }                                                                                                                               \
    }                                                                                                                                   \
    return accumulatedResult;                                                                                                           \
}                                                                                                                                       \

GENERATE_MAP_FUNCTION(map, float)
GENERATE_MAP_FUNCTION(mapWithMaterial, Surface)

vec3 calcNormalTetrahedron(vec3 p) {
    // https://iquilezles.org/articles/normalsSDF/
    const float h = 0.0001f;      // replace by an appropriate value
    vec3 n = vec3(0.0f);
    for (int i = ZERO; i < 4; i++) {
        vec3 e = 0.5773f * (2.0f * vec3((((i + 3) >> 1) & 1), ((i >> 1) & 1), (i & 1)) - 1.0f);
        n += e * map(p + e * h);
    }
    return normalize(n);
}

HitInfo trace(vec3 ro, vec3 rd) {
    // adapted from Accelerating Sphere Tracing 
    // https://diglib.eg.org/server/api/core/bitstreams/7537a378-9a0a-4ef4-b57d-877322b1441e/content    
    float omega = 1.2f;
    float t = 0.0f;
    float pixelRadius = EPSILON;
    float tMax = 100.0f;

    float rLast = 0.0f;
    float rCurr = 0.1f; // map(ro).distance; to reduce compilation time, because map() gets inlined
    float dPrev = 0.0f;

    float lowerBound = 0.001f; // lower bound for the stepsize when raymarching
    float upperBound = 0.01f; // upper bound for the stepsize when raymarching
    float lowerDistance = EPSILON; // distance at which stepsize = lowerBound
    float upperDistance = 0.01f; // distance at which stepsize = upperBound

    int directionalDerivativeZero = 0;

    for (int i = ZERO; i < 200; i++) {
        // Intersection found if raymarching
        bool raymarchingIntersection = rCurr < 0.0f;
        if (raymarchingIntersection) {
            float tLower = t - dPrev;
            float tUpper = t;
            float mid = 0.0f;

            for (int j = 0; j < 5; j++) {
                mid = (tLower + tUpper) * 0.5f;
                float sdfMid = map(ro + mid * rd);
                if (abs(sdfMid) < pixelRadius) {
                    break;
                }
                if (sdfMid < 0.0f) {
                    tUpper = mid;
                } else {
                    tLower = mid;
                }
            }
            // vec3 p = ro + t * rd;
            // return HitInfo(i, p, calcNormalTetrahedron(p), map(p));
        }

        // Hit condition
        if (raymarchingIntersection || rCurr < pixelRadius) {
            vec3 p = ro + t * rd;
            return HitInfo(i, p, calcNormalTetrahedron(p), mapWithMaterial(p));
        }

        if (t >= tMax) {
            vec3 p = ro + t * rd;
            break;
        }

        float dNext = rCurr;
        float denom = dPrev + rLast - rCurr;

        if (i > 0 && denom > EPSILON) {
            dNext = rCurr + omega * rCurr * (dPrev - rLast + rCurr) / denom;
        }

        // Detect parallel rays 
        if (rCurr < upperDistance && abs(rCurr - rLast) < EPSILON) {
            directionalDerivativeZero++;
        } else {
            directionalDerivativeZero = 0;
        }

        bool isParallel = directionalDerivativeZero >= 5;

        if (isParallel) {
            float tFactor = clamp((rCurr - lowerDistance) / (upperDistance - lowerDistance), 0.0f, 1.0f);
            float minStep = mix(lowerBound, upperBound, tFactor);
            // Allow over-relaxation to take a larger step if possible, but enforce minimum step
            dNext = max(dNext, minStep);
        }

        float rNext = map(ro + (t + dNext) * rd);

        // Overrelaxation was too big (only in the case where we don't do raymarching)
        if (!isParallel && dNext > rCurr + rNext) {
            dNext = rCurr;
            rNext = map(ro + (t + dNext) * rd);
        }

        t += dNext;
        dPrev = dNext;
        rLast = rCurr;
        rCurr = rNext;
    }

    return HitInfo(-1, vec3(0.0f), vec3(0.0f), Surface(vec3(0.0f), vec3(0.0f), vec3(0.0f), 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f));
}

// ╔══════════════════════════════════════════════════════════╗
// ║                         SHADING                          ║
// ╚══════════════════════════════════════════════════════════╝
/* float shadow(in vec3 ro, in vec3 rd, float mint, float maxt) {
    float t = mint;
    for (int i = ZERO; i < 256 && t < maxt; i++) {
        float h = map(ro + rd * t);
        if (h < EPSILON)
            return 0.0f;
        t += h;
    }
    return 1.0f;
}

// https://iquilezles.org/articles/rmshadows
float softshadow(in vec3 ro, in vec3 rd, float mint, float maxt, float w) {
    float res = 1.0f;
    float t = mint;
    for (int i = ZERO; i < 256 && t < maxt; i++) {
        float h = map(ro + t * rd);
        res = min(res, h / (w * t));
        t += clamp(h, 0.005f, 0.50f);
        if (res < -1.0f || t > maxt)
            break;
    }
    res = max(res, -1.0f);
    return 0.25f * (1.0f + res) * (1.0f + res) * (2.0f - res);
}

float calcSoftshadow(in vec3 ro, in vec3 rd, float tmin, float tmax, const float k) {
    float res = 1.0f;
    float t = tmin;
    for (int i = ZERO; i < 50; i++) {
        float h = map(ro + rd * t);
        res = min(res, k * h / t);
        t += clamp(h, 0.02f, 0.20f);
        if (res < 0.005f || t > tmax)
            break;
    }
    return clamp(res, 0.0f, 1.0f);
}
 */
float gaussian(float x, float mu, float sigma) {
    return exp(-1.0f * ((x - mu) * (x - mu)) / (2.0f * sigma * sigma));
}

vec3 shade(HitInfo hit) {
    /* if (hit.id == -1) {
        return vec3(1., 0., 1.);
    }
    if (hit.id == -2) {
        return vec3(0.,1.,1.);
    }
    if (hit.id < 20) {
        return vec3(0., float(hit.id) / 20., 0.);
    }
    if (hit.id < 50) {
        float val = float(hit.id) / 50.;
        return vec3(val, val, 0.);
    }
    return vec3(1, 0., 0.); */

    if (hit.id == -1) {
        return vec3(0);
    }

    const vec3 lightPos = vec3(0.5f, 0.5f, 10.0f);

    vec3 vecToLight = normalize(lightPos - hit.pos);
    vec3 vecFromLight = normalize(hit.pos - lightPos);

    Surface surface = hit.surface;

    float mixFacotr = gaussian(surface.mix, 0.5f, 0.07f);

    float ld = 1.f; // diffuse light intensity (light source dependent)
    float la = 1.f; // ambient light intensity (constant for scene)
    float ls = 1.f; // specular light intensity (light source dependent)

    float iDiffuse = surface.kd * ld * max(0.0f, dot(vecToLight, hit.normal));
    float iAmbient = surface.ka * la;
    float iSpecular = surface.ks * ls * pow(max(0.0f, dot(reflect(vecFromLight, hit.normal), vec3(0.0f, 0.0f, 1.0f))), surface.p);

    //float shadow = shadow(hit.pos, -sundir, 0.001f, 5.f);
    float shadow; // = softshadow(hit.pos, vecToLight, 0.001f, 5.f, 0.1f);
    //float shadow = calcSoftshadow(hit.pos, -sundir, 0.01f, 5.0f, 16.0f);
    // shadow = max(shadow, 0.1f);
    shadow = 1.0f;

    //return vec3(shadow);
    //return hit.id != -1 ? vec3(1.f) : vec3(0.f);
    return shadow * (iDiffuse * surface.colorDiffuse + iSpecular * surface.colorSpecular) + iAmbient * surface.colorAmbient;
}

struct ColorStop {
    vec3 color;
    float position;
};

#define COLOR_RAMP(colors, factor, finalColor) {                        \
    int index = 0;                                                      \
    for (int i = 0; i < colors.length() - 1; i ++) {                    \
        ColorStop currentColor = colors[i];                             \
        bool isInBetween = currentColor.position <= factor;             \
        index = isInBetween ? i : index;                                \
    }                                                                   \
    ColorStop currentColor = colors[index];                             \
    ColorStop nextColor = colors[index + 1];                            \
    float range = nextColor.position - currentColor.position;           \
    float lerpFactor = (factor - currentColor.position) / range;        \
    finalColor = mix(currentColor.color, nextColor.color, lerpFactor);  \
}                                                                       \

// ╔══════════════════════════════════════════════════════════╗
// ║                          MAIN                            ║
// ╚══════════════════════════════════════════════════════════╝
void main(void) {
    //const vec2 subPixleOffsets[] = vec2[](vec2(0.375f, 0.125f) - vec2(0.5f), vec2(0.875f, 0.375f) - vec2(0.5f), vec2(0.125f, 0.625f) - vec2(0.5f), vec2(0.625f, 0.875f) - vec2(0.5f));
    const vec2 subPixleOffsets[] = vec2[](vec2(0.0f, 0.0f));
    vec2 pixelSize = vec2(1.0f) / uResolution.x;

    vec3 color = vec3(0.0f);

    vec2 uv = vUv; // origin = top left
    uv *= vec2(uWindowWidth, uWindowHeight);
    uv += vec2(uLeftOffset, uTopOffset);

    /* float ddd = getBakedSDF(15,uv - vec2(0.5, 0.5), 0.5) / 1000.;
    ddd = fract(ddd * 10.);
    // ddd = textureLod(uSdfArray, vec3(pp, float(5)), 0.0).r;
    float ccc = ddd > .0 ? 0. : 1.;

    fragColor = vec4(ddd, ddd, ddd, 1.f);
    return; */

    /* if (pp.y <= 0. || pp.x >= 1.){
        fragColor = vec4(0., 0., 0., 1.);
    } */

    vec3 pos = vec3(uv, uCameraZ);
    vec3 dir = vec3(0.0f, 0.0f, -1.0f);
    vec3 posOffset;

    for (int i = 0; i < subPixleOffsets.length(); i++) {
        posOffset = pos + vec3(subPixleOffsets[i] * pixelSize, 0.0f);

        if (!uTwoDMode) {
            color += shade(trace(posOffset, dir));
        } else {
            posOffset.z = 0.0f;
            Surface surface; // mapWithMaterial(posOffset);
            float sdfValue = surface.distance * 80.0f;

            ColorStop[] colors = ColorStop[](
			    //ColorStop(surface.colorDiffuse, 0.000000),
            ColorStop(vec3(0.000000f, 0.000000f, 0.015996f), 0.000000f), ColorStop(vec3(0.008023f, 0.002428f, 0.162029f), 0.300000f), ColorStop(vec3(0.590619f, 0.964686f, 0.428690f), 0.400000f), ColorStop(vec3(0.991102f, 0.031896f, 0.814847f), 0.600000f), ColorStop(vec3(1.000000f, 0.000000f, 0.001821f), 0.800000f), ColorStop(vec3(0.008023f, 0.002428f, 0.162029f), 0.900000f), ColorStop(vec3(0.000000f, 0.000000f, 0.015996f), 1.000000f));
            vec3 finalColor;
            COLOR_RAMP(colors, sdfValue, finalColor);

            color += vec3(finalColor);
        }
    }

    color /= float(subPixleOffsets.length());

    fragColor = vec4(color, 1.0f);
}