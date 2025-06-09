#version 460

#define fragCoord gl_FragCoord.xy

layout(location = 0) out vec4 fragColor;

layout(push_constant) uniform PushConstant {
    vec3 resolution;
    float time;
    vec4 mouse;
} pushConstant;

const int MAX_STEPS = 128;
const float SURF_DIST = 0.001;
const float MAX_DIST = 100.0;

struct Material {
    int id;
    vec3 col;
};

struct Hit {
    float dist;
    Material mat;
};

// More at:
// https://iquilezles.org/articles/raymarchingdf/
// https://iquilezles.org/articles/distfunctions/

float sdSphere(vec3 p, float s) {
    return length(p) - s;
}

float sdBox(vec3 p, vec3 b) {
    vec3 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0);
}

float sdOctahedron(vec3 p, float s) {
    p = abs(p);
    return (p.x + p.y + p.z - s) * 0.57735027;
}

Hit opUnion(Hit a, Hit b) {
    return (a.dist < b.dist) ? a : b;
}

Hit opSmoothUnion(Hit a, Hit b, float k) {
    float h = clamp(0.5 + 0.5 * (b.dist - a.dist) / k, 0.0, 1.0);
    float d = mix(b.dist, a.dist, h) - k * h * (1.0 - h);
    Material m;
    m.id = (h < 0.5) ? a.mat.id : b.mat.id;
    m.col = mix(b.mat.col, a.mat.col, h);
    return Hit(d, m);
}

Hit scene(vec3 p) {
    // Sphere
    Hit h1 = Hit(sdSphere(p - vec3(0.0, 0.0, 4.0 * sin(pushConstant.time)), 1.5),
                 Material(1, vec3(1.0, 0.0, 0.0)));

    // Box
    Hit h2 = Hit(sdBox(p - vec3(0.0, 0.0, 4.0), vec3(1.8)),
                 Material(2, vec3(0.0, 0.5, 1.0)));

    // Octahedron
    Hit h3 = Hit(sdOctahedron(p - vec3(0.0, 0.0, -4.0), 2.0),
                 Material(3, vec3(0.2, 1.0, 0.4)));

    return opSmoothUnion(h1, opUnion(h2, h3), 0.8);
}

Hit rayMarch(vec3 ro, vec3 rd) {
    Hit hit;

    float t = 0.0;
    for (int i = 0; i < MAX_STEPS; i++) {
        vec3 p = ro + t * rd;
        hit = scene(p);
        if (hit.dist < SURF_DIST || t > MAX_DIST) break;
        t += hit.dist;
    }

    if (t > MAX_DIST) hit.mat.id = 0; // No hit
    hit.dist = t;
    return hit;
}

vec3 calcNormal(vec3 p) {
    const float eps = 0.001;
    const vec2 h = vec2(eps, 0);
    return normalize(vec3(scene(p + h.xyy).dist - scene(p - h.xyy).dist,
                          scene(p + h.yxy).dist - scene(p - h.yxy).dist,
                          scene(p + h.yyx).dist - scene(p - h.yyx).dist));
}

vec3 shade(vec3 p, vec3 rd, Material mat) {
    vec3 lightDir = normalize(vec3(0.5, 0.8, 1.0));
    vec3 n = calcNormal(p);
    float diff = max(dot(n, lightDir), 0.0);
    return mat.col * (diff + vec3(0.1)); // Adding some ambient light
}

vec3 getRayDir(vec2 uv, vec3 ro, vec3 lookAt) {
    vec3 f = normalize(lookAt - ro);
    vec3 r = normalize(cross(vec3(0,-1,0), f));
    vec3 u = cross(f, r);
    return normalize(uv.x * r + uv.y * u + 1.5 * f);
}

void main() {
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = (2.0 * fragCoord - pushConstant.resolution.xy) / pushConstant.resolution.y;
    const vec2 mouse = vec2(pushConstant.mouse.x, pushConstant.mouse.y) * 0.005;
    float cx = 8.0 * sin(mouse.y) * cos(mouse.x * 2.0);
    float cy = 8.0 * cos(mouse.y);
    float cz = 8.0 * sin(mouse.y) * sin(mouse.x * 2.0);

    // Ray construction
    vec3 ro = vec3(cx, cy, cz);
    vec3 rd = getRayDir(uv, ro, vec3(0.0));

    // Ray march
    Hit hit = rayMarch(ro, rd);

    // Shade
    vec3 col = vec3(0.0);
    if (hit.mat.id != 0) {
        vec3 p = ro + rd * hit.dist;
        col = shade(p, rd, hit.mat);
    }

    // Output to screen
    fragColor = vec4(col, 1.0);
}
