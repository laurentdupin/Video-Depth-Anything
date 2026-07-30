#version 450 core

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    float data[];
} input_buffer;

layout(push_constant) uniform Parameters {
    uint rows;
    uint inner;
} parameters;

float erf_approx(float value) {
    const float sign_value = value < 0.0 ? -1.0 : 1.0;
    const float x = abs(value);
    const float t = 1.0 / (1.0 + 0.3275911 * x);
    const float polynomial =
        (((((1.061405429 * t - 1.453152027) * t) +
            1.421413741) * t - 0.284496736) * t +
            0.254829592) * t;
    return sign_value * (1.0 - polynomial * exp(-x * x));
}

void main() {
    const uint index = gl_GlobalInvocationID.x;
    const uint count = parameters.rows * parameters.inner;
    if (index >= count) return;
    const uint row = index / parameters.inner;
    const uint channel = index % parameters.inner;
    const uint source = row * parameters.inner * 2;
    const float gate =
        input_buffer.data[source + parameters.inner + channel];
    const float gelu =
        0.5 * gate * (1.0 + erf_approx(gate * 0.7071067811865476));
    output_buffer.data[index] =
        input_buffer.data[source + channel] * gelu;
}
