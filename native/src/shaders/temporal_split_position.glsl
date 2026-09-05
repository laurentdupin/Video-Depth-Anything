#version 450 core

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Key {
    float data[];
} key_buffer;
layout(set = 0, binding = 1, std430) writeonly buffer Value {
    float data[];
} value_buffer;
layout(set = 0, binding = 2, std430) readonly buffer KeyValue {
    float data[];
} key_value_buffer;
layout(set = 0, binding = 3, std430) readonly buffer Position {
    float data[];
} position_buffer;

layout(push_constant) uniform Parameters {
    uint sequences;
    uint frames;
    uint channels;
} parameters;

void main() {
    const uint index = gl_GlobalInvocationID.x;
    const uint count =
        parameters.sequences * parameters.frames * parameters.channels;
    if (index >= count) return;
    const uint channel = index % parameters.channels;
    const uint frame =
        (index / parameters.channels) % parameters.frames;
    const uint row = index / parameters.channels;
    const uint combined_base = row * parameters.channels * 2;
    const uint position_base = frame * parameters.channels * 2;
    key_buffer.data[index] =
        key_value_buffer.data[combined_base + channel] +
        position_buffer.data[position_base + channel];
    value_buffer.data[index] =
        key_value_buffer.data[
            combined_base + parameters.channels + channel] +
        position_buffer.data[
            position_base + parameters.channels + channel];
}
