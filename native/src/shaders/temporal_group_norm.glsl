#version 450 core

layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    float data[];
} input_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Scale {
    float data[];
} scale_buffer;
layout(set = 0, binding = 3, std430) readonly buffer Bias {
    float data[];
} bias_buffer;

layout(push_constant) uniform Parameters {
    uint frames;
    uint channels;
    uint spatial;
    uint groups;
} parameters;

shared float partial_sum[128];
shared float partial_square[128];

void main() {
    const uint lane = gl_LocalInvocationID.x;
    const uint frame_group = gl_WorkGroupID.x;
    const uint frame = frame_group / parameters.groups;
    const uint group = frame_group % parameters.groups;
    if (frame >= parameters.frames) return;
    const uint group_channels = parameters.channels / parameters.groups;
    const uint values = group_channels * parameters.spatial;
    float sum = 0.0;
    float square = 0.0;
    for (uint index = lane; index < values; index += 128) {
        const uint local_channel = index / parameters.spatial;
        const uint position = index % parameters.spatial;
        const uint channel = group * group_channels + local_channel;
        const float value = input_buffer.data[
            (frame * parameters.channels + channel) *
                parameters.spatial +
            position];
        sum += value;
        square += value * value;
    }
    partial_sum[lane] = sum;
    partial_square[lane] = square;
    barrier();
    for (uint stride = 64; stride > 0; stride >>= 1) {
        if (lane < stride) {
            partial_sum[lane] += partial_sum[lane + stride];
            partial_square[lane] += partial_square[lane + stride];
        }
        barrier();
    }
    const float mean = partial_sum[0] / float(values);
    const float variance =
        max(partial_square[0] / float(values) - mean * mean, 0.0);
    const float inverse = inversesqrt(variance + 1.0e-6);
    for (uint index = lane; index < values; index += 128) {
        const uint local_channel = index / parameters.spatial;
        const uint position = index % parameters.spatial;
        const uint channel = group * group_channels + local_channel;
        const float value = input_buffer.data[
            (frame * parameters.channels + channel) *
                parameters.spatial +
            position];
        output_buffer.data[
            (frame * parameters.spatial + position) *
                parameters.channels +
            channel] =
            (value - mean) * inverse * scale_buffer.data[channel] +
            bias_buffer.data[channel];
    }
}
