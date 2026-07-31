layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    float data[];
} input_buffer;
#if defined(HALF_WEIGHT)
layout(set = 0, binding = 2, std430) readonly buffer Weight {
    uint data[];
} weight_buffer;
float read_weight(uint index) {
    const vec2 values =
        unpackHalf2x16(weight_buffer.data[index >> 1]);
    return (index & 1) == 0 ? values.x : values.y;
}
#else
layout(set = 0, binding = 2, std430) readonly buffer Weight {
    float data[];
} weight_buffer;
float read_weight(uint index) {
    return weight_buffer.data[index];
}
#endif
layout(set = 0, binding = 3, std430) readonly buffer Bias {
    float data[];
} bias_buffer;
layout(push_constant) uniform Parameters {
    uint input_width;
    uint input_height;
    uint input_channels;
    uint output_width;
    uint output_height;
    uint output_channels;
    uint kernel;
    uint stride;
    int padding;
    uint has_bias;
    uint batches;
    uint output_channel_blocks;
} parameters;

shared float spatial_tile[1440];
shared float kernel_tile[576];

void main() {
    const uint output_x = gl_GlobalInvocationID.x;
    const uint output_y = gl_GlobalInvocationID.y;
    const uint batch =
        gl_GlobalInvocationID.z / parameters.output_channel_blocks;
    const uint output_channel_base =
        (gl_GlobalInvocationID.z %
            parameters.output_channel_blocks) * 8;
    const bool valid =
        batch < parameters.batches &&
        output_x < parameters.output_width &&
        output_y < parameters.output_height &&
        output_channel_base < parameters.output_channels;
    float sums[8] = float[8](0, 0, 0, 0, 0, 0, 0, 0);
    const uint lane =
        gl_LocalInvocationID.y * 16 + gl_LocalInvocationID.x;
    const int origin_x = int(gl_WorkGroupID.x * 16) - 1;
    const int origin_y = int(gl_WorkGroupID.y * 8) - 1;
    for (uint channel_base = 0;
         channel_base < parameters.input_channels;
         channel_base += 8) {
        for (uint index = lane; index < 1440; index += 128) {
            const uint channel_offset = index / 180;
            const uint tile_index = index % 180;
            const uint channel = channel_base + channel_offset;
            const int input_x = origin_x + int(tile_index % 18);
            const int input_y = origin_y + int(tile_index / 18);
            spatial_tile[index] =
                channel < parameters.input_channels &&
                input_x >= 0 && input_x < int(parameters.input_width) &&
                input_y >= 0 && input_y < int(parameters.input_height)
                ? input_buffer.data[
                    ((batch * parameters.input_channels + channel) *
                        parameters.input_height + uint(input_y)) *
                        parameters.input_width + uint(input_x)]
                : 0.0;
        }
        for (uint index = lane; index < 576; index += 128) {
            const uint channel_offset = index / 72;
            const uint kernel_index = index % 72;
            const uint channel = channel_base + channel_offset;
            const uint output_offset = kernel_index / 9;
            const uint output_channel =
                output_channel_base + output_offset;
            kernel_tile[index] =
                channel < parameters.input_channels &&
                output_channel < parameters.output_channels
                ? read_weight(
                    (output_channel * parameters.input_channels +
                        channel) * 9 + kernel_index % 9)
                : 0.0;
        }
        barrier();
        if (valid) {
            for (uint channel_offset = 0;
                 channel_offset < 8 &&
                    channel_base + channel_offset <
                        parameters.input_channels;
                 ++channel_offset)
                for (uint kernel_y = 0; kernel_y < 3; ++kernel_y)
                    for (uint kernel_x = 0; kernel_x < 3; ++kernel_x) {
                        const float value = spatial_tile[
                            channel_offset * 180 +
                            (gl_LocalInvocationID.y + kernel_y) * 18 +
                            gl_LocalInvocationID.x + kernel_x];
                        const uint kernel_index =
                            kernel_y * 3 + kernel_x;
                        for (uint output_offset = 0;
                             output_offset < 8;
                             ++output_offset)
                            sums[output_offset] += value * kernel_tile[
                                channel_offset * 72 +
                                output_offset * 9 + kernel_index];
                    }
        }
        barrier();
    }
    if (!valid) return;
    for (uint output_offset = 0; output_offset < 8; ++output_offset) {
        const uint output_channel =
            output_channel_base + output_offset;
        if (output_channel < parameters.output_channels)
            output_buffer.data[
                ((batch * parameters.output_channels +
                    output_channel) * parameters.output_height +
                    output_y) * parameters.output_width + output_x] =
                sums[output_offset] +
                (parameters.has_bias != 0
                    ? bias_buffer.data[output_channel] : 0.0);
    }
}
