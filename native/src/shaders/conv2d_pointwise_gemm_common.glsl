layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

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

const uint inner_tile = 32;
shared float input_tile[32 * inner_tile];
shared float weight_tile[32 * inner_tile];

void main() {
    const uint batch = gl_GlobalInvocationID.z;
    const uint spatial_base =
        gl_WorkGroupID.x * 32 + gl_LocalInvocationID.x * 4;
    const uint output_channel_base =
        gl_WorkGroupID.y * 32 + gl_LocalInvocationID.y * 4;
    const uint spatial_count =
        parameters.output_width * parameters.output_height;
    const uint lane =
        gl_LocalInvocationID.y * gl_WorkGroupSize.x +
        gl_LocalInvocationID.x;
    float sums[4][4];
    for (uint output_offset = 0; output_offset < 4; ++output_offset)
        for (uint spatial_offset = 0; spatial_offset < 4; ++spatial_offset)
            sums[output_offset][spatial_offset] = 0.0;
    for (uint inner_base = 0;
         inner_base < parameters.input_channels;
         inner_base += inner_tile) {
        for (uint index = lane; index < 32 * inner_tile; index += 64) {
            const uint spatial_offset = index / inner_tile;
            const uint inner = inner_base + index % inner_tile;
            const uint spatial =
                gl_WorkGroupID.x * 32 + spatial_offset;
            input_tile[index] =
                batch < parameters.batches &&
                spatial < spatial_count &&
                inner < parameters.input_channels
                ? input_buffer.data[
                    (batch * parameters.input_channels + inner) *
                        spatial_count + spatial]
                : 0.0;
        }
        for (uint index = lane; index < 32 * inner_tile; index += 64) {
            const uint output_offset = index / inner_tile;
            const uint inner = inner_base + index % inner_tile;
            const uint output_channel =
                gl_WorkGroupID.y * 32 + output_offset;
            weight_tile[index] =
                output_channel < parameters.output_channels &&
                inner < parameters.input_channels
                ? read_weight(
                    output_channel * parameters.input_channels + inner)
                : 0.0;
        }
        barrier();
        const uint inner_count =
            min(inner_tile, parameters.input_channels - inner_base);
        for (uint inner_offset = 0;
             inner_offset < inner_count;
             ++inner_offset) {
            float input_values[4];
            float weight_values[4];
            for (uint spatial_offset = 0; spatial_offset < 4;
                 ++spatial_offset)
                input_values[spatial_offset] = input_tile[
                    (gl_LocalInvocationID.x * 4 + spatial_offset) *
                        inner_tile + inner_offset];
            for (uint output_offset = 0; output_offset < 4;
                 ++output_offset)
                weight_values[output_offset] = weight_tile[
                    (gl_LocalInvocationID.y * 4 + output_offset) *
                        inner_tile + inner_offset];
            for (uint output_offset = 0; output_offset < 4;
                 ++output_offset)
                for (uint spatial_offset = 0; spatial_offset < 4;
                     ++spatial_offset)
                    sums[output_offset][spatial_offset] +=
                        weight_values[output_offset] *
                        input_values[spatial_offset];
        }
        barrier();
    }
    if (batch >= parameters.batches) return;
    for (uint output_offset = 0; output_offset < 4; ++output_offset) {
        const uint output_channel =
            output_channel_base + output_offset;
        if (output_channel >= parameters.output_channels) continue;
        const float bias = parameters.has_bias != 0
            ? bias_buffer.data[output_channel] : 0.0;
        for (uint spatial_offset = 0; spatial_offset < 4;
             ++spatial_offset) {
            const uint spatial = spatial_base + spatial_offset;
            if (spatial < spatial_count)
                output_buffer.data[
                    (batch * parameters.output_channels +
                        output_channel) * spatial_count + spatial] =
                    sums[output_offset][spatial_offset] + bias;
        }
    }
}
