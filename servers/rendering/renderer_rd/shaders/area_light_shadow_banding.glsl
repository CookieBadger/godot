#[compute]

#version 450

#VERSION_DEFINES

#extension GL_EXT_samplerless_texture_functions : enable

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform texture2D reprojection_texture;

layout(set = 0, binding = 1, std430) restrict buffer Result {
	uint flag;
};

void main() {
	if (flag == 1) return;

	ivec2 pos = ivec2(gl_GlobalInvocationID.xy) + ivec2(1, 1);
	float center = texelFetch(reprojection_texture, pos, 0).r;

    if (center <= 0.0 || center >= 1.0) return;

	for(int dx = -1; dx <= 1; dx++) {
		for(int dy = -1; dy <= 1; dy++) {
			if (dx == 0 && dy == 0) continue;
			float neighbor = texelFetch(reprojection_texture, pos + ivec2(dx, dy), 0).r;
			if (neighbor != center) {
				return;
			}
		}
	}

	atomicExchange(flag, 1);
}
