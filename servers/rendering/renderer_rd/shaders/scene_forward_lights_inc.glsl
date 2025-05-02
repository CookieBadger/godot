// Functions related to lighting
float D_GGX(float cos_theta_m, float alpha) {
	float a = cos_theta_m * alpha;
	float k = alpha / (1.0 - cos_theta_m * cos_theta_m + a * a);
	return k * k * (1.0 / M_PI);
}

// From Earl Hammon, Jr. "PBR Diffuse Lighting for GGX+Smith Microsurfaces" https://www.gdcvault.com/play/1024478/PBR-Diffuse-Lighting-for-GGX
float V_GGX(float NdotL, float NdotV, float alpha) {
	return 0.5 / mix(2.0 * NdotL * NdotV, NdotL + NdotV, alpha);
}

float D_GGX_anisotropic(float cos_theta_m, float alpha_x, float alpha_y, float cos_phi, float sin_phi) {
	float alpha2 = alpha_x * alpha_y;
	highp vec3 v = vec3(alpha_y * cos_phi, alpha_x * sin_phi, alpha2 * cos_theta_m);
	highp float v2 = dot(v, v);
	float w2 = alpha2 / v2;
	float D = alpha2 * w2 * w2 * (1.0 / M_PI);
	return D;
}

float V_GGX_anisotropic(float alpha_x, float alpha_y, float TdotV, float TdotL, float BdotV, float BdotL, float NdotV, float NdotL) {
	float Lambda_V = NdotL * length(vec3(alpha_x * TdotV, alpha_y * BdotV, NdotV));
	float Lambda_L = NdotV * length(vec3(alpha_x * TdotL, alpha_y * BdotL, NdotL));
	return 0.5 / (Lambda_V + Lambda_L);
}

float SchlickFresnel(float u) {
	float m = 1.0 - u;
	float m2 = m * m;
	return m2 * m2 * m; // pow(m,5)
}

vec3 F0(float metallic, float specular, vec3 albedo) {
	float dielectric = 0.16 * specular * specular;
	// use albedo * metallic as colored specular reflectance at 0 angle for metallic materials;
	// see https://google.github.io/filament/Filament.md.html
	return mix(vec3(dielectric), albedo, vec3(metallic));
}
uint hash(uint value) {
	uint state = value * 747796405u + 2891336453u;
	uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}
uint random_seed(vec3 seed) {
	uint x = uint(abs(seed.x));
	if (seed.x < 0.0) {
		x = hash(x);
	}
	uint y = uint(abs(seed.y));
	if (seed.y < 0.0) {
		y = hash(y);
	}
	uint z = uint(abs(seed.z));
	if (seed.z < 0.0) {
		z = hash(z);
	}

	return hash(uint(x ^ hash(y) ^ hash(z)));
}
// generates a random value in range [0.0, 1.0)
float randomize(uint value) {
	value = hash(value);
	return float(value / 4294967296.0);
}

float lerp(float v, float a, float b) {
	return a + v * (b - a);
}

float sample_linear(float u, float a, float b) {
	if (u == 0 && a == 0) {
		return 0;
	}
	float x = u * (a + b) / (a + sqrt(lerp(u, a, b)));

	const float ONE_MINUS_EPSILON = 1.0 - 1e-6f;
	return min(x, ONE_MINUS_EPSILON);
}

float bilinear_PDF(float u, float v, vec4 w) {
	if (u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0) {
		return 0;
	}
	if (w[0] + w[1] + w[2] + w[3] == 0.0) {
		return 1;
	}
	return 2.0 * ((1.0 - u) * (1.0 - v) * w[0] + u * (1.0 - v) * w[1] + (1.0 - u) * v * w[2] + u * v * w[3]) / (w[0] + w[1] + w[2] + w[3]);
}

vec2 sample_bilinear(float u, float v, vec4 w) {
	// Sample  for bilinear marginal distribution
	float y = sample_linear(v, w[0] + w[1], w[2] + w[3]);

	// Sample  for bilinear conditional distribution
	float x = sample_linear(u, lerp(y, w[0], w[2]), lerp(y, w[1], w[3]));

	return vec2(x, y);
}

struct SphericalQuad {
	vec3 o, x, y, z; // local reference system ’R’
	float z0, z0sq; //
	float x0, y0, y0sq; // rectangle coords in ’R’
	float x1, y1, y1sq; //
	float b0, b1, b0sq, k; // misc precomputed constants
	float S; // solid angle of ’Q’'
};

SphericalQuad init_spherical_quad(vec3 s, vec3 ex, vec3 ey, vec3 o) {
	float exl = length(ex);
	float eyl = length(ey);
	// compute local reference system ’R’
	vec3 x = ex / exl;
	vec3 y = ey / eyl;
	vec3 z = cross(x, y);
	// compute rectangle coords in local reference system
	vec3 d = s - o;
	float z0 = dot(d, z);
	// flip ’z’ to make it point against ’Q’
	if (z0 > 0) {
		z *= -1;
		z0 *= -1;
	}
	float z0sq = z0 * z0;
	float x0 = dot(d, x);
	float y0 = dot(d, y);
	float x1 = x0 + exl;
	float y1 = y0 + eyl;
	float y0sq = y0 * y0;
	float y1sq = y1 * y1;
	// create vectors to four vertices
	vec3 v00 = { x0, y0, z0 };
	vec3 v01 = { x0, y1, z0 };
	vec3 v10 = { x1, y0, z0 };
	vec3 v11 = { x1, y1, z0 };
	// compute normals to edges
	vec3 n0 = normalize(cross(v00, v10));
	vec3 n1 = normalize(cross(v10, v11));
	vec3 n2 = normalize(cross(v11, v01));
	vec3 n3 = normalize(cross(v01, v00));
	// compute internal angles (gamma_i)
	float g0 = acos(-dot(n0, n1));
	float g1 = acos(-dot(n1, n2));
	float g2 = acos(-dot(n2, n3));
	float g3 = acos(-dot(n3, n0));
	// compute predefined constants
	float b0 = n0.z;
	float b1 = n2.z;
	float b0sq = b0 * b0;
	float k = 2 * M_PI - g2 - g3;
	// compute solid angle from internal angles (sum of internal angles - 2*PI)
	float S = g0 + g1 - k;

	return SphericalQuad(o, x, y, z, z0, z0sq, x0, y0, y0sq, x1, y1, y1sq, b0, b1, b0sq, k, S);
}

vec3 sample_squad(SphericalQuad squad, float u, float v) {
	// 1. compute ’cu’
	float au = u * squad.S + squad.k;
	float fu = (cos(au) * squad.b0 - squad.b1) / sin(au);
	float cu = 1 / sqrt(fu * fu + squad.b0sq) * (fu > 0 ? +1 : -1);
	cu = clamp(cu, -1, 1); // avoid NaNs
	// 2. compute ’xu’
	float xu = -(cu * squad.z0) / sqrt(1 - cu * cu);
	xu = clamp(xu, squad.x0, squad.x1); // avoid Infs
	// 3. compute ’yv’
	float d = sqrt(xu * xu + squad.z0sq);
	float h0 = squad.y0 / sqrt(d * d + squad.y0sq);
	float h1 = squad.y1 / sqrt(d * d + squad.y1sq);
	float hv = h0 + v * (h1 - h0), hv2 = hv * hv;
	const float ONE_MINUS_EPSILON = 1.0 - 1e-6f;
	float yv = (hv2 < ONE_MINUS_EPSILON) ? (hv * d) / sqrt(1 - hv2) : squad.y1;
	// 4. transform (xu,yv,z0) to world coords
	return (squad.o + xu * squad.x + yv * squad.y + squad.z0 * squad.z);
}

void light_compute(vec3 N, vec3 L_diff, vec3 L_spec, vec3 V, float A, vec3 light_color, bool is_directional, float attenuation, vec3 f0, uint orms, float specular_amount, vec3 albedo, inout float alpha,
#ifdef LIGHT_BACKLIGHT_USED
		vec3 backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
		vec4 transmittance_color,
		float transmittance_depth,
		float transmittance_boost,
		float transmittance_z,
#endif
#ifdef LIGHT_RIM_USED
		float rim, float rim_tint,
#endif
#ifdef LIGHT_CLEARCOAT_USED
		float clearcoat, float clearcoat_roughness, vec3 vertex_normal,
#endif
#ifdef LIGHT_ANISOTROPY_USED
		vec3 B, vec3 T, float anisotropy,
#endif
		inout vec3 diffuse_light, inout vec3 specular_light) {

	vec4 orms_unpacked = unpackUnorm4x8(orms);

	float roughness = orms_unpacked.y;
	float metallic = orms_unpacked.z;

#if defined(LIGHT_CODE_USED)
	// light is written by the light shader

	mat4 inv_view_matrix = scene_data_block.data.inv_view_matrix;

#ifdef USING_MOBILE_RENDERER
	mat4 read_model_matrix = instances.data[draw_call.instance_index].transform;
#else
	mat4 read_model_matrix = instances.data[instance_index_interp].transform;
#endif

	mat4 read_view_matrix = scene_data_block.data.view_matrix;

#undef projection_matrix
#define projection_matrix scene_data_block.data.projection_matrix
#undef inv_projection_matrix
#define inv_projection_matrix scene_data_block.data.inv_projection_matrix

	vec2 read_viewport_size = scene_data_block.data.viewport_size;

	vec3 normal = N;
	vec3 light = L_diff;
	vec3 view = V;

#CODE : LIGHT

#else

	float NdotL = min(A + dot(N, L_diff), 1.0);
	float cNdotL = max(NdotL, 0.0); // clamped NdotL
	float NdotL_spec = min(A + dot(N, L_spec), 1.0);
	float cNdotL_spec = max(NdotL_spec, 0.0); // clamped NdotL
	float NdotV = dot(N, V);
	float cNdotV = max(NdotV, 1e-4);

#if defined(DIFFUSE_BURLEY) || defined(SPECULAR_SCHLICK_GGX) || defined(LIGHT_CLEARCOAT_USED)
	vec3 H = normalize(V + L_diff);
	vec3 H_spec = normalize(V + L_spec);
#endif

#if defined(SPECULAR_SCHLICK_GGX)
	float cNdotH = clamp(A + dot(N, H), 0.0, 1.0);
	float cNdotH_spec = clamp(A + dot(N, H_spec), 0.0, 1.0);
#endif

#if defined(DIFFUSE_BURLEY) || defined(SPECULAR_SCHLICK_GGX) || defined(LIGHT_CLEARCOAT_USED)
	float cLdotH = clamp(A + dot(L_diff, H), 0.0, 1.0);
	float cLdotH_spec = clamp(A + dot(L_spec, H_spec), 0.0, 1.0);
#endif

	if (metallic < 1.0) {
		float diffuse_brdf_NL; // BRDF times N.L for calculating diffuse radiance

#if defined(DIFFUSE_LAMBERT_WRAP)
		// Energy conserving lambert wrap shader.
		// https://web.archive.org/web/20210228210901/http://blog.stevemcauley.com/2011/12/03/energy-conserving-wrapped-diffuse/
		diffuse_brdf_NL = max(0.0, (NdotL + roughness) / ((1.0 + roughness) * (1.0 + roughness))) * (1.0 / M_PI);
#elif defined(DIFFUSE_TOON)

		diffuse_brdf_NL = smoothstep(-roughness, max(roughness, 0.01), NdotL) * (1.0 / M_PI);

#elif defined(DIFFUSE_BURLEY)

		{
			float FD90_minus_1 = 2.0 * cLdotH * cLdotH * roughness - 0.5;
			float FdV = 1.0 + FD90_minus_1 * SchlickFresnel(cNdotV);
			float FdL = 1.0 + FD90_minus_1 * SchlickFresnel(cNdotL);
			diffuse_brdf_NL = (1.0 / M_PI) * FdV * FdL * cNdotL;
			/*
			float energyBias = mix(roughness, 0.0, 0.5);
			float energyFactor = mix(roughness, 1.0, 1.0 / 1.51);
			float fd90 = energyBias + 2.0 * VoH * VoH * roughness;
			float f0 = 1.0;
			float lightScatter = f0 + (fd90 - f0) * pow(1.0 - cNdotL, 5.0);
			float viewScatter = f0 + (fd90 - f0) * pow(1.0 - cNdotV, 5.0);

			diffuse_brdf_NL = lightScatter * viewScatter * energyFactor;
			*/
		}
#else
		// lambert
		diffuse_brdf_NL = cNdotL * (1.0 / M_PI);
#endif

		diffuse_light += light_color * diffuse_brdf_NL * attenuation;

#if defined(LIGHT_BACKLIGHT_USED)
		diffuse_light += light_color * (vec3(1.0 / M_PI) - diffuse_brdf_NL) * backlight * attenuation;
#endif

#if defined(LIGHT_RIM_USED)
		// Epsilon min to prevent pow(0, 0) singularity which results in undefined behavior.
		float rim_light = pow(max(1e-4, 1.0 - cNdotV), max(0.0, (1.0 - roughness) * 16.0));
		diffuse_light += rim_light * rim * mix(vec3(1.0), albedo, rim_tint) * light_color;
#endif

#ifdef LIGHT_TRANSMITTANCE_USED

		{
#ifdef SSS_MODE_SKIN
			float scale = 8.25 / transmittance_depth;
			float d = scale * abs(transmittance_z);
			float dd = -d * d;
			vec3 profile = vec3(0.233, 0.455, 0.649) * exp(dd / 0.0064) +
					vec3(0.1, 0.336, 0.344) * exp(dd / 0.0484) +
					vec3(0.118, 0.198, 0.0) * exp(dd / 0.187) +
					vec3(0.113, 0.007, 0.007) * exp(dd / 0.567) +
					vec3(0.358, 0.004, 0.0) * exp(dd / 1.99) +
					vec3(0.078, 0.0, 0.0) * exp(dd / 7.41);

			diffuse_light += profile * transmittance_color.a * light_color * clamp(transmittance_boost - NdotL, 0.0, 1.0) * (1.0 / M_PI);
#else

			float scale = 8.25 / transmittance_depth;
			float d = scale * abs(transmittance_z);
			float dd = -d * d;
			diffuse_light += exp(dd) * transmittance_color.rgb * transmittance_color.a * light_color * clamp(transmittance_boost - NdotL, 0.0, 1.0) * (1.0 / M_PI);
#endif
		}
#else

#endif //LIGHT_TRANSMITTANCE_USED
	}

	if (roughness > 0.0) { // FIXME: roughness == 0 should not disable specular light entirely

		// D

#if defined(SPECULAR_TOON)

		vec3 R = normalize(-reflect(L_diff, N));
		float RdotV = dot(R, V);
		float mid = 1.0 - roughness;
		mid *= mid;
		float intensity = smoothstep(mid - roughness * 0.5, mid + roughness * 0.5, RdotV) * mid;
		diffuse_light += light_color * intensity * attenuation * specular_amount; // write to diffuse_light, as in toon shading you generally want no reflection

#elif defined(SPECULAR_DISABLED)
		// none..

#elif defined(SPECULAR_SCHLICK_GGX)
		// shlick+ggx as default
		float alpha_ggx = roughness * roughness;
#if defined(LIGHT_ANISOTROPY_USED)

		float aspect = sqrt(1.0 - anisotropy * 0.9);
		float ax = alpha_ggx / aspect;
		float ay = alpha_ggx * aspect;
		float XdotH = dot(T, H_spec);
		float YdotH = dot(B, H_spec);
		float D = D_GGX_anisotropic(cNdotH_spec, ax, ay, XdotH, YdotH);
		float G = V_GGX_anisotropic(ax, ay, dot(T, V), dot(T, L_spec), dot(B, V), dot(B, L_spec), cNdotV, cNdotL);
#else // LIGHT_ANISOTROPY_USED
		float D = D_GGX(cNdotH_spec, alpha_ggx);
		float G = V_GGX(cNdotL_spec, cNdotV, alpha_ggx);
#endif // LIGHT_ANISOTROPY_USED
	   // F
		float cLdotH5 = SchlickFresnel(cLdotH_spec);
		// Calculate Fresnel using specular occlusion term from Filament:
		// https://google.github.io/filament/Filament.html#lighting/occlusion/specularocclusion
		float f90 = clamp(dot(f0, vec3(50.0 * 0.33)), metallic, 1.0);
		vec3 F = f0 + (f90 - f0) * cLdotH5;

		vec3 specular_brdf_NL = cNdotL * D * F * G;

		specular_light += specular_brdf_NL * light_color * attenuation * specular_amount;
#endif

#if defined(LIGHT_CLEARCOAT_USED)
		// Clearcoat ignores normal_map, use vertex normal instead
		float ccNdotL = max(min(A + dot(vertex_normal, L_spec), 1.0), 0.0);
		float ccNdotH = clamp(A + dot(vertex_normal, H_spec), 0.0, 1.0);
		float ccNdotV = max(dot(vertex_normal, V), 1e-4);

#if !defined(SPECULAR_SCHLICK_GGX)
		float cLdotH5 = SchlickFresnel(cLdotH_spec);
#endif
		float Dr = D_GGX(ccNdotH, mix(0.001, 0.1, clearcoat_roughness));
		float Gr = 0.25 / (cLdotH_spec * cLdotH_spec);
		float Fr = mix(.04, 1.0, cLdotH5);
		float clearcoat_specular_brdf_NL = clearcoat * Gr * Fr * Dr * cNdotL;

		specular_light += clearcoat_specular_brdf_NL * light_color * attenuation * specular_amount;
		// TODO: Clearcoat adds light to the scene right now (it is non-energy conserving), both diffuse and specular need to be scaled by (1.0 - FR)
		// but to do so we need to rearrange this entire function
#endif // LIGHT_CLEARCOAT_USED
	}

#ifdef USE_SHADOW_TO_OPACITY
	alpha = min(alpha, clamp(1.0 - attenuation, 0.0, 1.0));
#endif

#endif //defined(LIGHT_CODE_USED)
}

#ifndef SHADOWS_DISABLED

// Interleaved Gradient Noise
// https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare
float quick_hash(vec2 pos) {
	const vec3 magic = vec3(0.06711056f, 0.00583715f, 52.9829189f);
	return fract(magic.z * fract(dot(pos, magic.xy)));
}

float sample_directional_pcf_shadow(texture2D shadow, vec2 shadow_pixel_size, vec4 coord) {
	vec2 pos = coord.xy;
	float depth = coord.z;

	//if only one sample is taken, take it from the center
	if (sc_directional_soft_shadow_samples == 0) {
		return textureProj(sampler2DShadow(shadow, shadow_sampler), vec4(pos, depth, 1.0));
	}

	mat2 disk_rotation;
	{
		float r = quick_hash(gl_FragCoord.xy) * 2.0 * M_PI;
		float sr = sin(r);
		float cr = cos(r);
		disk_rotation = mat2(vec2(cr, -sr), vec2(sr, cr));
	}

	float avg = 0.0;

	for (uint i = 0; i < sc_directional_soft_shadow_samples; i++) {
		avg += textureProj(sampler2DShadow(shadow, shadow_sampler), vec4(pos + shadow_pixel_size * (disk_rotation * scene_data_block.data.directional_soft_shadow_kernel[i].xy), depth, 1.0));
	}

	return avg * (1.0 / float(sc_directional_soft_shadow_samples));
}

float sample_pcf_shadow(texture2D shadow, vec2 shadow_pixel_size, vec3 coord) {
	vec2 pos = coord.xy;
	float depth = coord.z;

	//if only one sample is taken, take it from the center
	if (sc_soft_shadow_samples == 0) {
		return textureProj(sampler2DShadow(shadow, shadow_sampler), vec4(pos, depth, 1.0));
	}

	mat2 disk_rotation;
	{
		float r = quick_hash(gl_FragCoord.xy) * 2.0 * M_PI;
		float sr = sin(r);
		float cr = cos(r);
		disk_rotation = mat2(vec2(cr, -sr), vec2(sr, cr));
	}

	float avg = 0.0;

	for (uint i = 0; i < sc_soft_shadow_samples; i++) {
		avg += textureProj(sampler2DShadow(shadow, shadow_sampler), vec4(pos + shadow_pixel_size * (disk_rotation * scene_data_block.data.soft_shadow_kernel[i].xy), depth, 1.0));
	}

	return avg * (1.0 / float(sc_soft_shadow_samples));
}

float sample_omni_pcf_shadow(texture2D shadow, float blur_scale, vec2 coord, vec4 uv_rect, vec2 flip_offset, float depth) {
	//if only one sample is taken, take it from the center
	if (sc_soft_shadow_samples == 0) {
		vec2 pos = coord * 0.5 + 0.5;
		pos = uv_rect.xy + pos * uv_rect.zw;
		return textureProj(sampler2DShadow(shadow, shadow_sampler), vec4(pos, depth, 1.0));
	}

	mat2 disk_rotation;
	{
		float r = quick_hash(gl_FragCoord.xy) * 2.0 * M_PI;
		float sr = sin(r);
		float cr = cos(r);
		disk_rotation = mat2(vec2(cr, -sr), vec2(sr, cr));
	}

	float avg = 0.0;
	vec2 offset_scale = blur_scale * 2.0 * scene_data_block.data.shadow_atlas_pixel_size / uv_rect.zw;

	for (uint i = 0; i < sc_soft_shadow_samples; i++) {
		vec2 offset = offset_scale * (disk_rotation * scene_data_block.data.soft_shadow_kernel[i].xy);
		vec2 sample_coord = coord + offset;

		float sample_coord_length_sqaured = dot(sample_coord, sample_coord);
		bool do_flip = sample_coord_length_sqaured > 1.0;

		if (do_flip) {
			float len = sqrt(sample_coord_length_sqaured);
			sample_coord = sample_coord * (2.0 / len - 1.0);
		}

		sample_coord = sample_coord * 0.5 + 0.5;
		sample_coord = uv_rect.xy + sample_coord * uv_rect.zw;

		if (do_flip) {
			sample_coord += flip_offset;
		}
		avg += textureProj(sampler2DShadow(shadow, shadow_sampler), vec4(sample_coord, depth, 1.0));
	}

	return avg * (1.0 / float(sc_soft_shadow_samples));
}

float sample_directional_soft_shadow(texture2D shadow, vec3 pssm_coord, vec2 tex_scale) {
	//find blocker
	float blocker_count = 0.0;
	float blocker_average = 0.0;

	mat2 disk_rotation;
	{
		float r = quick_hash(gl_FragCoord.xy) * 2.0 * M_PI;
		float sr = sin(r);
		float cr = cos(r);
		disk_rotation = mat2(vec2(cr, -sr), vec2(sr, cr));
	}

	for (uint i = 0; i < sc_directional_penumbra_shadow_samples; i++) {
		vec2 suv = pssm_coord.xy + (disk_rotation * scene_data_block.data.directional_penumbra_shadow_kernel[i].xy) * tex_scale;
		float d = textureLod(sampler2D(shadow, SAMPLER_LINEAR_CLAMP), suv, 0.0).r;
		if (d > pssm_coord.z) {
			blocker_average += d;
			blocker_count += 1.0;
		}
	}

	if (blocker_count > 0.0) {
		//blockers found, do soft shadow
		blocker_average /= blocker_count;
		float penumbra = (-pssm_coord.z + blocker_average) / (1.0 - blocker_average);
		tex_scale *= penumbra;

		float s = 0.0;
		for (uint i = 0; i < sc_directional_penumbra_shadow_samples; i++) {
			vec2 suv = pssm_coord.xy + (disk_rotation * scene_data_block.data.directional_penumbra_shadow_kernel[i].xy) * tex_scale;
			s += textureProj(sampler2DShadow(shadow, shadow_sampler), vec4(suv, pssm_coord.z, 1.0));
		}

		return s / float(sc_directional_penumbra_shadow_samples);

	} else {
		//no blockers found, so no shadow
		return 1.0;
	}
}

#endif // SHADOWS_DISABLED

float get_omni_attenuation(float distance, float inv_range, float decay) {
	float nd = distance * inv_range;
	nd *= nd;
	nd *= nd; // nd^4
	nd = max(1.0 - nd, 0.0);
	nd *= nd; // nd^2
	return nd * pow(max(distance, 0.0001), -decay);
}

float light_process_omni_shadow(uint idx, vec3 vertex, vec3 normal) {
#ifndef SHADOWS_DISABLED
	if (omni_lights.data[idx].shadow_opacity > 0.001) {
		// there is a shadowmap
		vec2 texel_size = scene_data_block.data.shadow_atlas_pixel_size;
		vec4 base_uv_rect = omni_lights.data[idx].atlas_rect;
		base_uv_rect.xy += texel_size;
		base_uv_rect.zw -= texel_size * 2.0;

		// Omni lights use direction.xy to store to store the offset between the two paraboloid regions
		vec2 flip_offset = omni_lights.data[idx].direction.xy;

		vec3 local_vert = (omni_lights.data[idx].shadow_matrix * vec4(vertex, 1.0)).xyz;

		float shadow_len = length(local_vert); //need to remember shadow len from here
		vec3 shadow_dir = normalize(local_vert);

		vec3 local_normal = normalize(mat3(omni_lights.data[idx].shadow_matrix) * normal);
		vec3 normal_bias = local_normal * omni_lights.data[idx].shadow_normal_bias * (1.0 - abs(dot(local_normal, shadow_dir)));

		float shadow;

		if (sc_use_light_soft_shadows && omni_lights.data[idx].soft_shadow_size > 0.0) {
			//soft shadow

			//find blocker

			float blocker_count = 0.0;
			float blocker_average = 0.0;

			mat2 disk_rotation;
			{
				float r = quick_hash(gl_FragCoord.xy) * 2.0 * M_PI;
				float sr = sin(r);
				float cr = cos(r);
				disk_rotation = mat2(vec2(cr, -sr), vec2(sr, cr));
			}

			vec3 basis_normal = shadow_dir;
			vec3 v0 = abs(basis_normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
			vec3 tangent = normalize(cross(v0, basis_normal));
			vec3 bitangent = normalize(cross(tangent, basis_normal));
			float z_norm = 1.0 - shadow_len * omni_lights.data[idx].inv_radius;

			tangent *= omni_lights.data[idx].soft_shadow_size * omni_lights.data[idx].soft_shadow_scale;
			bitangent *= omni_lights.data[idx].soft_shadow_size * omni_lights.data[idx].soft_shadow_scale;

			for (uint i = 0; i < sc_penumbra_shadow_samples; i++) {
				vec2 disk = disk_rotation * scene_data_block.data.penumbra_shadow_kernel[i].xy;

				vec3 pos = local_vert + tangent * disk.x + bitangent * disk.y;

				pos = normalize(pos);

				vec4 uv_rect = base_uv_rect;

				if (pos.z >= 0.0) {
					uv_rect.xy += flip_offset;
				}

				pos.z = 1.0 + abs(pos.z);
				pos.xy /= pos.z;

				pos.xy = pos.xy * 0.5 + 0.5;
				pos.xy = uv_rect.xy + pos.xy * uv_rect.zw;

				float d = textureLod(sampler2D(shadow_atlas, SAMPLER_LINEAR_CLAMP), pos.xy, 0.0).r;
				if (d > z_norm) {
					blocker_average += d;
					blocker_count += 1.0;
				}
			}

			if (blocker_count > 0.0) {
				//blockers found, do soft shadow
				blocker_average /= blocker_count;
				float penumbra = (-z_norm + blocker_average) / (1.0 - blocker_average);
				tangent *= penumbra;
				bitangent *= penumbra;

				z_norm += omni_lights.data[idx].inv_radius * omni_lights.data[idx].shadow_bias;

				shadow = 0.0;
				for (uint i = 0; i < sc_penumbra_shadow_samples; i++) {
					vec2 disk = disk_rotation * scene_data_block.data.penumbra_shadow_kernel[i].xy;
					vec3 pos = local_vert + tangent * disk.x + bitangent * disk.y;

					pos = normalize(pos);
					pos = normalize(pos + normal_bias);

					vec4 uv_rect = base_uv_rect;

					if (pos.z >= 0.0) {
						uv_rect.xy += flip_offset;
					}

					pos.z = 1.0 + abs(pos.z);
					pos.xy /= pos.z;

					pos.xy = pos.xy * 0.5 + 0.5;
					pos.xy = uv_rect.xy + pos.xy * uv_rect.zw;
					shadow += textureProj(sampler2DShadow(shadow_atlas, shadow_sampler), vec4(pos.xy, z_norm, 1.0));
				}

				shadow /= float(sc_penumbra_shadow_samples);
				shadow = mix(1.0, shadow, omni_lights.data[idx].shadow_opacity);

			} else {
				//no blockers found, so no shadow
				shadow = 1.0;
			}
		} else {
			vec4 uv_rect = base_uv_rect;

			vec3 shadow_sample = normalize(shadow_dir + normal_bias);
			if (shadow_sample.z >= 0.0) {
				uv_rect.xy += flip_offset;
				flip_offset *= -1.0;
			}

			shadow_sample.z = 1.0 + abs(shadow_sample.z);
			vec2 pos = shadow_sample.xy / shadow_sample.z;
			float depth = shadow_len - omni_lights.data[idx].shadow_bias; // shadow_len = distance from vertex to light
			depth *= omni_lights.data[idx].inv_radius; // depth = how many light radii away from light (more than 1 = no light)
			depth = 1.0 - depth; // shadow map depth range = radius of light (white or 1.0 on map)
			shadow = mix(1.0, sample_omni_pcf_shadow(shadow_atlas, omni_lights.data[idx].soft_shadow_scale / shadow_sample.z, pos, uv_rect, flip_offset, depth), omni_lights.data[idx].shadow_opacity);
		}

		return shadow;
	}
#endif

	return 1.0;
}

void light_process_omni(uint idx, vec3 vertex, vec3 eye_vec, vec3 normal, vec3 vertex_ddx, vec3 vertex_ddy, vec3 f0, uint orms, float shadow, vec3 albedo, inout float alpha,
#ifdef LIGHT_BACKLIGHT_USED
		vec3 backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
		vec4 transmittance_color,
		float transmittance_depth,
		float transmittance_boost,
#endif
#ifdef LIGHT_RIM_USED
		float rim, float rim_tint,
#endif
#ifdef LIGHT_CLEARCOAT_USED
		float clearcoat, float clearcoat_roughness, vec3 vertex_normal,
#endif
#ifdef LIGHT_ANISOTROPY_USED
		vec3 binormal, vec3 tangent, float anisotropy,
#endif
		inout vec3 diffuse_light, inout vec3 specular_light) {
	vec3 light_rel_vec = omni_lights.data[idx].position - vertex;
	float light_length = length(light_rel_vec);
	float omni_attenuation = get_omni_attenuation(light_length, omni_lights.data[idx].inv_radius, omni_lights.data[idx].attenuation);
	float light_attenuation = omni_attenuation;
	vec3 color = omni_lights.data[idx].color;

	float size_A = 0.0;

	if (sc_use_light_soft_shadows && omni_lights.data[idx].size > 0.0) {
		float t = omni_lights.data[idx].size / max(0.001, light_length);
		size_A = max(0.0, 1.0 - 1 / sqrt(1 + t * t));
	}

#ifdef LIGHT_TRANSMITTANCE_USED
	float transmittance_z = transmittance_depth; //no transmittance by default
	transmittance_color.a *= light_attenuation;
	{
		vec4 clamp_rect = omni_lights.data[idx].atlas_rect;

		//redo shadowmapping, but shrink the model a bit to avoid artifacts
		vec4 splane = (omni_lights.data[idx].shadow_matrix * vec4(vertex - normalize(normal_interp) * omni_lights.data[idx].transmittance_bias, 1.0));

		float shadow_len = length(splane.xyz);
		splane.xyz = normalize(splane.xyz);

		if (splane.z >= 0.0) {
			splane.z += 1.0;
			clamp_rect.y += clamp_rect.w;
		} else {
			splane.z = 1.0 - splane.z;
		}

		splane.xy /= splane.z;

		splane.xy = splane.xy * 0.5 + 0.5;
		splane.z = shadow_len * omni_lights.data[idx].inv_radius;
		splane.xy = clamp_rect.xy + splane.xy * clamp_rect.zw;
		//		splane.xy = clamp(splane.xy,clamp_rect.xy + scene_data_block.data.shadow_atlas_pixel_size,clamp_rect.xy + clamp_rect.zw - scene_data_block.data.shadow_atlas_pixel_size );
		splane.w = 1.0; //needed? i think it should be 1 already

		float shadow_z = textureLod(sampler2D(shadow_atlas, SAMPLER_LINEAR_CLAMP), splane.xy, 0.0).r;
		transmittance_z = (splane.z - shadow_z) / omni_lights.data[idx].inv_radius;
	}
#endif

	if (sc_use_light_projector && omni_lights.data[idx].projector_rect != vec4(0.0)) {
		vec3 local_v = (omni_lights.data[idx].shadow_matrix * vec4(vertex, 1.0)).xyz;
		local_v = normalize(local_v);

		vec4 atlas_rect = omni_lights.data[idx].projector_rect;

		if (local_v.z >= 0.0) {
			atlas_rect.y += atlas_rect.w;
		}

		local_v.z = 1.0 + abs(local_v.z);

		local_v.xy /= local_v.z;
		local_v.xy = local_v.xy * 0.5 + 0.5;
		vec2 proj_uv = local_v.xy * atlas_rect.zw;

		if (sc_projector_use_mipmaps) {
			vec2 proj_uv_ddx;
			vec2 proj_uv_ddy;
			{
				vec3 local_v_ddx = (omni_lights.data[idx].shadow_matrix * vec4(vertex + vertex_ddx, 1.0)).xyz;
				local_v_ddx = normalize(local_v_ddx);

				if (local_v_ddx.z >= 0.0) {
					local_v_ddx.z += 1.0;
				} else {
					local_v_ddx.z = 1.0 - local_v_ddx.z;
				}

				local_v_ddx.xy /= local_v_ddx.z;
				local_v_ddx.xy = local_v_ddx.xy * 0.5 + 0.5;

				proj_uv_ddx = local_v_ddx.xy * atlas_rect.zw - proj_uv;

				vec3 local_v_ddy = (omni_lights.data[idx].shadow_matrix * vec4(vertex + vertex_ddy, 1.0)).xyz;
				local_v_ddy = normalize(local_v_ddy);

				if (local_v_ddy.z >= 0.0) {
					local_v_ddy.z += 1.0;
				} else {
					local_v_ddy.z = 1.0 - local_v_ddy.z;
				}

				local_v_ddy.xy /= local_v_ddy.z;
				local_v_ddy.xy = local_v_ddy.xy * 0.5 + 0.5;

				proj_uv_ddy = local_v_ddy.xy * atlas_rect.zw - proj_uv;
			}

			vec4 proj = textureGrad(sampler2D(decal_atlas_srgb, light_projector_sampler), proj_uv + atlas_rect.xy, proj_uv_ddx, proj_uv_ddy);
			color *= proj.rgb * proj.a;
		} else {
			vec4 proj = textureLod(sampler2D(decal_atlas_srgb, light_projector_sampler), proj_uv + atlas_rect.xy, 0.0);
			color *= proj.rgb * proj.a;
		}
	}

	light_attenuation *= shadow;
	vec3 light_vec = normalize(light_rel_vec);
	light_compute(normal, light_vec, light_vec, eye_vec, size_A, color, false, light_attenuation, f0, orms, omni_lights.data[idx].specular_amount, albedo, alpha,
#ifdef LIGHT_BACKLIGHT_USED
			backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
			transmittance_color,
			transmittance_depth,
			transmittance_boost,
			transmittance_z,
#endif
#ifdef LIGHT_RIM_USED
			rim * omni_attenuation, rim_tint,
#endif
#ifdef LIGHT_CLEARCOAT_USED
			clearcoat, clearcoat_roughness, vertex_normal,
#endif
#ifdef LIGHT_ANISOTROPY_USED
			binormal, tangent, anisotropy,
#endif
			diffuse_light,
			specular_light);
}

float light_process_spot_shadow(uint idx, vec3 vertex, vec3 normal) {
#ifndef SHADOWS_DISABLED
	if (spot_lights.data[idx].shadow_opacity > 0.001) {
		vec3 light_rel_vec = spot_lights.data[idx].position - vertex;
		float light_length = length(light_rel_vec);
		vec3 spot_dir = spot_lights.data[idx].direction;

		vec3 shadow_dir = light_rel_vec / light_length;
		vec3 normal_bias = normal * light_length * spot_lights.data[idx].shadow_normal_bias * (1.0 - abs(dot(normal, shadow_dir)));

		//there is a shadowmap
		vec4 v = vec4(vertex + normal_bias, 1.0);

		vec4 splane = (spot_lights.data[idx].shadow_matrix * v);
		splane.z += spot_lights.data[idx].shadow_bias / (light_length * spot_lights.data[idx].inv_radius);
		splane /= splane.w;

		float shadow;
		if (sc_use_light_soft_shadows && spot_lights.data[idx].soft_shadow_size > 0.0) {
			//soft shadow

			//find blocker
			float z_norm = dot(spot_dir, -light_rel_vec) * spot_lights.data[idx].inv_radius;

			vec2 shadow_uv = splane.xy * spot_lights.data[idx].atlas_rect.zw + spot_lights.data[idx].atlas_rect.xy;

			float blocker_count = 0.0;
			float blocker_average = 0.0;

			mat2 disk_rotation;
			{
				float r = quick_hash(gl_FragCoord.xy) * 2.0 * M_PI;
				float sr = sin(r);
				float cr = cos(r);
				disk_rotation = mat2(vec2(cr, -sr), vec2(sr, cr));
			}

			float uv_size = spot_lights.data[idx].soft_shadow_size * z_norm * spot_lights.data[idx].soft_shadow_scale;
			vec2 clamp_max = spot_lights.data[idx].atlas_rect.xy + spot_lights.data[idx].atlas_rect.zw;
			for (uint i = 0; i < sc_penumbra_shadow_samples; i++) {
				vec2 suv = shadow_uv + (disk_rotation * scene_data_block.data.penumbra_shadow_kernel[i].xy) * uv_size;
				suv = clamp(suv, spot_lights.data[idx].atlas_rect.xy, clamp_max);
				float d = textureLod(sampler2D(shadow_atlas, SAMPLER_LINEAR_CLAMP), suv, 0.0).r;
				if (d > splane.z) {
					blocker_average += d;
					blocker_count += 1.0;
				}
			}

			if (blocker_count > 0.0) {
				//blockers found, do soft shadow
				blocker_average /= blocker_count;
				float penumbra = (-z_norm + blocker_average) / (1.0 - blocker_average);
				uv_size *= penumbra;

				shadow = 0.0;
				for (uint i = 0; i < sc_penumbra_shadow_samples; i++) {
					vec2 suv = shadow_uv + (disk_rotation * scene_data_block.data.penumbra_shadow_kernel[i].xy) * uv_size;
					suv = clamp(suv, spot_lights.data[idx].atlas_rect.xy, clamp_max);
					shadow += textureProj(sampler2DShadow(shadow_atlas, shadow_sampler), vec4(suv, splane.z, 1.0));
				}

				shadow /= float(sc_penumbra_shadow_samples);
				shadow = mix(1.0, shadow, spot_lights.data[idx].shadow_opacity);

			} else {
				//no blockers found, so no shadow
				shadow = 1.0;
			}
		} else {
			//hard shadow
			vec3 shadow_uv = vec3(splane.xy * spot_lights.data[idx].atlas_rect.zw + spot_lights.data[idx].atlas_rect.xy, splane.z);
			shadow = mix(1.0, sample_pcf_shadow(shadow_atlas, spot_lights.data[idx].soft_shadow_scale * scene_data_block.data.shadow_atlas_pixel_size, shadow_uv), spot_lights.data[idx].shadow_opacity);
		}

		return shadow;
	}

#endif // SHADOWS_DISABLED

	return 1.0;
}

vec2 normal_to_panorama(vec3 n) {
	n = normalize(n);
	vec2 panorama_coords = vec2(atan(n.x, n.z), acos(-n.y));

	if (panorama_coords.x < 0.0) {
		panorama_coords.x += M_PI * 2.0;
	}

	panorama_coords /= vec2(M_PI * 2.0, M_PI);
	return panorama_coords;
}

void light_process_spot(uint idx, vec3 vertex, vec3 eye_vec, vec3 normal, vec3 vertex_ddx, vec3 vertex_ddy, vec3 f0, uint orms, float shadow, vec3 albedo, inout float alpha,
#ifdef LIGHT_BACKLIGHT_USED
		vec3 backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
		vec4 transmittance_color,
		float transmittance_depth,
		float transmittance_boost,
#endif
#ifdef LIGHT_RIM_USED
		float rim, float rim_tint,
#endif
#ifdef LIGHT_CLEARCOAT_USED
		float clearcoat, float clearcoat_roughness, vec3 vertex_normal,
#endif
#ifdef LIGHT_ANISOTROPY_USED
		vec3 binormal, vec3 tangent, float anisotropy,
#endif
		inout vec3 diffuse_light,
		inout vec3 specular_light) {
	vec3 light_rel_vec = spot_lights.data[idx].position - vertex;
	float light_length = length(light_rel_vec);
	float spot_attenuation = get_omni_attenuation(light_length, spot_lights.data[idx].inv_radius, spot_lights.data[idx].attenuation);
	vec3 spot_dir = spot_lights.data[idx].direction;

	// This conversion to a highp float is crucial to prevent light leaking
	// due to precision errors in the following calculations (cone angle is mediump).
	highp float cone_angle = spot_lights.data[idx].cone_angle;
	float scos = max(dot(-normalize(light_rel_vec), spot_dir), cone_angle);
	float spot_rim = max(0.0001, (1.0 - scos) / (1.0 - cone_angle));

	spot_attenuation *= 1.0 - pow(spot_rim, spot_lights.data[idx].cone_attenuation);
	float light_attenuation = spot_attenuation;
	vec3 color = spot_lights.data[idx].color;
	float specular_amount = spot_lights.data[idx].specular_amount;

	float size_A = 0.0;

	if (sc_use_light_soft_shadows && spot_lights.data[idx].size > 0.0) {
		float t = spot_lights.data[idx].size / max(0.001, light_length);
		size_A = max(0.0, 1.0 - 1 / sqrt(1 + t * t));
	}

#ifdef LIGHT_TRANSMITTANCE_USED
	float transmittance_z = transmittance_depth;
	transmittance_color.a *= light_attenuation;
	{
		vec4 splane = (spot_lights.data[idx].shadow_matrix * vec4(vertex - normalize(normal_interp) * spot_lights.data[idx].transmittance_bias, 1.0));
		splane /= splane.w;
		splane.xy = splane.xy * spot_lights.data[idx].atlas_rect.zw + spot_lights.data[idx].atlas_rect.xy;

		float shadow_z = textureLod(sampler2D(shadow_atlas, SAMPLER_LINEAR_CLAMP), splane.xy, 0.0).r;

		shadow_z = shadow_z * 2.0 - 1.0;
		float z_far = 1.0 / spot_lights.data[idx].inv_radius;
		float z_near = 0.01;
		shadow_z = 2.0 * z_near * z_far / (z_far + z_near - shadow_z * (z_far - z_near));

		//distance to light plane
		float z = dot(spot_dir, -light_rel_vec);
		transmittance_z = z - shadow_z;
	}
#endif //LIGHT_TRANSMITTANCE_USED

	if (sc_use_light_projector && spot_lights.data[idx].projector_rect != vec4(0.0)) {
		vec4 splane = (spot_lights.data[idx].shadow_matrix * vec4(vertex, 1.0));
		splane /= splane.w;

		vec2 proj_uv = splane.xy * spot_lights.data[idx].projector_rect.zw;

		if (sc_projector_use_mipmaps) {
			//ensure we have proper mipmaps
			vec4 splane_ddx = (spot_lights.data[idx].shadow_matrix * vec4(vertex + vertex_ddx, 1.0));
			splane_ddx /= splane_ddx.w;
			vec2 proj_uv_ddx = splane_ddx.xy * spot_lights.data[idx].projector_rect.zw - proj_uv;

			vec4 splane_ddy = (spot_lights.data[idx].shadow_matrix * vec4(vertex + vertex_ddy, 1.0));
			splane_ddy /= splane_ddy.w;
			vec2 proj_uv_ddy = splane_ddy.xy * spot_lights.data[idx].projector_rect.zw - proj_uv;

			vec4 proj = textureGrad(sampler2D(decal_atlas_srgb, light_projector_sampler), proj_uv + spot_lights.data[idx].projector_rect.xy, proj_uv_ddx, proj_uv_ddy);
			color *= proj.rgb * proj.a;
		} else {
			vec4 proj = textureLod(sampler2D(decal_atlas_srgb, light_projector_sampler), proj_uv + spot_lights.data[idx].projector_rect.xy, 0.0);
			color *= proj.rgb * proj.a;
		}
	}
	light_attenuation *= shadow;

	vec3 light_vec = normalize(light_rel_vec);
	light_compute(normal, light_vec, light_vec, eye_vec, size_A, color, false, light_attenuation, f0, orms, spot_lights.data[idx].specular_amount, albedo, alpha,
#ifdef LIGHT_BACKLIGHT_USED
			backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
			transmittance_color,
			transmittance_depth,
			transmittance_boost,
			transmittance_z,
#endif
#ifdef LIGHT_RIM_USED
			rim * spot_attenuation, rim_tint,
#endif
#ifdef LIGHT_CLEARCOAT_USED
			clearcoat, clearcoat_roughness, vertex_normal,
#endif
#ifdef LIGHT_ANISOTROPY_USED
			binormal, tangent, anisotropy,
#endif
			diffuse_light, specular_light);
}

float light_process_area_shadow(uint idx, vec3 vertex, vec3 normal) {
#ifndef SHADOWS_DISABLED
	if (area_lights.data[idx].shadow_opacity > 0.001) {
		// there is a shadowmap

		vec2 texel_size = scene_data_block.data.area_shadow_atlas_pixel_size.xy;
		vec4 base_uv_rect = area_lights.data[idx].atlas_rect;
		// This offset is required if we decide to do soft sampling (sample kernel around actual point)
		base_uv_rect.xy += texel_size; // = 1 / 4096 = 0.000244140625
		base_uv_rect.zw -= texel_size * 2.0;

		//float quadrant_width = 0.5;
		//float quadrant_limit_x = area_lights.data[idx].atlas_rect.x >= 0.5 ? 1.0 : 0.5;

		float len_diagonal = area_lights.data[idx].size;
		vec3 world_side_a = mat3(scene_data_block.data.inv_view_matrix) * area_lights.data[idx].area_side_a;
		vec3 world_side_b = mat3(scene_data_block.data.inv_view_matrix) * area_lights.data[idx].area_side_b;
		float inv_depth_range = 1.0 / (1.0 / area_lights.data[idx].inv_radius + len_diagonal);

		float shadow_sum = 0.0;
		uint resolution = area_lights.data[idx].area_shadow_sample_resolution; // shorthand
		uint sample_count = resolution * resolution;

		for (uint i = 0; i < resolution; i++) {
			for(uint j = 0; j < resolution; j++) {
				vec2 sample_on_light = vec2(1.0 / (resolution - 1.0) * j, 1.0 / (resolution - 1.0) * i); // where is point i on the light, relative to the light's topright corner
				uint map_idx = area_lights.data[idx].map_idx[i * resolution + j]; // where is point i on the shadow map

				// TODO: area_map_subdivision needed? or can we just calculate subdivision from atlas_rect.size and size of area_shadow_atlas?
				uint row = map_idx / area_lights.data[idx].area_map_subdivision;
				uint col = map_idx % area_lights.data[idx].area_map_subdivision;

				// offset of position of point on light in world space
				vec3 sample_pos = (world_side_a + world_side_b) / 2.0 - (world_side_a * sample_on_light.x + world_side_b * sample_on_light.y);

				// shadow matrix is calculated as (view_matrix * light_sample_transform)^(-1) = inv_light_transform * inv_sample * inv_view_matrix
				mat4 sample_mat = scene_data_block.data.inv_view_matrix;
				sample_mat[3] -= vec4(sample_pos, 0.0);
				// this matrix transforms a point from viewspace to light sample local space
				mat4 shadow_sample_matrix = area_lights.data[idx].shadow_matrix * sample_mat;

				vec3 local_vert = (shadow_sample_matrix * vec4(vertex, 1.0)).xyz;

				float shadow_len = length(local_vert); //need to remember shadow len from here
				vec3 shadow_dir = normalize(local_vert);

				vec3 local_normal = normalize(mat3(shadow_sample_matrix) * normal);
				vec3 normal_bias = local_normal * area_lights.data[idx].shadow_normal_bias * (1.0 - abs(dot(local_normal, shadow_dir)));

				vec3 shadow_sample = normalize(shadow_dir + normal_bias);

				shadow_sample.z = 1.0 + abs(shadow_sample.z);
				vec2 pos = shadow_sample.xy / shadow_sample.z;
				float depth = shadow_len - area_lights.data[idx].shadow_bias; // shadow_len = distance from vertex to light
				depth *= inv_depth_range; // max depth = radius + diagonal
				depth = 1.0 - depth; // shadow map depth range = radius of light (white or 1.0 on map)

				vec4 uv_rect = base_uv_rect;
				vec2 sample_atlas_offset = vec2(col * area_lights.data[idx].atlas_rect.z, row * area_lights.data[idx].atlas_rect.w);

				// depending on the current area light sample point, select the right region on the atlas

				uv_rect.xy += sample_atlas_offset;

				pos = pos * 0.5 + 0.5;
				pos = uv_rect.xy + pos * uv_rect.zw;

				vec2 shadow_pixel_size = area_lights.data[idx].soft_shadow_scale / shadow_sample.z * scene_data_block.data.area_shadow_atlas_pixel_size.xy;

				shadow_sum += sample_pcf_shadow(area_shadow_atlas, shadow_pixel_size, vec3(pos, depth));
			}
			
		}

		float avg_shadow = shadow_sum / sample_count;

		return mix(1.0, avg_shadow, area_lights.data[idx].shadow_opacity);
	}

#endif // SHADOWS_DISABLED

	return 1.0;
}

// Component a with polynomial of degree 5 
// root mean squared error (RMSE): 0.00826631861246398
// error variance: 6.833202340276843e-05
float comp_a(float x, float y) {
	x += 1.0;
	y += 1.0;
	return (
	+ (40.022068) + (-21.786636)*y + (-6.711040)*pow(y,2) + (7.836000)*pow(y,3) + (-2.238067)*pow(y,4) + (0.207919)*pow(y,5) + (-113.968298)*x + (72.582053)*x*y + (-10.971895)*x*pow(y,2) + (-0.902927)*x*pow(y,3) + (0.364781)*x*pow(y,4) + (115.774423)*pow(x,2) + (-59.489534)*pow(x,2)*y + (8.060379)*pow(x,2)*pow(y,2) + (-0.421001)*pow(x,2)*pow(y,3) + (-57.958025)*pow(x,3) + (19.685421)*pow(x,3)*y + (-1.225889)*pow(x,3)*pow(y,2) + (15.368255)*pow(x,4) + (-2.398787)*pow(x,4)*y + (-1.802015)*pow(x,5)	);
}

// Component b with polynomial of degree 5 
// root mean squared error (RMSE): 0.00504682527171353
// error variance: 2.5470445323206342e-05
float comp_b(float x, float y) {
	x += 1.0;
	y += 1.0;
	return (
	+ (-32.371729) + (20.622668)*y + (-10.246609)*pow(y,2) + (5.042130)*pow(y,3) + (-1.433458)*pow(y,4) + (0.153338)*pow(y,5) + (98.859420)*x + (-37.623269)*x*y + (5.569236)*x*pow(y,2) + (-0.449615)*x*pow(y,3) + (0.097781)*x*pow(y,4) + (-124.243103)*pow(x,2) + (32.867644)*pow(x,2)*y + (-3.804820)*pow(x,2)*pow(y,2) + (-0.068663)*pow(x,2)*pow(y,3) + (76.969383)*pow(x,3) + (-10.908016)*pow(x,3)*y + (1.064901)*pow(x,3)*pow(y,2) + (-24.185140)*pow(x,4) + (0.910078)*pow(x,4)*y + (3.179056)*pow(x,5)	);
}

// Component c with polynomial of degree 5 
// root mean squared error (RMSE): 0.007847996395488764
// error variance: 6.159104742360464e-05
float comp_c(float x, float y) {
	x += 1.0;
	y += 1.0;
	return (
	+ (11.583158) + (16.195132)*y + (-21.187278)*pow(y,2) + (7.114936)*pow(y,3) + (-1.288357)*pow(y,4) + (0.090490)*pow(y,5) + (-52.766376)*x + (-2.864469)*x*y + (21.805870)*x*pow(y,2) + (-4.130327)*x*pow(y,3) + (0.381285)*x*pow(y,4) + (69.234074)*pow(x,2) + (-16.785586)*pow(x,2)*y + (-8.971769)*pow(x,2)*pow(y,2) + (0.573957)*pow(x,2)*pow(y,3) + (-40.192157)*pow(x,3) + (12.390364)*pow(x,3)*y + (1.546020)*pow(x,3)*pow(y,2) + (11.068497)*pow(x,4) + (-2.629684)*pow(x,4)*y + (-1.169427)*pow(x,5)	);
}

// Component d with polynomial of degree 5 
// root mean squared error (RMSE): 0.004446164724882146
// error variance: 1.9768380760786335e-05
float comp_d(float x, float y) {
	x += 1.0;
	y += 1.0;
	return (
	+ (56.560977) + (-26.447422)*y + (8.007063)*pow(y,2) + (-2.938371)*pow(y,3) + (0.778390)*pow(y,4) + (-0.079369)*pow(y,5) + (-174.029262)*x + (55.590827)*x*y + (-7.595242)*x*pow(y,2) + (0.586064)*x*pow(y,3) + (-0.052755)*x*pow(y,4) + (218.422343)*pow(x,2) + (-49.772234)*pow(x,2)*y + (4.705369)*pow(x,2)*pow(y,2) + (-0.110559)*pow(x,2)*pow(y,3) + (-135.668807)*pow(x,3) + (18.670509)*pow(x,3)*y + (-1.020345)*pow(x,3)*pow(y,2) + (42.042080)*pow(x,4) + (-2.349413)*pow(x,4)*y + (-5.275442)*pow(x,5)	);
}

// Component e with polynomial of degree 5 
// root mean squared error (RMSE): 0.007327637848224633
// error variance: 5.369427643473412e-05
float comp_e(float x, float y) {
	x += 1.0;
	y += 1.0;
	return (
	+ (38.635232) + (-53.895233)*y + (20.158410)*pow(y,2) + (-2.038722)*pow(y,3) + (-0.486485)*pow(y,4) + (0.086644)*pow(y,5) + (-83.242356)*x + (110.795093)*x*y + (-35.490292)*x*pow(y,2) + (4.713696)*x*pow(y,3) + (-0.117706)*x*pow(y,4) + (66.288968)*pow(x,2) + (-78.463860)*pow(x,2)*y + (16.126788)*pow(x,2)*pow(y,2) + (-1.280718)*pow(x,2)*pow(y,3) + (-23.625243)*pow(x,3) + (24.954272)*pow(x,3)*y + (-2.127836)*pow(x,3)*pow(y,2) + (3.208161)*pow(x,4) + (-3.198377)*pow(x,4)*y + (0.024170)*pow(x,5)	);
}

// Component mag with polynomial of degree 5 
// root mean squared error (RMSE): 0.009121039711851554
// error variance: 8.31933654251731e-05
float comp_mag(float x, float y) {
	x += 1.0;
	y += 1.0;
	return (
	+ (31.627915) + (-9.242613)*y + (-18.316942)*pow(y,2) + (12.243966)*pow(y,3) + (-2.814830)*pow(y,4) + (0.224888)*pow(y,5) + (-93.286029)*x + (63.261605)*x*y + (-1.313849)*x*pow(y,2) + (-4.396929)*x*pow(y,3) + (0.656442)*x*pow(y,4) + (88.299669)*pow(x,2) + (-61.632692)*pow(x,2)*y + (7.258499)*pow(x,2)*pow(y,2) + (0.096766)*pow(x,2)*pow(y,3) + (-32.964044)*pow(x,3) + (22.033172)*pow(x,3)*y + (-1.578708)*pow(x,3)*pow(y,2) + (3.228726)*pow(x,4) + (-2.787282)*pow(x,4)*y + (0.427175)*pow(x,5)	);
}

// Component fres with polynomial of degree 5 
// root mean squared error (RMSE): 0.010951139695856373
// error variance: 0.00011992746063816118
float comp_fres(float x, float y) {
	x += 1.0;
	y += 1.0;
	return (
	+ (-0.874912) + (2.405966)*y + (0.444356)*pow(y,2) + (-3.465719)*pow(y,3) + (0.463044)*pow(y,4) + (0.187222)*pow(y,5) + (0.660714)*x + (-7.715301)*x*y + (11.305963)*x*pow(y,2) + (1.880087)*x*pow(y,3) + (-1.186281)*x*pow(y,4) + (3.182991)*pow(x,2) + (-5.075303)*pow(x,2)*y + (-9.942256)*pow(x,2)*pow(y,2) + (1.837908)*pow(x,2)*pow(y,3) + (-0.220191)*pow(x,3) + (9.148312)*pow(x,3)*y + (0.226042)*pow(x,3)*pow(y,2) + (-2.274063)*pow(x,4) + (-1.613646)*pow(x,4)*y + (0.621338)*pow(x,5)	);
}


// Component a with polynomial of degree 5 
// root mean squared error (RMSE): 0.007847282004523323
// error variance: 6.15798348585156e-05
float comp_a_godot(float x, float y) {
	x += 1.0;
	y += 1.0;
	return (
	+ (40.290518) + (-18.170297)*y + (-7.811887)*pow(y,2) + (7.678010)*pow(y,3) + (-2.123667)*pow(y,4) + (0.193699)*pow(y,5) + (-118.397707)*x + (65.471498)*x*y + (-8.218117)*x*pow(y,2) + (-1.211069)*x*pow(y,3) + (0.370904)*x*pow(y,4) + (124.985159)*pow(x,2) + (-55.807916)*pow(x,2)*y + (6.778536)*pow(x,2)*pow(y,2) + (-0.339186)*pow(x,2)*pow(y,3) + (-64.848709)*pow(x,3) + (19.208735)*pow(x,3)*y + (-1.037636)*pow(x,3)*pow(y,2) + (17.502999)*pow(x,4) + (-2.461384)*pow(x,4)*y + (-2.021831)*pow(x,5)	);
}

// Component b with polynomial of degree 5 
// root mean squared error (RMSE): 0.004972015372305875
// error variance: 2.4720936862445926e-05
float comp_b_godot(float x, float y) {
	x += 1.0;
	y += 1.0;
	return (
	+ (-29.694721) + (22.317485)*y + (-10.250630)*pow(y,2) + (4.901086)*pow(y,3) + (-1.385772)*pow(y,4) + (0.150602)*pow(y,5) + (87.239706)*x + (-41.784029)*x*y + (5.866624)*x*pow(y,2) + (-0.456358)*x*pow(y,3) + (0.079256)*x*pow(y,4) + (-105.665162)*pow(x,2) + (36.451890)*pow(x,2)*y + (-3.890609)*pow(x,2)*pow(y,2) + (-0.027714)*pow(x,2)*pow(y,3) + (62.988809)*pow(x,3) + (-12.380499)*pow(x,3)*y + (1.022257)*pow(x,3)*pow(y,2) + (-19.110899)*pow(x,4) + (1.173446)*pow(x,4)*y + (2.455197)*pow(x,5)	);
}

// Component c with polynomial of degree 5 
// root mean squared error (RMSE): 0.008055450853627518
// error variance: 6.489028845520831e-05
float comp_c_godot(float x, float y) {
	x += 1.0;
	y += 1.0;
	return (
	+ (18.438152) + (17.506391)*y + (-22.001051)*pow(y,2) + (7.325071)*pow(y,3) + (-1.312048)*pow(y,4) + (0.091983)*pow(y,5) + (-77.630434)*x + (-4.326441)*x*y + (22.719814)*x*pow(y,2) + (-4.298572)*x*pow(y,3) + (0.387803)*x*pow(y,4) + (103.409544)*pow(x,2) + (-16.684480)*pow(x,2)*y + (-9.276583)*pow(x,2)*pow(y,2) + (0.611482)*pow(x,2)*pow(y,3) + (-62.684066)*pow(x,3) + (12.738066)*pow(x,3)*y + (1.570667)*pow(x,3)*pow(y,2) + (18.190300)*pow(x,4) + (-2.730086)*pow(x,4)*y + (-2.038562)*pow(x,5)	);
}

// Component d with polynomial of degree 5 
// root mean squared error (RMSE): 0.004203538161137005
// error variance: 1.7669733072135076e-05
float comp_d_godot(float x, float y) {
	x += 1.0;
	y += 1.0;
	return (
	+ (49.644937) + (-24.047666)*y + (7.248710)*pow(y,2) + (-2.895757)*pow(y,3) + (0.779058)*pow(y,4) + (-0.078956)*pow(y,5) + (-151.988444)*x + (50.518049)*x*y + (-6.113902)*x*pow(y,2) + (0.536562)*x*pow(y,3) + (-0.057818)*x*pow(y,4) + (190.288106)*pow(x,2) + (-46.222327)*pow(x,2)*y + (3.736295)*pow(x,2)*pow(y,2) + (-0.082829)*pow(x,2)*pow(y,3) + (-117.560503)*pow(x,3) + (17.843822)*pow(x,3)*y + (-0.827020)*pow(x,3)*pow(y,2) + (36.120635)*pow(x,4) + (-2.331960)*pow(x,4)*y + (-4.487602)*pow(x,5)	);
}

// Component e with polynomial of degree 5 
// root mean squared error (RMSE): 0.006821567579235692
// error variance: 4.6533784238079495e-05
float comp_e_godot(float x, float y) {
	x += 1.0;
	y += 1.0;
	return (
	+ (33.064271) + (-50.177714)*y + (18.981117)*pow(y,2) + (-1.958716)*pow(y,3) + (-0.452548)*pow(y,4) + (0.082617)*pow(y,5) + (-67.358879)*x + (103.132344)*x*y + (-33.368821)*x*pow(y,2) + (4.463898)*x*pow(y,3) + (-0.119857)*x*pow(y,4) + (47.954964)*pow(x,2) + (-73.023067)*pow(x,2)*y + (15.131075)*pow(x,2)*pow(y,2) + (-1.195099)*pow(x,2)*pow(y,3) + (-12.722984)*pow(x,3) + (23.275869)*pow(x,3)*y + (-2.002714)*pow(x,3)*pow(y,2) + (-0.129505)*pow(x,4) + (-2.995953)*pow(x,4)*y + (0.440648)*pow(x,5)	);
}

// Component mag with polynomial of degree 5 
// root mean squared error (RMSE): 0.021627974830001156
// error variance: 0.00046776929524716346
float comp_mag_godot(float x, float y) {
	x += 1.0;
	y += 1.0;
	return (
	+ (38.278041) + (-20.114468)*y + (-25.625500)*pow(y,2) + (22.902099)*pow(y,3) + (-6.340879)*pow(y,4) + (0.601630)*pow(y,5) + (-108.579649)*x + (109.445205)*x*y + (-21.421228)*x*pow(y,2) + (-2.934001)*x*pow(y,3) + (0.867130)*x*pow(y,4) + (89.463175)*pow(x,2) + (-86.998759)*pow(x,2)*y + (17.872421)*pow(x,2)*pow(y,2) + (-0.828128)*pow(x,2)*pow(y,3) + (-27.823761)*pow(x,3) + (26.037136)*pow(x,3)*y + (-2.842303)*pow(x,3)*pow(y,2) + (1.304907)*pow(x,4) + (-2.838796)*pow(x,4)*y + (0.606131)*pow(x,5)	);
}

// Component fres with polynomial of degree 5 
// root mean squared error (RMSE): 0.010689129633331566
// error variance: 0.000114257492318167
float comp_fres_godot(float x, float y) {
	x += 1.0;
	y += 1.0;
	return (
	+ (-1.859444) + (2.955921)*y + (-2.424913)*pow(y,2) + (-0.784027)*pow(y,3) + (-0.098735)*pow(y,4) + (0.207574)*pow(y,5) + (3.419159)*x + (-2.899320)*x*y + (7.985178)*x*pow(y,2) + (0.950729)*x*pow(y,3) + (-0.957429)*x*pow(y,4) + (-3.183815)*pow(x,2) + (-6.123560)*pow(x,2)*y + (-6.308591)*pow(x,2)*pow(y,2) + (1.653213)*pow(x,2)*pow(y,3) + (4.422340)*pow(x,3) + (7.010380)*pow(x,3)*y + (-0.361543)*pow(x,3)*pow(y,2) + (-3.271138)*pow(x,4) + (-0.957836)*pow(x,4)*y + (0.622685)*pow(x,5)	);
}



float integrate_edge_hill(vec3 p0, vec3 p1) {
	float cosTheta = dot(p0, p1);

	float x = cosTheta;
	float y = abs(x);
	float a = 5.42031 + (3.12829 + 0.0902326 * y) * y;
	float b = 3.45068 + (4.18814 + y) * y;
	float theta_sintheta = a / b;

	if (x < 0.0) {
		theta_sintheta = M_PI * inversesqrt(1.0-x * x) - theta_sintheta;
	}
	return theta_sintheta*cross(p0, p1).y;
}

float integrate_edge_acos(vec3 p0, vec3 p1) {

	// to integrate over an edge, we take the two vertices at the ends and calculate the angle between them.
	// then we take the z-coordinate of the cross product, which equals the signed area of the parallelogram formed by p0 and p1 and multiply it by theta/sin(theta)
    float EPSILON = 1e-6f;
	float cosTheta = dot(p0, p1);
	//if(cosTheta + 1.0 < EPSILON) {
	//	return 0.0; // avoid singularities
	//}

	/* float x = cosTheta;
	float y = abs(x);
	float a = 5.42031 + (3.12829 + 0.0902326 * y) * y;
	float b = 3.45068 + (4.18814 + y) * y;
	float theta_sintheta = a / b;

	if (x < 0.0) {
		theta_sintheta = M_PI * inversesqrt(1.0-x * x) - theta_sintheta;
	}
	return theta_sintheta*cross(p0, p1).z; */

    float theta = acos(cosTheta);
	float res = cross(p0, p1).y * ((theta > 0.001) ? theta/sin(theta) : 1.0);
    return res;
}

float integrate_edge(vec3 p_proj0, vec3 p_proj1, vec3 p0, vec3 p1) {
 	float epsilon = 0.00001;
	bool opposite_sides = dot(p_proj0, p_proj1) < -1.0 + epsilon;
	if(opposite_sides) {
		// calculate the point on the line p0 to p1 that is closest to the vertex (origin)
		vec3 half_point_t = p0 + normalize(p1-p0) * dot(p0, normalize(p0-p1));
		vec3 half_point = normalize(half_point_t);
		return integrate_edge_acos(p_proj0, half_point) + integrate_edge_acos(half_point, p_proj1);
	}
	return integrate_edge_acos(p_proj0, p_proj1);
}

void clip_quad_to_horizon(inout vec3 L[5], out int vertex_count)
{
    // detect clipping config
    int config = 0;
    if (L[0].y > 0.0) config += 1;
    if (L[1].y > 0.0) config += 2;
    if (L[2].y > 0.0) config += 4;
    if (L[3].y > 0.0) config += 8;

    // clip
    vertex_count = 0;

    if (config == 0)
    {
        // clip all
    }
    else if (config == 1) // V1 clip V2 V3 V4
    {
        vertex_count = 3;
        L[1] = -L[1].y * L[0] + L[0].y * L[1];
        L[2] = -L[3].y * L[0] + L[0].y * L[3];
    }
    else if (config == 2) // V2 clip V1 V3 V4
    {
        vertex_count = 3;
        L[0] = -L[0].y * L[1] + L[1].y * L[0];
        L[2] = -L[2].y * L[1] + L[1].y * L[2];
    }
    else if (config == 3) // V1 V2 clip V3 V4
    {
        vertex_count = 4;
        L[2] = -L[2].y * L[1] + L[1].y * L[2];
        L[3] = -L[3].y * L[0] + L[0].y * L[3];
    }
    else if (config == 4) // V3 clip V1 V2 V4
    {
        vertex_count = 3;
        L[0] = -L[3].y * L[2] + L[2].y * L[3];
        L[1] = -L[1].y * L[2] + L[2].y * L[1];
    }
    else if (config == 5) // V1 V3 clip V2 V4) impossible
    {
        vertex_count = 0;
    }
    else if (config == 6) // V2 V3 clip V1 V4
    {
        vertex_count = 4;
        L[0] = -L[0].y * L[1] + L[1].y * L[0];
        L[3] = -L[3].y * L[2] + L[2].y * L[3];
    }
    else if (config == 7) // V1 V2 V3 clip V4
    {
        vertex_count = 5;
        L[4] = -L[3].y * L[0] + L[0].y * L[3];
        L[3] = -L[3].y * L[2] + L[2].y * L[3];
    }
    else if (config == 8) // V4 clip V1 V2 V3
    {
        vertex_count = 3;
        L[0] = -L[0].y * L[3] + L[3].y * L[0];
        L[1] = -L[2].y * L[3] + L[3].y * L[2];
        L[2] =  L[3];
    }
    else if (config == 9) // V1 V4 clip V2 V3
    {
        vertex_count = 4;
        L[1] = -L[1].y * L[0] + L[0].y * L[1];
        L[2] = -L[2].y * L[3] + L[3].y * L[2];
    }
    else if (config == 10) // V2 V4 clip V1 V3) impossible
    {
        vertex_count = 0;
    }
    else if (config == 11) // V1 V2 V4 clip V3
    {
        vertex_count = 5;
        L[4] = L[3];
        L[3] = -L[2].y * L[3] + L[3].y * L[2];
        L[2] = -L[2].y * L[1] + L[1].y * L[2];
    }
    else if (config == 12) // V3 V4 clip V1 V2
    {
        vertex_count = 4;
        L[1] = -L[1].y * L[2] + L[2].y * L[1];
        L[0] = -L[0].y * L[3] + L[3].y * L[0];
    }
    else if (config == 13) // V1 V3 V4 clip V2
    {
        vertex_count = 5;
        L[4] = L[3];
        L[3] = L[2];
        L[2] = -L[1].y * L[2] + L[2].y * L[1];
        L[1] = -L[1].y * L[0] + L[0].y * L[1];
    }
    else if (config == 14) // V2 V3 V4 clip V1
    {
        vertex_count = 5;
        L[4] = -L[0].y * L[3] + L[3].y * L[0];
        L[0] = -L[0].y * L[1] + L[1].y * L[0];
    }
    else if (config == 15) // V1 V2 V3 V4
    {
        vertex_count = 4;
    }
    
    if (vertex_count == 3)
        L[3] = L[0];
    if (vertex_count == 4)
        L[4] = L[0];
}

vec3 ltc_evaluate(vec3 vertex, vec3 normal, vec3 eye_vec, mat3 M_inv, vec3 points[4]) {
    // construct the orthonormal basis around the normal vector
    vec3 x, z;
	/*if(dot(normal -  eye_vec, normal -  eye_vec) < 0.001) {
		if(dot(eye_vec, vec3(0,1,0)) < 0.99) {
			eye_vec = normalize(eye_vec + 0.01 * vec3(0, 1, 0));
		} else {
			eye_vec = normalize(eye_vec + 0.01 * vec3(1, 0, 0));
		}
	}*/
    z = -normalize(eye_vec - normal*dot(eye_vec, normal)); // expanding the angle between view and normal vector to 90 degrees, this gives a normal vector, unless view=normal. TODO: in that case, we have a problem.
    x = cross(normal, z);

    // rotate area light in (T1, normal, T2) basis
    M_inv = M_inv * transpose(mat3(x, normal, z));

	vec3 L[5];
	L[0] = M_inv * points[0];
	L[1] = M_inv * points[1];
	L[2] = M_inv * points[2];
	L[3] = M_inv * points[3];

	// before normalization, clamp the points such that they are not further away, than 100 times the shorter direction
	vec3 Lx = L[1]-L[0];
	vec3 Ly = L[2]-L[1];
	float len_x = length(Lx);
	float len_y = length(Ly);
	float ratio = len_x / len_y;
	float thresh = 100.0;
	if (ratio > thresh) { // x >> y
		float dx = (len_x - len_y * thresh) / len_x;
		
		L[0] *= dx;
		L[1] *= dx;
		L[2] *= dx;
		L[3] *=	dx;
	} else if (ratio < 1.0/thresh) { // b >> a
		float dy = (len_y - len_x * thresh) / len_y;
		
		L[0] *= dy;
		L[1] *= dy;
		L[2] *= dy;
		L[3] *= dy;
	}

    int n = 0;
    clip_quad_to_horizon(L, n);
    if (n == 0)
        return vec3(0, 0, 0);
	
	vec3 L_proj[5];
	// project onto unit sphere 
	L_proj[0] = normalize(L[0]);
	L_proj[1] = normalize(L[1]);
	L_proj[2] = normalize(L[2]);
	L_proj[3] = normalize(L[3]);
	L_proj[4] = normalize(L[4]);

	// Prevent abnormal values when the light goes through (or close to) the fragment
	// get the normal of this spherical polygon:
	vec3 pnorm = normalize(cross(L_proj[0] - L_proj[1], L_proj[2] - L_proj[1]));
	if(abs(dot(pnorm, L_proj[0])) < 1e-10) {
		// we could just return black, but that would lead to some black pixels in front of the light.
		// Better, we check if the fragment is on the light, and return white if so. 
		vec3 r10 = points[0] - points[1];
		vec3 r12 = points[2] - points[1];
		float alpha = -dot(points[1], r10)/dot(r10, r10);
		float beta = -dot(points[1], r12)/dot(r12, r12);
		if(0.0 < alpha && alpha < 1.0 && 0.0 < beta && beta < 1.0) { // fragment is on light {
			return vec3(2*M_PI);
		} else {
			return vec3(0.0);
		}
	}

	float I; // default case of 4 edges, need to adjust for case where light cuts view plane.
	I = integrate_edge(L_proj[0], L_proj[1], L[0], L[1]);
	I += integrate_edge(L_proj[1], L_proj[2], L[1], L[2]);
	I += integrate_edge(L_proj[2], L_proj[3], L[2], L[3]);
    if (n >= 4)
        I += integrate_edge(L_proj[3], L_proj[4], L[3], L[4]);
    if (n == 5)
        I += integrate_edge(L_proj[4], L_proj[0], L[4], L[0]);

	if(I <= 0) {
		//return vec3(1.0, 0.0, 1.0); // PINK
	}
	return vec3(max(0, I));
}

// implementation of area lights with Linearly Transformed Cosines (LTC): https://eheitzresearch.wordpress.com/415-2/
void light_process_area_ltc(uint idx, vec3 vertex, vec3 eye_vec, vec3 normal, vec3 vertex_ddx, vec3 vertex_ddy, vec3 f0, uint orms, float shadow, vec3 albedo, inout float alpha,
#ifdef LIGHT_BACKLIGHT_USED
		vec3 backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
		vec4 transmittance_color,
		float transmittance_depth,
		float transmittance_boost,
#endif
#ifdef LIGHT_RIM_USED
		float rim, float rim_tint,
#endif
#ifdef LIGHT_CLEARCOAT_USED
		float clearcoat, float clearcoat_roughness, vec3 vertex_normal,
#endif
#ifdef LIGHT_ANISOTROPY_USED
		vec3 binormal, vec3 tangent, float anisotropy,
#endif
		inout vec3 diffuse_light,
		inout vec3 specular_light) {
	float EPSILON = 1e-4f;
	vec3 area_side_a = area_lights.data[idx].area_side_a;
	vec3 area_side_b = area_lights.data[idx].area_side_b;

	if (dot(area_side_a, area_side_a) < EPSILON || dot(area_side_b, area_side_b) < EPSILON) { // area is 0
		return;
	}
	if (dot(-cross(area_side_a, area_side_b), vertex - area_lights.data[idx].position) <= 0) {
		return; // vertex is behind light
	}

	vec4 orms_unpacked = unpackUnorm4x8(orms);
	float roughness = orms_unpacked.y;
	float metallic = orms_unpacked.z;

	float theta = acos(dot(normal, eye_vec));

	vec2 lut_uv = vec2(roughness, theta/(0.5*M_PI));
	vec2 lut_uv1 = lut_uv*(63.0/64.0) + vec2(0.5/64.0); // offset by 1 pixel
	vec2 lut_uv2 = vec2(roughness, theta/(0.5*M_PI));
	lut_uv2 = lut_uv2*(31.0/32.0) + vec2(0.5/32.0); // offset by 1 pixel
	vec4 M_brdf_abcd = texture(ltc_lut1, lut_uv1);
	vec3 M_brdf_e_mag_fres = texture(ltc_lut2, lut_uv1).xyz;

	// M_brdf_abcd.x = comp_a(roughness, theta);
	// M_brdf_abcd.y = comp_b(roughness, theta);
	// M_brdf_abcd.z = comp_c(roughness, theta);
	// M_brdf_abcd.w = comp_d(roughness, theta);
	// M_brdf_e_mag_fres.x = comp_e(roughness, theta);
	// M_brdf_e_mag_fres.y = comp_mag(roughness, theta);
	// M_brdf_e_mag_fres.z = comp_fres(roughness, theta);

	// M_brdf_abcd.x = comp_a_godot(roughness, theta);
	// M_brdf_abcd.y = comp_b_godot(roughness, theta);
	// M_brdf_abcd.z = comp_c_godot(roughness, theta);
	// M_brdf_abcd.w = comp_d_godot(roughness, theta);
	// M_brdf_e_mag_fres.x = comp_e_godot(roughness, theta);
	// M_brdf_e_mag_fres.y = comp_mag_godot(roughness, theta);
	// M_brdf_e_mag_fres.z = comp_fres_godot(roughness, theta);

	float scale = 1.0 / (M_brdf_abcd.x * M_brdf_e_mag_fres.x - M_brdf_abcd.y * M_brdf_abcd.w);

	mat3 M_inv = mat3(
		vec3(0, 0, 1.0/M_brdf_abcd.z),
		vec3(-M_brdf_abcd.w * scale, M_brdf_abcd.x * scale, 0),
		vec3(-M_brdf_e_mag_fres.x * scale, M_brdf_abcd.y * scale, 0)
	);

	vec3 points[4];
	points[0] = area_lights.data[idx].position - vertex;
	points[1] = area_lights.data[idx].position + area_side_a - vertex;
	points[2] = area_lights.data[idx].position + area_side_a + area_side_b - vertex;
	points[3] = area_lights.data[idx].position + area_side_b - vertex;

	vec3 ltc_diffuse = max(ltc_evaluate(vertex, normal, eye_vec, mat3(1), points), vec3(0));
	vec3 ltc_specular = max(ltc_evaluate(vertex, normal, eye_vec, M_inv, points), vec3(0));
	
	float a_half_len = length(area_side_a) / 2.0;
	float b_half_len = length(area_side_b) / 2.0;
	mat4 light_mat = mat4(
		vec4(normalize(area_side_a), 0),
		vec4(normalize(area_side_b), 0),
		vec4(normalize(cross(area_side_a, area_side_b)), 0),
		vec4(area_lights.data[idx].position + (area_side_a + area_side_b)/2.0, 1)
	);
	mat4 light_mat_inv = inverse(light_mat);
	vec3 pos_local_to_light = (light_mat_inv * vec4(vertex, 1)).xyz;
	vec3 closest_point_local_to_light = vec3(clamp(pos_local_to_light.x, -a_half_len, a_half_len), clamp(pos_local_to_light.y, -b_half_len, b_half_len), 0);
	float dist = length(closest_point_local_to_light - pos_local_to_light);

	float light_length = max(0, dist);
	float light_attenuation = get_omni_attenuation(light_length, area_lights.data[idx].inv_radius, area_lights.data[idx].attenuation);
	light_attenuation = clamp(light_attenuation * shadow, 0, 1);

	if (metallic < 1.0) {
		diffuse_light += ltc_diffuse * area_lights.data[idx].color / (2*M_PI) * light_attenuation;
	}
	vec3 spec = ltc_specular * area_lights.data[idx].color;
	vec3 spec_color = F0(metallic, area_lights.data[idx].specular_amount, albedo);
	
	spec *= spec_color * max(M_brdf_e_mag_fres.y, 0.0) + (1.0 - spec_color) * max(M_brdf_e_mag_fres.z, 0.0); // TODO
	specular_light += spec / (2*M_PI) * area_lights.data[idx].specular_amount * light_attenuation;
	//alpha = ?; // ... SHADOW_TO_OPACITY might affect this.
}

// Functions for form factors
float polygon_solid_angle(vec3 vertex, vec3 L[5], int vertex_count)
{
	// The solid angle of a spherical rectangle is the difference of the sum of its angles
	// and the sum of the angles of a plane rectangle (2*PI)
	vec3 v0 = L[0];
	vec3 v1 = L[1];
	vec3 v2 = L[2];
	vec3 v3 = v0;
	vec3 v4 = v0; 
	if(vertex_count >= 4) {
		v3 = L[3];
	}
	if(vertex_count == 5) {
		v4 = L[4];
	}

	vec3 n0 = normalize(cross(v0, v1));
	vec3 n1 = normalize(cross(v1, v2));
	vec3 n2 = normalize(cross(v2, v3));
	vec3 n3 = n0;
	vec3 n4 = n0;
	if(vertex_count >= 4) {
		n3 = normalize(cross(v3, v4));
	}
	if(vertex_count == 5) {
		n4 = normalize(cross(v4, v0));
	}

	float g0 = acos(dot(-n0, n1));
	float g1 = acos(dot(-n1, n2));
	float g2 = acos(dot(-n2, n3));
	float g3 = 0;
	if(vertex_count >= 4) {
		g3 = acos(dot(-n3, n4));
	}
	float g4 = 0; 
	if(vertex_count == 5) {
		g4 = acos(dot(-n4, n0));
	}

	float angle_sum = g0 + g1 + g2 + g3 + g4;
	float plane_polygon_angle_sum = (vertex_count-2) * M_PI; // triangle: pi, quad: 2pi, pentagon: 3pi
	
	return angle_sum - plane_polygon_angle_sum;
}

void light_process_area_nearest_point(uint idx, vec3 vertex, vec3 eye_vec, vec3 normal, vec3 vertex_ddx, vec3 vertex_ddy, vec3 f0, uint orms, float shadow, vec3 albedo, inout float alpha,
#ifdef LIGHT_BACKLIGHT_USED
		vec3 backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
		vec4 transmittance_color,
		float transmittance_depth,
		float transmittance_boost,
#endif
#ifdef LIGHT_RIM_USED
		float rim, float rim_tint,
#endif
#ifdef LIGHT_CLEARCOAT_USED
		float clearcoat, float clearcoat_roughness, vec3 vertex_normal,
#endif
#ifdef LIGHT_ANISOTROPY_USED
		vec3 binormal, vec3 tangent, float anisotropy,
#endif
		inout vec3 diffuse_light, inout vec3 specular_light) {
	float EPSILON = 1e-4f;
	vec3 area_side_a = area_lights.data[idx].area_side_a;
	vec3 area_side_b = area_lights.data[idx].area_side_b;

	if (dot(area_side_a, area_side_a) < EPSILON || dot(area_side_b, area_side_b) < EPSILON) { // area is 0
		return;
	}
	if (dot(cross(area_side_b, area_side_a), vertex - area_lights.data[idx].position) <= 0) {
		return; // vertex is behind light
	}
	
	vec3 area_norm = normalize(cross(area_side_b, area_side_a));
	// calculate area of light above horizon of current pixel:
	
	vec3 points[4];
	points[0] = area_lights.data[idx].position - vertex;
	points[1] = area_lights.data[idx].position + area_side_a - vertex;
	points[2] = area_lights.data[idx].position + area_side_a + area_side_b - vertex;
	points[3] = area_lights.data[idx].position + area_side_b - vertex;
    vec3 x, z;
    z = -normalize(eye_vec - normal*dot(eye_vec, normal)); // expanding the angle between view and normal vector to 90 degrees, this gives a normal vector, unless view=normal. TODO: in that case, we have a problem.
    x = cross(normal, z);
    // rotate area light in (T1, normal, T2) basis
    mat3 M_vert = transpose(mat3(x, normal, z));

	vec3 L[5];
	L[0] = M_vert * points[0];
	L[1] = M_vert * points[1];
	L[2] = M_vert * points[2];
	L[3] = M_vert * points[3];
    int vertex_count = 0;
	clip_quad_to_horizon(L, vertex_count);
	float solid_angle = polygon_solid_angle(vertex, L, vertex_count);

	// for diffuse light, we take the nearest point on the light to the intersection of the light-plane 
	// with the half-vector between the nearest point above horizon on the light and the point with the least angle to the surface normal

	// First intersect the line defined by light normal and vertex with the light
	float d_light_norm = dot(area_lights.data[idx].position - vertex, -area_norm);
	vec3 vert_to_intersection = (d_light_norm * -area_norm);
	vec3 normal_light_intersection = vertex + vert_to_intersection; // intersection of light normal on vertex with light_plane 
	
	float p_proj = dot(vert_to_intersection, normal);
	if (p_proj < 0) { // we need to find a point above the horizon (only important when light goes below horizon)
		vec3 horizon_dir = vert_to_intersection + normal * (-p_proj);
		float d = d_light_norm / dot(horizon_dir, -area_norm);
		normal_light_intersection = vertex + d * horizon_dir; // intersection of vertex to horizon with plane
	}
	vec3 light_to_intersection = normal_light_intersection - area_lights.data[idx].position;
	float clamp_a = dot(light_to_intersection, area_side_a) / dot(area_side_a, area_side_a); // projection onto direction a
	float clamp_b = dot(light_to_intersection, area_side_b) / dot(area_side_b, area_side_b); // projection onto direction b
	vec3 closest_point_diff = area_lights.data[idx].position + clamp(clamp_a,0,1) * area_side_a + clamp(clamp_b,0,1) * area_side_b; //
	
	// TODO: steepest point still incorrect when rotating the area light.
	// we get the point with the lowest angle to the vertex normal.
	vec3 steepest_point_diff = area_lights.data[idx].position;
	if(dot(normal, area_norm) >= 0) { // light is pointing away from the vertex normal
		float sa = max(sign(dot(normal, area_side_a)), 0);
		float sb = max(sign(dot(normal, area_side_b)), 0);
		vec3 apex = area_lights.data[idx].position + sa * area_side_a + sb * area_side_b;
		vec3 Ap = vertex + normal * dot(apex-vertex, normal);
		vec3 ApA = area_norm * dot(area_norm, Ap - apex);
		vec3 norm_apex = Ap + normal * dot(normal, ApA); // the point from which we can intersect with the light normal
		// intersect
		float d = dot(area_lights.data[idx].position - norm_apex, area_norm) / dot(ApA, -area_norm);
		vec3 steepest_angle_intersection = norm_apex + d * ApA;
		
		light_to_intersection = steepest_angle_intersection - area_lights.data[idx].position;
		clamp_a = dot(light_to_intersection, area_side_a) / dot(area_side_a, area_side_a); // projection onto direction a
		clamp_b = dot(light_to_intersection, area_side_b) / dot(area_side_b, area_side_b); // projection onto direction b
		steepest_point_diff = area_lights.data[idx].position + clamp(clamp_a,0,1) * area_side_a + clamp(clamp_b,0,1) * area_side_b; //

	} else {
		float d = d_light_norm / dot(normal, -area_norm);
		vec3 steepest_angle_intersection = vertex + d * normal; // intersection of light normal on vertex with light_plane 
		light_to_intersection = steepest_angle_intersection - area_lights.data[idx].position;
		clamp_a = dot(light_to_intersection, area_side_a) / dot(area_side_a, area_side_a); // projection onto direction a
		clamp_b = dot(light_to_intersection, area_side_b) / dot(area_side_b, area_side_b); // projection onto direction b
		steepest_point_diff = area_lights.data[idx].position + clamp(clamp_a,0,1) * area_side_a + clamp(clamp_b,0,1) * area_side_b; //
	}

	vec3 halfway_vec = (closest_point_diff - vertex + steepest_point_diff - vertex) / length(closest_point_diff - vertex + steepest_point_diff - vertex);
	float d = d_light_norm / dot(halfway_vec, -area_norm);
	vec3 most_representative_point_diff = vertex + d * halfway_vec;
	most_representative_point_diff = steepest_point_diff;

	vec3 light_rel_vec_diff = most_representative_point_diff - vertex;
	vec3 light_rel_vec_spec;
	// for specular light, we take the point on the light, that has the smallest angle to the reflected view vector
	vec3 reflection_vec = -eye_vec + 2 * dot(eye_vec, normal) * normal;
	float h = dot(reflection_vec, -area_norm);
	float spec_size = 1.0;
	if(h < EPSILON) { // lines are parallel.
		light_rel_vec_spec = light_rel_vec_diff; 
	} else {
		d = d_light_norm/h;
		vec3 intersection_vec = d * reflection_vec;
		vec3 ref_light_intersection = vertex + intersection_vec; // intersection of reflection_vec with light_plane 
		light_to_intersection = ref_light_intersection - area_lights.data[idx].position;
		float len_a = length(area_side_a);
		float len_b = length(area_side_b);
		float isec_a = dot(light_to_intersection, area_side_a) / len_a; // projection onto direction a
		float isec_b = dot(light_to_intersection, area_side_b) / len_b; // projection onto direction b

		// Code to just take the clamped intersection of the reflection vector with the light (instead of the midpoint of the reflection-cone light intersection, as below)
		//clamp_a = dot(light_to_intersection, area_side_a) / dot(area_side_a, area_side_a); // projection onto direction a
		//clamp_b = dot(light_to_intersection, area_side_b) / dot(area_side_b, area_side_b); // projection onto direction b
		//vec3 closest_point_spec = area_lights.data[idx].position + clamp(clamp_a,0,1) * area_side_a + clamp(clamp_b,0,1) * area_side_b; //
		//light_rel_vec_spec = closest_point_spec - vertex;

		vec4 orms_unpacked = unpackUnorm4x8(orms);
		float roughness = orms_unpacked.y;
		//float half_apex_angle = sqrt(2/(1-roughness+2)); // Suggested apex angle formula (Drobot GPU Pro 5, 2014), produces way too much diffusion.
		float half_apex_angle = roughness*M_PI/8.0; // linear apex angle formula, empirically derived
		float r = tan(half_apex_angle) * length(intersection_vec);
		float a = 1.77245385 * r; //sqrt(pi)
		vec2 d1 = vec2(a,a);
		vec2 c1 = vec2(isec_a, isec_b);

		vec2 d0 = vec2(len_a, len_b); // diagonal
		vec2 bl0 = vec2(0);
		vec2 bl1 = vec2(c1 - d1/2);
		vec2 tr0 = vec2(bl0 + d0);
		vec2 tr1 = vec2(c1 + d1/2);
		vec2 bl = max(bl0, bl1);
		vec2 tr = min(tr0, tr1);
		vec2 cr0r1 = (bl + tr) / 2; // mid point of corners of intersection quad
		if (bl0.x < tr1.x && bl0.y < tr1.y && tr0.x > bl1.x && tr0.y > bl1.y) { // if there is an intersection, i.e. we have some specular reflection
			spec_size = max(tr.x - bl.x, 0) * max(tr.y - bl.y, 0);
		}

		vec3 closest_point_spec = area_lights.data[idx].position + area_side_a * clamp(cr0r1.x/len_a, 0, 1) + area_side_b * clamp(cr0r1.y/len_b, 0, 1);
		light_rel_vec_spec = closest_point_spec - vertex;
	}

	float light_length_diff = length(light_rel_vec_diff);
	float light_length = light_length_diff;

	float omni_attenuation = get_omni_attenuation(light_length, area_lights.data[idx].inv_radius, area_lights.data[idx].attenuation);
	float cos_falloff = max(dot(area_norm, -light_rel_vec_diff/light_length), 0.0); // cosine term for falloff at 90 degrees to light normal
	float light_attenuation = omni_attenuation * cos_falloff * solid_angle; 
	vec3 color = area_lights.data[idx].color;

	float size_A = 0.0;

	light_attenuation *= shadow;
	
	float specular_amount = area_lights.data[idx].specular_amount;

	if(spec_size > EPSILON) { // dot(specular_light, specular_light) > EPSILON && 
		//specular_light /= spec_size; // doesn't seem to work that well...
		//specular_amount /= spec_size;
	}

	light_compute(normal, normalize(light_rel_vec_diff), normalize(light_rel_vec_spec), eye_vec, size_A, color, false, light_attenuation, f0, orms, specular_amount, albedo, alpha,
#ifdef LIGHT_BACKLIGHT_USED
			backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
			transmittance_color,
			transmittance_depth,
			transmittance_boost,
			transmittance_z,
#endif
#ifdef LIGHT_RIM_USED
			rim * omni_attenuation, rim_tint,
#endif
#ifdef LIGHT_CLEARCOAT_USED
			clearcoat, clearcoat_roughness, vertex_normal,
#endif
#ifdef LIGHT_ANISOTROPY_USED
			binormal, tangent, anisotropy,
#endif
			diffuse_light,
			specular_light);
}

void light_process_area_montecarlo(uint idx, vec3 vertex, vec3 vertex_world, vec3 eye_vec, vec3 normal, vec3 vertex_ddx, vec3 vertex_ddy, vec3 f0, uint orms, float shadow, vec3 albedo, inout float alpha,
#ifdef LIGHT_BACKLIGHT_USED
		vec3 backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
		vec4 transmittance_color,
		float transmittance_depth,
		float transmittance_boost,
#endif
#ifdef LIGHT_RIM_USED
		float rim, float rim_tint,
#endif
#ifdef LIGHT_CLEARCOAT_USED
		float clearcoat, float clearcoat_roughness, vec3 vertex_normal,
#endif
#ifdef LIGHT_ANISOTROPY_USED
		vec3 binormal, vec3 tangent, float anisotropy,
#endif
		inout vec3 diffuse_light,
		inout vec3 specular_light) {

	float EPSILON = 1e-4f;
	vec3 area_side_a = area_lights.data[idx].area_side_a;
	vec3 area_side_b = area_lights.data[idx].area_side_b;

	if (dot(area_side_a, area_side_a) < EPSILON || dot(area_side_b, area_side_b) < EPSILON) { // area is 0
		return;
	}

	uint sample_nr = max(area_lights.data[idx].area_stochastic_samples, 1);
	vec3 diffuse_sum = vec3(0.0, 0.0, 0.0);
	vec3 specular_sum = vec3(0.0, 0.0, 0.0);
	float alpha_sum = 0.0;
	float length_sum = 0.0;

	vec3 color = area_lights.data[idx].color;
	vec3 sampling_vertex = vertex;
	vec3 vert_to_light = sampling_vertex - area_lights.data[idx].position;
	vec3 area_norm = cross(area_side_b, area_side_a);

	if (dot(area_norm, vert_to_light) <= 0) {
		return; // vertex is behind light
	}

	if (dot(vert_to_light, vert_to_light) < EPSILON) {
		sampling_vertex += vec3(0.01, 0.01, 0.01); // small offset
	}

	vec3 spec_squad_position = area_lights.data[idx].position;
	vec3 spec_squad_a = area_side_a;
	vec3 spec_squad_b = area_side_b;

	vec4 orms_unpacked = unpackUnorm4x8(orms);
	float roughness = orms_unpacked.y;
	if(roughness < 0.5) {
		vec3 reflection_vec = normalize(-eye_vec + 2 * dot(eye_vec, normal) * normal);
		float d = dot(area_lights.data[idx].position - vertex, -area_norm) / dot(reflection_vec, -area_norm);
		vec3 intersection_vec = d * reflection_vec;
		vec3 ref_light_intersection = vertex + intersection_vec; // intersection of reflection_vec with light_plane 
		vec3 light_to_intersection = ref_light_intersection - area_lights.data[idx].position;
		float len_a = length(area_side_a);
		float len_b = length(area_side_b);
		float isec_a = dot(light_to_intersection, area_side_a) / len_a; // projection onto direction a
		float isec_b = dot(light_to_intersection, area_side_b) / len_b; // projection onto direction b

		//float half_apex_angle = sqrt(2/(1-roughness+2)); // Suggested apex angle formula (Drobot GPU Pro 5, 2014), produces way too much diffusion.
		float half_apex_angle = roughness*M_PI/3.0; // linear apex angle formula, empirically derived
		float r = tan(half_apex_angle) * length(intersection_vec);
		float a = 1.77245385 * r; //sqrt(pi)
		vec2 d1 = vec2(a,a);
		vec2 c1 = vec2(isec_a, isec_b);

		vec2 d0 = vec2(len_a, len_b); // diagonal
		vec2 bl0 = vec2(0);
		vec2 bl1 = vec2(c1 - d1/2);
		vec2 tr0 = vec2(bl0 + d0);
		vec2 tr1 = vec2(c1 + d1/2);
		vec2 bl = max(bl0, bl1);
		vec2 tr = min(tr0, tr1);
		vec2 cr0r1 = (bl + tr) / 2; // mid point of corners of intersection quad

		if (bl0.x < tr1.x && bl0.y < tr1.y && tr0.x > bl1.x && tr0.y > bl1.y) { // if there is an intersection, i.e. we have some specular reflection
			spec_squad_position = area_lights.data[idx].position + area_side_a * clamp(bl.x/len_a, 0, 1) + area_side_b * clamp(bl.y/len_b, 0, 1);
			spec_squad_a = area_side_a/len_a * (tr.x-bl.x);//clamp(, 0, 1);
			spec_squad_b = area_side_b/len_b * (tr.y-bl.y);//clamp(, 0, 1);
		}
	}

	SphericalQuad squad = init_spherical_quad(area_lights.data[idx].position, area_side_a, area_side_b, sampling_vertex);
	SphericalQuad spec_squad = init_spherical_quad(spec_squad_position, spec_squad_a, spec_squad_b, sampling_vertex);

	if (squad.S == 0) { // area is 0
		return;
	}
	float inv_S = 1 / squad.S;
	float spec_inv_S = 1 / spec_squad.S;

	vec3 p00 = area_lights.data[idx].position;
	vec3 p10 = area_lights.data[idx].position + area_side_a;
	vec3 p01 = area_lights.data[idx].position + area_side_b;
	vec3 p11 = area_lights.data[idx].position + area_side_a + area_side_b;

	vec3 v00 = normalize(p00 - sampling_vertex);
	vec3 v10 = normalize(p10 - sampling_vertex);
	vec3 v01 = normalize(p01 - sampling_vertex);
	vec3 v11 = normalize(p11 - sampling_vertex);

	// Compute  weights for rectangle seen from reference point
	vec4 w = vec4(max(0.01, abs(dot(v00, normal))),
			max(0.01, abs(dot(v10, normal))), // TODO: double check order of weights here
			max(0.01, abs(dot(v01, normal))),
			max(0.01, abs(dot(v11, normal))));

	for (uint i = 0; i < sample_nr; i++) {
		// sampling of diffuse is based on a world position, so flickering is reduced
		float pdf = inv_S;
		float u = randomize(random_seed(vertex_world * 1e3) + i);
		float v = randomize(hash(random_seed(vertex_world * 1e3) + i));

		vec2 uv = sample_bilinear(u, v, w);
		u = uv[0];
		v = uv[1];
		pdf *= bilinear_PDF(u, v, w);

		// sampling for specular depends on the view
		vec3 screen_uvw = gl_FragCoord.xyz * vec3(scene_data_block.data.screen_pixel_size, 1.0);
		float s_u = randomize(random_seed(screen_uvw) + i);
		float s_v = randomize(hash(random_seed(screen_uvw) + i));

		vec2 s_uv = sample_bilinear(s_u, s_v, w);
		s_u = s_uv[0];
		s_v = s_uv[1];
		float s_pdf = spec_inv_S * bilinear_PDF(s_u, s_v, w);
		
		vec3 sampled_position = sample_squad(squad, u, v);
		vec3 sampled_position_spec = roughness < 0.5 ? sample_squad(spec_squad, u, v) : sampled_position; // TODO sample from directions within spec_cone_angle
		vec3 light_rel_vec_diff = sampled_position - sampling_vertex;
		vec3 light_rel_vec_spec = sampled_position_spec - sampling_vertex;

		// TODO: how to calculate attenuation value? probably easiest to base it on diffuse
		float light_length = length(light_rel_vec_diff);
		length_sum += light_length;

		float light_attenuation = get_omni_attenuation(light_length, area_lights.data[idx].inv_radius, area_lights.data[idx].attenuation);
		float specular_amount = area_lights.data[idx].specular_amount;

		float size_A = 0.0; // not sure about this one

// TODO: Fix this block
#ifdef LIGHT_TRANSMITTANCE_USED
		// TODO: neither SpotLight nor OmniLight implementation gives useful results here.
#endif //LIGHT_TRANSMITTANCE_USED

		light_attenuation *= shadow;
		vec3 diffuse_contribution = vec3(0.0, 0.0, 0.0);
		vec3 specular_contribution = vec3(0.0, 0.0, 0.0);

		vec3 light_vec_diff = normalize(light_rel_vec_diff);
		vec3 light_vec_spec = normalize(light_rel_vec_spec);
		light_compute(normal, light_vec_diff, light_vec_spec, eye_vec, size_A, color, false, light_attenuation, f0, orms, area_lights.data[idx].specular_amount, albedo, alpha,
#ifdef LIGHT_BACKLIGHT_USED
				backlight,
#endif
#ifdef LIGHT_TRANSMITTANCE_USED
				transmittance_color,
				transmittance_depth,
				transmittance_boost,
				transmittance_z,
#endif
#ifdef LIGHT_RIM_USED
				rim * light_attenuation, rim_tint,
#endif
#ifdef LIGHT_CLEARCOAT_USED
				clearcoat, clearcoat_roughness, vertex_normal,
#endif
#ifdef LIGHT_ANISOTROPY_USED
				binormal, tangent, anisotropy,
#endif
				diffuse_contribution, specular_contribution);

		if (pdf > 0) {
			diffuse_sum += diffuse_contribution / pdf;
			specular_sum += specular_contribution / s_pdf;
			alpha_sum += alpha;
		} else {
			sample_nr = max(sample_nr - 1, 1);
		}
	}

	float inv_sample_nr = 1.0 / sample_nr;
	diffuse_light += inv_sample_nr * diffuse_sum;
	specular_light += inv_sample_nr * specular_sum;
	alpha = inv_sample_nr * alpha_sum;
}

void reflection_process(uint ref_index, vec3 vertex, vec3 ref_vec, vec3 normal, float roughness, vec3 ambient_light, vec3 specular_light, inout vec4 ambient_accum, inout vec4 reflection_accum) {
	vec3 box_extents = reflections.data[ref_index].box_extents;
	vec3 local_pos = (reflections.data[ref_index].local_matrix * vec4(vertex, 1.0)).xyz;

	if (any(greaterThan(abs(local_pos), box_extents))) { //out of the reflection box
		return;
	}

	vec3 inner_pos = abs(local_pos / box_extents);
	float blend = max(inner_pos.x, max(inner_pos.y, inner_pos.z));
	//make blend more rounded
	blend = mix(length(inner_pos), blend, blend);
	blend *= blend;
	blend = max(0.0, 1.0 - blend);

	if (reflections.data[ref_index].intensity > 0.0) { // compute reflection

		vec3 local_ref_vec = (reflections.data[ref_index].local_matrix * vec4(ref_vec, 0.0)).xyz;

		if (reflections.data[ref_index].box_project) { //box project

			vec3 nrdir = normalize(local_ref_vec);
			vec3 rbmax = (box_extents - local_pos) / nrdir;
			vec3 rbmin = (-box_extents - local_pos) / nrdir;

			vec3 rbminmax = mix(rbmin, rbmax, greaterThan(nrdir, vec3(0.0, 0.0, 0.0)));

			float fa = min(min(rbminmax.x, rbminmax.y), rbminmax.z);
			vec3 posonbox = local_pos + nrdir * fa;
			local_ref_vec = posonbox - reflections.data[ref_index].box_offset;
		}

		vec4 reflection;

		reflection.rgb = textureLod(samplerCubeArray(reflection_atlas, DEFAULT_SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP), vec4(local_ref_vec, reflections.data[ref_index].index), roughness * MAX_ROUGHNESS_LOD).rgb * sc_luminance_multiplier;
		reflection.rgb *= reflections.data[ref_index].exposure_normalization;
		if (reflections.data[ref_index].exterior) {
			reflection.rgb = mix(specular_light, reflection.rgb, blend);
		}

		reflection.rgb *= reflections.data[ref_index].intensity; //intensity
		reflection.a = blend;
		reflection.rgb *= reflection.a;

		reflection_accum += reflection;
	}

	switch (reflections.data[ref_index].ambient_mode) {
		case REFLECTION_AMBIENT_DISABLED: {
			//do nothing
		} break;
		case REFLECTION_AMBIENT_ENVIRONMENT: {
			//do nothing
			vec3 local_amb_vec = (reflections.data[ref_index].local_matrix * vec4(normal, 0.0)).xyz;

			vec4 ambient_out;

			ambient_out.rgb = textureLod(samplerCubeArray(reflection_atlas, DEFAULT_SAMPLER_LINEAR_WITH_MIPMAPS_CLAMP), vec4(local_amb_vec, reflections.data[ref_index].index), MAX_ROUGHNESS_LOD).rgb;
			ambient_out.rgb *= reflections.data[ref_index].exposure_normalization;
			ambient_out.a = blend;
			if (reflections.data[ref_index].exterior) {
				ambient_out.rgb = mix(ambient_light, ambient_out.rgb, blend);
			}

			ambient_out.rgb *= ambient_out.a;
			ambient_accum += ambient_out;
		} break;
		case REFLECTION_AMBIENT_COLOR: {
			vec4 ambient_out;
			ambient_out.a = blend;
			ambient_out.rgb = reflections.data[ref_index].ambient;
			if (reflections.data[ref_index].exterior) {
				ambient_out.rgb = mix(ambient_light, ambient_out.rgb, blend);
			}
			ambient_out.rgb *= ambient_out.a;
			ambient_accum += ambient_out;
		} break;
	}
}

float blur_shadow(float shadow) {
	return shadow;
#if 0
	// TODO: what is this???
	//disabling for now, will investigate later
	float interp_shadow = shadow;
	if (gl_HelperInvocation) {
		interp_shadow = -4.0; // technically anything below -4 will do but just to make sure
	}

	uvec2 fc2 = uvec2(gl_FragCoord.xy);
	interp_shadow -= dFdx(interp_shadow) * (float(fc2.x & 1) - 0.5);
	interp_shadow -= dFdy(interp_shadow) * (float(fc2.y & 1) - 0.5);

	if (interp_shadow >= 0.0) {
		shadow = interp_shadow;
	}
	return shadow;
#endif
}
