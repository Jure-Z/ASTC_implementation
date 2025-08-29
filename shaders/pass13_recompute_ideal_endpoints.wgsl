const WORKGROUP_SIZE: u32 = 64u;
const BLOCK_MAX_TEXELS: u32 = 144u;
const BLOCK_MAX_WEIGHTS: u32 = 64u;
const BLOCK_MAX_PARTITIONS: u32 = 4u;

const TUNE_MAX_TRIAL_CANDIDATES = 8u;

struct UniformVariables {
    xdim : u32,
    ydim : u32,

    texel_count : u32,

    decimation_mode_count : u32,
    block_mode_count : u32,

    valid_decimation_mode_count: u32,
	valid_block_mode_count: u32,

    quant_limit : u32,
    partition_count : u32,
    tune_candidate_limit : u32,

    _padding1: u32,
    _padding2: u32,

    channel_weights : vec4<f32>,

    partitioning_count_selected : vec4<u32>,
    partitioning_count_all : vec4<u32>,
};

struct DecimationInfo {
    texel_count : u32,
    weight_count : u32,
    weight_x : u32,
    weight_y : u32,

    max_quant_level : u32,
    max_angular_steps : u32,
    max_quant_steps: u32,
    _padding: u32,

    texel_weight_count : array<u32, BLOCK_MAX_TEXELS>,
    texel_weights_offset : array<u32, BLOCK_MAX_TEXELS>,

    weight_texel_count : array<u32, BLOCK_MAX_WEIGHTS>,
    weight_texels_offset : array<u32, BLOCK_MAX_WEIGHTS>,
};

struct TexelToWeightMap {
	weight_index : u32,
	contribution : f32,

    _padding1 : u32,
    _padding2 : u32,
};

struct InputBlock {
    pixels: array<vec4<f32>, BLOCK_MAX_TEXELS>,
    texel_partitions: array<u32, BLOCK_MAX_TEXELS>,
    partition_pixel_counts: array<u32, 4>,

    partitioning_idx: u32,
    grayscale: u32,
    constant_alpha: u32,
    padding: u32,
};

struct BlockMode {
	mode_index : u32,
    decimation_mode : u32,
    quant_mode : u32,
    weight_bits : u32,
    is_dual_plane : u32,

    _padding1 : u32,
    _padding2 : u32,
    _padding3 : u32,
};

struct IdealEndpointsAndWeightsPartition {
    avg: vec4<f32>,
    dir: vec4<f32>,
    endpoint0: vec4<f32>,
    endpoint1: vec4<f32>,
};

struct FinalCandidate {
    block_mode_index: u32,
    block_mode_trial_index: u32,
    total_error: f32,
    quant_level: u32, // The original quant level
    quant_level_mod: u32,

    _padding1: u32,

	color_formats_matched: u32,
    final_quant_mode: u32, // The quant mode after checking the mod version

    formats: vec4<u32>,
    quantized_weights: array<u32, BLOCK_MAX_WEIGHTS>,
    candidate_partitions: array<IdealEndpointsAndWeightsPartition, 4>,

	final_formats: vec4<u32>,
	//8 integers per partition
    packed_color_values: array<u32, 32>,
};



@group(0) @binding(0) var<uniform> uniforms: UniformVariables;
@group(0) @binding(1) var<storage, read> decimation_infos: array<DecimationInfo>;
@group(0) @binding(2) var<storage, read> texel_to_weight_map: array<TexelToWeightMap>;
@group(0) @binding(3) var<storage, read> block_modes: array<BlockMode>;
@group(0) @binding(4) var<storage, read> input_blocks: array<InputBlock>;

@group(0) @binding(5) var<storage, read_write> final_candidates: array<FinalCandidate>;
@group(0) @binding(6) var<storage, read_write> candidate_rgbs_vectors: array<vec4<f32>>;



// Undecimated weights of block
var<workgroup> dec_weights: array<f32, BLOCK_MAX_WEIGHTS>;
var<workgroup> undec_weights: array<f32, BLOCK_MAX_TEXELS>;

//precomputed values
var<workgroup> averages: array<vec4<f32>, 4>;
var<workgroup> scale_dirs: array<vec3<f32>, 4>;

// Accumulators for the main reduction, one for each partition
var<workgroup> wmin1: array<atomic<u32>, 4>;
var<workgroup> wmax1: array<atomic<u32>, 4>;
var<workgroup> left_sum_s: array<atomic<u32>, 4>;
var<workgroup> middle_sum_s: array<atomic<u32>, 4>;
var<workgroup> right_sum_s: array<atomic<u32>, 4>;
var<workgroup> scale_min: array<atomic<u32>, 4>;
var<workgroup> scale_max: array<atomic<u32>, 4>;
var<workgroup> color_vec_x: array<atomic<u32>, 16>; // 4 partitions * 4 components
var<workgroup> color_vec_y: array<atomic<u32>, 16>;
var<workgroup> scale_vec: array<atomic<u32>, 8>; // 4 partitions * 2 components


fn atomicMin_f32(atomic_target: ptr<workgroup, atomic<u32>>, value: f32) {
    let val_uint = bitcast<u32>(value);
    var current_min = atomicLoad(atomic_target);
    while (value < bitcast<f32>(current_min)) {
        let result = atomicCompareExchangeWeak(atomic_target, current_min, val_uint);
        if (result.exchanged) {
            break;
        }
        current_min = result.old_value;
    }
}

fn atomicMax_f32(atomic_target: ptr<workgroup, atomic<u32>>, value: f32) {
    let val_uint = bitcast<u32>(value);
    var current_max = atomicLoad(atomic_target);
    while (value > bitcast<f32>(current_max)) {
        let result = atomicCompareExchangeWeak(atomic_target, current_max, val_uint);
        if (result.exchanged) {
            break;
        }
        current_max = result.old_value;
    }
}

fn atomicAdd_f32(atomic_target: ptr<workgroup, atomic<u32>>, value_to_add: f32) {
    loop {
        let original_val_uint = atomicLoad(atomic_target);
        let original_val_float = bitcast<f32>(original_val_uint);
        let new_val_float = original_val_float + value_to_add;
        let new_val_uint = bitcast<u32>(new_val_float);
        let result = atomicCompareExchangeWeak(atomic_target, original_val_uint, new_val_uint);
        if (result.exchanged) {
            break;
        }
    }
}


@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(workgroup_id) group_id: vec3<u32>, @builtin(local_invocation_index) local_idx: u32) {

    let block_idx = group_id.x;
    let candidate_idx = block_idx * uniforms.tune_candidate_limit + group_id.y;

    let candidate = final_candidates[candidate_idx];
    let bm = block_modes[candidate.block_mode_index];
    let di = decimation_infos[bm.decimation_mode];

    let block_mode_trial_index = candidate.block_mode_trial_index;

    let quantized_weights = candidate.quantized_weights;

    // Unquantize and bilinear infill
    for (var i = local_idx; i < di.weight_count; i += WORKGROUP_SIZE) {
        let weight_int = quantized_weights[i];
        dec_weights[i] = f32(weight_int) / 64.0;
    }
    workgroupBarrier();

    for (var i = local_idx; i < di.texel_count; i += WORKGROUP_SIZE) {
        if (di.texel_count == di.weight_count) {
            // Undecimated Case: Perform a direct copy.
            undec_weights[i] = dec_weights[i];
        } else {
            // Decimated Case: Perform the bilinear infill.
            var infill_val: f32 = 0.0;
            let weight_offset = di.texel_weights_offset[i];
            for (var j = 0u; j < di.texel_weight_count[i]; j = j + 1u) {
                let mapping = texel_to_weight_map[weight_offset + j];
                infill_val += dec_weights[mapping.weight_index] * mapping.contribution;
            }
            undec_weights[i] = infill_val;
        }
    }
    workgroupBarrier();


    let partition_count = uniforms.partition_count;
    let input_block = input_blocks[block_idx];

    // Initialize accumulators
    if(local_idx < partition_count) {
        atomicStore(&wmin1[local_idx], bitcast<u32>(1.0));
        atomicStore(&wmax1[local_idx], bitcast<u32>(0.0));
        atomicStore(&left_sum_s[local_idx], bitcast<u32>(0.0));
        atomicStore(&middle_sum_s[local_idx], bitcast<u32>(0.0));
        atomicStore(&right_sum_s[local_idx], bitcast<u32>(0.0));
        atomicStore(&scale_min[local_idx], bitcast<u32>(1e10));
        atomicStore(&scale_max[local_idx], bitcast<u32>(0.0));
        for (var j = 0u; j < 4u; j = j + 1u) {
			atomicStore(&color_vec_x[local_idx * 4 + j], bitcast<u32>(0.0));
			atomicStore(&color_vec_y[local_idx * 4 + j], bitcast<u32>(0.0));
		}
        for (var j = 0u; j < 2u; j = j + 1u) {
            atomicStore(&scale_vec[local_idx * 2 + j], bitcast<u32>(0.0));
        }

        //precompute averages and scale directions
        averages[local_idx] = candidate.candidate_partitions[local_idx].avg;

        let partition_size = f32(input_block.partition_pixel_counts[local_idx]);
        let rgba_sum = averages[local_idx] * partition_size * uniforms.channel_weights;
        let rgba_weight_sum = max(uniforms.channel_weights * partition_size, vec4<f32>(1e-17));
        scale_dirs[local_idx] = normalize(rgba_sum.xyz / rgba_weight_sum.xyz);
    }

    workgroupBarrier();
    
    
    // Acumulate multiple properties per-partition
    for (var i = local_idx; i < di.texel_count; i += WORKGROUP_SIZE) {

        let p = input_block.texel_partitions[i];
        let rgba = input_block.pixels[i];
        let weight = undec_weights[i];

        atomicMin_f32(&wmin1[p], weight);
        atomicMax_f32(&wmax1[p], weight);

        let scale_dir = scale_dirs[p];

        let scale = dot(rgba.xyz, scale_dir);

        atomicMin_f32(&scale_min[p], scale);
        atomicMax_f32(&scale_max[p], scale);

        let om_weight = 1.0 - weight;

        atomicAdd_f32(&left_sum_s[p], om_weight * om_weight);
        atomicAdd_f32(&middle_sum_s[p], om_weight * weight);
        atomicAdd_f32(&right_sum_s[p], weight * weight);

        let color_y_contrib = rgba * weight;
        atomicAdd_f32(&color_vec_y[p * 4u + 0u], color_y_contrib.r);
        atomicAdd_f32(&color_vec_y[p * 4u + 1u], color_y_contrib.g);
        atomicAdd_f32(&color_vec_y[p * 4u + 2u], color_y_contrib.b);
        atomicAdd_f32(&color_vec_y[p * 4u + 3u], color_y_contrib.a);
        atomicAdd_f32(&color_vec_x[p * 4u + 0u], rgba.r - color_y_contrib.r);
        atomicAdd_f32(&color_vec_x[p * 4u + 1u], rgba.g - color_y_contrib.g);
        atomicAdd_f32(&color_vec_x[p * 4u + 2u], rgba.b - color_y_contrib.b);
        atomicAdd_f32(&color_vec_x[p * 4u + 3u], rgba.a - color_y_contrib.a);

        let ls_weight = uniforms.channel_weights.x + uniforms.channel_weights.y + uniforms.channel_weights.z;

        atomicAdd_f32(&scale_vec[p * 2u + 0u], om_weight * scale * ls_weight);
        atomicAdd_f32(&scale_vec[p * 2u + 1u], weight * scale * ls_weight);
    }

    workgroupBarrier();


    if (local_idx < partition_count) {
        let p = local_idx;
        let color_weight = uniforms.channel_weights;

        // Load all final summed values from shared memory
        let wmin1_val = bitcast<f32>(atomicLoad(&wmin1[p]));
        let wmax1_val = bitcast<f32>(atomicLoad(&wmax1[p]));
        let left_s = bitcast<f32>(atomicLoad(&left_sum_s[p]));
        let middle_s = bitcast<f32>(atomicLoad(&middle_sum_s[p]));
        let right_s = bitcast<f32>(atomicLoad(&right_sum_s[p]));
        let scale_min_val = bitcast<f32>(atomicLoad(&scale_min[p]));
        let scale_max_val = bitcast<f32>(atomicLoad(&scale_max[p]));

        let color_x = vec4<f32>(
			bitcast<f32>(atomicLoad(&color_vec_x[p * 4u + 0u])),
			bitcast<f32>(atomicLoad(&color_vec_x[p * 4u + 1u])),
			bitcast<f32>(atomicLoad(&color_vec_x[p * 4u + 2u])),
			bitcast<f32>(atomicLoad(&color_vec_x[p * 4u + 3u]))
		) * color_weight;

        let color_y = vec4<f32>(
            bitcast<f32>(atomicLoad(&color_vec_y[p * 4u + 0u])),
            bitcast<f32>(atomicLoad(&color_vec_y[p * 4u + 1u])),
            bitcast<f32>(atomicLoad(&color_vec_y[p * 4u + 2u])),
            bitcast<f32>(atomicLoad(&color_vec_y[p * 4u + 3u]))
        ) * color_weight;

        let scale = vec2<f32>(
			bitcast<f32>(atomicLoad(&scale_vec[p * 2u + 0u])),
			bitcast<f32>(atomicLoad(&scale_vec[p * 2u + 1u]))
		);


        let left_sum_v = vec4<f32>(left_s) * color_weight;
        let middle_sum_v = vec4<f32>(middle_s) * color_weight;
        let right_sum_v = vec4<f32>(right_s) * color_weight;

        let ls_weight = uniforms.channel_weights.x + uniforms.channel_weights.y + uniforms.channel_weights.z;
        let lmrs_sum = vec3<f32>(left_s, middle_s, right_s) * ls_weight;
        
        // Initialize the luminance and scale vectors with a reasonable default
        let scalediv = clamp(scale_min_val / max(scale_max_val, 1e-10), 0.0, 1.0);


        let scale_dir = scale_dirs[p];
        let sds = scale_dir * scale_max_val;

        candidate_rgbs_vectors[candidate_idx * BLOCK_MAX_PARTITIONS + p] = vec4<f32>(sds.x, sds.y, sds.z, scalediv);
        let partition_ptr = &final_candidates[candidate_idx].candidate_partitions[p];

        if(wmin1_val >= wmax1_val * 0.999f) {
            // If all weights are equal set endpoints to average

            let partition_size = f32(input_block.partition_pixel_counts[p]);
            let rgba_weight_sum = max(uniforms.channel_weights * partition_size, vec4<f32>(1e-17));
            let avg_color = (color_x + color_y) / rgba_weight_sum;

            //check for NaN
            let notnan_mask = avg_color == avg_color;
            (*partition_ptr).endpoint0 = select(partition_ptr.endpoint0, avg_color, notnan_mask);
			(*partition_ptr).endpoint1 = select(partition_ptr.endpoint1, avg_color, notnan_mask);
            
            candidate_rgbs_vectors[candidate_idx * BLOCK_MAX_PARTITIONS + p] = vec4<f32>(sds.x, sds.y, sds.z, 1.0f);

        }
        else {
            // Complete the analytic calculation of ideal-endpoint-values for the given
			// set of texel weights and pixel colors

            let color_det1 = (left_sum_v * right_sum_v) - (middle_sum_v * middle_sum_v);
            let color_rdet1 = 1.0 / color_det1;

            let ls_det1 = (lmrs_sum.x * lmrs_sum.z) - (lmrs_sum.y * lmrs_sum.y);
            let ls_rdet1 = 1.0 / ls_det1;

            let color_mss1 = (left_sum_v * left_sum_v) + 2 * (middle_sum_v * middle_sum_v) + (right_sum_v * right_sum_v);
            let ls_mss1 = (lmrs_sum.x * lmrs_sum.x) + 2 * (lmrs_sum.y * lmrs_sum.y) + (lmrs_sum.z * lmrs_sum.z);

            let ep0 = (right_sum_v * color_x - middle_sum_v * color_y) * color_rdet1;
            let ep1 = (left_sum_v * color_y - middle_sum_v * color_x) * color_rdet1;

            let det_mask = abs(color_det1) > (color_mss1 * 1e-4);
            let notnan_mask = (ep0 == ep0) & (ep1 == ep1);
            let full_mask = det_mask & notnan_mask;

            (*partition_ptr).endpoint0 = select(partition_ptr.endpoint0, ep0, full_mask);
			(*partition_ptr).endpoint1 = select(partition_ptr.endpoint1, ep1, full_mask);

            let scale_ep0 = (lmrs_sum.z * scale.x - lmrs_sum.y * scale.y) * ls_rdet1;
            let scale_ep1 = (lmrs_sum.x * scale.y - lmrs_sum.y * scale.x) * ls_rdet1;

            if((abs(ls_det1) > (ls_mss1 * 1e-4)) && (scale_ep0 == scale_ep0) && (scale_ep1 == scale_ep1) && (scale_ep0 < scale_ep1)) {
                let scalediv2 = scale_ep0 / scale_ep1;
                let sdsm = scale_dir * scale_ep1;
                candidate_rgbs_vectors[candidate_idx * BLOCK_MAX_PARTITIONS + p] = vec4<f32>(sdsm.x, sdsm.y, sdsm.z, scalediv2);
            }
        }
    }
}