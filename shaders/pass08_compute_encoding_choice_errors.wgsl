const WORKGROUP_SIZE: u32 = 32u;
const BLOCK_MAX_TEXELS: u32 = 144u;

const DEFAULT_ALPHA: f32 = 65536.0;

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

struct IdealEndpointsAndWeightsPartition {
    avg: vec4<f32>,
    dir: vec4<f32>,
    endpoint0: vec4<f32>,
    endpoint1: vec4<f32>,
};

struct IdealEndpointsAndWeights {
    partitions: array<IdealEndpointsAndWeightsPartition, 4>,
    weights: array<f32, BLOCK_MAX_TEXELS>,

    weight_error_scale: array<f32, BLOCK_MAX_TEXELS>,

    is_constant_weight_error_scale : u32,
    min_weight_cuttof : f32,
    _padding1 : u32,
    _padding2 : u32,
};

struct EncodingChoiceErrors {
    rgb_scale_error: f32,
    rgb_luma_error: f32,
    luminance_error: f32,
    alpha_drop_error: f32,

    can_offset_encode: u32,
    can_blue_contract: u32,

    _padding1: u32,
    _padding2: u32,
}

struct ProcessedLine {
	amod: vec4<f32>,
    bs: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: UniformVariables;
@group(0) @binding(1) var<storage, read> inputBlocks: array<InputBlock>;
@group(0) @binding(2) var<storage, read> ideal_endpoints_and_weights: array<IdealEndpointsAndWeights>;

@group(0) @binding(3) var<storage, read_write> encoding_choice_errors: array<EncodingChoiceErrors>;


var<workgroup> shared_sum_xp: array<atomic<u32>, 16>;
var<workgroup> shared_sum_yp: array<atomic<u32>, 16>;
var<workgroup> shared_sum_zp: array<atomic<u32>, 16>;

var<workgroup> averages: array<vec4<f32>, 4>;
var<workgroup> directions: array<vec4<f32>, 4>;

var<workgroup> uncor_rgb_plines: array<ProcessedLine, 4>;
var<workgroup> samec_rgb_plines: array<ProcessedLine, 4>;
var<workgroup> rgb_luma_plines: array<ProcessedLine, 4>;
var<workgroup> luminance_plines: array<ProcessedLine, 4>;

// Accumulators for the 5 error types for each of the 4 partitions (5 * 4 = 20)
var<workgroup> error_accumulators: array<atomic<u32>, 20>;


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
    
    let block_index = group_id.x;
    let partitionCount = uniforms.partition_count;

    //compute averages and directions for partitions
    let input_block = inputBlocks[block_index];
    let ideal_endpoints_and_weights_block = ideal_endpoints_and_weights[block_index];

    //Initialize shared memory
    for(var i = local_idx; i < 16; i += WORKGROUP_SIZE) {
		atomicStore(&shared_sum_xp[i], bitcast<u32>(0.0));
        atomicStore(&shared_sum_yp[i], bitcast<u32>(0.0));
        atomicStore(&shared_sum_zp[i], bitcast<u32>(0.0));
	}

    workgroupBarrier();

    // Calculate averages and directions for all partitions simultaneously
    for(var i = local_idx; i < uniforms.texel_count; i += WORKGROUP_SIZE) {
        let p = input_block.texel_partitions[i];

		if (p < partitionCount) {
			let average = ideal_endpoints_and_weights_block.partitions[p].avg;
            var texel_datum = input_block.pixels[i] - average;
            texel_datum.w = 0.0; // Ignore alpha channel for direction calculation

            if(texel_datum.x > 0.0) {
                atomicAdd_f32(&shared_sum_xp[p * 4u + 0u], texel_datum.x);
                atomicAdd_f32(&shared_sum_xp[p * 4u + 1u], texel_datum.y);
                atomicAdd_f32(&shared_sum_xp[p * 4u + 2u], texel_datum.z);
                atomicAdd_f32(&shared_sum_xp[p * 4u + 3u], texel_datum.w);
            }
            if(texel_datum.y > 0.0) {
                atomicAdd_f32(&shared_sum_yp[p * 4u + 0u], texel_datum.x);
				atomicAdd_f32(&shared_sum_yp[p * 4u + 1u], texel_datum.y);
				atomicAdd_f32(&shared_sum_yp[p * 4u + 2u], texel_datum.z);
                atomicAdd_f32(&shared_sum_yp[p * 4u + 3u], texel_datum.w);
            }
            if(texel_datum.z > 0.0) {
				atomicAdd_f32(&shared_sum_zp[p * 4u + 0u], texel_datum.x);
                atomicAdd_f32(&shared_sum_zp[p * 4u + 1u], texel_datum.y);
                atomicAdd_f32(&shared_sum_zp[p * 4u + 2u], texel_datum.z);
                atomicAdd_f32(&shared_sum_zp[p * 4u + 3u], texel_datum.w);
            }
		}
	}

    workgroupBarrier();

    //One thread per partition calculates the best direction
    if(local_idx < partitionCount) {
        let p = local_idx;

        let sum_xp = vec4<f32>(
			bitcast<f32>(atomicLoad(&shared_sum_xp[p * 4u + 0u])),
			bitcast<f32>(atomicLoad(&shared_sum_xp[p * 4u + 1u])),
			bitcast<f32>(atomicLoad(&shared_sum_xp[p * 4u + 2u])),
            bitcast<f32>(atomicLoad(&shared_sum_xp[p * 4u + 3u]))
		);
        let sum_yp = vec4<f32>(
            bitcast<f32>(atomicLoad(&shared_sum_yp[p * 4u + 0u])),
            bitcast<f32>(atomicLoad(&shared_sum_yp[p * 4u + 1u])),
            bitcast<f32>(atomicLoad(&shared_sum_yp[p * 4u + 2u])),
            bitcast<f32>(atomicLoad(&shared_sum_yp[p * 4u + 3u]))
        );
        let sum_zp = vec4<f32>(
            bitcast<f32>(atomicLoad(&shared_sum_zp[p * 4u + 0u])),
            bitcast<f32>(atomicLoad(&shared_sum_zp[p * 4u + 1u])),
            bitcast<f32>(atomicLoad(&shared_sum_zp[p * 4u + 2u])),
            bitcast<f32>(atomicLoad(&shared_sum_zp[p * 4u + 3u]))
        );

        let prod_xp = dot(sum_xp, sum_xp);
        let prod_yp = dot(sum_yp, sum_yp);
        let prod_zp = dot(sum_zp, sum_zp);

        var best_vector = sum_xp;
        var best_sum = prod_xp;

        if(prod_yp > best_sum) {
			best_vector = sum_yp;
			best_sum = prod_yp;
		}
        if(prod_zp > best_sum) {
            best_vector = sum_zp;
            best_sum = prod_zp;
        }

        directions[p] = best_vector;
        averages[p] = vec4<f32>(ideal_endpoints_and_weights_block.partitions[p].avg.xyz, 0.0);
    }

    workgroupBarrier();


    //Prepare processed lines for different endpoint encodings
    if(local_idx < partitionCount) {
        let p = local_idx;
        let avg = averages[p];
        let dir = directions[p];
        //let avg = vec4<f32>(ideal_endpoints_and_weights_block.partitions[p].avg.xyz, 0.0);
        //let dir = vec4<f32>(ideal_endpoints_and_weights_block.partitions[p].dir.xyz, 0.0);

        //Uncorrelated RGB line
        let uncor_b = normalize(dir);
        uncor_rgb_plines[p] = ProcessedLine(avg - uncor_b * dot(avg, uncor_b), uncor_b);

        //Same Chroma line (goes through origin)
        let samec_b = normalize(avg);
        samec_rgb_plines[p] = ProcessedLine(vec4<f32>(0.0), samec_b);

        //RGB Luma line (direction is unit vector)
        let luma_b = normalize(vec4<f32>(1.0, 1.0, 1.0, 0.0));
        rgb_luma_plines[p] = ProcessedLine(avg - luma_b * dot(avg, luma_b), luma_b);

        //Luminance line (goes through origin, direction is unit vector)
        luminance_plines[p] = ProcessedLine(vec4<f32>(0.0), normalize(vec4<f32>(1.0, 1.0, 1.0, 0.0)));
    }

    workgroupBarrier();


    //Compute squared errors for encoding choices
    //initialize error accumulators
    for(var i = local_idx; i < 20; i += WORKGROUP_SIZE) {
		atomicStore(&error_accumulators[i], bitcast<u32>(0.0));
	}

    workgroupBarrier();

    for(var i = local_idx; i < uniforms.texel_count; i += WORKGROUP_SIZE) {
        let p = input_block.texel_partitions[i];

        if(p < partitionCount) {

            let cw = uniforms.channel_weights;
            let rgb_data = input_block.pixels[i].xyz;

            //Alpha drop error
            let alpha_diff = input_block.pixels[i].a - DEFAULT_ALPHA;
            let a_drop_error = alpha_diff * alpha_diff * cw.w;
            atomicAdd_f32(&error_accumulators[p * 5u + 0u], a_drop_error);

            //Uncorrelated RGB error
            let uncor_rgb_line_a = uncor_rgb_plines[p].amod.xyz;
            let uncor_rgb_line_b = uncor_rgb_plines[p].bs.xyz;
            let distance_uncor = (uncor_rgb_line_a + dot(rgb_data, uncor_rgb_line_b) * uncor_rgb_line_b) - rgb_data;
            let uncor_rgb_error = dot(distance_uncor * distance_uncor, cw.xyz);
            atomicAdd_f32(&error_accumulators[p * 5u + 1u], uncor_rgb_error);

            //Same Chroma RGB error
            let samec_rgb_line_b = samec_rgb_plines[p].bs.xyz;
            let distance_samec = dot(rgb_data, samec_rgb_line_b) * samec_rgb_line_b - rgb_data;
            let samec_rgb_error = dot(distance_samec * distance_samec, cw.xyz);
            atomicAdd_f32(&error_accumulators[p * 5u + 2u], samec_rgb_error);

            //RGB Luma error
            let rgb_luma_line_a = rgb_luma_plines[p].amod.xyz;
            let rgb_luma_line_b = rgb_luma_plines[p].bs.xyz;
            let distance_rgb_luma = (rgb_luma_line_a + dot(rgb_data, rgb_luma_line_b) * rgb_luma_line_b) - rgb_data;
            let rgb_luma_error = dot(distance_rgb_luma * distance_rgb_luma, cw.xyz);
            atomicAdd_f32(&error_accumulators[p * 5u + 3u], rgb_luma_error);

            //Luminance error
            let luminance_line_b = luminance_plines[p].bs.xyz;
            let distance_luminance = dot(rgb_data, luminance_line_b) * luminance_line_b - rgb_data;
            let luminance_error = dot(distance_luminance * distance_luminance, cw.xyz);
            atomicAdd_f32(&error_accumulators[p * 5u + 4u], luminance_error);
        }
    }

    workgroupBarrier();


    //Store the accumulated errors for each partition
    if(local_idx < partitionCount) {
        let p = local_idx;

        // Load the final summed errors from the accumulators
        let alpha_drop_err = bitcast<f32>(atomicLoad(&error_accumulators[p * 5u + 0u]));
        let uncorr_err =     bitcast<f32>(atomicLoad(&error_accumulators[p * 5u + 1u]));
        let samec_err =      bitcast<f32>(atomicLoad(&error_accumulators[p * 5u + 2u]));
        let rgb_luma_err =   bitcast<f32>(atomicLoad(&error_accumulators[p * 5u + 3u]));
        let luma_err =       bitcast<f32>(atomicLoad(&error_accumulators[p * 5u + 4u]));

        //determine if offset encoding is possible
        let ep0 = ideal_endpoints_and_weights_block.partitions[p].endpoint0;
        let ep1 = ideal_endpoints_and_weights_block.partitions[p].endpoint1;
        let ep_diff = abs(ep1 - ep0);
        let ep_can_offset = ep_diff < vec4<f32>(0.12f * 65536.0);
        let can_offset_encode = select(0u, 1u, ep_can_offset.x && ep_can_offset.y && ep_can_offset.z);

        var can_blue_contract = 1u; //0 if block is grayscale with constant alpha, 1 otherwise
        if(input_block.grayscale == 1u && input_block.constant_alpha == 1u) {
            can_blue_contract = 0u;
        }

        //Store errors
        //errors are weighted, the weights are determined empirically
        let output_idx = block_index * partitionCount + p;
        let outputPtr = &encoding_choice_errors[output_idx];

        (*outputPtr).rgb_scale_error = (samec_err - uncorr_err) * 0.7;
        (*outputPtr).rgb_luma_error  = (rgb_luma_err - uncorr_err) * 1.5;
        (*outputPtr).luminance_error = (luma_err - uncorr_err) * 3.0;
        (*outputPtr).alpha_drop_error = alpha_drop_err * 3.0;
        (*outputPtr).can_offset_encode = can_offset_encode;
        (*outputPtr).can_blue_contract = can_blue_contract;
    }
}