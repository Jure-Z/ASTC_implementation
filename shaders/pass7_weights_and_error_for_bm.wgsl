const WORKGROUP_SIZE: u32 = 64u;
const BLOCK_MAX_TEXELS: u32 = 144u;
const BLOCK_MAX_WEIGHTS: u32 = 64u;

const FREE_BITS_FOR_PARTITION_COUNT = array<i32, 4>(111, 111 - 4 - 6, 108 - 4 - 6, 105 - 4 - 6);
const QUANT_MODES = array<u32, 12>(2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32);

const QUANT_TABLE_OFFSETS = array<u32, 12>(0, 2, 5, 9, 14, 20, 28, 38, 50, 66, 86, 110);

//packed values of quant_to_unquant table from the ARM implementation
//get offset for quantization level from QUANT_TABLE_OFFSETS
const QUANT_TO_UNQUANT = array<u32, 142>(
    // QUANT_2
    0u, 64u,
    // QUANT_3
    0u, 32u, 64u,
    // QUANT_4
    0u, 21u, 43u, 64u,
    // QUANT_5
    0u, 16u, 32u, 48u, 64u,
    // QUANT_6
    0u, 12u, 25u, 39u, 52u, 64u,
    // QUANT_8
    0u, 9u, 18u, 27u, 37u, 46u, 55u, 64u,
    // QUANT_10
    0u, 7u, 14u, 21u, 28u, 36u, 43u, 50u, 57u, 64u,
    // QUANT_12
    0u, 5u, 11u, 17u, 23u, 28u, 36u, 41u, 47u, 53u, 59u, 64u,
    // QUANT_16
    0u, 4u, 8u, 12u, 17u, 21u, 25u, 29u, 35u, 39u, 43u, 47u, 52u, 56u, 60u, 64u,
    // QUANT_20
    0u, 3u, 6u, 9u, 13u, 16u, 19u, 23u, 26u, 29u, 35u, 38u, 41u, 45u, 48u, 51u, 55u, 58u, 61u, 64u,
    // QUANT_24
    0u, 2u, 5u, 8u, 11u, 13u, 16u, 19u, 22u, 24u, 27u, 30u, 34u, 37u, 40u, 42u, 45u, 48u, 51u, 53u, 56u, 59u, 62u, 64u,
    // QUANT_32
    0u, 2u, 4u, 6u, 8u, 10u, 12u, 14u, 16u, 18u, 20u, 22u, 24u, 26u, 28u, 30u, 34u, 36u, 38u, 40u, 42u, 44u, 46u, 48u, 50u, 52u, 54u, 56u, 58u, 60u, 62u, 64u,
);


struct UniformVariables {
    xdim : u32,
    ydim : u32,

    texel_count : u32,

    decimation_mode_count : u32,
    block_mode_count : u32,

    quant_limit : u32,
    partition_count : u32,
    tune_candidate_limit : u32,

    channel_weights : vec4<f32>,
};

struct BlockModeTrial {
    block_index : u32,
    block_mode_index : u32,
    decimation_mode_trial_index : u32,

    _padding1 : u32,
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

struct FinalValueRange {
    low : f32,
    high : f32,

    _padding1 : u32,
    _padding2 : u32,
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

struct TexelToWeightMap {
	weight_index : u32,
	contribution : f32,

    _padding1 : u32,
    _padding2 : u32,
};

struct QuantizationResult {
    error: f32,
    bitcount: i32,

    _padding1: u32,
    _padding2: u32,

    quantized_weights: array<u32, BLOCK_MAX_WEIGHTS>,
};

@group(0) @binding(0) var<uniform> uniforms: UniformVariables;
@group(0) @binding(1) var<storage, read> block_mode_trials: array<BlockModeTrial>;
@group(0) @binding(2) var<storage, read> block_modes: array<BlockMode>;
@group(0) @binding(3) var<storage, read> decimation_infos: array<DecimationInfo>;
@group(0) @binding(4) var<storage, read> ideal_decimated_weights: array<f32>;
@group(0) @binding(5) var<storage, read> final_value_ranges: array<FinalValueRange>;
@group(0) @binding(6) var<storage, read> ideal_endpoints_and_weights: array<IdealEndpointsAndWeights>;
@group(0) @binding(7) var<storage, read> texel_to_weight_map: array<TexelToWeightMap>;

@group(0) @binding(8) var<storage, read_write> output_quantization_results: array<QuantizationResult>;


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




var<workgroup> shared_skip_work: u32;
var<workgroup> shared_bitcount: i32; // Store bitcount for the final writer thread
var<workgroup> shared_quantized_weights_float: array<f32, BLOCK_MAX_WEIGHTS>;
var<workgroup> shared_quantized_weights_int: array<u32, BLOCK_MAX_WEIGHTS>;
var<workgroup> shared_infilled_weights: array<f32, BLOCK_MAX_TEXELS>;
var<workgroup> shared_total_error: atomic<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(workgroup_id) group_id: vec3<u32>, @builtin(local_invocation_index) local_idx: u32) {

    let block_mode_trial_index = group_id.x;

    let block_mode_index = block_mode_trials[block_mode_trial_index].block_mode_index;
    let bm = block_modes[block_mode_index];
    let block_index = block_mode_trials[block_mode_trial_index].block_index;

    //step 1: setup and filtering
    if (local_idx == 0u) {
        shared_skip_work = 0u;

        let max_weight_quant = min(uniforms.quant_limit, 11u); //QUANT_32

        if (bm.quant_mode > max_weight_quant) {
            output_quantization_results[block_mode_trial_index].error = 1e38;
            output_quantization_results[block_mode_trial_index].bitcount = 0;
            shared_skip_work = 1u;
        } else {
            let bc = FREE_BITS_FOR_PARTITION_COUNT[uniforms.partition_count - 1] - i32(bm.weight_bits);
            if (bc <= 0) {
                output_quantization_results[block_mode_trial_index].error = 1e38;
                output_quantization_results[block_mode_trial_index].bitcount = 0;
                shared_skip_work = 1u;
            } else {
                shared_bitcount = bc;
            }
        }
    }
    workgroupBarrier();

    if (workgroupUniformLoad(&shared_skip_work) == 1u) {
        return;
    }


    let di = decimation_infos[bm.decimation_mode];
    let ei = ideal_endpoints_and_weights[block_index];
    var value_range = final_value_ranges[block_mode_trial_index];

    if(value_range.high > 1.02f * ei.min_weight_cuttof) {
        value_range.high = 1.0;
    }

    if (value_range.high <= value_range.low) {
        value_range.low = 0.0;
        value_range.high = 1.0;
    }

    //step 2: compute quantized weights
    let num_weights = di.weight_count;
    let quant_steps = f32(QUANT_MODES[bm.quant_mode]);

    let quant_level = bm.quant_mode;
    let quant_level_m1 = f32(QUANT_MODES[quant_level] - 1u);

    var rscale = value_range.high - value_range.low;
    let scale = 1.0 / rscale;

    let scaled_low_bound = value_range.low * scale;
    rscale = rscale / 64.0f;

    let ideal_weight_base_idx = block_mode_trials[block_mode_trial_index].decimation_mode_trial_index * BLOCK_MAX_WEIGHTS;

    for (var i = local_idx; i < num_weights; i += WORKGROUP_SIZE) {
        let ideal_weight = ideal_decimated_weights[ideal_weight_base_idx + i];

        let ix = clamp(ideal_weight * scale - scaled_low_bound, 0.0, 1.0);

        let ix1 = ix * quant_level_m1;
        let weightl = u32(ix1);
        let weighth = min(weightl + 1u, u32(quant_level_m1));

        let table_base_idx = QUANT_TABLE_OFFSETS[quant_level];
        let ixli = QUANT_TO_UNQUANT[table_base_idx + weightl];
        let ixhi = QUANT_TO_UNQUANT[table_base_idx + weighth];
    
        let ixl = f32(ixli);
        let ixh = f32(ixhi);

        // Find the best match
        let use_high = (ixl + ixh) < (128.0 * ix);
        let weight_int = select(ixli, ixhi, use_high);
        let unquant_val = select(ixl, ixh, use_high);

        // Store the results
        shared_quantized_weights_int[i] = weight_int;
        // Rescale the unquantized value back to the original range
        shared_quantized_weights_float[i] = unquant_val * rscale + value_range.low;
    }
    workgroupBarrier();



    //step 3: Compute error of quantized weights
    if (local_idx == 0u) {
        atomicStore(&shared_total_error, bitcast<u32>(0.0));
    }
    workgroupBarrier();


    //bilinear infill using the newly quantized weights
    for (var i = local_idx; i < di.texel_count; i += WORKGROUP_SIZE) {
        let weight_count = di.texel_weight_count[i];
        let weight_offset = di.texel_weights_offset[i];
        var infill_val: f32 = 0.0;
        
        for (var j: u32 = 0u; j < weight_count; j = j + 1u) {
            let mapping = texel_to_weight_map[weight_offset + j];
            infill_val += shared_quantized_weights_float[mapping.weight_index] * mapping.contribution;
        }
        shared_infilled_weights[i] = infill_val / 16.0;
    }
    workgroupBarrier();
    
    //sum the squared error
    for (var i = local_idx; i < di.texel_count; i += WORKGROUP_SIZE) {
        let diff = shared_infilled_weights[i] - ei.weights[i];
        let error_scale = ei.weight_error_scale[select(0u, i, ei.is_constant_weight_error_scale == 0u)];
        atomicAdd_f32(&shared_total_error, diff * diff * error_scale);
    }
    workgroupBarrier();

    //step4: store final values
    //one thread writes the error and bitcount
    if (local_idx == 0u) {
        let result_ptr = &output_quantization_results[block_mode_trial_index];
        (*result_ptr).error = bitcast<f32>(atomicLoad(&shared_total_error));
        (*result_ptr).bitcount = shared_bitcount;
    }

    // All threads help copy the final integer weights to global memory.
    for (var i = local_idx; i < num_weights; i += WORKGROUP_SIZE) {
        let result_ptr = &output_quantization_results[block_mode_trial_index];
        (*result_ptr).quantized_weights[i] = shared_quantized_weights_int[i];
    }
}