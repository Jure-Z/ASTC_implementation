const WORKGROUP_SIZE: u32 = 64u;
const MAX_ANGULAR_QUANT = 12;
const MAX_BEST_RESULTS = 36;
const BLOCK_MAX_TEXELS: u32 = 144u;
const BLOCK_MAX_WEIGHTS: u32 = 64u;
const MAX_ANGULAR_STEPS: u32 = 16u;


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

struct FinalValueRange {
    low : f32,
    high : f32,

    _padding1 : u32,
    _padding2 : u32,
};


@group(0) @binding(0) var<uniform> uniforms: UniformVariables;
@group(0) @binding(1) var<storage, read> block_mode_trials: array<BlockModeTrial>;
@group(0) @binding(2) var<storage, read> block_modes: array<BlockMode>;
@group(0) @binding(3) var<storage, read> source_low_values: array<f32>;
@group(0) @binding(4) var<storage, read> source_high_values: array<f32>;

@group(0) @binding(5) var<storage, read_write> output_final_value_ranges: array<FinalValueRange>;


@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {

    let block_mode_trial_index = global_id.x;

    let block_mode_index = block_mode_trials[block_mode_trial_index].block_mode_index;
    let decimation_mode_trial_index = block_mode_trials[block_mode_trial_index].decimation_mode_trial_index;

    let bm = block_modes[block_mode_index];

    let quant_mode = bm.quant_mode;

    // Check if the quantization level is within the range handled by the angular search.
    if (quant_mode <= MAX_ANGULAR_QUANT) {   
        let source_idx = decimation_mode_trial_index * (MAX_ANGULAR_QUANT + 1) + quant_mode;

        output_final_value_ranges[block_mode_trial_index].low = source_low_values[source_idx];
        output_final_value_ranges[block_mode_trial_index].high = source_high_values[source_idx];
    } else {
        output_final_value_ranges[block_mode_trial_index].low = 0.0;
        output_final_value_ranges[block_mode_trial_index].high = 1.0;
    }
}