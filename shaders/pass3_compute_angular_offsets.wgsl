const WORKGROUP_SIZE: u32 = 64u;
const BLOCK_MAX_TEXELS: u32 = 144u; // Max texels (e.g., 12x12)
const BLOCK_MAX_WEIGHTS: u32 = 64u;  // Max decimated weights (e.g., 8x8)
const MAX_ANGULAR_STEPS: u32 = 16u;
const SINCOS_STEPS: f32 = 1024.0;
const PI: f32 = 3.14159265358979323846;


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

struct DecimationModeTrial {
    block_idx : u32,
    mode_idx : u32,

    _padding1 : u32,
    _padding2 : u32,
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


@group(0) @binding(0) var<uniform> uniforms: UniformVariables;
@group(0) @binding(1) var<storage, read> decimation_mode_trials: array<DecimationModeTrial>;
@group(0) @binding(2) var<storage, read> decimation_infos: array<DecimationInfo>;
@group(0) @binding(3) var<storage, read> sin_table_flat: array<f32>; //size is SINCOS_STEPS * MAX_ANGULAR_STEPS
@group(0) @binding(4) var<storage, read> cos_table_flat: array<f32>; //size is SINCOS_STEPS * MAX_ANGULAR_STEPS
@group(0) @binding(5) var<storage, read> ideal_decimated_weights: array<f32>; //output buffer of pass 2
@group(0) @binding(6) var<storage, read_write> output_angular_offsets: array<f32>; //output (size is decimation_mode_trails * MAX_ANGULAR_STEPS)


var<workgroup> shared_isamples: array<u32, BLOCK_MAX_WEIGHTS>;


@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32
) {
    let decimation_mode_trial_idx = group_id.x;

    let block_idx = decimation_mode_trials[decimation_mode_trial_idx].block_idx;
    let mode_idx = decimation_mode_trials[decimation_mode_trial_idx].mode_idx;

    if(mode_idx >= uniforms.decimation_mode_count) {
        return;
    }

    let di = decimation_infos[mode_idx];
    let num_weights = di.weight_count;

    //precompute the sample indices
    for (var i = local_idx; i < num_weights; i += WORKGROUP_SIZE) {
        let ideal_weight_base_idx = (block_idx * uniforms.decimation_mode_count + mode_idx) * BLOCK_MAX_WEIGHTS;
        let ideal_weight = ideal_decimated_weights[ideal_weight_base_idx + i];
        let sample = clamp(ideal_weight, 0.0, 1.0) * (SINCOS_STEPS - 1.0);
        shared_isamples[i] = u32(round(sample));
    }
    workgroupBarrier(); //sync

    //compute the final angular offsets
    for (var i = local_idx; i < MAX_ANGULAR_STEPS; i += WORKGROUP_SIZE) {
        var anglesum_x: f32 = 0.0;
        var anglesum_y: f32 = 0.0;
        
        // Each thread loops through all weights to calculate its assigned offset.
        for (var j: u32 = 0u; j < num_weights; j = j + 1u) {
            let isample = shared_isamples[j];
            
            //Calculate the 1D index in the flattened table
            let flat_idx = isample * MAX_ANGULAR_STEPS + i;

            anglesum_x += cos_table_flat[flat_idx];
            anglesum_y += sin_table_flat[flat_idx];
        }

        let angle = atan2(anglesum_y, anglesum_x);
        let offset = angle * (1.0 / (2.0 * PI));
        
        let output_base_idx = decimation_mode_trial_idx * MAX_ANGULAR_STEPS;
        output_angular_offsets[output_base_idx + i] = offset;
    }
}

