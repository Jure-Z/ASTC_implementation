const WORKGROUP_SIZE: u32 = 64u;
const MAX_ANGULAR_QUANT = 12; // The max number of distinct quant levels to test
const MAX_BEST_RESULTS = 36;  // A safe upper bound for the best_results array size
const BLOCK_MAX_TEXELS: u32 = 144u;
const BLOCK_MAX_WEIGHTS: u32 = 64u;
const MAX_ANGULAR_STEPS: u32 = 16u;

const STEPS_FOR_QUANT_LEVEL = array(
	2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32
);

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

struct HighestAndLowestWeight {
    lowest_weight : f32,
    weight_span : i32,
    error : f32,
    cut_low_error : f32,
    cut_high_error : f32,

    _padding1 : f32,
    _padding2 : f32,
    _padding3 : f32,
};


@group(0) @binding(0) var<uniform> uniforms: UniformVariables;
@group(0) @binding(1) var<storage, read> valid_decimation_modes: array<u32>;
@group(0) @binding(2) var<storage, read> decimation_infos: array<DecimationInfo>;
@group(0) @binding(3) var<storage, read> angular_offsets: array<f32>; //pass3 output
@group(0) @binding(4) var<storage, read> lowest_and_highest_weights: array<HighestAndLowestWeight>; //Output of pass4

@group(0) @binding(5) var<storage, read_write> output_final_low_values: array<f32>;
@group(0) @binding(6) var<storage, read_write> output_final_high_values: array<f32>;


//here we pack 3 values into a vector for efficiency and to be consistant with the ARM implementation
//val0: error
//val1: best_step_index (stored as f32)
//val2: cut_low_flag (0.0 if flase, 1.0 if true)
var<workgroup> shared_best_results: array<vec3<f32>, MAX_BEST_RESULTS>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(workgroup_id) group_id: vec3<u32>, @builtin(local_invocation_index) local_idx: u32) {

    let block_idx = group_id.x;
    let mode_lookup_idx = group_id.y;
    let mode_idx = valid_decimation_modes[mode_lookup_idx];

    let num_valid_modes = uniforms.valid_decimation_mode_count;
    let decimation_mode_trial_idx = block_idx * num_valid_modes + mode_lookup_idx;


    if(mode_idx >= uniforms.decimation_mode_count) {
        return;
    }

    let di = decimation_infos[mode_idx];

    //initalization
    for (var i = local_idx; i < di.max_quant_steps + 4u; i += WORKGROUP_SIZE) {
        if (i < MAX_BEST_RESULTS) {
            shared_best_results[i] = vec3(1e37, -1.0, 0.0);
        }
    }
    workgroupBarrier();

    //serial reduction. One thread performs the small, complex loop.
    if (local_idx == 0u) {
        let trial_results_base_idx = decimation_mode_trial_idx * MAX_ANGULAR_STEPS;
        for (var i = 0u; i < di.max_angular_steps; i = i + 1u) {
            let trial_idx = trial_results_base_idx + i;
            let i_flt = f32(i);
            
            let idx_span = lowest_and_highest_weights[trial_idx].weight_span;
            let error = lowest_and_highest_weights[trial_idx].error;
            let cut_low_err = lowest_and_highest_weights[trial_idx].cut_low_error;
            let cut_high_err = lowest_and_highest_weights[trial_idx].cut_high_error;
            
            let error_cut_low = error + cut_low_err;
            let error_cut_high = error + cut_high_err;
            let error_cut_low_high = error + cut_low_err + cut_high_err;


            // Check against record N (span)
            var best_result = shared_best_results[idx_span];
            var new_result = vec3(error, i_flt, 0.0);
            shared_best_results[idx_span] = select(best_result, new_result, error < best_result[0]); //best result[0] corresponds to error of best result

            // Check against record N-1 (span - 1) (cut_low)
            best_result = shared_best_results[idx_span - 1];
            new_result = vec3(error_cut_low, i_flt, 1.0);
            shared_best_results[idx_span - 1] = select(best_result, new_result, error_cut_low < best_result[0]);

            // Check against record N-1 (span - 1) (cut_high)
            best_result = shared_best_results[idx_span - 1];
            new_result = vec3(error_cut_high, i_flt, 0.0);
            shared_best_results[idx_span - 1] = select(best_result, new_result, error_cut_high < best_result[0]);

            // Check against record N-2 (span - 2)
            best_result = shared_best_results[idx_span - 2];
            new_result = vec3(error_cut_low_high, i_flt, 1.0);
            shared_best_results[idx_span - 2] = select(best_result, new_result, error_cut_low_high < best_result[0]);

        }
    }
    workgroupBarrier();

    //finalization
    for (var i = local_idx; i <= di.max_quant_level; i += WORKGROUP_SIZE) {
        let q_level_idx = i;
        let q_steps = STEPS_FOR_QUANT_LEVEL[q_level_idx]; // The number of steps, e.g., 12

        // Read the final winning result from shared memory
        let winner = shared_best_results[q_steps];
        var bsi_signed = i32(winner[1]); //best step index
        bsi_signed = max(bsi_signed, 0); // Handle the -1 "not found" case
        let bsi = u32(bsi_signed);

        let trial_results_base_idx = decimation_mode_trial_idx * MAX_ANGULAR_STEPS;
        let lwi = lowest_and_highest_weights[trial_results_base_idx + bsi].lowest_weight + winner[2]; //winner[2] corresponds to cut_low_flag
        let hwi = lwi + f32(q_steps) - 1.0;
        
        let stepsize = 1.0 / (1.0 + f32(bsi));
        let offset = angular_offsets[trial_results_base_idx + bsi];

        let low_val = (offset + lwi) * stepsize;
        let high_val = (offset + hwi) * stepsize;

        let output_base_idx = decimation_mode_trial_idx * (MAX_ANGULAR_QUANT + 1);
        output_final_low_values[output_base_idx + q_level_idx] = low_val;
        output_final_high_values[output_base_idx + q_level_idx] = high_val;
    }
}