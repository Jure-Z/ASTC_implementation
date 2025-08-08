const WORKGROUP_SIZE: u32 = 64u;
const BLOCK_MAX_TEXELS: u32 = 144u; // Max texels (e.g., 12x12)
const BLOCK_MAX_WEIGHTS: u32 = 64u;  // Max decimated weights (e.g., 8x8)

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

struct TexelToWeightMap {
	weight_index : u32,
	contribution : f32,

    _padding1 : u32,
    _padding2 : u32,
};

struct WeightToTexelMap {
    texel_index : u32,
	contribution : f32,

    _padding1 : u32,
    _padding2 : u32,
};

struct IdealEndpointsAndWeightsPartition { //same as OutputPartition in shader pass 1
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


@group(0) @binding(0) var<uniform> uniforms: UniformVariables;
@group(0) @binding(1) var<storage, read> decimation_mode_trials: array<DecimationModeTrial>;
@group(0) @binding(2) var<storage, read> decimation_infos: array<DecimationInfo>;
@group(0) @binding(3) var<storage, read> texel_to_weight_map: array<TexelToWeightMap>;
@group(0) @binding(4) var<storage, read> weight_to_texel_map: array<WeightToTexelMap>;
@group(0) @binding(5) var<storage, read> ideal_endpoints_and_weights: array<IdealEndpointsAndWeights>;
@group(0) @binding(6) var<storage, read_write> output_ideal_decimated_weights: array<f32>; //output buffer


//Workgroup shared memory
//decimated weights
var<workgroup> shared_decimated_weights: array<f32, BLOCK_MAX_WEIGHTS>;

//Reconstructed weight grid from the initial estimation of decimated weights
var<workgroup> shared_infilled_weights: array<f32, BLOCK_MAX_TEXELS>;

// Temporary storage for the refinement step's error calculation (the values are actually f32, but are stored as u32)
var<workgroup> shared_error_change0: array<atomic<u32>, BLOCK_MAX_WEIGHTS>;
var<workgroup> shared_error_change1: array<atomic<u32>, BLOCK_MAX_WEIGHTS>;


//Utility function for atomic add of f32 values, since this is not supproted directly
fn atomicAdd_f32(atomic_target: ptr<workgroup, atomic<u32>>, value_to_add: f32) {
    //compare-and-swap loop: continues until the atomic exchange is successful
    loop {
        //atomically load the current value as an integer
        let original_val_uint = atomicLoad(atomic_target);

        //re-interpret the bits of the integer as a float, without conversion
        let original_val_float = bitcast<f32>(original_val_uint);

        //add float value
        let new_val_float = original_val_float + value_to_add;

        //re-interpret the bits of the new float value back to integer
        let new_val_uint = bitcast<u32>(new_val_float);

        //attempt to swap
        let result = atomicCompareExchangeWeak(atomic_target, original_val_uint, new_val_uint);

        //if the swap was successful (result.exchanged is true), break the loop
        if (result.exchanged) {
            break;
        }
    }
}


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
    let ei = ideal_endpoints_and_weights[block_idx];


    // Shortcut for 1:1 mapping (direct copy)
    if (di.texel_count == di.weight_count) {
        
        for (var i = local_idx; i < di.weight_count; i += WORKGROUP_SIZE) {
            //let output_idx = (block_idx * uniforms.decimation_mode_count + mode_idx) * BLOCK_MAX_WEIGHTS + i;
            let output_idx = decimation_mode_trial_idx * BLOCK_MAX_WEIGHTS + i;
            output_ideal_decimated_weights[output_idx] = ei.weights[i];
        }
        return;
    }


    //step1: Initial estimaton of decimated weights
    for (var w = local_idx; w < di.weight_count; w += WORKGROUP_SIZE) {
        let texel_count = di.weight_texel_count[w];
        if (texel_count == 0u) {
            shared_decimated_weights[w] = 0.0;
            continue;
        }

        let texel_offset = di.weight_texels_offset[w];
        var weight_sum: f32 = 0.0;
        var total_contrib: f32 = 1e-10; // Avoid div-by-zero

        for (var j: u32 = 0u; j < texel_count; j = j + 1u) {
            let packed_idx = texel_offset + j;

            let texel_idx = weight_to_texel_map[packed_idx].texel_index;
            let contrib = weight_to_texel_map[packed_idx].contribution;

            let error_scale = ei.weight_error_scale[select(0u, texel_idx, ei.is_constant_weight_error_scale == 0u)]; //optimise memory access
            let contrib_weight = contrib * error_scale;

            weight_sum += ei.weights[texel_idx] * contrib_weight;
            total_contrib += contrib_weight;
        }
        shared_decimated_weights[w] = weight_sum / total_contrib;
    }
    workgroupBarrier(); // SYNC: Ensure all initial weights are calculated


    //step2: Bilinear Infilling
    for (var i = local_idx; i < di.texel_count; i += WORKGROUP_SIZE) {
        let weight_count = di.texel_weight_count[i];
        let weight_offset = di.texel_weights_offset[i];
        var infill_val: f32 = 0.0;

        for (var j: u32 = 0u; j < weight_count; j = j + 1u) {
            let packed_idx = weight_offset + j;

            let weight_idx = texel_to_weight_map[packed_idx].weight_index;
            let contrib = texel_to_weight_map[packed_idx].contribution;
            
            // Read from shared memory
            infill_val += shared_decimated_weights[weight_idx] * contrib;
        }

        shared_infilled_weights[i] = infill_val;
    }
    workgroupBarrier(); // SYNC: Ensure the entire infilled grid is complete


    //step3: Single refinement iteration

    //initialize the atomic error accumulators.
    if (local_idx < di.weight_count) {
        atomicStore(&shared_error_change0[local_idx], bitcast<u32>(1e-10)); // Avoid div-by-zero
        atomicStore(&shared_error_change1[local_idx], bitcast<u32>(0.0));
    }
    workgroupBarrier(); // SYNC: Ensure initialization is done.

    for (var w = local_idx; w < di.weight_count; w += WORKGROUP_SIZE) {
        let texel_count = di.weight_texel_count[w];
        if (texel_count == 0u) {
            continue;
        }

        let texel_offset = di.weight_texels_offset[w];
        for (var j: u32 = 0u; j < texel_count; j = j + 1u) {
            let packed_idx = texel_offset + j;

            let texel_idx = weight_to_texel_map[packed_idx].texel_index;
            let contrib = weight_to_texel_map[packed_idx].contribution;

            // Read from shared memory and global memory
            let old_weight = shared_infilled_weights[texel_idx];
            let ideal_weight = ei.weights[texel_idx];
            
            let error_scale = ei.weight_error_scale[select(0u, texel_idx, ei.is_constant_weight_error_scale == 0u)];
            let scale = error_scale * contrib;

            // Atomically accumulate the error changes
            atomicAdd_f32(&shared_error_change0[w], contrib * scale);
            atomicAdd_f32(&shared_error_change1[w], (old_weight - ideal_weight) * scale);
        }
    }
    workgroupBarrier(); // SYNC: Ensure all error contributions are summed

    //one thread per weight calculates the final step and updates the weight
    if (local_idx < di.weight_count) {
        let err0 = bitcast<f32>(atomicLoad(&shared_error_change0[local_idx]));
        let err1 = bitcast<f32>(atomicLoad(&shared_error_change1[local_idx]));

        let step = (err1 * -16.0) / err0; // chd_scale is -WEIGHTS_TEXEL_SUM = -16
        let clamped_step = clamp(step, -0.25, 0.25); //step size is 0.25
        
        shared_decimated_weights[local_idx] += clamped_step;
    }
    workgroupBarrier(); // SYNC: Ensure all weights are updated with the step.


    //Write refined weights from shared memory to global output
    for (var w = local_idx; w < di.weight_count; w += WORKGROUP_SIZE) {
        //let output_idx = (block_idx * uniforms.decimation_mode_count + mode_idx) * BLOCK_MAX_WEIGHTS + w;
        let output_idx = decimation_mode_trial_idx * BLOCK_MAX_WEIGHTS + w;
        output_ideal_decimated_weights[output_idx] = shared_decimated_weights[w];
    }

}

