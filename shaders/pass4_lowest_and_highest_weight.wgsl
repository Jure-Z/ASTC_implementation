const WORKGROUP_SIZE: u32 = 64u;
const BLOCK_MAX_WEIGHTS: u32 = 64u;
const MAX_ANGULAR_STEPS: u32 = 16u;
const BLOCK_MAX_TEXELS: u32 = 144u;

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

//output struct
struct HighestAndLowestWeight {
    lowest_weight : f32,
    weight_span : i32,
    error : f32,
    cut_low_error : f32,
    cut_high_error : f32,

    _padding1 : u32,
    _padding2 : u32,
    _padding3 : u32,
};


@group(0) @binding(0) var<uniform> uniforms: UniformVariables;
@group(0) @binding(1) var<storage, read> valid_decimation_modes: array<u32>;
@group(0) @binding(2) var<storage, read> decimation_infos: array<DecimationInfo>;
@group(0) @binding(3) var<storage, read> ideal_decimated_weights: array<f32>; //pass2 output
@group(0) @binding(4) var<storage, read> angular_offsets: array<f32>; //pass3 output

@group(0) @binding(5) var<storage, read_write> lowest_and_highest_weights: array<HighestAndLowestWeight>; //Output of pass4


var<workgroup> shared_min_w: atomic<u32>;
var<workgroup> shared_max_w: atomic<u32>;
var<workgroup> shared_errval: atomic<u32>;
var<workgroup> shared_cut_low: atomic<u32>;
var<workgroup> shared_cut_high: atomic<u32>;


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
fn main(
   @builtin(workgroup_id) group_id: vec3<u32>,
   @builtin(local_invocation_index) local_idx: u32
) {

    let block_idx = group_id.x;
    let mode_lookup_idx = group_id.y;
    let mode_idx = valid_decimation_modes[mode_lookup_idx];

    let num_valid_modes = uniforms.valid_decimation_mode_count;
    let decimation_mode_trial_idx = block_idx * num_valid_modes + mode_lookup_idx;


    if(mode_idx >= uniforms.decimation_mode_count) {
        return;
    }
    
    let di = decimation_infos[mode_idx];
    let num_weights = di.weight_count;

    if (num_weights == 0u) {
        return;
    }


    //Parallel reduction to find global min/max ideal weights
    if (local_idx == 0u) {
        atomicStore(&shared_min_w, bitcast<u32>(3.4e38)); // FLT_MAX
        atomicStore(&shared_max_w, bitcast<u32>(-3.4e38)); // -FLT_MAX
    }
    workgroupBarrier();

    let ideal_weights_base_idx = decimation_mode_trial_idx * BLOCK_MAX_WEIGHTS;
    for (var i = local_idx; i < num_weights; i += WORKGROUP_SIZE) {
        let weight = ideal_decimated_weights[ideal_weights_base_idx + i];
        atomicMin_f32(&shared_min_w, weight);
        atomicMax_f32(&shared_max_w, weight);
    }
    workgroupBarrier();

    let min_weight = bitcast<f32>(atomicLoad(&shared_min_w));
    let max_weight = bitcast<f32>(atomicLoad(&shared_max_w));
    


    //outer loop over all angular steps. This loop is serial.
    //inner loop parallel reduction over all weights for each step.
    let angular_offsets_base_idx = decimation_mode_trial_idx * MAX_ANGULAR_STEPS;

    for (var sp = 0u; sp < di.max_angular_steps; sp = sp + 1u) {
        let rcp_stepsize = f32(sp + 1u);
        let offset = angular_offsets[angular_offsets_base_idx + sp];
        
        //pre-calculate min/max indices for this angular step
        let minidx = round(min_weight * rcp_stepsize - offset);
        let maxidx = round(max_weight * rcp_stepsize - offset);

        //initialize shared accumulators for this angular step `sp`.
        if (local_idx == 0u) {
            atomicStore(&shared_errval, bitcast<u32>(0.0));
            atomicStore(&shared_cut_low, bitcast<u32>(0.0));
            atomicStore(&shared_cut_high, bitcast<u32>(0.0));
        }
        workgroupBarrier();

        for (var j = local_idx; j < num_weights; j += WORKGROUP_SIZE) {
            let sval = ideal_decimated_weights[ideal_weights_base_idx + j] * rcp_stepsize - offset;
            let svalrte = round(sval);
            let diff = sval - svalrte;
            
            atomicAdd_f32(&shared_errval, diff * diff);

            //if (svalrte == minidx) {
            //    atomicAdd_f32(&shared_cut_low, 1.0 - (2.0 * diff));
            //}
            //if (svalrte == maxidx) {
            //    atomicAdd_f32(&shared_cut_high, 1.0 + (2.0 * diff));
            //}

            //without if statements to avoid divergent threads
            let cut_low_val_to_add = select(0.0, 1.0 - (2.0 * diff), svalrte == minidx);
            atomicAdd_f32(&shared_cut_low, cut_low_val_to_add);

            let cut_high_val_to_add = select(0.0, 1.0 + (2.0 * diff), svalrte == maxidx);
            atomicAdd_f32(&shared_cut_high, cut_high_val_to_add);
        }
        workgroupBarrier();

        //one thread finalizes results for this angular step
        if (local_idx == 0u) {
            let span = i32(maxidx - minidx + 1.0);
            let clamped_span = clamp(span, 2, i32(di.max_quant_steps + 3u));
            
            let ssize = 1.0 / rcp_stepsize;
            let errscale = ssize * ssize;
            
            let final_error = bitcast<f32>(atomicLoad(&shared_errval)) * errscale;
            let final_cut_low = bitcast<f32>(atomicLoad(&shared_cut_low)) * errscale;
            let final_cut_high = bitcast<f32>(atomicLoad(&shared_cut_high)) * errscale;

            let output_base_idx = decimation_mode_trial_idx * MAX_ANGULAR_STEPS;
            let output_ptr = &lowest_and_highest_weights[output_base_idx + sp];

            (*output_ptr).lowest_weight = minidx;
            (*output_ptr).weight_span = clamped_span;
            (*output_ptr).error = final_error;
            (*output_ptr).cut_low_error = final_cut_low;
            (*output_ptr).cut_high_error = final_cut_high;
        }
        //no barrier needed here, as the next loop iteration is independent.
    }
}