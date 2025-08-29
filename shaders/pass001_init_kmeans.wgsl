const BLOCK_MAX_TEXELS : u32 = 144;
const WORKGROUP_SIZE: u32 = 256u;

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

struct InputBlock {
    pixels: array<vec4<f32>, BLOCK_MAX_TEXELS>,
    texel_partitions: array<u32, BLOCK_MAX_TEXELS>,
    partition_pixel_counts: array<u32, 4>,

    partitioning_idx: u32,
    grayscale: u32,
    constant_alpha: u32,
    padding: u32,
};


@group(0) @binding(0) var<uniform> uniforms: UniformVariables;
@group(0) @binding(1) var<storage, read> inputBlocks: array<InputBlock>;
@group(0) @binding(2) var<storage, read_write> cluster_centers: array<vec4<f32>>;


var<workgroup> pixels: array<vec4<f32>, BLOCK_MAX_TEXELS>;
var<workgroup> distances: array<f32, BLOCK_MAX_TEXELS>;
var<workgroup> centers: array<vec4<f32>, 4>;

var<workgroup> s_reduction_value: atomic<u32>;
var<workgroup> s_reduction_index: atomic<u32>;


fn dist_sq(c1: vec4<f32>, c2: vec4<f32>) -> f32 {
    let diff = c1 - c2;
    let diff2 = diff * diff;
    return dot(diff2, uniforms.channel_weights);
}


@compute @workgroup_size(WORKGROUP_SIZE)
fn main( @builtin(workgroup_id) group_id: vec3<u32>, @builtin(local_invocation_index) local_idx: u32) {

    let block_idx = group_id.x;

    if (local_idx < uniforms.texel_count) {
        pixels[local_idx] = inputBlocks[block_idx].pixels[local_idx];
    }
    workgroupBarrier();

    //pick random center for first cluster
    if (local_idx == 0) {
        let sample_index = 145897 % uniforms.texel_count;
        centers[0] = pixels[sample_index];
    }
    workgroupBarrier();

    //compute distances to first center
    if (local_idx < uniforms.texel_count) {
        distances[local_idx] = dist_sq(centers[0], pixels[local_idx]);
    }
    workgroupBarrier();


    //find the remaining centers
    for(var p = 1u; p < uniforms.partition_count; p += 1u) {
        
        //find the farthest point from any center
        if(local_idx == 0) {
            atomicStore(&s_reduction_value, 0u);
            atomicStore(&s_reduction_index, 0u);
        }
        workgroupBarrier();

        //find max distance
        if(local_idx < uniforms.texel_count) {
            atomicMax(&s_reduction_value, bitcast<u32>(distances[local_idx]));
        }
        workgroupBarrier();

        //each thread compares its distance to the max
        if (local_idx < uniforms.texel_count) {
            let max_dist_u32 = atomicLoad(&s_reduction_value);
            if(bitcast<u32>(distances[local_idx]) == max_dist_u32) {
                atomicStore(&s_reduction_index, local_idx);
            }
        }
        workgroupBarrier();

        //thread 0 stores the new center
        if(local_idx == 0) {
			let new_center_index = atomicLoad(&s_reduction_index);
			centers[p] = pixels[new_center_index];
		}
        workgroupBarrier();

        //update distances
        if(p < uniforms.partition_count - 1u) {
            if (local_idx < uniforms.texel_count) {
                let new_dist = dist_sq(centers[p], pixels[local_idx]);
                distances[local_idx] = min(distances[local_idx], new_dist);
            }
            workgroupBarrier();
        }
    }

    //write out the centers
    if(local_idx < uniforms.partition_count) {
        let out_idx = block_idx * 4u + local_idx;
        cluster_centers[out_idx] = centers[local_idx];
    }

}