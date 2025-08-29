const BLOCK_MAX_TEXELS: u32 = 144u;
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
@group(0) @binding(2) var<storage, read> cluster_centers: array<vec4<f32>>;

@group(0) @binding(3) var<storage, read_write> texel_assignments : array<u32>;


fn dist_sq(c1: vec4f, c2: vec4f) -> f32 {
    let diff = c1 - c2;
    let diff2 = diff * diff;
    return dot(diff2, uniforms.channel_weights);
}


var<workgroup> centers: array<vec4f, 4>;

var<workgroup> partition_counts: array<atomic<u32>, 4>;
var<workgroup> is_problematic: atomic<u32>;


@compute @workgroup_size(WORKGROUP_SIZE)
fn main( @builtin(workgroup_id) group_id: vec3<u32>, @builtin(local_invocation_index) local_idx: u32) {

    let block_idx = group_id.x;

    //copy cluster centers to local memory
    if(local_idx < uniforms.partition_count) {
        centers[local_idx] = cluster_centers[block_idx * 4u + local_idx];
        atomicStore(&partition_counts[local_idx], 0u);
    }
    workgroupBarrier();


    //find colosest center for every texel
    if(local_idx < uniforms.texel_count) {
        let pixel = inputBlocks[block_idx].pixels[local_idx];

        var best_dist = 1e30; // Initialize with a very large number
        var best_partition_idx = 0u;

        for (var p = 0u; p < uniforms.partition_count; p = p + 1u) {
		    let center = centers[p];
		    let d = dist_sq(pixel, center);
		    if (d < best_dist) {
			    best_dist = d;
			    best_partition_idx = p;
		    }
	    }

        let out_idx = block_idx * BLOCK_MAX_TEXELS + local_idx;
        texel_assignments[out_idx] = best_partition_idx;
        atomicAdd(&partition_counts[best_partition_idx], 1u);
    }
    workgroupBarrier();


    if (local_idx == 0) {
        atomicStore(&is_problematic, 0u);
    }
    workgroupBarrier();

    //check final counts
    if (local_idx < uniforms.partition_count) {
        if (atomicLoad(&partition_counts[local_idx]) == 0u) {
            // If a partition is empty, set the shared flag.
            atomicStore(&is_problematic, 1u);
        }
    }
    workgroupBarrier();

    //if any of the clusters were empty, forcibly reasign texels
    let problem_found = (atomicLoad(&is_problematic) == 1u);
    if (problem_found) {
        if (local_idx < uniforms.partition_count) {
            let out_idx = block_idx * BLOCK_MAX_TEXELS + local_idx;
            texel_assignments[out_idx] = local_idx;
        }
    }
}