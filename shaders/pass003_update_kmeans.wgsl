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
@group(0) @binding(2) var<storage, read> texel_assignments : array<u32>;

@group(0) @binding(3) var<storage, read_write> cluster_centers: array<vec4<f32>>;


var<workgroup> partition_sums: array<array<atomic<u32>, 4>, 4>;
var<workgroup> partition_counts: array<atomic<u32>, 4>;


@compute @workgroup_size(WORKGROUP_SIZE)
fn main( @builtin(workgroup_id) group_id: vec3<u32>, @builtin(local_invocation_index) local_idx: u32) {
    
    let block_idx = group_id.x;

    //init atomic variables to zero
    if (local_idx < uniforms.partition_count) {
        atomicStore(&partition_counts[local_idx], 0u);
        atomicStore(&partition_sums[local_idx][0], 0u); // R
        atomicStore(&partition_sums[local_idx][1], 0u); // G
        atomicStore(&partition_sums[local_idx][2], 0u); // B
        atomicStore(&partition_sums[local_idx][3], 0u); // A
    }
    workgroupBarrier();

    // accumulate partition sums and counts
    if (local_idx < uniforms.texel_count) {
        let global_idx = block_idx * BLOCK_MAX_TEXELS + local_idx;
        let p = texel_assignments[global_idx];
        let pixel = inputBlocks[block_idx].pixels[local_idx];
        
        // Convert f32 color to u32 fixed-point for atomic operations.
        let pixel_u32 = vec4<u32>(pixel);

        // Add to the appropriate partition's sum and count.
        atomicAdd(&partition_sums[p][0], pixel_u32.r);
        atomicAdd(&partition_sums[p][1], pixel_u32.g);
        atomicAdd(&partition_sums[p][2], pixel_u32.b);
        atomicAdd(&partition_sums[p][3], pixel_u32.a);
        atomicAdd(&partition_counts[p], 1u);
    }
    workgroupBarrier();


    // Compute final cluster centers
    if (local_idx < uniforms.partition_count) {

        let p = local_idx;
        let count = atomicLoad(&partition_counts[p]);

        var new_center = vec4<f32>(0.0);

        if(count > 0u) {
            let sum_r = f32(atomicLoad(&partition_sums[p][0]));
            let sum_g = f32(atomicLoad(&partition_sums[p][1]));
            let sum_b = f32(atomicLoad(&partition_sums[p][2]));
            let sum_a = f32(atomicLoad(&partition_sums[p][3]));

            let f_count = f32(count);

            new_center = vec4f(
                sum_r / f_count,
                sum_g / f_count,
                sum_b / f_count,
                sum_a / f_count
            );
        }

        cluster_centers[block_idx * 4u + p] = new_center;
        
    }
}