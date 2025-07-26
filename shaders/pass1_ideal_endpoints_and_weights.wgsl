const BLOCK_MAX_TEXELS : u32 = 144;

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

struct Pixel {
    data: vec4<f32>,
    partitionNum: u32,

    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
};

struct InputBlock {
    pixels: array<Pixel, BLOCK_MAX_TEXELS>,
    partition_pixel_counts: array<u32, 4>,
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


@group(0) @binding(0) var<uniform> uniforms: UniformVariables;
@group(0) @binding(1) var<storage, read> inputBlocks: array<InputBlock>;
@group(0) @binding(2) var<storage, read_write> outputBlocks: array<IdealEndpointsAndWeights>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let blockIndex = global_id.x;
    let inputBlock = inputBlocks[blockIndex];

    let texelCount = uniforms.texel_count;

    let cw = uniforms.channel_weights;
    let error_weight = (cw.x + cw.y + cw.z + cw.w) / 4.0f;

    var is_constant_wes = true;

    //compute per partition averages
    var sum: array<vec4<f32>, 4>;
    var count: array<f32, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        sum[i] = vec4<f32>(0.0);
        count[i] = 0.0;
    }

    for (var i = 0u; i < texelCount; i = i + 1u) {
        let p = inputBlock.pixels[i].partitionNum;
        if (p < 4u) {
            sum[p] += inputBlock.pixels[i].data;
            count[p] += 1.0;
        }
    }

    var avg: array<vec4<f32>, 4>;
    for (var p = 0u; p < 4u; p = p + 1u) {
        avg[p] = select(vec4<f32>(0.0), sum[p] / count[p], count[p] > 0.0);

        //store avg
        outputBlocks[blockIndex].partitions[p].avg = avg[p];
    }


    //compute dominant direction for each partition (using the method of positive deviations)

    var direction: array<vec4<f32>, 4>;
    var minProj: array<f32, 4>;
    var maxProj: array<f32, 4>;

    for (var p = 0u; p < 4u; p = p + 1u) {
        direction[p] = vec4<f32>(0.0);
        minProj[p] = 1e10;
        maxProj[p] = -1e10;
    }

    for (var i = 0u; i < texelCount; i = i + 1u) {
        let pix = inputBlock.pixels[i];
        let p = pix.partitionNum;
        if (p < 4u) {
            let diff = pix.data - avg[p];
            let pd = max(diff, vec4<f32>(0.0));
            direction[p] += pd * pd;
        }
    }

    for (var p = 0u; p < 4u; p = p + 1u) {
        direction[p] = normalize(direction[p]);

        //store dir
        outputBlocks[blockIndex].partitions[p].dir = direction[p];
    }


    //find ideal endpoints
    for (var i = 0u; i < texelCount; i = i + 1u) {
        let pix = inputBlock.pixels[i];
        let p = pix.partitionNum;
        if (p < 4u) {
            let proj = dot(pix.data, direction[p]);
            minProj[p] = min(minProj[p], proj);
            maxProj[p] = max(maxProj[p], proj);
        }
    }

    var partition0_lenSqr = 0.0f;
    var lenghtsSqr = vec4<f32>(0.0);

    for (var p = 0u; p < 4u; p = p + 1u) {
        let d = direction[p];

        if(minProj[p] >= maxProj[p]) {
            minProj[p] = 0.0f;
            maxProj[p] = 1e-7f;
        }

        let length = maxProj[p] - minProj[p];
        let lengthSqr = length * length;

        lenghtsSqr[p] = lengthSqr;

        if (all(d == vec4<f32>(0.0))) {
            outputBlocks[blockIndex].partitions[p].endpoint0 = avg[p];
            outputBlocks[blockIndex].partitions[p].endpoint1 = avg[p];
        }
        else {
            outputBlocks[blockIndex].partitions[p].endpoint0 = d * minProj[p];
            outputBlocks[blockIndex].partitions[p].endpoint1 = d * maxProj[p];
        }

        if(p == 0) {
            partition0_lenSqr = lengthSqr;
        }
        else {
            is_constant_wes = is_constant_wes && lengthSqr == partition0_lenSqr;
        }
    }

    //assign weights
    for (var i = 0u; i < texelCount; i = i + 1u) {
        let pix = inputBlock.pixels[i];
        let p = pix.partitionNum;

        if (p < 4u) {
            let proj = dot(pix.data, direction[p]);
            let span = max(maxProj[p] - minProj[p], 1e-6);
            let w = clamp((proj - minProj[p]) / span, 0.0, 1.0);
            outputBlocks[blockIndex].weights[i] = w;

            outputBlocks[blockIndex].weight_error_scale[i] = lenghtsSqr[p] * error_weight;
        }
    }

    outputBlocks[blockIndex].is_constant_weight_error_scale = select(0u, 1u, is_constant_wes);


    //calculate min_endpoint for endpoint quality metric
    var min_ep = vec4<f32>(10.0);

    for (var p = 0u; p < 4u; p = p + 1u) {
        let ep0 = outputBlocks[blockIndex].partitions[p].endpoint0;
        let ep1 = outputBlocks[blockIndex].partitions[p].endpoint1;

        let ep = (vec4<f32>(1.0) - ep0) / (ep1 - ep0 + vec4<f32>(1e-9));  //add small value to prevent division by zero

        let use_ep_mask = (ep > vec4<f32>(0.5)) & (ep < min_ep);

        min_ep = select(min_ep, ep, use_ep_mask);
    }


    outputBlocks[blockIndex].min_weight_cuttof = min(min_ep[0], min(min_ep[1], min(min_ep[2], min_ep[3])));
}