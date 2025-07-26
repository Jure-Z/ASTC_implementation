#include <iostream>

#include "astc.h"
#include "webgpu_utils.h"


std::vector<InputBlock> SplitImageIntoBlocks(
    uint8_t* imageData,
    int width,
    int height,
    int blockWidth,
    int blockHeight
) {

    std::vector<InputBlock> blocks;

    int blocksX = width / blockWidth;
    int blocksY = height / blockHeight;

    for (int by = 0; by < blocksY; ++by) {
        for (int bx = 0; bx < blocksX; ++bx) {
            InputBlock block = {};

            int pixelIndex = 0;
            for (int dy = 0; dy < blockHeight; ++dy) {
                for (int dx = 0; dx < blockWidth; ++dx) {
                    int x = bx * blockWidth + dx;
                    int y = by * blockHeight + dy;
                    int idx = (y * width + x) * 4;

                    block.pixels[pixelIndex].data[0] = (imageData[idx + 0] / 255.0f) * 65536.0f;
                    block.pixels[pixelIndex].data[1] = (imageData[idx + 1] / 255.0f) * 65536.0f;;
                    block.pixels[pixelIndex].data[2] = (imageData[idx + 2] / 255.0f) * 65536.0f;;
                    block.pixels[pixelIndex].data[3] = (imageData[idx + 3] / 255.0f) * 65536.0f;;

                    block.pixels[pixelIndex].partition = 0;

                    ++pixelIndex;
                }
            }

			block.partition_pixel_counts[0] = blockWidth * blockHeight;
            block.partition_pixel_counts[1] = 0;
			block.partition_pixel_counts[2] = 0;
			block.partition_pixel_counts[3] = 0;

            blocks.push_back(block);
        }
    }

    return blocks;
}


ASTCEncoder::ASTCEncoder(wgpu::Device device, uint32_t textureWidth, uint32_t textureHeight, uint8_t blockXDim, uint8_t blockYDim) {

    if (!device) {
        throw std::runtime_error("Invalid WebGPU Device");
    }

    this->device = device;
    this->queue = device.GetQueue();

    this->textureWidth = textureWidth;
    this->textureHeight = textureHeight;
    this->blocksX = textureWidth / blockXDim;
    this->blocksY = textureHeight / blockYDim;
    this->numBlocks = blocksX * blocksY;
    this->blockXDim = blockXDim;
    this->blockYDim = blockYDim;
}

void ASTCEncoder::initMetadata() {
    construct_metadata_structures(blockXDim, blockYDim, block_descriptor);

    construct_angular_tables(sin_table, cos_table);


    //construcrt decimation mode trials
    int max_weight_quant = std::min(static_cast<int>(QUANT_32), quant_limit);

    //store index of decimation mode trial, so we can later link it in block mode trial
    std::vector<int> inverseArray(numBlocks * WEIGHTS_MAX_DECIMATION_MODES, -1);

    int trialCounter = 0;

    for (uint32_t block_idx = 0; block_idx < numBlocks; ++block_idx) {
        for (uint32_t mode_idx = 0; mode_idx < block_descriptor.uniform_variables.decimation_mode_count; ++mode_idx) {
            const decimation_mode& dm = block_descriptor.decimation_modes[mode_idx];

            static_cast<quant_method>(max_weight_quant);

            uint32_t mask = static_cast<uint32_t>((1 << (max_weight_quant + 1)) - 1);
            if ((dm.refprec_1plane & mask) == 0) {
                continue;
            }

            decimation_mode_trials.push_back({ block_idx, mode_idx, 0, 0 });

            inverseArray[block_idx * WEIGHTS_MAX_DECIMATION_MODES + mode_idx] = trialCounter;
            trialCounter++;
        }
    }

    modes_per_block.resize(numBlocks);
	block_mode_trial_offsets.resize(numBlocks);
    uint32_t block_mode_trial_offset = 0;

    for (uint32_t block_idx = 0; block_idx < numBlocks; ++block_idx) {

		modes_per_block[block_idx] = 0;
		block_mode_trial_offsets[block_idx] = block_mode_trial_offset;

        for (uint32_t mode_idx = 0; mode_idx < block_descriptor.uniform_variables.block_mode_count; ++mode_idx) {

            const block_mode& bm = block_descriptor.block_modes[mode_idx];
            
            if (bm.is_dual_plane) {
                continue;
            }

            if (inverseArray[block_idx * WEIGHTS_MAX_DECIMATION_MODES + bm.decimation_mode] < 0) {
                continue;
            }

            block_mode_trials.push_back({ block_idx, mode_idx, uint32_t(inverseArray[block_idx * WEIGHTS_MAX_DECIMATION_MODES + bm.decimation_mode]) });
			modes_per_block[block_idx]++;
			block_mode_trial_offset++;
        }
    }

    block_descriptor.uniform_variables.quant_limit = QUANT_32;
    block_descriptor.uniform_variables.partition_count = 1;
    block_descriptor.uniform_variables.tune_candidate_limit = TUNE_MAX_TRIAL_CANDIDATES;
}

void ASTCEncoder::initBindGroupLayouts() {

    //bind group layout for pass 1
    std::vector<wgpu::BindGroupLayoutEntry> bg1_entries;
    bg1_entries.push_back({ .binding = 0, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Uniform} }); //Uniforms buffer
    bg1_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage}}); //Input block buffer
    bg1_entries.push_back({ .binding = 2, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage} }); //Output of pass1

    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc1 = {};
    bindGroupLayoutDesc1.entryCount = (uint32_t)bg1_entries.size();
    bindGroupLayoutDesc1.entries = bg1_entries.data();
    pass1_bindGroupLayout = device.CreateBindGroupLayout(&bindGroupLayoutDesc1);


    //bind group layout for pass 2
    std::vector<wgpu::BindGroupLayoutEntry> bg2_entries;
    bg2_entries.push_back({ .binding = 0, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Uniform} }); //Uniforms Buffer
    bg2_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Decimation mode trials buffer
    bg2_entries.push_back({ .binding = 2, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Decimation infos buffer
    bg2_entries.push_back({ .binding = 3, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Texel to weight map buffer
    bg2_entries.push_back({ .binding = 4, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Weight to texel map buffer
    bg2_entries.push_back({ .binding = 5, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Output of pass1
    bg2_entries.push_back({ .binding = 6, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage} }); //Output of pass2

    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc2 = {};
    bindGroupLayoutDesc2.entryCount = (uint32_t)bg2_entries.size();
    bindGroupLayoutDesc2.entries = bg2_entries.data();
    pass2_bindGroupLayout = device.CreateBindGroupLayout(&bindGroupLayoutDesc2);

    //bind group layout for pass 3
    std::vector<wgpu::BindGroupLayoutEntry> bg3_entries;
    bg3_entries.push_back({ .binding = 0, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Uniform} }); //Uniforms Buffer
    bg3_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Decimation mode trials buffer
    bg3_entries.push_back({ .binding = 2, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Decimation infos buffer
    bg3_entries.push_back({ .binding = 3, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Sin table buffer
    bg3_entries.push_back({ .binding = 4, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Cos table buffer
    bg3_entries.push_back({ .binding = 5, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Output of pass2 (decimated weights)
    bg3_entries.push_back({ .binding = 6, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage} }); //Output of pass3

    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc3 = {};
    bindGroupLayoutDesc3.entryCount = (uint32_t)bg3_entries.size();
    bindGroupLayoutDesc3.entries = bg3_entries.data();
    pass3_bindGroupLayout = device.CreateBindGroupLayout(&bindGroupLayoutDesc3);


    //bind group layout for pass 4
    std::vector<wgpu::BindGroupLayoutEntry> bg4_entries;
    bg4_entries.push_back({ .binding = 0, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Uniform} }); //Uniforms Buffer
    bg4_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Decimation mode trials buffer
    bg4_entries.push_back({ .binding = 2, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Decimation infos buffer
    bg4_entries.push_back({ .binding = 3, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Output of pass2 (decimated weights)
    bg4_entries.push_back({ .binding = 4, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Output of pass3 (angular offsets)
    bg4_entries.push_back({ .binding = 5, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage} }); //Output of pass4

    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc4 = {};
    bindGroupLayoutDesc4.entryCount = (uint32_t)bg4_entries.size();
    bindGroupLayoutDesc4.entries = bg4_entries.data();
    pass4_bindGroupLayout = device.CreateBindGroupLayout(&bindGroupLayoutDesc4);


    //bind group layout for pass 5
    std::vector<wgpu::BindGroupLayoutEntry> bg5_entries;
    bg5_entries.push_back({ .binding = 0, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Uniform} }); //Uniforms Buffer
    bg5_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Decimation mode trials buffer
    bg5_entries.push_back({ .binding = 2, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Decimation infos buffer
    bg5_entries.push_back({ .binding = 3, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Output of pass3 (angular offsets)
    bg5_entries.push_back({ .binding = 4, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Output of pass4 (lowest and highest weights)
    bg5_entries.push_back({ .binding = 5, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage} }); //Output of pass5 (low values)
    bg5_entries.push_back({ .binding = 6, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage} }); //Output of pass5 (high values)

    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc5 = {};
    bindGroupLayoutDesc5.entryCount = (uint32_t)bg5_entries.size();
    bindGroupLayoutDesc5.entries = bg5_entries.data();
    pass5_bindGroupLayout = device.CreateBindGroupLayout(&bindGroupLayoutDesc5);


    //bind group layout for pass 6
    std::vector<wgpu::BindGroupLayoutEntry> bg6_entries;
    bg6_entries.push_back({ .binding = 0, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Uniform} }); //Uniforms Buffer
    bg6_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Block mode trials buffer
    bg6_entries.push_back({ .binding = 2, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Block modes buffer
    bg6_entries.push_back({ .binding = 3, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Output of pass5 (low values)
    bg6_entries.push_back({ .binding = 4, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Output of pass5 (high values)
    bg6_entries.push_back({ .binding = 5, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage} }); //Output of pass6 (final value ranges)

    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc6 = {};
    bindGroupLayoutDesc6.entryCount = (uint32_t)bg6_entries.size();
    bindGroupLayoutDesc6.entries = bg6_entries.data();
    pass6_bindGroupLayout = device.CreateBindGroupLayout(&bindGroupLayoutDesc6);


    //bind group layout for pass 7
    std::vector<wgpu::BindGroupLayoutEntry> bg7_entries;
    bg7_entries.push_back({ .binding = 0, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Uniform} }); //Uniforms Buffer
    bg7_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Block mode trials buffer
    bg7_entries.push_back({ .binding = 2, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Block modes buffer
    bg7_entries.push_back({ .binding = 3, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Decimation infos buffer
    bg7_entries.push_back({ .binding = 4, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Output of pass2 (decimated weights)
    bg7_entries.push_back({ .binding = 5, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Output of pass6 (final value ranges)
    bg7_entries.push_back({ .binding = 6, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Output of pass1 (ideal endpoints and weights)
    bg7_entries.push_back({ .binding = 7, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Texel to weight map buffer
    bg7_entries.push_back({ .binding = 8, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage} }); //Output of pass7 (quantization results)

    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc7 = {};
    bindGroupLayoutDesc7.entryCount = (uint32_t)bg7_entries.size();
    bindGroupLayoutDesc7.entries = bg7_entries.data();
    pass7_bindGroupLayout = device.CreateBindGroupLayout(&bindGroupLayoutDesc7);

    //bind grup layout for pass 8
	std::vector<wgpu::BindGroupLayoutEntry> bg8_entries;
    bg8_entries.push_back({ .binding = 0, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Uniform} }); //Uniforms Buffer
    bg8_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Input blocks buffer
    bg8_entries.push_back({ .binding = 2, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Output of pass1 (ideal endpoints and weights)
    bg8_entries.push_back({ .binding = 3, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage} }); //Output of pass8 (encoding choice errors)

	wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc8 = {};
	bindGroupLayoutDesc8.entryCount = (uint32_t)bg8_entries.size();
	bindGroupLayoutDesc8.entries = bg8_entries.data();
	pass8_bindGroupLayout = device.CreateBindGroupLayout(&bindGroupLayoutDesc8);

    //bind grup layout for pass 9
    std::vector<wgpu::BindGroupLayoutEntry> bg9_entries;
    bg9_entries.push_back({ .binding = 0, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Uniform} }); //Uniforms Buffer
    bg9_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Input blocks buffer
    bg9_entries.push_back({ .binding = 2, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Output of pass1 (ideal endpoints and weights)
	bg9_entries.push_back({ .binding = 3, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Output of pass8 (encoding choice errors)
    bg9_entries.push_back({ .binding = 4, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage} }); //Output of pass9 (color format errors)
    bg9_entries.push_back({ .binding = 5, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage} }); //Output of pass9 (color formats)

    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc9 = {};
    bindGroupLayoutDesc9.entryCount = (uint32_t)bg9_entries.size();
    bindGroupLayoutDesc9.entries = bg9_entries.data();
    pass9_bindGroupLayout = device.CreateBindGroupLayout(&bindGroupLayoutDesc9);

    //bind grup layout for pass 10
    std::vector<wgpu::BindGroupLayoutEntry> bg10_entries;
    bg10_entries.push_back({ .binding = 0, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Uniform} }); //Uniforms Buffer
    bg10_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Output of pass9 (color format errors)
    bg10_entries.push_back({ .binding = 2, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Output of pass9 (color formats)
    bg10_entries.push_back({ .binding = 3, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage} }); //Output of pass10 (color format combinations)

    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc10 = {};
    bindGroupLayoutDesc10.entryCount = (uint32_t)bg10_entries.size();
    bindGroupLayoutDesc10.entries = bg10_entries.data();
    pass10_bindGroupLayout = device.CreateBindGroupLayout(&bindGroupLayoutDesc10);

    //bind grup layout for pass 11, 1 partition
    std::vector<wgpu::BindGroupLayoutEntry> bg11_1_entries;
    bg11_1_entries.push_back({ .binding = 0, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Uniform} }); //Uniforms Buffer
	bg11_1_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Block mode trials buffer
    bg11_1_entries.push_back({ .binding = 2, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Output of pass7 (quantization results)
    bg11_1_entries.push_back({ .binding = 3, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Output of pass9 (color format errors)
    bg11_1_entries.push_back({ .binding = 4, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Output of pass9 (color formats)
    bg11_1_entries.push_back({ .binding = 5, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage} }); //Output of pass11 (best endpoint combiantions for mode)

    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc11_1 = {};
    bindGroupLayoutDesc11_1.entryCount = (uint32_t)bg11_1_entries.size();
    bindGroupLayoutDesc11_1.entries = bg11_1_entries.data();
    pass11_bindGroupLayout_1part = device.CreateBindGroupLayout(&bindGroupLayoutDesc11_1);

	//bind grup layout for pass 11, 2,3,4 partitions
    std::vector<wgpu::BindGroupLayoutEntry> bg11_234_entries;
    bg11_234_entries.push_back({ .binding = 0, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Uniform} }); //Uniforms Buffer
    bg11_234_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Block mode trials buffer
    bg11_234_entries.push_back({ .binding = 2, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Output of pass7 (quantization results)
    bg11_234_entries.push_back({ .binding = 3, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Output of pass10 (color format combinations)
    bg11_234_entries.push_back({ .binding = 4, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage} }); //Output of pass11 (best endpoint combiantions for mode)

    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc11_234 = {};
    bindGroupLayoutDesc11_234.entryCount = (uint32_t)bg11_234_entries.size();
    bindGroupLayoutDesc11_234.entries = bg11_234_entries.data();
    pass11_bindGroupLayout_234part = device.CreateBindGroupLayout(&bindGroupLayoutDesc11_234);

    //bind grup layout for pass 12
    std::vector<wgpu::BindGroupLayoutEntry> bg12_entries;
    bg12_entries.push_back({ .binding = 0, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Uniform} }); //Uniforms Buffer
    bg12_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Block mode trials buffer
    bg12_entries.push_back({ .binding = 2, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Modes per block buffer
	bg12_entries.push_back({ .binding = 3, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Block mode trial offsets buffer
    bg12_entries.push_back({ .binding = 4, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Output of pass11 (best endpoint combiantions for mode)
    bg12_entries.push_back({ .binding = 5, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage} }); //Output of pass12 (final candidates)

    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc12 = {};
    bindGroupLayoutDesc12.entryCount = (uint32_t)bg12_entries.size();
    bindGroupLayoutDesc12.entries = bg12_entries.data();
    pass12_bindGroupLayout = device.CreateBindGroupLayout(&bindGroupLayoutDesc12);
}

void ASTCEncoder::initPipelines() {

    // Load shader modules
#if defined(__EMSCRIPTEN__)
    pass1_idealEndpointsShader = prepareShaderModule(device, "shaders/pass1_ideal_endpoints_and_weights.wgsl", "Ideal endpoints and weights (pass1)");
    pass2_decimatedWeightsShader = prepareShaderModule(device, "shaders/pass2_decimated_weights.wgsl", "decimated weights (pass2)");
    pass3_angularOffsetsShader = prepareShaderModule(device, "shaders/pass3_compute_angular_offsets.wgsl", "angular offsets (pass3)");
    pass4_lowestAndHighestWeightShader = prepareShaderModule(device, "shaders/pass4_lowest_and_highest_weight.wgsl", "lowest and highest weight (pass4)");
    pass5_valuesForQuantLevelsShader = prepareShaderModule(device, "shaders/pass5_best_values_for_quant_levels.wgsl", "best values for quant levels (pass5)");
    pass6_remapLowAndHighValuesShader = prepareShaderModule(device, "shaders/pass6_remap_low_and_high_values.wgsl", "remap low and high values (pass6)");
    pass7_weightsAndErrorForBMShader = prepareShaderModule(device, "shaders/pass7_weights_and_error_for_bm.wgsl", "weights and error for block mode (pass7)");
	pass8_encodingChoiceErrorsShader = prepareShaderModule(device, "shaders/pass8_compute_encoding_choice_errors.wgsl", "encoding choice errors (pass8)");
	pass9_computeColorErrorShader = prepareShaderModule(device, "shaders/pass9_compute_color_error.wgsl", "color format errors (pass9)");
	pass10_colorEndpointCombinationsShader_2part = prepareShaderModule(device, "shaders/pass10_color_combinations_for_quant_2part.wgsl", "color endpoint combinations (pass10, 2part)");
	pass10_colorEndpointCombinationsShader_3part = prepareShaderModule(device, "shaders/pass10_color_combinations_for_quant_3part.wgsl", "color endpoint combinations (pass10, 3part)");
	pass10_colorEndpointCombinationsShader_4part = prepareShaderModule(device, "shaders/pass10_color_combinations_for_quant_4part.wgsl", "color endpoint combinations (pass10, 4part)");
	pass11_bestEndpointCombinationsForModeShader_1part = prepareShaderModule(device, "shaders/pass11_best_color_combination_for_mode_1part.wgsl", "best endpoint combinations for mode (pass11, 1part)");
	pass11_bestEndpointCombinationsForModeShader_2part = prepareShaderModule(device, "shaders/pass11_best_color_combination_for_mode_2part.wgsl", "best endpoint combinations for mode (pass11, 2part)");
	pass11_bestEndpointCombinationsForModeShader_3part = prepareShaderModule(device, "shaders/pass11_best_color_combination_for_mode_3part.wgsl", "best endpoint combinations for mode (pass11, 3part)");
	pass11_bestEndpointCombinationsForModeShader_4part = prepareShaderModule(device, "shaders/pass11_best_color_combination_for_mode_4part.wgsl", "best endpoint combinations for mode (pass11, 4part)");
	pass12_findTopNCandidatesShader = prepareShaderModule(device, "shaders/pass12_find_top_N_candidates.wgsl", "find top N candidates (pass12)");
#else
    pass1_idealEndpointsShader = prepareShaderModule(device, SHADER_DIR "/pass1_ideal_endpoints_and_weights.wgsl", "Ideal endpoints and weights (pass1)");
    pass2_decimatedWeightsShader = prepareShaderModule(device, SHADER_DIR "/pass2_decimated_weights.wgsl", "decimated weights (pass2)");
    pass3_angularOffsetsShader = prepareShaderModule(device, SHADER_DIR "/pass3_compute_angular_offsets.wgsl", "angular offsets (pass3)");
    pass4_lowestAndHighestWeightShader = prepareShaderModule(device, SHADER_DIR "/pass4_lowest_and_highest_weight.wgsl", "lowest and highest weight (pass4)");
    pass5_valuesForQuantLevelsShader = prepareShaderModule(device, SHADER_DIR "/pass5_best_values_for_quant_levels.wgsl", "best values for quant levels (pass5)");
    pass6_remapLowAndHighValuesShader = prepareShaderModule(device, SHADER_DIR "/pass6_remap_low_and_high_values.wgsl", "remap low and high values (pass6)");
    pass7_weightsAndErrorForBMShader = prepareShaderModule(device, SHADER_DIR "/pass7_weights_and_error_for_bm.wgsl", "weights and error for block mode (pass7)");
	pass8_encodingChoiceErrorsShader = prepareShaderModule(device, SHADER_DIR "/pass8_compute_encoding_choice_errors.wgsl", "encoding choice errors (pass8)");
	pass9_computeColorErrorShader = prepareShaderModule(device, SHADER_DIR "/pass9_compute_color_error.wgsl", "color format errors (pass9)");
	pass10_colorEndpointCombinationsShader_2part = prepareShaderModule(device, SHADER_DIR "/pass10_color_combinations_for_quant_2part.wgsl", "color endpoint combinations (pass10, 2part)");
    pass10_colorEndpointCombinationsShader_3part = prepareShaderModule(device, SHADER_DIR "/pass10_color_combinations_for_quant_3part.wgsl", "color endpoint combinations (pass10, 3part)");
    pass10_colorEndpointCombinationsShader_4part = prepareShaderModule(device, SHADER_DIR "/pass10_color_combinations_for_quant_4part.wgsl", "color endpoint combinations (pass10, 4part)");
	pass11_bestEndpointCombinationsForModeShader_1part = prepareShaderModule(device, SHADER_DIR "/pass11_best_color_combination_for_mode_1part.wgsl", "best endpoint combinations for mode (pass11, 1part)");
    pass11_bestEndpointCombinationsForModeShader_2part = prepareShaderModule(device, SHADER_DIR "/pass11_best_color_combination_for_mode_2part.wgsl", "best endpoint combinations for mode (pass11, 2part)");
    pass11_bestEndpointCombinationsForModeShader_3part = prepareShaderModule(device, SHADER_DIR "/pass11_best_color_combination_for_mode_3part.wgsl", "best endpoint combinations for mode (pass11, 3part)");
    pass11_bestEndpointCombinationsForModeShader_4part = prepareShaderModule(device, SHADER_DIR "/pass11_best_color_combination_for_mode_4part.wgsl", "best endpoint combinations for mode (pass11, 4part)");
	pass12_findTopNCandidatesShader = prepareShaderModule(device, SHADER_DIR "/pass12_find_top_N_candidates.wgsl", "find top N candidates (pass12)");
#endif

    //pass1 compute pipeline
    wgpu::PipelineLayoutDescriptor pass1_layoutDesc = {};
    pass1_layoutDesc.bindGroupLayoutCount = 1;
    pass1_layoutDesc.bindGroupLayouts = &pass1_bindGroupLayout;
    wgpu::PipelineLayout pass1_pipelineLayout = device.CreatePipelineLayout(&pass1_layoutDesc);

    wgpu::ComputePipelineDescriptor pass1_pipelineDesc = {};
    pass1_pipelineDesc.compute.constantCount = 0;
    pass1_pipelineDesc.compute.constants = nullptr;
    pass1_pipelineDesc.compute.entryPoint = "main";
    pass1_pipelineDesc.compute.module = pass1_idealEndpointsShader;
    pass1_pipelineDesc.layout = pass1_pipelineLayout;

    pass1_pipeline = device.CreateComputePipeline(&pass1_pipelineDesc);


    //pass2 compute pipeline
    wgpu::PipelineLayoutDescriptor pass2_layoutDesc = {};
    pass2_layoutDesc.bindGroupLayoutCount = 1;
    pass2_layoutDesc.bindGroupLayouts = &pass2_bindGroupLayout;
    wgpu::PipelineLayout pass2_pipelineLayout = device.CreatePipelineLayout(&pass2_layoutDesc);

    wgpu::ComputePipelineDescriptor pass2_pipelineDesc = {};
    pass2_pipelineDesc.compute.constantCount = 0;
    pass2_pipelineDesc.compute.constants = nullptr;
    pass2_pipelineDesc.compute.entryPoint = "main";
    pass2_pipelineDesc.compute.module = pass2_decimatedWeightsShader;
    pass2_pipelineDesc.layout = pass2_pipelineLayout;

    pass2_pipeline = device.CreateComputePipeline(&pass2_pipelineDesc);


    //pass3 compute pipeline
    wgpu::PipelineLayoutDescriptor pass3_layoutDesc = {};
    pass3_layoutDesc.bindGroupLayoutCount = 1;
    pass3_layoutDesc.bindGroupLayouts = &pass3_bindGroupLayout;
    wgpu::PipelineLayout pass3_pipelineLayout = device.CreatePipelineLayout(&pass3_layoutDesc);

    wgpu::ComputePipelineDescriptor pass3_pipelineDesc = {};
    pass3_pipelineDesc.compute.constantCount = 0;
    pass3_pipelineDesc.compute.constants = nullptr;
    pass3_pipelineDesc.compute.entryPoint = "main";
    pass3_pipelineDesc.compute.module = pass3_angularOffsetsShader;
    pass3_pipelineDesc.layout = pass3_pipelineLayout;

    pass3_pipeline = device.CreateComputePipeline(&pass3_pipelineDesc);


    //pass4 compute pipeline
    wgpu::PipelineLayoutDescriptor pass4_layoutDesc = {};
    pass4_layoutDesc.bindGroupLayoutCount = 1;
    pass4_layoutDesc.bindGroupLayouts = &pass4_bindGroupLayout;
    wgpu::PipelineLayout pass4_pipelineLayout = device.CreatePipelineLayout(&pass4_layoutDesc);

    wgpu::ComputePipelineDescriptor pass4_pipelineDesc = {};
    pass4_pipelineDesc.compute.constantCount = 0;
    pass4_pipelineDesc.compute.constants = nullptr;
    pass4_pipelineDesc.compute.entryPoint = "main";
    pass4_pipelineDesc.compute.module = pass4_lowestAndHighestWeightShader;
    pass4_pipelineDesc.layout = pass4_pipelineLayout;

    pass4_pipeline = device.CreateComputePipeline(&pass4_pipelineDesc);


    //pass5 compute pipeline
    wgpu::PipelineLayoutDescriptor pass5_layoutDesc = {};
    pass5_layoutDesc.bindGroupLayoutCount = 1;
    pass5_layoutDesc.bindGroupLayouts = &pass5_bindGroupLayout;
    wgpu::PipelineLayout pass5_pipelineLayout = device.CreatePipelineLayout(&pass5_layoutDesc);

    wgpu::ComputePipelineDescriptor pass5_pipelineDesc = {};
    pass5_pipelineDesc.compute.constantCount = 0;
    pass5_pipelineDesc.compute.constants = nullptr;
    pass5_pipelineDesc.compute.entryPoint = "main";
    pass5_pipelineDesc.compute.module = pass5_valuesForQuantLevelsShader;
    pass5_pipelineDesc.layout = pass5_pipelineLayout;

    pass5_pipeline = device.CreateComputePipeline(&pass5_pipelineDesc);


    //pass6 compute pipeline
    wgpu::PipelineLayoutDescriptor pass6_layoutDesc = {};
    pass6_layoutDesc.bindGroupLayoutCount = 1;
    pass6_layoutDesc.bindGroupLayouts = &pass6_bindGroupLayout;
    wgpu::PipelineLayout pass6_pipelineLayout = device.CreatePipelineLayout(&pass6_layoutDesc);

    wgpu::ComputePipelineDescriptor pass6_pipelineDesc = {};
    pass6_pipelineDesc.compute.constantCount = 0;
    pass6_pipelineDesc.compute.constants = nullptr;
    pass6_pipelineDesc.compute.entryPoint = "main";
    pass6_pipelineDesc.compute.module = pass6_remapLowAndHighValuesShader;
    pass6_pipelineDesc.layout = pass6_pipelineLayout;

    pass6_pipeline = device.CreateComputePipeline(&pass6_pipelineDesc);


    //pass7 compute pipeline
    wgpu::PipelineLayoutDescriptor pass7_layoutDesc = {};
    pass7_layoutDesc.bindGroupLayoutCount = 1;
    pass7_layoutDesc.bindGroupLayouts = &pass7_bindGroupLayout;
    wgpu::PipelineLayout pass7_pipelineLayout = device.CreatePipelineLayout(&pass7_layoutDesc);

    wgpu::ComputePipelineDescriptor pass7_pipelineDesc = {};
    pass7_pipelineDesc.compute.constantCount = 0;
    pass7_pipelineDesc.compute.constants = nullptr;
    pass7_pipelineDesc.compute.entryPoint = "main";
    pass7_pipelineDesc.compute.module = pass7_weightsAndErrorForBMShader;
    pass7_pipelineDesc.layout = pass7_pipelineLayout;

    pass7_pipeline = device.CreateComputePipeline(&pass7_pipelineDesc);

	//pass8 compute pipeline
	wgpu::PipelineLayoutDescriptor pass8_layoutDesc = {};
	pass8_layoutDesc.bindGroupLayoutCount = 1;
	pass8_layoutDesc.bindGroupLayouts = &pass8_bindGroupLayout;
	wgpu::PipelineLayout pass8_pipelineLayout = device.CreatePipelineLayout(&pass8_layoutDesc);

	wgpu::ComputePipelineDescriptor pass8_pipelineDesc = {};
	pass8_pipelineDesc.compute.constantCount = 0;
	pass8_pipelineDesc.compute.constants = nullptr;
	pass8_pipelineDesc.compute.entryPoint = "main";
	pass8_pipelineDesc.compute.module = pass8_encodingChoiceErrorsShader;
	pass8_pipelineDesc.layout = pass8_pipelineLayout;

	pass8_pipeline = device.CreateComputePipeline(&pass8_pipelineDesc);

	//pass9 compute pipeline
    wgpu::PipelineLayoutDescriptor pass9_layoutDesc = {};
    pass9_layoutDesc.bindGroupLayoutCount = 1;
    pass9_layoutDesc.bindGroupLayouts = &pass9_bindGroupLayout;
    wgpu::PipelineLayout pass9_pipelineLayout = device.CreatePipelineLayout(&pass9_layoutDesc);

    wgpu::ComputePipelineDescriptor pass9_pipelineDesc = {};
    pass9_pipelineDesc.compute.constantCount = 0;
    pass9_pipelineDesc.compute.constants = nullptr;
    pass9_pipelineDesc.compute.entryPoint = "main";
    pass9_pipelineDesc.compute.module = pass9_computeColorErrorShader;
    pass9_pipelineDesc.layout = pass9_pipelineLayout;

    pass9_pipeline = device.CreateComputePipeline(&pass9_pipelineDesc);

	//pass10 2partitions compute pipeline
    wgpu::PipelineLayoutDescriptor pass10_2_layoutDesc = {};
    pass10_2_layoutDesc.bindGroupLayoutCount = 1;
    pass10_2_layoutDesc.bindGroupLayouts = &pass10_bindGroupLayout;
    wgpu::PipelineLayout pass10_2_pipelineLayout = device.CreatePipelineLayout(&pass10_2_layoutDesc);

    wgpu::ComputePipelineDescriptor pass10_2_pipelineDesc = {};
    pass10_2_pipelineDesc.compute.constantCount = 0;
    pass10_2_pipelineDesc.compute.constants = nullptr;
    pass10_2_pipelineDesc.compute.entryPoint = "main";
    pass10_2_pipelineDesc.compute.module = pass10_colorEndpointCombinationsShader_2part;
    pass10_2_pipelineDesc.layout = pass10_2_pipelineLayout;

    pass10_pipeline_2part = device.CreateComputePipeline(&pass10_2_pipelineDesc);

	//pass10 3partitions compute pipeline
	wgpu::PipelineLayoutDescriptor pass10_3_layoutDesc = {};
	pass10_3_layoutDesc.bindGroupLayoutCount = 1;
	pass10_3_layoutDesc.bindGroupLayouts = &pass10_bindGroupLayout;
	wgpu::PipelineLayout pass10_3_pipelineLayout = device.CreatePipelineLayout(&pass10_3_layoutDesc);

	wgpu::ComputePipelineDescriptor pass10_3_pipelineDesc = {};
	pass10_3_pipelineDesc.compute.constantCount = 0;
	pass10_3_pipelineDesc.compute.constants = nullptr;
	pass10_3_pipelineDesc.compute.entryPoint = "main";
	pass10_3_pipelineDesc.compute.module = pass10_colorEndpointCombinationsShader_3part;
	pass10_3_pipelineDesc.layout = pass10_3_pipelineLayout;

	pass10_pipeline_3part = device.CreateComputePipeline(&pass10_3_pipelineDesc);

	//pass10 4partitions compute pipeline
	wgpu::PipelineLayoutDescriptor pass10_4_layoutDesc = {};
	pass10_4_layoutDesc.bindGroupLayoutCount = 1;
	pass10_4_layoutDesc.bindGroupLayouts = &pass10_bindGroupLayout;
	wgpu::PipelineLayout pass10_4_pipelineLayout = device.CreatePipelineLayout(&pass10_4_layoutDesc);

	wgpu::ComputePipelineDescriptor pass10_4_pipelineDesc = {};
	pass10_4_pipelineDesc.compute.constantCount = 0;
	pass10_4_pipelineDesc.compute.constants = nullptr;
	pass10_4_pipelineDesc.compute.entryPoint = "main";
	pass10_4_pipelineDesc.compute.module = pass10_colorEndpointCombinationsShader_4part;
	pass10_4_pipelineDesc.layout = pass10_4_pipelineLayout;

	pass10_pipeline_4part = device.CreateComputePipeline(&pass10_4_pipelineDesc);

    //pass11 1partition compute pipeline
    wgpu::PipelineLayoutDescriptor pass11_1_layoutDesc = {};
    pass11_1_layoutDesc.bindGroupLayoutCount = 1;
    pass11_1_layoutDesc.bindGroupLayouts = &pass11_bindGroupLayout_1part;
    wgpu::PipelineLayout pass11_1_pipelineLayout = device.CreatePipelineLayout(&pass11_1_layoutDesc);

    wgpu::ComputePipelineDescriptor pass11_1_pipelineDesc = {};
    pass11_1_pipelineDesc.compute.constantCount = 0;
    pass11_1_pipelineDesc.compute.constants = nullptr;
    pass11_1_pipelineDesc.compute.entryPoint = "main";
    pass11_1_pipelineDesc.compute.module = pass11_bestEndpointCombinationsForModeShader_1part;
    pass11_1_pipelineDesc.layout = pass11_1_pipelineLayout;

    pass11_pipeline_1part = device.CreateComputePipeline(&pass11_1_pipelineDesc);

    //pass11 2partitions compute pipeline
    wgpu::PipelineLayoutDescriptor pass11_2_layoutDesc = {};
    pass11_2_layoutDesc.bindGroupLayoutCount = 1;
    pass11_2_layoutDesc.bindGroupLayouts = &pass11_bindGroupLayout_234part;
    wgpu::PipelineLayout pass11_2_pipelineLayout = device.CreatePipelineLayout(&pass11_2_layoutDesc);

    wgpu::ComputePipelineDescriptor pass11_2_pipelineDesc = {};
    pass11_2_pipelineDesc.compute.constantCount = 0;
    pass11_2_pipelineDesc.compute.constants = nullptr;
    pass11_2_pipelineDesc.compute.entryPoint = "main";
    pass11_2_pipelineDesc.compute.module = pass11_bestEndpointCombinationsForModeShader_2part;
    pass11_2_pipelineDesc.layout = pass11_2_pipelineLayout;

    pass11_pipeline_2part = device.CreateComputePipeline(&pass11_2_pipelineDesc);

	//pass11 3partitions compute pipeline
	wgpu::PipelineLayoutDescriptor pass11_3_layoutDesc = {};
	pass11_3_layoutDesc.bindGroupLayoutCount = 1;
	pass11_3_layoutDesc.bindGroupLayouts = &pass11_bindGroupLayout_234part;
	wgpu::PipelineLayout pass11_3_pipelineLayout = device.CreatePipelineLayout(&pass11_3_layoutDesc);

	wgpu::ComputePipelineDescriptor pass11_3_pipelineDesc = {};
	pass11_3_pipelineDesc.compute.constantCount = 0;
	pass11_3_pipelineDesc.compute.constants = nullptr;
	pass11_3_pipelineDesc.compute.entryPoint = "main";
	pass11_3_pipelineDesc.compute.module = pass11_bestEndpointCombinationsForModeShader_3part;
	pass11_3_pipelineDesc.layout = pass11_3_pipelineLayout;

	pass11_pipeline_3part = device.CreateComputePipeline(&pass11_3_pipelineDesc);

	//pass11 4partitions compute pipeline
	wgpu::PipelineLayoutDescriptor pass11_4_layoutDesc = {};
	pass11_4_layoutDesc.bindGroupLayoutCount = 1;
	pass11_4_layoutDesc.bindGroupLayouts = &pass11_bindGroupLayout_234part;
	wgpu::PipelineLayout pass11_4_pipelineLayout = device.CreatePipelineLayout(&pass11_4_layoutDesc);

	wgpu::ComputePipelineDescriptor pass11_4_pipelineDesc = {};
	pass11_4_pipelineDesc.compute.constantCount = 0;
	pass11_4_pipelineDesc.compute.constants = nullptr;
	pass11_4_pipelineDesc.compute.entryPoint = "main";
	pass11_4_pipelineDesc.compute.module = pass11_bestEndpointCombinationsForModeShader_4part;
	pass11_4_pipelineDesc.layout = pass11_4_pipelineLayout;

	pass11_pipeline_4part = device.CreateComputePipeline(&pass11_4_pipelineDesc);

    //pass12 compute pipeline
    wgpu::PipelineLayoutDescriptor pass12_layoutDesc = {};
    pass12_layoutDesc.bindGroupLayoutCount = 1;
    pass12_layoutDesc.bindGroupLayouts = &pass12_bindGroupLayout;
    wgpu::PipelineLayout pass12_pipelineLayout = device.CreatePipelineLayout(&pass12_layoutDesc);

    wgpu::ComputePipelineDescriptor pass12_pipelineDesc = {};
    pass12_pipelineDesc.compute.constantCount = 0;
    pass12_pipelineDesc.compute.constants = nullptr;
    pass12_pipelineDesc.compute.entryPoint = "main";
    pass12_pipelineDesc.compute.module = pass12_findTopNCandidatesShader;
    pass12_pipelineDesc.layout = pass12_pipelineLayout;

    pass12_pipeline = device.CreateComputePipeline(&pass12_pipelineDesc);


}

void ASTCEncoder::initBuffers() {

    //Buffer for uniform variables
    wgpu::BufferDescriptor uniformDesc;
    uniformDesc.size = sizeof(uniform_variables);
    uniformDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    uniformsBuffer = device.CreateBuffer(&uniformDesc);

    //Buffer for decimation mode trials
    wgpu::BufferDescriptor decimationModeTrialsDesc;
    decimationModeTrialsDesc.size = decimation_mode_trials.size() * sizeof(DecimationModeTrial);
    decimationModeTrialsDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    decimationModeTrialsBuffer = device.CreateBuffer(&decimationModeTrialsDesc);

    //Buffer for block mode trials
    wgpu::BufferDescriptor blockModeTrialsDesc;
    blockModeTrialsDesc.size = block_mode_trials.size() * sizeof(BlockModeTrial);
    blockModeTrialsDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    blockModeTrialsBuffer = device.CreateBuffer(&blockModeTrialsDesc);

    //Buffer for number of block modes considered for each block
	wgpu::BufferDescriptor modesPerBlockDesc;
	modesPerBlockDesc.size = numBlocks * sizeof(uint32_t);
	modesPerBlockDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
	modesPerBlockBuffer = device.CreateBuffer(&modesPerBlockDesc);

	//Buffer for block mode trial offsets (for each block)
	wgpu::BufferDescriptor blockModeTrialOffsetsDesc;
	blockModeTrialOffsetsDesc.size = numBlocks * sizeof(uint32_t);
	blockModeTrialOffsetsDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
	blockModeTrialOffsetsBuffer = device.CreateBuffer(&blockModeTrialOffsetsDesc);

    //Buffer for block modes
    wgpu::BufferDescriptor blockModesDesc;
    blockModesDesc.size = block_descriptor.uniform_variables.block_mode_count * sizeof(block_mode);
    blockModesDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    blockModesBuffer = device.CreateBuffer(&blockModesDesc);

    //Buffer for block mode index
    wgpu::BufferDescriptor blockModeIndexDesc;
    blockModeIndexDesc.size = block_descriptor.uniform_variables.block_mode_count * sizeof(uint32_t);
    blockModeIndexDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    blockModeIndexBuffer = device.CreateBuffer(&blockModeIndexDesc);

    //Buffer for decimation modes
    wgpu::BufferDescriptor decimationModesDesc;
    decimationModesDesc.size = block_descriptor.uniform_variables.decimation_mode_count * sizeof(decimation_mode);
    decimationModesDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    decimationModesBuffer = device.CreateBuffer(&decimationModesDesc);

    //Buffer for decimation infos
    wgpu::BufferDescriptor decimatioInfoDesc;
    decimatioInfoDesc.size = block_descriptor.uniform_variables.decimation_mode_count * sizeof(decimation_info);
    decimatioInfoDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    decimationInfoBuffer = device.CreateBuffer(&decimatioInfoDesc);

    //Buffer stores info for reconstructing original weights from decimated weights  decimated weights -> reconstructed weights (for every decimation mode)
    wgpu::BufferDescriptor texelToWeightMapDesc;
    texelToWeightMapDesc.size = block_descriptor.decimation_info_packed.texel_to_weight_map_data.size() * sizeof(TexelToWeightMap);
    texelToWeightMapDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    texelToWeightMapBuffer = device.CreateBuffer(&texelToWeightMapDesc);

    //Buffer stores info for decimation of texel weights  ideal weights -> decimated weights  (for every decimation mode)
    wgpu::BufferDescriptor weightToTexelMapDesc;
    weightToTexelMapDesc.size = block_descriptor.decimation_info_packed.weight_to_texel_map_data.size() * sizeof(WeightToTexelMap);
    weightToTexelMapDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    weightToTexelMapBuffer = device.CreateBuffer(&weightToTexelMapDesc);

    //Buffer for sin table
    wgpu::BufferDescriptor sinTableDesc;
    sinTableDesc.size = SINCOS_STEPS * ANGULAR_STEPS * sizeof(float);
    sinTableDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    sinBuffer = device.CreateBuffer(&sinTableDesc);

    //Buffer for cos table
    wgpu::BufferDescriptor cosTableDesc;
    cosTableDesc.size = SINCOS_STEPS * ANGULAR_STEPS * sizeof(float);
    cosTableDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    cosBuffer = device.CreateBuffer(&cosTableDesc);

    //Buffer for input blocks (read-only in shaders)
    wgpu::BufferDescriptor inputDesc;
    inputDesc.size = numBlocks * sizeof(InputBlock);
    inputDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    inputBlocksBuffer = device.CreateBuffer(&inputDesc);

    //Output buffer of pass 1 (ideal endpoints and weights)
    wgpu::BufferDescriptor pass1Desc = {};
    pass1Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    pass1Desc.size = numBlocks * sizeof(IdealEndpointsAndWeights);
    pass1_output_idealEndpointsAndWeights = device.CreateBuffer(&pass1Desc);

    //Output buffer of pass 2 (decimated weights)
    //indexing pattern: decimation_mode_trial_index * BLOCK_MAX_WEIGHTS + weight_index
    wgpu::BufferDescriptor pass2Desc = {};
    pass2Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    //pass2Desc.size = numBlocks * block_descriptor.uniform_variables.decimation_mode_count * BLOCK_MAX_WEIGHTS * sizeof(float);
    pass2Desc.size = decimation_mode_trials.size() * BLOCK_MAX_WEIGHTS * sizeof(float);
    pass2_output_decimatedWeights = device.CreateBuffer(&pass2Desc);

    //Output buffer of pass 3 (angular offsets)
    //indexing pattern: decimation_mode_trial_index * ANGULAR_STEPS + angular_offset_index
    wgpu::BufferDescriptor pass3Desc = {};
    pass3Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    //pass3Desc.size = numBlocks * block_descriptor.uniform_variables.decimation_mode_count * ANGULAR_STEPS * sizeof(float);
    pass3Desc.size = decimation_mode_trials.size() * ANGULAR_STEPS * sizeof(float);
    pass3_output_angular_offsets = device.CreateBuffer(&pass3Desc);

    //Output buffer of pass 4 (lowest and highest weights)
    wgpu::BufferDescriptor pass4Desc = {};
    pass4Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    pass4Desc.size = decimation_mode_trials.size() * ANGULAR_STEPS * sizeof(HighestAndLowestWeight);
    pass4_output_lowestAndHighestWeight = device.CreateBuffer(&pass4Desc);

    //Output buffer of pass 5 (low values)
    wgpu::BufferDescriptor pass5_1Desc = {};
    pass5_1Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    pass5_1Desc.size = decimation_mode_trials.size() * (MAX_ANGULAR_QUANT + 1) * sizeof(float);
    pass5_output_lowValues = device.CreateBuffer(&pass5_1Desc);

    //Output buffer of pass 5 (high values)
    wgpu::BufferDescriptor pass5_2Desc = {};
    pass5_2Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    pass5_2Desc.size = decimation_mode_trials.size() * (MAX_ANGULAR_QUANT + 1) * sizeof(float);
    pass5_output_highValues = device.CreateBuffer(&pass5_2Desc);

    //Output buffer of pass 6 (final value ranges)
    wgpu::BufferDescriptor pass6Desc = {};
    pass6Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    pass6Desc.size = block_mode_trials.size() * sizeof(FinalValueRange);
    pass6_output_finalValueRanges = device.CreateBuffer(&pass6Desc);

    //Output buffer of pass 7 (quantization results)
    wgpu::BufferDescriptor pass7Desc = {};
    pass7Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    pass7Desc.size = block_mode_trials.size() * sizeof(QuantizationResult);
    pass7_output_quantizationResults = device.CreateBuffer(&pass7Desc);

	//Output buffer of pass 8 (encoding choice errors)
	wgpu::BufferDescriptor pass8Desc = {};
	pass8Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
	pass8Desc.size = numBlocks * sizeof(EncodingChoiceErrors);
	pass8_output_encodingChoiceErrors = device.CreateBuffer(&pass8Desc);

	//Output buffer of pass 9 (color format errors)
	//indexing pattern: ((block_index * BLOCK_MAX_PARTITIONS + partition_index) * QUANT_LEVELS + quant_level_index) * NUM_INT_COUNTS + integer_count
	wgpu::BufferDescriptor pass9_1Desc = {};
	pass9_1Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
	pass9_1Desc.size = numBlocks * BLOCK_MAX_PARTITIONS * QUANT_LEVELS * NUM_INT_COUNTS * sizeof(float);
	pass9_output_colorFormatErrors = device.CreateBuffer(&pass9_1Desc);

    //Output buffer of pass 9 (color formats)
    //indexing pattern: ((block_index * BLOCK_MAX_PARTITIONS + partition_index) * QUANT_LEVELS + quant_level_index) * NUM_INT_COUNTS + integer_count
    wgpu::BufferDescriptor pass9_2Desc = {};
    pass9_2Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    pass9_2Desc.size = numBlocks * BLOCK_MAX_PARTITIONS * QUANT_LEVELS * NUM_INT_COUNTS * sizeof(uint32_t);
    pass9_output_colorFormats = device.CreateBuffer(&pass9_2Desc);

	//Output buffer of pass 10 (color format combinations)
	//indexing pattern: (block_index * QUANT_LEVELS + quant_level) * MAX_INT_COUNT_COMBINATIONS + integer_count
    wgpu::BufferDescriptor pass10Desc = {};
    pass10Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
	pass10Desc.size = numBlocks * QUANT_LEVELS * MAX_INT_COUNT_COMBINATIONS * sizeof(CombinedEndpointFormats);
    pass10_output_colorEndpointCombinations = device.CreateBuffer(&pass10Desc);

	//Output buffer of pass 11 (best endpoint combinations for mode)
    wgpu::BufferDescriptor pass11Desc = {};
    pass11Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
	pass11Desc.size = block_mode_trials.size() * sizeof(ColorCombinationResult);
    pass11_output_bestEndpointCombinationsForMode = device.CreateBuffer(&pass11Desc);

	//Output buffer of pass 12 (top N candidates)
	//indexing pattern: (block_index * block_descriptor.uniform_variables.tune_candidate_limit + i-th best candidate)
	wgpu::BufferDescriptor pass12Desc = {};
	pass12Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
	pass12Desc.size = numBlocks * block_descriptor.uniform_variables.tune_candidate_limit * sizeof(FinalCandidate);
	pass12_output_finalCandidates = device.CreateBuffer(&pass12Desc);

    //Readback buffer for debuging (pass1)
    wgpu::BufferDescriptor readBackDesc = {};
    readBackDesc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    readBackDesc.size = numBlocks * sizeof(IdealEndpointsAndWeights);
    readbackBuffer = device.CreateBuffer(&readBackDesc);

    //Readback buffer for debuging (pass2)
    wgpu::BufferDescriptor readBackDesc2 = {};
    readBackDesc2.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    readBackDesc2.size = decimation_mode_trials.size() * BLOCK_MAX_WEIGHTS * sizeof(float);
    readbackBuffer2 = device.CreateBuffer(&readBackDesc2);

    //Readback buffer for debuging (pass6)
    wgpu::BufferDescriptor readBackDesc6_1 = {};
    readBackDesc6_1.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    readBackDesc6_1.size = block_mode_trials.size() * sizeof(FinalValueRange);
    readbackBuffer6_1 = device.CreateBuffer(&readBackDesc6_1);

    //Readback buffer for debuging (pass7)
    wgpu::BufferDescriptor readBackDesc7 = {};
    readBackDesc7.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    readBackDesc7.size = block_mode_trials.size() * sizeof(QuantizationResult);
    readbackBuffer7 = device.CreateBuffer(&readBackDesc7);

    //Readback buffer for debuging (pass10)
    wgpu::BufferDescriptor readBackDesc10 = {};
    readBackDesc10.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    readBackDesc10.size = numBlocks * QUANT_LEVELS * MAX_INT_COUNT_COMBINATIONS * sizeof(CombinedEndpointFormats);
    readbackBuffer10 = device.CreateBuffer(&readBackDesc10);

	//Readback buffer for debuging (pass11)
	wgpu::BufferDescriptor readBackDesc11 = {};
	readBackDesc11.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
	readBackDesc11.size = block_mode_trials.size() * sizeof(ColorCombinationResult);
	readbackBuffer11 = device.CreateBuffer(&readBackDesc11);

    //Readback buffer for debuging (pass12)
    wgpu::BufferDescriptor readBackDesc12 = {};
    readBackDesc12.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    readBackDesc12.size = numBlocks * block_descriptor.uniform_variables.tune_candidate_limit * sizeof(FinalCandidate);
    readbackBuffer12 = device.CreateBuffer(&readBackDesc12);
}

void ASTCEncoder::initBindGroups() {

    //bind group for pass1 (ideal endpoints and weights)
    std::vector<wgpu::BindGroupEntry> bg1_entries;
    bg1_entries.push_back({ .binding = 0, .buffer = uniformsBuffer, .offset = 0, .size = uniformsBuffer.GetSize() });
    bg1_entries.push_back({ .binding = 1, .buffer = inputBlocksBuffer, .offset = 0, .size = inputBlocksBuffer.GetSize() });
    bg1_entries.push_back({ .binding = 2, .buffer = pass1_output_idealEndpointsAndWeights, .offset = 0, .size = pass1_output_idealEndpointsAndWeights.GetSize() });

    wgpu::BindGroupDescriptor bg1_desc = {};
    bg1_desc.layout = pass1_bindGroupLayout;
    bg1_desc.entryCount = bg1_entries.size();
    bg1_desc.entries = bg1_entries.data();
    pass1_bindGroup = device.CreateBindGroup(&bg1_desc);


    //bind group for pass2 (decimated weights)
    std::vector<wgpu::BindGroupEntry> bg2_entries;
    bg2_entries.push_back({ .binding = 0, .buffer = uniformsBuffer, .offset = 0, .size = uniformsBuffer.GetSize() });
    bg2_entries.push_back({ .binding = 1, .buffer = decimationModeTrialsBuffer, .offset = 0, .size = decimationModeTrialsBuffer.GetSize() });
    bg2_entries.push_back({ .binding = 2, .buffer = decimationInfoBuffer, .offset = 0, .size = decimationInfoBuffer.GetSize() });
    bg2_entries.push_back({ .binding = 3, .buffer = texelToWeightMapBuffer, .offset = 0, .size = texelToWeightMapBuffer.GetSize() });
    bg2_entries.push_back({ .binding = 4, .buffer = weightToTexelMapBuffer, .offset = 0, .size = weightToTexelMapBuffer.GetSize() });
    bg2_entries.push_back({ .binding = 5, .buffer = pass1_output_idealEndpointsAndWeights, .offset = 0, .size = pass1_output_idealEndpointsAndWeights.GetSize() });
    bg2_entries.push_back({ .binding = 6, .buffer = pass2_output_decimatedWeights, .offset = 0, .size = pass2_output_decimatedWeights.GetSize() });

    wgpu::BindGroupDescriptor bg2_desc = {};
    bg2_desc.layout = pass2_bindGroupLayout;
    bg2_desc.entryCount = bg2_entries.size();
    bg2_desc.entries = bg2_entries.data();
    pass2_bindGroup = device.CreateBindGroup(&bg2_desc);


    //bind group for pass3 (angular offsets)
    std::vector<wgpu::BindGroupEntry> bg3_entries;
    bg3_entries.push_back({ .binding = 0, .buffer = uniformsBuffer, .offset = 0, .size = uniformsBuffer.GetSize() });
    bg3_entries.push_back({ .binding = 1, .buffer = decimationModeTrialsBuffer, .offset = 0, .size = decimationModeTrialsBuffer.GetSize() });
    bg3_entries.push_back({ .binding = 2, .buffer = decimationInfoBuffer, .offset = 0, .size = decimationInfoBuffer.GetSize() });
    bg3_entries.push_back({ .binding = 3, .buffer = sinBuffer, .offset = 0, .size = sinBuffer.GetSize() });
    bg3_entries.push_back({ .binding = 4, .buffer = cosBuffer, .offset = 0, .size = cosBuffer.GetSize() });
    bg3_entries.push_back({ .binding = 5, .buffer = pass2_output_decimatedWeights, .offset = 0, .size = pass2_output_decimatedWeights.GetSize() });
    bg3_entries.push_back({ .binding = 6, .buffer = pass3_output_angular_offsets, .offset = 0, .size = pass3_output_angular_offsets.GetSize() });

    wgpu::BindGroupDescriptor bg3_desc = {};
    bg3_desc.layout = pass3_bindGroupLayout;
    bg3_desc.entryCount = bg3_entries.size();
    bg3_desc.entries = bg3_entries.data();
    pass3_bindGroup = device.CreateBindGroup(&bg3_desc);


    //bind group for pass4 (lowest nad highest weight)
    std::vector<wgpu::BindGroupEntry> bg4_entries;
    bg4_entries.push_back({ .binding = 0, .buffer = uniformsBuffer, .offset = 0, .size = uniformsBuffer.GetSize() });
    bg4_entries.push_back({ .binding = 1, .buffer = decimationModeTrialsBuffer, .offset = 0, .size = decimationModeTrialsBuffer.GetSize() });
    bg4_entries.push_back({ .binding = 2, .buffer = decimationInfoBuffer, .offset = 0, .size = decimationInfoBuffer.GetSize() });
    bg4_entries.push_back({ .binding = 3, .buffer = pass2_output_decimatedWeights, .offset = 0, .size = pass2_output_decimatedWeights.GetSize() });
    bg4_entries.push_back({ .binding = 4, .buffer = pass3_output_angular_offsets, .offset = 0, .size = pass3_output_angular_offsets.GetSize() });
    bg4_entries.push_back({ .binding = 5, .buffer = pass4_output_lowestAndHighestWeight, .offset = 0, .size = pass4_output_lowestAndHighestWeight.GetSize() });

    wgpu::BindGroupDescriptor bg4_desc = {};
    bg4_desc.layout = pass4_bindGroupLayout;
    bg4_desc.entryCount = bg4_entries.size();
    bg4_desc.entries = bg4_entries.data();
    pass4_bindGroup = device.CreateBindGroup(&bg4_desc);
    

    //bind group for pass5 (best values for quant levels)
    std::vector<wgpu::BindGroupEntry> bg5_entries;
    bg5_entries.push_back({ .binding = 0, .buffer = uniformsBuffer, .offset = 0, .size = uniformsBuffer.GetSize() });
    bg5_entries.push_back({ .binding = 1, .buffer = decimationModeTrialsBuffer, .offset = 0, .size = decimationModeTrialsBuffer.GetSize() });
    bg5_entries.push_back({ .binding = 2, .buffer = decimationInfoBuffer, .offset = 0, .size = decimationInfoBuffer.GetSize() });
    bg5_entries.push_back({ .binding = 3, .buffer = pass3_output_angular_offsets, .offset = 0, .size = pass3_output_angular_offsets.GetSize() });
    bg5_entries.push_back({ .binding = 4, .buffer = pass4_output_lowestAndHighestWeight, .offset = 0, .size = pass4_output_lowestAndHighestWeight.GetSize() });
    bg5_entries.push_back({ .binding = 5, .buffer = pass5_output_lowValues, .offset = 0, .size = pass5_output_lowValues.GetSize() });
    bg5_entries.push_back({ .binding = 6, .buffer = pass5_output_highValues, .offset = 0, .size = pass5_output_highValues.GetSize() });

    wgpu::BindGroupDescriptor bg5_desc = {};
    bg5_desc.layout = pass5_bindGroupLayout;
    bg5_desc.entryCount = bg5_entries.size();
    bg5_desc.entries = bg5_entries.data();
    pass5_bindGroup = device.CreateBindGroup(&bg5_desc);


    //bind group for pass6 (remap low and high values)
    std::vector<wgpu::BindGroupEntry> bg6_entries;
    bg6_entries.push_back({ .binding = 0, .buffer = uniformsBuffer, .offset = 0, .size = uniformsBuffer.GetSize() });
    bg6_entries.push_back({ .binding = 1, .buffer = blockModeTrialsBuffer, .offset = 0, .size = blockModeTrialsBuffer.GetSize() });
    bg6_entries.push_back({ .binding = 2, .buffer = blockModesBuffer, .offset = 0, .size = blockModesBuffer.GetSize() });
    bg6_entries.push_back({ .binding = 3, .buffer = pass5_output_lowValues, .offset = 0, .size = pass5_output_lowValues.GetSize() });
    bg6_entries.push_back({ .binding = 4, .buffer = pass5_output_highValues, .offset = 0, .size = pass5_output_highValues.GetSize() });
    bg6_entries.push_back({ .binding = 5, .buffer = pass6_output_finalValueRanges, .offset = 0, .size = pass6_output_finalValueRanges.GetSize() });

    wgpu::BindGroupDescriptor bg6_desc = {};
    bg6_desc.layout = pass6_bindGroupLayout;
    bg6_desc.entryCount = bg6_entries.size();
    bg6_desc.entries = bg6_entries.data();
    pass6_bindGroup = device.CreateBindGroup(&bg6_desc);


    //bind group for pass7 (weights and error for block mode)
    std::vector<wgpu::BindGroupEntry> bg7_entries;
    bg7_entries.push_back({ .binding = 0, .buffer = uniformsBuffer, .offset = 0, .size = uniformsBuffer.GetSize() });
    bg7_entries.push_back({ .binding = 1, .buffer = blockModeTrialsBuffer, .offset = 0, .size = blockModeTrialsBuffer.GetSize() });
    bg7_entries.push_back({ .binding = 2, .buffer = blockModesBuffer, .offset = 0, .size = blockModesBuffer.GetSize() });
    bg7_entries.push_back({ .binding = 3, .buffer = decimationInfoBuffer, .offset = 0, .size = decimationInfoBuffer.GetSize() });
    bg7_entries.push_back({ .binding = 4, .buffer = pass2_output_decimatedWeights, .offset = 0, .size = pass2_output_decimatedWeights.GetSize() });
    bg7_entries.push_back({ .binding = 5, .buffer = pass6_output_finalValueRanges, .offset = 0, .size = pass6_output_finalValueRanges.GetSize() });
    bg7_entries.push_back({ .binding = 6, .buffer = pass1_output_idealEndpointsAndWeights, .offset = 0, .size = pass1_output_idealEndpointsAndWeights.GetSize() });
    bg7_entries.push_back({ .binding = 7, .buffer = texelToWeightMapBuffer, .offset = 0, .size = texelToWeightMapBuffer.GetSize() });
    bg7_entries.push_back({ .binding = 8, .buffer = pass7_output_quantizationResults, .offset = 0, .size = pass7_output_quantizationResults.GetSize() });

    wgpu::BindGroupDescriptor bg7_desc = {};
    bg7_desc.layout = pass7_bindGroupLayout;
    bg7_desc.entryCount = bg7_entries.size();
    bg7_desc.entries = bg7_entries.data();
    pass7_bindGroup = device.CreateBindGroup(&bg7_desc);

	//bind group for pass8 (encoding choice errors)
	std::vector<wgpu::BindGroupEntry> bg8_entries;
	bg8_entries.push_back({ .binding = 0, .buffer = uniformsBuffer, .offset = 0, .size = uniformsBuffer.GetSize() });
	bg8_entries.push_back({ .binding = 1, .buffer = inputBlocksBuffer, .offset = 0, .size = inputBlocksBuffer.GetSize() });
	bg8_entries.push_back({ .binding = 2, .buffer = pass1_output_idealEndpointsAndWeights, .offset = 0, .size = pass1_output_idealEndpointsAndWeights.GetSize() });
	bg8_entries.push_back({ .binding = 3, .buffer = pass8_output_encodingChoiceErrors, .offset = 0, .size = pass8_output_encodingChoiceErrors.GetSize() });

	wgpu::BindGroupDescriptor bg8_desc = {};
	bg8_desc.layout = pass8_bindGroupLayout;
	bg8_desc.entryCount = bg8_entries.size();
	bg8_desc.entries = bg8_entries.data();
	pass8_bindGroup = device.CreateBindGroup(&bg8_desc);

	//bind group for pass9 (color format errors)
	std::vector<wgpu::BindGroupEntry> bg9_entries;
	bg9_entries.push_back({ .binding = 0, .buffer = uniformsBuffer, .offset = 0, .size = uniformsBuffer.GetSize() });
	bg9_entries.push_back({ .binding = 1, .buffer = inputBlocksBuffer, .offset = 0, .size = inputBlocksBuffer.GetSize() });
	bg9_entries.push_back({ .binding = 2, .buffer = pass1_output_idealEndpointsAndWeights, .offset = 0, .size = pass1_output_idealEndpointsAndWeights.GetSize() });
	bg9_entries.push_back({ .binding = 3, .buffer = pass8_output_encodingChoiceErrors, .offset = 0, .size = pass8_output_encodingChoiceErrors.GetSize() });
	bg9_entries.push_back({ .binding = 4, .buffer = pass9_output_colorFormatErrors, .offset = 0, .size = pass9_output_colorFormatErrors.GetSize() });
	bg9_entries.push_back({ .binding = 5, .buffer = pass9_output_colorFormats, .offset = 0, .size = pass9_output_colorFormats.GetSize() });

	wgpu::BindGroupDescriptor bg9_desc = {};
	bg9_desc.layout = pass9_bindGroupLayout;
	bg9_desc.entryCount = bg9_entries.size();
	bg9_desc.entries = bg9_entries.data();
	pass9_bindGroup = device.CreateBindGroup(&bg9_desc);

    //bind group for pass10 (color endpoint combinations)
    std::vector<wgpu::BindGroupEntry> bg10_entries;
    bg10_entries.push_back({ .binding = 0, .buffer = uniformsBuffer, .offset = 0, .size = uniformsBuffer.GetSize() });
    bg10_entries.push_back({ .binding = 1, .buffer = pass9_output_colorFormatErrors, .offset = 0, .size = pass9_output_colorFormatErrors.GetSize() });
    bg10_entries.push_back({ .binding = 2, .buffer = pass9_output_colorFormats, .offset = 0, .size = pass9_output_colorFormats.GetSize() });
    bg10_entries.push_back({ .binding = 3, .buffer = pass10_output_colorEndpointCombinations, .offset = 0, .size = pass10_output_colorEndpointCombinations.GetSize() });

    wgpu::BindGroupDescriptor bg10_desc = {};
    bg10_desc.layout = pass10_bindGroupLayout;
    bg10_desc.entryCount = bg10_entries.size();
    bg10_desc.entries = bg10_entries.data();
    pass10_bindGroup = device.CreateBindGroup(&bg10_desc);

	//bind group for pass11, 1 partition (best endpoint combinations for mode)
	std::vector<wgpu::BindGroupEntry> bg11_1_entries;
	bg11_1_entries.push_back({ .binding = 0, .buffer = uniformsBuffer, .offset = 0, .size = uniformsBuffer.GetSize() });
	bg11_1_entries.push_back({ .binding = 1, .buffer = blockModeTrialsBuffer, .offset = 0, .size = blockModeTrialsBuffer.GetSize() });
	bg11_1_entries.push_back({ .binding = 2, .buffer = pass7_output_quantizationResults, .offset = 0, .size = pass7_output_quantizationResults.GetSize() });
	bg11_1_entries.push_back({ .binding = 3, .buffer = pass9_output_colorFormatErrors, .offset = 0, .size = pass9_output_colorFormatErrors.GetSize() });
	bg11_1_entries.push_back({ .binding = 4, .buffer = pass9_output_colorFormats, .offset = 0, .size = pass9_output_colorFormats.GetSize() });
    bg11_1_entries.push_back({ .binding = 5, .buffer = pass11_output_bestEndpointCombinationsForMode, .offset = 0, .size = pass11_output_bestEndpointCombinationsForMode.GetSize() });

	wgpu::BindGroupDescriptor bg11_1_desc = {};
	bg11_1_desc.layout = pass11_bindGroupLayout_1part;
	bg11_1_desc.entryCount = bg11_1_entries.size();
	bg11_1_desc.entries = bg11_1_entries.data();
	pass11_bindGroup_1part = device.CreateBindGroup(&bg11_1_desc);

	//bind group for pass11, 2,3,4 partitions (best endpoint combinations for mode)
	std::vector<wgpu::BindGroupEntry> bg11_234_entries;
	bg11_234_entries.push_back({ .binding = 0, .buffer = uniformsBuffer, .offset = 0, .size = uniformsBuffer.GetSize() });
	bg11_234_entries.push_back({ .binding = 1, .buffer = blockModeTrialsBuffer, .offset = 0, .size = blockModeTrialsBuffer.GetSize() });
	bg11_234_entries.push_back({ .binding = 2, .buffer = pass7_output_quantizationResults, .offset = 0, .size = pass7_output_quantizationResults.GetSize() });
	bg11_234_entries.push_back({ .binding = 3, .buffer = pass10_output_colorEndpointCombinations, .offset = 0, .size = pass10_output_colorEndpointCombinations.GetSize() });
	bg11_234_entries.push_back({ .binding = 4, .buffer = pass11_output_bestEndpointCombinationsForMode, .offset = 0, .size = pass11_output_bestEndpointCombinationsForMode.GetSize() });

	wgpu::BindGroupDescriptor bg11_234_desc = {};
	bg11_234_desc.layout = pass11_bindGroupLayout_234part;
	bg11_234_desc.entryCount = bg11_234_entries.size();
	bg11_234_desc.entries = bg11_234_entries.data();
	pass11_bindGroup_234part = device.CreateBindGroup(&bg11_234_desc);

	//bind group for pass12 (final candidates)
    std::vector<wgpu::BindGroupEntry> bg12_entries;
    bg12_entries.push_back({ .binding = 0, .buffer = uniformsBuffer, .offset = 0, .size = uniformsBuffer.GetSize() });
    bg12_entries.push_back({ .binding = 1, .buffer = blockModeTrialsBuffer, .offset = 0, .size = blockModeTrialsBuffer.GetSize() });
    bg12_entries.push_back({ .binding = 2, .buffer = modesPerBlockBuffer, .offset = 0, .size = modesPerBlockBuffer.GetSize() });
    bg12_entries.push_back({ .binding = 3, .buffer = blockModeTrialOffsetsBuffer, .offset = 0, .size = blockModeTrialOffsetsBuffer.GetSize() });
    bg12_entries.push_back({ .binding = 4, .buffer = pass11_output_bestEndpointCombinationsForMode, .offset = 0, .size = pass11_output_bestEndpointCombinationsForMode.GetSize() });
    bg12_entries.push_back({ .binding = 5, .buffer = pass12_output_finalCandidates, .offset = 0, .size = pass12_output_finalCandidates.GetSize() });

    wgpu::BindGroupDescriptor bg12_desc = {};
    bg12_desc.layout = pass12_bindGroupLayout;
    bg12_desc.entryCount = bg12_entries.size();
    bg12_desc.entries = bg12_entries.data();
    pass12_bindGroup = device.CreateBindGroup(&bg12_desc);

}

void ASTCEncoder::encode(uint8_t* imageData) {

    //initialize compression metadata
    initMetadata();

    //initialize buffers, pipelines and bind groups
    initBindGroupLayouts();
    initBuffers();
    initPipelines();
    initBindGroups();

    //write to uniforms buffer
    queue.WriteBuffer(uniformsBuffer, 0, &block_descriptor.uniform_variables, sizeof(uniform_variables));

    //write to decimation mode trials buffer
    queue.WriteBuffer(decimationModeTrialsBuffer, 0, decimation_mode_trials.data(), decimation_mode_trials.size() * sizeof(DecimationModeTrial));

    //write to block mode trials buffer
    queue.WriteBuffer(blockModeTrialsBuffer, 0, block_mode_trials.data(), block_mode_trials.size() * sizeof(BlockModeTrial));
	queue.WriteBuffer(modesPerBlockBuffer, 0, modes_per_block.data(), numBlocks * sizeof(uint32_t));
	queue.WriteBuffer(blockModeTrialOffsetsBuffer, 0, block_mode_trial_offsets.data(), numBlocks * sizeof(uint32_t));

    //write to block mode and decimation mode buffers
    queue.WriteBuffer(blockModesBuffer, 0, block_descriptor.block_modes, block_descriptor.uniform_variables.block_mode_count * sizeof(block_mode));
    queue.WriteBuffer(blockModeIndexBuffer, 0, block_descriptor.block_mode_index, block_descriptor.uniform_variables.block_mode_count * sizeof(uint32_t));
    queue.WriteBuffer(decimationModesBuffer, 0, block_descriptor.decimation_modes, block_descriptor.uniform_variables.decimation_mode_count * sizeof(decimation_mode));
    queue.WriteBuffer(decimationInfoBuffer, 0, block_descriptor.decimation_info_metadata, block_descriptor.uniform_variables.decimation_mode_count * sizeof(decimation_info));
    queue.WriteBuffer(texelToWeightMapBuffer, 0, block_descriptor.decimation_info_packed.texel_to_weight_map_data.data(), block_descriptor.decimation_info_packed.texel_to_weight_map_data.size() * sizeof(TexelToWeightMap));
    queue.WriteBuffer(weightToTexelMapBuffer, 0, block_descriptor.decimation_info_packed.weight_to_texel_map_data.data(), block_descriptor.decimation_info_packed.weight_to_texel_map_data.size() * sizeof(WeightToTexelMap));

    //write to sin and cos table buffers
    queue.WriteBuffer(sinBuffer, 0, sin_table.data(), SINCOS_STEPS * ANGULAR_STEPS * sizeof(float));
    queue.WriteBuffer(cosBuffer, 0, cos_table.data(), SINCOS_STEPS * ANGULAR_STEPS * sizeof(float));

    //Split texture into blocks and populate input buffer
    std::vector<InputBlock> blocks = SplitImageIntoBlocks(imageData, textureWidth, textureHeight, blockXDim, blockYDim);

    queue.WriteBuffer(inputBlocksBuffer, 0, blocks.data(), numBlocks * sizeof(InputBlock));

    int decimation_mode_trials_num = decimation_mode_trials.size();
    int block_mode_trials_num = block_mode_trials.size();

    // Command encoder
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    {
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
        pass.SetPipeline(pass1_pipeline);
        pass.SetBindGroup(0, pass1_bindGroup, 0, nullptr);
        pass.DispatchWorkgroups(numBlocks, 1, 1);
        pass.End();
    }
    {
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
        pass.SetPipeline(pass2_pipeline);
        pass.SetBindGroup(0, pass2_bindGroup, 0, nullptr);
        pass.DispatchWorkgroups(decimation_mode_trials_num, 1, 1); //run shader for each (block, deciamtion_mode) pair from decimation_mode_trials
        pass.End();
    }
    {
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
        pass.SetPipeline(pass3_pipeline);
        pass.SetBindGroup(0, pass3_bindGroup, 0, nullptr);
        pass.DispatchWorkgroups(decimation_mode_trials_num, 1, 1); //run shader for each (block, deciamtion_mode) pair from decimation_mode_trials
        pass.End();
    }
    {
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
        pass.SetPipeline(pass4_pipeline);
        pass.SetBindGroup(0, pass4_bindGroup, 0, nullptr);
        pass.DispatchWorkgroups(decimation_mode_trials_num, 1, 1); //run shader for each (block, deciamtion_mode) pair from decimation_mode_trials
        pass.End();
    }
    {
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
        pass.SetPipeline(pass5_pipeline);
        pass.SetBindGroup(0, pass5_bindGroup, 0, nullptr);
        pass.DispatchWorkgroups(decimation_mode_trials_num, 1, 1); //run shader for each (block, deciamtion_mode) pair from decimation_mode_trials
        pass.End();
    }
    {
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
        pass.SetPipeline(pass6_pipeline);
        pass.SetBindGroup(0, pass6_bindGroup, 0, nullptr);
        pass.DispatchWorkgroups(block_mode_trials_num, 1, 1); //run shader for each (block, deciamtion_mode) pair from decimation_mode_trials
        pass.End();
    }
    {
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
        pass.SetPipeline(pass7_pipeline);
        pass.SetBindGroup(0, pass7_bindGroup, 0, nullptr);
        pass.DispatchWorkgroups(block_mode_trials_num, 1, 1); //run shader for each (block, deciamtion_mode) pair from decimation_mode_trials
        pass.End();
    }
    {
		wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
		pass.SetPipeline(pass8_pipeline);
		pass.SetBindGroup(0, pass8_bindGroup, 0, nullptr);
		pass.DispatchWorkgroups(numBlocks, 1, 1); //run shader for each block
		pass.End();
    }
    {
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
        pass.SetPipeline(pass9_pipeline);
        pass.SetBindGroup(0, pass9_bindGroup, 0, nullptr);
        pass.DispatchWorkgroups(numBlocks, 1, 1); //run shader for each block
        pass.End();
    }
    {
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
        pass.SetPipeline(pass11_pipeline_1part);
        pass.SetBindGroup(0, pass11_bindGroup_1part, 0, nullptr);
        pass.DispatchWorkgroups(block_mode_trials_num, 1, 1); //run shader for each (block, deciamtion_mode) pair from decimation_mode_trials
        pass.End();
    }
    {
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
        pass.SetPipeline(pass12_pipeline);
        pass.SetBindGroup(0, pass12_bindGroup, 0, nullptr);
        pass.DispatchWorkgroups(numBlocks, 1, 1); //run shader for each block
        pass.End();
    }


    // Copy to readback buffer
    encoder.CopyBufferToBuffer(pass1_output_idealEndpointsAndWeights, 0, readbackBuffer, 0, numBlocks * sizeof(IdealEndpointsAndWeights));
    encoder.CopyBufferToBuffer(pass2_output_decimatedWeights, 0, readbackBuffer2, 0, decimation_mode_trials.size() * BLOCK_MAX_WEIGHTS * sizeof(float));
    encoder.CopyBufferToBuffer(pass6_output_finalValueRanges, 0, readbackBuffer6_1, 0, block_mode_trials_num * sizeof(FinalValueRange));
    encoder.CopyBufferToBuffer(pass7_output_quantizationResults, 0, readbackBuffer7, 0, block_mode_trials_num * sizeof(QuantizationResult));
	encoder.CopyBufferToBuffer(pass10_output_colorEndpointCombinations, 0, readbackBuffer10, 0, numBlocks* QUANT_LEVELS* MAX_INT_COUNT_COMBINATIONS * sizeof(CombinedEndpointFormats));
	encoder.CopyBufferToBuffer(pass11_output_bestEndpointCombinationsForMode, 0, readbackBuffer11, 0, block_mode_trials_num * sizeof(ColorCombinationResult));
	encoder.CopyBufferToBuffer(pass12_output_finalCandidates, 0, readbackBuffer12, 0, numBlocks* block_descriptor.uniform_variables.tune_candidate_limit * sizeof(FinalCandidate));
    wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);


    //print out output of pass 1
    std::vector<IdealEndpointsAndWeights> output(numBlocks);

    mapOutputBufferSync<IdealEndpointsAndWeights>(device, readbackBuffer, numBlocks, output);

    std::cout << "== Compressed blocks (dummy values) ==" << std::endl;
    for (uint32_t i = 0; i < numBlocks; ++i) {
        std::cout << "Block " << i << ":\n";
        std::cout << "avg: " << output[i].partitions[0].avg[0] << " " << output[i].partitions[0].avg[1] << " " << output[i].partitions[0].avg[2] << " " << output[i].partitions[0].avg[3] << "\n";
        std::cout << "dir: " << output[i].partitions[0].dir[0] << " " << output[i].partitions[0].dir[1] << " " << output[i].partitions[0].dir[2] << " " << output[i].partitions[0].dir[3] << "\n";
        std::cout << std::endl;
    }

    //map output
    std::vector<float> output2(decimation_mode_trials.size() * BLOCK_MAX_WEIGHTS);
    mapOutputBufferSync<float>(device, readbackBuffer2, decimation_mode_trials.size() * BLOCK_MAX_WEIGHTS, output2);

    std::vector<FinalValueRange> output6_1(block_mode_trials_num);
    mapOutputBufferSync<FinalValueRange>(device, readbackBuffer6_1, block_mode_trials_num, output6_1);

    std::vector<QuantizationResult> output7(block_mode_trials_num);
    mapOutputBufferSync<QuantizationResult>(device, readbackBuffer7, block_mode_trials_num, output7);

	std::vector<CombinedEndpointFormats> output10(numBlocks* QUANT_LEVELS* MAX_INT_COUNT_COMBINATIONS);
	mapOutputBufferSync<CombinedEndpointFormats>(device, readbackBuffer10, numBlocks * QUANT_LEVELS * MAX_INT_COUNT_COMBINATIONS, output10);

	std::vector<ColorCombinationResult> output11(block_mode_trials_num);
	mapOutputBufferSync<ColorCombinationResult>(device, readbackBuffer11, block_mode_trials_num, output11);

	std::vector<FinalCandidate> output12(numBlocks* block_descriptor.uniform_variables.tune_candidate_limit);
	mapOutputBufferSync<FinalCandidate>(device, readbackBuffer12, numBlocks * block_descriptor.uniform_variables.tune_candidate_limit, output12);

    readbackBuffer.Unmap();
    readbackBuffer6_1.Unmap();
    readbackBuffer7.Unmap();
	readbackBuffer2.Unmap();
	readbackBuffer10.Unmap();
    readbackBuffer11.Unmap();
}

ASTCEncoder::~ASTCEncoder() {
    // Release all WebGPU resources
}