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

            block.ypos = by;
            block.xpos = bx;

            for (int c = 0; c < 4; c++) {
                block.data_min[c] = 1e38f;
                block.data_max[c] = -1e38f;
            }

            bool grayscale = true;

            int pixelIndex = 0;
            for (int dy = 0; dy < blockHeight; ++dy) {
                for (int dx = 0; dx < blockWidth; ++dx) {
                    int x = bx * blockWidth + dx;
                    int y = by * blockHeight + dy;
                    int idx = (y * width + x) * 4;


                    for (int c = 0; c < 4; c++) {
                        float channel_value = (imageData[idx + c] / 255.0f) * 65536.0f;

                        block.pixels[pixelIndex].data[c] = channel_value;

                        if (channel_value > block.data_max[c]) {
                            block.data_max[c] = channel_value;
                        }

                        if (channel_value < block.data_min[c]) {
                            block.data_min[c] = channel_value;
                        }
                    }

                    grayscale = grayscale && (block.pixels[pixelIndex].data[0] == block.pixels[pixelIndex].data[1] && block.pixels[pixelIndex].data[0] == block.pixels[pixelIndex].data[2]);

                    block.pixels[pixelIndex].partition = 0;

                    ++pixelIndex;
                }
            }

            block.grayscale = grayscale;

			block.partition_pixel_counts[0] = blockWidth * blockHeight;
            block.partition_pixel_counts[1] = 0;
			block.partition_pixel_counts[2] = 0;
			block.partition_pixel_counts[3] = 0;

            blocks.push_back(block);
        }
    }

    return blocks;
}

//generate best partitionings for provided input blocks
static int generateBlockPartitionings(
    const block_descriptor& block_descriptor,
    const std::vector<InputBlock>& unpartitionedBlocks,
    int partition_count,
    int partition_search_limint,
    int requested_candidates,
    std::vector<InputBlock>& partitionedBlocks,
    std::vector<int>& blockOffsets
) {
    int imageBlocksNum = unpartitionedBlocks.size();
    int texel_count = block_descriptor.uniform_variables.texel_count;

    partitionedBlocks.clear();
    blockOffsets.clear();

    blockOffsets.resize(imageBlocksNum);
    int nextOffset = 0;

    for (int i = 0; i < imageBlocksNum; i++) {
        unsigned int partition_indices[TUNE_MAX_PARTITIONING_CANDIDATES];
        int candidatecount = find_best_partition_candidates(block_descriptor, unpartitionedBlocks[i], partition_count, partition_search_limint, partition_indices, requested_candidates);

        blockOffsets[i] = nextOffset;
        nextOffset += candidatecount;

        for (int j = 0; j < candidatecount; j++) {

            partitionedBlocks.emplace_back(unpartitionedBlocks[i]);
            InputBlock& partitionedBlock = partitionedBlocks.back();

            auto& pi = block_descriptor.get_partition_info(partition_count, partition_indices[j]);

            for (int tex = 0; tex < texel_count; tex++) {
                partitionedBlock.pixels[tex].partition = pi.partition_of_texel[tex];
            }

            partitionedBlock.partitioning_idx = pi.partition_index;

            for (int p = 0; p < partition_count; p++) {
                partitionedBlock.partition_pixel_counts[p] = pi.partition_texel_count[p];
            }
        }

    }

    return nextOffset;
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

    init_partition_tables(block_descriptor, false, 4);

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

            unsigned int mode_packed_index = block_descriptor.block_mode_index[mode_idx];

            if (mode_packed_index == BLOCK_BAD_BLOCK_MODE) {
                continue;
            }

            const block_mode& bm = block_descriptor.block_modes[mode_packed_index];
            
            if (bm.is_dual_plane) {
                continue;
            }

            if (inverseArray[block_idx * WEIGHTS_MAX_DECIMATION_MODES + bm.decimation_mode] < 0) {
                continue;
            }

            block_mode_trials.push_back({ block_idx, mode_packed_index, uint32_t(inverseArray[block_idx * WEIGHTS_MAX_DECIMATION_MODES + bm.decimation_mode]) });
			modes_per_block[block_idx]++;
			block_mode_trial_offset++;
        }
    }

    block_descriptor.uniform_variables.quant_limit = QUANT_32;
    block_descriptor.uniform_variables.partition_count = 1;
    block_descriptor.uniform_variables.tune_candidate_limit = TUNE_MAX_TRIAL_CANDIDATES;
}


void ASTCEncoder::encode(uint8_t* imageData, uint8_t* dataOut, size_t dataLen) {

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
    int numCandidates = block_descriptor.uniform_variables.tune_candidate_limit;

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
        pass.DispatchWorkgroups(block_mode_trials_num, 1, 1); //run shader for each (block, block_mode) pair from block_mode_trials
        pass.End();
    }
    {
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
        pass.SetPipeline(pass7_pipeline);
        pass.SetBindGroup(0, pass7_bindGroup, 0, nullptr);
        pass.DispatchWorkgroups(block_mode_trials_num, 1, 1); //run shader for each (block, block_mode) pair from block_mode_trials
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
        pass.DispatchWorkgroups(block_mode_trials_num, 1, 1); //run shader for each (block, block_mode) pair from block_mode_trials
        pass.End();
    }
    {
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
        pass.SetPipeline(pass12_pipeline);
        pass.SetBindGroup(0, pass12_bindGroup, 0, nullptr);
        pass.DispatchWorkgroups(numBlocks, 1, 1); //run shader for each block
        pass.End();
    }
	{
		wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
		pass.SetPipeline(pass13_pipeline);
		pass.SetBindGroup(0, pass13_bindGroup, 0, nullptr);
		pass.DispatchWorkgroups(numBlocks * numCandidates, 1, 1); //run shader for each final candidate
		pass.End();
	}
	{
		wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
		pass.SetPipeline(pass14_pipeline);
		pass.SetBindGroup(0, pass14_bindGroup, 0, nullptr);
		pass.DispatchWorkgroups(numBlocks * numCandidates, 1, 1); //run shader for each final candidate
		pass.End();
	}
    {
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
        pass.SetPipeline(pass15_pipeline);
        pass.SetBindGroup(0, pass15_bindGroup, 0, nullptr);
        pass.DispatchWorkgroups(numBlocks * numCandidates, 1, 1); //run shader for each final candidate
        pass.End();
    }
    {
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
        pass.SetPipeline(pass16_pipeline);
        pass.SetBindGroup(0, pass16_bindGroup, 0, nullptr);
        pass.DispatchWorkgroups(numBlocks * numCandidates, 1, 1); //run shader for each final candidate
        pass.End();
    }
    {
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
        pass.SetPipeline(pass17_pipeline);
        pass.SetBindGroup(0, pass17_bindGroup, 0, nullptr);
        pass.DispatchWorkgroups(numBlocks * numCandidates, 1, 1); //run shader for each final candidate
        pass.End();
    }
    {
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
        pass.SetPipeline(pass18_pipeline);
        pass.SetBindGroup(0, pass18_bindGroup, 0, nullptr);
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
    encoder.CopyBufferToBuffer(pass18_output_symbolicBlocks, 0, outputReadbackBuffer, 0, numBlocks * sizeof(SymbolicBlock));
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

    std::vector<SymbolicBlock> output18(numBlocks);
    mapOutputBufferSync<SymbolicBlock>(device, outputReadbackBuffer, numBlocks, output18);

    readbackBuffer.Unmap();
    readbackBuffer6_1.Unmap();
    readbackBuffer7.Unmap();
	readbackBuffer2.Unmap();
	readbackBuffer10.Unmap();
    readbackBuffer11.Unmap();

    for (int i = 0; i < numBlocks; i++) {
        int offset = i * 16;
        uint8_t* outputBlock = dataOut + offset;
        symbolic_to_physical(block_descriptor, output18[i], outputBlock);
    }

    std::vector<InputBlock> partitionedBlocks;
    std::vector<int> blockOffsets;

    generateBlockPartitionings(block_descriptor, blocks, 2, 18, 4, partitionedBlocks, blockOffsets);
    
}

ASTCEncoder::~ASTCEncoder() {
    // Release all WebGPU resources
}