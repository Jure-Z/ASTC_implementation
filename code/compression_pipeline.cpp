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

    int blocksX = (width + blockWidth - 1) / blockWidth;
    int blocksY = (height + blockHeight - 1) / blockHeight;

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

                int y = by * blockHeight + dy;
                int clamped_y = std::min(y, height - 1);

                for (int dx = 0; dx < blockWidth; ++dx) {

                    int x = bx * blockWidth + dx;
                    int clamped_x = std::min(x, width - 1);

                    int idx = (clamped_y * width + clamped_x) * 4;


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

            if (grayscale) {
                block.grayscale = 1;
            }
            else {
                block.grayscale = 0;
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
    this->blocksX = (textureWidth + blockXDim - 1) / blockXDim;
    this->blocksY = (textureHeight + blockYDim - 1) / blockYDim;
    this->numBlocks = blocksX * blocksY;
    this->blockXDim = blockXDim;
    this->blockYDim = blockYDim;
}

void ASTCEncoder::initMetadata() {
    construct_metadata_structures(blockXDim, blockYDim, block_descriptor);

    init_partition_tables(block_descriptor, false, 4);

    construct_angular_tables(sin_table, cos_table);
}

void ASTCEncoder::initTrialModes() {
    valid_decimation_modes.clear();
    valid_block_modes.clear();

    std::vector<uint32_t> decimation_mode_remap_table(WEIGHTS_MAX_DECIMATION_MODES, BLOCK_BAD_BLOCK_MODE);

    //Find all valid decimation modes
    int max_weight_quant = std::min(static_cast<int>(QUANT_32), quant_limit);
    static_cast<quant_method>(max_weight_quant);
    uint32_t mask = static_cast<uint32_t>((1 << (max_weight_quant + 1)) - 1);

    int trialCounter = 0;

    for (uint32_t mode_idx = 0; mode_idx < block_descriptor.uniform_variables.decimation_mode_count; ++mode_idx) {

        const decimation_mode& dm = block_descriptor.decimation_modes[mode_idx];
        if ((dm.refprec_1plane & mask) == 0) {
            continue;
        }

        valid_decimation_modes.push_back(mode_idx);

        decimation_mode_remap_table[mode_idx] = trialCounter;
        trialCounter++;
    }

    //Find all valid block modes
    for (uint32_t i = 0; i < WEIGHTS_MAX_BLOCK_MODES; ++i) {

        unsigned int mode_packed_index = block_descriptor.block_mode_index[i];

        if (mode_packed_index == BLOCK_BAD_BLOCK_MODE) {
            continue;
        }

        const block_mode& bm = block_descriptor.block_modes[mode_packed_index];

        if (bm.is_dual_plane) {
            continue;
        }

        // Check if its associated decimation mode is valid
        uint32_t remapped_dm_idx = decimation_mode_remap_table[bm.decimation_mode];
        if (remapped_dm_idx == BLOCK_BAD_BLOCK_MODE) {
            continue;
        }

        valid_block_modes.push_back({ mode_packed_index, remapped_dm_idx, 0, 0 });
    }

    block_descriptor.uniform_variables.valid_decimation_mode_count = valid_decimation_modes.size();
    block_descriptor.uniform_variables.valid_block_mode_count = valid_block_modes.size();
}


void ASTCEncoder::encode(uint8_t* imageData, uint8_t* dataOut, size_t dataLen) {

    //initialize compression metadata
    std::cout << "Precomputing compression data..." << std::endl;
    initMetadata();
    initTrialModes();

    //initialize buffers, pipelines and bind groups
    std::cout << "Initializing bind group layouts..." << std::endl;
    initBindGroupLayouts();
    std::cout << "Initializing storage buffers..." << std::endl;
    initBuffers();
    std::cout << "Initializing pipelines..." << std::endl;
    initPipelines();
    std::cout << "Initializing bind groups..." << std::endl;
    initBindGroups();

    //write the precomputed metadata to the buffers
    //write to block mode and decimation mode buffers
    std::cout << "Writing precomputed data to buffers..." << std::endl;
    queue.WriteBuffer(blockModesBuffer, 0, block_descriptor.block_modes, block_descriptor.uniform_variables.block_mode_count * sizeof(block_mode));
    queue.WriteBuffer(blockModeIndexBuffer, 0, block_descriptor.block_mode_index, block_descriptor.uniform_variables.block_mode_count * sizeof(uint32_t));
    queue.WriteBuffer(decimationModesBuffer, 0, block_descriptor.decimation_modes, block_descriptor.uniform_variables.decimation_mode_count * sizeof(decimation_mode));
    queue.WriteBuffer(decimationInfoBuffer, 0, block_descriptor.decimation_info_metadata, block_descriptor.uniform_variables.decimation_mode_count * sizeof(decimation_info));
    queue.WriteBuffer(texelToWeightMapBuffer, 0, block_descriptor.decimation_info_packed.texel_to_weight_map_data.data(), block_descriptor.decimation_info_packed.texel_to_weight_map_data.size() * sizeof(TexelToWeightMap));
    queue.WriteBuffer(weightToTexelMapBuffer, 0, block_descriptor.decimation_info_packed.weight_to_texel_map_data.data(), block_descriptor.decimation_info_packed.weight_to_texel_map_data.size() * sizeof(WeightToTexelMap));

    queue.WriteBuffer(validDecimationModesBuffer, 0, valid_decimation_modes.data(), valid_decimation_modes.size() * sizeof(uint32_t));
    queue.WriteBuffer(validBlockModesBuffer, 0, valid_block_modes.data(), valid_block_modes.size() * sizeof(PackedBlockModeLookup));

    //write to sin and cos table buffers
    queue.WriteBuffer(sinBuffer, 0, sin_table.data(), SINCOS_STEPS * ANGULAR_STEPS * sizeof(float));
    queue.WriteBuffer(cosBuffer, 0, cos_table.data(), SINCOS_STEPS * ANGULAR_STEPS * sizeof(float));


    //get image blocks
    std::cout << "Preparing image blocks..." << std::endl;
    std::vector<InputBlock> original_blocks = SplitImageIntoBlocks(imageData, textureWidth, textureHeight, blockXDim, blockYDim);

    // This vector will store the best encoding found for each block across all partition counts
    std::vector<SymbolicBlock> best_symbolic_blocks(numBlocks);
    for (auto& block : best_symbolic_blocks) {
        block.errorval = ERROR_CALC_DEFAULT;
    }

    // Iterate through all supported partition counts
    for (unsigned int p_count = 1; p_count <= 1; ++p_count) {
        std::cout << "Compressing with " << p_count << " partition(s)..." << std::endl;

        // Prepare block data for the current partition count
        std::vector<InputBlock> current_blocks;
        std::vector<int> block_offsets;
        int current_partitioned_blocks_num;
        block_descriptor.uniform_variables.partition_count = p_count;

        if (p_count == 1) {
            current_blocks = original_blocks;
            current_partitioned_blocks_num = original_blocks.size();
            block_offsets.resize(numBlocks);
            for (uint32_t i = 0; i < numBlocks; ++i) block_offsets[i] = i;
        }
        else {
            unsigned int search_limit = 18;
            unsigned int candidates = TUNE_MAX_PARTITIONING_CANDIDATES;
            current_partitioned_blocks_num = generateBlockPartitionings(
                block_descriptor, original_blocks, p_count, search_limit, candidates,
                current_blocks, block_offsets);
        }

        if (current_partitioned_blocks_num == 0) {
            std::cout << "No valid partitionings found for " << p_count << " partitions. Skipping." << std::endl;
            continue;
        }

        block_descriptor.uniform_variables.tune_candidate_limit = TUNE_MAX_TRIAL_CANDIDATES;
        block_descriptor.uniform_variables.quant_limit = QUANT_32;

        // Write data that is specific to this run
        queue.WriteBuffer(uniformsBuffer, 0, &block_descriptor.uniform_variables, sizeof(uniform_variables));
        queue.WriteBuffer(inputBlocksBuffer, 0, current_blocks.data(), current_partitioned_blocks_num * sizeof(InputBlock));

        int decimation_mode_trials_num = current_partitioned_blocks_num * valid_decimation_modes.size();
        int block_mode_trials_num = current_partitioned_blocks_num * valid_block_modes.size();
        int numCandidates = block_descriptor.uniform_variables.tune_candidate_limit;

        // Run the full compute pipeline
        wgpu::CommandEncoder encoder = device.CreateCommandEncoder();

        // Passes 1-9
        { wgpu::ComputePassEncoder pass = encoder.BeginComputePass(); pass.SetPipeline(pass1_pipeline); pass.SetBindGroup(0, pass1_bindGroup, 0, nullptr); pass.DispatchWorkgroups(current_partitioned_blocks_num, 1, 1); pass.End(); }
        { wgpu::ComputePassEncoder pass = encoder.BeginComputePass(); pass.SetPipeline(pass2_pipeline); pass.SetBindGroup(0, pass2_bindGroup, 0, nullptr); pass.DispatchWorkgroups(decimation_mode_trials_num, 1, 1); pass.End(); }
        { wgpu::ComputePassEncoder pass = encoder.BeginComputePass(); pass.SetPipeline(pass3_pipeline); pass.SetBindGroup(0, pass3_bindGroup, 0, nullptr); pass.DispatchWorkgroups(decimation_mode_trials_num, 1, 1); pass.End(); }
        { wgpu::ComputePassEncoder pass = encoder.BeginComputePass(); pass.SetPipeline(pass4_pipeline); pass.SetBindGroup(0, pass4_bindGroup, 0, nullptr); pass.DispatchWorkgroups(decimation_mode_trials_num, 1, 1); pass.End(); }
        { wgpu::ComputePassEncoder pass = encoder.BeginComputePass(); pass.SetPipeline(pass5_pipeline); pass.SetBindGroup(0, pass5_bindGroup, 0, nullptr); pass.DispatchWorkgroups(decimation_mode_trials_num, 1, 1); pass.End(); }
        { wgpu::ComputePassEncoder pass = encoder.BeginComputePass(); pass.SetPipeline(pass6_pipeline); pass.SetBindGroup(0, pass6_bindGroup, 0, nullptr); pass.DispatchWorkgroups(block_mode_trials_num, 1, 1); pass.End(); }
        { wgpu::ComputePassEncoder pass = encoder.BeginComputePass(); pass.SetPipeline(pass7_pipeline); pass.SetBindGroup(0, pass7_bindGroup, 0, nullptr); pass.DispatchWorkgroups(block_mode_trials_num, 1, 1); pass.End(); }
        { wgpu::ComputePassEncoder pass = encoder.BeginComputePass(); pass.SetPipeline(pass8_pipeline); pass.SetBindGroup(0, pass8_bindGroup, 0, nullptr); pass.DispatchWorkgroups(current_partitioned_blocks_num, 1, 1); pass.End(); }
        { wgpu::ComputePassEncoder pass = encoder.BeginComputePass(); pass.SetPipeline(pass9_pipeline); pass.SetBindGroup(0, pass9_bindGroup, 0, nullptr); pass.DispatchWorkgroups(current_partitioned_blocks_num, 1, 1); pass.End(); }

        // Pass 10 (Color Endpoint Combinations) is different for partition counts
        if (p_count > 1) {
            wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
            switch (p_count) {
            case 2: pass.SetPipeline(pass10_pipeline_2part); break;
            case 3: pass.SetPipeline(pass10_pipeline_3part); break;
            case 4: pass.SetPipeline(pass10_pipeline_4part); break;
            }
            pass.SetBindGroup(0, pass10_bindGroup, 0, nullptr);
            pass.DispatchWorkgroups(current_partitioned_blocks_num, 1, 1);
            pass.End();
        }

        // Pass 11 (Best Combination for Mode) Is different for partition counts
        {
            wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
            switch (p_count) {
            case 1: pass.SetPipeline(pass11_pipeline_1part); pass.SetBindGroup(0, pass11_bindGroup_1part, 0, nullptr); break;
            case 2: pass.SetPipeline(pass11_pipeline_2part); pass.SetBindGroup(0, pass11_bindGroup_234part, 0, nullptr); break;
            case 3: pass.SetPipeline(pass11_pipeline_3part); pass.SetBindGroup(0, pass11_bindGroup_234part, 0, nullptr); break;
            case 4: pass.SetPipeline(pass11_pipeline_4part); pass.SetBindGroup(0, pass11_bindGroup_234part, 0, nullptr); break;
            }
            pass.DispatchWorkgroups(block_mode_trials_num, 1, 1);
            pass.End();
        }

        // Pass 12 (Find Top N Candidates)
        { wgpu::ComputePassEncoder pass = encoder.BeginComputePass(); pass.SetPipeline(pass12_pipeline); pass.SetBindGroup(0, pass12_bindGroup, 0, nullptr); pass.DispatchWorkgroups(current_partitioned_blocks_num, 1, 1); pass.End(); }

        // Passes 13-16 (Refinement Loop)
        for (int a = 0; a < 4; a++) {
            { wgpu::ComputePassEncoder pass = encoder.BeginComputePass(); pass.SetPipeline(pass13_pipeline); pass.SetBindGroup(0, pass13_bindGroup, 0, nullptr); pass.DispatchWorkgroups(current_partitioned_blocks_num * numCandidates, 1, 1); pass.End(); }
            { wgpu::ComputePassEncoder pass = encoder.BeginComputePass(); pass.SetPipeline(pass14_pipeline); pass.SetBindGroup(0, pass14_bindGroup, 0, nullptr); pass.DispatchWorkgroups(current_partitioned_blocks_num * numCandidates, 1, 1); pass.End(); }
            { wgpu::ComputePassEncoder pass = encoder.BeginComputePass(); pass.SetPipeline(pass15_pipeline); pass.SetBindGroup(0, pass15_bindGroup, 0, nullptr); pass.DispatchWorkgroups(current_partitioned_blocks_num * numCandidates, 1, 1); pass.End(); }
            { wgpu::ComputePassEncoder pass = encoder.BeginComputePass(); pass.SetPipeline(pass16_pipeline); pass.SetBindGroup(0, pass16_bindGroup, 0, nullptr); pass.DispatchWorkgroups(current_partitioned_blocks_num * numCandidates, 1, 1); pass.End(); }
        }

        // Passes 17 and 18
        { wgpu::ComputePassEncoder pass = encoder.BeginComputePass(); pass.SetPipeline(pass17_pipeline); pass.SetBindGroup(0, pass17_bindGroup, 0, nullptr); pass.DispatchWorkgroups(current_partitioned_blocks_num * numCandidates, 1, 1); pass.End(); }
        { wgpu::ComputePassEncoder pass = encoder.BeginComputePass(); pass.SetPipeline(pass18_pipeline); pass.SetBindGroup(0, pass18_bindGroup, 0, nullptr); pass.DispatchWorkgroups(current_partitioned_blocks_num, 1, 1); pass.End(); }


        // Read data from the final output buffer
        encoder.CopyBufferToBuffer(pass18_output_symbolicBlocks, 0, outputReadbackBuffer, 0, current_partitioned_blocks_num * sizeof(SymbolicBlock));
        wgpu::CommandBuffer commands = encoder.Finish();
        queue.Submit(1, &commands);

        std::vector<SymbolicBlock> current_results(current_partitioned_blocks_num);
        mapOutputBufferSync<SymbolicBlock>(device, outputReadbackBuffer, current_partitioned_blocks_num, current_results);


        //Choose the best partitioning candidate for each block & compare to the current best
        for (uint32_t i = 0; i < numBlocks; ++i) {
            int start_offset = block_offsets[i];
            int end_offset = (i + 1 < block_offsets.size()) ? block_offsets[i + 1] : current_partitioned_blocks_num;

            if (start_offset >= end_offset)
                continue; // No candidates for this block

            const SymbolicBlock* best_candidate_for_block = &current_results[start_offset];
            for (int j = start_offset + 1; j < end_offset; ++j) {
                if (current_results[j].errorval < best_candidate_for_block->errorval) {
                    best_candidate_for_block = &current_results[j];
                }
            }

            // Compare its error with the best overall encoding found so far
            if (best_candidate_for_block->errorval < best_symbolic_blocks[i].errorval) {
                best_symbolic_blocks[i] = *best_candidate_for_block;
            }
        }
    }

    // Final physical encoding of the best symbolic blocks found
    std::cout << "Performing final physical encoding..." << std::endl;
    for (uint32_t i = 0; i < numBlocks; i++) {
        int offset = i * 16;
        uint8_t* outputBlock = dataOut + offset;
        symbolic_to_physical(block_descriptor, best_symbolic_blocks[i], outputBlock);
    }

    std::cout << "Encoding complete." << std::endl;
    
    
}

ASTCEncoder::~ASTCEncoder() {
    // Release all WebGPU resources
}