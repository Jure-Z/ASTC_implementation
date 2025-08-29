#include "astc.h"
#include "webgpu_utils.h"

#if !defined(EMSCRIPTEN)
#include <shaders_pass001_init_kmeans_wgsl.h>
#include <shaders_pass002_assign_kmeans_wgsl.h>
#include <shaders_pass003_update_kmeans_wgsl.h>
#include <shaders_pass004_count_partition_mismatch_wgsl.h>
#include <shaders_pass005_partition_ordering_wgsl.h>
#include <shaders_pass006_evaluate_partition_candidates_wgsl.h>
#include <shaders_pass007_prepare_partitioned_blocks_wgsl.h>
#include <shaders_pass01_ideal_endpoints_and_weights_wgsl.h>
#include <shaders_pass02_decimated_weights_wgsl.h>
#include <shaders_pass03_compute_angular_offsets_wgsl.h>
#include <shaders_pass04_lowest_and_highest_weight_wgsl.h>
#include <shaders_pass05_best_values_for_quant_levels_wgsl.h>
#include <shaders_pass06_remap_low_and_high_values_wgsl.h>
#include <shaders_pass07_weights_and_error_for_bm_wgsl.h>
#include <shaders_pass08_compute_encoding_choice_errors_wgsl.h>
#include <shaders_pass09_compute_color_error_wgsl.h>
#include <shaders_pass10_color_combinations_for_quant_2part_wgsl.h>
#include <shaders_pass10_color_combinations_for_quant_3part_wgsl.h>
#include <shaders_pass10_color_combinations_for_quant_4part_wgsl.h>
#include <shaders_pass11_best_color_combination_for_mode_1part_wgsl.h>
#include <shaders_pass11_best_color_combination_for_mode_2part_wgsl.h>
#include <shaders_pass11_best_color_combination_for_mode_3part_wgsl.h>
#include <shaders_pass11_best_color_combination_for_mode_4part_wgsl.h>
#include <shaders_pass12_find_top_N_candidates_wgsl.h>
#include <shaders_pass13_recompute_ideal_endpoints_wgsl.h>
#include <shaders_pass14_pack_color_endpoints_wgsl.h>
#include <shaders_pass15_unpack_color_endpoints_wgsl.h>
#include <shaders_pass16_realign_weights_wgsl.h>
#include <shaders_pass17_compute_final_error_wgsl.h>
#include <shaders_pass18_pick_best_candidate_wgsl.h>
#endif


void ASTCEncoder::initBindGroupLayouts() {

    //bind group layout for pass 001
    std::vector<wgpu::BindGroupLayoutEntry> bg001_entries;
    bg001_entries.push_back({ .binding = 0, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Uniform} }); //Uniforms buffer
    bg001_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Input block buffer
    bg001_entries.push_back({ .binding = 2, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage} }); //Output of pass001 (cluster centers)

    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc001 = {};
    bindGroupLayoutDesc001.entryCount = (uint32_t)bg001_entries.size();
    bindGroupLayoutDesc001.entries = bg001_entries.data();
    pass001_bindGroupLayout = device.CreateBindGroupLayout(&bindGroupLayoutDesc001);

    //bind group layout for pass 002
    std::vector<wgpu::BindGroupLayoutEntry> bg002_entries;
    bg002_entries.push_back({ .binding = 0, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Uniform} }); //Uniforms buffer
    bg002_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Input block buffer
    bg002_entries.push_back({ .binding = 2, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Output of pass001 (cluster centers)
    bg002_entries.push_back({ .binding = 3, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage} }); //Output of pass002 (texel assignments)

    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc002 = {};
    bindGroupLayoutDesc002.entryCount = (uint32_t)bg002_entries.size();
    bindGroupLayoutDesc002.entries = bg002_entries.data();
    pass002_bindGroupLayout = device.CreateBindGroupLayout(&bindGroupLayoutDesc002);


    //bind group layout for pass 003
    std::vector<wgpu::BindGroupLayoutEntry> bg003_entries;
    bg003_entries.push_back({ .binding = 0, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Uniform} }); //Uniforms buffer
    bg003_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Input block buffer
    bg003_entries.push_back({ .binding = 2, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Output of pass002 (texel assignments)
    bg003_entries.push_back({ .binding = 3, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage} }); //Output of pass001 (cluster centers)

    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc003 = {};
    bindGroupLayoutDesc003.entryCount = (uint32_t)bg003_entries.size();
    bindGroupLayoutDesc003.entries = bg003_entries.data();
    pass003_bindGroupLayout = device.CreateBindGroupLayout(&bindGroupLayoutDesc003);


    //bind group layout for pass 004
    std::vector<wgpu::BindGroupLayoutEntry> bg004_entries;
    bg004_entries.push_back({ .binding = 0, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Uniform} }); //Uniforms buffer
    bg004_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //k-means texels buffer
    bg004_entries.push_back({ .binding = 2, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Coverage bitmaps 2 buffer
    bg004_entries.push_back({ .binding = 3, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Coverage bitmaps 3 buffer
    bg004_entries.push_back({ .binding = 4, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Coverage bitmaps 4 buffer
    bg004_entries.push_back({ .binding = 5, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Output of pass002 (texel assignments)
    bg004_entries.push_back({ .binding = 6, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage} }); //Output of pass004 (mismatch counts)

    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc004 = {};
    bindGroupLayoutDesc004.entryCount = (uint32_t)bg004_entries.size();
    bindGroupLayoutDesc004.entries = bg004_entries.data();
    pass004_bindGroupLayout = device.CreateBindGroupLayout(&bindGroupLayoutDesc004);


    //bind group layout for pass 005
    std::vector<wgpu::BindGroupLayoutEntry> bg005_entries;
    bg005_entries.push_back({ .binding = 0, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Uniform} }); //Uniforms buffer
    bg005_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Output of pass004 (mismatch counts)
    bg005_entries.push_back({ .binding = 2, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage} }); //Output of pass005 (partition ordering)

    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc005 = {};
    bindGroupLayoutDesc005.entryCount = (uint32_t)bg005_entries.size();
    bindGroupLayoutDesc005.entries = bg005_entries.data();
    pass005_bindGroupLayout = device.CreateBindGroupLayout(&bindGroupLayoutDesc005);


    //bind group layout for pass 006
    std::vector<wgpu::BindGroupLayoutEntry> bg006_entries;
    bg006_entries.push_back({ .binding = 0, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Uniform} }); //Uniforms buffer
    bg006_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Partition infos buffer
	bg006_entries.push_back({ .binding = 2, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Input block buffer
    bg006_entries.push_back({ .binding = 3, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Output of pass005 (partition ordering)
    bg006_entries.push_back({ .binding = 4, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage} }); //Output of pass006 (final partition errors)

    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc006 = {};
    bindGroupLayoutDesc006.entryCount = (uint32_t)bg006_entries.size();
    bindGroupLayoutDesc006.entries = bg006_entries.data();
    pass006_bindGroupLayout = device.CreateBindGroupLayout(&bindGroupLayoutDesc006);


    //bind group layout for pass 007
    std::vector<wgpu::BindGroupLayoutEntry> bg007_entries;
    bg007_entries.push_back({ .binding = 0, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Uniform} }); //Uniforms buffer
    bg007_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Partition infos buffer
    bg007_entries.push_back({ .binding = 2, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Input block buffer
    bg007_entries.push_back({ .binding = 3, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Output of pass006 (final partition errors)
    bg007_entries.push_back({ .binding = 4, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage} }); //Output of pass007 (partitioned blocks)

    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc007 = {};
    bindGroupLayoutDesc007.entryCount = (uint32_t)bg007_entries.size();
    bindGroupLayoutDesc007.entries = bg007_entries.data();
    pass007_bindGroupLayout = device.CreateBindGroupLayout(&bindGroupLayoutDesc007);

    //bind group layout for pass 1
    std::vector<wgpu::BindGroupLayoutEntry> bg1_entries;
    bg1_entries.push_back({ .binding = 0, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Uniform} }); //Uniforms buffer
    bg1_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Input block buffer
    bg1_entries.push_back({ .binding = 2, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage} }); //Output of pass1

    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc1 = {};
    bindGroupLayoutDesc1.entryCount = (uint32_t)bg1_entries.size();
    bindGroupLayoutDesc1.entries = bg1_entries.data();
    pass1_bindGroupLayout = device.CreateBindGroupLayout(&bindGroupLayoutDesc1);


    //bind group layout for pass 2
    std::vector<wgpu::BindGroupLayoutEntry> bg2_entries;
    bg2_entries.push_back({ .binding = 0, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Uniform} }); //Uniforms Buffer
    bg2_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Valid decimation modes buffer
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
    bg3_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Valid decimation modes buffer
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
    bg4_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Valid decimation modes buffer
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
    bg5_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Valid decimation modes buffer
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
    bg6_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Valid block modes buffer
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
    bg7_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Valid block modes buffer
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
    bg11_1_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Valid block modes buffer
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
    bg11_234_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Valid block modes buffer
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
    bg12_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Valid block modes buffer
    bg12_entries.push_back({ .binding = 2, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Output of pass 1 (ideal endpoints and weights)
    bg12_entries.push_back({ .binding = 3, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //output of pass 7 (quantization results)
    bg12_entries.push_back({ .binding = 4, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Output of pass11 (best endpoint combiantions for mode)
    bg12_entries.push_back({ .binding = 5, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage} }); //Output of pass12 (final candidates)
    bg12_entries.push_back({ .binding = 6, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage} }); //Output of pass12 (top candidates)

    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc12 = {};
    bindGroupLayoutDesc12.entryCount = (uint32_t)bg12_entries.size();
    bindGroupLayoutDesc12.entries = bg12_entries.data();
    pass12_bindGroupLayout = device.CreateBindGroupLayout(&bindGroupLayoutDesc12);

    //bind grup layout for pass 13
    std::vector<wgpu::BindGroupLayoutEntry> bg13_entries;
    bg13_entries.push_back({ .binding = 0, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Uniform} }); //Uniforms Buffer
    bg13_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Decimation infos buffer
    bg13_entries.push_back({ .binding = 2, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Texel to weight map buffer
    bg13_entries.push_back({ .binding = 3, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Block modes buffer
    bg13_entries.push_back({ .binding = 4, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Input blocks buffer
    bg13_entries.push_back({ .binding = 5, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage} }); //Output of pass12 (final candidates)
    bg13_entries.push_back({ .binding = 6, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage} }); //Output of pass13 (rgbs vectors)

    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc13 = {};
    bindGroupLayoutDesc13.entryCount = (uint32_t)bg13_entries.size();
    bindGroupLayoutDesc13.entries = bg13_entries.data();
    pass13_bindGroupLayout = device.CreateBindGroupLayout(&bindGroupLayoutDesc13);

    //bind grup layout for pass 14
    std::vector<wgpu::BindGroupLayoutEntry> bg14_entries;
    bg14_entries.push_back({ .binding = 0, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Uniform} }); //Uniforms Buffer
    bg14_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Output of pass13 (rgbs vectors)
    bg14_entries.push_back({ .binding = 2, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage} }); //Output of pass12 (final candidates)

    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc14 = {};
    bindGroupLayoutDesc14.entryCount = (uint32_t)bg14_entries.size();
    bindGroupLayoutDesc14.entries = bg14_entries.data();
    pass14_bindGroupLayout = device.CreateBindGroupLayout(&bindGroupLayoutDesc14);

    //bind grup layout for pass 15
    std::vector<wgpu::BindGroupLayoutEntry> bg15_entries;
    bg15_entries.push_back({ .binding = 0, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Uniform} }); //Uniforms Buffer
    bg15_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Output of pass12 (final candidates)
    bg15_entries.push_back({ .binding = 2, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage} }); //Output of pass15 (unpacked endpoints)

    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc15 = {};
    bindGroupLayoutDesc15.entryCount = (uint32_t)bg15_entries.size();
    bindGroupLayoutDesc15.entries = bg15_entries.data();
    pass15_bindGroupLayout = device.CreateBindGroupLayout(&bindGroupLayoutDesc15);

    //bind grup layout for pass 16
    std::vector<wgpu::BindGroupLayoutEntry> bg16_entries;
    bg16_entries.push_back({ .binding = 0, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Uniform} }); //Uniforms Buffer
    bg16_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Block modes
    bg16_entries.push_back({ .binding = 2, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Decimation infos
    bg16_entries.push_back({ .binding = 3, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Texel to weight map
    bg16_entries.push_back({ .binding = 4, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Weight to texel map
    bg16_entries.push_back({ .binding = 5, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Input blocks
    bg16_entries.push_back({ .binding = 6, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Output of pass15 (unpacked endpoints)
    bg16_entries.push_back({ .binding = 7, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage} }); //Output of pass12 (final candidates)

    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc16 = {};
    bindGroupLayoutDesc16.entryCount = (uint32_t)bg16_entries.size();
    bindGroupLayoutDesc16.entries = bg16_entries.data();
    pass16_bindGroupLayout = device.CreateBindGroupLayout(&bindGroupLayoutDesc16);

    //bind grup layout for pass 17
    std::vector<wgpu::BindGroupLayoutEntry> bg17_entries;
    bg17_entries.push_back({ .binding = 0, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Uniform} }); //Uniforms Buffer
    bg17_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Block modes
    bg17_entries.push_back({ .binding = 2, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Decimation infos
    bg17_entries.push_back({ .binding = 3, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Texel to weight map
    bg17_entries.push_back({ .binding = 4, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Input blocks
    bg17_entries.push_back({ .binding = 5, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Output of pass15 (unpacked endpoints)
    bg17_entries.push_back({ .binding = 6, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage} }); //Output of pass12 (final candidates)
    bg17_entries.push_back({ .binding = 7, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage} }); //Output of pass12 (top candidates)

    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc17 = {};
    bindGroupLayoutDesc17.entryCount = (uint32_t)bg17_entries.size();
    bindGroupLayoutDesc17.entries = bg17_entries.data();
    pass17_bindGroupLayout = device.CreateBindGroupLayout(&bindGroupLayoutDesc17);

    //bind grup layout for pass 18
    std::vector<wgpu::BindGroupLayoutEntry> bg18_entries;
    bg18_entries.push_back({ .binding = 0, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Uniform} }); //Uniforms Buffer
    bg18_entries.push_back({ .binding = 1, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Input blocks
    bg18_entries.push_back({ .binding = 2, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Output of pass12 (top candidates)
    bg18_entries.push_back({ .binding = 3, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::ReadOnlyStorage} }); //Block modes
    bg18_entries.push_back({ .binding = 4, .visibility = wgpu::ShaderStage::Compute, .buffer = {.type = wgpu::BufferBindingType::Storage} }); //Output of pass18 (symbolic blocks)

    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc18 = {};
    bindGroupLayoutDesc18.entryCount = (uint32_t)bg18_entries.size();
    bindGroupLayoutDesc18.entries = bg18_entries.data();
    pass18_bindGroupLayout = device.CreateBindGroupLayout(&bindGroupLayoutDesc18);
}

#if !defined(EMSCRIPTEN)
void ASTCEncoder::initPipelines() {

    // Load shader modules form embbeded shader code
    pass001_initKmeansShader = prepareShaderModule(device, Shaders::shaders_pass001_init_kmeans_wgsl, Shaders::shaders_pass001_init_kmeans_wgsl_len, "Init k-means (pass001)");
    pass002_assignKmeansShader = prepareShaderModule(device, Shaders::shaders_pass002_assign_kmeans_wgsl, Shaders::shaders_pass002_assign_kmeans_wgsl_len, "Assign k-means (pass002)");
    pass003_updateKmeansShader = prepareShaderModule(device, Shaders::shaders_pass003_update_kmeans_wgsl, Shaders::shaders_pass003_update_kmeans_wgsl_len, "Update k-means (pass003)");
    pass004_partitionMismatchShader = prepareShaderModule(device, Shaders::shaders_pass004_count_partition_mismatch_wgsl, Shaders::shaders_pass004_count_partition_mismatch_wgsl_len, "Count partition mismatch (pass004)");
    pass005_partitionOrderingShader = prepareShaderModule(device, Shaders::shaders_pass005_partition_ordering_wgsl, Shaders::shaders_pass005_partition_ordering_wgsl_len, "Partition ordering (pass005)");
    pass006_evaluatePartitionShader = prepareShaderModule(device, Shaders::shaders_pass006_evaluate_partition_candidates_wgsl, Shaders::shaders_pass006_evaluate_partition_candidates_wgsl_len, "Evaluate partition candidates (pass006)");
	pass007_preparePartitionedBlocksShader = prepareShaderModule(device, Shaders::shaders_pass007_prepare_partitioned_blocks_wgsl, Shaders::shaders_pass007_prepare_partitioned_blocks_wgsl_len, "Prepare partitioned blocks (pass007)");
    pass1_idealEndpointsShader = prepareShaderModule(device, Shaders::shaders_pass01_ideal_endpoints_and_weights_wgsl, Shaders::shaders_pass01_ideal_endpoints_and_weights_wgsl_len, "Ideal endpoints and weights (pass1)");
    pass2_decimatedWeightsShader = prepareShaderModule(device, Shaders::shaders_pass02_decimated_weights_wgsl, Shaders::shaders_pass02_decimated_weights_wgsl_len, "decimated weights (pass2)");
    pass3_angularOffsetsShader = prepareShaderModule(device, Shaders::shaders_pass03_compute_angular_offsets_wgsl, Shaders::shaders_pass03_compute_angular_offsets_wgsl_len, "angular offsets (pass3)");
    pass4_lowestAndHighestWeightShader = prepareShaderModule(device, Shaders::shaders_pass04_lowest_and_highest_weight_wgsl, Shaders::shaders_pass04_lowest_and_highest_weight_wgsl_len, "lowest and highest weight (pass4)");
    pass5_valuesForQuantLevelsShader = prepareShaderModule(device, Shaders::shaders_pass05_best_values_for_quant_levels_wgsl, Shaders::shaders_pass05_best_values_for_quant_levels_wgsl_len, "best values for quant levels (pass5)");
    pass6_remapLowAndHighValuesShader = prepareShaderModule(device, Shaders::shaders_pass06_remap_low_and_high_values_wgsl, Shaders::shaders_pass06_remap_low_and_high_values_wgsl_len, "remap low and high values (pass6)");
    pass7_weightsAndErrorForBMShader = prepareShaderModule(device, Shaders::shaders_pass07_weights_and_error_for_bm_wgsl, Shaders::shaders_pass07_weights_and_error_for_bm_wgsl_len, "weights and error for block mode (pass7)");
    pass8_encodingChoiceErrorsShader = prepareShaderModule(device, Shaders::shaders_pass08_compute_encoding_choice_errors_wgsl, Shaders::shaders_pass08_compute_encoding_choice_errors_wgsl_len, "encoding choice errors (pass8)");
    pass9_computeColorErrorShader = prepareShaderModule(device, Shaders::shaders_pass09_compute_color_error_wgsl, Shaders::shaders_pass09_compute_color_error_wgsl_len, "color format errors (pass9)");
    pass10_colorEndpointCombinationsShader_2part = prepareShaderModule(device, Shaders::shaders_pass10_color_combinations_for_quant_2part_wgsl, Shaders::shaders_pass10_color_combinations_for_quant_2part_wgsl_len, "color endpoint combinations (pass10, 2part)");
    pass10_colorEndpointCombinationsShader_3part = prepareShaderModule(device, Shaders::shaders_pass10_color_combinations_for_quant_3part_wgsl, Shaders::shaders_pass10_color_combinations_for_quant_3part_wgsl_len, "color endpoint combinations (pass10, 3part)");
    pass10_colorEndpointCombinationsShader_4part = prepareShaderModule(device, Shaders::shaders_pass10_color_combinations_for_quant_4part_wgsl, Shaders::shaders_pass10_color_combinations_for_quant_4part_wgsl_len, "color endpoint combinations (pass10, 4part)");
    pass11_bestEndpointCombinationsForModeShader_1part = prepareShaderModule(device, Shaders::shaders_pass11_best_color_combination_for_mode_1part_wgsl, Shaders::shaders_pass11_best_color_combination_for_mode_1part_wgsl_len, "best endpoint combinations for mode (pass11, 1part)");
    pass11_bestEndpointCombinationsForModeShader_2part = prepareShaderModule(device, Shaders::shaders_pass11_best_color_combination_for_mode_2part_wgsl, Shaders::shaders_pass11_best_color_combination_for_mode_2part_wgsl_len, "best endpoint combinations for mode (pass11, 2part)");
    pass11_bestEndpointCombinationsForModeShader_3part = prepareShaderModule(device, Shaders::shaders_pass11_best_color_combination_for_mode_3part_wgsl, Shaders::shaders_pass11_best_color_combination_for_mode_3part_wgsl_len, "best endpoint combinations for mode (pass11, 3part)");
    pass11_bestEndpointCombinationsForModeShader_4part = prepareShaderModule(device, Shaders::shaders_pass11_best_color_combination_for_mode_4part_wgsl, Shaders::shaders_pass11_best_color_combination_for_mode_4part_wgsl_len, "best endpoint combinations for mode (pass11, 4part)");
    pass12_findTopNCandidatesShader = prepareShaderModule(device, Shaders::shaders_pass12_find_top_N_candidates_wgsl, Shaders::shaders_pass12_find_top_N_candidates_wgsl_len, "find top N candidates (pass12)");
    pass13_recomputeIdealEndpointsShader = prepareShaderModule(device, Shaders::shaders_pass13_recompute_ideal_endpoints_wgsl, Shaders::shaders_pass13_recompute_ideal_endpoints_wgsl_len, "recompute ideal endpoints (pass13)");
    pass14_packColorEndpointsShader = prepareShaderModule(device, Shaders::shaders_pass14_pack_color_endpoints_wgsl, Shaders::shaders_pass14_pack_color_endpoints_wgsl_len, "pack color endpoints (pass14)");
    pass15_unpackColorEndpointsShader = prepareShaderModule(device, Shaders::shaders_pass15_unpack_color_endpoints_wgsl, Shaders::shaders_pass15_unpack_color_endpoints_wgsl_len, "unpack color endpoints (pass15)");
    pass16_realignWeightsShader = prepareShaderModule(device, Shaders::shaders_pass16_realign_weights_wgsl, Shaders::shaders_pass16_realign_weights_wgsl_len, "realign weights (pass16)");
    pass17_computeFinalErrorShader = prepareShaderModule(device, Shaders::shaders_pass17_compute_final_error_wgsl, Shaders::shaders_pass17_compute_final_error_wgsl_len, "compute final error (pass17)");
    pass18_pickBestCandidateShader = prepareShaderModule(device, Shaders::shaders_pass18_pick_best_candidate_wgsl, Shaders::shaders_pass18_pick_best_candidate_wgsl_len, "pick best candidate (pass18)");


    if (!pass1_idealEndpointsShader) {
        std::cerr << "FATAL ERROR: Failed to create shader module for pass 1. Check shader file paths." << std::endl;
        // You could even throw an exception here to halt execution
        throw std::runtime_error("Failed to load a critical shader.");
    }

    //pass001 compute pipeline
    wgpu::PipelineLayoutDescriptor pass001_layoutDesc = {};
    pass001_layoutDesc.bindGroupLayoutCount = 1;
    pass001_layoutDesc.bindGroupLayouts = &pass001_bindGroupLayout;
    wgpu::PipelineLayout pass001_pipelineLayout = device.CreatePipelineLayout(&pass001_layoutDesc);

    wgpu::ComputePipelineDescriptor pass001_pipelineDesc = {};
    pass001_pipelineDesc.compute.constantCount = 0;
    pass001_pipelineDesc.compute.constants = nullptr;
    pass001_pipelineDesc.compute.entryPoint = "main";
    pass001_pipelineDesc.compute.module = pass001_initKmeansShader;
    pass001_pipelineDesc.layout = pass001_pipelineLayout;

    pass001_pipeline = device.CreateComputePipeline(&pass001_pipelineDesc);
    

    //pass002 compute pipeline
    wgpu::PipelineLayoutDescriptor pass002_layoutDesc = {};
    pass002_layoutDesc.bindGroupLayoutCount = 1;
    pass002_layoutDesc.bindGroupLayouts = &pass002_bindGroupLayout;
    wgpu::PipelineLayout pass002_pipelineLayout = device.CreatePipelineLayout(&pass002_layoutDesc);

    wgpu::ComputePipelineDescriptor pass002_pipelineDesc = {};
    pass002_pipelineDesc.compute.constantCount = 0;
    pass002_pipelineDesc.compute.constants = nullptr;
    pass002_pipelineDesc.compute.entryPoint = "main";
    pass002_pipelineDesc.compute.module = pass002_assignKmeansShader;
    pass002_pipelineDesc.layout = pass002_pipelineLayout;

    pass002_pipeline = device.CreateComputePipeline(&pass002_pipelineDesc);


    //pass003 compute pipeline
    wgpu::PipelineLayoutDescriptor pass003_layoutDesc = {};
    pass003_layoutDesc.bindGroupLayoutCount = 1;
    pass003_layoutDesc.bindGroupLayouts = &pass003_bindGroupLayout;
    wgpu::PipelineLayout pass003_pipelineLayout = device.CreatePipelineLayout(&pass003_layoutDesc);

    wgpu::ComputePipelineDescriptor pass003_pipelineDesc = {};
    pass003_pipelineDesc.compute.constantCount = 0;
    pass003_pipelineDesc.compute.constants = nullptr;
    pass003_pipelineDesc.compute.entryPoint = "main";
    pass003_pipelineDesc.compute.module = pass003_updateKmeansShader;
    pass003_pipelineDesc.layout = pass003_pipelineLayout;

    pass003_pipeline = device.CreateComputePipeline(&pass003_pipelineDesc);


    //pass004 compute pipeline
    wgpu::PipelineLayoutDescriptor pass004_layoutDesc = {};
    pass004_layoutDesc.bindGroupLayoutCount = 1;
    pass004_layoutDesc.bindGroupLayouts = &pass004_bindGroupLayout;
    wgpu::PipelineLayout pass004_pipelineLayout = device.CreatePipelineLayout(&pass004_layoutDesc);

    wgpu::ComputePipelineDescriptor pass004_pipelineDesc = {};
    pass004_pipelineDesc.compute.constantCount = 0;
    pass004_pipelineDesc.compute.constants = nullptr;
    pass004_pipelineDesc.compute.entryPoint = "main";
    pass004_pipelineDesc.compute.module = pass004_partitionMismatchShader;
    pass004_pipelineDesc.layout = pass004_pipelineLayout;

    pass004_pipeline = device.CreateComputePipeline(&pass004_pipelineDesc);


    //pass005 compute pipeline
    wgpu::PipelineLayoutDescriptor pass005_layoutDesc = {};
    pass005_layoutDesc.bindGroupLayoutCount = 1;
    pass005_layoutDesc.bindGroupLayouts = &pass005_bindGroupLayout;
    wgpu::PipelineLayout pass005_pipelineLayout = device.CreatePipelineLayout(&pass005_layoutDesc);

    wgpu::ComputePipelineDescriptor pass005_pipelineDesc = {};
    pass005_pipelineDesc.compute.constantCount = 0;
    pass005_pipelineDesc.compute.constants = nullptr;
    pass005_pipelineDesc.compute.entryPoint = "main";
    pass005_pipelineDesc.compute.module = pass005_partitionOrderingShader;
    pass005_pipelineDesc.layout = pass005_pipelineLayout;

    pass005_pipeline = device.CreateComputePipeline(&pass005_pipelineDesc);


    //pass006 compute pipeline
    wgpu::PipelineLayoutDescriptor pass006_layoutDesc = {};
    pass006_layoutDesc.bindGroupLayoutCount = 1;
    pass006_layoutDesc.bindGroupLayouts = &pass006_bindGroupLayout;
    wgpu::PipelineLayout pass006_pipelineLayout = device.CreatePipelineLayout(&pass006_layoutDesc);

    wgpu::ComputePipelineDescriptor pass006_pipelineDesc = {};
    pass006_pipelineDesc.compute.constantCount = 0;
    pass006_pipelineDesc.compute.constants = nullptr;
    pass006_pipelineDesc.compute.entryPoint = "main";
    pass006_pipelineDesc.compute.module = pass006_evaluatePartitionShader;
    pass006_pipelineDesc.layout = pass006_pipelineLayout;

    pass006_pipeline = device.CreateComputePipeline(&pass006_pipelineDesc);


    //pass007 compute pipeline
    wgpu::PipelineLayoutDescriptor pass007_layoutDesc = {};
    pass007_layoutDesc.bindGroupLayoutCount = 1;
    pass007_layoutDesc.bindGroupLayouts = &pass007_bindGroupLayout;
    wgpu::PipelineLayout pass007_pipelineLayout = device.CreatePipelineLayout(&pass007_layoutDesc);

    wgpu::ComputePipelineDescriptor pass007_pipelineDesc = {};
    pass007_pipelineDesc.compute.constantCount = 0;
    pass007_pipelineDesc.compute.constants = nullptr;
    pass007_pipelineDesc.compute.entryPoint = "main";
	pass007_pipelineDesc.compute.module = pass007_preparePartitionedBlocksShader;
    pass007_pipelineDesc.layout = pass007_pipelineLayout;

    pass007_pipeline = device.CreateComputePipeline(&pass007_pipelineDesc);


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

    //pass13 compute pipeline
    wgpu::PipelineLayoutDescriptor pass13_layoutDesc = {};
    pass13_layoutDesc.bindGroupLayoutCount = 1;
    pass13_layoutDesc.bindGroupLayouts = &pass13_bindGroupLayout;
    wgpu::PipelineLayout pass13_pipelineLayout = device.CreatePipelineLayout(&pass13_layoutDesc);

    wgpu::ComputePipelineDescriptor pass13_pipelineDesc = {};
    pass13_pipelineDesc.compute.constantCount = 0;
    pass13_pipelineDesc.compute.constants = nullptr;
    pass13_pipelineDesc.compute.entryPoint = "main";
    pass13_pipelineDesc.compute.module = pass13_recomputeIdealEndpointsShader;
    pass13_pipelineDesc.layout = pass13_pipelineLayout;

    pass13_pipeline = device.CreateComputePipeline(&pass13_pipelineDesc);

    //pass14 compute pipeline
    wgpu::PipelineLayoutDescriptor pass14_layoutDesc = {};
    pass14_layoutDesc.bindGroupLayoutCount = 1;
    pass14_layoutDesc.bindGroupLayouts = &pass14_bindGroupLayout;
    wgpu::PipelineLayout pass14_pipelineLayout = device.CreatePipelineLayout(&pass14_layoutDesc);

    wgpu::ComputePipelineDescriptor pass14_pipelineDesc = {};
    pass14_pipelineDesc.compute.constantCount = 0;
    pass14_pipelineDesc.compute.constants = nullptr;
    pass14_pipelineDesc.compute.entryPoint = "main";
    pass14_pipelineDesc.compute.module = pass14_packColorEndpointsShader;
    pass14_pipelineDesc.layout = pass14_pipelineLayout;

    pass14_pipeline = device.CreateComputePipeline(&pass14_pipelineDesc);

    //pass15 compute pipeline
    wgpu::PipelineLayoutDescriptor pass15_layoutDesc = {};
    pass15_layoutDesc.bindGroupLayoutCount = 1;
    pass15_layoutDesc.bindGroupLayouts = &pass15_bindGroupLayout;
    wgpu::PipelineLayout pass15_pipelineLayout = device.CreatePipelineLayout(&pass15_layoutDesc);

    wgpu::ComputePipelineDescriptor pass15_pipelineDesc = {};
    pass15_pipelineDesc.compute.constantCount = 0;
    pass15_pipelineDesc.compute.constants = nullptr;
    pass15_pipelineDesc.compute.entryPoint = "main";
    pass15_pipelineDesc.compute.module = pass15_unpackColorEndpointsShader;
    pass15_pipelineDesc.layout = pass15_pipelineLayout;

    pass15_pipeline = device.CreateComputePipeline(&pass15_pipelineDesc);

    //pass16 compute pipeline
    wgpu::PipelineLayoutDescriptor pass16_layoutDesc = {};
    pass16_layoutDesc.bindGroupLayoutCount = 1;
    pass16_layoutDesc.bindGroupLayouts = &pass16_bindGroupLayout;
    wgpu::PipelineLayout pass16_pipelineLayout = device.CreatePipelineLayout(&pass16_layoutDesc);

    wgpu::ComputePipelineDescriptor pass16_pipelineDesc = {};
    pass16_pipelineDesc.compute.constantCount = 0;
    pass16_pipelineDesc.compute.constants = nullptr;
    pass16_pipelineDesc.compute.entryPoint = "main";
    pass16_pipelineDesc.compute.module = pass16_realignWeightsShader;
    pass16_pipelineDesc.layout = pass16_pipelineLayout;

    pass16_pipeline = device.CreateComputePipeline(&pass16_pipelineDesc);

    //pass17 compute pipeline
    wgpu::PipelineLayoutDescriptor pass17_layoutDesc = {};
    pass17_layoutDesc.bindGroupLayoutCount = 1;
    pass17_layoutDesc.bindGroupLayouts = &pass17_bindGroupLayout;
    wgpu::PipelineLayout pass17_pipelineLayout = device.CreatePipelineLayout(&pass17_layoutDesc);

    wgpu::ComputePipelineDescriptor pass17_pipelineDesc = {};
    pass17_pipelineDesc.compute.constantCount = 0;
    pass17_pipelineDesc.compute.constants = nullptr;
    pass17_pipelineDesc.compute.entryPoint = "main";
    pass17_pipelineDesc.compute.module = pass17_computeFinalErrorShader;
    pass17_pipelineDesc.layout = pass17_pipelineLayout;

    pass17_pipeline = device.CreateComputePipeline(&pass17_pipelineDesc);

    //pass18 compute pipeline
    wgpu::PipelineLayoutDescriptor pass18_layoutDesc = {};
    pass18_layoutDesc.bindGroupLayoutCount = 1;
    pass18_layoutDesc.bindGroupLayouts = &pass18_bindGroupLayout;
    wgpu::PipelineLayout pass18_pipelineLayout = device.CreatePipelineLayout(&pass18_layoutDesc);

    wgpu::ComputePipelineDescriptor pass18_pipelineDesc = {};
    pass18_pipelineDesc.compute.constantCount = 0;
    pass18_pipelineDesc.compute.constants = nullptr;
    pass18_pipelineDesc.compute.entryPoint = "main";
    pass18_pipelineDesc.compute.module = pass18_pickBestCandidateShader;
    pass18_pipelineDesc.layout = pass18_pipelineLayout;

    pass18_pipeline = device.CreateComputePipeline(&pass18_pipelineDesc);
}
#endif

#if defined(EMSCRIPTEN)
void ASTCEncoder::initPipelinesAsync(std::function<void()> on_all_pipelines_created) {
    m_pipeline_build_queue = {
        {&pass001_initKmeansShader, "/shaders/pass001_init_kmeans.wgsl", "Init k-means (pass001)", &pass001_pipeline, &pass001_bindGroupLayout},
        {&pass002_assignKmeansShader, "/shaders/pass002_assign_kmeans.wgsl", "Assign k-means (pass002)", &pass002_pipeline, &pass002_bindGroupLayout},
        {&pass003_updateKmeansShader, "/shaders/pass003_update_kmeans.wgsl", "Update k-means (pass003)", &pass003_pipeline, &pass003_bindGroupLayout},
        {&pass004_partitionMismatchShader, "/shaders/pass004_count_partition_mismatch.wgsl", "Count partition mismatch (pass004)", &pass004_pipeline, &pass004_bindGroupLayout},
        {&pass005_partitionOrderingShader, "/shaders/pass005_partition_ordering.wgsl", "Partition ordering (pass005)", &pass005_pipeline, &pass005_bindGroupLayout},
        {&pass006_evaluatePartitionShader, "/shaders/pass006_evaluate_partition_candidates.wgsl", "Evaluate partition candidates (pass006)", &pass006_pipeline, &pass006_bindGroupLayout},
        {&pass007_preparePartitionedBlocksShader, "/shaders/pass007_prepare_partitioned_blocks.wgsl", "Prepare partitioned blocks (pass007)", &pass007_pipeline, &pass007_bindGroupLayout},
        {&pass1_idealEndpointsShader, "/shaders/pass01_ideal_endpoints_and_weights.wgsl", "Ideal endpoints and weights (pass1)", &pass1_pipeline, &pass1_bindGroupLayout},
        {&pass2_decimatedWeightsShader, "/shaders/pass02_decimated_weights.wgsl", "decimated weights (pass2)", &pass2_pipeline, &pass2_bindGroupLayout},
        {&pass3_angularOffsetsShader, "/shaders/pass03_compute_angular_offsets.wgsl", "angular offsets (pass3)", &pass3_pipeline, &pass3_bindGroupLayout},
        {&pass4_lowestAndHighestWeightShader, "/shaders/pass04_lowest_and_highest_weight.wgsl", "lowest and highest weight (pass4)", &pass4_pipeline, &pass4_bindGroupLayout},
        {&pass5_valuesForQuantLevelsShader, "/shaders/pass05_best_values_for_quant_levels.wgsl", "best values for quant levels (pass5)", &pass5_pipeline, &pass5_bindGroupLayout},
        {&pass6_remapLowAndHighValuesShader, "/shaders/pass06_remap_low_and_high_values.wgsl", "remap low and high values (pass6)", &pass6_pipeline, &pass6_bindGroupLayout},
        {&pass7_weightsAndErrorForBMShader, "/shaders/pass07_weights_and_error_for_bm.wgsl", "weights and error for block mode (pass7)", &pass7_pipeline, &pass7_bindGroupLayout},
        {&pass8_encodingChoiceErrorsShader, "/shaders/pass08_compute_encoding_choice_errors.wgsl", "encoding choice errors (pass8)", &pass8_pipeline, &pass8_bindGroupLayout},
        {&pass9_computeColorErrorShader, "/shaders/pass09_compute_color_error.wgsl", "color format errors (pass9)", &pass9_pipeline, &pass9_bindGroupLayout},
        {&pass10_colorEndpointCombinationsShader_2part, "/shaders/pass10_color_combinations_for_quant_2part.wgsl", "color endpoint combinations (pass10, 2part)", &pass10_pipeline_2part, &pass10_bindGroupLayout},
        {&pass10_colorEndpointCombinationsShader_3part, "/shaders/pass10_color_combinations_for_quant_3part.wgsl", "color endpoint combinations (pass10, 3part)", &pass10_pipeline_3part, &pass10_bindGroupLayout},
        {&pass10_colorEndpointCombinationsShader_4part, "/shaders/pass10_color_combinations_for_quant_4part.wgsl", "color endpoint combinations (pass10, 4part)", &pass10_pipeline_4part, &pass10_bindGroupLayout},
        {&pass11_bestEndpointCombinationsForModeShader_1part, "/shaders/pass11_best_color_combination_for_mode_1part.wgsl", "best endpoint combinations for mode (pass11, 1part)", &pass11_pipeline_1part, &pass11_bindGroupLayout_1part},
        {&pass11_bestEndpointCombinationsForModeShader_2part, "/shaders/pass11_best_color_combination_for_mode_2part.wgsl", "best endpoint combinations for mode (pass11, 2part)", &pass11_pipeline_2part, &pass11_bindGroupLayout_234part},
        {&pass11_bestEndpointCombinationsForModeShader_3part, "/shaders/pass11_best_color_combination_for_mode_3part.wgsl", "best endpoint combinations for mode (pass11, 3part)", &pass11_pipeline_3part, &pass11_bindGroupLayout_234part},
        {&pass11_bestEndpointCombinationsForModeShader_4part, "/shaders/pass11_best_color_combination_for_mode_4part.wgsl", "best endpoint combinations for mode (pass11, 4part)", &pass11_pipeline_4part, &pass11_bindGroupLayout_234part},
        {&pass12_findTopNCandidatesShader, "/shaders/pass12_find_top_N_candidates.wgsl", "find top N candidates (pass12)", &pass12_pipeline, &pass12_bindGroupLayout},
        {&pass13_recomputeIdealEndpointsShader, "/shaders/pass13_recompute_ideal_endpoints.wgsl", "recompute ideal endpoints (pass13)", &pass13_pipeline, &pass13_bindGroupLayout},
        {&pass14_packColorEndpointsShader, "/shaders/pass14_pack_color_endpoints.wgsl", "pack color endpoints (pass14)", &pass14_pipeline, &pass14_bindGroupLayout},
        {&pass15_unpackColorEndpointsShader, "/shaders/pass15_unpack_color_endpoints.wgsl", "unpack color endpoints (pass15)", &pass15_pipeline, &pass15_bindGroupLayout},
        {&pass16_realignWeightsShader, "/shaders/pass16_realign_weights.wgsl", "realign weights (pass16)", &pass16_pipeline, &pass16_bindGroupLayout},
        {&pass17_computeFinalErrorShader, "/shaders/pass17_compute_final_error.wgsl", "compute final error (pass17)", &pass17_pipeline, &pass17_bindGroupLayout},
        {&pass18_pickBestCandidateShader, "/shaders/pass18_pick_best_candidate.wgsl", "pick best candidate (pass18)", &pass18_pipeline, &pass18_bindGroupLayout},
    };

    m_current_pipeline_index = 0;
    createNextPipeline(on_all_pipelines_created);
}

void ASTCEncoder::OnPipelineCreated(WGPUCreatePipelineAsyncStatus status, WGPUComputePipeline pipeline, char const* message, void* userdata) {

    auto* context = static_cast<PipelineCreationContext*>(userdata);

    if (status != WGPUCreatePipelineAsyncStatus_Success) {
        std::cerr << "FATAL: Failed to create compute pipeline: " << message << std::endl;
    }
    else {
        *context->targetPipeline = wgpu::ComputePipeline::Acquire(pipeline);
    }

    ASTCEncoder* encoder = context->encoderInstance;
    encoder->m_current_pipeline_index++;

    emscripten_sleep(0);

    encoder->createNextPipeline(context->onAllPipelinesCreated);

    delete context;
}

void ASTCEncoder::createNextPipeline(std::function<void()> on_all_pipelines_created) {
    if (m_current_pipeline_index >= m_pipeline_build_queue.size()) {
        if (on_all_pipelines_created) {
            on_all_pipelines_created();
        }
        return;
    }

    const auto& info = m_pipeline_build_queue[m_current_pipeline_index];

    std::cout << "Creating pipeline " << (m_current_pipeline_index + 1) << "/" << m_pipeline_build_queue.size() << ": " << info.shaderLabel << std::endl;

    *info.shaderModule = prepareShaderModule(device, info.shaderPath, info.shaderLabel.c_str());

    wgpu::PipelineLayoutDescriptor layoutDesc = {};
    layoutDesc.bindGroupLayoutCount = 1;
    layoutDesc.bindGroupLayouts = info.bindGroupLayout;
    wgpu::PipelineLayout pipelineLayout = device.CreatePipelineLayout(&layoutDesc);

    wgpu::ComputePipelineDescriptor pipelineDesc = {};
    pipelineDesc.layout = pipelineLayout;
    pipelineDesc.compute.module = *info.shaderModule;
    pipelineDesc.compute.entryPoint = "main";


    auto* context = new PipelineCreationContext{
        this,
        info.targetPipeline,
        on_all_pipelines_created
    };

    device.CreateComputePipelineAsync(&pipelineDesc, &ASTCEncoder::OnPipelineCreated, context);
}
#endif

void ASTCEncoder::initBuffers() {

    int max_partitioned_blocks = batchSize * TUNE_MAX_PARTITIONING_CANDIDATES;
    int max_decimation_mode_trials = max_partitioned_blocks * valid_decimation_modes.size();
    int max_block_mode_trials = max_partitioned_blocks * valid_block_modes.size();

    //Buffer for uniform variables
    wgpu::BufferDescriptor uniformDesc;
    uniformDesc.size = sizeof(uniform_variables);
    uniformDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    uniformsBuffer = device.CreateBuffer(&uniformDesc);

    //K-means texels
    wgpu::BufferDescriptor kmeansDesc;
    kmeansDesc.size = BLOCK_MAX_KMEANS_TEXELS * sizeof(uint32_t);
    kmeansDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    kmeansTexelsBuffer = device.CreateBuffer(&kmeansDesc);

    //Coverage bitmaps for 2 partitions
    wgpu::BufferDescriptor covarageBitmaps2Desc;
    covarageBitmaps2Desc.size = BLOCK_MAX_PARTITIONINGS * 2 * sizeof(uint64_t);
    covarageBitmaps2Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    coverageBitmaps2Buffer = device.CreateBuffer(&covarageBitmaps2Desc);

    //Coverage bitmaps for 3 partitions
    wgpu::BufferDescriptor covarageBitmaps3Desc;
    covarageBitmaps3Desc.size = BLOCK_MAX_PARTITIONINGS * 3 * sizeof(uint64_t);
    covarageBitmaps3Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    coverageBitmaps3Buffer = device.CreateBuffer(&covarageBitmaps3Desc);

    //Coverage bitmaps for 4 partitions
    wgpu::BufferDescriptor covarageBitmaps4Desc;
    covarageBitmaps4Desc.size = BLOCK_MAX_PARTITIONINGS * 4 * sizeof(uint64_t);
    covarageBitmaps4Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    coverageBitmaps4Buffer = device.CreateBuffer(&covarageBitmaps4Desc);

    //Buffer for partition infos
    wgpu::BufferDescriptor partitionInfoDesc;
    partitionInfoDesc.size = ((3 * BLOCK_MAX_PARTITIONINGS) + 1) * sizeof(partition_info_GPU);
    partitionInfoDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    partitionInfoBuffer = device.CreateBuffer(&partitionInfoDesc);

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

    //Buffer for deciamtion mode indeces consirered during compression
    wgpu::BufferDescriptor validDecModesDesc;
    validDecModesDesc.size = valid_decimation_modes.size() * sizeof(uint32_t);
    validDecModesDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    validDecimationModesBuffer = device.CreateBuffer(&validDecModesDesc);

    //Buffer for block mode indeces consirered during compression
    wgpu::BufferDescriptor validBlockModesDesc;
    validBlockModesDesc.size = valid_block_modes.size() * sizeof(PackedBlockModeLookup);
    validBlockModesDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    validBlockModesBuffer = device.CreateBuffer(&validBlockModesDesc);

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

    //Buffer for input blocks
    wgpu::BufferDescriptor inputDesc;
    inputDesc.size = batchSize * sizeof(InputBlock);
    inputDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    inputBlocksBuffer = device.CreateBuffer(&inputDesc);

    //Output buffer of pass 001 (cluster centers)
    wgpu::BufferDescriptor pass001Desc = {};
    pass001Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
	pass001Desc.size = batchSize * 4 * 4 * sizeof(float); //4 cluster centers, each with RGBA channels
    pass001_output_clusterCenters = device.CreateBuffer(&pass001Desc);

    //Output buffer of pass 002 (texel assignments)
    wgpu::BufferDescriptor pass002Desc = {};
    pass002Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    pass002Desc.size = batchSize * BLOCK_MAX_TEXELS * sizeof(uint32_t);
    pass002_output_texelAssignments = device.CreateBuffer(&pass002Desc);

    //Output buffer of pass 004 (mismatch counts)
    wgpu::BufferDescriptor pass004Desc = {};
    pass004Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    pass004Desc.size = batchSize * BLOCK_MAX_PARTITIONINGS * sizeof(uint32_t);
    pass004_output_mismatchCounts = device.CreateBuffer(&pass004Desc);

    //Output buffer of pass 005 (partition ordering)
    wgpu::BufferDescriptor pass005Desc = {};
    pass005Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    pass005Desc.size = batchSize * TUNE_MAX_PARTITIONING_CANDIDATE_LIMIT * sizeof(uint32_t);
    pass005_output_partitionOrdering = device.CreateBuffer(&pass005Desc);

    //Output buffer of pass 006 (final partitioning errors)
    wgpu::BufferDescriptor pass006Desc = {};
    pass006Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    pass006Desc.size = batchSize * TUNE_MAX_PARTITIONING_CANDIDATE_LIMIT * 2 * sizeof(uint32_t);
    pass006_output_partitioningErrors = device.CreateBuffer(&pass006Desc);

    //Buffer for partitioned blocks
    wgpu::BufferDescriptor partBlocksDesc;
    partBlocksDesc.size = max_partitioned_blocks * sizeof(InputBlock);
    partBlocksDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc;
    partitionedBlocksBuffer = device.CreateBuffer(&partBlocksDesc);

    //Output buffer of pass 1 (ideal endpoints and weights)
    wgpu::BufferDescriptor pass1Desc = {};
    pass1Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    pass1Desc.size = max_partitioned_blocks * sizeof(IdealEndpointsAndWeights);
    pass1_output_idealEndpointsAndWeights = device.CreateBuffer(&pass1Desc);

    //Output buffer of pass 2 (decimated weights)
    //indexing pattern: decimation_mode_trial_index * BLOCK_MAX_WEIGHTS + weight_index
    wgpu::BufferDescriptor pass2Desc = {};
    pass2Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    pass2Desc.size = max_decimation_mode_trials * BLOCK_MAX_WEIGHTS * sizeof(float);
    pass2_output_decimatedWeights = device.CreateBuffer(&pass2Desc);

    //Output buffer of pass 3 (angular offsets)
    //indexing pattern: decimation_mode_trial_index * ANGULAR_STEPS + angular_offset_index
    wgpu::BufferDescriptor pass3Desc = {};
    pass3Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    pass3Desc.size = max_decimation_mode_trials * ANGULAR_STEPS * sizeof(float);
    pass3_output_angular_offsets = device.CreateBuffer(&pass3Desc);

    //Output buffer of pass 4 (lowest and highest weights)
    wgpu::BufferDescriptor pass4Desc = {};
    pass4Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    pass4Desc.size = max_decimation_mode_trials * ANGULAR_STEPS * sizeof(HighestAndLowestWeight);
    pass4_output_lowestAndHighestWeight = device.CreateBuffer(&pass4Desc);

    //Output buffer of pass 5 (low values)
    wgpu::BufferDescriptor pass5_1Desc = {};
    pass5_1Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    pass5_1Desc.size = max_decimation_mode_trials * (MAX_ANGULAR_QUANT + 1) * sizeof(float);
    pass5_output_lowValues = device.CreateBuffer(&pass5_1Desc);

    //Output buffer of pass 5 (high values)
    wgpu::BufferDescriptor pass5_2Desc = {};
    pass5_2Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    pass5_2Desc.size = max_decimation_mode_trials * (MAX_ANGULAR_QUANT + 1) * sizeof(float);
    pass5_output_highValues = device.CreateBuffer(&pass5_2Desc);

    //Output buffer of pass 6 (final value ranges)
    wgpu::BufferDescriptor pass6Desc = {};
    pass6Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    pass6Desc.size = max_block_mode_trials * sizeof(FinalValueRange);
    pass6_output_finalValueRanges = device.CreateBuffer(&pass6Desc);

    //Output buffer of pass 7 (quantization results)
    wgpu::BufferDescriptor pass7Desc = {};
    pass7Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    pass7Desc.size = max_block_mode_trials * sizeof(QuantizationResult);
    pass7_output_quantizationResults = device.CreateBuffer(&pass7Desc);

    //Output buffer of pass 8 (encoding choice errors)
    wgpu::BufferDescriptor pass8Desc = {};
    pass8Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    pass8Desc.size = max_partitioned_blocks * BLOCK_MAX_PARTITIONS * sizeof(EncodingChoiceErrors);
    pass8_output_encodingChoiceErrors = device.CreateBuffer(&pass8Desc);

    //Output buffer of pass 9 (color format errors)
    //indexing pattern: ((block_index * BLOCK_MAX_PARTITIONS + partition_index) * QUANT_LEVELS + quant_level_index) * NUM_INT_COUNTS + integer_count
    wgpu::BufferDescriptor pass9_1Desc = {};
    pass9_1Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    pass9_1Desc.size = max_partitioned_blocks * BLOCK_MAX_PARTITIONS * QUANT_LEVELS * NUM_INT_COUNTS * sizeof(float);
    pass9_output_colorFormatErrors = device.CreateBuffer(&pass9_1Desc);

    //Output buffer of pass 9 (color formats)
    //indexing pattern: ((block_index * BLOCK_MAX_PARTITIONS + partition_index) * QUANT_LEVELS + quant_level_index) * NUM_INT_COUNTS + integer_count
    wgpu::BufferDescriptor pass9_2Desc = {};
    pass9_2Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    pass9_2Desc.size = max_partitioned_blocks * BLOCK_MAX_PARTITIONS * QUANT_LEVELS * NUM_INT_COUNTS * sizeof(uint32_t);
    pass9_output_colorFormats = device.CreateBuffer(&pass9_2Desc);

    //Output buffer of pass 10 (color format combinations)
    //indexing pattern: (block_index * QUANT_LEVELS + quant_level) * MAX_INT_COUNT_COMBINATIONS + integer_count
    wgpu::BufferDescriptor pass10Desc = {};
    pass10Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    pass10Desc.size = max_partitioned_blocks * QUANT_LEVELS * MAX_INT_COUNT_COMBINATIONS * sizeof(CombinedEndpointFormats);
    pass10_output_colorEndpointCombinations = device.CreateBuffer(&pass10Desc);

    //Output buffer of pass 11 (best endpoint combinations for mode)
    wgpu::BufferDescriptor pass11Desc = {};
    pass11Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    pass11Desc.size = max_block_mode_trials * sizeof(ColorCombinationResult);
    pass11_output_bestEndpointCombinationsForMode = device.CreateBuffer(&pass11Desc);

    //Output buffer of pass 12 (final candidates)
    //indexing pattern: (block_index * block_descriptor.uniform_variables.tune_candidate_limit + i-th best candidate)
    wgpu::BufferDescriptor pass12Desc = {};
    pass12Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    pass12Desc.size = max_partitioned_blocks * TUNE_MAX_TRIAL_CANDIDATES * sizeof(FinalCandidate);
    pass12_output_finalCandidates = device.CreateBuffer(&pass12Desc);

    //Output buffer of pass 12 (best iteration of each final candidate)
    //indexing pattern: (block_index * block_descriptor.uniform_variables.tune_candidate_limit + i-th best candidate)
    wgpu::BufferDescriptor pass12Desc1 = {};
    pass12Desc1.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    pass12Desc1.size = max_partitioned_blocks * TUNE_MAX_TRIAL_CANDIDATES * sizeof(FinalCandidate);
    pass12_output_topCandidates = device.CreateBuffer(&pass12Desc1);

    //Output buffer of pass 13 (recomputed ideal endpoints)
    wgpu::BufferDescriptor pass13Desc = {};
    pass13Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    pass13Desc.size = max_partitioned_blocks * TUNE_MAX_TRIAL_CANDIDATES * BLOCK_MAX_PARTITIONS * 4 * sizeof(float);
    pass13_output_rgbsVectors = device.CreateBuffer(&pass13Desc);

    //Output buffer of pass 15 (unpacked endpoints)
    wgpu::BufferDescriptor pass15Desc = {};
    pass15Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    pass15Desc.size = max_partitioned_blocks * TUNE_MAX_TRIAL_CANDIDATES * sizeof(UnpackedEndpoints);
    pass15_output_unpackedEndpoints = device.CreateBuffer(&pass15Desc);

    //Output buffer of pass 18 (symbolic blocks)
    wgpu::BufferDescriptor pass18Desc = {};
    pass18Desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    pass18Desc.size = max_partitioned_blocks * sizeof(SymbolicBlock);
    pass18_output_symbolicBlocks = device.CreateBuffer(&pass18Desc);

    //Readback buffer for final output
    wgpu::BufferDescriptor outputDesc = {};
    outputDesc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    outputDesc.size = max_partitioned_blocks * sizeof(SymbolicBlock);
    outputReadbackBuffer = device.CreateBuffer(&outputDesc);

    //Readback buffer for debugging
    wgpu::BufferDescriptor outputDesc1 = {};
    outputDesc1.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    outputDesc1.size = batchSize * 8 * sizeof(InputBlock);
    pass1111ReadbackBuffer = device.CreateBuffer(&outputDesc1);

}

void ASTCEncoder::initBindGroups() {

    //bind group for pass001 (init k-means)
    std::vector<wgpu::BindGroupEntry> bg001_entries;
    bg001_entries.push_back({ .binding = 0, .buffer = uniformsBuffer, .offset = 0, .size = uniformsBuffer.GetSize() });
    bg001_entries.push_back({ .binding = 1, .buffer = inputBlocksBuffer, .offset = 0, .size = inputBlocksBuffer.GetSize() });
    bg001_entries.push_back({ .binding = 2, .buffer = pass001_output_clusterCenters, .offset = 0, .size = pass001_output_clusterCenters.GetSize() });

    wgpu::BindGroupDescriptor bg001_desc = {};
    bg001_desc.layout = pass001_bindGroupLayout;
    bg001_desc.entryCount = bg001_entries.size();
    bg001_desc.entries = bg001_entries.data();
    pass001_bindGroup = device.CreateBindGroup(&bg001_desc);


    //bind group for pass002 (assign k-means)
    std::vector<wgpu::BindGroupEntry> bg002_entries;
    bg002_entries.push_back({ .binding = 0, .buffer = uniformsBuffer, .offset = 0, .size = uniformsBuffer.GetSize() });
    bg002_entries.push_back({ .binding = 1, .buffer = inputBlocksBuffer, .offset = 0, .size = inputBlocksBuffer.GetSize() });
    bg002_entries.push_back({ .binding = 2, .buffer = pass001_output_clusterCenters, .offset = 0, .size = pass001_output_clusterCenters.GetSize() });
    bg002_entries.push_back({ .binding = 3, .buffer = pass002_output_texelAssignments, .offset = 0, .size = pass002_output_texelAssignments.GetSize() });

    wgpu::BindGroupDescriptor bg002_desc = {};
    bg002_desc.layout = pass002_bindGroupLayout;
    bg002_desc.entryCount = bg002_entries.size();
    bg002_desc.entries = bg002_entries.data();
    pass002_bindGroup = device.CreateBindGroup(&bg002_desc);


    //bind group for pass003 (update k-means)
    std::vector<wgpu::BindGroupEntry> bg003_entries;
    bg003_entries.push_back({ .binding = 0, .buffer = uniformsBuffer, .offset = 0, .size = uniformsBuffer.GetSize() });
    bg003_entries.push_back({ .binding = 1, .buffer = inputBlocksBuffer, .offset = 0, .size = inputBlocksBuffer.GetSize() });
    bg003_entries.push_back({ .binding = 2, .buffer = pass002_output_texelAssignments, .offset = 0, .size = pass002_output_texelAssignments.GetSize() });
    bg003_entries.push_back({ .binding = 3, .buffer = pass001_output_clusterCenters, .offset = 0, .size = pass001_output_clusterCenters.GetSize() });

    wgpu::BindGroupDescriptor bg003_desc = {};
    bg003_desc.layout = pass003_bindGroupLayout;
    bg003_desc.entryCount = bg003_entries.size();
    bg003_desc.entries = bg003_entries.data();
    pass003_bindGroup = device.CreateBindGroup(&bg003_desc);


    //bind group for pass004 (count partition mismatch)
    std::vector<wgpu::BindGroupEntry> bg004_entries;
    bg004_entries.push_back({ .binding = 0, .buffer = uniformsBuffer, .offset = 0, .size = uniformsBuffer.GetSize() });
    bg004_entries.push_back({ .binding = 1, .buffer = kmeansTexelsBuffer, .offset = 0, .size = kmeansTexelsBuffer.GetSize() });
    bg004_entries.push_back({ .binding = 2, .buffer = coverageBitmaps2Buffer, .offset = 0, .size = coverageBitmaps2Buffer.GetSize() });
    bg004_entries.push_back({ .binding = 3, .buffer = coverageBitmaps3Buffer, .offset = 0, .size = coverageBitmaps3Buffer.GetSize() });
    bg004_entries.push_back({ .binding = 4, .buffer = coverageBitmaps4Buffer, .offset = 0, .size = coverageBitmaps4Buffer.GetSize() });
    bg004_entries.push_back({ .binding = 5, .buffer = pass002_output_texelAssignments, .offset = 0, .size = pass002_output_texelAssignments.GetSize() });
    bg004_entries.push_back({ .binding = 6, .buffer = pass004_output_mismatchCounts, .offset = 0, .size = pass004_output_mismatchCounts.GetSize() });

    wgpu::BindGroupDescriptor bg004_desc = {};
    bg004_desc.layout = pass004_bindGroupLayout;
    bg004_desc.entryCount = bg004_entries.size();
    bg004_desc.entries = bg004_entries.data();
    pass004_bindGroup = device.CreateBindGroup(&bg004_desc);


    //bind group for pass005 (partition ordering)
    std::vector<wgpu::BindGroupEntry> bg005_entries;
    bg005_entries.push_back({ .binding = 0, .buffer = uniformsBuffer, .offset = 0, .size = uniformsBuffer.GetSize() });
    bg005_entries.push_back({ .binding = 1, .buffer = pass004_output_mismatchCounts, .offset = 0, .size = pass004_output_mismatchCounts.GetSize() });
    bg005_entries.push_back({ .binding = 2, .buffer = pass005_output_partitionOrdering, .offset = 0, .size = pass005_output_partitionOrdering.GetSize() });

    wgpu::BindGroupDescriptor bg005_desc = {};
    bg005_desc.layout = pass005_bindGroupLayout;
    bg005_desc.entryCount = bg005_entries.size();
    bg005_desc.entries = bg005_entries.data();
    pass005_bindGroup = device.CreateBindGroup(&bg005_desc);


    //bind group for pass006 (evaluate partition candidates)
    std::vector<wgpu::BindGroupEntry> bg006_entries;
    bg006_entries.push_back({ .binding = 0, .buffer = uniformsBuffer, .offset = 0, .size = uniformsBuffer.GetSize() });
    bg006_entries.push_back({ .binding = 1, .buffer = partitionInfoBuffer, .offset = 0, .size = partitionInfoBuffer.GetSize() });
    bg006_entries.push_back({ .binding = 2, .buffer = inputBlocksBuffer, .offset = 0, .size = inputBlocksBuffer.GetSize() });
    bg006_entries.push_back({ .binding = 3, .buffer = pass005_output_partitionOrdering, .offset = 0, .size = pass005_output_partitionOrdering.GetSize() });
    bg006_entries.push_back({ .binding = 4, .buffer = pass006_output_partitioningErrors, .offset = 0, .size = pass006_output_partitioningErrors.GetSize() });

    wgpu::BindGroupDescriptor bg006_desc = {};
    bg006_desc.layout = pass006_bindGroupLayout;
    bg006_desc.entryCount = bg006_entries.size();
    bg006_desc.entries = bg006_entries.data();
    pass006_bindGroup = device.CreateBindGroup(&bg006_desc);


    //bind group for pass007 (evaluate partition candidates)
    std::vector<wgpu::BindGroupEntry> bg007_entries;
    bg007_entries.push_back({ .binding = 0, .buffer = uniformsBuffer, .offset = 0, .size = uniformsBuffer.GetSize() });
    bg007_entries.push_back({ .binding = 1, .buffer = partitionInfoBuffer, .offset = 0, .size = partitionInfoBuffer.GetSize() });
    bg007_entries.push_back({ .binding = 2, .buffer = inputBlocksBuffer, .offset = 0, .size = inputBlocksBuffer.GetSize() });
    bg007_entries.push_back({ .binding = 3, .buffer = pass006_output_partitioningErrors, .offset = 0, .size = pass006_output_partitioningErrors.GetSize() });
    bg007_entries.push_back({ .binding = 4, .buffer = partitionedBlocksBuffer, .offset = 0, .size = partitionedBlocksBuffer.GetSize() });

    wgpu::BindGroupDescriptor bg007_desc = {};
    bg007_desc.layout = pass007_bindGroupLayout;
    bg007_desc.entryCount = bg007_entries.size();
    bg007_desc.entries = bg007_entries.data();
    pass007_bindGroup = device.CreateBindGroup(&bg007_desc);


    //bind group for pass1 (ideal endpoints and weights)
    std::vector<wgpu::BindGroupEntry> bg1_entries;
    bg1_entries.push_back({ .binding = 0, .buffer = uniformsBuffer, .offset = 0, .size = uniformsBuffer.GetSize() });
    bg1_entries.push_back({ .binding = 1, .buffer = partitionedBlocksBuffer, .offset = 0, .size = partitionedBlocksBuffer.GetSize() });
    bg1_entries.push_back({ .binding = 2, .buffer = pass1_output_idealEndpointsAndWeights, .offset = 0, .size = pass1_output_idealEndpointsAndWeights.GetSize() });

    wgpu::BindGroupDescriptor bg1_desc = {};
    bg1_desc.layout = pass1_bindGroupLayout;
    bg1_desc.entryCount = bg1_entries.size();
    bg1_desc.entries = bg1_entries.data();
    pass1_bindGroup = device.CreateBindGroup(&bg1_desc);


    //bind group for pass2 (decimated weights)
    std::vector<wgpu::BindGroupEntry> bg2_entries;
    bg2_entries.push_back({ .binding = 0, .buffer = uniformsBuffer, .offset = 0, .size = uniformsBuffer.GetSize() });
    bg2_entries.push_back({ .binding = 1, .buffer = validDecimationModesBuffer, .offset = 0, .size = validDecimationModesBuffer.GetSize() });
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
    bg3_entries.push_back({ .binding = 1, .buffer = validDecimationModesBuffer, .offset = 0, .size = validDecimationModesBuffer.GetSize() });
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
    bg4_entries.push_back({ .binding = 1, .buffer = validDecimationModesBuffer, .offset = 0, .size = validDecimationModesBuffer.GetSize() });
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
    bg5_entries.push_back({ .binding = 1, .buffer = validDecimationModesBuffer, .offset = 0, .size = validDecimationModesBuffer.GetSize() });
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
    bg6_entries.push_back({ .binding = 1, .buffer = validBlockModesBuffer, .offset = 0, .size = validBlockModesBuffer.GetSize() });
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
    bg7_entries.push_back({ .binding = 1, .buffer = validBlockModesBuffer, .offset = 0, .size = validBlockModesBuffer.GetSize() });
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
    bg8_entries.push_back({ .binding = 1, .buffer = partitionedBlocksBuffer, .offset = 0, .size = partitionedBlocksBuffer.GetSize() });
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
    bg9_entries.push_back({ .binding = 1, .buffer = partitionedBlocksBuffer, .offset = 0, .size = partitionedBlocksBuffer.GetSize() });
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
    bg11_1_entries.push_back({ .binding = 1, .buffer = validBlockModesBuffer, .offset = 0, .size = validBlockModesBuffer.GetSize() });
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
    bg11_234_entries.push_back({ .binding = 1, .buffer = validBlockModesBuffer, .offset = 0, .size = validBlockModesBuffer.GetSize() });
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
    bg12_entries.push_back({ .binding = 1, .buffer = validBlockModesBuffer, .offset = 0, .size = validBlockModesBuffer.GetSize() });
    bg12_entries.push_back({ .binding = 2, .buffer = pass1_output_idealEndpointsAndWeights, .offset = 0, .size = pass1_output_idealEndpointsAndWeights.GetSize() });
    bg12_entries.push_back({ .binding = 3, .buffer = pass7_output_quantizationResults, .offset = 0, .size = pass7_output_quantizationResults.GetSize() });
    bg12_entries.push_back({ .binding = 4, .buffer = pass11_output_bestEndpointCombinationsForMode, .offset = 0, .size = pass11_output_bestEndpointCombinationsForMode.GetSize() });
    bg12_entries.push_back({ .binding = 5, .buffer = pass12_output_finalCandidates, .offset = 0, .size = pass12_output_finalCandidates.GetSize() });
    bg12_entries.push_back({ .binding = 6, .buffer = pass12_output_topCandidates, .offset = 0, .size = pass12_output_topCandidates.GetSize() });

    wgpu::BindGroupDescriptor bg12_desc = {};
    bg12_desc.layout = pass12_bindGroupLayout;
    bg12_desc.entryCount = bg12_entries.size();
    bg12_desc.entries = bg12_entries.data();
    pass12_bindGroup = device.CreateBindGroup(&bg12_desc);

    //bind group for pass13 (recompute ideal endpoints)
    std::vector<wgpu::BindGroupEntry> bg13_entries;
    bg13_entries.push_back({ .binding = 0, .buffer = uniformsBuffer, .offset = 0, .size = uniformsBuffer.GetSize() });
    bg13_entries.push_back({ .binding = 1, .buffer = decimationInfoBuffer, .offset = 0, .size = decimationInfoBuffer.GetSize() });
    bg13_entries.push_back({ .binding = 2, .buffer = texelToWeightMapBuffer, .offset = 0, .size = texelToWeightMapBuffer.GetSize() });
    bg13_entries.push_back({ .binding = 3, .buffer = blockModesBuffer, .offset = 0, .size = blockModesBuffer.GetSize() });
    bg13_entries.push_back({ .binding = 4, .buffer = partitionedBlocksBuffer, .offset = 0, .size = partitionedBlocksBuffer.GetSize() });
    bg13_entries.push_back({ .binding = 5, .buffer = pass12_output_finalCandidates, .offset = 0, .size = pass12_output_finalCandidates.GetSize() });
    bg13_entries.push_back({ .binding = 6, .buffer = pass13_output_rgbsVectors, .offset = 0, .size = pass13_output_rgbsVectors.GetSize() });

    wgpu::BindGroupDescriptor bg13_desc = {};
    bg13_desc.layout = pass13_bindGroupLayout;
    bg13_desc.entryCount = bg13_entries.size();
    bg13_desc.entries = bg13_entries.data();
    pass13_bindGroup = device.CreateBindGroup(&bg13_desc);

    //bind group for pass14 (pack color endpoints)
    std::vector<wgpu::BindGroupEntry> bg14_entries;
    bg14_entries.push_back({ .binding = 0, .buffer = uniformsBuffer, .offset = 0, .size = uniformsBuffer.GetSize() });
    bg14_entries.push_back({ .binding = 1, .buffer = pass13_output_rgbsVectors, .offset = 0, .size = pass13_output_rgbsVectors.GetSize() });
    bg14_entries.push_back({ .binding = 2, .buffer = pass12_output_finalCandidates, .offset = 0, .size = pass12_output_finalCandidates.GetSize() });

    wgpu::BindGroupDescriptor bg14_desc = {};
    bg14_desc.layout = pass14_bindGroupLayout;
    bg14_desc.entryCount = bg14_entries.size();
    bg14_desc.entries = bg14_entries.data();
    pass14_bindGroup = device.CreateBindGroup(&bg14_desc);

    //bind group for pass15 (unpack color endpoints)
    std::vector<wgpu::BindGroupEntry> bg15_entries;
    bg15_entries.push_back({ .binding = 0, .buffer = uniformsBuffer, .offset = 0, .size = uniformsBuffer.GetSize() });
    bg15_entries.push_back({ .binding = 1, .buffer = pass12_output_finalCandidates, .offset = 0, .size = pass12_output_finalCandidates.GetSize() });
    bg15_entries.push_back({ .binding = 2, .buffer = pass15_output_unpackedEndpoints, .offset = 0, .size = pass15_output_unpackedEndpoints.GetSize() });

    wgpu::BindGroupDescriptor bg15_desc = {};
    bg15_desc.layout = pass15_bindGroupLayout;
    bg15_desc.entryCount = bg15_entries.size();
    bg15_desc.entries = bg15_entries.data();
    pass15_bindGroup = device.CreateBindGroup(&bg15_desc);

    //bind group for pass16 (realign weights)
    std::vector<wgpu::BindGroupEntry> bg16_entries;
    bg16_entries.push_back({ .binding = 0, .buffer = uniformsBuffer, .offset = 0, .size = uniformsBuffer.GetSize() });
    bg16_entries.push_back({ .binding = 1, .buffer = blockModesBuffer, .offset = 0, .size = blockModesBuffer.GetSize() });
    bg16_entries.push_back({ .binding = 2, .buffer = decimationInfoBuffer, .offset = 0, .size = decimationInfoBuffer.GetSize() });
    bg16_entries.push_back({ .binding = 3, .buffer = texelToWeightMapBuffer, .offset = 0, .size = texelToWeightMapBuffer.GetSize() });
    bg16_entries.push_back({ .binding = 4, .buffer = weightToTexelMapBuffer, .offset = 0, .size = weightToTexelMapBuffer.GetSize() });
    bg16_entries.push_back({ .binding = 5, .buffer = partitionedBlocksBuffer, .offset = 0, .size = partitionedBlocksBuffer.GetSize() });
    bg16_entries.push_back({ .binding = 6, .buffer = pass15_output_unpackedEndpoints, .offset = 0, .size = pass15_output_unpackedEndpoints.GetSize() });
    bg16_entries.push_back({ .binding = 7, .buffer = pass12_output_finalCandidates, .offset = 0, .size = pass12_output_finalCandidates.GetSize() });

    wgpu::BindGroupDescriptor bg16_desc = {};
    bg16_desc.layout = pass16_bindGroupLayout;
    bg16_desc.entryCount = bg16_entries.size();
    bg16_desc.entries = bg16_entries.data();
    pass16_bindGroup = device.CreateBindGroup(&bg16_desc);

    //bind group for pass17 (compute final error)
    std::vector<wgpu::BindGroupEntry> bg17_entries;
    bg17_entries.push_back({ .binding = 0, .buffer = uniformsBuffer, .offset = 0, .size = uniformsBuffer.GetSize() });
    bg17_entries.push_back({ .binding = 1, .buffer = blockModesBuffer, .offset = 0, .size = blockModesBuffer.GetSize() });
    bg17_entries.push_back({ .binding = 2, .buffer = decimationInfoBuffer, .offset = 0, .size = decimationInfoBuffer.GetSize() });
    bg17_entries.push_back({ .binding = 3, .buffer = texelToWeightMapBuffer, .offset = 0, .size = texelToWeightMapBuffer.GetSize() });
    bg17_entries.push_back({ .binding = 4, .buffer = partitionedBlocksBuffer, .offset = 0, .size = partitionedBlocksBuffer.GetSize() });
    bg17_entries.push_back({ .binding = 5, .buffer = pass15_output_unpackedEndpoints, .offset = 0, .size = pass15_output_unpackedEndpoints.GetSize() });
    bg17_entries.push_back({ .binding = 6, .buffer = pass12_output_finalCandidates, .offset = 0, .size = pass12_output_finalCandidates.GetSize() });
    bg17_entries.push_back({ .binding = 7, .buffer = pass12_output_topCandidates, .offset = 0, .size = pass12_output_topCandidates.GetSize() });

    wgpu::BindGroupDescriptor bg17_desc = {};
    bg17_desc.layout = pass17_bindGroupLayout;
    bg17_desc.entryCount = bg17_entries.size();
    bg17_desc.entries = bg17_entries.data();
    pass17_bindGroup = device.CreateBindGroup(&bg17_desc);

    //bind group for pass18 (pick best candidate)
    std::vector<wgpu::BindGroupEntry> bg18_entries;
    bg18_entries.push_back({ .binding = 0, .buffer = uniformsBuffer, .offset = 0, .size = uniformsBuffer.GetSize() });
    bg18_entries.push_back({ .binding = 1, .buffer = partitionedBlocksBuffer, .offset = 0, .size = partitionedBlocksBuffer.GetSize() });
    bg18_entries.push_back({ .binding = 2, .buffer = pass12_output_topCandidates, .offset = 0, .size = pass12_output_topCandidates.GetSize() });
    bg18_entries.push_back({ .binding = 3, .buffer = blockModesBuffer, .offset = 0, .size = blockModesBuffer.GetSize() });
    bg18_entries.push_back({ .binding = 4, .buffer = pass18_output_symbolicBlocks, .offset = 0, .size = pass18_output_symbolicBlocks.GetSize() });

    wgpu::BindGroupDescriptor bg18_desc = {};
    bg18_desc.layout = pass18_bindGroupLayout;
    bg18_desc.entryCount = bg18_entries.size();
    bg18_desc.entries = bg18_entries.data();
    pass18_bindGroup = device.CreateBindGroup(&bg18_desc);
}

void ASTCEncoder::releasePerImageResources() {
    // This function explicitly destroys all GPU buffers that are created
    // on a per-image basis, preventing memory leaks.
    if (uniformsBuffer) uniformsBuffer.Destroy();
	if (kmeansTexelsBuffer) kmeansTexelsBuffer.Destroy();
	if (coverageBitmaps2Buffer) coverageBitmaps2Buffer.Destroy();
	if (coverageBitmaps3Buffer) coverageBitmaps3Buffer.Destroy();
	if (coverageBitmaps4Buffer) coverageBitmaps4Buffer.Destroy();
    if (blockModesBuffer) blockModesBuffer.Destroy();
    if (blockModeIndexBuffer) blockModeIndexBuffer.Destroy();
    if (decimationModesBuffer) decimationModesBuffer.Destroy();
    if (decimationInfoBuffer) decimationInfoBuffer.Destroy();
    if (texelToWeightMapBuffer) texelToWeightMapBuffer.Destroy();
    if (weightToTexelMapBuffer) weightToTexelMapBuffer.Destroy();
    if (validDecimationModesBuffer) validDecimationModesBuffer.Destroy();
    if (validBlockModesBuffer) validBlockModesBuffer.Destroy();
    if (sinBuffer) sinBuffer.Destroy();
    if (cosBuffer) cosBuffer.Destroy();
    if (inputBlocksBuffer) inputBlocksBuffer.Destroy();
	if (pass001_output_clusterCenters) pass001_output_clusterCenters.Destroy();
    if (pass002_output_texelAssignments) pass002_output_texelAssignments.Destroy();
	if (pass004_output_mismatchCounts) pass004_output_mismatchCounts.Destroy();
    if (pass005_output_partitionOrdering) pass005_output_partitionOrdering.Destroy();
	if (pass006_output_partitioningErrors) pass006_output_partitioningErrors.Destroy();
	if (partitionedBlocksBuffer) partitionedBlocksBuffer.Destroy();
    if (pass1_output_idealEndpointsAndWeights) pass1_output_idealEndpointsAndWeights.Destroy();
    if (pass2_output_decimatedWeights) pass2_output_decimatedWeights.Destroy();
    if (pass3_output_angular_offsets) pass3_output_angular_offsets.Destroy();
    if (pass4_output_lowestAndHighestWeight) pass4_output_lowestAndHighestWeight.Destroy();
    if (pass5_output_lowValues) pass5_output_lowValues.Destroy();
    if (pass5_output_highValues) pass5_output_highValues.Destroy();
    if (pass6_output_finalValueRanges) pass6_output_finalValueRanges.Destroy();
    if (pass7_output_quantizationResults) pass7_output_quantizationResults.Destroy();
    if (pass8_output_encodingChoiceErrors) pass8_output_encodingChoiceErrors.Destroy();
    if (pass9_output_colorFormatErrors) pass9_output_colorFormatErrors.Destroy();
    if (pass9_output_colorFormats) pass9_output_colorFormats.Destroy();
    if (pass10_output_colorEndpointCombinations) pass10_output_colorEndpointCombinations.Destroy();
    if (pass11_output_bestEndpointCombinationsForMode) pass11_output_bestEndpointCombinationsForMode.Destroy();
    if (pass12_output_finalCandidates) pass12_output_finalCandidates.Destroy();
    if (pass12_output_topCandidates) pass12_output_topCandidates.Destroy();
    if (pass13_output_rgbsVectors) pass13_output_rgbsVectors.Destroy();
    if (pass15_output_unpackedEndpoints) pass15_output_unpackedEndpoints.Destroy();
    if (pass18_output_symbolicBlocks) pass18_output_symbolicBlocks.Destroy();
    if (outputReadbackBuffer) outputReadbackBuffer.Destroy();
}

void ASTCEncoder::printBufferSizes() {
    std::cout << "Input_blocks_buffer: " << (float)(inputBlocksBuffer.GetSize()) / 1000000 << std::endl;
    std::cout << "Pass001_output_clusterCenters: " << (float)(pass001_output_clusterCenters.GetSize()) / 1000000 << std::endl;
    std::cout << "Pass002_output_texelAssignments: " << (float)(pass002_output_texelAssignments.GetSize()) / 1000000 << std::endl;
	std::cout << "Pass004_output_mismatchCounts: " << (float)(pass004_output_mismatchCounts.GetSize()) / 1000000 << std::endl;
	std::cout << "Pass005_output_partitionOrdering: " << (float)(pass005_output_partitionOrdering.GetSize()) / 1000000 << std::endl;
	std::cout << "Pass006_output_partitioningErrors: " << (float)(pass006_output_partitioningErrors.GetSize()) / 1000000 << std::endl;
	std::cout << "Partitioned_blocks_buffer: " << (float)(partitionedBlocksBuffer.GetSize()) / 1000000 << std::endl;
	std::cout << "Pass1_output_idealEndpointsAndWeights: " << (float)(pass1_output_idealEndpointsAndWeights.GetSize()) / 1000000 << std::endl;
	std::cout << "Pass2_output_decimatedWeights: " << (float)(pass2_output_decimatedWeights.GetSize()) / 1000000 << std::endl;
	std::cout << "Pass3_output_angular_offsets: " << (float)(pass3_output_angular_offsets.GetSize()) / 1000000 << std::endl;
	std::cout << "Pass4_output_lowestAndHighestWeight: " << (float)(pass4_output_lowestAndHighestWeight.GetSize()) / 1000000 << std::endl;
	std::cout << "Pass5_output_lowValues: " << (float)(pass5_output_lowValues.GetSize()) / 1000000 << std::endl;
	std::cout << "Pass5_output_highValues: " << (float)(pass5_output_highValues.GetSize()) / 1000000 << std::endl;
	std::cout << "Pass6_output_finalValueRanges: " << (float)(pass6_output_finalValueRanges.GetSize()) / 1000000 << std::endl;
	std::cout << "Pass7_output_quantizationResults: " << (float)(pass7_output_quantizationResults.GetSize()) / 1000000 << std::endl;
	std::cout << "Pass8_output_encodingChoiceErrors: " << (float)(pass8_output_encodingChoiceErrors.GetSize()) / 1000000 << std::endl;
	std::cout << "Pass9_output_colorFormatErrors: " << (float)(pass9_output_colorFormatErrors.GetSize()) / 1000000 << std::endl;
	std::cout << "Pass9_output_colorFormats: " << (float)(pass9_output_colorFormats.GetSize()) / 1000000 << std::endl;
	std::cout << "Pass10_output_colorEndpointCombinations: " << (float)(pass10_output_colorEndpointCombinations.GetSize()) / 1000000 << std::endl;
	std::cout << "Pass11_output_bestEndpointCombinationsForMode: " << (float)(pass11_output_bestEndpointCombinationsForMode.GetSize()) / 1000000 << std::endl;
	std::cout << "Pass12_output_finalCandidates: " << (float)(pass12_output_finalCandidates.GetSize()) / 1000000 << std::endl;
	std::cout << "Pass12_output_topCandidates: " << (float)(pass12_output_topCandidates.GetSize()) / 1000000 << std::endl;
	std::cout << "Pass13_output_rgbsVectors: " << (float)(pass13_output_rgbsVectors.GetSize()) / 1000000 << std::endl;
	std::cout << "Pass15_output_unpackedEndpoints: " << (float)(pass15_output_unpackedEndpoints.GetSize()) / 1000000 << std::endl;
	std::cout << "Pass18_output_symbolicBlocks: " << (float)(pass18_output_symbolicBlocks.GetSize()) / 1000000 << std::endl;
}