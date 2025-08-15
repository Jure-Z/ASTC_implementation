#include "astc.h"
#include "webgpu_utils.h"

void ASTCEncoder::initBindGroupLayouts() {

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
    pass13_recomputeIdealEndpointsShader = prepareShaderModule(device, "shaders/pass13_recompute_ideal_endpoints.wgsl", "recompute ideal endpoints (pass13)");
    pass14_packColorEndpointsShader = prepareShaderModule(device, "shaders/pass14_pack_color_endpoints.wgsl", "pack color endpoints (pass14)");
    pass15_unpackColorEndpointsShader = prepareShaderModule(device, "shaders/pass15_unpack_color_endpoints.wgsl", "unpack color endpoints (pass15)");
    pass16_realignWeightsShader = prepareShaderModule(device, "shaders/pass16_realign_weights.wgsl", "realign weights (pass16)");
    pass17_computeFinalErrorShader = prepareShaderModule(device, "shaders/pass17_compute_final_error.wgsl", "compute final error (pass17)");
    pass18_pickBestCandidateShader = prepareShaderModule(device, "shaders/pass18_pick_best_candidate.wgsl", "pick best candidate (pass18)");
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
    pass13_recomputeIdealEndpointsShader = prepareShaderModule(device, SHADER_DIR "/pass13_recompute_ideal_endpoints.wgsl", "recompute ideal endpoints (pass13)");
    pass14_packColorEndpointsShader = prepareShaderModule(device, SHADER_DIR "/pass14_pack_color_endpoints.wgsl", "pack color endpoints (pass14)");
    pass15_unpackColorEndpointsShader = prepareShaderModule(device, SHADER_DIR "/pass15_unpack_color_endpoints.wgsl", "unpack color endpoints (pass15)");
    pass16_realignWeightsShader = prepareShaderModule(device, SHADER_DIR "/pass16_realign_weights.wgsl", "realign weights (pass16)");
    pass17_computeFinalErrorShader = prepareShaderModule(device, SHADER_DIR "/pass17_compute_final_error.wgsl", "compute final error (pass17)");
    pass18_pickBestCandidateShader = prepareShaderModule(device, SHADER_DIR "/pass18_pick_best_candidate.wgsl", "pick best candidate (pass18)");
#endif

    if (!pass1_idealEndpointsShader) {
        std::cerr << "FATAL ERROR: Failed to create shader module for pass 1. Check shader file paths." << std::endl;
        // You could even throw an exception here to halt execution
        throw std::runtime_error("Failed to load a critical shader.");
    }

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

#if defined(EMSCRIPTEN)
void ASTCEncoder::initPipelinesAsync(std::function<void()> on_all_pipelines_created) {

    //callback that will be used for all pipeline creations
    auto master_callback_ptr = std::make_shared<std::function<void(WGPUCreatePipelineAsyncStatus, WGPUComputePipeline, char const*, void*)>>(
        [=, this](WGPUCreatePipelineAsyncStatus status, WGPUComputePipeline pipeline, char const* message, void* userdata) {

            ASTCEncoder* encoder = static_cast<ASTCEncoder*>(userdata);
            if (status != WGPUCreatePipelineAsyncStatus_Success) {
                std::cerr << "FATAL: Failed to create a compute pipeline: " << message << std::endl;
            }

            encoder->m_pending_pipelines--;
            //std::cout << "Pipelines remaining:" << m_pending_pipelines << std::endl;

            if (encoder->m_pending_pipelines == 0) {
                std::cout << "All compute pipelines have been created successfully." << std::endl;
                on_all_pipelines_created();
            }
        });

    pass1_idealEndpointsShader = prepareShaderModule(device, "/shaders/pass1_ideal_endpoints_and_weights.wgsl", "Ideal endpoints and weights (pass1)");
    pass2_decimatedWeightsShader = prepareShaderModule(device, "/shaders/pass2_decimated_weights.wgsl", "decimated weights (pass2)");
    pass3_angularOffsetsShader = prepareShaderModule(device, "/shaders/pass3_compute_angular_offsets.wgsl", "angular offsets (pass3)");
    pass4_lowestAndHighestWeightShader = prepareShaderModule(device, "/shaders/pass4_lowest_and_highest_weight.wgsl", "lowest and highest weight (pass4)");
    pass5_valuesForQuantLevelsShader = prepareShaderModule(device, "/shaders/pass5_best_values_for_quant_levels.wgsl", "best values for quant levels (pass5)");
    pass6_remapLowAndHighValuesShader = prepareShaderModule(device, "/shaders/pass6_remap_low_and_high_values.wgsl", "remap low and high values (pass6)");
    pass7_weightsAndErrorForBMShader = prepareShaderModule(device, "/shaders/pass7_weights_and_error_for_bm.wgsl", "weights and error for block mode (pass7)");
    pass8_encodingChoiceErrorsShader = prepareShaderModule(device, "/shaders/pass8_compute_encoding_choice_errors.wgsl", "encoding choice errors (pass8)");
    pass9_computeColorErrorShader = prepareShaderModule(device, "/shaders/pass9_compute_color_error.wgsl", "color format errors (pass9)");
    pass10_colorEndpointCombinationsShader_2part = prepareShaderModule(device, "/shaders/pass10_color_combinations_for_quant_2part.wgsl", "color endpoint combinations (pass10, 2part)");
    pass10_colorEndpointCombinationsShader_3part = prepareShaderModule(device, "/shaders/pass10_color_combinations_for_quant_3part.wgsl", "color endpoint combinations (pass10, 3part)");
    pass10_colorEndpointCombinationsShader_4part = prepareShaderModule(device, "/shaders/pass10_color_combinations_for_quant_4part.wgsl", "color endpoint combinations (pass10, 4part)");
    pass11_bestEndpointCombinationsForModeShader_1part = prepareShaderModule(device, "/shaders/pass11_best_color_combination_for_mode_1part.wgsl", "best endpoint combinations for mode (pass11, 1part)");
    pass11_bestEndpointCombinationsForModeShader_2part = prepareShaderModule(device, "/shaders/pass11_best_color_combination_for_mode_2part.wgsl", "best endpoint combinations for mode (pass11, 2part)");
    pass11_bestEndpointCombinationsForModeShader_3part = prepareShaderModule(device, "/shaders/pass11_best_color_combination_for_mode_3part.wgsl", "best endpoint combinations for mode (pass11, 3part)");
    pass11_bestEndpointCombinationsForModeShader_4part = prepareShaderModule(device, "/shaders/pass11_best_color_combination_for_mode_4part.wgsl", "best endpoint combinations for mode (pass11, 4part)");
    pass12_findTopNCandidatesShader = prepareShaderModule(device, "/shaders/pass12_find_top_N_candidates.wgsl", "find top N candidates (pass12)");
    pass13_recomputeIdealEndpointsShader = prepareShaderModule(device, "/shaders/pass13_recompute_ideal_endpoints.wgsl", "recompute ideal endpoints (pass13)");
    pass14_packColorEndpointsShader = prepareShaderModule(device, "/shaders/pass14_pack_color_endpoints.wgsl", "pack color endpoints (pass14)");
    pass15_unpackColorEndpointsShader = prepareShaderModule(device, "/shaders/pass15_unpack_color_endpoints.wgsl", "unpack color endpoints (pass15)");
    pass16_realignWeightsShader = prepareShaderModule(device, "/shaders/pass16_realign_weights.wgsl", "realign weights (pass16)");
    pass17_computeFinalErrorShader = prepareShaderModule(device, "/shaders/pass17_compute_final_error.wgsl", "compute final error (pass17)");
    pass18_pickBestCandidateShader = prepareShaderModule(device, "/shaders/pass18_pick_best_candidate.wgsl", "pick best candidate (pass18)");


    const int total_pipelines = 23;
    m_pending_pipelines = total_pipelines;

    struct PipelineCreationInfo {
        ASTCEncoder* encoder;
        wgpu::ComputePipeline* target;
        std::shared_ptr<std::function<void(WGPUCreatePipelineAsyncStatus, WGPUComputePipeline, char const*, void*)>> master_callback;
    };

    auto pipeline_callback = [](WGPUCreatePipelineAsyncStatus status, WGPUComputePipeline p, char const* msg, void* userdata) {

        auto* info = static_cast<PipelineCreationInfo*>(userdata);

        if (status == WGPUCreatePipelineAsyncStatus_Success) {
            *info->target = wgpu::ComputePipeline::Acquire(p);
        }

        (*info->master_callback)(status, p, msg, info->encoder);

        delete info;
    };

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

    auto* info1 = new PipelineCreationInfo{ this, &pass1_pipeline, master_callback_ptr };
    device.CreateComputePipelineAsync(&pass1_pipelineDesc, pipeline_callback, info1);

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

    auto* info2 = new PipelineCreationInfo{ this, &pass2_pipeline, master_callback_ptr };
    device.CreateComputePipelineAsync(&pass2_pipelineDesc, pipeline_callback, info2);


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

    auto* info3 = new PipelineCreationInfo{ this, &pass3_pipeline, master_callback_ptr };
    device.CreateComputePipelineAsync(&pass3_pipelineDesc, pipeline_callback, info3);


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

    auto* info4 = new PipelineCreationInfo{ this, &pass4_pipeline, master_callback_ptr };
    device.CreateComputePipelineAsync(&pass4_pipelineDesc, pipeline_callback, info4);


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

    auto* info5 = new PipelineCreationInfo{ this, &pass5_pipeline, master_callback_ptr };
    device.CreateComputePipelineAsync(&pass5_pipelineDesc, pipeline_callback, info5);


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

    auto* info6 = new PipelineCreationInfo{ this, &pass6_pipeline, master_callback_ptr };
    device.CreateComputePipelineAsync(&pass6_pipelineDesc, pipeline_callback, info6);


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

    auto* info7 = new PipelineCreationInfo{ this, &pass7_pipeline, master_callback_ptr };
    device.CreateComputePipelineAsync(&pass7_pipelineDesc, pipeline_callback, info7);

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

    auto* info8 = new PipelineCreationInfo{ this, &pass8_pipeline, master_callback_ptr };
    device.CreateComputePipelineAsync(&pass8_pipelineDesc, pipeline_callback, info8);

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

    auto* info9 = new PipelineCreationInfo{ this, &pass9_pipeline, master_callback_ptr };
    device.CreateComputePipelineAsync(&pass9_pipelineDesc, pipeline_callback, info9);

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

    auto* info10_2 = new PipelineCreationInfo{ this, &pass10_pipeline_2part, master_callback_ptr };
    device.CreateComputePipelineAsync(&pass10_2_pipelineDesc, pipeline_callback, info10_2);

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

    auto* info10_3 = new PipelineCreationInfo{ this, &pass10_pipeline_3part, master_callback_ptr };
    device.CreateComputePipelineAsync(&pass10_3_pipelineDesc, pipeline_callback, info10_3);

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

    auto* info10_4 = new PipelineCreationInfo{ this, &pass10_pipeline_4part, master_callback_ptr };
    device.CreateComputePipelineAsync(&pass10_4_pipelineDesc, pipeline_callback, info10_4);

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

    auto* info11_1 = new PipelineCreationInfo{ this, &pass11_pipeline_1part, master_callback_ptr };
    device.CreateComputePipelineAsync(&pass11_1_pipelineDesc, pipeline_callback, info11_1);

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

    auto* info11_2 = new PipelineCreationInfo{ this, &pass11_pipeline_2part, master_callback_ptr };
    device.CreateComputePipelineAsync(&pass11_2_pipelineDesc, pipeline_callback, info11_2);

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

    auto* info11_3 = new PipelineCreationInfo{ this, &pass11_pipeline_3part, master_callback_ptr };
    device.CreateComputePipelineAsync(&pass11_3_pipelineDesc, pipeline_callback, info11_3);

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

    auto* info11_4 = new PipelineCreationInfo{ this, &pass11_pipeline_4part, master_callback_ptr };
    device.CreateComputePipelineAsync(&pass11_4_pipelineDesc, pipeline_callback, info11_4);

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

    auto* info12 = new PipelineCreationInfo{ this, &pass12_pipeline, master_callback_ptr };
    device.CreateComputePipelineAsync(&pass12_pipelineDesc, pipeline_callback, info12);

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

    auto* info13 = new PipelineCreationInfo{ this, &pass13_pipeline, master_callback_ptr };
    device.CreateComputePipelineAsync(&pass13_pipelineDesc, pipeline_callback, info13);

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

    auto* info14 = new PipelineCreationInfo{ this, &pass14_pipeline, master_callback_ptr };
    device.CreateComputePipelineAsync(&pass14_pipelineDesc, pipeline_callback, info14);

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

    auto* info15 = new PipelineCreationInfo{ this, &pass15_pipeline, master_callback_ptr };
    device.CreateComputePipelineAsync(&pass15_pipelineDesc, pipeline_callback, info15);

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

    auto* info16 = new PipelineCreationInfo{ this, &pass16_pipeline, master_callback_ptr };
    device.CreateComputePipelineAsync(&pass16_pipelineDesc, pipeline_callback, info16);

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

    auto* info17 = new PipelineCreationInfo{ this, &pass17_pipeline, master_callback_ptr };
    device.CreateComputePipelineAsync(&pass17_pipelineDesc, pipeline_callback, info17);

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

    auto* info18 = new PipelineCreationInfo{ this, &pass18_pipeline, master_callback_ptr };
    device.CreateComputePipelineAsync(&pass18_pipelineDesc, pipeline_callback, info18);
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
    inputDesc.size = max_partitioned_blocks * sizeof(InputBlock);
    inputDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    inputBlocksBuffer = device.CreateBuffer(&inputDesc);

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
    bg13_entries.push_back({ .binding = 4, .buffer = inputBlocksBuffer, .offset = 0, .size = inputBlocksBuffer.GetSize() });
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
    bg16_entries.push_back({ .binding = 5, .buffer = inputBlocksBuffer, .offset = 0, .size = inputBlocksBuffer.GetSize() });
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
    bg17_entries.push_back({ .binding = 4, .buffer = inputBlocksBuffer, .offset = 0, .size = inputBlocksBuffer.GetSize() });
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
    bg18_entries.push_back({ .binding = 1, .buffer = inputBlocksBuffer, .offset = 0, .size = inputBlocksBuffer.GetSize() });
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