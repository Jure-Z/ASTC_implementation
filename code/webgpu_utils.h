#pragma once

#include <webgpu/webgpu.h>
#include <webgpu/webgpu_cpp.h>

#if defined(EMSCRIPTEN)
#include <emscripten.h>
#endif

WGPUAdapter requestAdapterSync(WGPUInstance instance, WGPURequestAdapterOptions const* options);

WGPUDevice requestDeviceSync(WGPUAdapter adapter, WGPUDeviceDescriptor const* descriptor);

std::string LoadWGSL(const std::string& path);

wgpu::ShaderModule prepareShaderModule(wgpu::Device device, std::string filePath, const char* label);

template <typename T>
void mapOutputBufferSync(wgpu::Device device, wgpu::Buffer buffer, uint64_t blockCount, std::vector<T>& output) {

    uint64_t outputSize = sizeof(T) * blockCount;

    struct BufferMapContext {
        bool done;
        wgpu::Buffer buffer;
        uint64_t bufferSize;
        std::vector<T>& output;
    };

    BufferMapContext context = { false, buffer, outputSize, output };

    auto onMapComplete = [](WGPUBufferMapAsyncStatus status, void* userdata) {
        BufferMapContext* ctx = reinterpret_cast<BufferMapContext*>(userdata);

        if (status == WGPUBufferMapAsyncStatus_Success) {
            std::cout << "Buffer mapped successfully!" << std::endl;

            const void* mappedData = ctx->buffer.GetConstMappedRange(0, ctx->bufferSize);
            if (mappedData) {
                std::memcpy(ctx->output.data(), mappedData, ctx->bufferSize);
            }
            else {
                std::cout << "Output buffer could not be mapped (mappedData is null)!" << std::endl;
            }
            ctx->buffer.Unmap();
        }
        else {
            std::cout << "Output buffer mapping failed! Status: " << status << std::endl;

        }
        ctx->done = true;
        };

    buffer.MapAsync(wgpu::MapMode::Read, 0, outputSize, onMapComplete, &context);

    while (!context.done) {
#if defined(__EMSCRIPTEN__)
        emscripten_sleep(100);
#else
        device.Tick();
#endif
    }
}