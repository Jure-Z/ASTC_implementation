#include <webgpu/webgpu.h>
#include <webgpu/webgpu_cpp.h>

#if defined(EMSCRIPTEN)
#include <emscripten.h>
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>
#include <cassert>
#include <vector>
#include <cstring>

#include "webgpu_utils.h"
#include "astc.h"
#include "astc_store.h"

using namespace wgpu;

struct ImageData {
    uint8_t* pixels;
    int width;
    int height;
    int channels;
};

// Loads an image from file into RGBA8 format
ImageData LoadImageRGBA(const std::string& filename) {
    ImageData image;
    image.pixels = stbi_load(filename.c_str(), &image.width, &image.height, &image.channels, 4);

    if (!image.pixels) {
        throw std::runtime_error("Failed to load image: " + filename);
    }

    image.channels = 4; // Forced to RGBA
    return image;
}

void FreeImage(ImageData& image) {
    if (image.pixels) {
        stbi_image_free(image.pixels);
        image.pixels = nullptr;
    }
}


int main(int, char**) {

	// We create a descriptor
	WGPUInstanceDescriptor desc = {};
	desc.nextInChain = nullptr;

	// We create the instance using this descriptor
#ifdef WEBGPU_BACKEND_EMSCRIPTEN
	WGPUInstance instance = wgpuCreateInstance(nullptr);
#else //  WEBGPU_BACKEND_EMSCRIPTEN
	WGPUInstance instance = wgpuCreateInstance(&desc);
#endif //  WEBGPU_BACKEND_EMSCRIPTEN

	// We can check whether there is actually an instance created
	if (!instance) {
		std::cerr << "Could not initialize WebGPU!" << std::endl;
		return 1;
	}

	// Display the object (WGPUInstance is a simple pointer, it may be
	// copied around without worrying about its size).
	std::cout << "WGPU instance: " << instance << std::endl;

	std::cout << "Requesting adapter..." << std::endl;
	WGPURequestAdapterOptions adapterOpts = {};
	adapterOpts.nextInChain = nullptr;
	WGPUAdapter adapter = requestAdapterSync(instance, &adapterOpts);
	std::cout << "Got adapter: " << adapter << std::endl;

	//clean up instance
	wgpuInstanceRelease(instance);	

	std::cout << "Requesting device..." << std::endl;
	WGPUDeviceDescriptor deviceDesc = {};
	deviceDesc.nextInChain = nullptr;
	deviceDesc.label = "My Device"; // anything works here, that's your call
	deviceDesc.requiredFeatureCount = 0; // we do not require any specific feature
	deviceDesc.requiredLimits = nullptr; // we do not require any specific limit
	deviceDesc.defaultQueue.nextInChain = nullptr;
	deviceDesc.defaultQueue.label = "The default queue";
	// A function that is invoked whenever the device stops being available.
	deviceDesc.deviceLostCallback = [](WGPUDeviceLostReason reason, char const* message, void* /* pUserData */) {
		std::cout << "Device lost: reason " << reason;
		if (message) std::cout << " (" << message << ")";
		std::cout << std::endl;
		};
	WGPUDevice device = requestDeviceSync(adapter, &deviceDesc);
	std::cout << "Got device: " << device << std::endl;

	// A function that is invoked whenever there is an error in the use of the device
	auto onDeviceError = [](WGPUErrorType type, char const* message, void* /* pUserData */) {
		std::cout << "Uncaptured device error: type " << type;
		if (message) std::cout << " (" << message << ")";
		std::cout << std::endl;
		};
	wgpuDeviceSetUncapturedErrorCallback(device, onDeviceError, nullptr /* pUserData */);

#if defined(__EMSCRIPTEN__)
    ImageData image = LoadImageRGBA("image1.jpg");
#else
    ImageData image = LoadImageRGBA(TEST_IMAGE_DIR "/image1.jpg");
#endif

	unsigned int blockXDim = 10;
	unsigned int blockYDim = 10;
    
	ASTCEncoder* encoder = new ASTCEncoder(device, image.width, image.height, blockXDim, blockYDim);

	unsigned int blocksX = encoder->blocksX;
	unsigned int blocksY = encoder->blocksY;
	size_t dataLen = blocksX * blocksY * 16; //number of blocks * 128 bits

	uint8_t* dataOut = new uint8_t[dataLen];

	encoder->encode(image.pixels, dataOut, dataLen);

	store_image(blockXDim, blockYDim, image.width, image.height, dataOut, dataLen, TEST_IMAGE_DIR "/imageOut.astc");

    FreeImage(image);

	//clean up adapter
	wgpuAdapterRelease(adapter);

	//clean up device
	wgpuDeviceRelease(device);

	return 0;
}