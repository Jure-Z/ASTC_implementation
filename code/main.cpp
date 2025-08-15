#include <webgpu/webgpu.h>
#include <webgpu/webgpu_cpp.h>

#if defined(EMSCRIPTEN)
#include <emscripten.h>
#include <emscripten/bind.h>
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

Instance instance = nullptr;
Adapter adapter = nullptr;
Device device = nullptr;
ASTCEncoder* encoder = nullptr;
int image_width = 0;
int image_height = 0;

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

#if defined(EMSCRIPTEN)

void keep_runtime_alive_loop() {}

extern "C" EMSCRIPTEN_KEEPALIVE void process_image(uintptr_t data, size_t size, int width, int height, int blockXDim, int blockYDim) {

	//safety check
	if (!encoder || !encoder->is_initialized) {
		std::cerr << "Error: process_image called before ASTCEncoder was ready." << std::endl;
		return;
	}

    std::cout << "Processing image of size " << width << "x" << height << std::endl;

    image_width = width;
    image_height = height;
    uint8_t* image_data = reinterpret_cast<uint8_t*>(data);


	encoder->secondaryInit(width, height, blockXDim, blockYDim);

    unsigned int blocksX = encoder->blocksX;
    unsigned int blocksY = encoder->blocksY;
    size_t dataLen = blocksX * blocksY * 16;
    uint8_t* dataOut = new uint8_t[dataLen];


    encoder->encode(image_data, dataOut, dataLen);


	AstcFile astcFile = create_astc_file_in_memory(blockXDim, blockYDim, width, height, dataOut, dataLen);

    // Expose the compressed data to JavaScript for download
	EM_ASM_({
		let dataPtr = $0;
		let dataSize = $1;

		let dataArray = new Uint8Array(HEAPU8.subarray(dataPtr, dataPtr + dataSize));
		let blob = new Blob([dataArray], {type: "application/octet-stream"});
		let url = URL.createObjectURL(blob);

		let downloadLink = document.createElement('a');
		downloadLink.href = url;
		downloadLink.download = 'compressed_image.astc';
		document.body.appendChild(downloadLink);
		downloadLink.click();
		document.body.removeChild(downloadLink);
		URL.revokeObjectURL(url);
		}, astcFile.data.get(), astcFile.size);

	EM_ASM_({
			window.onCompressionFinished(true);
	});


    delete[] dataOut;
}
#endif

int main(int, char**) {

#if defined(__EMSCRIPTEN__)

	instance = wgpu::CreateInstance();
	if (!instance) {
		std::cerr << "Failed to create instance" << std::endl;
		return 1;
	}

	wgpu::RequestAdapterOptions adapterOpts = {};
	adapterOpts.nextInChain = nullptr;

	// --- Start the async chain ---
	requestAdapterAsync(instance, &adapterOpts, [](wgpu::Adapter receivedAdapter) {
		// This is the first callback. It runs when the adapter is ready.
		if (!receivedAdapter) {
			std::cerr << "Adapter request failed" << std::endl;
			return;
		}
		adapter = receivedAdapter;

		AdapterInfo properties = {};
		properties.nextInChain = nullptr;
		adapter.GetInfo(&properties);

		std::cout << "--- Adapter Info ---" << std::endl;
		std::cout << "Vendor: " << (properties.vendor ? properties.vendor : "N/A") << std::endl;
		std::cout << "Architecture: " << (properties.architecture ? properties.architecture : "N/A") << std::endl;
		std::cout << "Device: " << (properties.device ? properties.device : "N/A - by browser design") << std::endl;
		std::cout << "Description: " << (properties.description ? properties.description : "N/A") << std::endl;
		std::cout << "--------------------" << std::endl;

		DeviceDescriptor deviceDesc = {};
		deviceDesc.nextInChain = nullptr;
		deviceDesc.label = "My Device";
		deviceDesc.requiredFeatureCount = 0;
		deviceDesc.requiredLimits = nullptr;
		deviceDesc.defaultQueue.nextInChain = nullptr;
		deviceDesc.defaultQueue.label = "The default queue";

		deviceDesc.deviceLostCallback = [](WGPUDeviceLostReason reason, char const* message, void* /* pUserData */) {
			std::cout << "Device lost: reason " << reason;
			if (message) std::cout << " (" << message << ")";
			std::cout << std::endl;
		};

		// --- Continue the chain ---
		requestDeviceAsync(adapter, &deviceDesc, [](wgpu::Device receivedDevice) {
			// This is the second callback. It runs when the device is ready.
			if (!receivedDevice) {
				std::cerr << "Device request failed" << std::endl;
				return;
			}
			device = receivedDevice;

			auto onDeviceError = [](WGPUErrorType type, char const* message, void* /* pUserData */) {
				std::cout << "Uncaptured device error: type " << type;
				if (message) std::cout << " (" << message << ")";
				std::cout << std::endl;
				};
			device.SetUncapturedErrorCallback(onDeviceError, nullptr /* pUserData */);

			//create encoder object
			encoder = new ASTCEncoder(device);

			//callback for the encoder init function
			auto on_encoder_ready = []() {
				std::cout << "ASTCEncoder is fully initialized. Application is ready." << std::endl;

				//signal javascript that initialization is complete
				EM_ASM_({
					Module.isReady = true;
					if (window.onWasmReady) window.onWasmReady(); //enable the file upload button
				});

				emscripten_cancel_main_loop();
			};

			//start async initialization of the encoder
			encoder->initAsync(on_encoder_ready);

		});
	});

	emscripten_set_main_loop(keep_runtime_alive_loop, 0, 1);

#else

	std::cerr << "Preparing webgpu adapter..." << std::endl;

	InstanceDescriptor desc = {};
	desc.nextInChain = nullptr;

	instance = CreateInstance(&desc);

	if (!instance) {
		std::cerr << "Could not initialize WebGPU!" << std::endl;
		return 1;
	}

	RequestAdapterOptions adapterOpts = {};
	adapterOpts.powerPreference = PowerPreference::HighPerformance;
	adapterOpts.nextInChain = nullptr;
	Adapter adapter = requestAdapterSync(instance, &adapterOpts);

	if (!adapter) {
		std::cerr << "Could not get adapter!" << std::endl;
		return 1;
	}

	AdapterInfo properties = {};
	properties.nextInChain = nullptr;
	adapter.GetInfo(&properties);

	std::cout << "--- Adapter Info ---" << std::endl;
	std::cout << "Vendor: " << (properties.vendor ? properties.vendor : "N/A") << std::endl;
	std::cout << "Architecture: " << (properties.architecture ? properties.architecture : "N/A") << std::endl;
	std::cout << "Device: " << (properties.device ? properties.device : "N/A - by browser design") << std::endl;
	std::cout << "Description: " << (properties.description ? properties.description : "N/A") << std::endl;
	std::cout << "--------------------" << std::endl;

	//clean up instance
	//wgpuInstanceRelease(instance);	

	std::cout << "Requesting device..." << std::endl;
	DeviceDescriptor deviceDesc = {};
	deviceDesc.nextInChain = nullptr;
	deviceDesc.label = "My Device";
	deviceDesc.requiredFeatureCount = 0;
	deviceDesc.requiredLimits = nullptr;
	deviceDesc.defaultQueue.nextInChain = nullptr;
	deviceDesc.defaultQueue.label = "The default queue";

	deviceDesc.deviceLostCallback = [](WGPUDeviceLostReason reason, char const* message, void* /* pUserData */) {
		std::cout << "Device lost: reason " << reason;
		if (message) std::cout << " (" << message << ")";
		std::cout << std::endl;
		};

	device = requestDeviceSync(adapter, &deviceDesc);

	if (!device) {
		std::cerr << "Could not get device!" << std::endl;
		return 1;
	}

	auto onDeviceError = [](WGPUErrorType type, char const* message, void* /* pUserData */) {
		std::cout << "Uncaptured device error: type " << type;
		if (message) std::cout << " (" << message << ")";
		std::cout << std::endl;
		};
	device.SetUncapturedErrorCallback(onDeviceError, nullptr /* pUserData */);

    ImageData image = LoadImageRGBA(TEST_IMAGE_DIR "/image.jpeg");

	unsigned int blockXDim = 10;
	unsigned int blockYDim = 10;

	if (encoder) {
		delete encoder;
	}
	encoder = new ASTCEncoder(device);

	encoder->init();
	encoder->secondaryInit(image.width, image.height, blockXDim, blockYDim);

	unsigned int blocksX = encoder->blocksX;
	unsigned int blocksY = encoder->blocksY;
	size_t dataLen = blocksX * blocksY * 16; //number of blocks * 128 bits

	uint8_t* dataOut = new uint8_t[dataLen];

	encoder->encode(image.pixels, dataOut, dataLen);

	store_image(blockXDim, blockYDim, image.width, image.height, dataOut, dataLen, TEST_IMAGE_DIR "/imageOut.astc");

	FreeImage(image);
#endif

	return 0;
}