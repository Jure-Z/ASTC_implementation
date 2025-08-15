#include <webgpu/webgpu.h>
#include <webgpu/webgpu_cpp.h>

#if defined(EMSCRIPTEN)
#include <emscripten.h>
#endif

#include <iostream>
#include <cassert>
#include <mutex>
#include <condition_variable>
#include <tuple>
#include <fstream>

wgpu::Adapter requestAdapterSync(wgpu::Instance instance, wgpu::RequestAdapterOptions const* options) {

	struct UserData {
		wgpu::Adapter adapter = nullptr;
		bool requestEnded = false;
	};
	UserData userData;

	auto onAdapterRequestEnded = [](WGPURequestAdapterStatus status, WGPUAdapter receivedAdapter, char const* message, void* pUserData) {
		UserData& userData = *reinterpret_cast<UserData*>(pUserData);
		if (status == WGPURequestAdapterStatus_Success) {
			userData.adapter = wgpu::Adapter::Acquire(receivedAdapter);
		}
		else {
			std::cout << "Could not get WebGPU adapter: " << message << std::endl;
		}
		userData.requestEnded = true;
		};

	instance.RequestAdapter(options, onAdapterRequestEnded, &userData);

#ifdef __EMSCRIPTEN__
	while (!userData.requestEnded) {
		emscripten_sleep(100);
	}
#endif // __EMSCRIPTEN__

	return userData.adapter;
}

void requestAdapterAsync(
	wgpu::Instance instance,
	wgpu::RequestAdapterOptions const* options,
	std::function<void(wgpu::Adapter)> callback
) {
	// The C++ std::function is a complex object. We can't pass it directly
	// as a C-style userdata pointer. The solution is to allocate it on the heap
	// and pass the pointer to that allocation.
	auto* userdata = new std::function<void(wgpu::Adapter)>(callback);

	// The C-style callback that WebGPU will execute when the async operation is complete.
	auto onAdapterReady = [](WGPURequestAdapterStatus status, WGPUAdapter receivedAdapter, char const* message, void* pUserData) {
		// Cast the userdata pointer back to the C++ function type.
		auto* cb = static_cast<std::function<void(wgpu::Adapter)>*>(pUserData);

		if (status == WGPURequestAdapterStatus_Success) {
			// Acquire the raw C handle into a C++ wrapper and call the callback.
			(*cb)(wgpu::Adapter::Acquire(receivedAdapter));
		}
		else {
			std::cout << "Could not get WebGPU adapter: " << message << std::endl;
			// On failure, call the callback with a null adapter.
			(*cb)(nullptr);
		}

		// IMPORTANT: Clean up the memory we allocated for the callback.
		delete cb;
		};

	// Make the non-blocking async call. This function returns immediately.
	instance.RequestAdapter(options, onAdapterReady, userdata);
}

/*
WGPUAdapter requestAdapterSync(WGPUInstance instance, WGPURequestAdapterOptions const* options) {
	// A simple structure holding the local information shared with the
	// onAdapterRequestEnded callback.
	struct UserData {
		WGPUAdapter adapter = nullptr;
		bool requestEnded = false;
	};
	UserData userData;

	// Callback called by wgpuInstanceRequestAdapter when the request returns
	// This is a C++ lambda function, but could be any function defined in the
	// global scope. It must be non-capturing (the brackets [] are empty) so
	// that it behaves like a regular C function pointer, which is what
	// wgpuInstanceRequestAdapter expects (WebGPU being a C API). The workaround
	// is to convey what we want to capture through the pUserData pointer,
	// provided as the last argument of wgpuInstanceRequestAdapter and received
	// by the callback as its last argument.
	auto onAdapterRequestEnded = [](WGPURequestAdapterStatus status, WGPUAdapter adapter, char const* message, void* pUserData) {
		UserData& userData = *reinterpret_cast<UserData*>(pUserData);
		if (status == WGPURequestAdapterStatus_Success) {
			userData.adapter = adapter;
		}
		else {
			std::cout << "Could not get WebGPU adapter: " << message << std::endl;
		}
		userData.requestEnded = true;
		};

	// Call to the WebGPU request adapter procedure
	wgpuInstanceRequestAdapter(
		instance,
		options,
		onAdapterRequestEnded,
		(void*)&userData
	);

	// We wait until userData.requestEnded gets true
#ifdef __EMSCRIPTEN__
	while (!userData.requestEnded) {
		emscripten_sleep(100);
	}
#endif // __EMSCRIPTEN__

	assert(userData.requestEnded);

	return userData.adapter;
}
*/

wgpu::Device requestDeviceSync(wgpu::Adapter adapter, wgpu::DeviceDescriptor const* descriptor) {

	struct UserData {
		wgpu::Device device = nullptr;
		bool requestEnded = false;
	};
	UserData userData;

	auto onDeviceRequestEnded = [](WGPURequestDeviceStatus status, WGPUDevice receivedDevice, char const* message, void* pUserData) {
		UserData& userData = *reinterpret_cast<UserData*>(pUserData);
		if (status == WGPURequestDeviceStatus_Success) {
			userData.device = wgpu::Device::Acquire(receivedDevice);
		}
		else {
			std::cout << "Could not get WebGPU device: " << message << std::endl;
		}
		userData.requestEnded = true;
		};

	adapter.RequestDevice(descriptor, onDeviceRequestEnded, &userData);

#ifdef __EMSCRIPTEN__
	while (!userData.requestEnded) {
		emscripten_sleep(100);
	}
#endif // __EMSCRIPTEN__

	return userData.device;
}

void requestDeviceAsync(
	wgpu::Adapter adapter,
	wgpu::DeviceDescriptor const* descriptor,
	std::function<void(wgpu::Device)> callback
) {
	// Allocate the C++ callback on the heap to pass it through the C API.
	auto* userdata = new std::function<void(wgpu::Device)>(callback);

	// The C-style callback that will be executed by the browser.
	auto onDeviceReady = [](WGPURequestDeviceStatus status, WGPUDevice receivedDevice, char const* message, void* pUserData) {
		// Cast the userdata pointer back to our C++ std::function.
		auto* cb = static_cast<std::function<void(wgpu::Device)>*>(pUserData);

		if (status == WGPURequestDeviceStatus_Success) {
			// Acquire the raw C handle into a C++ wrapper and call the callback.
			(*cb)(wgpu::Device::Acquire(receivedDevice));
		}
		else {
			std::cout << "Could not get WebGPU device: " << message << std::endl;
			// On failure, call the callback with a null device.
			(*cb)(nullptr);
		}

		// IMPORTANT: Clean up the heap-allocated callback.
		delete cb;
		};

	// Make the non-blocking async call.
	adapter.RequestDevice(descriptor, onDeviceReady, userdata);
}

/*
WGPUDevice requestDeviceSync(WGPUAdapter adapter, WGPUDeviceDescriptor const* descriptor) {
	struct UserData {
		WGPUDevice device = nullptr;
		bool requestEnded = false;
	};
	UserData userData;

	auto onDeviceRequestEnded = [](WGPURequestDeviceStatus status, WGPUDevice device, char const* message, void* pUserData) {
		UserData& userData = *reinterpret_cast<UserData*>(pUserData);
		if (status == WGPURequestDeviceStatus_Success) {
			userData.device = device;
		}
		else {
			std::cout << "Could not get WebGPU device: " << message << std::endl;
		}
		userData.requestEnded = true;
		};

	wgpuAdapterRequestDevice(
		adapter,
		descriptor,
		onDeviceRequestEnded,
		(void*)&userData
	);

#ifdef __EMSCRIPTEN__
	while (!userData.requestEnded) {
		emscripten_sleep(100);
	}
#endif // __EMSCRIPTEN__

	assert(userData.requestEnded);

	return userData.device;
}
*/

// === Load WGSL shader from file ===
std::string LoadWGSL(const std::string& path) {
	std::ifstream file(path);
	if (!file.is_open()) {
		std::cerr << "!!! ERROR: Failed to open shader file at path: " << path << std::endl;
		return nullptr;
	}
	return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

wgpu::ShaderModule prepareShaderModule(wgpu::Device device, std::string filePath, const char* label) {

	std::string shaderSource = LoadWGSL(filePath);

	//std::cout << "shader code: " << shaderSource << std::endl;

	// Create a WGSL shader descriptor
#if defined(__EMSCRIPTEN__)
	wgpu::ShaderModuleWGSLDescriptor wgslDescriptor;
	wgslDescriptor.code = shaderSource.c_str();
	wgslDescriptor.nextInChain = nullptr;

	wgslDescriptor.sType = wgpu::SType::ShaderModuleWGSLDescriptor;

	wgpu::ShaderModuleDescriptor shaderDesc = {};
	shaderDesc.label = label;
	shaderDesc.nextInChain = &wgslDescriptor;

	wgpu::ShaderModule shaderModule = device.CreateShaderModule(&shaderDesc);
#else
	wgpu::ShaderSourceWGSL wgslDescriptor = {};
	wgslDescriptor.code = shaderSource.c_str();
	wgslDescriptor.nextInChain = nullptr;

	wgslDescriptor.sType = wgpu::SType::ShaderSourceWGSL;

	wgpu::ShaderModuleDescriptor shaderDesc = {};
	shaderDesc.label = label;
	shaderDesc.nextInChain = &wgslDescriptor;

	wgpu::ShaderModule shaderModule = device.CreateShaderModule(&shaderDesc);
#endif

	return shaderModule;
}