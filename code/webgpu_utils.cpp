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
		instance /* equivalent of navigator.gpu */,
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

// === Load WGSL shader from file ===
std::string LoadWGSL(const std::string& path) {
	std::ifstream file(path);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open shader file");
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