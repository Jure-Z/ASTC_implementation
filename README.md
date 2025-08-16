Requirements
-----

The following tools need to be installed on your system to build the program:
- **python**
- **cmake**
- **emscripten** (to build for web)

For emscripten you can follow the installation instructions here: [emscripten installation](https://emscripten.org/docs/getting_started/downloads.html)

Usage
---

### Native build

Run the following commands:

```bash
cmake -B build-native
cmake --build build-native
```

### Emscripten build

Run the following commands:

```bash
emcmake cmake -B build-web
cmake --build build-web
```

After that you can start the server:

```bash
python -m http.server 80 -d build-web
```

The website will be accessible here:

`http://localhost/webgpu_astc.html`