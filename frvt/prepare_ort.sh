#!/bin/sh
# Install the Linux x64 C++ SDK from the Microsoft.ML.OnnxRuntime package.
set -eu
dest=${1:?Pass the SDK installation directory}
version=1.21.0
pkg=microsoft.ml.onnxruntime
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
curl -fsSL --retry 2 "https://api.nuget.org/v3-flatcontainer/$pkg/$version/$pkg.$version.nupkg" -o "$work/sdk.nupkg"
echo '359b4782cfa2e288e6c4bbb9da1c231b085fb8e0d149faf8c3c843f54576559b  sdk.nupkg' | (cd "$work" && sha256sum --check)
unzip -q "$work/sdk.nupkg" 'build/native/include/*' 'runtimes/linux-x64/native/*' -d "$work"
mkdir -p "$dest/include" "$dest/lib"
cp -R "$work/build/native/include/." "$dest/include/"
cp "$work/runtimes/linux-x64/native/libonnxruntime.so" "$dest/lib/libonnxruntime.so.$version"
ln -sf "libonnxruntime.so.$version" "$dest/lib/libonnxruntime.so.1"
ln -sf libonnxruntime.so.1 "$dest/lib/libonnxruntime.so"
for provider in "$work"/runtimes/linux-x64/native/libonnxruntime_providers*.so; do
    if [ -f "$provider" ]; then cp "$provider" "$dest/lib/"; fi
done
