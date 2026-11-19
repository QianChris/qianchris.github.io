# environment-lighting

Environment lighting capability with an atomic runtime resource set and static
image source loading.

The plugin computes:

- SH9 diffuse irradiance with Lambert band convolution.
- An unfiltered, configurable 128-2048-per-face sky-radiance cubemap.
- A GGX-prefiltered `rgba16float` cubemap mip chain.
- A split-sum DFG lookup texture.
- A visible-sky renderer and a resource contract for material consumers.

## Scene Component

Add zero or one `EnvironmentLightComponent`:

```json
{
  "source": "../../common/textures/skybox.png",
  "lightingEnabled": 1,
  "skyVisible": 1,
  "intensity": 1.0,
  "diffuseIntensity": 1.0,
  "specularIntensity": 1.0,
  "backgroundIntensity": 1.0,
  "rotation": 0,
  "skyResolution": 2048,
  "specularResolution": 64
}
```

`rotation` is degrees around world Y. `skyResolution` and
`specularResolution` are rounded to powers of two and clamped to 128-2048 and
32-256 respectively. A 2048 LDR sky cubemap is about 96 MiB; use 1024 or 512
for memory-constrained targets.

`intensity` is the master distant-light intensity. `diffuseIntensity` and
`specularIntensity` scale the environment's diffuse irradiance and reflected
radiance globally; material response remains owned by base color, metallic,
roughness, F0/specular, and occlusion fields on material components.

`lightingEnabled`, `skyVisible`, the three intensity controls, and `rotation`
are runtime fields. Lighting and the visible background can therefore be
switched independently.
`source`, `skyResolution`, and `specularResolution` affect resource acquisition
or preprocessing. They are read when the app loads or the component is added.
Editing them on a live component does not reprocess the environment; edit the
persisted scene and load the app again.

No component, or an empty `source`, keeps the complete neutral fallback set
active. Removing the component immediately disables IBL and visible sky,
destroys the prepared environment textures and buffer, and installs only the
tiny neutral resources required by statically declared pipeline bindings.
Adding the component again loads its source again. A missing or invalid source
fails app loading or component activation.

## Consumer Contract

Material consumers use `environmentLighting`:

```text
environment.data
environment.sampler
environment.specular
environment.brdf-lut
```

Visible-sky consumers use `environmentSky` with `environment.data`,
`environment.sampler`, and `environment.sky-radiance`. The complete set is
registered under the owner-local set name `environment`, replaced with exact
membership checks, and removed atomically.

The reference PBR consumer lives in
`public/apps/demo11_environmentLighting/pipelines/PbrEnvironmentPipeline.json`.
It belongs to the demo rather than this plugin because material BRDFs own how
they consume diffuse irradiance, prefiltered radiance, and the DFG LUT.

## Responsibility Boundary

The plugin owns environment-source acquisition, preprocessing, shared resource
publication, neutral fallbacks, runtime environment parameters, and visible-sky
rendering. It does not own PBR, clearcoat, cloth, or other material shading
models. Those consumers depend on the resource contract above and provide their
own pipelines.

## Source Loading

The plugin decodes static 2:1 sRGB images into an equirectangular source
payload. The source loader owns image acquisition and decoding; the environment
processor owns SH projection and GPU resource generation.

`source` is a project URL, not an operating-system file handle. A durable file
picker requires an asset-import workflow that copies the selected file into the
project and writes the resulting URL into scene data; a temporary browser Blob
would stop working after reload.

## Current Limits

- Source decoding is sRGB LDR, not HDR/EXR.
- Runtime preprocessing is performed when the app loads or the component is
  added. Source and resolution edits do not update prepared resources live.
- Static images are decoded up to `skyResolution * 4` wide and the device 2D
  texture limit. The demo therefore uses the full 8192x4096 source.
- Runtime fields update every frame.
- The demo PBR shader performs ACES-style tone mapping plus sRGB output. The
  display-referred LDR sky avoids a second tone-map and only applies sRGB
  output, pending a dedicated HDR color pipeline.
- Local reflection probes, parallax correction, multi-scattering compensation,
  and dynamic scene capture are not supported.
