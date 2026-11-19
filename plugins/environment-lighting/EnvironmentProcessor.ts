import type { EnvironmentSettings } from './EnvironmentLightingSystem.ts';
import type { EnvironmentSource } from './EnvironmentSourceProvider.ts';
import {
    createEnvironmentDataBuffer,
    createEnvironmentSampler,
    type EnvironmentResourceSet,
} from './EnvironmentResourceSet.ts';

const PREFILTER_SIZE_MIN = 32;
const PREFILTER_SIZE_MAX = 256;
const SKY_RADIANCE_SIZE_MIN = 128;
const SKY_RADIANCE_SIZE_MAX = 2048;
const BRDF_LUT_SIZE = 128;

export function runtimeSourceWidth(skyResolution: number, deviceLimit: number): number {
    return Math.min(deviceLimit, clampPowerOfTwo(
        skyResolution,
        SKY_RADIANCE_SIZE_MIN,
        SKY_RADIANCE_SIZE_MAX,
    ) * 4);
}

export async function buildEnvironment(
    device: GPUDevice,
    source: EnvironmentSource,
    settings: EnvironmentSettings,
    shaderBase: string,
): Promise<EnvironmentResourceSet> {
    let sourceTexture: GPUTexture | null = null;
    let skyRadiance: GPUTexture | null = null;
    let specular: GPUTexture | null = null;
    let brdfLut: GPUTexture | null = null;
    let dataBuffer: GPUBuffer | null = null;
    const transientBuffers: GPUBuffer[] = [];

    try {
        const sh = computeDiffuseSh(source);
        const resolution = clampPowerOfTwo(
            settings.specularResolution,
            PREFILTER_SIZE_MIN,
            PREFILTER_SIZE_MAX,
        );
        const mipCount = Math.floor(Math.log2(resolution)) + 1;
        const skyResolution = clampPowerOfTwo(
            settings.skyResolution,
            SKY_RADIANCE_SIZE_MIN,
            Math.min(SKY_RADIANCE_SIZE_MAX, device.limits.maxTextureDimension2D),
        );
        const skyFormat: GPUTextureFormat = source.encoding === 'srgb-ldr'
            ? 'rgba8unorm-srgb'
            : 'rgba16float';

        sourceTexture = device.createTexture({
            label: `environment-source:${source.sourceUrl}`,
            size: { width: source.width, height: source.height },
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING
                | GPUTextureUsage.COPY_DST,
        });
        device.queue.writeTexture(
            { texture: sourceTexture },
            source.pixels,
            { bytesPerRow: source.width * 4, rowsPerImage: source.height },
            { width: source.width, height: source.height },
        );

        const sampler = createEnvironmentSampler(device);
        const sourceSampler = device.createSampler({
            label: 'environment-equirect-sampler',
            addressModeU: 'repeat',
            addressModeV: 'clamp-to-edge',
            magFilter: 'linear',
            minFilter: 'linear',
        });

        skyRadiance = device.createTexture({
            label: 'environment-sky-radiance',
            size: {
                width: skyResolution,
                height: skyResolution,
                depthOrArrayLayers: 6,
            },
            dimension: '2d',
            format: skyFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        specular = device.createTexture({
            label: 'environment-specular',
            size: { width: resolution, height: resolution, depthOrArrayLayers: 6 },
            dimension: '2d',
            format: 'rgba16float',
            mipLevelCount: mipCount,
            usage: GPUTextureUsage.RENDER_ATTACHMENT
                | GPUTextureUsage.TEXTURE_BINDING,
        });
        brdfLut = device.createTexture({
            label: 'environment-brdf-lut',
            size: { width: BRDF_LUT_SIZE, height: BRDF_LUT_SIZE },
            format: 'rgba16float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });

        const [cubeSource, prefilterSource, brdfSource] = await Promise.all([
            fetchShader(`${shaderBase}/EquirectToCube.wgsl`),
            fetchShader(`${shaderBase}/PrefilterEnvironment.wgsl`),
            fetchShader(`${shaderBase}/IntegrateBrdf.wgsl`),
        ]);
        const [cubePipeline, prefilterPipeline, brdfPipeline] = await Promise.all([
            createBakePipeline(device, 'environment-sky-radiance', cubeSource, skyFormat),
            createBakePipeline(device, 'environment-prefilter', prefilterSource, 'rgba16float'),
            device.createRenderPipelineAsync({
                label: 'environment-brdf-integration',
                layout: 'auto',
                vertex: {
                    module: device.createShaderModule({ code: brdfSource }),
                    entryPoint: 'vs',
                },
                fragment: {
                    module: device.createShaderModule({ code: brdfSource }),
                    entryPoint: 'fs',
                    targets: [{ format: 'rgba16float' }],
                },
                primitive: { topology: 'triangle-list', cullMode: 'none' },
            }),
        ]);

        const encoder = device.createCommandEncoder({ label: 'environment-preprocess' });
        const sourceView = sourceTexture.createView();
        for (let face = 0; face < 6; face++) {
            const params = createUniformBuffer(device, new Float32Array([face, 0, 0, 0]));
            transientBuffers.push(params);
            const bindGroup = device.createBindGroup({
                layout: cubePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: sourceSampler },
                    { binding: 1, resource: sourceView },
                    { binding: 2, resource: { buffer: params } },
                ],
            });
            const pass = encoder.beginRenderPass({
                colorAttachments: [{
                    view: skyRadiance.createView({
                        dimension: '2d',
                        baseArrayLayer: face,
                        arrayLayerCount: 1,
                    }),
                    loadOp: 'clear',
                    storeOp: 'store',
                    clearValue: [0, 0, 0, 1],
                }],
            });
            pass.setPipeline(cubePipeline);
            pass.setBindGroup(0, bindGroup);
            pass.draw(3);
            pass.end();
        }
        for (let mip = 0; mip < mipCount; mip++) {
            const roughness = mipCount === 1 ? 0 : mip / (mipCount - 1);
            for (let face = 0; face < 6; face++) {
                const params = createUniformBuffer(device, new Float32Array([face, roughness, 0, 0]));
                transientBuffers.push(params);
                const bindGroup = device.createBindGroup({
                    layout: prefilterPipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: sourceSampler },
                        { binding: 1, resource: sourceView },
                        { binding: 2, resource: { buffer: params } },
                    ],
                });
                const pass = encoder.beginRenderPass({
                    colorAttachments: [{
                        view: specular.createView({
                            dimension: '2d',
                            baseMipLevel: mip,
                            mipLevelCount: 1,
                            baseArrayLayer: face,
                            arrayLayerCount: 1,
                        }),
                        loadOp: 'clear',
                        storeOp: 'store',
                        clearValue: [0, 0, 0, 1],
                    }],
                });
                pass.setPipeline(prefilterPipeline);
                pass.setBindGroup(0, bindGroup);
                pass.draw(3);
                pass.end();
            }
        }

        const brdfPass = encoder.beginRenderPass({
            colorAttachments: [{
                view: brdfLut.createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: [0, 0, 0, 1],
            }],
        });
        brdfPass.setPipeline(brdfPipeline);
        brdfPass.draw(3);
        brdfPass.end();

        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        dataBuffer = createLightingDataBuffer(device, sh, settings, mipCount);
        return {
            dataBuffer,
            sampler,
            skyRadiance,
            specular,
            specularMipCount: mipCount,
            brdfLut,
        };
    } catch (err) {
        skyRadiance?.destroy();
        specular?.destroy();
        brdfLut?.destroy();
        dataBuffer?.destroy();
        throw err;
    } finally {
        for (const buffer of transientBuffers) buffer.destroy();
        sourceTexture?.destroy();
    }
}

function createBakePipeline(
    device: GPUDevice,
    label: string,
    source: string,
    format: GPUTextureFormat,
): Promise<GPURenderPipeline> {
    const module = device.createShaderModule({ label, code: source });
    return device.createRenderPipelineAsync({
        label,
        layout: 'auto',
        vertex: { module, entryPoint: 'vs' },
        fragment: { module, entryPoint: 'fs', targets: [{ format }] },
        primitive: { topology: 'triangle-list', cullMode: 'none' },
    });
}

async function fetchShader(url: string): Promise<string> {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Environment preprocessing shader not found: ${url}`);
    return await response.text();
}

function clampPowerOfTwo(value: number, min: number, max: number): number {
    const finite = Number.isFinite(value) ? value : min;
    const clamped = Math.min(max, Math.max(min, finite));
    return 2 ** Math.round(Math.log2(clamped));
}

function createUniformBuffer(device: GPUDevice, values: Float32Array): GPUBuffer {
    const buffer = device.createBuffer({
        size: Math.max(16, values.byteLength),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Float32Array(buffer.getMappedRange()).set(values);
    buffer.unmap();
    return buffer;
}

function createLightingDataBuffer(
    device: GPUDevice,
    sh: Float32Array,
    settings: EnvironmentSettings,
    mipCount: number,
): GPUBuffer {
    const data = new Float32Array(44);
    data.set(sh, 0);
    const intensity = settings.lightingEnabled ? settings.intensity : 0;
    data.set([
        intensity,
        settings.lightingEnabled ? settings.diffuseIntensity : 0,
        settings.lightingEnabled ? settings.specularIntensity : 0,
        mipCount - 1,
    ], 36);
    const angle = settings.rotationDegrees * Math.PI / 180;
    data.set([
        Math.sin(angle),
        Math.cos(angle),
        settings.skyVisible ? settings.backgroundIntensity : 0,
        0,
    ], 40);
    return createEnvironmentDataBuffer(device, data);
}

/** Compute radiance SH9 and apply the Lambert convolution per frequency band. */
function computeDiffuseSh(source: EnvironmentSource): Float32Array {
    const width = 128;
    const height = 64;
    const pixels = new Uint8Array(source.pixels);
    const coefficients = new Float64Array(9 * 3);
    const dPhi = 2 * Math.PI / width;
    const dTheta = Math.PI / height;

    for (let y = 0; y < height; y++) {
        const theta = (y + 0.5) * dTheta;
        const sinTheta = Math.sin(theta);
        const directionY = Math.cos(theta);
        const weight = dPhi * dTheta * sinTheta;
        const sourceY = Math.min(
            source.height - 1,
            Math.floor((y + 0.5) * source.height / height),
        );
        for (let x = 0; x < width; x++) {
            const phi = (x + 0.5) * dPhi - Math.PI;
            const directionX = Math.cos(phi) * sinTheta;
            const directionZ = Math.sin(phi) * sinTheta;
            const basis = shBasis(directionX, directionY, directionZ);
            const sourceX = Math.min(
                source.width - 1,
                Math.floor((x + 0.5) * source.width / width),
            );
            const pixel = (sourceY * source.width + sourceX) * 4;
            const color = [
                srgbToLinear(pixels[pixel] / 255),
                srgbToLinear(pixels[pixel + 1] / 255),
                srgbToLinear(pixels[pixel + 2] / 255),
            ];
            for (let i = 0; i < 9; i++) {
                for (let channel = 0; channel < 3; channel++) {
                    coefficients[i * 3 + channel] += color[channel] * basis[i] * weight;
                }
            }
        }
    }

    const result = new Float32Array(9 * 4);
    for (let i = 0; i < 9; i++) {
        const convolution = i === 0 ? Math.PI : (i <= 3 ? 2 * Math.PI / 3 : Math.PI / 4);
        result[i * 4] = coefficients[i * 3] * convolution;
        result[i * 4 + 1] = coefficients[i * 3 + 1] * convolution;
        result[i * 4 + 2] = coefficients[i * 3 + 2] * convolution;
    }
    return result;
}

function shBasis(x: number, y: number, z: number): number[] {
    return [
        0.282095,
        0.488603 * y,
        0.488603 * z,
        0.488603 * x,
        1.092548 * x * y,
        1.092548 * y * z,
        0.315392 * (3 * z * z - 1),
        1.092548 * x * z,
        0.546274 * (x * x - y * y),
    ];
}

function srgbToLinear(value: number): number {
    return value <= 0.04045
        ? value / 12.92
        : ((value + 0.055) / 1.055) ** 2.4;
}
