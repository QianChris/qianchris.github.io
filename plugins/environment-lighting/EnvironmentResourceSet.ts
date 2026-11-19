import type { GpuResourceSet } from '@shaderlab/api';

export const ENVIRONMENT_RESOURCE_NAMES = {
    data: 'environment.data',
    sampler: 'environment.sampler',
    skyRadiance: 'environment.sky-radiance',
    specular: 'environment.specular',
    brdfLut: 'environment.brdf-lut',
} as const;

export const ENVIRONMENT_RESOURCE_SET_NAME = 'environment';

export interface EnvironmentResourceSet {
    dataBuffer: GPUBuffer;
    sampler: GPUSampler;
    skyRadiance: GPUTexture;
    specular: GPUTexture;
    specularMipCount: number;
    brdfLut: GPUTexture;
}

export function environmentResourceDescriptors(set: EnvironmentResourceSet): GpuResourceSet {
    return {
        [ENVIRONMENT_RESOURCE_NAMES.data]: {
            kind: 'buffer', buffer: set.dataBuffer, owned: true,
        },
        [ENVIRONMENT_RESOURCE_NAMES.sampler]: {
            kind: 'sampler', sampler: set.sampler,
        },
        [ENVIRONMENT_RESOURCE_NAMES.skyRadiance]: {
            kind: 'texture', texture: set.skyRadiance,
            viewDescriptor: { dimension: 'cube' }, owned: true,
        },
        [ENVIRONMENT_RESOURCE_NAMES.specular]: {
            kind: 'texture', texture: set.specular,
            viewDescriptor: { dimension: 'cube' }, owned: true,
        },
        [ENVIRONMENT_RESOURCE_NAMES.brdfLut]: {
            kind: 'texture', texture: set.brdfLut, owned: true,
        },
    };
}

export function createNeutralEnvironmentResourceSet(device: GPUDevice): EnvironmentResourceSet {
    const data = new Float32Array(44);
    data[41] = 1;
    const dataBuffer = createEnvironmentDataBuffer(device, data, 'environment-fallback-data');
    const sampler = createEnvironmentSampler(device, 'environment-fallback-sampler');
    const skyRadiance = createZeroTexture(device, 'environment-fallback-sky', {
        width: 1, height: 1, depthOrArrayLayers: 6,
    });
    const specular = createZeroTexture(device, 'environment-fallback-specular', {
        width: 1, height: 1, depthOrArrayLayers: 6,
    });
    const brdfLut = createZeroTexture(device, 'environment-fallback-dfg', {
        width: 1, height: 1,
    });
    return {
        dataBuffer,
        sampler,
        skyRadiance,
        specular,
        specularMipCount: 1,
        brdfLut,
    };
}

export function createEnvironmentSampler(device: GPUDevice, label = 'environment-sampler'): GPUSampler {
    return device.createSampler({
        label,
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge',
        addressModeW: 'clamp-to-edge',
        magFilter: 'linear',
        minFilter: 'linear',
        mipmapFilter: 'linear',
    });
}

export function createEnvironmentDataBuffer(
    device: GPUDevice,
    data: Float32Array,
    label = 'environment-data',
): GPUBuffer {
    const buffer = device.createBuffer({
        label,
        size: Math.max(16, data.byteLength),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Float32Array(buffer.getMappedRange()).set(data);
    buffer.unmap();
    return buffer;
}

export function destroyUnpublishedEnvironmentResourceSet(set: EnvironmentResourceSet): void {
    set.dataBuffer.destroy();
    set.skyRadiance.destroy();
    set.specular.destroy();
    set.brdfLut.destroy();
}

function createZeroTexture(device: GPUDevice, label: string, size: GPUExtent3D): GPUTexture {
    return device.createTexture({
        label,
        size,
        format: 'rgba16float',
        usage: GPUTextureUsage.TEXTURE_BINDING,
    });
}
