import {
    hasComponent,
    schemaRegistry,
    type FrameContext,
    type System,
} from '@shaderlab/api';

export interface EnvironmentSettings {
    source: string;
    lightingEnabled: boolean;
    skyVisible: boolean;
    intensity: number;
    diffuseIntensity: number;
    specularIntensity: number;
    backgroundIntensity: number;
    rotationDegrees: number;
    skyResolution: number;
    specularResolution: number;
}

export function readEnvironmentSettings(eid: number): EnvironmentSettings {
    const comp = schemaRegistry.get('EnvironmentLightComponent');
    if (!comp) throw new Error(`EnvironmentLightComponent is not registered`);
    return {
        source: String(schemaRegistry.getComposite('EnvironmentLightComponent', comp, eid, 'source')),
        lightingEnabled: schemaRegistry.getScalar(comp, eid, 'lightingEnabled') > 0.5,
        skyVisible: schemaRegistry.getScalar(comp, eid, 'skyVisible') > 0.5,
        intensity: schemaRegistry.getScalar(comp, eid, 'intensity'),
        diffuseIntensity: schemaRegistry.getScalar(comp, eid, 'diffuseIntensity'),
        specularIntensity: schemaRegistry.getScalar(comp, eid, 'specularIntensity'),
        backgroundIntensity: schemaRegistry.getScalar(comp, eid, 'backgroundIntensity'),
        rotationDegrees: schemaRegistry.getScalar(comp, eid, 'rotation'),
        skyResolution: schemaRegistry.getScalar(comp, eid, 'skyResolution'),
        specularResolution: schemaRegistry.getScalar(comp, eid, 'specularResolution'),
    };
}

export class EnvironmentLightingSystem implements System {
    private dataBuffer: GPUBuffer | null = null;
    private specularMipCount = 1;
    private componentEid: number | null = null;
    private componentChanged: (settings: EnvironmentSettings | null) => void;
    private pendingError: Error | null = null;

    constructor(componentChanged: (settings: EnvironmentSettings | null) => void) {
        this.componentChanged = componentChanged;
    }

    setRuntime(dataBuffer: GPUBuffer, specularMipCount: number): void {
        this.dataBuffer = dataBuffer;
        this.specularMipCount = specularMipCount;
    }

    setComponentEid(eid: number | null): void {
        this.componentEid = eid;
    }

    fail(error: unknown): void {
        this.pendingError = error instanceof Error ? error : new Error(String(error));
    }

    clear(): void {
        this.dataBuffer = null;
        this.specularMipCount = 1;
        this.componentEid = null;
        this.pendingError = null;
    }

    update(ctx: FrameContext): void {
        if (this.pendingError) {
            const error = this.pendingError;
            this.pendingError = null;
            throw error;
        }
        if (!this.dataBuffer) return;
        const comp = schemaRegistry.get('EnvironmentLightComponent');
        if (!comp) throw new Error(`EnvironmentLightComponent is not registered`);

        let activeEid: number | null = null;
        for (const [, eid] of ctx.scene.entityKeyMap) {
            if (!hasComponent(ctx.scene.world, comp, eid)) continue;
            if (activeEid !== null) {
                throw new Error(`environment-lighting supports at most one EnvironmentLightComponent`);
            }
            activeEid = eid;
        }

        const settings = activeEid === null ? null : readEnvironmentSettings(activeEid);
        if (activeEid !== this.componentEid) {
            this.componentEid = activeEid;
            this.componentChanged(settings);
        }
        const lightingEnabled = settings?.lightingEnabled ?? false;
        const skyVisible = settings?.skyVisible ?? false;
        const angle = (settings?.rotationDegrees ?? 0) * Math.PI / 180;
        const params = new Float32Array([
            lightingEnabled ? settings!.intensity : 0,
            lightingEnabled ? settings!.diffuseIntensity : 0,
            lightingEnabled ? settings!.specularIntensity : 0,
            this.specularMipCount - 1,
            Math.sin(angle),
            Math.cos(angle),
            skyVisible ? settings!.backgroundIntensity : 0,
            0,
        ]);
        ctx.device.queue.writeBuffer(this.dataBuffer, 9 * 16, params);
    }
}
