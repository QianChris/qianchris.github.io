import { EnginePlugin, resourceManager, type PluginContext, type ValueContext, type AtomResolver } from '@shaderlab/api';

/**
 * Shared material assets. Publishes a `material` value-atom namespace so
 * pipeline JSON can write `material.baseColor` etc. instead of
 * `PbrMaterial.baseColor`. The resolver checks the entity's
 * `PbrMaterial.material` field: if it names a registered material asset,
 * the field is read from `ResourceManager.getMaterial(name)`; otherwise it
 * falls back to the entity's inline `PbrMaterial.<field>` (back-compat).
 *
 * A single `Gold` material asset can drive 100 entities; editing
 * `Gold.roughness` in the registry immediately affects all of them next
 * frame (the per-entity UBO write re-resolves the atom each frame).
 *
 * App-scoped: apps declare `"plugins": ["materials"]`. registerValueAtoms
 * is swept on unload via the plugin ledger.
 */
export default class MaterialsPlugin extends EnginePlugin {
    readonly meta = { id: 'materials', dependencies: ['core'] };

    setup(ctx: PluginContext): void {
        const atoms: Record<string, AtomResolver> = {};
        for (const f of ['baseColor', 'metallic', 'roughness', 'ao', 'emissive', 'shadowCast', 'shadowReceive']) {
            atoms[f] = (vc: ValueContext) => resolveMaterialField(vc, f);
        }
        // Texture handles live on the entity by default, but shared materials
        // can also override them: resolve from the asset when present.
        for (const f of ['texBaseColor', 'texMetalRough', 'texOcclusion', 'texEmissive', 'texNormal']) {
            atoms[f] = (vc: ValueContext) => resolveMaterialField(vc, f);
        }
        ctx.registerValueAtoms('material', atoms);
    }

    /** Fetch the app's materials.json (a { name: { baseColor, metallic, ... } }
     *  map) and register each as a shared material asset. Missing file is
     *  fine (apps may use only inline PbrMaterial fields). */
    async appLoaded(ctx: PluginContext, appBase: string): Promise<void> {
        const url = `${appBase}/materials.json`;
        try {
            const resp = await fetch(url);
            if (!resp.ok) return;
            const mats = await resp.json() as Record<string, Record<string, unknown>>;
            for (const [name, data] of Object.entries(mats)) {
                resourceManager.registerMaterial(name, data);
            }
        } catch {
            // No materials.json or malformed — silently use inline materials.
        }
    }
}

/** Resolve a material field for the current entity: shared asset (if
 *  PbrMaterial.material names one) or inline PbrMaterial fallback. */
function resolveMaterialField(ctx: ValueContext, field: string): number | number[] {
    const matName = ctx.scene.getField(ctx.eid, 'PbrMaterial', 'material') as string | undefined;
    if (matName) {
        const mat = resourceManager.getMaterial(matName);
        if (mat) {
            const v = mat[field];
            if (Array.isArray(v)) return v.map(Number);
            return Number(v ?? 0);
        }
    }
    // Inline fallback: read from the entity's PbrMaterial.<field>.
    const v = ctx.scene.getField(ctx.eid, 'PbrMaterial', field);
    if (Array.isArray(v)) return v.map(Number);
    return Number(v ?? 0);
}
