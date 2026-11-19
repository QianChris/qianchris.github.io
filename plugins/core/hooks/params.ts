// Render escape-hatch value scripts, registered as script:params.<export>.
// Each receives a ValueContext and returns a number[] for a uniform member.
import type { ValueContext } from '@shaderlab/api';

/** Point sprite size — params.xy in NDC half-extents (matches TestPoint.wgsl).
 *  tag.extra holds the point size in pixels. */
export function point(ctx: ValueContext): number[] {
    const px = ctx.scene.getTagExtra(ctx.eid, ctx.tag);
    return [(px * 2) / ctx.screenH, (px * 2) / ctx.screenW, 0, 0];
}

/** Edge line width — params (x = width in NDC, y = aspect). */
export function edge(ctx: ValueContext): number[] {
    const w = ctx.scene.getTagExtra(ctx.eid, ctx.tag);
    return [w / ctx.screenH, ctx.aspect, 0, 0];
}
