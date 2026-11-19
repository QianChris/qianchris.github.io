export type EnvironmentSourceEncoding = 'srgb-ldr';

export interface EnvironmentSourceRequest {
    sourceUrl: string;
    maxWidth: number;
}

export interface EnvironmentSource {
    sourceUrl: string;
    encoding: EnvironmentSourceEncoding;
    width: number;
    height: number;
    pixels: ArrayBuffer;
}

export interface EnvironmentSourceProvider {
    load(request: EnvironmentSourceRequest): Promise<EnvironmentSource>;
}

/** Static sRGB equirectangular image provider (PNG/JPEG and browser codecs). */
export class StaticImageEnvironmentSourceProvider implements EnvironmentSourceProvider {
    async load(request: EnvironmentSourceRequest): Promise<EnvironmentSource> {
        const response = await fetch(request.sourceUrl);
        if (!response.ok) {
            throw new Error(`Environment image not found: ${request.sourceUrl}`);
        }
        const bitmap = await createImageBitmap(await response.blob());
        try {
            const aspect = bitmap.width / bitmap.height;
            if (Math.abs(aspect - 2) > 0.02) {
                throw new Error(
                    `Environment image must be 2:1 equirectangular, got ${bitmap.width}x${bitmap.height}`,
                );
            }
            const width = Math.max(2, Math.min(bitmap.width, request.maxWidth));
            const height = Math.max(1, Math.round(width / 2));
            return {
                sourceUrl: request.sourceUrl,
                encoding: 'srgb-ldr',
                width,
                height,
                pixels: rasterize(bitmap, width, height),
            };
        } finally {
            bitmap.close();
        }
    }
}

function rasterize(bitmap: ImageBitmap, width: number, height: number): ArrayBuffer {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    if (!context) throw new Error(`Could not create a 2D canvas for environment decoding`);
    context.drawImage(bitmap, 0, 0, width, height);
    const source = context.getImageData(0, 0, width, height).data;
    const pixels = new Uint8Array(source.byteLength);
    pixels.set(source);
    return pixels.buffer;
}
