import { extname } from "path";
import type { LLMImageContent } from "./types";

export interface ImageInput {
  base64: string;
  mimeType: string;
}

export type ImageSize = "low" | "mid" | "high" | "xhigh" | "raw" | number;

const IMAGE_SIZE_MAP: Record<string, number> = {
  low: 256,
  mid: 512,
  high: 1024,
  xhigh: 1280,
};

const IMAGE_MIME_TYPES: Record<string, string> = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".webp": "image/webp",
};

const MIME_TO_SHARP_FORMAT: Record<string, string> = {
  "image/jpeg": "jpeg",
  "image/png": "png",
  "image/webp": "webp",
  "image/gif": "gif",
};

export function images(input: ImageInput | ImageInput[]): LLMImageContent[] {
  const inputs = Array.isArray(input) ? input : [input];
  return inputs.map(({ base64, mimeType }) => ({
    type: "image_url",
    image_url: { url: `data:${mimeType};base64,${base64}` },
  }));
}

export async function resizeImage(
  source: string | Uint8Array | ArrayBuffer,
  size: ImageSize,
  mimeType?: string,
): Promise<ImageInput> {
  const resolvedMime =
    mimeType ??
    (typeof source === "string"
      ? (IMAGE_MIME_TYPES[extname(source).toLowerCase()] ?? "image/jpeg")
      : "image/jpeg");

  let sharp: import("sharp").SharpConstructor;
  try {
    sharp = (await import("sharp")).default;
  } catch {
    throw new Error(
      'resizeImage() requires "sharp" to be installed. Run: bun add sharp'
    );
  }

  const input = source instanceof ArrayBuffer ? Buffer.from(source) : source;
  let img = sharp(input as Parameters<typeof sharp>[0]);

  if (size !== "raw") {
    const targetPx = typeof size === "number" ? size : IMAGE_SIZE_MAP[size]!;
    img = img.resize(targetPx, targetPx, { fit: "inside", withoutEnlargement: true });
  }

  const sharpFormat = MIME_TO_SHARP_FORMAT[resolvedMime] ?? "jpeg";
  const outputMime = MIME_TO_SHARP_FORMAT[resolvedMime] ? resolvedMime : "image/jpeg";
  const buf = await img.toFormat(sharpFormat as Parameters<typeof img.toFormat>[0]).toBuffer();
  return { base64: buf.toString("base64"), mimeType: outputMime };
}
