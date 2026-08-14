import { readFile } from "fs/promises";
import { extname } from "path";
import { fileURLToPath } from "url";
import type { LLMImageContent } from "./types";

export interface ImageInput {
  base64: string;
  mimeType: string;
}

/** Everything `images()` can normalize without any I/O. */
export type SyncImageSource =
  | string
  | URL
  | Uint8Array
  | ArrayBuffer
  | ImageInput;

/** Everything `loadImages()` accepts — sync sources plus file paths and blobs. */
export type ImageSource = SyncImageSource | Blob;

const IMAGE_MIME_TYPES: Record<string, string> = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".webp": "image/webp",
  ".avif": "image/avif",
  ".heic": "image/heic",
  ".bmp": "image/bmp",
};

const SUPPORTED_EXTENSIONS = Object.keys(IMAGE_MIME_TYPES).join(", ");

/**
 * Detects an image mime type from its magic bytes.
 * Returns `undefined` when the format is not recognized.
 */
export function sniffMimeType(bytes: Uint8Array): string | undefined {
  const at = (index: number): number => bytes[index] ?? -1;
  const ascii = (start: number, end: number): string =>
    String.fromCharCode(...bytes.subarray(start, end));

  if (
    at(0) === 0x89 && at(1) === 0x50 && at(2) === 0x4e && at(3) === 0x47 &&
    at(4) === 0x0d && at(5) === 0x0a && at(6) === 0x1a && at(7) === 0x0a
  ) {
    return "image/png";
  }

  if (at(0) === 0xff && at(1) === 0xd8 && at(2) === 0xff) {
    return "image/jpeg";
  }

  if (ascii(0, 4) === "GIF8") {
    return "image/gif";
  }

  if (ascii(0, 4) === "RIFF" && ascii(8, 12) === "WEBP") {
    return "image/webp";
  }

  if (ascii(4, 8) === "ftyp") {
    // Major brand at 8..12, minor version at 12..16, then the compatible brands.
    // `mif1` alone proves nothing: an AVIF may declare it as major brand and list
    // `avif` among the compatible ones, so every brand is inspected.
    const boxSize = Math.min(
      (at(0) << 24) | (at(1) << 16) | (at(2) << 8) | at(3),
      bytes.length,
    );
    const brands = [ascii(8, 12)];
    for (let offset = 16; offset + 4 <= boxSize; offset += 4) {
      brands.push(ascii(offset, offset + 4));
    }

    if (brands.some((brand) => brand === "avif" || brand === "avis")) {
      return "image/avif";
    }
    if (
      brands.some(
        (brand) =>
          brand.startsWith("hei") ||
          brand.startsWith("hev") ||
          brand === "mif1" ||
          brand === "msf1",
      )
    ) {
      return "image/heic";
    }
  }

  if (ascii(0, 2) === "BM") {
    return "image/bmp";
  }

  return undefined;
}

function toBytes(source: Uint8Array | ArrayBuffer): Uint8Array {
  return source instanceof ArrayBuffer ? new Uint8Array(source) : source;
}

function toDataURL(bytes: Uint8Array, mimeType: string): string {
  return `data:${mimeType};base64,${Buffer.from(bytes).toString("base64")}`;
}

function resolveBytesMimeType(bytes: Uint8Array, hint?: string): string {
  const mimeType = hint ?? sniffMimeType(bytes);
  if (!mimeType) {
    throw new Error(
      "Unable to detect the image format from its bytes. Pass { base64, mimeType } instead, " +
        "or use a file with one of these extensions: " +
        SUPPORTED_EXTENSIONS,
    );
  }
  return mimeType;
}

function isImageInput(value: unknown): value is ImageInput {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as ImageInput).base64 === "string" &&
    typeof (value as ImageInput).mimeType === "string"
  );
}

// URI schemes are case-insensitive: `HTTPS://…` and `Data:…` are valid.
function isRemoteURL(value: string): boolean {
  return /^https?:\/\//i.test(value);
}

function isDataURL(value: string): boolean {
  return /^data:/i.test(value);
}

/** Resolves a `URL` to either a passthrough URL string or a local file path. */
function splitURL(url: URL): { url?: string; path?: string } {
  if (url.protocol === "file:") {
    return { path: fileURLToPath(url) };
  }
  if (url.protocol === "http:" || url.protocol === "https:" || url.protocol === "data:") {
    return { url: url.href };
  }
  throw new Error(`Unsupported image URL protocol: ${url.protocol}`);
}

/**
 * Normalizes a source that requires no I/O into a URL string usable by providers.
 * Data URLs and http(s) URLs are passed through byte for byte — nothing is re-encoded.
 */
function toImageURL(source: SyncImageSource): string {
  if (typeof source === "string") {
    if (isDataURL(source) || isRemoteURL(source)) {
      return source;
    }
    throw new Error(
      `images() only accepts data URLs, http(s) URLs, raw bytes or { base64, mimeType }. ` +
        `"${source}" looks like a file path — use await loadImages(...) instead.`,
    );
  }

  if (source instanceof URL) {
    const { url, path } = splitURL(source);
    if (url) return url;
    throw new Error(
      `images() cannot read the local file "${path}" — use await loadImages(...) instead.`,
    );
  }

  if (source instanceof Uint8Array || source instanceof ArrayBuffer) {
    const bytes = toBytes(source);
    return toDataURL(bytes, resolveBytesMimeType(bytes));
  }

  if (isImageInput(source)) {
    return `data:${source.mimeType};base64,${source.base64}`;
  }

  if (typeof Blob !== "undefined" && (source as unknown) instanceof Blob) {
    throw new Error("images() cannot read a Blob — use await loadImages(...) instead.");
  }

  throw new Error("Unsupported image source.");
}

function toContent(url: string): LLMImageContent {
  return { type: "image_url", image_url: { url } };
}

/**
 * Builds image content blocks from sources that need no I/O:
 * data URLs, http(s) URLs, raw bytes, or `{ base64, mimeType }`.
 *
 * File paths and `Blob`s require `loadImages()`.
 */
export function images(input: SyncImageSource | SyncImageSource[]): LLMImageContent[] {
  const inputs = Array.isArray(input) ? input : [input];
  return inputs.map((source) => toContent(toImageURL(source)));
}

async function loadImageURL(source: ImageSource): Promise<string> {
  if (typeof Blob !== "undefined" && source instanceof Blob) {
    const bytes = new Uint8Array(await source.arrayBuffer());
    return toDataURL(bytes, resolveBytesMimeType(bytes, source.type || undefined));
  }

  let path: string | undefined;

  if (typeof source === "string" && !isDataURL(source) && !isRemoteURL(source)) {
    path = source;
  } else if (source instanceof URL) {
    path = splitURL(source).path;
  }

  if (path !== undefined) {
    const bytes = new Uint8Array(await readFile(path));
    const extensionMime = IMAGE_MIME_TYPES[extname(path).toLowerCase()];
    return toDataURL(bytes, sniffMimeType(bytes) ?? resolveBytesMimeType(bytes, extensionMime));
  }

  return toImageURL(source as SyncImageSource);
}

/**
 * Builds image content blocks from any source, including file paths and `Blob`s.
 * Images are transmitted as-is: nothing is decoded, resized or re-encoded.
 */
export async function loadImages(
  input: ImageSource | ImageSource[],
): Promise<LLMImageContent[]> {
  const inputs = Array.isArray(input) ? input : [input];
  const urls = await Promise.all(inputs.map(loadImageURL));
  return urls.map(toContent);
}
