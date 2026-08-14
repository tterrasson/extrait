import { describe, expect, test } from "bun:test";
import { join } from "path";
import { pathToFileURL } from "url";
import { images, loadImages, sniffMimeType } from "@/image";

const FIXTURE_PATH = join(import.meta.dir, "fixtures", "pixel.png");
const FIXTURE_BASE64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC";

const PNG_BYTES = Uint8Array.from(Buffer.from(FIXTURE_BASE64, "base64"));
const JPEG_BYTES = Uint8Array.from([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46]);
const GIF_BYTES = Uint8Array.from(Buffer.from("GIF89a-payload", "latin1"));
const WEBP_BYTES = Uint8Array.from(Buffer.from("RIFF\0\0\0\0WEBPVP8 ", "latin1"));
const AVIF_BYTES = Uint8Array.from(Buffer.from("\0\0\0\x20ftypavif", "latin1"));
// Major brand `mif1` (structural HEIF), `avif` only among the compatible brands.
const AVIF_MIF1_BYTES = Uint8Array.from(
  Buffer.from("\0\0\0\x18ftypmif1\0\0\0\0mif1avif", "latin1"),
);
const HEIC_BYTES = Uint8Array.from(
  Buffer.from("\0\0\0\x18ftypheic\0\0\0\0mif1heic", "latin1"),
);
const UNKNOWN_BYTES = Uint8Array.from([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]);

describe("images()", () => {
  test("keeps { base64, mimeType } support", () => {
    const result = images({ base64: "abc123", mimeType: "image/png" });

    expect(result).toEqual([
      { type: "image_url", image_url: { url: "data:image/png;base64,abc123" } },
    ]);
  });

  test("builds multiple image content blocks from an array", () => {
    const result = images([
      { base64: "abc123", mimeType: "image/png" },
      { base64: "def456", mimeType: "image/jpeg" },
    ]);

    expect(result).toEqual([
      { type: "image_url", image_url: { url: "data:image/png;base64,abc123" } },
      { type: "image_url", image_url: { url: "data:image/jpeg;base64,def456" } },
    ]);
  });

  test("always returns an array", () => {
    const single = images({ base64: "x", mimeType: "image/webp" });
    const multi = images([{ base64: "x", mimeType: "image/webp" }]);

    expect(Array.isArray(single)).toBe(true);
    expect(Array.isArray(multi)).toBe(true);
  });

  test("passes a data URL through byte for byte", () => {
    const url = `data:image/png;base64,${FIXTURE_BASE64}`;
    const [block] = images(url);
    expect(block?.image_url.url).toBe(url);
  });

  test("passes http(s) URLs through untouched", () => {
    const [http] = images("http://example.com/a.png");
    const [https] = images("https://example.com/b.jpg?v=1#frag");

    expect(http?.image_url.url).toBe("http://example.com/a.png");
    expect(https?.image_url.url).toBe("https://example.com/b.jpg?v=1#frag");
  });

  test("treats URI schemes as case-insensitive", () => {
    const upperData = `DATA:image/png;BASE64,${FIXTURE_BASE64}`;
    expect(images(upperData)[0]?.image_url.url).toBe(upperData);
    expect(images("HTTPS://example.com/a.png")[0]?.image_url.url).toBe(
      "HTTPS://example.com/a.png",
    );
    expect(images("Http://example.com/a.png")[0]?.image_url.url).toBe(
      "Http://example.com/a.png",
    );
  });

  test("accepts a URL object", () => {
    const [block] = images(new URL("https://example.com/photo.webp"));
    expect(block?.image_url.url).toBe("https://example.com/photo.webp");
  });

  test("sniffs PNG bytes", () => {
    const [block] = images(PNG_BYTES);
    expect(block?.image_url.url).toBe(`data:image/png;base64,${FIXTURE_BASE64}`);
  });

  test("sniffs JPEG bytes", () => {
    const [block] = images(JPEG_BYTES);
    expect(block?.image_url.url.startsWith("data:image/jpeg;base64,")).toBe(true);
  });

  test("accepts an ArrayBuffer", () => {
    const buffer = PNG_BYTES.buffer.slice(
      PNG_BYTES.byteOffset,
      PNG_BYTES.byteOffset + PNG_BYTES.byteLength,
    );
    const [block] = images(buffer as ArrayBuffer);
    expect(block?.image_url.url).toBe(`data:image/png;base64,${FIXTURE_BASE64}`);
  });

  test("preserves order across mixed sources", () => {
    const result = images([
      "https://example.com/1.png",
      PNG_BYTES,
      { base64: "zzz", mimeType: "image/gif" },
    ]);

    expect(result.map((block) => block.image_url.url)).toEqual([
      "https://example.com/1.png",
      `data:image/png;base64,${FIXTURE_BASE64}`,
      "data:image/gif;base64,zzz",
    ]);
  });

  test("rejects a file path and points to loadImages()", () => {
    expect(() => images("./photo.png")).toThrow(/loadImages/);
    expect(() => images(pathToFileURL(FIXTURE_PATH))).toThrow(/loadImages/);
  });

  test("rejects a bare base64 string", () => {
    expect(() => images(FIXTURE_BASE64)).toThrow(/file path/);
  });

  test("rejects bytes in an unknown format", () => {
    expect(() => images(UNKNOWN_BYTES)).toThrow(/Unable to detect the image format/);
  });
});

describe("loadImages()", () => {
  test("reads a file and infers its mime type", async () => {
    const [block] = await loadImages(FIXTURE_PATH);
    expect(block?.image_url.url).toBe(`data:image/png;base64,${FIXTURE_BASE64}`);
  });

  test("reads a file: URL", async () => {
    const [block] = await loadImages(pathToFileURL(FIXTURE_PATH));
    expect(block?.image_url.url).toBe(`data:image/png;base64,${FIXTURE_BASE64}`);
  });

  test("reads a Blob and uses its declared type", async () => {
    const blob = new Blob([JPEG_BYTES], { type: "image/jpeg" });
    const [block] = await loadImages(blob);
    expect(block?.image_url.url.startsWith("data:image/jpeg;base64,")).toBe(true);
  });

  test("sniffs a Blob with no declared type", async () => {
    const blob = new Blob([PNG_BYTES]);
    const [block] = await loadImages(blob);
    expect(block?.image_url.url).toBe(`data:image/png;base64,${FIXTURE_BASE64}`);
  });

  test("rejects a Blob whose bytes and type are both unusable", async () => {
    const blob = new Blob([UNKNOWN_BYTES]);
    expect(loadImages(blob)).rejects.toThrow(/Unable to detect the image format/);
  });

  test("mixes file paths, data URLs and remote URLs in one call, preserving order", async () => {
    const result = await loadImages([
      FIXTURE_PATH,
      "data:image/gif;base64,zzz",
      "https://example.com/remote.webp",
    ]);

    expect(result.map((block) => block.image_url.url)).toEqual([
      `data:image/png;base64,${FIXTURE_BASE64}`,
      "data:image/gif;base64,zzz",
      "https://example.com/remote.webp",
    ]);
  });

  test("does not read an upper-case scheme as a file path", async () => {
    const result = await loadImages(["HTTPS://example.com/a.png", "DATA:image/gif;base64,zzz"]);
    expect(result.map((block) => block.image_url.url)).toEqual([
      "HTTPS://example.com/a.png",
      "DATA:image/gif;base64,zzz",
    ]);
  });

  test("fails when the file does not exist", async () => {
    expect(loadImages(join(import.meta.dir, "fixtures", "missing.png"))).rejects.toThrow();
  });
});

describe("sniffMimeType()", () => {
  test("recognizes the common formats", () => {
    expect(sniffMimeType(PNG_BYTES)).toBe("image/png");
    expect(sniffMimeType(JPEG_BYTES)).toBe("image/jpeg");
    expect(sniffMimeType(GIF_BYTES)).toBe("image/gif");
    expect(sniffMimeType(WEBP_BYTES)).toBe("image/webp");
    expect(sniffMimeType(AVIF_BYTES)).toBe("image/avif");
    expect(sniffMimeType(HEIC_BYTES)).toBe("image/heic");
  });

  test("prefers avif over a structural mif1 major brand", () => {
    expect(sniffMimeType(AVIF_MIF1_BYTES)).toBe("image/avif");
  });

  test("returns undefined for unknown bytes", () => {
    expect(sniffMimeType(UNKNOWN_BYTES)).toBeUndefined();
    expect(sniffMimeType(new Uint8Array())).toBeUndefined();
  });

  test("an explicit mime type wins over the sniffed one", async () => {
    // PNG bytes declared as WebP by the Blob: the declaration is trusted.
    const blob = new Blob([PNG_BYTES], { type: "image/webp" });
    const [block] = await loadImages(blob);
    expect(block?.image_url.url).toBe(`data:image/webp;base64,${FIXTURE_BASE64}`);
  });
});
