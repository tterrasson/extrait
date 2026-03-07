import { describe, expect, test } from "bun:test";
import { images, resizeImage } from "@/image";

// @ts-ignore - sharp is an optional peer dependency
let sharp: any = null;
try {
  // @ts-ignore
  sharp = (await import("sharp")).default;
} catch {
  // sharp not installed, skip related tests
}

const testWithSharp = sharp ? test : test.skip;

async function makeImage(width: number, height: number): Promise<Buffer> {
  return sharp({ create: { width, height, channels: 3, background: { r: 128, g: 128, b: 128 } } })
    .png()
    .toBuffer();
}

describe("images()", () => {
  test("builds a single image content block", () => {
    const result = images({ base64: "abc123", mimeType: "image/png" });

    expect(result).toEqual([
      {
        type: "image_url",
        image_url: { url: "data:image/png;base64,abc123" },
      },
    ]);
  });

  test("builds multiple image content blocks from an array", () => {
    const result = images([
      { base64: "abc123", mimeType: "image/png" },
      { base64: "def456", mimeType: "image/jpeg" },
    ]);

    expect(result).toEqual([
      {
        type: "image_url",
        image_url: { url: "data:image/png;base64,abc123" },
      },
      {
        type: "image_url",
        image_url: { url: "data:image/jpeg;base64,def456" },
      },
    ]);
  });

  test("always returns an array", () => {
    const single = images({ base64: "x", mimeType: "image/webp" });
    const multi = images([{ base64: "x", mimeType: "image/webp" }]);

    expect(Array.isArray(single)).toBe(true);
    expect(Array.isArray(multi)).toBe(true);
  });

  test("constructs the data URL correctly", () => {
    const [block] = images({ base64: "SGVsbG8=", mimeType: "image/gif" });
    expect(block?.image_url.url).toBe("data:image/gif;base64,SGVsbG8=");
  });
});

describe("resizeImage()", () => {
  testWithSharp("resizes landscape image: longest side becomes target", async () => {
    const buf = await makeImage(2000, 800);
    const result = await resizeImage(buf, "high"); // 1024px
    const { width, height } = await sharp(Buffer.from(result.base64, "base64")).metadata();
    expect(width).toBe(1024);
    expect(height).toBe(410); // 800 * (1024/2000)
  });

  testWithSharp("resizes portrait image: longest side becomes target", async () => {
    const buf = await makeImage(800, 2000);
    const result = await resizeImage(buf, "mid"); // 512px
    const { width, height } = await sharp(Buffer.from(result.base64, "base64")).metadata();
    expect(height).toBe(512);
    expect(width).toBe(205); // 800 * (512/2000)
  });

  testWithSharp("does not enlarge images smaller than target", async () => {
    const buf = await makeImage(200, 100);
    const result = await resizeImage(buf, "low"); // 256px
    const { width, height } = await sharp(Buffer.from(result.base64, "base64")).metadata();
    expect(width).toBe(200);
    expect(height).toBe(100);
  });

  testWithSharp("raw: no resize", async () => {
    const buf = await makeImage(3000, 1500);
    const result = await resizeImage(buf, "raw");
    const { width, height } = await sharp(Buffer.from(result.base64, "base64")).metadata();
    expect(width).toBe(3000);
    expect(height).toBe(1500);
  });

  testWithSharp("xhigh: resizes to 1280px", async () => {
    const buf = await makeImage(2000, 800);
    const result = await resizeImage(buf, "xhigh"); // 1280px
    const { width, height } = await sharp(Buffer.from(result.base64, "base64")).metadata();
    expect(width).toBe(1280);
    expect(height).toBe(512); // 800 * (1280/2000)
  });

  testWithSharp("numeric size", async () => {
    const buf = await makeImage(1000, 500);
    const result = await resizeImage(buf, 640);
    const { width } = await sharp(Buffer.from(result.base64, "base64")).metadata();
    expect(width).toBe(640);
  });

  testWithSharp("auto-detects mimeType from file extension", async () => {
    const buf = await makeImage(100, 100);
    await Bun.write("/tmp/test-extrait.png", buf);
    const result = await resizeImage("/tmp/test-extrait.png", "raw");
    expect(result.mimeType).toBe("image/png");
  });

  testWithSharp("explicit mimeType overrides auto-detection", async () => {
    const buf = await makeImage(100, 100);
    const result = await resizeImage(buf, "raw", "image/webp");
    expect(result.mimeType).toBe("image/webp");
  });

  testWithSharp("returns valid base64", async () => {
    const buf = await makeImage(100, 100);
    const result = await resizeImage(buf, "low");
    expect(() => Buffer.from(result.base64, "base64")).not.toThrow();
    expect(result.base64.length).toBeGreaterThan(0);
  });
});
