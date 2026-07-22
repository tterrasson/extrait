import { describe, expect, test } from "bun:test";
import { consumeSSE } from "@/providers/stream-utils";

function mockResponse(chunks: string[]): Response {
  const encoder = new TextEncoder();
  let index = 0;
  const stream = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (index < chunks.length) {
        controller.enqueue(encoder.encode(chunks[index]));
        index++;
      } else {
        controller.close();
      }
    },
  });
  return new Response(stream);
}

function emptyBodyResponse(): Response {
  return new Response(null);
}

describe("consumeSSE", () => {
  test("does nothing when body is null", async () => {
    const events: string[] = [];
    await consumeSSE(emptyBodyResponse(), (data) => events.push(data));
    expect(events).toEqual([]);
  });

  test("parses a single SSE event with LF boundary", async () => {
    const events: string[] = [];
    const response = mockResponse(["data: hello\n\n"]);
    await consumeSSE(response, (data) => events.push(data));
    expect(events).toEqual(["hello"]);
  });

  test("parses a single SSE event with CRLF boundary", async () => {
    const events: string[] = [];
    const response = mockResponse(["data: world\r\n\r\n"]);
    await consumeSSE(response, (data) => events.push(data));
    expect(events).toEqual(["world"]);
  });

  test("parses multiple events in one chunk", async () => {
    const events: string[] = [];
    const response = mockResponse(["data: first\n\ndata: second\n\n"]);
    await consumeSSE(response, (data) => events.push(data));
    expect(events).toEqual(["first", "second"]);
  });

  test("joins multi-line data events", async () => {
    const events: string[] = [];
    const response = mockResponse(["data: line1\ndata: line2\n\n"]);
    await consumeSSE(response, (data) => events.push(data));
    expect(events).toEqual(["line1\nline2"]);
  });

  test("ignores events with no data: lines (comments)", async () => {
    const events: string[] = [];
    const response = mockResponse([": this is a comment\n\ndata: real\n\n"]);
    await consumeSSE(response, (data) => events.push(data));
    expect(events).toEqual(["real"]);
  });

  test("forwards [DONE] as raw data", async () => {
    const events: string[] = [];
    const response = mockResponse(["data: [DONE]\n\n"]);
    await consumeSSE(response, (data) => events.push(data));
    expect(events).toEqual(["[DONE]"]);
  });

  test("handles data split across chunks", async () => {
    const events: string[] = [];
    const response = mockResponse(["data: hel", "lo\n\n"]);
    await consumeSSE(response, (data) => events.push(data));
    expect(events).toEqual(["hello"]);
  });

  test("handles boundary split across chunks", async () => {
    const events: string[] = [];
    const response = mockResponse(["data: msg\n", "\ndata: next\n\n"]);
    await consumeSSE(response, (data) => events.push(data));
    expect(events).toEqual(["msg", "next"]);
  });

  test("emits remainder with data: lines after stream ends", async () => {
    const events: string[] = [];
    const response = mockResponse(["data: trailing"]);
    await consumeSSE(response, (data) => events.push(data));
    expect(events).toEqual(["trailing"]);
  });

  test("does not emit remainder without data: lines", async () => {
    const events: string[] = [];
    const response = mockResponse(["some non-data text"]);
    await consumeSSE(response, (data) => events.push(data));
    expect(events).toEqual([]);
  });

  test("handles mixed LF and CRLF boundaries", async () => {
    const events: string[] = [];
    const response = mockResponse(["data: a\r\n\r\ndata: b\n\n"]);
    await consumeSSE(response, (data) => events.push(data));
    expect(events).toEqual(["a", "b"]);
  });

  test("removes at most one optional space after the data field separator", async () => {
    const events: string[] = [];
    const response = mockResponse(["data:   spaced  \n\n"]);
    await consumeSSE(response, (data) => events.push(data));
    expect(events).toEqual(["  spaced  "]);
  });
});
