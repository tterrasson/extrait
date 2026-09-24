// A stream is unbounded by design, but a single SSE event is not: without this
// cap a provider that never emits an event boundary grows the buffer forever.
const MAX_SSE_EVENT_CHARS = 8_000_000;

/**
 * Reads a `text/event-stream` body and hands each event's data to `onEvent`.
 *
 * Follows the WHATWG line grammar: a line ends with CRLF, LF or a lone CR, a
 * blank line dispatches the event, and an event whose data buffer is empty is
 * dropped. Only `data` matters to the providers (their payloads carry their own
 * `type`), so `event`, `id` and `retry` are ignored. Every character is scanned
 * once, however the body is chunked.
 */
export async function consumeSSE(
  response: Response,
  onEvent: (data: string) => void,
): Promise<void> {
  if (!response.body) {
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  // The unterminated line, kept as fragments so a long line arriving in many
  // chunks is joined once instead of being re-copied on every chunk.
  let partial: string[] = [];
  let partialChars = 0;
  // A CR ends a line at once; an LF right after it (possibly in the next
  // chunk) is the second half of a CRLF, not an empty line.
  let skipLF = false;
  let dataLines: string[] = [];
  let dataChars = 0;

  const processLine = (line: string): void => {
    if (line.length === 0) {
      if (dataLines.length > 0) {
        const data = dataLines.join("\n");
        dataLines = [];
        dataChars = 0;
        if (data.length > 0) {
          onEvent(data);
        }
      }
      return;
    }

    if (line === "data") {
      dataLines.push("");
      return;
    }

    if (!line.startsWith("data:")) {
      return;
    }

    const value = line.charCodeAt(5) === 32 ? line.slice(6) : line.slice(5);
    dataLines.push(value);
    dataChars += value.length;
  };

  const takeLine = (tail: string): string => {
    if (partial.length === 0) {
      return tail;
    }
    partial.push(tail);
    const line = partial.join("");
    partial = [];
    partialChars = 0;
    return line;
  };

  const processChunk = (chunk: string): void => {
    let lineStart = 0;
    if (skipLF && chunk.charCodeAt(0) === 10) {
      lineStart = 1;
    }
    skipLF = false;

    for (let index = lineStart; index < chunk.length; index += 1) {
      const code = chunk.charCodeAt(index);
      if (code !== 10 && code !== 13) {
        continue;
      }

      processLine(takeLine(chunk.slice(lineStart, index)));
      if (code === 13) {
        if (index + 1 === chunk.length) {
          skipLF = true;
        } else if (chunk.charCodeAt(index + 1) === 10) {
          index += 1;
        }
      }
      lineStart = index + 1;
    }

    if (lineStart < chunk.length) {
      partial.push(chunk.slice(lineStart));
      partialChars += chunk.length - lineStart;
    }
  };

  // `onEvent` throws on provider `error` events and malformed payloads. Without
  // the cancel below, that exception would escape while the reader still holds
  // the response body, leaking the connection until GC.
  let drained = false;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      processChunk(decoder.decode(value, { stream: true }));

      if (partialChars + dataChars > MAX_SSE_EVENT_CHARS) {
        throw new Error(
          `SSE stream exceeded ${MAX_SSE_EVENT_CHARS} characters without an event boundary.`,
        );
      }
    }

    processChunk(decoder.decode());
    // Lenient about a missing final blank line: a body that ends right after
    // its last `data:` line (or without any terminator) still delivers it.
    if (partial.length > 0) {
      processLine(takeLine(""));
    }
    processLine("");

    drained = true;
  } finally {
    if (!drained) {
      await reader.cancel().catch(() => {});
    }
  }
}
