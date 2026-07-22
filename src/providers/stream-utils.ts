const RE_LINE_ENDING = /\r?\n/;

export async function consumeSSE(
  response: Response,
  onEvent: (data: string) => void,
): Promise<void> {
  if (!response.body) {
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });

    while (true) {
      const boundary = findSSEBoundary(buffer);
      if (boundary < 0) {
        break;
      }

      const rawEvent = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + (buffer.startsWith("\r\n\r\n", boundary) ? 4 : 2));

      const dataLines = extractSSEDataLines(rawEvent);

      if (dataLines.length === 0) {
        continue;
      }

      onEvent(dataLines.join("\n"));
    }
  }

  buffer += decoder.decode();
  if (buffer.trim().length > 0) {
    const dataLines = extractSSEDataLines(buffer);

    if (dataLines.length > 0) {
      onEvent(dataLines.join("\n"));
    }
  }
}

function extractSSEDataLines(rawEvent: string): string[] {
  return rawEvent
    .split(RE_LINE_ENDING)
    .filter((line) => line === "data" || line.startsWith("data:"))
    .map((line) => {
      if (line === "data") {
        return "";
      }

      const value = line.slice(5);
      return value.startsWith(" ") ? value.slice(1) : value;
    });
}

function findSSEBoundary(buffer: string): number {
  const crlfIndex = buffer.indexOf("\r\n\r\n");
  const lfIndex = buffer.indexOf("\n\n");

  if (crlfIndex >= 0 && lfIndex >= 0) {
    return Math.min(crlfIndex, lfIndex);
  }

  return Math.max(crlfIndex, lfIndex);
}
