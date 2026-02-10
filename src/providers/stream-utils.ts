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

      const dataLines = rawEvent
        .split(/\r?\n/)
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trim());

      if (dataLines.length === 0) {
        continue;
      }

      onEvent(dataLines.join("\n"));
    }
  }

  const remainder = buffer.trim();
  if (remainder.length > 0) {
    const dataLines = remainder
      .split(/\r?\n/)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trim());

    if (dataLines.length > 0) {
      onEvent(dataLines.join("\n"));
    }
  }
}

export async function consumeNDJSON(
  response: Response,
  onLine: (line: string) => void,
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
      const newLine = buffer.indexOf("\n");
      if (newLine < 0) {
        break;
      }

      const line = buffer.slice(0, newLine).trim();
      buffer = buffer.slice(newLine + 1);

      if (line.length > 0) {
        onLine(line);
      }
    }
  }

  const rest = buffer.trim();
  if (rest.length > 0) {
    onLine(rest);
  }
}

function findSSEBoundary(buffer: string): number {
  const crlfIndex = buffer.indexOf("\r\n\r\n");
  const lfIndex = buffer.indexOf("\n\n");

  if (crlfIndex >= 0 && lfIndex >= 0) {
    return Math.min(crlfIndex, lfIndex);
  }

  return Math.max(crlfIndex, lfIndex);
}
