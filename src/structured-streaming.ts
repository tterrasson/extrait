import { jsonrepair } from "jsonrepair";

// Takes the already-normalized visible text (think blocks stripped). The
// reasoning channel is intentionally not scanned: the previous implementation
// re-sanitized `<think>${reasoning}</think>${text}`, which masked the entire
// reasoning with whitespace before searching for a JSON root, so the search
// only ever operated on the visible text.
export function parseStreamingStructuredData(visibleText: string): unknown | null {
  const start = findFirstJsonRootStart(visibleText);
  if (start < 0) {
    return null;
  }

  const candidate = visibleText.slice(start).trim();
  if (!candidate) {
    return null;
  }

  try {
    const repaired = jsonrepair(candidate);
    const parsed = JSON.parse(repaired);
    if (typeof parsed !== "object" || parsed === null) {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

function findFirstJsonRootStart(input: string): number {
  const unquotedRootStart = findFirstUnquotedJsonRootStart(input);
  if (unquotedRootStart >= 0) {
    return unquotedRootStart;
  }

  return findFirstRawJsonRootStart(input);
}

function findFirstUnquotedJsonRootStart(input: string): number {
  let inString = false;
  let escaped = false;

  for (let index = 0; index < input.length; index += 1) {
    const char = input[index];
    if (!char) {
      continue;
    }

    if (inString) {
      if (escaped) {
        escaped = false;
        continue;
      }

      if (char === "\\") {
        escaped = true;
        continue;
      }

      if (char === '"') {
        inString = false;
      }

      continue;
    }

    if (char === '"') {
      inString = true;
      continue;
    }

    if (char === "{" || char === "[") {
      return index;
    }
  }

  return -1;
}

function findFirstRawJsonRootStart(input: string): number {
  const objectStart = input.indexOf("{");
  const arrayStart = input.indexOf("[");

  if (objectStart < 0) {
    return arrayStart;
  }
  if (arrayStart < 0) {
    return objectStart;
  }

  return Math.min(objectStart, arrayStart);
}
