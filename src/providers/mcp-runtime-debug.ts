import type { LLMRequest, LLMToolDebugOptions, LLMToolExecution } from "../types";

export function formatToolExecutionDebugLine(execution: LLMToolExecution): string {
  const status = execution.error ? "error" : "ok";
  const scope = [execution.provider, execution.model].filter(Boolean).join("/") || "unknown";
  const toolRef = execution.clientId ? `${execution.clientId}:${execution.name ?? "unknown"}` : execution.name ?? "unknown";
  const duration = typeof execution.durationMs === "number" ? ` ${execution.durationMs}ms` : "";
  const base = `[tool:mcp:${status}] ${scope} ${toolRef}#${execution.callId}${duration}`;

  if (execution.error) {
    return `${base} -> ${execution.error}`;
  }

  return base;
}

export function emitToolExecution(request: LLMRequest, execution: LLMToolExecution): void {
  request.onToolExecution?.(execution);

  const debug = resolveToolDebugOptions(request.toolDebug);
  if (!debug.enabled) {
    return;
  }

  debug.logger(formatToolExecutionDebugLine(execution));
  if (debug.includeRequest) {
    debug.logger(formatToolExecutionRequestDebugLine(execution, debug));
  }
  if (debug.includeResult && (!execution.error || debug.includeResultOnError)) {
    debug.logger(formatToolExecutionResultDebugLine(execution, debug));
  }
}

function resolveToolDebugOptions(value: LLMRequest["toolDebug"]): Required<LLMToolDebugOptions> {
  if (value === true) {
    return {
      enabled: true,
      logger: defaultToolDebugLogger,
      includeRequest: true,
      includeResult: true,
      includeResultOnError: true,
      pretty: false,
    };
  }

  if (value === undefined || value === false) {
    return {
      enabled: false,
      logger: () => undefined,
      includeRequest: false,
      includeResult: false,
      includeResultOnError: false,
      pretty: false,
    };
  }

  return {
    enabled: value.enabled ?? true,
    logger: value.logger ?? defaultToolDebugLogger,
    includeRequest: value.includeRequest ?? true,
    includeResult: value.includeResult ?? true,
    includeResultOnError: value.includeResultOnError ?? true,
    pretty: value.pretty ?? false,
  };
}

function defaultToolDebugLogger(line: string): void {
  const { log } = globalThis.console;
  log(line);
}

function formatToolExecutionRequestDebugLine(
  execution: LLMToolExecution,
  debug: Required<LLMToolDebugOptions>,
): string {
  const scope = [execution.provider, execution.model].filter(Boolean).join("/") || "unknown";
  const toolRef = execution.clientId ? `${execution.clientId}:${execution.name ?? "unknown"}` : execution.name ?? "unknown";
  const payload = formatDebugPayload(execution.arguments, debug.pretty);
  return `[tool:mcp:request] ${scope} ${toolRef}#${execution.callId} arguments=${payload}`;
}

function formatToolExecutionResultDebugLine(
  execution: LLMToolExecution,
  debug: Required<LLMToolDebugOptions>,
): string {
  const scope = [execution.provider, execution.model].filter(Boolean).join("/") || "unknown";
  const toolRef = execution.clientId ? `${execution.clientId}:${execution.name ?? "unknown"}` : execution.name ?? "unknown";
  if (execution.error) {
    const payload = formatDebugPayload({ error: execution.error }, debug.pretty);
    return `[tool:mcp:result:error] ${scope} ${toolRef}#${execution.callId} output=${payload}`;
  }

  const payload = formatDebugPayload(execution.output, debug.pretty);
  return `[tool:mcp:result:ok] ${scope} ${toolRef}#${execution.callId} output=${payload}`;
}

function formatDebugPayload(value: unknown, pretty: boolean): string {
  if (value === undefined) {
    return "undefined";
  }

  try {
    const serialized = JSON.stringify(value, null, pretty ? 2 : 0);
    return serialized ?? "undefined";
  } catch {
    return String(value);
  }
}
