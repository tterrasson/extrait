import type {
  LLMRequest,
  LLMToolCall,
  LLMToolDebugOptions,
  LLMToolExecution,
  MCPToolClient,
  MCPToolDescriptor,
} from "../types";

export interface RuntimeToolCall {
  id?: string;
  type?: string;
  name?: string;
  arguments?: unknown;
}

export interface ResolvedMCPTool {
  name: string;
  remoteName: string;
  description?: string;
  inputSchema: Record<string, unknown>;
  clientId: string;
  client: MCPToolClient;
}

export interface ResolvedMCPToolset {
  tools: ResolvedMCPTool[];
  byName: Map<string, ResolvedMCPTool>;
}

export interface ExecuteMCPToolCallsOptions {
  round: number;
  request: LLMRequest;
  provider?: string;
  model?: string;
}

export interface ExecutedMCPToolCall {
  call: LLMToolCall;
  execution: LLMToolExecution;
}

export const DEFAULT_MAX_TOOL_ROUNDS = 8;

export async function resolveMCPToolset(clients: MCPToolClient[] | undefined): Promise<ResolvedMCPToolset> {
  if (!Array.isArray(clients) || clients.length === 0) {
    return {
      tools: [],
      byName: new Map(),
    };
  }

  const listed: Array<{ client: MCPToolClient; tool: MCPToolDescriptor }> = [];

  for (const client of clients) {
    let cursor: string | undefined;

    do {
      const page = await client.listTools(cursor ? { cursor } : undefined);
      for (const tool of page.tools ?? []) {
        listed.push({ client, tool });
      }
      cursor = page.nextCursor;
    } while (cursor);
  }

  const collisions = countNameCollisions(listed.map((entry) => entry.tool.name));
  const tools: ResolvedMCPTool[] = [];
  const byName = new Map<string, ResolvedMCPTool>();

  for (const entry of listed) {
    const name = collisions.get(entry.tool.name)! > 1
      ? `${sanitizeToolName(entry.client.id)}__${sanitizeToolName(entry.tool.name)}`
      : sanitizeToolName(entry.tool.name);

    const resolved: ResolvedMCPTool = {
      name,
      remoteName: entry.tool.name,
      description: describeTool(entry.client.id, entry.tool, collisions.get(entry.tool.name)! > 1),
      inputSchema: normalizeInputSchema(entry.tool.inputSchema),
      clientId: entry.client.id,
      client: entry.client,
    };

    tools.push(resolved);
    byName.set(name, resolved);
  }

  return {
    tools,
    byName,
  };
}

export function toProviderFunctionTools(toolset: ResolvedMCPToolset): Array<Record<string, unknown>> | undefined {
  if (toolset.tools.length === 0) {
    return undefined;
  }

  return toolset.tools.map((tool) => ({
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.inputSchema,
    },
  }));
}

export async function executeMCPToolCalls(
  calls: RuntimeToolCall[],
  toolset: ResolvedMCPToolset,
  context: ExecuteMCPToolCallsOptions,
): Promise<ExecutedMCPToolCall[]> {
  const out: ExecutedMCPToolCall[] = [];

  for (const call of calls) {
    const callId = call.id;
    const toolName = call.name;
    if (!callId || !toolName) {
      throw new Error("Received a function tool call without id or name.");
    }

    const tool = toolset.byName.get(toolName);
    const parsedArguments = parseToolArguments(call.arguments);

    if (!tool) {
      const errorMessage = context.request.unknownToolError
        ? context.request.unknownToolError(toolName)
        : `Tool "${toolName}" is not registered in the current toolset.`;

        const metadata: LLMToolCall = {
        id: callId,
        type: call.type ?? "function",
        name: toolName,
        arguments: parsedArguments,
        error: errorMessage,
      };

      const startedAt = new Date().toISOString();

      const execution: LLMToolExecution = {
        callId,
        type: metadata.type,
        name: toolName,
        clientId: "__unregistered__",
        remoteName: toolName,
        arguments: parsedArguments,
        error: errorMessage,
        round: context.round,
        provider: context.provider,
        model: context.model,
        handledLocally: true,
        startedAt,
        durationMs: 0,
      };

      emitToolExecution(context.request, execution);
      out.push({ call: metadata, execution });

      continue;
    }

    const rawArgs = isRecord(parsedArguments) ? (parsedArguments as Record<string, unknown>) : {};
    const args = context.request.transformToolArguments
      ? await context.request.transformToolArguments(rawArgs, {
          name: toolName,
          remoteName: tool.remoteName,
          clientId: tool.clientId,
        })
      : rawArgs;

    const metadata: LLMToolCall = {
      id: callId,
      type: call.type ?? "function",
      name: toolName,
      arguments: parsedArguments,
    };

    const startedAt = new Date().toISOString();
    const startedAtMs = Date.now();

    try {
      const output = await tool.client.callTool({
        name: tool.remoteName,
        arguments: args,
      });

      const executionContext = {
        callId,
        type: call.type ?? "function",
        name: toolName,
        clientId: tool.clientId,
        remoteName: tool.remoteName,
        arguments: parsedArguments,
        round: context.round,
        provider: context.provider,
        model: context.model,
        handledLocally: true as const,
        startedAt,
        error: undefined,
      };

      const transformedOutput = context.request.transformToolOutput
        ? await context.request.transformToolOutput(output, executionContext)
        : output;

      metadata.output = transformedOutput;
      const execution: LLMToolExecution = {
        ...executionContext,
        output: transformedOutput,
        durationMs: Date.now() - startedAtMs,
      };

      emitToolExecution(context.request, execution);
      out.push({ call: metadata, execution });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      metadata.error = message;

      const execution: LLMToolExecution = {
        callId,
        type: metadata.type,
        name: toolName,
        clientId: tool.clientId,
        remoteName: tool.remoteName,
        arguments: parsedArguments,
        error: message,
        round: context.round,
        provider: context.provider,
        model: context.model,
        handledLocally: true,
        startedAt,
        durationMs: Date.now() - startedAtMs,
      };

      emitToolExecution(context.request, execution);
      out.push({ call: metadata, execution });
    }
  }

  return out;
}

export function hasMCPClients(clients: MCPToolClient[] | undefined): boolean {
  return Array.isArray(clients) && clients.length > 0;
}

export function normalizeMaxToolRounds(value: number | undefined): number {
  if (!Number.isFinite(value)) {
    return DEFAULT_MAX_TOOL_ROUNDS;
  }

  return Math.max(0, Math.floor(value as number));
}

export function parseToolArguments(value: unknown): unknown {
  if (typeof value !== "string") {
    return value ?? {};
  }

  try {
    return JSON.parse(value);
  } catch {
    return {};
  }
}

export function stringifyToolOutput(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }

  return JSON.stringify(value ?? null);
}

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

function emitToolExecution(request: LLMRequest, execution: LLMToolExecution): void {
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
      logger: (line: string) => console.log(line),
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
    logger: value.logger ?? ((line: string) => console.log(line)),
    includeRequest: value.includeRequest ?? true,
    includeResult: value.includeResult ?? true,
    includeResultOnError: value.includeResultOnError ?? true,
    pretty: value.pretty ?? false,
  };
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

function countNameCollisions(names: string[]): Map<string, number> {
  const out = new Map<string, number>();
  for (const name of names) {
    out.set(name, (out.get(name) ?? 0) + 1);
  }
  return out;
}

function normalizeInputSchema(schema: unknown): Record<string, unknown> {
  if (!isRecord(schema)) {
    return {
      type: "object",
      properties: {},
    };
  }

  const out = { ...schema };
  if (out.type === undefined) {
    out.type = "object";
  }

  if (!isRecord(out.properties)) {
    out.properties = {};
  }

  return out;
}

function describeTool(clientId: string, tool: MCPToolDescriptor, hasCollision: boolean): string | undefined {
  const prefix = hasCollision ? `[${clientId}] ` : "";
  if (tool.description && tool.description.length > 0) {
    return `${prefix}${tool.description}`;
  }

  if (prefix.length > 0) {
    return `${prefix}${tool.name}`;
  }

  return undefined;
}

const RE_NON_ALPHANUMERIC = /[^A-Za-z0-9_]/g;
const RE_MULTIPLE_UNDERSCORES = /_+/g;
const RE_LEADING_UNDERSCORES = /^_+/;
const RE_TRAILING_UNDERSCORES = /_+$/;
const RE_STARTS_WITH_DIGIT = /^[0-9]/;

export function sanitizeToolName(input: string): string {
  const sanitized = input.replace(RE_NON_ALPHANUMERIC, "_").replace(RE_MULTIPLE_UNDERSCORES, "_");
  const trimmed = sanitized.replace(RE_LEADING_UNDERSCORES, "").replace(RE_TRAILING_UNDERSCORES, "");
  if (!trimmed) {
    return "tool";
  }

  if (RE_STARTS_WITH_DIGIT.test(trimmed)) {
    return `tool_${trimmed}`;
  }

  return trimmed;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
