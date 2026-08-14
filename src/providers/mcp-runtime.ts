import type {
  LLMRequest,
  LLMToolCall,
  LLMToolExecution,
  MCPToolClient,
  MCPToolDescriptor,
} from "../types";
import { emitToolExecution } from "./mcp-runtime-debug";

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

export const DEFAULT_MAX_TOOL_ROUNDS = 100;

// OpenAI and Anthropic both reject function names longer than 64 characters.
// Remote MCP servers are free to expose longer ones, and the `clientId__toolName`
// prefix added on collision makes it easy to cross the limit.
export const MAX_TOOL_NAME_LENGTH = 64;

// Budget for the disambiguating `clientId__` prefix, so a long client id can
// never eat the whole allowance and collapse every tool onto the same name.
const MAX_TOOL_NAME_CLIENT_PREFIX_LENGTH = 20;

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
    const name = uniqueToolName(
      byName,
      collisions.get(entry.tool.name)! > 1
        ? prefixedToolName(entry.client.id, entry.tool.name)
        : capToolNameLength(sanitizeToolName(entry.tool.name)),
    );

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
    const parsed = parseToolArgumentsResult(call.arguments);
    const parsedArguments = parsed.value;

    // Malformed arguments are reported back instead of running the tool with an
    // empty payload, which would hide the failure and may hit unintended defaults.
    const errorMessage = tool
      ? parsed.error
      : context.request.unknownToolError
        ? context.request.unknownToolError(toolName)
        : `Tool "${toolName}" is not registered in the current toolset.`;

    if (!tool || errorMessage !== undefined) {
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
        clientId: tool?.clientId ?? "__unregistered__",
        remoteName: tool?.remoteName ?? toolName,
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

    const toolParams = context.request.transformToolCallParams
      ? await context.request.transformToolCallParams(
          {
            name: tool.remoteName,
            arguments: args,
          },
          {
            name: toolName,
            remoteName: tool.remoteName,
            clientId: tool.clientId,
          },
        )
      : {
          name: tool.remoteName,
          arguments: args,
        };

    const metadata: LLMToolCall = {
      id: callId,
      type: call.type ?? "function",
      name: toolName,
      arguments: parsedArguments,
    };

    const startedAt = new Date().toISOString();
    const startedAtMs = Date.now();

    try {
      const output = await tool.client.callTool(toolParams);

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
  return parseToolArgumentsResult(value).value;
}

export interface ParsedToolArguments {
  value: unknown;
  error?: string;
}

/**
 * Same parsing as {@link parseToolArguments}, but reports malformed JSON instead
 * of silently degrading to `{}` — a tool must not run with no arguments just
 * because the model emitted a broken payload.
 */
export function parseToolArgumentsResult(value: unknown): ParsedToolArguments {
  if (typeof value !== "string") {
    return { value: value ?? {} };
  }

  try {
    return { value: JSON.parse(value) };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { value: {}, error: `Tool arguments are not valid JSON: ${message}` };
  }
}

export function stringifyToolOutput(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }

  return JSON.stringify(value ?? null);
}

export { emitToolExecution, formatToolExecutionDebugLine } from "./mcp-runtime-debug";

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

export function capToolNameLength(name: string, maxLength = MAX_TOOL_NAME_LENGTH): string {
  if (name.length <= maxLength) {
    return name;
  }

  const truncated = name.slice(0, maxLength).replace(RE_TRAILING_UNDERSCORES, "");
  return truncated || "tool";
}

function prefixedToolName(clientId: string, toolName: string): string {
  const prefix = capToolNameLength(sanitizeToolName(clientId), MAX_TOOL_NAME_CLIENT_PREFIX_LENGTH);
  const available = MAX_TOOL_NAME_LENGTH - prefix.length - "__".length;
  return `${prefix}__${capToolNameLength(sanitizeToolName(toolName), available)}`;
}

// Truncation (and sanitization before it) can map two distinct remote names onto
// the same provider-facing name, which would silently drop a tool from `byName`.
function uniqueToolName(taken: Map<string, ResolvedMCPTool>, name: string): string {
  if (!taken.has(name)) {
    return name;
  }

  // Terminates: for a given digit count every suffix yields a distinct candidate
  // of the same shape, so at worst it exhausts the names already taken.
  for (let suffix = 2; ; suffix += 1) {
    const marker = `_${suffix}`;
    const stem = name
      .slice(0, MAX_TOOL_NAME_LENGTH - marker.length)
      .replace(RE_TRAILING_UNDERSCORES, "");
    const candidate = `${stem}${marker}`;
    if (!taken.has(candidate)) {
      return candidate;
    }
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
