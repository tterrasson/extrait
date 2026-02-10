import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import {
  StdioClientTransport,
  type StdioServerParameters,
} from "@modelcontextprotocol/sdk/client/stdio.js";
import {
  StreamableHTTPClientTransport,
  type StreamableHTTPClientTransportOptions,
} from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import type { Transport } from "@modelcontextprotocol/sdk/shared/transport.js";
import type { Implementation } from "@modelcontextprotocol/sdk/types.js";
import type { MCPListToolsResult, MCPToolClient } from "./types";

export interface MCPClientInfo {
  name: string;
  version: string;
}

export interface MCPStdioTransportConfig extends Omit<StdioServerParameters, "command"> {
  type: "stdio";
  command: string;
}

export interface MCPStreamableHTTPTransportConfig {
  type: "streamable-http";
  url: string | URL;
  options?: StreamableHTTPClientTransportOptions;
}

export interface MCPInMemoryTransportConfig {
  type: "in-memory";
  transport: InMemoryTransport;
}

export type MCPTransportConfig =
  | MCPStdioTransportConfig
  | MCPStreamableHTTPTransportConfig
  | MCPInMemoryTransportConfig;

export interface CreateMCPClientOptions {
  id: string;
  transport: MCPTransportConfig;
  clientInfo?: MCPClientInfo;
}

export interface ManagedMCPToolClient extends MCPToolClient {
  sdkClient: Client;
  transport?: Transport;
}

export async function createMCPClient(options: CreateMCPClientOptions): Promise<ManagedMCPToolClient> {
  const client = new Client(toImplementation(options.clientInfo), {
    capabilities: {},
  });

  const transport = createTransport(options.transport);
  await client.connect(transport);

  return wrapMCPClient({
    id: options.id,
    client,
    transport,
  });
}

export interface WrapMCPClientOptions {
  id: string;
  client: Client;
  transport?: Transport;
}

export function wrapMCPClient(options: WrapMCPClientOptions): ManagedMCPToolClient {
  return {
    id: options.id,
    sdkClient: options.client,
    transport: options.transport,
    async listTools(params): Promise<MCPListToolsResult> {
      const response = await options.client.listTools(params);
      return {
        tools: response.tools.map((tool) => ({
          name: tool.name,
          description: tool.description,
          inputSchema: tool.inputSchema,
        })),
        nextCursor: response.nextCursor,
      };
    },
    async callTool(params) {
      return options.client.callTool(params);
    },
    async close() {
      await options.client.close();
      if (options.transport) {
        await options.transport.close();
      }
    },
  };
}

function createTransport(config: MCPTransportConfig): Transport {
  if (config.type === "stdio") {
    const stdio: StdioServerParameters = {
      command: config.command,
      args: config.args,
      env: config.env,
      stderr: config.stderr,
      cwd: config.cwd,
    };
    return new StdioClientTransport(stdio);
  }

  if (config.type === "streamable-http") {
    const url = typeof config.url === "string" ? new URL(config.url) : config.url;
    return new StreamableHTTPClientTransport(url, config.options);
  }

  return config.transport;
}

function toImplementation(clientInfo: MCPClientInfo | undefined): Implementation {
  return {
    name: clientInfo?.name ?? "extrait-mcp-client",
    version: clientInfo?.version ?? "0.1.0",
  };
}
