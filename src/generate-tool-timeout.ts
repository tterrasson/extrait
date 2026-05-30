import type { MCPToolClient } from "./types";

export function withToolTimeout(client: MCPToolClient, toolTimeoutMs: number): MCPToolClient {
  return {
    id: client.id,
    listTools: client.listTools.bind(client),
    close: client.close?.bind(client),
    async callTool(params) {
      let timeoutId: ReturnType<typeof setTimeout> | undefined;
      const timeoutPromise = new Promise<never>((_, reject) => {
        timeoutId = setTimeout(
          () => reject(new Error(`Tool call timed out after ${toolTimeoutMs}ms`)),
          toolTimeoutMs,
        );
      });
      try {
        return await Promise.race([client.callTool(params), timeoutPromise]);
      } finally {
        clearTimeout(timeoutId);
      }
    },
  };
}

export function applyToolTimeout(clients: MCPToolClient[], toolTimeoutMs: number): MCPToolClient[] {
  return clients.map((client) => withToolTimeout(client, toolTimeoutMs));
}
