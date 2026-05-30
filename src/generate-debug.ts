import { color, dim, title } from "./utils/debug-colors";
import type { LLMRequest, LLMUsage } from "./types";
import type { NormalizedDebugConfig } from "./generate-shared";

export interface DebugRequestInput {
  label: string;
  provider?: string;
  model?: string;
  attempt: number;
  selfHealAttempt: boolean;
  selfHealEnabled: boolean;
  stream: boolean;
  requestPayload: LLMRequest;
}

export interface DebugResponseInput {
  label: string;
  attempt: number;
  selfHealAttempt: boolean;
  selfHealEnabled: boolean;
  via: "complete" | "stream";
  text: string;
  reasoning: string;
  parseSource: string;
  usage?: LLMUsage;
  finishReason?: string;
}

export function emitDebugRequest(
  config: NormalizedDebugConfig,
  input: DebugRequestInput,
): void {
  const requestBody =
    input.requestPayload.body !== undefined
      ? JSON.stringify(input.requestPayload.body, null, 2)
      : "(none)";
  const requestMessages =
    input.requestPayload.messages !== undefined
      ? JSON.stringify(input.requestPayload.messages, null, 2)
      : "(none)";

  const lines = [
    color(
      config,
      title(
        config,
        [
          `[${input.label}][request]`,
          `attempt=${input.attempt}`,
          `selfHealEnabled=${input.selfHealEnabled}`,
          `selfHealAttempt=${input.selfHealAttempt}`,
        ].join(" "),
      ),
      "cyan",
    ),
    dim(
      config,
      [
        `provider=${input.provider ?? "unknown"}`,
        `model=${input.model ?? "unknown"}`,
        `stream=${input.stream}`,
      ].join(" "),
    ),
    color(config, "prompt:", "yellow"),
    input.requestPayload.prompt ?? "(none)",
    color(config, "messages:", "yellow"),
    requestMessages,
    color(config, "systemPrompt:", "yellow"),
    input.requestPayload.systemPrompt ?? "(none)",
    color(config, "request.body:", "yellow"),
    requestBody,
  ];

  emitDebug(config, lines.join("\n"));
}

export function emitDebugResponse(
  config: NormalizedDebugConfig,
  input: DebugResponseInput,
): void {
  const text = input.text.length > 0 ? input.text : "(none)";
  const reasoning = input.reasoning.length > 0 ? input.reasoning : "(none)";
  const metadata = [
    `via=${input.via}`,
    `textChars=${input.text.length}`,
    `reasoningChars=${input.reasoning.length}`,
  ];
  if (config.verbose) {
    metadata.push(`parseSourceChars=${input.parseSource.length}`);
  }
  metadata.push(
    `finishReason=${input.finishReason ?? "unknown"}`,
    `usage=${JSON.stringify(input.usage ?? {})}`,
  );
  const lines = [
    color(
      config,
      title(
        config,
        [
          `[${input.label}][response]`,
          `attempt=${input.attempt}`,
          `selfHealEnabled=${input.selfHealEnabled}`,
          `selfHealAttempt=${input.selfHealAttempt}`,
        ].join(" "),
      ),
      "green",
    ),
    dim(config, metadata.join(" ")),
    color(config, "text:", "yellow"),
    text,
    color(config, "reasoning:", "yellow"),
    reasoning,
  ];
  if (config.verbose) {
    lines.push(color(config, "parseSource:", "yellow"), input.parseSource);
  }

  emitDebug(config, lines.join("\n"));
}

function emitDebug(config: NormalizedDebugConfig, message: string): void {
  if (!config.enabled) {
    return;
  }

  config.logger(message);
}
