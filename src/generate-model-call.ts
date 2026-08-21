import type {
  LLMAdapter,
  LLMRequest,
  LLMToolCall,
  LLMUsage,
  ReasoningBlock,
} from "./types";
import { preferLatestUsage as preferLatestStreamUsage } from "./providers/utils";
import {
  appendReasoningBlock,
  normalizeModelOutput,
  toStreamDataFingerprint,
  withoutTrailingThinkTagPrefix,
} from "./generate-output";
import type { ModelCallOptions, ModelCallResult } from "./generate-shared";
import { emitDebugRequest, emitDebugResponse } from "./generate-debug";

/**
 * Folds the caller's cancellation signal and the configured per-request timeout
 * into the single signal the adapter runs under.
 *
 * Both matter, and they are not interchangeable: the caller's signal ends a call
 * nobody is waiting for any more, the timeout ends one the provider never
 * answers. Honoring only the caller's — as skipping the timeout whenever a
 * signal is present would — leaves a stalled upstream holding the call forever,
 * which is precisely the case the timeout exists for.
 */
function withRequestTimeout(
  signal: AbortSignal | undefined,
  timeoutMs: number | undefined,
): AbortSignal | undefined {
  if (timeoutMs === undefined) {
    return signal;
  }
  const timeout = AbortSignal.timeout(timeoutMs);
  return signal ? AbortSignal.any([signal, timeout]) : timeout;
}

export async function callModel<TSnapshot, TTraceEvent>(
  adapter: LLMAdapter,
  options: ModelCallOptions<TSnapshot, TTraceEvent>,
): Promise<ModelCallResult> {
  const requestSignal = withRequestTimeout(options.request?.signal, options.timeout?.request);

  const requestPayload: LLMRequest = {
    prompt: options.prompt,
    messages: options.messages,
    systemPrompt: options.systemPrompt,
    temperature: options.request?.temperature,
    reasoningEffort: options.request?.reasoningEffort,
    maxTokens: options.request?.maxTokens,
    topLogprobs: options.request?.topLogprobs,
    mcpClients: options.request?.mcpClients,
    toolChoice: options.request?.toolChoice,
    parallelToolCalls: options.request?.parallelToolCalls,
    maxToolRounds: options.request?.maxToolRounds,
    onToolExecution: options.request?.onToolExecution,
    transformToolOutput: options.request?.transformToolOutput,
    transformToolArguments: options.request?.transformToolArguments,
    transformToolCallParams: options.request?.transformToolCallParams,
    unknownToolError: options.request?.unknownToolError,
    toolDebug: options.request?.toolDebug,
    onTurnTransition: options.stream.onTurnTransition,
    body: options.request?.body,
    signal: requestSignal,
  };

  emitDebugRequest(options.debug, {
    label: options.debugLabel,
    provider: adapter.provider,
    model: adapter.model,
    attempt: options.attempt,
    selfHealAttempt: options.selfHeal,
    selfHealEnabled: options.selfHealEnabled,
    stream: options.stream.enabled && !!adapter.stream,
    requestPayload,
  });

  options.observe?.(
    options.buildEvent({
      stage: "llm.request",
      message: "Sending LLM request.",
      details: {
        provider: adapter.provider,
        model: adapter.model,
        stream: options.stream.enabled && !!adapter.stream,
      },
    }),
  );

  if (options.stream.enabled && adapter.stream) {
    let latestUsage: LLMUsage | undefined;
    let latestFinishReason: string | undefined;
    let streamedProviderText = "";
    let streamedDedicatedReasoning = "";
    let currentTurnIndex: number | undefined;
    let currentToolCalls: LLMToolCall[] | undefined;
    let streamedReasoningBlocks: ReasoningBlock[] | undefined;
    let emittedOnce = false;
    let lastEmittedText: string | undefined;
    let lastEmittedReasoning: string | undefined;
    let lastEmittedReasoningBlocks: ReasoningBlock[] | undefined;
    let lastEmittedTurnIndex: number | undefined;
    let lastEmittedToolCallsFingerprint: string | undefined;
    let previousSnapshotText = "";
    let previousSnapshotReasoning = "";

    const emitStreamingData = (
      done: boolean,
      usage?: LLMUsage,
      finishReason?: string,
    ): void => {
      const normalized = normalizeModelOutput(
        streamedProviderText,
        streamedDedicatedReasoning,
        streamedReasoningBlocks,
      );
      // Dedup on the rendered output (not on the raw accumulators): chunks that
      // grow the input but render to an identical snapshot — e.g. `<think>`
      // then `</think>` — must not emit twice. `toolCalls` is the only field
      // still serialized; it stays small. Everything else the snapshot exposes
      // is a deterministic function of these source fields.
      const toolCallsFingerprint = currentToolCalls
        ? toStreamDataFingerprint(currentToolCalls)
        : undefined;
      if (
        !done &&
        emittedOnce &&
        normalized.text === lastEmittedText &&
        normalized.reasoning === lastEmittedReasoning &&
        streamedReasoningBlocks === lastEmittedReasoningBlocks &&
        currentTurnIndex === lastEmittedTurnIndex &&
        toolCallsFingerprint === lastEmittedToolCallsFingerprint
      ) {
        return;
      }
      const snapshot = options.buildSnapshot(normalized, { done });

      // Withhold a trailing partial `<think>`/`</think>` fragment while more
      // chunks may still arrive; the final (done) emit releases it in full.
      const stableText = done ? normalized.text : withoutTrailingThinkTagPrefix(normalized.text);
      const stableReasoning = done
        ? normalized.reasoning
        : withoutTrailingThinkTagPrefix(normalized.reasoning);

      const delta = {
        text: stableText.startsWith(previousSnapshotText)
          ? stableText.slice(previousSnapshotText.length)
          : "",
        reasoning: stableReasoning.startsWith(previousSnapshotReasoning)
          ? stableReasoning.slice(previousSnapshotReasoning.length)
          : "",
      };

      emittedOnce = true;
      lastEmittedText = normalized.text;
      lastEmittedReasoning = normalized.reasoning;
      lastEmittedReasoningBlocks = streamedReasoningBlocks;
      lastEmittedTurnIndex = currentTurnIndex;
      lastEmittedToolCallsFingerprint = toolCallsFingerprint;
      previousSnapshotText = stableText;
      previousSnapshotReasoning = stableReasoning;
      options.stream.onData?.({
        delta,
        snapshot,
        done,
        usage,
        finishReason,
        turnIndex: currentTurnIndex,
        toolCalls: currentToolCalls,
      });

      if (options.stream.to === "stdout" && delta.text) {
        process.stdout.write(delta.text);
      }

      options.observe?.(
        options.buildEvent({
          stage: "llm.stream.data",
          message: done ? "Streaming response completed." : "Streaming response updated.",
          details: {
            done,
            finishReason,
          },
        }),
      );
    };

    const handleTextDelta = (delta: string): void => {
      if (!delta) {
        return;
      }

      streamedProviderText += delta;

      options.observe?.(
        options.buildEvent({
          stage: "llm.stream.delta",
          message: "Received stream delta.",
          details: {
            chars: delta.length,
          },
        }),
      );

      emitStreamingData(false);
    };

    const handleReasoningDelta = (delta: string): void => {
      if (!delta) {
        return;
      }

      streamedDedicatedReasoning += delta;
      emitStreamingData(false);
    };

    const streamRequestPayload: LLMRequest = {
      ...requestPayload,
      onTurnTransition: (transition) => {
        if (transition.kind === "reasoningComplete") {
          streamedReasoningBlocks = appendReasoningBlock(streamedReasoningBlocks, transition);
        }
        options.stream.onTurnTransition?.(transition);
      },
    };

    const response = await adapter.stream(streamRequestPayload, {
      onChunk: (chunk) => {
        if (chunk.turnIndex !== undefined) {
          currentTurnIndex = chunk.turnIndex;
        }
        currentToolCalls = chunk.toolCalls;

        if (chunk.textDelta) {
          handleTextDelta(chunk.textDelta);
        }

        if (chunk.reasoningDelta) {
          handleReasoningDelta(chunk.reasoningDelta);
        }

        if (chunk.usage) {
          latestUsage = preferLatestStreamUsage(latestUsage, chunk.usage);
        }

        if (chunk.finishReason) {
          latestFinishReason = chunk.finishReason;
        }

        if (!chunk.textDelta && !chunk.reasoningDelta && (chunk.turnIndex !== undefined || chunk.toolCalls)) {
          emitStreamingData(false, chunk.usage, chunk.finishReason);
        }
      },
    });

    streamedProviderText =
      typeof response.text === "string" ? response.text : streamedProviderText;
    streamedDedicatedReasoning =
      typeof response.reasoning === "string" ? response.reasoning : streamedDedicatedReasoning;
    streamedReasoningBlocks = response.reasoningBlocks ?? streamedReasoningBlocks;
    const finalNormalized = normalizeModelOutput(
      streamedProviderText,
      streamedDedicatedReasoning,
      streamedReasoningBlocks,
    );
    const usage = preferLatestStreamUsage(latestUsage, response.usage);
    const finishReason = response.finishReason ?? latestFinishReason;
    emitStreamingData(true, usage, finishReason);

    options.observe?.(
      options.buildEvent({
        stage: "llm.response",
        message: "Streaming response completed.",
        details: {
          via: "stream",
          chars: finalNormalized.parseSource.length,
          finishReason,
        },
      }),
    );

    emitDebugResponse(options.debug, {
      label: options.debugLabel,
      attempt: options.attempt,
      selfHealAttempt: options.selfHeal,
      selfHealEnabled: options.selfHealEnabled,
      via: "stream",
      text: finalNormalized.text,
      reasoning: finalNormalized.reasoning,
      parseSource: finalNormalized.parseSource,
      usage,
      finishReason,
      logprobs: response.logprobs,
    });

    return {
      text: finalNormalized.text,
      reasoning: finalNormalized.reasoning,
      thinkBlocks: finalNormalized.thinkBlocks,
      parseSource: finalNormalized.parseSource,
      via: "stream",
      usage,
      finishReason,
      ...(response.logprobs ? { logprobs: response.logprobs } : {}),
      reasoningBlocks: finalNormalized.reasoningBlocks,
    };
  }

  const response = await adapter.complete(requestPayload);
  const normalized = normalizeModelOutput(response.text, response.reasoning, response.reasoningBlocks);

  // Streaming was requested but the adapter cannot stream. Consumers finalize on
  // `done`, so emit the completed response as a single terminal event rather than
  // silently never calling `onData` at all.
  if (options.stream.enabled) {
    options.stream.onData?.({
      delta: { text: normalized.text, reasoning: normalized.reasoning },
      snapshot: options.buildSnapshot(normalized, { done: true }),
      done: true,
      usage: response.usage,
      finishReason: response.finishReason,
    });

    if (options.stream.to === "stdout" && normalized.text) {
      process.stdout.write(normalized.text);
    }
  }

  options.observe?.(
    options.buildEvent({
      stage: "llm.response",
      message: "Completion response received.",
      details: {
        via: "complete",
        chars: normalized.parseSource.length,
        finishReason: response.finishReason,
      },
    }),
  );

  emitDebugResponse(options.debug, {
    label: options.debugLabel,
    attempt: options.attempt,
    selfHealAttempt: options.selfHeal,
    selfHealEnabled: options.selfHealEnabled,
    via: "complete",
    text: normalized.text,
    reasoning: normalized.reasoning,
    parseSource: normalized.parseSource,
    usage: response.usage,
    finishReason: response.finishReason,
    logprobs: response.logprobs,
  });

  return {
    text: normalized.text,
    reasoning: normalized.reasoning,
    thinkBlocks: normalized.thinkBlocks,
    parseSource: normalized.parseSource,
    via: "complete",
    usage: response.usage,
    finishReason: response.finishReason,
    ...(response.logprobs ? { logprobs: response.logprobs } : {}),
    reasoningBlocks: normalized.reasoningBlocks,
  };
}
