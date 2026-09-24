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
  normalizeReasoningBlocks,
  sameToolCalls,
} from "./generate-output";
import { createStreamNormalizer, type StreamNormalizerUpdate } from "./stream-normalizer";
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
    // Raw accumulators, only read once the stream ends.
    let streamedProviderText = "";
    let streamedDedicatedReasoning = "";
    let currentTurnIndex: number | undefined;
    let currentToolCalls: LLMToolCall[] | undefined;
    let streamedReasoningBlocks: ReasoningBlock[] | undefined;
    let emittedOnce = false;
    let lastEmittedReasoningBlocks: ReasoningBlock[] | undefined;
    let lastEmittedTurnIndex: number | undefined;
    let lastEmittedToolCalls: LLMToolCall[] | undefined;
    let normalizedBlocksSource: ReasoningBlock[] | undefined;
    let normalizedBlocks: ReasoningBlock[] | undefined;

    // Reads each chunk once: the snapshot strings it hands out are only copied
    // if the consumer reads them, so a consumer of deltas alone stays linear.
    const normalizer = createStreamNormalizer();

    const emitStreamingData = (
      update: StreamNormalizerUpdate,
      done: boolean,
      usage?: LLMUsage,
      finishReason?: string,
    ): void => {
      // Dedup on the rendered output (not on the raw accumulators): chunks that
      // grow the input but render to an identical snapshot — e.g. `<think>`
      // then `</think>` — must not emit twice. Everything else the snapshot
      // exposes is a deterministic function of these source fields.
      if (
        !done &&
        emittedOnce &&
        !update.changed.text &&
        !update.changed.reasoning &&
        streamedReasoningBlocks === lastEmittedReasoningBlocks &&
        currentTurnIndex === lastEmittedTurnIndex &&
        sameToolCalls(currentToolCalls, lastEmittedToolCalls)
      ) {
        return;
      }

      if (streamedReasoningBlocks !== normalizedBlocksSource) {
        normalizedBlocksSource = streamedReasoningBlocks;
        normalizedBlocks = normalizeReasoningBlocks(streamedReasoningBlocks);
      }

      emittedOnce = true;
      lastEmittedReasoningBlocks = streamedReasoningBlocks;
      lastEmittedTurnIndex = currentTurnIndex;
      lastEmittedToolCalls = currentToolCalls;

      if (options.stream.onData) {
        options.stream.onData({
          delta: update.delta,
          snapshot: options.buildSnapshot(
            { text: update.text, reasoning: update.reasoning, reasoningBlocks: normalizedBlocks },
            { done, textExtends: update.textExtends },
          ),
          done,
          usage,
          finishReason,
          turnIndex: currentTurnIndex,
          toolCalls: currentToolCalls,
          ...(update.resync.text || update.resync.reasoning ? { resync: update.resync } : {}),
        });
      }

      if (options.stream.to === "stdout" && update.delta.text) {
        process.stdout.write(update.delta.text);
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

      options.observe?.(
        options.buildEvent({
          stage: "llm.stream.delta",
          message: "Received stream delta.",
          details: {
            chars: delta.length,
          },
        }),
      );

      streamedProviderText += delta;
      emitStreamingData(normalizer.push({ text: delta }), false);
    };

    const handleReasoningDelta = (delta: string): void => {
      if (!delta) {
        return;
      }

      streamedDedicatedReasoning += delta;
      emitStreamingData(normalizer.push({ reasoning: delta }), false);
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
          emitStreamingData(normalizer.push({}), false, chunk.usage, chunk.finishReason);
        }
      },
    });

    streamedReasoningBlocks = response.reasoningBlocks ?? streamedReasoningBlocks;
    const usage = preferLatestStreamUsage(latestUsage, response.usage);
    const finishReason = response.finishReason ?? latestFinishReason;
    // The final response is authoritative over the accumulated chunks.
    const finalText = typeof response.text === "string" ? response.text : streamedProviderText;
    const finalReasoning =
      typeof response.reasoning === "string" ? response.reasoning : streamedDedicatedReasoning;
    emitStreamingData(normalizer.finish({ text: finalText, reasoning: finalReasoning }), true, usage, finishReason);
    const finalNormalized = normalizeModelOutput(finalText, finalReasoning, streamedReasoningBlocks);

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
      snapshot: options.buildSnapshot(normalized, { done: true, textExtends: false }),
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
