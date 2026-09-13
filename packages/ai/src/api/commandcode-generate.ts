/**
 * Command Code `/alpha/generate` API implementation.
 *
 * Command Code exposes two request surfaces:
 *
 * - `provider/v1` — an OpenAI- and Anthropic-compatible pair of endpoints that
 *   require a Provider-or-higher plan. `provider/v1/chat/completions` and
 *   `provider/v1/messages` answer `403 upgrade_required` on Go plans.
 * - `alpha/generate` — the streaming endpoint the official `cmd` CLI uses. It
 *   accepts a single JSON POST and answers with an SSE stream of AI-SDK-style
 *   events (`start`, `reasoning-*`, `text-*`, `tool-input-*`, `tool-call`,
 *   `finish`). Every plan that can use the CLI can use this endpoint.
 *
 * The provider registers `provider/v1` models through the native OpenAI and
 * Anthropic adapters and falls back to this API when the account cannot use
 * them, so Go and Provider plans share one catalog.
 */

import { calculateCost } from "../models.ts";
/** Ground-truth CLI version sent upstream; the endpoint rejects unknown clients. */
import { COMMAND_CODE_CLI_VERSION } from "../providers/commandcode.catalog.ts";
import { commandCodeErrorMessage, readableCommandCodeError } from "../providers/commandcode-errors.ts";
import type {
	AssistantMessage,
	Context,
	Model,
	ProviderEnv,
	SimpleStreamOptions,
	StreamFunction,
	StreamOptions,
	TextContent,
	ThinkingContent,
	Tool,
	ToolCall,
} from "../types.ts";
import { AssistantMessageEventStream } from "../utils/event-stream.ts";
import { headersToRecord, providerHeadersToRecord } from "../utils/headers.ts";
import { parseStreamingJson } from "../utils/json-parse.ts";

const DEFAULT_API_BASE = "https://api.commandcode.ai";
const DEFAULT_GENERATE_MAX_TOKENS = 64_000;
const DEFAULT_MAX_RETRY_DELAY_MS = 60_000;
const BASE_RETRY_DELAY_MS = 500;

export interface CommandCodeGenerateOptions extends StreamOptions {
	reasoning?: string;
}

type CommandCodeTool = Tool & { parameters?: unknown };

interface CommandCodeToolSchema {
	type: "function";
	name: string;
	description: string;
	input_schema: unknown;
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringValue(value: unknown): string | undefined {
	return typeof value === "string" ? value : undefined;
}

function numberValue(value: unknown): number | undefined {
	return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function recordArray(value: unknown): Record<string, unknown>[] {
	return Array.isArray(value) ? value.filter(isRecord) : [];
}

function recordOrEmpty(value: unknown): Record<string, unknown> {
	if (isRecord(value)) return value;
	if (typeof value === "string") {
		const parsed = parseStreamingJson<unknown>(value);
		if (isRecord(parsed)) return parsed;
	}
	return {};
}

function emptyUsage(): AssistantMessage["usage"] {
	return {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function abortError(message = "The operation was aborted"): Error {
	const error = new Error(message);
	error.name = "AbortError";
	return error;
}

function isAbortError(error: unknown): boolean {
	return error instanceof Error && error.name === "AbortError";
}

/**
 * The endpoint fingerprints the client, so requests carry the identity the
 * official CLI sends. `x-project-slug` is derived from the working directory.
 */
export function projectSlugFromPath(pathName: string): string {
	const slug = pathName
		.toLowerCase()
		.replace(/^[a-z]:/i, "")
		.replace(/[^a-z0-9]+/g, "-")
		.replace(/^-+|-+$/g, "");
	return slug || "project";
}

function getEnvironmentInfo(): string {
	return `${process.platform}-${process.arch}, Node.js ${process.version}`;
}

function toJsonSchema(schema: unknown): unknown {
	// TypeBox schemas are already JSON Schema; anything else is passed through
	// verbatim so custom providers keep working.
	if (!isRecord(schema)) return {};
	return schema;
}

function toolsToGenerate(tools?: readonly CommandCodeTool[]): CommandCodeToolSchema[] {
	if (!tools) return [];
	return tools.map((tool) => ({
		type: "function",
		name: tool.name,
		description: tool.description,
		input_schema: tool.parameters ? toJsonSchema(tool.parameters) : {},
	}));
}

function imageParts(value: unknown): Record<string, unknown>[] {
	if (isRecord(value)) return value.type === "image" ? [value] : [];
	return recordArray(value).filter((part) => part.type === "image");
}

function textContentOf(message: { content?: unknown }): string {
	if (typeof message.content === "string") return message.content;
	if (!Array.isArray(message.content)) return "";
	return recordArray(message.content)
		.filter((part) => part.type === "text")
		.map((part) => stringValue(part.text) ?? "")
		.join("\n");
}

function imageToGenerate(part: Record<string, unknown>): Record<string, string> {
	const data = stringValue(part.data);
	const mimeType = stringValue(part.mimeType);
	if (!data || !mimeType) throw new Error("Invalid image content: expected base64 data and mimeType");
	return { type: "image", image: `data:${mimeType};base64,${data}`, mimeType };
}

function userContentToGenerate(content: unknown, allowImages: boolean): unknown {
	if (typeof content === "string") return content;
	return recordArray(content).flatMap((part) => {
		if (part.type === "text") return [{ type: "text", text: stringValue(part.text) ?? "" }];
		if (part.type === "image") {
			if (!allowImages) throw new Error("Selected Command Code model does not support image content");
			return [imageToGenerate(part)];
		}
		return [];
	});
}

function modelSupportsImages(model: Model<string>): boolean {
	return model.input.includes("image");
}

/**
 * Convert pi messages to the `{role, content}` shape `/alpha/generate` accepts.
 * Only `user`, `assistant`, and `tool` roles are valid, so any other role is
 * forwarded as a user message rather than dropped.
 */
export function messagesToGenerate(messages: Context["messages"], options: { allowImages: boolean }): unknown[] {
	const out: unknown[] = [];
	const callIds = new Set<string>();
	const resultIds = new Set<string>();

	for (const message of messages) {
		if (message.role === "assistant") {
			for (const content of message.content) {
				if (content.type === "toolCall") callIds.add(content.id);
			}
		} else if (message.role === "toolResult") {
			resultIds.add(message.toolCallId);
		}
	}

	for (const message of messages) {
		if (message.role === "user") {
			out.push({ role: "user", content: userContentToGenerate(message.content, options.allowImages) });
		} else if (message.role === "assistant") {
			const parts: unknown[] = [];
			const missingResults: unknown[] = [];
			for (const content of message.content) {
				if (content.type === "text") {
					parts.push({ type: "text", text: content.text });
				} else if (content.type === "toolCall") {
					parts.push({
						type: "tool-call",
						toolCallId: content.id,
						toolName: content.name,
						input: content.arguments ?? {},
					});
					if (!resultIds.has(content.id)) {
						missingResults.push({
							type: "tool-result",
							toolCallId: content.id,
							toolName: content.name,
							output: {
								type: "error-text",
								value: "No result - the tool call did not complete (interrupted or lost).",
							},
						});
					}
				}
			}
			if (parts.length > 0) out.push({ role: "assistant", content: parts });
			if (missingResults.length > 0) out.push({ role: "tool", content: missingResults });
		} else if (message.role === "toolResult") {
			if (!callIds.has(message.toolCallId)) continue;
			const images = imageParts(message.content);
			const text = textContentOf(message);
			const outputText =
				text || (images.length > 0 && !options.allowImages ? "[Image omitted: model does not support images]" : "");
			out.push({
				role: "tool",
				content: [
					{
						type: "tool-result",
						toolCallId: message.toolCallId,
						toolName: message.toolName,
						output: message.isError
							? { type: "error-text", value: outputText }
							: { type: "text", value: outputText },
					},
				],
			});
			if (images.length > 0 && options.allowImages) {
				out.push({ role: "user", content: images.map(imageToGenerate) });
			}
		}
	}
	return out;
}

function mapFinishReason(reason: unknown): "stop" | "length" | "toolUse" {
	if (reason === "tool-calls" || reason === "tool_calls") return "toolUse";
	if (reason === "length" || reason === "max_tokens" || reason === "max-tokens" || reason === "max_output_tokens") {
		return "length";
	}
	return "stop";
}

function parseStreamLine(line: string): Record<string, unknown> | undefined {
	let trimmed = line.trim();
	if (!trimmed || trimmed.startsWith(":") || trimmed.startsWith("event:")) return undefined;
	if (trimmed.startsWith("data:")) trimmed = trimmed.slice(5).trim();
	if (!trimmed || trimmed === "[DONE]") return undefined;
	try {
		const parsed: unknown = JSON.parse(trimmed);
		return isRecord(parsed) ? parsed : undefined;
	} catch {
		return undefined;
	}
}

function isRetryableStatus(status: number): boolean {
	return status === 429 || (status >= 500 && status < 600);
}

function parseRetryAfterSeconds(value: string | null): number | undefined {
	if (!value) return undefined;
	const seconds = Number(value);
	if (Number.isFinite(seconds) && seconds >= 0) return seconds;
	const date = Date.parse(value);
	return Number.isNaN(date) ? undefined : Math.max(0, (date - Date.now()) / 1000);
}

function retryDelayMs(attempt: number, retryAfterHeader: string | null, maxDelayMs: number): number {
	const retryAfterMs = parseRetryAfterSeconds(retryAfterHeader);
	if (retryAfterMs !== undefined) {
		if (retryAfterMs * 1000 > maxDelayMs) return -1;
		return retryAfterMs * 1000;
	}
	const exponential = BASE_RETRY_DELAY_MS * 2 ** attempt;
	return Math.min(exponential + exponential * 0.2 * Math.random(), maxDelayMs);
}

function effectiveMaxRetryDelayMs(value: number | undefined): number {
	if (value === undefined) return DEFAULT_MAX_RETRY_DELAY_MS;
	if (value === 0) return Number.POSITIVE_INFINITY;
	return value;
}

function generateMaxTokens(model: Model<string>, options?: StreamOptions): number {
	return Math.min(options?.maxTokens ?? model.maxTokens, model.maxTokens, DEFAULT_GENERATE_MAX_TOKENS);
}

function mappedReasoningEffort(
	model: Model<string>,
	options?: StreamOptions & { reasoning?: string },
): string | undefined {
	const level = options?.reasoning;
	if (!level || level === "off" || !model.reasoning) return undefined;
	const mapped = model.thinkingLevelMap?.[level as keyof typeof model.thinkingLevelMap];
	return typeof mapped === "string" && mapped !== "off" ? mapped : undefined;
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
	if (signal.aborted) return Promise.reject(abortError());
	return new Promise((resolve, reject) => {
		const timer = setTimeout(() => {
			signal.removeEventListener("abort", onAbort);
			resolve();
		}, ms);
		const onAbort = () => {
			clearTimeout(timer);
			reject(abortError());
		};
		signal.addEventListener("abort", onAbort, { once: true });
	});
}

/**
 * `/alpha/generate` sits at the API root, while models are addressed through
 * `/provider/v1`. Hosts may pass either shape (and may rewrite the path), so
 * derive the root from the endpoint the model is actually addressed at.
 */
export function generateBaseUrl(model: Model<string>, env?: ProviderEnv): string {
	const fromEnv = env?.COMMANDCODE_API_BASE ?? process.env.COMMANDCODE_API_BASE;
	const source = fromEnv ?? model.baseUrl;
	try {
		const url = new URL(source);
		return `${url.protocol}//${url.host}`;
	} catch {
		return source.replace(/\/+$/u, "").replace(/\/provider(?:\/v1)?$/u, "") || DEFAULT_API_BASE;
	}
}

export function createGenerateStream(
	model: Model<string>,
	context: Context,
	options?: CommandCodeGenerateOptions,
): AssistantMessageEventStream {
	const eventStream = new AssistantMessageEventStream();
	const output: AssistantMessage = {
		role: "assistant",
		content: [],
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: emptyUsage(),
		stopReason: "pending",
		timestamp: Date.now(),
	};

	void (async () => {
		const controller = new AbortController();
		const onOuterAbort = () => controller.abort();
		if (options?.signal?.aborted) controller.abort();
		else options?.signal?.addEventListener("abort", onOuterAbort, { once: true });

		let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
		let textBlock: TextContent | undefined;
		let thinkingBlock: ThinkingContent | undefined;
		const streamingToolCalls = new Map<string, { contentIndex: number; toolCall: ToolCall; partialArgs: string }>();
		let finished = false;

		const endText = () => {
			if (!textBlock) return;
			eventStream.push({
				type: "text_end",
				contentIndex: output.content.indexOf(textBlock),
				content: textBlock.text,
				partial: output,
			});
			textBlock = undefined;
		};

		const endThinking = () => {
			if (!thinkingBlock) return;
			eventStream.push({
				type: "thinking_end",
				contentIndex: output.content.indexOf(thinkingBlock),
				content: thinkingBlock.thinking,
				partial: output,
			});
			thinkingBlock = undefined;
		};

		const handleEvent = (event: Record<string, unknown>) => {
			switch (event.type) {
				case "text-delta": {
					endThinking();
					if (!textBlock) {
						textBlock = { type: "text", text: "" };
						output.content.push(textBlock);
						eventStream.push({ type: "text_start", contentIndex: output.content.length - 1, partial: output });
					}
					const delta = stringValue(event.text) ?? "";
					textBlock.text += delta;
					eventStream.push({
						type: "text_delta",
						contentIndex: output.content.indexOf(textBlock),
						delta,
						partial: output,
					});
					break;
				}
				case "reasoning-start":
					endText();
					break;
				case "reasoning-delta": {
					endText();
					const delta = stringValue(event.text) ?? "";
					if (!thinkingBlock) {
						thinkingBlock = { type: "thinking", thinking: "" };
						output.content.push(thinkingBlock);
						eventStream.push({
							type: "thinking_start",
							contentIndex: output.content.length - 1,
							partial: output,
						});
					}
					thinkingBlock.thinking += delta;
					eventStream.push({
						type: "thinking_delta",
						contentIndex: output.content.indexOf(thinkingBlock),
						delta,
						partial: output,
					});
					break;
				}
				case "reasoning-end":
					endThinking();
					break;
				case "tool-input-start": {
					endText();
					endThinking();
					const id = stringValue(event.id);
					if (!id || streamingToolCalls.has(id)) break;
					const toolCall: ToolCall = {
						type: "toolCall",
						id,
						name: stringValue(event.toolName) ?? "",
						arguments: {},
					};
					output.content.push(toolCall);
					streamingToolCalls.set(id, { contentIndex: output.content.length - 1, toolCall, partialArgs: "" });
					eventStream.push({ type: "toolcall_start", contentIndex: output.content.length - 1, partial: output });
					break;
				}
				case "tool-input-delta": {
					const id = stringValue(event.id);
					const delta = stringValue(event.delta);
					if (!id || delta === undefined) break;
					const active = streamingToolCalls.get(id);
					if (!active) break;
					active.partialArgs += delta;
					active.toolCall.arguments = recordOrEmpty(active.partialArgs);
					eventStream.push({
						type: "toolcall_delta",
						contentIndex: active.contentIndex,
						delta,
						partial: output,
					});
					break;
				}
				case "tool-call": {
					endText();
					endThinking();
					const id = stringValue(event.toolCallId) ?? "";
					const active = streamingToolCalls.get(id);
					const toolCall: ToolCall = active?.toolCall ?? {
						type: "toolCall",
						id,
						name: stringValue(event.toolName) ?? "",
						arguments: {},
					};
					toolCall.name = stringValue(event.toolName) ?? toolCall.name;
					toolCall.arguments = recordOrEmpty(event.input ?? event.args ?? event.arguments);

					let contentIndex: number;
					if (active) {
						contentIndex = active.contentIndex;
						streamingToolCalls.delete(id);
					} else {
						output.content.push(toolCall);
						contentIndex = output.content.length - 1;
						eventStream.push({ type: "toolcall_start", contentIndex, partial: output });
					}
					eventStream.push({ type: "toolcall_end", contentIndex, toolCall, partial: output });
					break;
				}
				case "finish": {
					const rawFinishReason = stringValue(event.rawFinishReason);
					if (rawFinishReason && /^(?:network|connection|upstream)[-_\s]?error$/i.test(rawFinishReason)) {
						throw new Error(
							`Provider finished with reason "${rawFinishReason}" - upstream connection failed mid-stream`,
						);
					}
					const usage = isRecord(event.totalUsage) ? event.totalUsage : undefined;
					if (usage) {
						const details = isRecord(usage.inputTokenDetails) ? usage.inputTokenDetails : undefined;
						const totalInput = numberValue(usage.inputTokens) ?? 0;
						const cacheRead = numberValue(details?.cacheReadTokens) ?? 0;
						const cacheWrite = numberValue(details?.cacheWriteTokens) ?? 0;
						output.usage.input =
							numberValue(details?.noCacheTokens) ?? Math.max(0, totalInput - cacheRead - cacheWrite);
						output.usage.output = numberValue(usage.outputTokens) ?? 0;
						output.usage.cacheRead = cacheRead;
						output.usage.cacheWrite = cacheWrite;
						output.usage.totalTokens =
							output.usage.input + output.usage.output + output.usage.cacheRead + output.usage.cacheWrite;
						calculateCost(model, output.usage);
					}
					output.stopReason = mapFinishReason(event.finishReason);
					finished = true;
					break;
				}
				case "error": {
					const message =
						commandCodeErrorMessage(event) ?? stringValue(event.message) ?? "Command Code stream error";
					throw new Error(message);
				}
				default:
					break;
			}
		};

		try {
			eventStream.push({ type: "start", partial: output });
			if (controller.signal.aborted) throw abortError();

			const apiKey = options?.apiKey;
			if (!apiKey) throw new Error(`No API key provided for provider "${model.provider}"`);

			const allowImages = modelSupportsImages(model);
			const reasoningEffort = mappedReasoningEffort(model, options);
			const workingDir = process.cwd();
			const timeoutMs = options?.timeoutMs;

			let body: unknown = {
				config: {
					workingDir,
					date: new Date().toISOString().split("T")[0],
					environment: getEnvironmentInfo(),
					structure: [],
					isGitRepo: false,
					currentBranch: "",
					mainBranch: "",
					gitStatus: "",
					recentCommits: [],
				},
				memory: null,
				taste: null,
				skills: null,
				params: {
					model: model.id,
					messages: messagesToGenerate(context.messages, { allowImages }),
					tools: toolsToGenerate(context.tools as CommandCodeTool[] | undefined),
					system: context.systemPrompt ?? "",
					max_tokens: generateMaxTokens(model, options),
					stream: true,
					...(options?.temperature !== undefined ? { temperature: options.temperature } : {}),
					...(reasoningEffort ? { reasoning_effort: reasoningEffort } : {}),
				},
				threadId: options?.sessionId,
			};
			const nextBody = await options?.onPayload?.(body, model);
			if (nextBody !== undefined) body = nextBody;

			const headers = {
				"Content-Type": "application/json",
				Authorization: `Bearer ${apiKey}`,
				"x-command-code-version": COMMAND_CODE_CLI_VERSION,
				"x-cli-environment": "production",
				"x-project-slug": projectSlugFromPath(workingDir),
				"User-Agent": "cli",
				...(options?.sessionId ? { "x-session-id": options.sessionId } : {}),
				...providerHeadersToRecord(options?.headers),
			};

			if (process.env.CMD_ZDR === "1" || process.env.COMMANDCODE_ZDR === "1") {
				(headers as Record<string, string>)["x-cmd-zdr"] = "1";
			}

			const url = `${generateBaseUrl(model, options?.env)}/alpha/generate`;
			const bodyStr = JSON.stringify(body);
			const maxRetries = options?.maxRetries ?? 0;
			const maxRetryDelayMs = effectiveMaxRetryDelayMs(options?.maxRetryDelayMs);

			let response: Response | undefined;
			for (let attempt = 0; ; attempt++) {
				try {
					const requestSignal =
						timeoutMs === undefined
							? controller.signal
							: AbortSignal.any([controller.signal, AbortSignal.timeout(timeoutMs)]);
					response = await (options?.fetch ?? globalThis.fetch)(url, {
						method: "POST",
						headers,
						body: bodyStr,
						signal: requestSignal,
					});
				} catch (error) {
					if (controller.signal.aborted) throw abortError();
					if (timeoutMs !== undefined && error instanceof Error && error.name === "TimeoutError") {
						if (attempt < maxRetries) continue;
						throw new Error(`Command Code API request timed out after ${timeoutMs}ms`);
					}
					throw error;
				}

				if (!response.ok && isRetryableStatus(response.status) && attempt < maxRetries) {
					const waitMs = retryDelayMs(attempt, response.headers.get("retry-after"), maxRetryDelayMs);
					if (waitMs < 0) break;
					await response.text().catch(() => "");
					if (waitMs > 0) await sleep(waitMs, controller.signal);
					continue;
				}
				break;
			}

			const finalResponse = response!;
			await options?.onResponse?.(
				{ status: finalResponse.status, headers: headersToRecord(finalResponse.headers) },
				model,
			);

			if (!finalResponse.ok) {
				const errBody = await finalResponse.text().catch(() => "");
				const detail = commandCodeErrorMessage(errBody) ?? errBody.slice(0, 500);
				throw new Error(`Command Code API error ${finalResponse.status}: ${detail}`);
			}
			if (!finalResponse.body) throw new Error("Command Code response has no body");

			reader = finalResponse.body.getReader();
			const decoder = new TextDecoder();
			let buffer = "";

			readLoop: for (;;) {
				if (controller.signal.aborted) throw abortError();
				const { done, value } = await reader.read();
				if (done) {
					if (buffer.trim()) {
						const event = parseStreamLine(buffer);
						if (event) handleEvent(event);
					}
					if (!finished) {
						throw new Error(
							"Stream ended unexpectedly before completion (no finish event) - response was truncated",
						);
					}
					break;
				}
				buffer += decoder.decode(value, { stream: true });
				const lines = buffer.split("\n");
				buffer = lines.pop() ?? "";
				for (const line of lines) {
					const event = parseStreamLine(line);
					if (!event) continue;
					handleEvent(event);
					if (finished) break readLoop;
				}
			}

			endText();
			endThinking();
			if (output.stopReason === "pending") output.stopReason = "stop";
			eventStream.push({
				type: "done",
				reason: output.stopReason === "length" || output.stopReason === "toolUse" ? output.stopReason : "stop",
				message: output,
			});
			eventStream.end(output);
		} catch (error) {
			output.stopReason = controller.signal.aborted || isAbortError(error) ? "aborted" : "error";
			output.errorMessage =
				output.stopReason === "aborted"
					? "Request aborted"
					: readableCommandCodeError(error instanceof Error ? error.message : String(error));
			eventStream.push({ type: "error", reason: output.stopReason, error: output });
			eventStream.end(output);
		} finally {
			options?.signal?.removeEventListener("abort", onOuterAbort);
			try {
				await reader?.cancel();
			} catch {
				// Reader may already be closed or cancelled by the abort path.
			}
		}
	})();

	return eventStream;
}

export const stream: StreamFunction<string, CommandCodeGenerateOptions> = (
	model: Model<string>,
	context: Context,
	options?: CommandCodeGenerateOptions,
): AssistantMessageEventStream => createGenerateStream(model, context, options);

export const streamSimple: StreamFunction<string, SimpleStreamOptions> = (
	model: Model<string>,
	context: Context,
	options?: SimpleStreamOptions,
): AssistantMessageEventStream =>
	createGenerateStream(model, context, {
		...options,
		reasoning: options?.reasoning,
	});
